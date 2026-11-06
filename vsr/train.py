"""Fresh official initialization and bounded, recorded VSR training experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".deps"))

import torch
import torch.nn.functional as F
from .audit import dump, environment, file_hash, verify_official
from .alignment import DinoTeacher, latent_geometry, point_alignment
from .data import decode_clip, validate_manifest, write_video
from .model import ConditionalWan, gradient_norms
from .preflight import autocast, encode_text, encode_videos, release
from .evaluation import evaluate_restoration
from .cache import PairedClipSequence, validate_cache
from .flow import FlowTeacher
from .relation import relation_loss
from .feature_diagnostics import feature_statistics, same_anchor_patch_identity_retrieval
from .probes import evaluate_denoising_probe
from .reproducibility import configure_reproducibility,tensor_state_hashes
from .restoration_architecture import validate_restoration_architecture


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(config, variant, device="cuda"):
    projection = validate_restoration_architecture(config,variant)
    from diffusers import WanTransformer3DModel
    release()
    training_seed=config["training"].get("seed",config["seed"])
    seed_all(training_seed)
    native, info = WanTransformer3DModel.from_pretrained(
        config["model"]["local_dir"], subfolder="transformer", dtype=torch.bfloat16 if torch.device(device).type=="cuda" else torch.float32,
        local_files_only=True, use_safetensors=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Official initialization loading mismatch: {info}")
    original_count = sum(p.numel() for p in native.parameters())
    for name, p in native.named_parameters():
        if any(k in name for k in (native._keep_in_fp32_modules or [])) and p.dtype != torch.float32:
            raise ValueError(f"Native FP32 exception changed: {name}")
    c = config["conditioning"]
    model_options={k:c[k] for k in ("rank", "alpha", "injection_layers", "alignment_layer", "stem_channels", "projector_hidden", "teacher_dim")}
    model_options.update(projection=projection,gradient_checkpointing=config["training"]["gradient_checkpointing"])
    reference=config.get("reference",{})
    if reference.get("enabled",False):
        from .reference_model import ReferenceConditionalWan
        model=ReferenceConditionalWan(native,**model_options,reference_layer=reference.get("layer",9),
            reference_stem_dim=reference.get("stem_dim",128),reference_seed=reference.get("initialization_seed",training_seed+100)).to(device)
    else:
        model=ConditionalWan(native,**model_options).to(device)
    trainable = {name:p.numel() for name,p in model.named_parameters() if p.requires_grad}
    return model, {"official_revision":config["model"]["revision"], "loading_info":info, "native_parameter_count":original_count,
                   "trainable_parameters":trainable,"trainable_count":sum(trainable.values()),
                   "seed":training_seed,"data_seed":config["seed"],"training_seed":training_seed,
                   "reference_enabled":reference.get("enabled",False),"trainable_initialization_hashes":tensor_state_hashes(model),
                   "fresh_official_initialization":True,"legacy_state_loaded":False,"optimizer_fresh":True}


def select_records(manifest, config):
    if config["data"]["selection"] == "debug":
        records = list(manifest["debug_records"])
    else:
        records = [r for r in manifest["records"] if r["split"] == "train"]
        # Keep selection/data order independent of any model results.
        records.sort(key=lambda r: __import__("hashlib").sha256(f"train:{config['seed']}:{r['id']}".encode()).hexdigest())
        limit = config["data"].get("train_limit")
        if limit:
            records = records[:limit]
    if any(r["split"] != "train" for r in records):
        raise ValueError("Non-training video requested for optimizer updates")
    validation = [r for r in manifest["records"] if r["split"] == "val"]
    validation.sort(key=lambda r: __import__("hashlib").sha256(f"val:{config['seed']}:{r['id']}".encode()).hexdigest())
    return records, validation[:config["data"].get("val_limit",0)]


def prepare(config, run, env):
    manifest = json.loads((ROOT / config["data"]["manifest"]).read_text())
    validate_manifest(manifest)
    training, validation = select_records(manifest, config)
    records = training + validation
    dump(run / "selection.json", {"manifest_sha256":manifest["manifest_sha256"],"training_ids":[r["id"] for r in training],
         "validation_ids":[r["id"] for r in validation],"test_accessed":False,"caption_policy":config["caption_policy"]})
    if config["data"].get("cache_from"):
        source=Path(config["data"]["cache_from"])
        verified=validate_cache(source,config,manifest["manifest_sha256"],
            {key:file_hash(ROOT/key) for key in ("vsr/data.py","vsr/preflight.py")},expected_manifest=manifest)
        original=json.loads((source/"selection.json").read_text())
        if original["training_ids"]!=[r["id"] for r in training] or original["validation_ids"]!=[r["id"] for r in validation]:
            raise ValueError("Requested training/validation selection differs from frozen RGB cache")
        clips=PairedClipSequence(source)
        frozen=torch.load(source/"fresh_training_cache.pt",map_location="cpu",weights_only=True)
        latents=frozen["latents"]
        embeddings=torch.load(source/"fresh_text_embeddings.pt",map_location="cpu",weights_only=True)
        geometry=json.loads((source/"correspondence.json").read_text())
        base=Path(config["model"]["local_dir"])
        actual_geometry=latent_geometry(json.loads((base/"vae/config.json").read_text()),
            json.loads((base/"transformer/config.json").read_text()),config["data"]["num_frames"],
            config["data"]["height"],config["data"]["width"])
        if geometry!=actual_geometry:
            raise ValueError("Cached geometry differs from current official model configurations")
        if len(latents)!=len(clips):
            raise ValueError("Frozen latent cache length differs from RGB selection")
        if frozen.get("binding")!=clips[0]["binding"]:
            raise ValueError("Top-level frozen latent binding differs from originating RGB cache")
        expected_shape=(1,16,*geometry["latent_grid"])
        for i,pair in enumerate(latents):
            if any(tuple(pair[key].shape)!=expected_shape or not bool(torch.isfinite(pair[key]).all()) for key in ("clean","low")):
                raise ValueError("Frozen latent cache geometry/nonfinite values")
            per_clip=torch.load(source/f"clip_{i}/fresh_latents.pt",map_location="cpu",weights_only=True)
            if per_clip.get("binding")!=clips[i]["binding"] or any(not torch.equal(pair[key],per_clip[key]) for key in ("clean","low")):
                raise ValueError("Latent cache clip order or per-clip binding mismatch")
        verified["artifact_sha256"]={name:file_hash(source/name) for name in ("fresh_training_cache.pt","fresh_text_embeddings.pt")}
        seal=source/"cache_asset_seal.json"
        if not seal.is_file():
            raise ValueError("Source experiment requires a completed cache_asset_seal.json before reuse")
        sealed=json.loads(seal.read_text())
        for name,expected in sealed["artifact_sha256"].items():
            resolved=(source/name).resolve()
            if not resolved.is_relative_to(source.resolve()) or file_hash(resolved)!=expected:
                raise ValueError(f"Frozen cache asset checksum mismatch: {name}")
        verified["source_asset_seal_sha256"]=file_hash(seal)
        dump(run/"cache_reuse.json",verified)
        dump(run/"correspondence.json",geometry)
        features,flows=prepare_auxiliary(config,clips,geometry,run)
        return clips,latents,embeddings,features,flows,geometry,len(training)
    # Encode text before accumulating RGB clips, bounding host RAM for full HQ-VSR.
    embeddings=encode_text(ROOT,config,[record["caption"] for record in records],run)
    clips = []
    for i, record in enumerate(records):
        clip = decode_clip(record, **{k:config["data"][k] for k in ("num_frames","height","width","scale")},seed=config["seed"],
                           temporal_stride=config["data"].get("temporal_stride",1))
        clip["binding"] = {"data_cache_key":clip["cache_key"],"manifest_sha256":manifest["manifest_sha256"],
                           "official_revision":config["model"]["revision"],"code_commit":env["git_commit"],
                           "code_sha256":env["code_sha256"],"config_sha256":config["_sha256"]}
        dump(run / f"clip_{i}/metadata.json", {**clip["metadata"],"binding":clip["binding"]})
        # Lossless paired targets are retained; metrics never read MP4 encodings.
        torch.save({k:clip[k] for k in ("hr","lr","lr_up")}, run / f"clip_{i}/paired_rgb.pt")
        if i < config["data"].get("save_input_videos",2):
            for key in ("hr","lr","lr_up"):
                write_video(run / f"clip_{i}/{key}.mp4",clip[key],clip["fps"])
        clips.append(clip)
        if (i+1)%25==0: print(f"Prepared RGB {i+1}/{len(records)}",flush=True)
    base=Path(config["model"]["local_dir"])
    geometry=latent_geometry(json.loads((base/"vae/config.json").read_text()), json.loads((base/"transformer/config.json").read_text()),
                             config["data"]["num_frames"],config["data"]["height"],config["data"]["width"])
    dump(run / "correspondence.json",geometry)
    print("Encoding fresh paired VAE latents",flush=True)
    latents=encode_videos(config,clips,run,geometry)
    teacher_features,flows=prepare_auxiliary(config,clips,geometry,run)
    # Cache belongs to this new experiment and is not an initialization state.
    torch.save({"latents":latents,"teacher_features":teacher_features,"binding":clips[0]["binding"]},run/"fresh_training_cache.pt")
    return clips,latents,embeddings,teacher_features,flows,geometry,len(training)


def prepare_auxiliary(config,clips,geometry,run):
    # Auxiliary caches may be a strict superset of optimized groups (for
    # example, HRI Base/Align repeats still seal frozen teacher/flow targets).
    # Validate before synchronization or construction; never mutate config.
    options=config["training"]
    trained=options["variants"]
    variants=options.get("preparation_variants",trained)
    known={"Base","Align","Relation","Both"}
    for field,names in (("variants",trained),("preparation_variants",variants)):
        if (not isinstance(names,(list,tuple)) or not names
                or any(not isinstance(name,str) or name not in known for name in names)
                or len(set(names))!=len(names)):
            raise ValueError(f"training.{field} must contain distinct known variants")
    if not set(trained).issubset(variants):
        raise ValueError("training.preparation_variants must include every training variant")
    teacher_features,flows=None,None
    timings={"teacher_wall_seconds":0.0,"teacher_inference_seconds":0.0,"flow_wall_seconds":0.0,"flow_inference_seconds":0.0}
    if any(v!="Base" for v in variants):
        torch.cuda.synchronize()
        teacher_started=time.perf_counter()
        teacher=DinoTeacher(ROOT/config["teacher"]["source_dir"],ROOT/config["teacher"]["weights"]).to("cuda")
        torch.cuda.synchronize()
        inference_started=time.perf_counter()
        teacher_features=[teacher(c["hr"].to("cuda"),geometry["anchors"],geometry["token_grid"]).cpu() for c in clips]
        torch.cuda.synchronize()
        timings["teacher_inference_seconds"]=time.perf_counter()-inference_started
        timings["teacher_wall_seconds"]=time.perf_counter()-teacher_started
        dump(run/"teacher_cache_contract.json",{"weights_sha256":config["teacher"]["weights_sha256"],
             "source_revision":config["teacher"]["revision"],"alignment_source_sha256":file_hash(ROOT/"vsr/alignment.py"),
             "shapes":[list(x.shape) for x in teacher_features],"frozen":True,"training_supervision_only":True})
        del teacher
        release()
        torch.save(teacher_features,run/"fresh_teacher_features.pt")
    if any(v in ("Relation","Both") for v in variants):
        torch.cuda.synchronize()
        flow_started=time.perf_counter()
        options=config["flow"]
        estimator=FlowTeacher(ROOT/options["weights"],num_flow_updates=options["updates"],expected_sha256=options["weights_sha256"]).to("cuda")
        flows=[]
        torch.cuda.synchronize()
        inference_started=time.perf_counter()
        for i,clip in enumerate(clips):
            forward,backward=estimator.estimate_anchor_flows(clip["hr"].to("cuda"),geometry["anchors"])
            flows.append({"forward":forward.detach().cpu().clone(),"backward":backward.detach().cpu().clone()})
            if (i+1)%16==0:print(f"Prepared direct HR-anchor flows {i+1}/{len(clips)}",flush=True)
        torch.cuda.synchronize()
        timings["flow_inference_seconds"]=time.perf_counter()-inference_started
        timings["flow_wall_seconds"]=time.perf_counter()-flow_started
        dump(run/"flow_provenance.json",{**estimator.provenance,"anchors":geometry["anchors"],
            "clip_count":len(clips),"training_supervision_only":True})
        torch.save(flows,run/"fresh_anchor_flows.pt")
        del estimator
        release()
    dump(run/"auxiliary_preprocessing_cost.json",{**timings,"clip_count":len(clips),
        "scope":"shared frozen targets; wall seconds include local checkpoint load; inference includes HR transfers",
        "teacher_cache_gpu_hours":timings["teacher_inference_seconds"]/3600,"flow_cache_gpu_hours":timings["flow_inference_seconds"]/3600})
    return teacher_features,flows


def save_checkpoint(path,model,optimizer,step,config,initialization,noise_rng,index_rng):
    state={name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
    torch.save({"format":"vsr_training_v1","step":step,"trainable_state":state,
                "optimizer":optimizer.state_dict(),"noise_rng":noise_rng.get_state(),"index_rng":index_rng.get_state(),
                "config":config,"initialization":initialization},path)


def compute_relation_objective(student,teacher,grid,forward,backward,mask,config):
    """Identical edge objective; CPU route supports strict deterministic backward."""
    cpu=config.get("reproducibility",{}).get("mode")=="strict"
    device=torch.device("cpu") if cpu else student.device
    loss,diagnostics=relation_loss(student.to(device),teacher.detach().to(device),grid,
        forward.detach().to(device),backward.detach().to(device),mask.to(device),
        fb_alpha=config["flow"]["fb_alpha"],fb_beta=config["flow"]["fb_beta"])
    diagnostics.update(loss_device=str(device),strict_cpu_backward=cpu)
    return loss.to(student.device),diagnostics


def auxiliary_audit(model,variant,config,clips,latents,texts,features,flows,geometry,ntrain,target):
    """Separate auxiliary-only generator gradients from validation diagnostics."""
    mask=torch.ones((1,math.prod(geometry["token_grid"])),device="cuda",dtype=torch.bool)
    model.train()
    model.zero_grad(set_to_none=True)
    clean,low=(latents[0][key].to("cuda") for key in ("clean","low"))
    rng=torch.Generator(device="cuda").manual_seed(config["seed"]+91)
    noise=torch.randn(clean.shape,generator=rng,device="cuda")
    noisy=.5*clean+.5*noise
    timestep=torch.tensor([500.],device="cuda")
    with autocast():
        _,student=model(noisy,timestep,texts[clips[0]["metadata"]["caption"]],low)
    alignment=point_alignment(student,features[0].to("cuda"),mask)
    relation=alignment.new_zeros(())
    if variant in ("Relation","Both"):
        relation,_=compute_relation_objective(student,features[0],geometry["token_grid"],flows[0]["forward"],
            flows[0]["backward"],mask,config)
    auxiliary=(config["training"]["lambda_align"]*alignment if variant in ("Align","Both") else alignment.new_zeros(()))
    auxiliary=auxiliary+config["training"].get("lambda_relation",0)*relation
    auxiliary.backward()
    gradients=gradient_norms(model)
    reached=all(gradients[k]["finite"] and gradients[k]["l2"]>0 for k in ("lrv_stem","lrv_residuals","lora","projector"))
    dump(target/"auxiliary_only_gradients.json",{"variant":variant,"alignment":float(alignment.detach()),
        "relation":float(relation.detach()),"weighted_auxiliary":float(auxiliary.detach()),"gradients":gradients,
        "generator_gradient_reached":reached,"training_clip_only":True})
    model.clear_features()
    model.zero_grad(set_to_none=True)
    rows=[]
    selected=list(range(ntrain,len(clips))) or [0]
    model.eval()
    with torch.no_grad():
        for i in selected:
            clean,low=(latents[i][key].to("cuda") for key in ("clean","low"))
            rng=torch.Generator(device="cuda").manual_seed(config["seed"]+91+i)
            noisy=.5*clean+.5*torch.randn(clean.shape,generator=rng,device="cuda")
            with autocast():
                _,student=model(noisy,timestep,texts[clips[i]["metadata"]["caption"]],low)
            teacher=features[i].to("cuda")
            stats=feature_statistics(student,teacher,mask)
            retrieval=same_anchor_patch_identity_retrieval(student,teacher,mask,geometry["token_grid"])
            rows.append({"record_id":clips[i]["metadata"]["record_id"],"split":clips[i]["metadata"]["split"],
                "statistics":stats,"same_anchor_retrieval":retrieval})
            model.clear_features()
    dump(target/"feature_diagnostics.json",{"scope":"oracle_noised_HR_feature_diagnostic_not_restoration", "sigma":.5,
        "per_video":rows,"teacher_frozen":True,"interpretation":"alignment/rank/retrieval alone cannot prove restoration benefit"})


def train_variant(variant,config,run,clips,latents,embeddings,features,flows,geometry,ntrain):
    from diffusers import AutoencoderKLWan,UniPCMultistepScheduler
    options=config["training"]
    target=run/variant
    target.mkdir()
    print(f"{variant}: official fresh initialization",flush=True)
    load_started=time.perf_counter()
    model,initialization=make_model(config,variant)
    torch.cuda.synchronize()
    initialization["load_initialize_seconds"]=time.perf_counter()-load_started
    dump(target/"initialization.json",initialization)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=options["learning_rate"],weight_decay=options["weight_decay"])
    scheduler=UniPCMultistepScheduler.from_pretrained(config["model"]["local_dir"],subfolder="scheduler",local_files_only=True)
    scheduler.set_timesteps(1000,device="cuda")
    training_seed=options.get("seed",config["seed"])
    noise_rng=torch.Generator(device="cuda").manual_seed(training_seed+1)
    index_rng=torch.Generator(device="cpu").manual_seed(training_seed+2)
    # Evaluation VAE is frozen and detached, never part of the optimizer.
    vae=AutoencoderKLWan.from_pretrained(config["model"]["local_dir"],subfolder="vae",dtype=torch.float32,
                                        local_files_only=True,use_safetensors=True).requires_grad_(False).eval().to("cuda")
    texts={k:v.to("cuda") for k,v in embeddings.items()}
    # Fixed targets remain on CPU; each update transfers only its own clip.
    # This keeps teacher/flow GPU memory independent of dataset size.
    metadata=[clips[i]["metadata"] for i in range(len(clips))]
    mask=torch.ones((1,math.prod(geometry["token_grid"])),device="cuda",dtype=torch.bool)
    eval_history=[]
    train_seconds=0.0
    torch.cuda.reset_peak_memory_stats()
    elapsed_start=time.perf_counter()
    for step in range(options["updates"]+1):
        if step in options["evaluate_at"]:
            eval_start=time.perf_counter()
            final=step==options["updates"]
            eval_cfg=config["evaluation"]
            # Debug eval includes memorized clips plus a separately marked val set.
            selected=list(range(ntrain)) if config["data"]["selection"]=="debug" else list(range(ntrain,len(clips)))
            if not selected:
                selected=list(range(min(ntrain,2)))
            summary=evaluate_restoration(model,vae,latents,clips,embeddings,config,target/f"step_{step:06d}",
                modes=tuple(eval_cfg["modes"] if final else ["correct"]),
                seeds=tuple(eval_cfg["seeds"] if final else eval_cfg["monitor_seeds"]),
                steps=eval_cfg["sampling_steps"],train_sample_indices=selected)
            eval_history.append({"step":step,"metrics":summary,"seconds":time.perf_counter()-eval_start})
            dump(target/"evaluation_history.json",eval_history)
            print(f"{variant}: evaluation step {step} complete",flush=True)
            model.train()
        if step==options["updates"]:
            break
        update_start=time.perf_counter()
        model.train()
        i=step%ntrain
        clean,low=(latents[i][key].to("cuda") for key in ("clean","low"))
        text=texts[metadata[i]["caption"]]
        index=int(torch.randint(0,1000,(),generator=index_rng))
        sigma=scheduler.sigmas[index].to("cuda")
        t=scheduler.timesteps[index].expand(1)
        noise=torch.randn(clean.shape,generator=noise_rng,device="cuda")
        noisy=(1-sigma)*clean+sigma*noise
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            velocity,student=model(noisy,t,text,low)
        fm=F.mse_loss(velocity.float(),noise-clean)
        target_features=features[i].to("cuda") if features is not None and variant!="Base" else None
        alignment=point_alignment(student,target_features,mask) if variant in ("Align","Both") else fm.new_zeros(())
        relation,relation_diagnostics=fm.new_zeros(()),None
        if variant in ("Relation","Both"):
            relation,relation_diagnostics=compute_relation_objective(student,target_features,geometry["token_grid"],
                flows[i]["forward"],flows[i]["backward"],mask,config)
        loss=fm+options["lambda_align"]*alignment+options.get("lambda_relation",0.0)*relation
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss at {step+1}")
        loss.backward()
        norms=gradient_norms(model) if (step+1)%options["gradient_log_every"]==0 or step<3 else None
        if norms is not None and (not all(x["finite"] for x in norms.values()) or norms["frozen"]["elements_with_grad"]):
            raise RuntimeError(f"Invalid trainable/frozen gradients at {step+1}: {norms}")
        total_norm=torch.nn.utils.clip_grad_norm_(params,options["max_grad_norm"],error_if_nonfinite=True)
        optimizer.step()
        model.clear_features()
        torch.cuda.synchronize()
        update_seconds=time.perf_counter()-update_start
        train_seconds+=update_seconds
        row={"step":step+1,"clip_id":metadata[i]["record_id"],"timestep_index":index,"sigma":float(sigma),
             "timestep":float(t[0]),"fm":float(fm.detach()),"alignment":float(alignment.detach()),"loss":float(loss.detach()),
             "relation":float(relation.detach()),"relation_diagnostics":relation_diagnostics,
             "grad_norm_before_clip":float(total_norm),"gradient_groups":norms,"update_seconds":update_seconds,
             "optimizer_updates_seconds":train_seconds}
        with (target/"training.jsonl").open("a") as log:
            log.write(json.dumps(row,allow_nan=False)+"\n")
        if (step+1)%options["log_every"]==0:
            print(f"{variant} {step+1}/{options['updates']} FM={row['fm']:.6f} A={row['alignment']:.6f} update_seconds={train_seconds:.2f}",flush=True)
            dump(run/"progress.json",{"variant":variant,"step":step+1,"updates":options["updates"],"loss":row["loss"]})
        if (step+1) in options["checkpoint_at"]:
            save_checkpoint(target/f"checkpoint_{step+1:06d}.pt",model,optimizer,step+1,config,initialization,noise_rng,index_rng)
    peak=torch.cuda.max_memory_allocated()
    if variant!="Base":
        auxiliary_audit(model,variant,config,clips,latents,texts,features,flows,geometry,ntrain,target)
    probe_indices=list(range(ntrain,len(clips))) or list(range(min(ntrain,2)))
    evaluate_denoising_probe(model,latents,clips,embeddings,config,target/"denoising_probe",sample_indices=probe_indices)
    summary={"status":"completed","variant":variant,"updates":options["updates"],"training_clip_count":ntrain,
        "validation_clip_count":len(clips)-ntrain,"test_accessed":False,"optimizer_updates_seconds":train_seconds,
        "training_gpu_hours":train_seconds/3600,"phase_wall_seconds":time.perf_counter()-elapsed_start,
        "peak_allocated_bytes":peak,"evaluation":eval_history[-1],"caption_policy":config["caption_policy"]}
    dump(target/"summary.json",summary)
    del model,optimizer,vae,params,texts
    release()
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config_path=ROOT/args.config
    config=json.loads(config_path.read_text())
    config["_sha256"]=file_hash(config_path)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run=ROOT/"runs"/f"{config['experiment_name']}_{stamp}"
    run.mkdir(parents=True)
    dump(run/"config.json",config)
    dump(ROOT/"runs/active_training.json",{"status":"running","run":str(run)})
    print(f"New independent experiment: {run}",flush=True)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false")
    try:
        phase_started=time.perf_counter()
        dump(run/"reproducibility.json",configure_reproducibility(config))
        env=environment(ROOT)
        dump(run/"environment.json",env)
        if not env["cuda_available"]:
            raise RuntimeError("CUDA unavailable; host GPU execution required")
        audit_started=time.perf_counter()
        dump(run/"official_artifacts.json",verify_official(ROOT,config))
        audit_seconds=time.perf_counter()-audit_started
        preparation_started=time.perf_counter()
        data=prepare(config,run,env)
        preparation_seconds=time.perf_counter()-preparation_started
        summaries={v:train_variant(v,config,run,*data) for v in config["training"]["variants"]}
        status={"status":"completed","stage":config["stage"],"run":str(run),"variants":list(summaries),
                "caption_policy":config["caption_policy"],"test_accessed":False,
                "note":"completion means budget exhausted; recovery and controls require assessment"}
        dump(run/"phase_timings.json",{"artifact_verification_seconds":audit_seconds,
             "preparation_seconds":preparation_seconds,"entire_experiment_wall_seconds":time.perf_counter()-phase_started,
             "optimizer_updates_seconds":{name:s["optimizer_updates_seconds"] for name,s in summaries.items()},
             "note":"wall times are local observations, not controlled cross-hardware benchmarks"})
        dump(run/"status.json",status)
        dump(ROOT/"runs/active_training.json",status)
        print(json.dumps(status),flush=True)
    except Exception as error:
        status={"status":"failed","run":str(run),"error":str(error),"traceback":traceback.format_exc()}
        dump(run/"status.json",status)
        dump(ROOT/"runs/active_training.json",status)
        raise


if __name__=="__main__":
    main()
