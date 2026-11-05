"""Training-only, zero-update relation-weight calibration before ablations."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/".deps"))
import torch
from .audit import dump,environment,file_hash
from .cache import PairedClipSequence,validate_cache
from .calibration import collect_auxiliary_gradient_scales
from .flow import FlowTeacher
from .train import make_model,release


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--teacher-run",required=True)
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    source=Path(config["data"]["cache_from"]).resolve()
    teacher_source=Path(args.teacher_run).resolve()
    if not source.is_relative_to(ROOT/"runs") or not teacher_source.is_relative_to(ROOT/"runs"):
        raise ValueError("Only new VSR experiment caches are allowed")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run=ROOT/"runs"/f"stage4_relation_calibration_{stamp}"
    run.mkdir(parents=True)
    dump(run/"config.json",config)
    dump(run/"environment.json",environment(ROOT))
    manifest=json.loads((ROOT/config["data"]["manifest"]).read_text())
    contract=validate_cache(source,config,manifest["manifest_sha256"],
        {key:file_hash(ROOT/key) for key in ("vsr/data.py","vsr/preflight.py")},expected_manifest=manifest)
    dump(run/"cache_contract.json",contract)
    first_selection=json.loads((source/"selection.json").read_text())
    if json.loads((teacher_source/"selection.json").read_text())!=first_selection:
        raise ValueError("Teacher feature ordering differs from RGB/latent cache")
    geometry=json.loads((source/"correspondence.json").read_text())
    teacher_contract=json.loads((teacher_source/"teacher_cache_contract.json").read_text())
    if teacher_contract["weights_sha256"]!=config["teacher"]["weights_sha256"] or teacher_contract["alignment_source_sha256"]!=file_hash(ROOT/"vsr/alignment.py"):
        raise ValueError("Teacher weights/implementation differ from calibration configuration")
    clips=PairedClipSequence(source)
    latents=torch.load(source/"fresh_training_cache.pt",map_location="cpu",weights_only=True)["latents"]
    features=torch.load(teacher_source/"fresh_teacher_features.pt",map_location="cpu",weights_only=True)
    embeddings=torch.load(source/"fresh_text_embeddings.pt",map_location="cpu",weights_only=True)
    if len(clips)!=len(latents) or len(features)!=len(latents):
        raise ValueError("Calibration cache length mismatch")
    options=config["flow"]
    estimator=FlowTeacher(ROOT/options["weights"],num_flow_updates=options["updates"],expected_sha256=options["weights_sha256"]).to("cuda")
    flows=[]
    for i in range(3):
        if clips[i]["metadata"]["split"]!="train":
            raise ValueError("Calibration uses train only")
        f,b=estimator.estimate_anchor_flows(clips[i]["hr"].to("cuda"),geometry["anchors"])
        flows.append({"forward":f.cpu().clone(),"backward":b.cpu().clone()})
    dump(run/"flow_provenance.json",estimator.provenance)
    del estimator
    release()
    model,initialization=make_model(config,"Relation")
    dump(run/"initialization.json",initialization)
    results=[]
    for i in range(3):
        result=collect_auxiliary_gradient_scales(model,latents[i]["clean"].to("cuda"),latents[i]["low"].to("cuda"),
            embeddings[clips[i]["metadata"]["caption"]].to("cuda"),features[i].to("cuda"),geometry["token_grid"],flows[i],
            seed=config["seed"]+200+i,sigmas=(.3,.7,.99),target_fraction=.1,weight_bounds=(.01,10),
            fb_alpha=options["fb_alpha"],fb_beta=options["fb_beta"])
        result["record_id"]=clips[i]["metadata"]["record_id"]
        results.append(result)
        print(f"Calibration train clip {i}: suggested lambdaR={result['suggested_lambda_R']}",flush=True)
    values=[row["lambda_R"]["raw"] for result in results for row in result["rows"] if row["lambda_R"]["raw"] is not None]
    if not values:
        raise ValueError("Relation generator gradients are all zero; cannot choose weight")
    median=statistics.median(values)
    bounded=min(10.,max(.01,median))
    candidates=[.01,.03,.1,.3,1.,3.,10.]
    chosen=min(candidates,key=lambda x:abs(math.log(x)-math.log(bounded)))
    summary={"status":"completed","run":str(run),"training_clips":3,"optimizer_updates":0,"model_state_modified":False,
        "raw_median_lambda_R":median,"bounded_median_lambda_R":bounded,"selected_lambda_R":chosen,
        "candidate_set":candidates,"target_generator_gradient_fraction":.1,
        "selection_rule":"nearest log candidate to median 0.1*gFM/gR over 3 train clips x sigmas .3/.7/.99; projector excluded",
        "fixed_lambda_A":.1,"test_or_validation_used_for_calibration":False,"results":results,
        "formal_initialization":"calibration model discarded; Relation and Both independently reload official weights"}
    dump(run/"summary.json",summary)
    dump(ROOT/"runs/relation_calibration_latest.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="results"}),flush=True)


if __name__=="__main__":
    main()
