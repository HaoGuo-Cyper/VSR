"""Real-model strict backward and three-update repeat test; never a comparison run."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/".deps"))
import torch
import torch.nn.functional as F
from vsr.audit import dump
from vsr.reproducibility import configure_reproducibility,tensor_state_hashes
from vsr.train import make_model,compute_relation_objective
from vsr.preflight import autocast
from vsr.alignment import point_alignment
from vsr.evaluation import sample_conditioned_latent


def grad_hash(model):
    digest=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(b"none" if p.grad is None else p.grad.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    source=ROOT/"runs/stage2_base_pilot64_20261102T122328.690735Z"
    auxiliary=ROOT/"runs/stage4_relation_both_pilot64_20261102T132829.177751Z"
    config=json.loads((ROOT/"configs/stage4_relation_both_pilot64.json").read_text())
    config["reproducibility"]={"mode":"strict","relation_loss_device":"cpu"}
    config["reference"]={"enabled":False}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run=ROOT/"runs"/f"strict_determinism_probe_{stamp}"
    run.mkdir()
    dump(run/"reproducibility.json",configure_reproducibility(config))
    dump(run/"config.json",config)
    pairs=torch.load(source/"fresh_training_cache.pt",weights_only=True,map_location="cpu")["latents"]
    features=torch.load(auxiliary/"fresh_teacher_features.pt",weights_only=True,map_location="cpu")
    flows=torch.load(auxiliary/"fresh_anchor_flows.pt",weights_only=True,map_location="cpu")
    texts=torch.load(source/"fresh_text_embeddings.pt",weights_only=True,map_location="cpu")
    text=next(iter(texts.values())).to("cuda")
    grid=json.loads((source/"correspondence.json").read_text())["token_grid"]
    model,initialization=make_model(config,"Both")
    dump(run/"initialization.json",initialization)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    clean,low=(pairs[0][k].to("cuda") for k in ("clean","low"))
    teacher=features[0].to("cuda")
    mask=torch.ones(teacher.shape[:2],dtype=torch.bool,device="cuda")
    def backward(step):
        model.train();model.zero_grad(set_to_none=True)
        sigma=[.3,.7,.99][step%3]
        rng=torch.Generator(device="cuda").manual_seed(199+step)
        noise=torch.randn(clean.shape,generator=rng,device="cuda")
        with autocast():velocity,student=model((1-sigma)*clean+sigma*noise,torch.tensor([sigma*1000],device="cuda"),text,low)
        fm=F.mse_loss(velocity.float(),noise-clean)
        alignment=point_alignment(student,teacher,mask)
        relation,edges=compute_relation_objective(student,teacher,grid,flows[0]["forward"],flows[0]["backward"],mask,config)
        loss=fm+.01*alignment+relation
        loss.backward()
        row={"fm":float(fm.detach()),"alignment":float(alignment.detach()),"relation":float(relation.detach()),
             "gradient_sha256":grad_hash(model),"edges":edges}
        model.clear_features()
        return row
    repeats=[backward(0),backward(0)]
    model.eval()
    sample_conditioned_latent(model,low,text,config,steps=2,seed=99)
    repeats.append(backward(0))
    exact_gradients=len({r["gradient_sha256"] for r in repeats})==1
    trajectories=[]
    for before_eval in (False,True):
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.requires_grad:p.copy_(initial[name].to(p.device))
        if before_eval:
            model.eval();sample_conditioned_latent(model,low,text,config,steps=2,seed=99)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.01)
        rows=[]
        for step in range(3):
            rows.append(backward(step));torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);optimizer.step()
        trajectories.append({"before_eval":before_eval,"rows":rows,"task_state":tensor_state_hashes(model)["all_trainable"]["sha256"]})
        del optimizer
    exact_updates=trajectories[0]["task_state"]==trajectories[1]["task_state"]
    summary={"status":"passed" if exact_gradients and exact_updates else "failed","same_state_backward_exact":exact_gradients,
        "three_update_trajectories_exact":exact_updates,"repeats":repeats,"trajectories":trajectories,
        "formal_initialization":"temporary diagnostic model discarded; all formal groups reload official initialization",
        "scope":"one real frozen training sample, three sigmas, A+CPU relation+FM; not proof of cross-hardware reproducibility",
        "source_cache_protocol":"historical pilot frozen data used only for numeric diagnostic; not reused for protocol-v2 training"}
    dump(run/"summary.json",summary);dump(ROOT/"runs/strict_determinism_latest.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("repeats","trajectories")}),flush=True)
    if summary["status"]!="passed":raise RuntimeError("Strict numeric probe did not reproduce exactly")


if __name__=="__main__":main()
