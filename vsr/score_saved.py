"""Score only new VSR lossless saved samples with pinned perceptual metrics."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/".deps"))
import torch
from .audit import dump,file_hash
from .cache import PairedClipSequence
from .perceptual_metrics import PerceptualMetrics,LPIPS_KEY,TLP_KEY,TLP100_KEY,TOF_KEY


def scalar_values(result):
    return {LPIPS_KEY:result[LPIPS_KEY],**{k:result["tecogan"][k] for k in (TLP_KEY,TLP100_KEY,TOF_KEY)}}


def mean_values(rows):
    return {k:sum(row[k] for row in rows)/len(rows) for k in rows[0]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--variant",required=True)
    parser.add_argument("--step",type=int,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--modes",nargs="+",default=["correct"])
    args=parser.parse_args()
    run=Path(args.run).resolve()
    if not run.is_relative_to(ROOT/"runs"):
        raise ValueError("Only this independent VSR project's saved samples may be scored")
    directory=run/args.variant/f"step_{args.step:06d}"
    output=directory/"perceptual_metrics_v2.json"
    if output.exists():
        raise FileExistsError("Bound scoring record exists; preserve its original input hashes")
    metrics=json.loads((directory/"metrics.json").read_text())
    if metrics["status"]!="completed" or metrics["sampling"].get("matched_noise_across_modes") is not True:
        raise ValueError("Requires completed matched-noise sampling")
    source=run
    if (run/"cache_reuse.json").exists():
        source=Path(json.loads((run/"cache_reuse.json").read_text())["source_run"])
    source=source.resolve()
    if not source.is_relative_to(ROOT/"runs"):
        raise ValueError("RGB cache must originate in this independent project")
    clips=PairedClipSequence(source)
    evaluator=PerceptualMetrics(ROOT/"weights/alexnet-owt-7be5be79.pth",ROOT/"weights/lpips_v0.1_alex.pth",device=args.device)
    started=time.perf_counter()
    videos=[]
    for row in metrics["per_video"]:
        clip=clips[row["clip_index"]]
        if (row["record_id"],row["source_id"],row["split"])!=(clip["metadata"]["record_id"],clip["metadata"]["source_id"],clip["metadata"]["split"]):
            raise ValueError("RGB metric identity differs from its bound target cache")
        baseline=evaluator.evaluate(clip["lr_up"],clip["hr"])
        result={"record_id":row["record_id"],"source_id":row["source_id"],"split":row["split"],"bicubic":baseline,"modes":{},
            "source_rgb_sha256":file_hash(source/f"clip_{row['clip_index']}/paired_rgb.pt"),
            "source_metadata_sha256":file_hash(source/f"clip_{row['clip_index']}/metadata.json")}
        for mode in args.modes:
            seeds=[]
            if {sample["seed"] for sample in row["modes"][mode]["seeds"]}!=set(metrics["sampling"]["seeds"]):
                raise ValueError("Mode lacks declared sampling seeds")
            for sample in row["modes"][mode]["seeds"]:
                if sample.get("noise_seed")!=sample["seed"]:
                    raise ValueError("Stored actual noise seed differs from its sampling label")
                tensor=Path(sample["rgb_path"]).resolve()
                if not tensor.is_relative_to(run):
                    raise ValueError("Saved generated sample escapes selected experiment")
                rgb_sha256=file_hash(tensor)
                video=torch.load(tensor,map_location="cpu",weights_only=True)
                score=evaluator.evaluate(video,clip["hr"])
                if file_hash(tensor)!=rgb_sha256:
                    raise ValueError("Generated RGB changed during scoring")
                seeds.append({"seed":sample["seed"],"noise_seed":sample["noise_seed"],"rgb_path":str(tensor),
                    "rgb_sha256":rgb_sha256,"metrics":score})
            result["modes"][mode]={"seeds":seeds,"mean":mean_values([scalar_values(s["metrics"]) for s in seeds])}
        videos.append(result)
        print(f"Scored {len(videos)}/{len(metrics['per_video'])}: {row['record_id']}",flush=True)
    aggregate={"bicubic":mean_values([scalar_values(r["bicubic"]) for r in videos]),
        "modes":{mode:mean_values([r["modes"][mode]["mean"] for r in videos]) for mode in args.modes}}
    summary={"format":"vsr_saved_perceptual_v2","status":"completed","scope":metrics["scope"],"phase":"posthoc_saved_lossless_samples","audit":evaluator.audit,
        "rgb_metrics_sha256":file_hash(directory/"metrics.json"),"scorer_sha256":file_hash(Path(__file__)),
        "source_rgb_cache":str(source),"sampling":metrics["sampling"],
        "per_video":videos,"aggregate":aggregate,"device":args.device,"wall_seconds":time.perf_counter()-started,
        "aggregation":"frames/pairs per video, then all seeds, then equal video weights; no seed selection"}
    dump(output,summary)
    print(json.dumps(aggregate),flush=True)


if __name__=="__main__":
    main()
