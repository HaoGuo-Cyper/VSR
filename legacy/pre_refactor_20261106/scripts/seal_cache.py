"""Seal completed VSR frozen data artifacts before reuse across ablations."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from vsr.audit import dump,file_hash

parser=argparse.ArgumentParser()
parser.add_argument("run")
args=parser.parse_args()
run=Path(args.run).resolve()
if not run.is_relative_to(ROOT/"runs"):
    raise SystemExit("Only this VSR experiment's runs may be sealed")
status=json.loads((run/"status.json").read_text())
if status["status"]!="completed":
    raise SystemExit("Wait for the originating experiment to complete")
names=["config.json","selection.json","correspondence.json","fresh_training_cache.pt","fresh_text_embeddings.pt"]
for directory in sorted(run.glob("clip_*")):
    if directory.is_dir():
        names.extend(str(p.relative_to(run)) for p in (directory/"metadata.json",directory/"paired_rgb.pt",directory/"fresh_latents.pt"))
seal={"source_run":str(run),"artifact_sha256":{name:file_hash(run/name) for name in names},
      "contains_task_initialization":False,"purpose":"only frozen RGB, normalized VAE targets and text encodings"}
target=run/"cache_asset_seal.json"
if target.exists() and json.loads(target.read_text())!=seal:
    raise SystemExit("Existing frozen cache seal differs; refusing overwrite")
dump(target,seal)
print(f"Verified/sealed {len(names)} artifacts: {target}")
