"""Regenerate an audit manifest from original HQ-VSR files, without old caches."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vsr.data import build_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/stage1.json")
parser.add_argument("--output", default="manifests/hq_vsr.json")
args = parser.parse_args()
config = json.loads((ROOT / args.config).read_text())
manifest = build_manifest(config["data"]["raw_root"], seed=config["seed"])
destination = ROOT / args.output
if destination.exists():
    previous = json.loads(destination.read_text())
    if previous != manifest:
        raise SystemExit("Existing manifest differs; choose a new --output to preserve the sealed split")
    print("Existing manifest reproduced exactly:", manifest["manifest_sha256"])
else:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Created", destination, manifest["manifest_sha256"])
