"""Validate and seal frozen preparation after an interrupted VSR training run."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vsr.prepared_cache import seal_prepared_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Interrupted original preparation run inside this VSR project's runs")
    args = parser.parse_args()
    seal = seal_prepared_cache(args.run, project_root=ROOT, progress=lambda message: print(message, flush=True))
    print(f"Verified and sealed {len(seal['artifact_sha256'])} frozen preparation artifacts: "
          f"{Path(args.run).resolve() / 'cache_asset_seal.json'}", flush=True)
    print("Origin training remains interrupted; no task checkpoint/optimizer state was read.", flush=True)


if __name__ == "__main__":
    main()
