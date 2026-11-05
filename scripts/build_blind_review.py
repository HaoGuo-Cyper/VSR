"""Build an offline anonymous VideoLQ package from a completed sealed PNG run."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/".deps"))
from vsr.blind_review import build_blind_package


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-run",required=True,type=Path)
    parser.add_argument("--labels",required=True,nargs="+")
    parser.add_argument("--dataset",choices=("VideoLQ",),default="VideoLQ")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--max-sequences",type=int,default=None,help="Optional SHA-ordered preselection; default all frozen sources")
    parser.add_argument("--selection-seed",type=int,default=20261102)
    parser.add_argument("--randomization-seed",type=int,default=20261205)
    parser.add_argument("--max-side",type=int,default=960)
    parser.add_argument("--crf",type=int,default=18)
    args=parser.parse_args()
    result=build_blind_package(args.project_root,args.evaluation_run,args.labels,args.output,dataset=args.dataset,
        max_sequences=args.max_sequences,selection_seed=args.selection_seed,randomization_seed=args.randomization_seed,
        max_side=args.max_side,crf=args.crf)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
