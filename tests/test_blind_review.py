"""Synthetic sealed PNG evaluations only; no real VideoLQ pixels or GPU."""
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image, PngImagePlugin

from vsr.blind_review import _digest, _hash, _viewer, build_blind_package


LABELS = ("SecretBaselineMethod", "SecretRelationMethod")
NODE_BINARY = os.environ.get("VSR_TEST_NODE") or shutil.which("node")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def pngs(run, directory, *, size, tag, count=3):
    directory = run/directory; directory.mkdir(parents=True,exist_ok=True)
    result = []
    for frame in range(count):
        path = directory/f"frame_{frame:08d}.png"
        info = PngImagePlugin.PngInfo(); info.add_text("private_method_tag",tag)
        image = np.full((size,size,3), 40+frame*30, dtype=np.uint8)
        Image.fromarray(image).save(path,pnginfo=info)
        result.append({"path":path.relative_to(run).as_posix(),"bytes":path.stat().st_size,"sha256":_hash(path)})
    return result


def sealed_run(root, sources=("000","001")):
    run = root/"runs"/"synthetic_frozen_evaluation"; run.mkdir(parents=True)
    specs = {label:{"path":f"/private/newVSR/{label}/checkpoint.pt","checkpoint_sha256":hashlib.sha256(label.encode()).hexdigest(),
                    "config":{"reference":{"enabled":index==1}}} for index,label in enumerate(LABELS)}
    protocol = {"schema_version":1,"checkpoints":specs,"datasets":{"VideoLQ":list(sources)},"seeds":[42,43],"frames":3,"start":0}
    save_json(run/"protocol.json",protocol)
    save_json(run/"status.json",{"status":"completed","protocol_sha256":_digest(protocol),"training_updates":0})
    for source in sources:
        source_artifacts = pngs(run,f"inputs/VideoLQ/{source}/lr",size=4,tag="private-LR")
        source_artifacts += pngs(run,f"inputs/VideoLQ/{source}/bicubic",size=16,tag="bicubic")
        for index,label in enumerate(LABELS):
            mode = "hri_off" if index else "no_reference"
            entries, artifacts = [], copy.deepcopy(source_artifacts)
            for seed in protocol["seeds"]:
                directory = f"results/{label}/VideoLQ/{source}/{mode}/seed_{seed}/png"
                artifacts += pngs(run,directory,size=16,tag=label)
                entries.append({"seed":seed,"hri_supplied":False,"reference_index":None,"png_count":3,"png_directory":directory,
                                # Opposite fake scores prove package selection never chooses
                                # a preferred seed by a quality value.
                                "metrics":{"synthetic_score":1000 if seed==42 else -1000}})
            contract = _digest({"protocol":_digest(protocol),"label":label,"dataset":"VideoLQ","id":source})
            row = {"status":"completed","contract_sha256":contract,"checkpoint_sha256":specs[label]["checkpoint_sha256"],
                   "label":label,"dataset":"VideoLQ","sequence_id":source,"source_fps":24.,"prefix_frames":3,
                   "source":{"sequence_id":source,"source_frame_indices":[0,1,2],"source_archive_sha256":"f"*64},
                   "modes":{mode:{"seeds":entries}},"artifacts":artifacts}
            directory = run/"results"/label/"VideoLQ"/source
            save_json(directory/"metrics.json",row)
            save_json(directory/"complete.json",{"contract_sha256":contract,"metrics_sha256":_hash(directory/"metrics.json")})
    return run


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "CPU FFmpeg/ffprobe required")
class BlindReviewTests(unittest.TestCase):
    def test_viewer_javascript_preserves_literal_escape_sequences(self):
        html=_viewer({"cases":[]})
        script=html.rsplit("<script>",1)[1].split("</script>",1)[0]
        self.assertIn(r".join('\r\n')",script)
        self.assertNotIn("\r",script)
        self.assertIn("""replaceAll('"','""')""",script)
        # The viewer is offline file:// compatible: manifest is embedded, no
        # fetch/CORS exception or external script is required.
        self.assertIn('type="application/json"',html)
        self.assertNotIn("fetch(",script)

    @unittest.skipUnless(NODE_BINARY, "node unavailable; escape regression still runs")
    def test_viewer_inline_javascript_node_parse_only(self):
        script=_viewer({"cases":[]}).rsplit("<script>",1)[1].split("</script>",1)[0]
        result=subprocess.run([NODE_BINARY,"--check","-"],input=script,text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_anonymous_zip_contains_all_matched_seeds_and_no_private_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); run=sealed_run(root,sources=("000",))
            result=build_blind_package(root,run,LABELS,root/"blind",max_side=16)
            self.assertEqual(result["case_count"],2)
            self.assertEqual(result["stimuli_per_case"],3)
            self.assertFalse(result["ratings_collected"])
            answer=json.loads(Path(result["private_answer_key"]).read_text())
            self.assertEqual({case["sampling_seed"] for case in answer["cases"]},{42,43})
            self.assertTrue(all({entry["identity"] for entry in case["stimuli"].values()}=={"bicubic",*LABELS} for case in answer["cases"]))
            with zipfile.ZipFile(result["participant_zip"]) as package:
                self.assertTrue({"index.html","manifest.json","ratings.csv","ratings.json"}<=set(package.namelist()))
                self.assertTrue(all("private" not in name and "answer" not in name for name in package.namelist()))
                for name in package.namelist():
                    payload=package.read(name)
                    for forbidden in (*LABELS,"bicubic","/private/newVSR",str(root),"order_rng_seed","sampling_seed"):
                        self.assertNotIn(forbidden.encode(),payload,name)
                ratings=json.loads(package.read("ratings.json"))["ratings"]
                self.assertEqual(len(ratings),6)
                self.assertTrue(all(row["clarity_naturalness"] is None and row["ghosting_cleanliness"] is None for row in ratings))
                csv_rows=list(csv.DictReader(io.StringIO(package.read("ratings.csv").decode("utf-8-sig"))))
                self.assertTrue(all(row["content_preservation"]=="" for row in csv_rows))
                html=package.read("index.html").decode()
                self.assertNotIn("fetch(",html)
                self.assertNotIn("https://",html)
                self.assertIn("同步播放",html)
                self.assertIn("分数越高越好",html)
            for case in answer["cases"]:
                for stimulus in case["stimuli"].values():
                    encoded=stimulus["encoding"]
                    self.assertEqual(encoded["frame_count"],3)
                    self.assertEqual(encoded["fps"],"24/1")
                    self.assertEqual(encoded["preview_hw"],[16,16])
                    self.assertFalse(encoded["GPU_used"])

    def test_sha_preselection_and_local_anonymous_order_reproduce(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); run=sealed_run(root)
            original_random=random.getstate()
            first=build_blind_package(root,run,LABELS,root/"first",max_sequences=1,max_side=16,randomization_seed=99)
            second=build_blind_package(root,run,LABELS,root/"second",max_sequences=1,max_side=16,randomization_seed=99)
            self.assertEqual(original_random,random.getstate())
            a=json.loads(Path(first["private_answer_key"]).read_text())
            b=json.loads(Path(second["private_answer_key"]).read_text())
            self.assertEqual(a["preparation"]["selected_source_ids"],b["preparation"]["selected_source_ids"])
            self.assertEqual(len(a["preparation"]["selected_source_ids"]),1)
            for one,two in zip(a["cases"],b["cases"]):
                self.assertEqual(one["sampling_seed"],two["sampling_seed"])
                self.assertEqual({letter:item["identity"] for letter,item in one["stimuli"].items()},
                                 {letter:item["identity"] for letter,item in two["stimuli"].items()})
            self.assertEqual(a["preparation"]["all_preset_seeds"],[42,43])
            self.assertFalse(a["preparation"]["method_scores_read_for_selection"])

    def test_tampered_png_and_missing_seed_are_rejected_before_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); run=sealed_run(root,sources=("000",))
            bad=next((run/"inputs/VideoLQ/000/lr").glob("*.png"))
            original=bad.read_bytes();bad.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"PNG seal"):
                build_blind_package(root,run,LABELS,root/"bad")
            self.assertFalse((root/"bad").exists())
            bad.write_bytes(original)
            metrics=run/"results"/LABELS[0]/"VideoLQ/000/metrics.json"
            row=json.loads(metrics.read_text());row["modes"]["no_reference"]["seeds"].pop()
            save_json(metrics,row)
            complete=metrics.parent/"complete.json";seal=json.loads(complete.read_text());seal["metrics_sha256"]=_hash(metrics);save_json(complete,seal)
            with self.assertRaisesRegex(ValueError,"Every predeclared"):
                build_blind_package(root,run,LABELS,root/"missing")


if __name__=="__main__":unittest.main()
