"""CPU-only anonymous VideoLQ playback packages from sealed evaluation PNGs.

No model inference, metric-based selection, scoring or reviewer response is
performed. Participant media are uniformly encoded from verified PNG pixels;
private mappings and source/encoding provenance remain outside the ZIP.
"""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import zipfile

from PIL import Image

from .robustness import fps_fraction


RATING_KEYS = ("clarity_naturalness", "content_preservation", "flicker_stability", "ghosting_cleanliness")


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Frozen artifact escapes its selected evaluation run")
    return path


def _letter(index):
    result = ""
    while True:
        index, remainder = divmod(index, 26)
        result = chr(65+remainder)+result
        if index == 0:
            return result
        index -= 1


def _sequence(run, directory, artifacts, frames):
    directory = _inside(run/directory, run)
    expected = [directory/f"frame_{i:08d}.png" for i in range(frames)]
    if not directory.is_dir() or set(directory.glob("*.png")) != set(expected):
        raise ValueError("Frozen PNG directory must contain exactly every predeclared frame")
    result = []
    for path in expected:
        path = _inside(path, run)
        relative = path.relative_to(run).as_posix()
        item = artifacts.get(relative)
        if item is None or path.stat().st_size != item["bytes"] or _hash(path) != item["sha256"]:
            raise ValueError(f"Frozen PNG seal mismatch: {relative}")
        result.append({"path": str(path), "relative_path": relative, "bytes": item["bytes"], "sha256": item["sha256"]})
    return result


def _sealed_row(run, protocol_hash, protocol, label, sequence):
    directory = _inside(run/"results"/label/"VideoLQ"/sequence, run)
    metrics = directory/"metrics.json"
    complete = json.loads((directory/"complete.json").read_text())
    contract = _digest({"protocol": protocol_hash, "label": label, "dataset": "VideoLQ", "id": sequence})
    if complete.get("contract_sha256") != contract or complete.get("metrics_sha256") != _hash(metrics):
        raise ValueError("Per-sequence completion seal differs from its frozen protocol/metrics")
    row = json.loads(metrics.read_text())
    if (row.get("status") != "completed" or row.get("contract_sha256") != contract or row.get("label") != label
            or row.get("dataset") != "VideoLQ" or row.get("sequence_id") != sequence
            or row.get("checkpoint_sha256") != protocol["checkpoints"][label]["checkpoint_sha256"]
            or row.get("prefix_frames") != protocol["frames"]):
        raise ValueError("Completed sequence identity/checkpoint/frame contract changed")
    mode = "hri_off" if protocol["checkpoints"][label]["config"].get("reference", {}).get("enabled", False) else "no_reference"
    if set(row["modes"]) != {mode}:
        raise ValueError("VideoLQ blind review accepts missing-HRI/no-reference inference only")
    entries = row["modes"][mode]["seeds"]
    if len(entries) != len(protocol["seeds"]) or {entry["seed"] for entry in entries} != set(protocol["seeds"]):
        raise ValueError("Every predeclared sampling seed must be included; no best-seed selection")
    index = {}
    for item in row["artifacts"]:
        name = item["path"]
        if name in index and index[name] != item:
            raise ValueError("Conflicting sealed artifacts")
        index[name] = item
    frames = protocol["frames"]
    outputs = {}
    for entry in entries:
        if entry.get("png_count") != frames or entry.get("hri_supplied", False) or entry.get("reference_index") is not None:
            raise ValueError("Seed output frame/reference contract changed")
        outputs[entry["seed"]] = _sequence(run, entry["png_directory"], index, frames)
    lr = _sequence(run, f"inputs/VideoLQ/{sequence}/lr", index, frames)
    bicubic = _sequence(run, f"inputs/VideoLQ/{sequence}/bicubic", index, frames)
    return {"source": row["source"], "fps": fps_fraction(row["source_fps"]), "lr": lr, "bicubic": bicubic,
            "outputs": outputs, "mode": mode, "metrics_sha256": _hash(metrics), "complete_sha256": _hash(directory/"complete.json")}


def _binary(name):
    executable = shutil.which(str(name))
    if executable is None:
        raise FileNotFoundError(f"Required CPU encoder/probe not found: {name}")
    return str(Path(executable).resolve())


def _command(args, *, data=None):
    try:
        return subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode("utf-8", errors="replace")[-3000:]) from error


def _encode_pngs(frames, output, fps, target_hw, encoder, probe, crf):
    raw, shape = [], None
    for item in frames:
        payload = Path(item["path"]).read_bytes()
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError("Frozen PNG changed before pixel decoding")
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("Saved metric stimuli must be uint8 RGB PNGs")
            current = (image.height, image.width)
            if shape is not None and current != shape:
                raise ValueError("Stimulus frame geometry changed")
            shape = current
            raw.append(image.tobytes())  # PNG text/paths never enter MP4 metadata.
    height, width = target_hw
    rate = f"{fps.numerator}/{fps.denominator}"
    args = [encoder, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{shape[1]}x{shape[0]}", "-framerate", rate, "-i", "pipe:0", "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"scale={width}:{height}:flags=lanczos", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-threads", "1", "-fps_mode", "passthrough", "-map_metadata", "-1",
            "-metadata", "creation_time=", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-movflags", "+faststart", str(output)]
    _command(args, data=b"".join(raw))
    probe_args = [probe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                  "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_read_frames", "-of", "json", str(output)]
    streams = json.loads(_command(probe_args).stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("Expected one encoded preview stream")
    stream = streams[0]
    if (stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p"
            or (stream["height"], stream["width"]) != (height, width) or int(stream["nb_read_frames"]) != len(frames)
            or Fraction(stream["avg_frame_rate"]) != fps or Fraction(stream["r_frame_rate"]) != fps):
        raise ValueError("Uniform encoding changed frame count, geometry, codec or FPS")
    return {"source_pngs": frames, "source_rgb24_sha256": hashlib.sha256(b"".join(raw)).hexdigest(),
            "source_hw": list(shape), "preview_hw": list(target_hw), "frame_count": len(frames), "fps": rate,
            "encode_args": args, "probe_args": probe_args, "stream": stream,
            "output_sha256": _hash(output), "output_bytes": output.stat().st_size,
            "codec": "libx264", "crf": crf, "preset": "medium", "threads": 1,
            "PNG_metadata_removed": True, "audio": False, "GPU_used": False}


def _preview_hw(shape, max_side):
    factor = min(1., max_side/max(shape))
    return tuple(max(2, int(size*factor)//2*2) for size in shape)


def _viewer(public):
    embedded = json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视频播放盲评</title><style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#20252c}button,select,input,textarea{font:inherit;margin:4px;padding:6px}video{width:100%;max-height:55vh;background:#111}#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:16px}.card{background:white;padding:16px;border-radius:10px}.reference{max-width:480px}label{display:block}textarea{width:90%}small{color:#555}</style>
<h1>视频播放盲评</h1><p>本材料仅评价预先固定的有限帧窗口与预设重复，不代表全长视频或总体质量。没有预填评分。</p>
<p>先观看“输入内容参照”，再同步播放所有匿名刺激。四项均为1—5分，<b>分数越高越好</b>：清晰自然；输入内容保持；闪烁稳定（5=无明显闪烁）；拖影干净（5=无明显拖影）。空白表示未评分。</p>
<label>评审者代号 <input id="reviewer" autocomplete="off"></label><label>编号 <select id="case"></select></label>
<button id="play">同步播放</button><button id="pause">全部暂停</button><button id="reset">同步回到开头</button><label><input id="loop" type="checkbox" checked>循环播放</label>
<button id="json">下载评分JSON</button><button id="csv">下载评分CSV</button><p id="details"></p><div class="card reference"><b>输入内容参照（不评分）</b><video id="reference" controls muted playsinline preload="metadata"></video></div><div id="cards"></div>
<script id="manifest" type="application/json">''' + embedded + r'''</script><script>
const data=JSON.parse(document.getElementById('manifest').textContent), fields=['clarity_naturalness','content_preservation','flicker_stability','ghosting_cleanliness'], titles=['清晰自然','输入内容保持','闪烁稳定','拖影干净'];
const byId=id=>document.getElementById(id), answers={}, choose=byId('case'); let current=null, videos=[];
data.cases.forEach(c=>{const o=document.createElement('option');o.value=c.case_id;o.textContent=c.case_id;choose.appendChild(o)});
function save(){if(!current)return; current.stimuli.forEach(s=>{const key=current.case_id+'/'+s.letter;const row={case_id:current.case_id,stimulus:s.letter};fields.forEach(f=>{const value=byId(s.letter+'_'+f).value;row[f]=value?Number(value):null});row.notes=byId(s.letter+'_notes').value;answers[key]=row})}
function pause(){videos.forEach(v=>v.pause())}function play(){const time=videos[0]?.currentTime||0;videos.forEach(v=>{v.currentTime=time;v.loop=byId('loop').checked});Promise.allSettled(videos.map(v=>v.play()))}
function load(){save();pause();current=data.cases.find(c=>c.case_id===choose.value);byId('reference').src=current.reference;byId('cards').replaceChildren();current.stimuli.forEach(s=>{const card=document.createElement('div');card.className='card';const h=document.createElement('h2');h.textContent=s.letter;card.appendChild(h);const v=document.createElement('video');v.src=s.path;v.controls=true;v.muted=true;v.playsInline=true;v.preload='metadata';card.appendChild(v);const old=answers[current.case_id+'/'+s.letter]||{};fields.forEach((f,i)=>{const label=document.createElement('label');label.textContent=titles[i]+' ';const select=document.createElement('select');select.id=s.letter+'_'+f;['',1,2,3,4,5].forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n||'未评分';select.appendChild(o)});select.value=old[f]??'';label.appendChild(select);card.appendChild(label)});const note=document.createElement('textarea');note.id=s.letter+'_notes';note.placeholder='备注（可空）';note.value=old.notes||'';card.appendChild(note);byId('cards').appendChild(card)});videos=[...document.querySelectorAll('video')];byId('details').textContent='窗口 '+current.frames+' 帧；播放帧率 '+current.fps+'；所有刺激使用相同预览尺寸和编码。';const master=videos[1]||videos[0];master.addEventListener('timeupdate',()=>{if(!master.paused)videos.forEach(v=>{if(v!==master&&Math.abs(v.currentTime-master.currentTime)>.08)v.currentTime=master.currentTime})});videos.forEach(v=>v.loop=byId('loop').checked)}
function rows(){save();return data.cases.flatMap(c=>c.stimuli.map(s=>({reviewer_id:byId('reviewer').value,...(answers[c.case_id+'/'+s.letter]||{case_id:c.case_id,stimulus:s.letter,...Object.fromEntries(fields.map(f=>[f,null])),notes:''})})))}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
byId('play').onclick=play;byId('pause').onclick=pause;byId('reset').onclick=()=>{pause();videos.forEach(v=>v.currentTime=0)};byId('loop').onchange=()=>videos.forEach(v=>v.loop=byId('loop').checked);choose.onchange=load;
byId('json').onclick=()=>download('ratings.json',JSON.stringify({schema_version:1,ratings:rows()},null,2),'application/json');byId('csv').onclick=()=>{const columns=['reviewer_id','case_id','stimulus',...fields,'notes'],quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';download('ratings.csv',[columns,...rows().map(r=>columns.map(k=>r[k]))].map(row=>row.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8')};load();
</script></html>'''


def build_blind_package(project_root, evaluation_run, labels, output_dir, *, dataset="VideoLQ", max_sequences=None,
                        selection_seed=20261102, randomization_seed=20261205, max_side=960, crf=18,
                        ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe"):
    """Build an anonymous participant ZIP and separate private answer key."""
    root = Path(project_root).resolve(); run = _inside(evaluation_run, root/"runs")
    if dataset != "VideoLQ":
        raise ValueError("This blind-review protocol is for unpaired VideoLQ only")
    labels = tuple(labels)
    if (not labels or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", label) for label in labels)):
        raise ValueError("Provide an explicit distinct checkpoint label list")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (selection_seed, randomization_seed)):
        raise ValueError("Selection/randomization seeds must be nonnegative integers")
    if not isinstance(max_side, int) or isinstance(max_side, bool) or max_side < 2 or not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError("Preview max side/CRF outside supported ranges")
    protocol = json.loads((run/"protocol.json").read_text()); contract = _digest(protocol)
    status = json.loads((run/"status.json").read_text())
    if status.get("status") != "completed" or status.get("protocol_sha256") != contract:
        raise ValueError("Use a completed evaluation with its exact frozen protocol")
    if any(label not in protocol["checkpoints"] for label in labels):
        raise ValueError("Explicit label not present in the frozen evaluation")
    ids, seeds, frames = protocol["datasets"].get(dataset), protocol["seeds"], protocol["frames"]
    if (not ids or len(set(ids)) != len(ids) or any(not isinstance(source,str) or not re.fullmatch(r"\d{3}",source) for source in ids)
            or not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
            or not isinstance(frames, int) or isinstance(frames, bool) or frames < 1):
        raise ValueError("Frozen source/seed/frame selection is invalid")
    ordered = sorted(ids, key=lambda source: (_digest(["blind-sequence-selection-v1", selection_seed, source]), source))
    if max_sequences is not None:
        if not isinstance(max_sequences, int) or isinstance(max_sequences, bool) or max_sequences < 1:
            raise ValueError("max_sequences must be a positive preselection budget")
        ordered = ordered[:max_sequences]
    plans = []
    for sequence in ordered:
        rows = {label: _sealed_row(run, contract, protocol, label, sequence) for label in labels}
        first = rows[labels[0]]
        for row in rows.values():
            if (row["source"] != first["source"] or row["fps"] != first["fps"]
                    or row["lr"] != first["lr"] or row["bicubic"] != first["bicubic"]):
                raise ValueError("Matched labels do not share identical source pixels/timing")
        for seed in seeds:
            identities = ["__bicubic__", *labels]
            case_rng_seed = int(_digest(["blind-case-order-v1", randomization_seed, sequence, seed])[:16], 16)
            random.Random(case_rng_seed).shuffle(identities)
            plans.append({"case_id": f"case_{len(plans)+1:04d}", "source_id": sequence, "sampling_seed": seed,
                          "order_rng_seed": case_rng_seed, "identities": identities, "rows": rows})
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty blind-review output directory")
    output.mkdir(parents=True, exist_ok=True)
    participant, private = output/"participant", output/"private"
    participant.mkdir(); private.mkdir()
    encoder, probe = _binary(ffmpeg_bin), _binary(ffprobe_bin)
    encoding = {"ffmpeg_path": encoder, "ffmpeg_sha256": _hash(encoder), "ffmpeg_version": _command([encoder,"-version"]).stdout.decode(),
                "ffprobe_path": probe, "ffprobe_sha256": _hash(probe), "ffprobe_version": _command([probe,"-version"]).stdout.decode(),
                "max_side": max_side, "codec": "libx264", "crf": crf, "preset": "medium", "threads": 1,
                "resize": "one shared per-case target size; Lanczos; no upscale; even dimensions", "GPU_used": False}
    public_cases, answers, blank = [], [], []
    for plan in plans:
        case_id, source, seed = plan["case_id"], plan["source_id"], plan["sampling_seed"]
        first = plan["rows"][labels[0]]
        with Image.open(first["bicubic"][0]["path"]) as image:
            target_hw = _preview_hw((image.height, image.width), max_side)
        folder = participant/"media"/case_id; folder.mkdir(parents=True)
        with Image.open(first["lr"][0]["path"]) as image:
            reference_hw = _preview_hw((image.height, image.width), max_side)
        reference_encoding = _encode_pngs(first["lr"], folder/"input.mp4", first["fps"], reference_hw, encoder, probe, crf)
        public_case = {"case_id": case_id, "frames": frames, "fps": str(first["fps"]), "preview_hw": list(target_hw),
                       "reference": f"media/{case_id}/input.mp4", "stimuli": []}
        answer = {key: plan[key] for key in ("case_id", "source_id", "sampling_seed", "order_rng_seed")}
        answer.update(source_metadata=first["source"], reference_encoding=reference_encoding, stimuli={})
        for index, identity in enumerate(plan["identities"]):
            letter = _letter(index)
            pngs = first["bicubic"] if identity == "__bicubic__" else plan["rows"][identity]["outputs"][seed]
            with Image.open(pngs[0]["path"]) as image:
                if (image.height, image.width) != tuple(reference_encoding["source_hw"][i]*4 for i in range(2)):
                    raise ValueError("Every rated stimulus must share exact x4 target geometry")
            encoded = _encode_pngs(pngs, folder/f"{letter}.mp4", first["fps"], target_hw, encoder, probe, crf)
            public_case["stimuli"].append({"letter": letter, "path": f"media/{case_id}/{letter}.mp4", "sha256": encoded["output_sha256"]})
            answer["stimuli"][letter] = {"identity": "bicubic" if identity == "__bicubic__" else identity,
                "mode": "bicubic" if identity == "__bicubic__" else plan["rows"][identity]["mode"],
                "checkpoint_sha256": None if identity == "__bicubic__" else protocol["checkpoints"][identity]["checkpoint_sha256"],
                "encoding": encoded}
            blank.append({"reviewer_id": "", "case_id": case_id, "stimulus": letter, **{key: None for key in RATING_KEYS}, "notes": ""})
        public_cases.append(public_case); answers.append(answer)
    public = {"schema_version": 1, "title": "视频播放盲评", "cases": public_cases,
              "scope": "预先固定的有限帧窗口；不是完整视频或总体结论", "all_dimensions_high_score_is_better": True,
              "rating_range": [1,5], "ratings_prefilled": False}
    _dump(participant/"manifest.json", public)
    (participant/"index.html").write_text(_viewer(public), encoding="utf-8")
    _dump(participant/"ratings.json", {"schema_version":1, "ratings":blank})
    with (participant/"ratings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["reviewer_id","case_id","stimulus",*RATING_KEYS,"notes"])
        writer.writeheader(); writer.writerows(blank)
    preparation = {"source_evaluation_run": str(run), "protocol_sha256": contract, "protocol_file_sha256": _hash(run/"protocol.json"),
        "dataset": dataset, "explicit_labels": list(labels), "source_count_available": len(ids), "selected_source_ids": ordered,
        "selection_seed": selection_seed, "selection_rule": "SHA256(['blind-sequence-selection-v1',seed,id]) ascending; optional fixed prefix",
        "max_sequences": max_sequences, "all_preset_seeds": list(seeds), "seed_selection_by_score": False,
        "randomization_seed": randomization_seed, "case_order_rule": "local RNG from SHA256(global_randomization_seed,source_id,sampling_seed)",
        "window_frames": frames, "finite_window_only": True, "encoding": encoding,
        "method_scores_read_for_selection": False, "ratings_collected": False, "training_performed": False,
        "quality_scoring_performed": False, "model_inference_performed": False, "module_sha256": _hash(__file__)}
    _dump(private/"preparation_protocol.json", preparation)
    _dump(private/"answer_key.json", {"schema_version":1, "preparation":preparation, "cases":answers})
    (private/"answer_key.json").chmod(0o600)
    archive = output/"participant.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for path in sorted(participant.rglob("*")):
            if path.is_file():
                entry = zipfile.ZipInfo(path.relative_to(participant).as_posix(), date_time=(1980,1,1,0,0,0))
                entry.compress_type = zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                package.writestr(entry, path.read_bytes())
    return {"status":"completed", "participant_zip":str(archive), "participant_zip_sha256":_hash(archive),
            "private_answer_key":str(private/"answer_key.json"), "case_count":len(public_cases),
            "source_count":len(ordered), "seeds_per_source":len(seeds), "stimuli_per_case":len(labels)+1,
            "ratings_collected":False, "quality_conclusion":None, "GPU_used":False}
