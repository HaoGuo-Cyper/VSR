"""Export explicit allowlisted TRD text; never open weights/media/tensor caches."""
from pathlib import Path
import argparse,hashlib,json

ROOT_FILES={'protocol.json','status.json','progress.json','summary.json','launch.json','failure.json',
            'gradient_calibration.json','same_input_semantics.json','calibration_rows.jsonl','benchmark.jsonl','console.log',
            'GPU_admission.json','runtime.json'}
ARM_FILES={'config.json','initialization.json','training.jsonl','completion.json'}
DYNAMIC={'status.json','progress.json'}
def need(value,message):
    if not value:raise ValueError(message)
def export(run,run_root,repo):
    run,run_root,repo=map(lambda p:Path(p).resolve(),(run,run_root,repo))
    need(run!=run_root and run.is_relative_to(run_root),'Explicit child run required')
    protocol=json.loads((run/'protocol.json').read_text())
    need(protocol['schema'] in ('wan_trd_calibration_protocol_v1','wan_trd_training_protocol_v1','wan_trd_evaluation_protocol_v1'),'Not a TRD run')
    psha=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()
    state=json.loads((run/'status.json').read_text());need(state['protocol_sha256']==psha,'Protocol/status differs')
    dest=repo/'docs/ema_teacher_targets_20261108/experiments'/run.name
    files=[p for p in run.iterdir() if p.name in ROOT_FILES or (p.name.startswith('attempt_') and p.suffix=='.json')
           or (p.name.startswith('failure_attempt_') and p.suffix=='.json') or (p.name.startswith('completion_attempt_') and p.suffix=='.json')]
    for arm in ('baseline','frozen','ema'):
        directory=run/'arm_runs'/arm
        if directory.is_dir():files.extend(p for p in directory.iterdir() if p.name in ARM_FILES)
    exported=[]
    for path in sorted(files):
        need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run),'Unsafe source text path')
        relative=path.relative_to(run);raw=path.read_bytes()
        if path.suffix=='.json':json.loads(raw)
        else:
            # Append logs only through their last complete line.
            if raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
            raw.decode('utf-8')
            if path.suffix=='.jsonl':
                rows=[json.loads(line) for line in raw.splitlines() if line]
                if path.name=='training.jsonl':need([x['step'] for x in rows]==list(range(1,len(rows)+1)),'Training prefix not contiguous')
        need(b'\x00' not in raw,'NUL-containing file rejected')
        if path.name=='training.jsonl':
            lines=raw.splitlines(keepends=True);parts=[]
            for start in range(0,len(lines),512):
                chunk=b''.join(lines[start:start+512]);part=relative.parent/'training_parts'/f'part_{start//512:04d}.jsonl'
                target=dest/part;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():need(chunk.startswith(target.read_bytes()),'Published training part changed')
                target.write_bytes(chunk)
                item={'path':str(part),'first_step':start+1,'last_step':start+len(lines[start:start+512]),
                      'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()}
                parts.append(item);exported.append(item)
            manifest={'canonical_source':str(relative),'canonical_bytes':len(raw),'canonical_sha256':hashlib.sha256(raw).hexdigest(),
                      'complete_rows':len(lines),'reconstruct':'Concatenate parts in listed order, byte for byte.','parts':parts}
            (dest/relative.parent/'training_parts_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            need(b''.join((dest/x['path']).read_bytes() for x in parts)==raw,'Lossless training-part reconstruction failed')
            continue
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            previous=target.read_bytes()
            if path.suffix in ('.jsonl','.log'):need(raw.startswith(previous),'Append-only text changed')
            elif path.name not in DYNAMIC:need(raw==previous,'Fixed experiment text changed')
        target.write_bytes(raw)
        exported.append({'path':str(relative),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    receipt={'status':'text_snapshot','run':str(run),'protocol_sha256':psha,'run_status':state['status'],
             'files':exported,'binary_files_opened':0,'media_exported':False,'quality_conclusion':False}
    (dest/'text_export_snapshot.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return {'status':'text_exported','destination':str(dest),'files':len(exported),'run_status':state['status']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--run-root',required=True);p.add_argument('--repo',required=True)
    a=p.parse_args();print(json.dumps(export(a.run,a.run_root,a.repo)))
