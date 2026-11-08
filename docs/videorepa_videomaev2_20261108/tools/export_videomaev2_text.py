"""Lossless bounded text-only publication of explicitly declared VideoMAEv2 runs.

No weights, RGB tensors, PNG, MP4 or symlink targets are opened. Fixed chunks are
immutable. Append logs retain their entire previously exported byte prefix;
status/progress snapshots may change. Oversized UTF-8 text is represented by
immutable parts and a canonical SHA manifest, never by a shortened JSON object.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

THRESHOLD = 350000
MAX_PART = 400000
ARMS = ('baseline', 'videomaev2')
SCHEMAS = {'vsr_videomaev2_preflight_protocol_v1', 'vsr_videomaev2_training_protocol_v1',
           'vsr_videomaev2_evaluation_protocol_v1'}
ROOT_FILES = {'protocol.json', 'status.json', 'progress.json', 'summary.json', 'launch.json',
              'failure.json', 'teacher_repeat.json', 'target_viability.json',
              'gradient_probe_rows.jsonl', 'zero_weight.json', 'resume_report.json',
              'benchmark.jsonl', 'console.log', 'GPU_admission.json', 'runtime.json',
              'observations.json', 'role_audits.json', 'execution_wall_time.json'}
ARM_FILES = {'config.json', 'initialization.json', 'training.jsonl', 'completion.json'}
SNAPSHOTS = {'status.json', 'progress.json'}
ROOT_PATTERN = re.compile(r'(?:attempt|failure_attempt|completion_attempt|GPU_admission_attempt)_\d+\.json\Z')
ARM_PATTERN = re.compile(r'(?:resume_loader_audit_attempt|rollback_attempt)_\d+\.json\Z|interrupted_log_attempt_\d+\.jsonl\Z')
ATTEMPT_PATTERN = re.compile(r'(?:attempt|GPU_admission|failure)_\d+\.json\Z')
ROLE_PATTERN = re.compile(r'(?:load_audit|state_audit)_\d+\.json\Z')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(path, root):
    path, root = Path(path), Path(root)
    need(path.is_file() and path.resolve().is_relative_to(root), 'Text path outside explicit run')
    for ancestor in (path, *path.parents):
        need(not ancestor.is_symlink(), 'Symlink source/destination rejected')
        if ancestor == root:
            break
    return path


def utf8_chunks(raw, maximum=MAX_PART):
    """Prefer entire lines; a line > maximum is split only at UTF-8 boundaries."""
    raw.decode('utf-8')
    offset = 0
    while offset < len(raw):
        end = min(offset + maximum, len(raw))
        if end < len(raw):
            newline = raw.rfind(b'\n', offset, end)
            if newline >= offset:
                end = newline + 1
            else:
                while end > offset and raw[end] & 0xC0 == 0x80:
                    end -= 1
        need(end > offset, 'UTF-8 splitting made no progress')
        block = raw[offset:end]
        block.decode('utf-8')
        yield block
        offset = end


def write_file(path, raw, root, *, immutable=False):
    need(path.resolve().is_relative_to(root), 'Output path escaped text destination')
    for p in (path, *path.parents):
        need(not p.is_symlink(), 'Symlink output rejected')
        if p == root:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and immutable:
        need(path.read_bytes() == raw, 'Immutable text/chunk changed')
        return
    temporary = path.with_name(path.name + '.tmp-export')
    need(not temporary.exists() and not temporary.is_symlink(), 'Export temporary path occupied')
    temporary.write_bytes(raw)
    temporary.replace(path)


def read_manifest_bytes(manifest, dest):
    blocks = []
    for item in manifest['parts']:
        path = safe(dest / item['path'], dest)
        raw = path.read_bytes()
        need(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'Published chunk changed')
        need(len(raw) <= MAX_PART, 'Published chunk exceeds cap')
        raw.decode('utf-8')
        blocks.append(raw)
    raw = b''.join(blocks)
    need(len(raw) == manifest['canonical_bytes'] and sha(raw) == manifest['canonical_sha256'],
         'Published canonical text differs')
    return raw


def chunked_export(raw, relative, dest, mode):
    manifest_path = dest / relative.parent / (relative.name + '.parts_manifest.json')
    parts = []
    prefix = b''
    if (dest / relative).exists() and mode == 'fixed':
        need(safe(dest / relative, dest).read_bytes() == raw, 'Previously published fixed text changed')
    if manifest_path.exists():
        old = json.loads(safe(manifest_path, dest).read_text())
        need(old['canonical_source'] == str(relative) and old['mode'] == mode, 'Manifest identity changed')
        previous = read_manifest_bytes(old, dest)
        if mode == 'fixed':
            need(raw == previous, 'Fixed experiment text changed')
            return old
        if mode == 'append':
            need(raw.startswith(previous), 'Append-only source changed or shrank; preserve rollback explicitly')
            parts = old['parts']
            prefix = previous
    for block in utf8_chunks(raw[len(prefix):]):
        digest = sha(block)
        part = relative.parent / (relative.name + '.parts') / f'part_{len(parts):05d}_{digest}.txt'
        write_file(dest / part, block, dest, immutable=True)
        parts.append({'path': str(part), 'bytes': len(block), 'sha256': digest})
    result = {'schema': 'lossless_UTF8_text_parts_v1', 'canonical_source': str(relative), 'mode': mode,
              'canonical_bytes': len(raw), 'canonical_sha256': sha(raw), 'parts': parts,
              'parts_are_complete_lines_unless_one_line_exceeds_cap': True,
              'reconstruct': 'Concatenate every listed part in order, without separators, byte for byte.'}
    need(read_manifest_bytes(result, dest) == raw, 'Lossless reconstruction failed')
    encoded = (json.dumps(result, indent=2) + '\n').encode()
    need(len(encoded) <= MAX_PART, 'Manifest exceeds publication cap; explicit new strategy required')
    write_file(manifest_path, encoded, dest)
    return result


def export(run, run_root, repo):
    run, run_root, repo = map(Path, (run, run_root, repo))
    need(not run.is_symlink() and not run_root.is_symlink() and not repo.is_symlink(), 'Root symlink rejected')
    run, run_root, repo = (p.resolve() for p in (run, run_root, repo))
    need(run != run_root and run.is_relative_to(run_root), 'Explicit child run required')
    protocol_raw = safe(run/'protocol.json', run).read_bytes()
    protocol = json.loads(protocol_raw)
    need(protocol.get('schema') in SCHEMAS, 'Not a VideoMAEv2 run')
    psha = sha(protocol_raw)
    state = json.loads(safe(run/'status.json', run).read_text())
    need(state.get('protocol_sha256') == psha, 'Protocol/status differs')
    if 'study' in protocol:
        need(protocol['study']['arms'] == [{'label':'baseline','weight':0.0}, {'label':'videomaev2','weight':0.5}],
             'VideoMAEv2 arm settings differ')
    dest = repo/'docs/videorepa_videomaev2_20261108/experiments'/run.name
    need(not dest.is_symlink() and dest.resolve().is_relative_to(repo), 'Unsafe destination')
    files = [p for p in run.iterdir() if p.name in ROOT_FILES or ROOT_PATTERN.fullmatch(p.name)]
    for arm in ARMS:
        directory = run/'arm_runs'/arm
        if directory.exists():
            need(not directory.is_symlink(), 'Arm directory symlink rejected')
            files.extend(p for p in directory.iterdir() if p.name in ARM_FILES or ARM_PATTERN.fullmatch(p.name))
    attempts = run/'attempts'
    if attempts.exists():
        need(not attempts.is_symlink(), 'Attempts symlink rejected')
        files.extend(p for p in attempts.iterdir() if ATTEMPT_PATTERN.fullmatch(p.name))
    if protocol['schema'] == 'vsr_videomaev2_evaluation_protocol_v1':
        for role in protocol.get('roles', []):
            label = role['label']
            need(re.fullmatch(r'[A-Za-z0-9_-]+', label), 'Unsafe role name')
            directory = run/'roles'/label
            if directory.exists():
                need(not directory.is_symlink(), 'Role directory symlink rejected')
                files.extend(p for p in directory.iterdir() if ROLE_PATTERN.fullmatch(p.name))
    exported = []
    for path in sorted(set(files)):
        relative = path.relative_to(run)
        raw = safe(path, run).read_bytes()
        append = path.suffix in ('.log', '.jsonl')
        if append and raw and not raw.endswith(b'\n'):
            raw = raw[:raw.rfind(b'\n')+1]
        need(b'\x00' not in raw, 'NUL-containing file rejected')
        raw.decode('utf-8')
        if path.suffix == '.json':
            json.loads(raw)
        elif path.suffix == '.jsonl':
            rows = [json.loads(x) for x in raw.splitlines() if x]
            if path.name == 'training.jsonl':
                need([x['step'] for x in rows] == list(range(1,len(rows)+1)), 'Training prefix not contiguous')
                need(all(x['arm'] == path.parent.name for x in rows), 'Training arm differs')
        mode = 'append' if append else 'snapshot' if path.name in SNAPSHOTS else 'fixed'
        if len(raw) > THRESHOLD or append or (dest/relative.parent/(relative.name+'.parts_manifest.json')).exists():
            result = chunked_export(raw, relative, dest, mode)
            exported.append({'source':str(relative),'bytes':len(raw),'sha256':sha(raw),
                             'parts_manifest':str(relative.parent/(relative.name+'.parts_manifest.json')),
                             'parts':len(result['parts'])})
        else:
            write_file(dest/relative, raw, dest, immutable=mode=='fixed')
            exported.append({'source':str(relative),'path':str(relative),'bytes':len(raw),'sha256':sha(raw)})
    receipt = {'schema':'videomaev2_public_text_snapshot_v1','status':'text_snapshot','run':str(run),
               'protocol_sha256':psha,'run_status':state['status'],'files':exported,
               'binary_files_opened':0,'media_exported':False,'quality_conclusion':False,
               'max_part_bytes':MAX_PART,'oversized_text_threshold_bytes':THRESHOLD}
    encoded = (json.dumps(receipt,indent=2)+'\n').encode()
    need(len(encoded)<=MAX_PART, 'Snapshot receipt exceeds publication cap')
    write_file(dest/'text_export_snapshot.json',encoded,dest)
    return {'status':'text_exported','destination':str(dest),'files':len(exported),'run_status':state['status']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--run-root',required=True);parser.add_argument('--repo',required=True)
    args=parser.parse_args();print(json.dumps(export(args.run,args.run_root,args.repo)))
