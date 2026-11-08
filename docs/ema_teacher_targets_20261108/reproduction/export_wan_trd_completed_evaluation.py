#!/usr/bin/env python3
"""Export only fully verified Wan TRD evaluation text, preserving source bytes.

This unbound postprocessor never imports the model, opens checkpoints/media, or
repeats native PNG/MP4 verification. It requires the saved native verify receipt.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


SCHEMA = 'wan_trd_evaluation_protocol_v1'
METRICS = ('psnr_rgb_db', 'ssim_rgb', 'lpips_alex_v0_1',
           'tLP_alex_v0_1_TecoGAN_crop', 'tLP100_alex_v0_1_TecoGAN_crop',
           'tOF_Farneback_TecoGAN_crop')
SAFE_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}')
SHA = re.compile(r'[0-9a-f]{64}')
SINGLE_LIMIT = 1_000_000
CHUNK_BYTES = 800_000  # Escaped connector JSON stays below its 2.6 MB payload cap.
MAX_FILE_BYTES = 64_000_000
MAX_TOTAL_BYTES = 384_000_000


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(raw):
    need(b'\x00' not in raw, 'NUL byte in an allowed text file')
    return raw.decode('utf-8')


def parse(raw):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(decode(raw), object_pairs_hook=strict_pairs, parse_constant=invalid)


def absolute(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and str(path) == os.path.normpath(str(path)),
         'Use an absolute normalized path without parent traversal')
    return path


def no_symlinks(path):
    """Use lstat only; never follow a source or destination symbolic link."""
    path = absolute(path)
    for part in reversed((path, *path.parents)):
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError:
            continue
        need(not stat.S_ISLNK(mode), 'Symbolic links are not allowed: ' + str(part))
    return path


def reference(path, parent, suffix):
    """Validate a binary reference lexically/with lstat, without opening it."""
    path = no_symlinks(path)
    need(path != parent and path.is_relative_to(parent) and path.suffix == suffix,
         'Artifact reference escaped its expected directory')


class TextStore:
    def __init__(self):
        self.raw = {}
        self.signatures = {}
        self.total = 0

    def read(self, path):
        path = no_symlinks(path)
        if path in self.raw:
            return self.raw[path]
        before = path.lstat()
        need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE_BYTES, 'Expected bounded regular text file')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            current = os.fstat(stream.fileno())
            need((before.st_dev, before.st_ino) == (current.st_dev, current.st_ino), 'Text file changed before opening')
            raw = stream.read(MAX_FILE_BYTES + 1)
            after = os.fstat(stream.fileno())
        need(len(raw) <= MAX_FILE_BYTES and self.signature(before) == self.signature(after), 'Text changed during read')
        decode(raw)
        self.total += len(raw)
        need(self.total <= MAX_TOTAL_BYTES, 'Text export exceeds the declared memory bound')
        self.raw[path] = raw
        self.signatures[path] = self.signature(after)
        return raw

    @staticmethod
    def signature(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def json(self, path):
        return parse(self.read(path))

    def unchanged(self):
        for path, before in self.signatures.items():
            no_symlinks(path)
            need(self.signature(path.lstat()) == before, 'Allowed text changed during export validation: ' + str(path))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_sha(value):
    need(isinstance(value, str) and SHA.fullmatch(value), 'Invalid SHA256 declaration')


def identity(protocol, role, item, sample):
    return dict(role=role['label'], arm=role['arm'], weight=role['weight'], update=role['update'],
                step=role['step'], cumulative_training_seconds=role['cumulative_training_seconds'],
                training_seed=20261102, record_id=item['record']['id'], source_id=item['record']['source_id'],
                sampling_seed=sample, condition='clean', checkpoint_sha256=role['checkpoint']['sha256'],
                cohort_seal_digest=protocol['cohort_seal_digest'], family=role['family'], setting=role['setting'],
                split=protocol['scope']['source_split'])


def protocol_shape(protocol):
    phase = protocol.get('phase')
    need(protocol.get('schema') == SCHEMA and phase in ('val', 'test'), 'Require Wan TRD val/test schema')
    total = 640 if phase == 'val' else 192
    scope = protocol['scope']
    need(scope['total_restores'] == total and scope['roles'] == (10 if phase == 'val' else 3)
         and scope['clips'] == 32 and scope['sampling_seeds'] == [42, 43]
         and scope['conditions'] == ['clean'] and scope['source_split'] == phase
         and scope['frames'] == 33 and scope['training_updates'] == 0
         and scope['weight_or_checkpoint_selection'] is False, 'Full fixed TRD matrix scope differs')
    roles, cohort = protocol['roles'], protocol['cohort']
    need(len(roles) == scope['roles'] and len(cohort) == 32, 'Incomplete roles/cohort')
    labels, role_keys = set(), set()
    for role in roles:
        label = role['label']
        need(isinstance(label, str) and SAFE_NAME.fullmatch(label) and label not in labels, 'Unsafe/duplicate role label')
        labels.add(label)
        key = ('shared', 0) if role['update'] == 0 else (role['arm'], role['update'])
        need(key not in role_keys, 'Duplicate role arm/update')
        role_keys.add(key)
        need(finite(role['weight']) and role['weight'] >= 0 and finite(role['cumulative_training_seconds'])
             and role['cumulative_training_seconds'] >= 0, 'Invalid role weight/time')
        checked_sha(role['checkpoint']['sha256'])
        if role['update'] == 0:
            need(phase == 'val' and role['step'] == 8192 and role['weight'] == 0, 'Wrong shared initial role')
        else:
            need(role['arm'] in ('baseline', 'frozen', 'ema') and role['step'] == role['update'], 'Wrong trained role')
    expected = ({('shared', 0)} | {(arm, step) for arm in ('baseline', 'frozen', 'ema') for step in (512, 1024, 2048)}
                if phase == 'val' else {(arm, 2048) for arm in ('baseline', 'frozen', 'ema')})
    need(role_keys == expected, 'Missing declared arm/checkpoint role')
    records, sources = set(), set()
    for item in cohort:
        record, source = item['record']['id'], item['record']['source_id']
        need(isinstance(record, str) and SAFE_NAME.fullmatch(record) and record not in records, 'Unsafe/duplicate clip')
        need(isinstance(source, str) and source not in sources, 'Duplicate source video')
        records.add(record); sources.add(source)
        need(isinstance(item['source_indices'], list) and len(item['source_indices']) == 33, 'Wrong source window')
    return phase, total


def collect(run, verification, source_root):
    """Gate completed status and native receipt before reading any quality file."""
    run, verification, source_root = (no_symlinks(p) for p in (run, verification, source_root))
    need(run.is_dir() and source_root.is_dir(), 'Missing run/source root')
    store = TextStore()
    state = store.json(run / 'status.json')
    need(state.get('status') == 'completed', 'Evaluation must be completed before any quality text is read')
    protocol = store.json(run / 'protocol.json'); psha = digest(store.raw[run / 'protocol.json'])
    phase, total = protocol_shape(protocol)
    receipt = store.json(verification)
    need(state.get('protocol_sha256') == psha and state.get('restores') == total and state.get('training_updates') == 0,
         'Completion status/protocol chain differs')
    need(receipt.get('status') == 'verified_complete_clean_evaluation' and receipt.get('run') == str(run)
         and receipt.get('protocol_sha256') == psha and receipt.get('restores') == total
         and receipt.get('prediction_PNGs_verified') == total * 33 and receipt.get('prediction_MP4s_verified') == total
         and receipt.get('GT_PNGs_verified') == 32 * 33 and receipt.get('GT_MP4s_verified') == 32
         and receipt.get('training_updates') == 0 and receipt.get('training_only_checkpoint_roles_verified') is True
         and receipt.get('all_three_arms_retained') is True and receipt.get('weight_or_checkpoint_selection') is False
         and receipt.get('validation_not_heldout_test') is (phase == 'val'), 'Require full native TRD verification receipt')
    summary = store.json(run / 'summary.json'); summary_sha = digest(store.raw[run / 'summary.json'])
    need(summary_sha == state['summary_sha256'] == receipt.get('summary_sha256')
         and summary.get('status') == 'completed_clean_evaluation' and summary.get('protocol_sha256') == psha
         and summary.get('restores') == total and summary.get('training_updates') == 0
         and summary.get('roles') == len(protocol['roles']) and summary.get('phase') == phase
         and summary.get('quality_gate_applied') is False, 'Native summary receipt chain differs')
    checked_sha(receipt.get('observations_sha256'))

    # Gate above is complete. Sources are text; external checkpoint/media/tensor
    # declarations remain references only and are never opened by this helper.
    sources = protocol['sources']
    expected_names = {'__init__.py', 'checkpoints.py', 'data.py', 'evaluate.py', 'infer.py', 'losses.py',
                      'metrics.py', 'model.py', 'teacher.py', 'train.py', 'vae.py'}
    need(set(sources) == expected_names, 'Require all eleven frozen sources')
    for name, binding in sources.items():
        path = source_root / 'vsr' / name
        need(binding['path'] == str(path), 'Source escaped the declared source root')
        raw = store.read(path)
        need(len(raw) == binding['bytes'] and digest(raw) == binding['sha256'], 'Frozen source binding changed')

    observations_raw = store.read(run / 'observations.json')
    need(digest(observations_raw) == receipt['observations_sha256'], 'Native observations binding changed')
    aggregate = parse(observations_raw)
    need(set(aggregate) == {'observations', 'protocol_sha256'} and aggregate['protocol_sha256'] == psha,
         'Observation aggregate protocol differs')
    observations = aggregate['observations']
    need(isinstance(observations, list) and len(observations) == total, 'Missing/extra observations')
    exported = {str(p.relative_to(run)): p for p in (run / 'status.json', run / 'protocol.json', run / 'summary.json', run / 'observations.json')}
    seen, ordinal = set(), 0
    for role in protocol['roles']:
        for item in protocol['cohort']:
            for sample in (42, 43):
                observation = observations[ordinal]; ordinal += 1
                wanted = identity(protocol, role, item, sample)
                key = role['label'], item['record']['id'], sample
                need(key not in seen, 'Duplicate role/clip/seed observation'); seen.add(key)
                directory = run / 'results' / role['label'] / item['record']['id'] / f'seed_{sample}'
                metrics_path, marker_path = directory / 'metrics.json', directory / 'complete.json'
                need(observation.get('row_metrics_path') == str(metrics_path), 'Row metric path escaped canonical directory')
                raw = store.read(metrics_path); msha = digest(raw)
                need(observation.get('row_metrics_sha256') == msha, 'Observation metric SHA changed')
                row, marker = parse(raw), store.json(marker_path)
                need(marker.get('protocol_sha256') == psha and marker.get('metrics_sha256') == msha
                     and marker.get('PNG_count') == 33, 'Row complete marker differs')
                pool = row['full_frame_pair_metrics']['all_frames']
                scalars = {name: pool[name] for name in METRICS}
                need(all(finite(v) for v in scalars.values()) and pool['frame_count'] == 33 and len(pool['per_frame']) == 33
                     and pool['temporal_pair_count'] == 28
                     and {tuple(p['indices']) for p in pool['per_temporal_pair']} == {(i, i + 1) for i in range(2, 30)},
                     'Incomplete/nonfinite native metric pool')
                wanted = wanted | {'metrics': scalars}
                need(row.get('status') == 'completed' and row.get('training_updates') == 0 and row['observation'] == wanted
                     and observation == wanted | {'row_metrics_path': str(metrics_path), 'row_metrics_sha256': msha},
                     'Missing/duplicate/reordered role/clip/seed or mismatched metrics')
                need(row['source_metadata']['split'] == phase and row['source_metadata']['source_frame_indices'] == item['source_indices']
                     and row['teacher_or_HR_input_to_inferrer'] is False
                     and row['inference_inputs'] == ['LRV', 'native text embedding', 'student checkpoint']
                     and row['restoration']['inference_inputs'] == ['LRV', 'native text embedding'], 'Native window/inference provenance differs')
                pngs, video = row['PNG_artifacts'], row['video_artifact']
                need(len(pngs) == 33 and len({p['path'] for p in pngs}) == 33, 'Missing/duplicate prediction declarations')
                for artifact in pngs:
                    reference(artifact['path'], directory / 'png', '.png'); checked_sha(artifact['sha256'])
                need(video == marker.get('video_artifact') and video['path'] == str(directory / 'prediction.mp4'), 'MP4 declaration differs')
                reference(video['path'], directory, '.mp4'); checked_sha(video['sha256'])
                for path in (metrics_path, marker_path): exported[str(path.relative_to(run))] = path
    need(len(seen) == total, 'Incomplete matrix identity coverage')

    role_audits_path = run / 'role_audits.json'
    role_audits = store.json(role_audits_path)
    need(role_audits == [{'role': r['label'], 'restores': 64, 'training_updates': 0, 'inference_weights': 'student'} for r in protocol['roles']], 'Incomplete role summary audit')
    exported['role_audits.json'] = role_audits_path
    for role in protocol['roles']:
        directory = no_symlinks(run / 'roles' / role['label'])
        state_paths, load_paths = [], []
        for path in directory.iterdir():
            if re.fullmatch(r'state_audit_[0-9]{3}\.json', path.name): state_paths.append(path)
            elif re.fullmatch(r'load_audit_[0-9]{3}\.json', path.name): load_paths.append(path)
        need(state_paths and load_paths, 'Missing persisted load/state audits')
        for path in sorted(state_paths):
            value = store.json(path)
            need(value['role'] == role['label'] and value['trainable_before'] == value['trainable_after']
                 and value['training_updates'] == 0 and value['inference_weights'] == 'student' and value['restores_expected'] == 64,
                 'Student state audit differs')
            exported[str(path.relative_to(run))] = path
        for path in sorted(load_paths):
            value = store.json(path)
            need(value['step'] == role['step'] and value['checkpoint_sha256'] == role['checkpoint']['sha256'] and value['config'] == role['config']
                 and value.get('training_only_state_restored') is False and value.get('EMA_weights_selected') is False, 'Student-only load audit differs')
            if role['update']:
                expected = dict(schema='vsr_trd_training_state_v1', protocol_sha256=protocol['training_evidence']['protocol_sha256'],
                                arm=role['arm'], completed_updates=role['update'], training_seed=20261102,
                                weight=role['weight'], student_layer=9, EMA_updates=role['update'] if role['arm'] == 'ema' else 0)
                need(value.get('trd_training_metadata') == expected, 'TRD checkpoint metadata differs')
            exported[str(path.relative_to(run))] = path
    for item in protocol['cohort']:
        path = run / 'ground_truth' / item['record']['id'] / 'complete.json'
        value = store.json(path)
        need(value['protocol_sha256'] == psha and value['record_id'] == item['record']['id']
             and value['source_id'] == item['record']['source_id'] and value['frames'] == 33, 'Ground-truth text marker differs')
        exported[str(path.relative_to(run))] = path
    for name in ('runtime.json', 'progress.json', 'execution_wall_time.json', 'console.log'):
        path = run / name
        if name == 'runtime.json' or path.exists() or path.is_symlink():
            if path.suffix == '.json': store.json(path)
            else: store.read(path)
            exported[name] = path
    attempts = no_symlinks(run / 'attempts')
    need(attempts.is_dir(), 'Missing executed attempt records')
    attempt_count = 0
    for path in sorted(attempts.iterdir()):
        if re.fullmatch(r'(attempt|GPU_admission|failure)_[0-9]{3}\.json', path.name):
            value = store.json(path)
            if path.name.startswith(('attempt_', 'failure_')):
                need(value.get('protocol_sha256') == psha, 'Attempt protocol binding changed')
            exported[str(path.relative_to(run))] = path
            attempt_count += path.name.startswith('attempt_')
    need(attempt_count > 0, 'Missing real execution attempt')
    exported['verification/native_verify.json'] = verification
    store.unchanged()
    return store, exported, dict(phase=phase, restores=total, protocol_sha256=psha,
                                 observations_sha256=receipt['observations_sha256'], summary_sha256=summary_sha)


def utf8_chunks(raw):
    start = 0
    while start < len(raw):
        end = min(start + CHUNK_BYTES, len(raw))
        while end < len(raw) and raw[end] & 0xC0 == 0x80:
            end -= 1
        need(end > start, 'Invalid UTF8 chunk boundary')
        part = raw[start:end]; decode(part)
        yield part
        start = end


def export(run, verification, source_root, dest=None):
    source_root = no_symlinks(source_root)
    store, files, evidence = collect(run, verification, source_root)
    run = absolute(run)
    base = no_symlinks(source_root / 'docs/ema_teacher_targets_20261108/experiments')
    dest = no_symlinks(dest or base / f"{evidence['phase']}_{run.name}_{evidence['protocol_sha256'][:12]}")
    need(dest.parent == base and dest.name.startswith(evidence['phase'] + '_') and SAFE_NAME.fullmatch(dest.name), 'Use a unique phase-prefixed child of this source study experiments directory')
    need(not dest.exists(), 'Text export destination already exists; never overwrite')
    store.unchanged()
    base.mkdir(parents=True, exist_ok=True); no_symlinks(base)
    dest.mkdir()
    entries = []
    for relative, path in sorted(files.items()):
        raw = store.raw[path]
        entry = dict(path=relative, original_source=str(path), bytes=len(raw), sha256=digest(raw))
        if len(raw) < SINGLE_LIMIT:
            target = dest / relative; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(raw)
            entry['storage'] = 'copy'
        else:
            entry['storage'] = 'utf8_byte_chunks'; entry['parts'] = []
            for ordinal, part in enumerate(utf8_chunks(raw), 1):
                name = '.text_parts/' + relative + f'/part_{ordinal:05d}.txt'
                target = dest / name; target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream: stream.write(part)
                entry['parts'].append(dict(path=name, bytes=len(part), sha256=digest(part)))
        entries.append(entry)
    store.unchanged()
    manifest = dict(schema='wan_trd_complete_evaluation_text_export_v1', status='complete_verified_text_export',
                    run=str(run), verification_receipt=str(verification), source_root=str(source_root),
                    evidence=evidence, files=entries, source_files_checked_not_exported=11,
                    complete_observations_preserved=True, per_row_metrics_and_markers_preserved=evidence['restores'],
                    checkpoint_or_tensor_files_opened=False, media_files_opened=False, PNG_hashes_recomputed=False,
                    source_bytes_and_CRLF_preserved=True, UTF8_chunk_bytes=CHUNK_BYTES,
                    unknown_files_excluded=True, new_training_or_restores=0)
    with (dest / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write('\n')
    (dest / 'README.md').write_text('''# 完整 Wan TRD 评价文本\n\n只有 completed640-val 或192-test及保存的原生trd-verify收据通过后才导出。包含完整observations、每行metrics/complete、每role载入/状态审计、GT文本marker和根/attempt白名单文本。所有来源文件按原字节保存（含CRLF）；未知文件、媒体、权重和tensor不读取也不复制。媒体已由原生verify检查，本工具不重复hash PNG。\n\nmanifest.json给出每个原始相对路径、完整SHA和字节数。storage=copy直接复制；storage=utf8_byte_chunks按parts列表顺序以二进制方式拼接，逐part核SHA，再核完整bytes/SHA即可恢复原文件字节。分块保持有效UTF8，每块最多800000字节，适合受限GitHub上传payload。文本中的模型/媒体路径和hash是引用，不代表包含对应二进制。\n''')
    return dict(status=manifest['status'], destination=str(dest), phase=evidence['phase'], restores=evidence['restores'],
                exported_text_files=len(entries), manifest_sha256=digest((dest / 'manifest.json').read_bytes()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--verification-json', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dest', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.run_dir, args.verification_json, args.source_root, args.dest), indent=2))


if __name__ == '__main__':
    main()
