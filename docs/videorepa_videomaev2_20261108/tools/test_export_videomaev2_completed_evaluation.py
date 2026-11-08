import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HELPER = Path('/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/export_videomaev2_completed_evaluation.py')
spec = importlib.util.spec_from_file_location('completed_videomaev2_export', HELPER)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\r\n').encode())


def fixture(root, phase='test', leading_hyphen=False):
    root = Path(root); run = root / 'run'; source = root / 'source'; verify = root / 'native_verify.json'
    run.mkdir(); source.mkdir()
    source_names = ('__init__.py', 'checkpoints.py', 'data.py', 'evaluate.py', 'infer.py', 'losses.py', 'metrics.py', 'model.py', 'teacher.py', 'train.py', 'vae.py')
    sources = {}
    for name in source_names:
        path = source / 'vsr' / name; path.parent.mkdir(exist_ok=True)
        raw = ('# synthetic ' + name + '\r\n').encode(); path.write_bytes(raw)
        sources[name] = dict(path=str(path), bytes=len(raw), sha256=module.digest(raw))
    steps = (512, 1024, 2048) if phase == 'val' else (2048,)
    roles = []
    if phase == 'val':
        roles.append(dict(label='shared0', arm='shared', update=0, step=8192, weight=0., family='baseline', setting='shared_initial', cumulative_training_seconds=0., checkpoint={'sha256': '0' * 64}, config={'synthetic': True}))
    for arm in ('baseline', 'videomaev2'):
        for step in steps:
            roles.append(dict(label=arm + str(step), arm=arm, update=step, step=step, weight=0. if arm == 'baseline' else .5, family=arm, setting='common', cumulative_training_seconds=float(step), checkpoint={'sha256': str(len(roles) % 10) * 64}, config={'synthetic': True}))
    cohort = [dict(record=dict(id=f'clip{i:03d}', source_id=f'source{i:03d}'), source_indices=list(range(33))) for i in range(32)]
    if leading_hyphen: cohort[0]['record']['id'] = '-mSvhSt_Q8g_85_92to239'
    total = len(roles) * 64
    protocol = dict(schema=module.SCHEMA, phase=phase, sources=sources, roles=roles, cohort=cohort, cohort_seal_digest='d' * 64,
                    training_evidence={'protocol_sha256': 'a' * 64},
                    scope=dict(total_restores=total, roles=len(roles), clips=32, sampling_seeds=[42, 43], conditions=['clean'], source_split=phase, frames=33, training_updates=0, weight_or_checkpoint_selection=False))
    write(run / 'protocol.json', protocol); psha = module.digest((run / 'protocol.json').read_bytes())
    observations = []
    for role in roles:
        for item in cohort:
            for seed in (42, 43):
                directory = run / 'results' / role['label'] / item['record']['id'] / f'seed_{seed}'
                scalars = dict.fromkeys(module.METRICS, .123)
                obs = module.identity(protocol, role, item, seed) | {'metrics': scalars}
                video = dict(path=str(directory / 'prediction.mp4'), bytes=123, sha256='b' * 64)
                pool = scalars | dict(frame_count=33, per_frame=[{}] * 33, temporal_pair_count=28,
                                      per_temporal_pair=[{'indices': [i, i + 1]} for i in range(2, 30)])
                row = dict(status='completed', training_updates=0, observation=obs, full_frame_pair_metrics={'all_frames': pool},
                           source_metadata={'split': phase, 'source_frame_indices': list(range(33))},
                           teacher_or_HR_input_to_inferrer=False, inference_inputs=['LRV', 'native text embedding', 'student checkpoint'],
                           restoration={'inference_inputs': ['LRV', 'native text embedding']},
                           PNG_artifacts=[dict(path=str(directory / 'png' / f'{i:06d}.png'), bytes=123, sha256='c' * 64) for i in range(33)], video_artifact=video)
                write(directory / 'metrics.json', row); msha = module.digest((directory / 'metrics.json').read_bytes())
                write(directory / 'complete.json', dict(protocol_sha256=psha, metrics_sha256=msha, PNG_count=33, video_artifact=video))
                observations.append(obs | dict(row_metrics_path=str(directory / 'metrics.json'), row_metrics_sha256=msha))
        directory = run / 'roles' / role['label']
        write(directory / 'state_audit_001.json', dict(role=role['label'], trainable_before={'digest': 'same'}, trainable_after={'digest': 'same'}, training_updates=0, inference_weights='student', restores_expected=64))
        load = dict(step=role['step'], checkpoint_sha256=role['checkpoint']['sha256'], config=role['config'], training_only_state_restored=False, EMA_weights_selected=False)
        if role['update']:
            load['videomaev2_training_metadata'] = dict(schema='vsr_videomaev2_training_state_v1', protocol_sha256='a' * 64, arm=role['arm'], completed_updates=role['update'], training_seed=20261102, weight=role['weight'], student_layer=17, EMA_updates=0, teacher_parameter_updates=0, teacher_forward_count=role['update'] if role['arm'] == 'videomaev2' else 0)
        write(directory / 'load_audit_001.json', load)
    for item in cohort:
        write(run / 'ground_truth' / item['record']['id'] / 'complete.json', dict(protocol_sha256=psha, record_id=item['record']['id'], source_id=item['record']['source_id'], frames=33))
    write(run / 'observations.json', dict(observations=observations, protocol_sha256=psha))
    write(run / 'role_audits.json', [dict(role=r['label'], restores=64, training_updates=0, inference_weights='student') for r in roles])
    write(run / 'runtime.json', {'synthetic': 'Only CPU fixture; never a real verification receipt'})
    write(run / 'attempts/attempt_001.json', dict(protocol_sha256=psha, resume_requested=False))
    write(run / 'attempts/GPU_admission_001.json', {'synthetic': True})
    write(run / 'summary.json', dict(status='completed_clean_evaluation', protocol_sha256=psha, restores=total, training_updates=0, roles=len(roles), phase=phase, quality_gate_applied=False))
    ssha = module.digest((run / 'summary.json').read_bytes())
    write(run / 'status.json', dict(status='completed', protocol_sha256=psha, restores=total, training_updates=0, summary_sha256=ssha))
    write(verify, dict(status='verified_complete_clean_evaluation', run=str(run), protocol_sha256=psha, restores=total, prediction_PNGs_verified=total * 33, prediction_MP4s_verified=total,
                       GT_PNGs_verified=32 * 33, GT_MP4s_verified=32, training_updates=0, training_only_checkpoint_roles_verified=True, both_arms_retained=True, weight_or_checkpoint_selection=False,
                       validation_not_heldout_test=phase == 'val', summary_sha256=ssha, observations_sha256=module.digest((run / 'observations.json').read_bytes())))
    return run, verify, source


def manifest_entries(target, manifest):
    if manifest['file_entries_storage'] == 'inline':
        entries = manifest['files']
    else:
        assert manifest['files'] == []
        entries = []
        for part in manifest['manifest_parts']:
            raw = (target / part['path']).read_bytes()
            assert len(raw) == part['bytes'] <= module.CHUNK_BYTES
            assert module.digest(raw) == part['sha256']
            values = json.loads(raw); assert len(values) == part['entries']
            entries.extend(values)
    assert len(entries) == manifest['file_entries_count']
    return entries


def update_observations_binding(run, verify, value=None):
    if value is not None: write(run / 'observations.json', value)
    receipt = json.loads(verify.read_text()); receipt['observations_sha256'] = module.digest((run / 'observations.json').read_bytes()); write(verify, receipt)


class ExportTests(unittest.TestCase):
    def test_leading_hyphen_clip_supported_without_allowing_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, verify, source = fixture(tmp, leading_hyphen=True)
            result = module.export(run, verify, source)
            self.assertEqual(result['restores'], 128)
            original = json.loads((run / 'protocol.json').read_text())
            for name in ('', '..', '../escape', 'a/b', 'a\\b'):
                with self.subTest(unsafe_clip=name):
                    protocol = json.loads(json.dumps(original)); protocol['cohort'][0]['record']['id'] = name
                    with self.assertRaisesRegex(ValueError, 'Unsafe/duplicate clip'):
                        module.protocol_shape(protocol)

    def test_complete_val448_and_test128_preserve_every_text_byte(self):
        for phase in ('val', 'test'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                run, verify, source = fixture(tmp, phase)
                result = module.export(run, verify, source)
                self.assertEqual(result['restores'], 448 if phase == 'val' else 128)
                target = Path(result['destination']); manifest = json.loads((target / 'manifest.json').read_text())
                self.assertEqual(len([e for e in manifest_entries(target, manifest) if e['path'].endswith('/metrics.json')]), result['restores'])
                for entry in manifest_entries(target, manifest):
                    raw = (target / entry['path']).read_bytes() if entry['storage'] == 'copy' else b''.join((target / p['path']).read_bytes() for p in entry['parts'])
                    self.assertEqual(raw, Path(entry['original_source']).read_bytes())
                    self.assertLessEqual((target/'manifest.json').stat().st_size, module.CHUNK_BYTES)
                    self.assertEqual(module.digest(raw), entry['sha256'])
                with self.assertRaisesRegex(ValueError, 'destination already exists'):
                    module.export(run, verify, source)

    def test_running_rejected_before_protocol_receipt_or_quality_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, source = root / 'run', root / 'source'; run.mkdir(); source.mkdir(); write(run / 'status.json', {'status': 'running'})
            original = module.TextStore.read; opened = []
            def guarded(store, path):
                opened.append(Path(path)); return original(store, path)
            with mock.patch.object(module.TextStore, 'read', guarded), self.assertRaisesRegex(ValueError, 'must be completed'):
                module.export(run, root / 'missing_verify.json', source)
            self.assertEqual(opened, [run / 'status.json'])

    def test_wrong_schema_budget_weights_and_teacher_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, verify, source = fixture(tmp)
            original = json.loads((run/'protocol.json').read_text())
            for key, value in (('schema', 'wan_trd_evaluation_protocol_v1'), ('phase', 'validation')):
                wrong = json.loads(json.dumps(original)); wrong[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError): module.protocol_shape(wrong)
            for edit in ('budget', 'weight', 'order'):
                wrong = json.loads(json.dumps(original))
                if edit == 'budget': wrong['scope']['total_restores'] = 192
                elif edit == 'weight': wrong['roles'][1]['weight'] = .3
                else: wrong['roles'] = wrong['roles'][::-1]
                with self.subTest(edit=edit), self.assertRaises(ValueError): module.protocol_shape(wrong)
            path = run/'roles/videomaev22048/load_audit_001.json'
            load = json.loads(path.read_text()); load['videomaev2_training_metadata']['teacher_forward_count'] = 2047
            write(path, load)
            with self.assertRaisesRegex(ValueError, 'checkpoint metadata differs'): module.export(run, verify, source)

    def test_missing_native_receipt_rejected_before_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, verify, source = fixture(tmp); verify.unlink()
            original = module.TextStore.read; opened = []
            def guarded(store, path):
                opened.append(str(path)); return original(store, path)
            with mock.patch.object(module.TextStore, 'read', guarded), self.assertRaises(FileNotFoundError):
                module.export(run, verify, source)
            self.assertFalse(any('observations' in p or p.endswith('/metrics.json') for p in opened))

    def test_missing_and_duplicate_observations_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as tmp:
                run, verify, source = fixture(tmp)
                value = json.loads((run / 'observations.json').read_text())
                if duplicate: value['observations'][1] = value['observations'][0]
                else: value['observations'].pop()
                update_observations_binding(run, verify, value)
                with self.assertRaises(ValueError): module.export(run, verify, source)

    def test_duplicate_temporal_pair_rejected_even_when_declared_count_and_set_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, verify, source = fixture(tmp)
            aggregate = json.loads((run / 'observations.json').read_text())
            observation = aggregate['observations'][0]
            metrics = Path(observation['row_metrics_path'])
            row = json.loads(metrics.read_text())
            pool = row['full_frame_pair_metrics']['all_frames']
            pool['per_temporal_pair'].append(dict(pool['per_temporal_pair'][0]))
            self.assertEqual(pool['temporal_pair_count'], 28)
            self.assertEqual(len({tuple(p['indices']) for p in pool['per_temporal_pair']}), 28)
            write(metrics, row); msha = module.digest(metrics.read_bytes())
            observation['row_metrics_sha256'] = msha
            marker = metrics.with_name('complete.json'); value = json.loads(marker.read_text())
            value['metrics_sha256'] = msha; write(marker, value)
            update_observations_binding(run, verify, aggregate)
            with self.assertRaisesRegex(ValueError, 'Incomplete/nonfinite native metric pool'):
                module.export(run, verify, source)

    def test_path_escape_rejected_before_external_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, verify, source = fixture(tmp)
            value = json.loads((run / 'observations.json').read_text()); escaped = Path(tmp) / 'not_allowed.json'
            value['observations'][0]['row_metrics_path'] = str(escaped)
            update_observations_binding(run, verify, value)
            with self.assertRaisesRegex(ValueError, 'escaped canonical'): module.export(run, verify, source)

    def test_symbolic_linked_whitelist_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, verify, source = fixture(tmp)
            obs = json.loads((run / 'observations.json').read_text())['observations'][0]; path = Path(obs['row_metrics_path'])
            original = path.with_name('original.json'); path.rename(original); path.symlink_to(original)
            with self.assertRaisesRegex(ValueError, 'Symbolic links'): module.export(run, verify, source)

    def test_changed_metric_SHA_and_marker_rejected(self):
        for marker in (False, True):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                run, verify, source = fixture(tmp)
                first = json.loads((run / 'observations.json').read_text())['observations'][0]
                if marker:
                    path = Path(first['row_metrics_path']).with_name('complete.json'); value = json.loads(path.read_text()); value['metrics_sha256'] = '0' * 64; write(path, value)
                else:
                    with Path(first['row_metrics_path']).open('ab') as stream: stream.write(b' \r\n')
                with self.assertRaisesRegex(ValueError, 'SHA changed|marker differs'): module.export(run, verify, source)

    def test_unknown_binary_and_unknown_JSON_never_opened_or_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, verify, source = fixture(tmp)
            unknown = [run / 'checkpoint.pt', run / 'credentials.json', run / 'roles/baseline2048/unknown.tensor', run / 'attempts/unknown.json']
            for path in unknown: path.write_bytes(b'\x00\xffnot text')
            real_open = os.open
            def guarded(path, flags, *args, **kwargs):
                self.assertNotIn(Path(path), unknown)
                self.assertNotIn(Path(path).suffix, ('.pt', '.png', '.mp4', '.tensor'))
                return real_open(path, flags, *args, **kwargs)
            with mock.patch.object(module.os, 'open', guarded): result = module.export(run, verify, source)
            target = Path(result['destination']); manifest = json.loads((target / 'manifest.json').read_text())
            self.assertTrue(all(entry['original_source'] not in {str(p) for p in unknown} for entry in manifest_entries(target, manifest)))

    def test_large_observations_lossless_UTF8_and_CRLF_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, verify, source = fixture(tmp)
            path = run / 'observations.json'
            with path.open('ab') as stream: stream.write(b'\r\n' * 600_000)
            update_observations_binding(run, verify)
            result = module.export(run, verify, source); target = Path(result['destination'])
            manifest = json.loads((target / 'manifest.json').read_text()); entry = next(e for e in manifest_entries(target, manifest) if e['path'] == 'observations.json')
            self.assertEqual(entry['storage'], 'utf8_byte_chunks')
            parts = [(target / p['path']).read_bytes() for p in entry['parts']]
            self.assertEqual(b''.join(parts), path.read_bytes())
            self.assertTrue(all(len(part) <= module.CHUNK_BYTES for part in parts))
            for part in parts: part.decode('utf-8')
            unicode_raw = ('你好🙂\r\n' * 150_000).encode()
            self.assertEqual(b''.join(module.utf8_chunks(unicode_raw)), unicode_raw)


if __name__ == '__main__':
    unittest.main(verbosity=2)
