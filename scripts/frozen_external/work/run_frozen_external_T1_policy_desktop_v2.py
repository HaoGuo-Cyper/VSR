"""Fresh single external phase under the root-frozen numerical protocol.

Only real VAE construction and public per-restore policy rotation are wrapped.
Import and metadata fixtures do not import Torch or open source pixels. A real
run requires explicit root phase-code seal and predecessor-completion review.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
import weakref
import zipfile

P = Path('/home/hugo/Projects/VSR')
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
FINAL_PROTOCOL_SHA = '418e9e36cf75b8929b63aaf0eaa5222c38fb4ad485b55f012a4d558aa7b2f746'
HQ_HELPER = W/'work/run_frozen_HQ_test_T1_policy.py'
HQ_HELPER_SHA = '6304f49d5a30ad0a26fd8ae5e8a52d782dd12ee5dd6148b87b099f03d93c50c7'
COMMON = W/'work/probe_global_HRI_encode_allocator.py'
COMMON_SHA = 'f6f1da551bdeff048253ec86aeb5679e7a5cfa609b0023dc92029664f8446514'
QA_HELPER = W/'work/run_native_QA_T1_policy.py'
QA_HELPER_SHA = 'dd99144dace5f9c7543881ad6c6daf2718936ea17d6ab639729a09fb525100e0'
POLICY = W/'work/t1_vae_cache_policy.py'
POLICY_SHA = 'd94340b33868912f43633c71ef4a34a97b2dd7eb7b74e9d40ded363b09417270'
ACCEPTED = W/'outputs/VSR_evaluation_primary_checkpoint_cohort_acceptance_v1.json'
ACCEPTED_SHA = 'c2e054cd00c2eca25a8c190bee4282ea1af7c0d272c36453bbe20f2d3a3a3557'
CAPTION = W/'outputs/VSR_final_evaluation_text_cache_preflight.json'
CAPTION_SHA = 'c903115c195e5f8f818c46743e82d8233309c92400fb8cdad964308263a1568e'
CAPACITY = W/'outputs/VSR_external_metadata_capacity_audit.json'
CAPACITY_SHA = 'ecc8f8081c1a737fa85f3d73c9d6e0e1c4fc8e86803f7301074e2865f7bf2090'
ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'PYTORCH_ALLOC_CONF': 'expandable_segments:True', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
PHASE_DATASETS = {'paired': ('REDS4', 'YouHQ40'), 'real': ('VideoLQ',)}
LIFECYCLE = 'public_remove_reinstall_same_VAE_after_each_successful_restore'
STARTUP_GUARD = W/'work/external_startup_gpu_guard_v2.py'


def need(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    need(isinstance(value, dict), 'Expected JSON object: ' + str(path))
    return value


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def phase_selection_note(phase, outer_contract):
    need(phase in PHASE_DATASETS and re.fullmatch('[0-9a-f]{64}', outer_contract), 'Explicit phase/outer execution contract required')
    return ('all accepted original8; immutable raw protocol=' + FINAL_PROTOCOL_SHA + '; phase=' + phase
        + '; frozen outer execution_policy_sha256=' + outer_contract)


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def inside(path, root):
    value = Path(path).resolve()
    need(value.is_relative_to(Path(root).resolve()), 'Path escaped declared root: ' + str(value))
    return value


def tile_positions(length, requested, overlap=64):
    """Metadata copy of pinned inference.tile_positions, verified by fixtures."""
    need(isinstance(length, int) and length > 0 and length % 16 == 0, 'Padded extent must be16-aligned')
    tile = min(length, requested)
    if length <= tile:
        return [0], tile
    step = tile - overlap
    need(step > 0, 'Invalid tile overlap')
    positions = list(range(0, length-tile+1, step))
    if positions[-1] != length-tile:
        positions.append(length-tile)
    return positions, tile


def geometry(lr_hw):
    need(len(lr_hw) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in lr_hw), 'Invalid native LR geometry')
    hr = [x*4 for x in lr_hw]; padded = [((x+15)//16)*16 for x in hr]
    ys, th = tile_positions(padded[0], 256); xs, tw = tile_positions(padded[1], 448)
    return {'LR_hw': list(lr_hw), 'HR_hw': hr, 'padded_HR_hw': padded,
        'tile_count': len(ys)*len(xs), 'tile_hw_actual': [th, tw]}


def phase_metadata(protocol, phase, manifest, streams, capacity):
    """Existing JSON/header metadata only. No archive/pixel/model access."""
    need(phase in PHASE_DATASETS, 'Choose exactly paired or real')
    ext = protocol['external_native_prefix']
    need(ext['schema'] == 'vsr_final_external_native_prefix_protocol_v1'
        and ext['frames'] == 17 and ext['start'] == 0 and ext['sampling_seeds'] == [42, 43]
        and ext['steps'] == 20 and ext['scale'] == 4 and ext['tile_hw'] == [256, 448]
        and ext['overlap'] == 64 and ext['cfg'] == 1 and ext['spatial_resize_for_metrics'] is False,
        'Frozen external native-prefix numerical protocol changed')
    need(streams['status'] == 'passed', 'All cached paired/header metadata must pass')
    cached = {(r['dataset'], r['id']): r for r in capacity['records']}
    records = []
    for dataset in PHASE_DATASETS[phase]:
        declared = ext['datasets'][dataset]; rows = manifest['datasets'][dataset]['records']
        expected = ['000', '011', '015', '020'] if dataset == 'REDS4' else [f'{x:03d}' for x in range(40 if dataset == 'YouHQ40' else 50)]
        need(declared['ids'] == expected and sorted(r['id'] for r in rows) == expected
            and declared['sequence_count'] == len(expected), 'Missing, duplicate or substituted dataset sequence')
        stream_rows = {r['id']: r for r in streams['datasets'][dataset]['records']}
        need(set(stream_rows) == set(expected), 'Stream catalogue sequence set differs')
        for row in sorted(rows, key=lambda x: x['id']):
            header = stream_rows[row['id']]
            need(header['status'] == 'passed', 'Native stream/header failed')
            lr = header['lq']; hw = [lr['height'], lr['width']]
            if dataset == 'REDS4':
                need(row['frame_count'] >= 17 and len(row['lq']['frames']) == row['frame_count']
                    and len(row['gt']['frames']) == row['frame_count'], 'REDS original frame catalogue incomplete')
            if phase == 'paired':
                gt = header['gt']
                need([gt['height'], gt['width']] == [hw[0]*4, hw[1]*4], 'GT/LQ native x4 shape differs')
            geo = geometry(hw); prior = cached[(dataset, row['id'])]
            need(prior['LR_hw'] == hw and prior['SR_hw'] == geo['HR_hw']
                and prior['padded_SR_hw'] == geo['padded_HR_hw'] and prior['tiles'] == geo['tile_count'],
                'Independent frozen metadata capacity geometry differs')
            records.append({'dataset': dataset, 'sequence_id': row['id'], **geo})
    count = len(records); tiles = sum(r['tile_count'] for r in records)
    need(count == (44 if phase == 'paired' else 50) and tiles == (1228 if phase == 'paired' else 2795), 'Complete fixed phase geometry budget differs')
    nr = {'restores': count*2, 'T1': 0, 'original_delegate': tiles*2, 'events': tiles*2, 'rotations': count*2}
    hr = ({'restores': count*4, 'T1': count*2, 'original_delegate': tiles*4,
        'events': tiles*4+count*2, 'rotations': count*4} if phase == 'paired' else dict(nr))
    totals = {key: 4*nr[key]+4*hr[key] for key in nr}
    need(totals['restores'] == (1056 if phase == 'paired' else 800), 'Unexpected inference budget')
    return {'phase': phase, 'datasets': list(PHASE_DATASETS[phase]), 'records': records,
        'sequence_count': count, 'sum_tiles': tiles, 'expected_per_role': {'no_reference': nr, 'reference': hr},
        'expected_totals': totals, 'geometry_source': 'frozen manifest/cached ffprobe headers and independent capacity JSON; actual PNG geometry rechecked by original reader'}


def validate_stage_seal(seal, protocol_path, protocol_sha, phase, worker_path, worker_sha, review_path, review_sha):
    need(seal['schema'] == 'vsr_final_test_execution_seal_v1'
        and seal['status'] == 'frozen_before_test_pixels' and seal['phase'] == phase
        and seal['protocol'] == {'path': str(Path(protocol_path).resolve()), 'sha256': protocol_sha}
        and seal['stage_worker'] == {'path': str(Path(worker_path).resolve()), 'sha256': worker_sha}, 'Phase code seal does not bind exact raw protocol/worker')
    need(seal['prerequisites']['previous_phase_review'] == {'path': str(Path(review_path).resolve()), 'sha256': review_sha}, 'Explicit predecessor review differs from phase seal')
    prior = seal['previous_phase']; prior_run = Path(prior['run'])
    need(prior_run.is_absolute() and prior_run == prior_run.resolve() and prior_run != P/'runs'
        and prior_run.is_relative_to(P/'runs') and isinstance(prior['original_PID'], int)
        and not isinstance(prior['original_PID'], bool) and prior['original_PID'] > 0
        and re.fullmatch(r'[A-Za-z0-9_.@-]+\.service', prior['unit'])
        and re.fullmatch('[0-9a-f]{64}', prior['summary_sha256'])
        and re.fullmatch('[0-9a-f]{64}', prior['execution_policy_sha256']), 'Explicit real predecessor run/unit/PID and both file SHAs required')


def startup_guard_module(seal):
    """Exact separately sealed new metadata guard; old producer guard is not called."""
    source = seal['startup_gpu_guard']['helper']
    need(source['path'] == str(STARTUP_GUARD) and sha(STARTUP_GUARD) == source['sha256'],
        'Explicit new startup helper source binding required')
    guard = module_from(STARTUP_GUARD, 'external_startup_GPU_guard_v2')
    guard.verify_bindings(seal['startup_gpu_guard'])
    return guard


def capture_startup_guard(guard, args, run, save, observations, filename):
    """Persist the complete real/failing observation before enforcing its gate."""
    observation = guard.worker_guard(P, expected_unit=args.unit,
        worker_path=Path(__file__).resolve(), worker_sha256=sha(__file__), run_dir=run,
        phase=args.phase, root_protocol_sha256=FINAL_PROTOCOL_SHA, startup_environment=ENVIRONMENT)
    observations.append(observation)
    save(run/filename, observation)
    guard.ensure_passed(observation)
    return observation


def validate_predecessor_metadata(protocol, phase, predecessor, execution, status, summary,
                                  child_protocol, child_status, child_summary, review):
    expected_stage = 'HQ_internal_test' if phase == 'paired' else 'external_paired'
    need(review['schema'] == 'vsr_frozen_phase_completion_review_v1' and review['status'] == 'passed'
        and review['blocking_findings'] == [] and review['stage'] == expected_stage
        and review['run'] == predecessor['run'] and review['summary_sha256'] == predecessor['summary_sha256']
        and review['root_protocol_sha256'] == FINAL_PROTOCOL_SHA and review['next_phase'] == phase
        and review.get('quality_used_for_gate', False) is False, 'Need explicit root-reviewed completeness/source gate, not a quality threshold')
    need(status['status'] == summary['status'] == 'passed'
        and summary['training_updates'] == 0 and summary['test_used_for_selection'] is False
        and status['summary_sha256'] == predecessor['summary_sha256']
        and execution['root_protocol_sha256'] == execution['root_protocol_file_sha256'] == FINAL_PROTOCOL_SHA
        and execution['root_protocol'] == protocol
        and execution['code106_sha256'] == protocol['bindings']['producer']['code_sha256']
        and execution['policy_sha256'] == POLICY_SHA
        and status['protocol_sha256'] == summary['protocol_sha256'] == digest(execution), 'Predecessor raw protocol/source/policy or successful outer contract differs')
    need(child_status['status'] == child_summary['status'] == 'completed'
        and child_summary['training_updates'] == child_protocol['training_updates'] == 0
        and child_summary['best_seed_selected'] is False
        and child_status['protocol_sha256'] == child_summary['protocol_sha256'] == digest(child_protocol)
        and child_summary['resumed_complete_sequences'] == [] and child_protocol['seeds'] == [42, 43]
        and child_protocol['steps'] == 20 and child_protocol['tile_hw'] == [256, 448] and child_protocol['overlap'] == 64,
        'Predecessor child completion or sampling protocol differs')
    modules = {Path(name).name: value for name, value in protocol['bindings']['producer']['code_sha256'].items()
               if len(Path(name).parts) == 2 and Path(name).parts[0] == 'vsr' and name.endswith('.py')}
    need(child_protocol['code_sha256'] == modules, 'Predecessor actual evaluator module source differs')
    need(set(child_protocol['checkpoints']) == set(protocol['checkpoints']), 'Predecessor accepted8 cohort incomplete')
    for label, spec in protocol['checkpoints'].items():
        actual = child_protocol['checkpoints'][label]
        need(actual['path'] == spec['path'] and actual['checkpoint_sha256'] == spec['checkpoint_sha256']
            and actual['variant'] == spec['variant'] and actual['step'] == 8192, 'Predecessor actual checkpoint differs')
    if phase == 'paired':
        expected_jobs = {name: protocol['hq_internal_test']['record_ids'] for name in protocol['hq_internal_test']['datasets']}
        need(child_protocol['frames'] == 9 and child_protocol['datasets'] == expected_jobs
            and summary['policy_totals']['restores'] == 25320 and summary['policy_totals']['events'] == 33760
            and summary['policy_totals']['T1'] == 8440, 'HQ full held-out phase is incomplete')
    else:
        expected_jobs = {name: protocol['external_native_prefix']['datasets'][name]['ids'] for name in PHASE_DATASETS['paired']}
        need(execution['phase'] == 'paired' and child_protocol['frames'] == 17 and child_protocol['datasets'] == expected_jobs
            and summary['phase'] == 'paired' and summary['policy_totals']['restores'] == 1056
            and summary['policy_totals']['events'] == 29824 and summary['policy_totals']['T1'] == 352,
            'The paired external phase, not merely HQ, must finish before real VideoLQ')
    need(set(child_summary['results']) == set(protocol['checkpoints']) and all(set(groups) == set(expected_jobs)
        and all(value['video_count'] == len(expected_jobs[name]) for name, value in groups.items())
        for groups in child_summary['results'].values()), 'Predecessor per-checkpoint/dataset completion budget differs')


def closed_service(service, *, runner=subprocess.run, pid_exists=lambda value: Path('/proc')/str(value)):
    """Runtime read-only check; callbacks make this CPU-fixture testable."""
    command = runner(['systemctl', '--user', 'show', service['unit'],
        '--property=LoadState,ActiveState,MainPID,ExecMainPID,ExecMainCode,ExecMainStatus'], capture_output=True, text=True)
    props = dict(row.split('=', 1) for row in command.stdout.splitlines() if '=' in row)
    need(command.returncode == 0 or props.get('LoadState') == 'not-found', 'Cannot inspect predecessor service')
    exists = pid_exists(service['PID'])
    exists = exists.exists() if isinstance(exists, Path) else bool(exists)
    need(props.get('LoadState') in {'loaded', 'not-found'} and props.get('ActiveState') == 'inactive'
        and props.get('MainPID') == '0' and not exists, 'Predecessor service/PID has not stopped')
    if props['LoadState'] == 'not-found':
        return {'actual_properties': props, 'actual_last_exit_code_available': False, 'actual_last_exit_code': None,
            'default_ExecMainStatus_is_historical_exit_evidence': False}
    need(props.get('ExecMainPID') == str(service['PID']) and props.get('ExecMainCode') == '1'
        and props.get('ExecMainStatus') == '0', 'Loaded predecessor lacks matching actual exit0')
    return {'actual_properties': props, 'actual_last_exit_code_available': True, 'actual_last_exit_code': 0}


class RotatingExternalRestore:
    """Same mathematical delegate; records metadata and public small ledgers."""
    def __init__(self, restore, install, validate_ledger, metadata, *, finite, contexts_empty, progress=None):
        self.restore, self.install, self.validate_ledger = restore, install, validate_ledger
        self.metadata, self.finite, self.contexts_empty, self.progress = metadata, finite, contexts_empty, progress
        self.roles, self.refs, self.last_audit = [], [], None
        self.geometries = {tuple(r['LR_hw']): r for r in metadata['records']}

    def register(self, vae, config):
        need(all(ref() is None for ref in self.refs), 'Previous role VAE leaked')
        need('encode' not in vae.__dict__ and getattr(vae, '_vsr_t1_cache_policy', None) is None, 'Expected original real VAE encode')
        role = {'role_id': len(self.roles), 'config_sha256': digest(config),
            'reference_enabled': bool(config.get('reference', {}).get('enabled', False)),
            'restores': 0, 'T1': 0, 'original_delegate': 0, 'events': 0, 'rotations': 0, 'request_counts': {}}
        self.roles.append(role); self.refs.append(weakref.ref(vae)); vae._vsr_external_policy_role_id = role['role_id']
        self.install(vae, max_records=6330)
        return vae

    @staticmethod
    def removed(vae):
        return 'encode' not in vae.__dict__ and getattr(vae, '_vsr_t1_cache_policy', None) is None

    def __call__(self, model, vae, lrv, text, config, **options):
        active = getattr(vae, '_vsr_t1_cache_policy', None); role = self.roles[vae._vsr_external_policy_role_id]
        enabled = options.get('reference_image') is not None
        audit = {'status': 'running', 'policy_sha256': POLICY_SHA, 'lifecycle': LIFECYCLE,
            'role_id': role['role_id'], 'max_records': 6330, 'numeric_restore_calls': 0, 'cleanup_errors': []}
        self.last_audit = audit
        try:
            need(active is not None and active.dropped_records == 0 and active.ledger == [], 'Fresh public restore policy required')
            need(role['config_sha256'] == digest(config) and (role['reference_enabled'] or not enabled)
                and (self.metadata['phase'] != 'real' or not enabled), 'Changed family/config or forbidden real HRI')
            shape = list(lrv.shape); geo = self.geometries[tuple(shape[-2:])]
            need(shape == [1, 3, 17, *geo['LR_hw']] and options['steps'] == 20 and options['seed'] in (42, 43)
                and list(options['tile_hw']) == [256, 448] and options['overlap'] == 64
                and options.get('reference_index') == (0 if enabled else None), 'Native prefix/sampling/reference scope changed')
            audit['numeric_restore_calls'] = 1
            video, meta = self.restore(model, vae, lrv, text, config, **options)
            audit['ledger'] = active.ledger
            need(active.dropped_records == 0 and meta['tile_count'] == geo['tile_count']
                and len(meta['tiles']) == geo['tile_count'] and meta['reference']['enabled'] is enabled
                and meta['reference']['encode_count'] == int(enabled) and meta['sampling']['cfg'] == 1.0
                and meta['sampling']['seed'] == options['seed'], 'Actual native restore metadata changed')
            self.validate_ledger(audit['ledger'], enabled, geo['tile_count'])
            need(list(video.shape) == [1, 3, 17, *geo['HR_hw']] and meta['output_shape'] == list(video.shape)
                and self.finite(video) and self.contexts_empty(model), 'Native output geometry/finite/context failure')
            active.remove(); need(self.removed(vae), 'Public remove failed to restore original encode')
            self.install(vae, max_records=6330)
            key = json.dumps([geo['LR_hw'], options['seed'], enabled], separators=(',', ':'))
            role['request_counts'][key] = role['request_counts'].get(key, 0) + 1
            for field, value in {'restores': 1, 'T1': int(enabled), 'original_delegate': geo['tile_count'],
                                 'events': len(audit['ledger']), 'rotations': 1}.items(): role[field] += value
            audit.update(status='passed', observed_T1=int(enabled), observed_original_delegate=geo['tile_count'],
                original_encode_restored_before_reinstall=True, next_policy_installed=True, no_parameter_hash_per_restore=True)
            meta['T1_cache_policy_audit'] = copy.deepcopy(audit)
            if self.progress and sum(r['restores'] for r in self.roles) % 10 == 0: self.progress(self.roles)
            return video, meta
        except BaseException as error:
            audit.update(status='failed', error=f'{type(error).__name__}: {error}')
            try:
                if 'ledger' not in audit and active is not None: audit['ledger'] = active.ledger
            except BaseException as caught: audit['cleanup_errors'].append('partial public ledger: ' + str(caught))
            for candidate in (getattr(vae, '_vsr_t1_cache_policy', None), active):
                if candidate is not None:
                    try: candidate.remove()
                    except BaseException as caught: audit['cleanup_errors'].append('remove: ' + str(caught))
            try: vae.clear_cache()
            except BaseException as caught: audit['cleanup_errors'].append('official clear_cache: ' + str(caught))
            audit['original_encode_restored_after_failure'] = self.removed(vae)
            for note in audit['cleanup_errors']: error.add_note(note)
            raise

    def verify_totals(self):
        need(len(self.roles) == 8 and sum(r['reference_enabled'] for r in self.roles) == 4
            and all(ref() is None for ref in self.refs), 'All8 real role VAEs must be released')
        for role in self.roles:
            family = 'reference' if role['reference_enabled'] else 'no_reference'
            expected = self.metadata['expected_per_role'][family]
            need(all(role[k] == v for k, v in expected.items()), 'Incomplete role numerical/encode budget')
            counts = {}
            modes = (False, True) if family == 'reference' and self.metadata['phase'] == 'paired' else (False,)
            for record in self.metadata['records']:
                for seed in (42, 43):
                    for enabled in modes:
                        key = json.dumps([record['LR_hw'], seed, enabled], separators=(',', ':'))
                        counts[key] = counts.get(key, 0) + 1
            need(role['request_counts'] == counts, 'Incomplete geometry/seed/reference requests')
        return {'status': 'passed', 'roles': self.roles, **self.metadata['expected_totals'], 'all_role_VAEs_collected': True}


def original_frame_catalogue(root, manifest, phase):
    """Archive SHA and ZIP CENTRAL DIRECTORY only; never decode image/video."""
    result = {}
    for name in PHASE_DATASETS[phase]:
        ds = manifest['datasets'][name]; source = ds['source']; original = Path(source['archive_path'])
        owned = root/'data/external/archives'/original.name; archive = owned if owned.is_file() else original
        need(archive.stat().st_size == source['archive_bytes'] and sha(archive) == source['archive_sha256'], 'Original archive bytes changed')
        rows = {}
        if name == 'REDS4':
            for row in ds['records']:
                lr = row['lq']['frames']; hr = row['gt']['frames']
                left = [Path(x['path']).name for x in lr]; right = [Path(x['path']).name for x in hr]
                need(left == right and len(left) >= 17 and left == sorted(set(left), key=lambda x: int(Path(x).stem)), 'Original paired PNG names differ')
                rows[row['id']] = {'count': len(left), 'first17_names': left[:17],
                    'first17_LQ': lr[:17], 'first17_GT': hr[:17]}
        else:
            with zipfile.ZipFile(archive) as z:
                indexed = {}
                for item in z.infolist():
                    parts = Path(item.filename).parts
                    if len(parts) == 4 and parts[0] == name and parts[1] in ('LQ', 'GT') and re.fullmatch(r'\d{3}', parts[2]) and re.fullmatch(r'\d+\.png', parts[3]):
                        key = (parts[2], parts[1]); indexed.setdefault(key, []).append(parts[3])
                for row in ds['records']:
                    lr = sorted(indexed.get((row['id'], 'LQ'), []), key=lambda x: int(Path(x).stem))
                    need(len(lr) >= 17 and len(lr) == len(set(lr)), 'Original LQ PNG directory incomplete/duplicate')
                    if name == 'YouHQ40':
                        hr = sorted(indexed.get((row['id'], 'GT'), []), key=lambda x: int(Path(x).stem))
                        need(lr == hr, 'Original paired PNG frame names differ')
                    rows[row['id']] = {'count': len(lr), 'first17_names': lr[:17]}
        result[name] = {'archive_path': str(archive), 'archive_sha256': source['archive_sha256'],
            'archive_bytes': source['archive_bytes'], 'records': rows, 'pixels_opened': False}
    return result


def completed_rows(child, protocol, phase, metadata, catalogue):
    """JSON row seals and restore metadata only; no image/metric recomputation."""
    actual_protocol, status, summary = (read(child/n) for n in ('protocol.json', 'status.json', 'summary.json'))
    need(status['status'] == summary['status'] == 'completed' and summary['resumed_complete_sequences'] == []
        and status['protocol_sha256'] == summary['protocol_sha256'] == digest(actual_protocol), 'Child frozen evaluation did not complete freshly')
    expected_jobs = {d: protocol['external_native_prefix']['datasets'][d]['ids'] for d in PHASE_DATASETS[phase]}
    need(actual_protocol['datasets'] == expected_jobs and actual_protocol['frames'] == 17 and actual_protocol['seeds'] == [42, 43]
        and actual_protocol['steps'] == 20 and actual_protocol['tile_hw'] == [256, 448] and actual_protocol['overlap'] == 64
        and actual_protocol['metric_device'] == 'cpu' and actual_protocol['device'] in ('cuda', 'cuda:0')
        and actual_protocol['caption'] == 'A high quality video.'
        and set(actual_protocol['checkpoints']) == set(protocol['checkpoints']), 'Actual child source/seed/checkpoint set differs')
    modules = {Path(name).name: value for name, value in protocol['bindings']['producer']['code_sha256'].items()
               if len(Path(name).parts) == 2 and Path(name).parts[0] == 'vsr' and name.endswith('.py')}
    need(actual_protocol['code_sha256'] == modules and summary['training_updates'] == 0
        and summary['external_tuning'] is False and summary['best_seed_selected'] is False,
        'Actual external metric/source implementation or no-selection contract differs')
    seals, count = {}, 0
    for label, spec in protocol['checkpoints'].items():
        actual = actual_protocol['checkpoints'][label]
        need(actual['path'] == spec['path'] and actual['checkpoint_sha256'] == spec['checkpoint_sha256']
            and actual['variant'] == spec['variant'] and actual['step'] == 8192, 'Actual evaluated accepted checkpoint differs')
        for dataset, ids in expected_jobs.items():
            for rid in ids:
                directory = child/'results'/label/dataset/rid; row = read(directory/'metrics.json'); seal = read(directory/'complete.json')
                need(row['status'] == 'completed' and row['label'] == label and row['dataset'] == dataset and row['sequence_id'] == rid
                    and row['checkpoint_sha256'] == spec['checkpoint_sha256'] and row['prefix_frames'] == 17
                    and seal['metrics_sha256'] == sha(directory/'metrics.json')
                    and row['contract_sha256'] == seal['contract_sha256'] == digest({'protocol': digest(actual_protocol), 'label': label, 'dataset': dataset, 'id': rid}), 'Actual completed row seal/identity differs')
                source = row['source']; names = catalogue[dataset]['records'][rid]['first17_names']
                need(source['source_frame_indices'] == list(range(17)) and source['source_frame_names'] == names
                    and source['source_archive_sha256'] == catalogue[dataset]['archive_sha256']
                    and source['spatial_resize'] is False, 'Actual native ordered-frame provenance differs')
                modes = {'hri_on', 'hri_off'} if phase == 'paired' and spec['family'] == 'reference' else {'hri_off'} if spec['family'] == 'reference' else {'no_reference'}
                need(set(row['modes']) == modes, 'Actual paired/reference-missing inference modes differ')
                for mode, values in row['modes'].items():
                    entries = values['seeds']; need(len(entries) == 2 and {r['seed'] for r in entries} == {42, 43}, 'Sampling seeds missing/duplicated')
                    for entry in entries:
                        audit = entry['restoration']['T1_cache_policy_audit']
                        need(audit['status'] == 'passed' and audit['policy_sha256'] == POLICY_SHA and audit['lifecycle'] == LIFECYCLE
                            and audit['numeric_restore_calls'] == 1 and entry['hri_supplied'] is (mode == 'hri_on')
                            and entry['reference_index'] == (0 if mode == 'hri_on' else None) and entry['png_count'] == 17, 'Actual per-restore memory-policy/reference record failed')
                        count += 1
                seals[str((directory/'metrics.json').relative_to(child))] = {'sha256': seal['metrics_sha256'], 'complete_sha256': sha(directory/'complete.json')}
    need(count == metadata['expected_totals']['restores'] and set(summary['results']) == set(protocol['checkpoints'])
        and all(set(v) == set(expected_jobs) and all(v[d]['video_count'] == len(ids) for d, ids in expected_jobs.items()) for v in summary['results'].values()), 'Full phase budget incomplete')
    return {'rows': seals, 'restore_count': count, 'artifact_policy': 'authoritative PNG/MP4 hashes retained in each sealed original metrics row; no pixels recomputed here'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'run-dir', 'execution-seal', 'predecessor-review'): parser.add_argument('--'+name, type=Path, required=True)
    for name in ('expected-protocol-sha', 'expected-execution-seal-sha', 'expected-predecessor-review-sha', 'unit'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--phase', choices=tuple(PHASE_DATASETS), required=True)
    args = parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9_.@-]+\.service', args.unit), 'Explicit direct service unit required')
    need(args.expected_protocol_sha == FINAL_PROTOCOL_SHA and sha(args.protocol) == FINAL_PROTOCOL_SHA, 'Use the identical root-frozen raw numerical protocol')
    for path, expected in ((args.execution_seal, args.expected_execution_seal_sha), (args.predecessor_review, args.expected_predecessor_review_sha)):
        need(re.fullmatch('[0-9a-f]{64}', expected) and sha(path) == expected, 'Explicit root seal/review raw SHA differs')
    seal, protocol, review = read(args.execution_seal), read(args.protocol), read(args.predecessor_review)
    validate_stage_seal(seal, args.protocol, FINAL_PROTOCOL_SHA, args.phase, __file__, sha(__file__), args.predecessor_review, args.expected_predecessor_review_sha)
    run = inside(args.run_dir, P/'runs'); child = run/'external'
    need(run != P/'runs' and (not run.exists() or not any(run.iterdir())), 'Fresh own phase run only; preserve failed/partial runs, no naked child resume')
    run.mkdir(parents=True, exist_ok=True)
    def save(path, value):
        temp = path.with_name(path.name+'.partial'); temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'); temp.replace(path)
    started = time.perf_counter(); contract = None; stage = 'CPU_metadata_preflight'; restore = None
    startup_guard = None; startup_guard_observations = []
    try:
        for path, expected in ((HQ_HELPER, HQ_HELPER_SHA), (COMMON, COMMON_SHA), (QA_HELPER, QA_HELPER_SHA),
            (POLICY, POLICY_SHA), (ACCEPTED, ACCEPTED_SHA), (CAPTION, CAPTION_SHA), (CAPACITY, CAPACITY_SHA)):
            need(sha(path) == expected, 'Frozen helper/evidence changed: '+str(path))
        startup_guard = startup_guard_module(seal)
        hq = module_from(HQ_HELPER, 'vsr_external_HQ_metadata_helper'); common = module_from(COMMON, 'vsr_external_common'); qa = module_from(QA_HELPER, 'vsr_external_resource_helper')
        accepted = read(ACCEPTED); bindings = protocol['bindings']; config, old_failure, environment = common.metadata_preflight()
        members = hq.validate_protocol(protocol, accepted, [r for r in read(P/'manifests/hq_vsr.json')['records'] if r['split'] == 'test'])
        need(bindings['producer']['commit'] == common.COMMIT and bindings['producer']['code_sha256'] == environment['code_sha256']
            and sha(bindings['producer']['environment_path']) == bindings['producer']['environment_sha256'], 'Same producer106 source required')
        ext = protocol['external_native_prefix']
        for field in ('manifest', 'stream_metadata', 'provenance'):
            need(sha(ext[field]['path']) == ext[field]['sha256'], 'Frozen external metadata changed: '+field)
        manifest, streams = read(ext['manifest']['path']), read(ext['stream_metadata']['path'])
        metadata = phase_metadata(protocol, args.phase, manifest, streams, read(CAPACITY))
        resource_proof = qa.verify_resource(bindings['resource']['run'], bindings['resource']['summary_sha256'], environment['code_sha256'])
        qrun = Path(bindings['QA']['run']); qsum = read(qrun/'summary.json')
        need(sha(qrun/'summary.json') == bindings['QA']['summary_sha256'] and qsum['status'] == 'passed'
            and qsum['native_20step_operator_QA_passed'] is True, 'Actual native20step QA source gate missing')
        for name, expected in qsum['child_file_sha256'].items(): need(sha(Path(qsum['child_run'])/name) == expected, 'QA child evidence changed')
        for key in ('CPU_parity', 'caption_preflight', 'Wan_VAE_source'):
            need(sha(bindings[key]['path']) == bindings[key]['sha256'], 'Frozen parity/caption/VAE source changed')
        need(bindings['caption_preflight'] == {'path': str(CAPTION), 'sha256': CAPTION_SHA}
            and bindings['CUDA_parity']['run'] == str(qa.PARITY) and bindings['CUDA_parity']['summary_sha256'] == qa.PARITY_SUMMARY_SHA
            and sha(qa.PARITY/'summary.json') == qa.PARITY_SUMMARY_SHA, 'Real CUDA/CPU parity or all8 caption binding differs')
        pre = seal['previous_phase']; pre_run = inside(pre['run'], P/'runs')
        need(sha(pre_run/'summary.json') == pre['summary_sha256'], 'Explicit predecessor actual summary SHA changed')
        need(sha(pre_run/'execution_policy.json') == pre['execution_policy_sha256'], 'Explicit predecessor execution-policy SHA changed')
        pre_exec, pre_status, pre_summary = (read(pre_run/n) for n in ('execution_policy.json', 'status.json', 'summary.json'))
        need(read(pre_run/'protocol.json') == pre_exec, 'Predecessor protocol alias differs')
        pre_child = inside(pre_summary['child_run'], pre_run)
        need(set(pre_summary['child_file_sha256']) == {'protocol.json', 'status.json', 'summary.json'}, 'Predecessor child proof must bind all three files')
        for name, expected in pre_summary['child_file_sha256'].items(): need(sha(pre_child/name) == expected, 'Predecessor child file changed')
        validate_predecessor_metadata(protocol, args.phase, pre, pre_exec, pre_status, pre_summary,
            read(pre_child/'protocol.json'), read(pre_child/'status.json'), read(pre_child/'summary.json'), review)
        old_service = closed_service({'unit': pre['unit'], 'PID': pre['original_PID']})
        for key, value in ENVIRONMENT.items(): need(os.environ.get(key) == value, 'Startup environment changed: '+key)
        catalogue = original_frame_catalogue(P, manifest, args.phase)
        caption = read(CAPTION); need(caption['status'] == 'passed' and set(caption['entries']) == set(members), 'All8 actual caption receipts required')
        import torch
        import importlib
        from vsr.frozen_evaluation import _default_preview, _default_vae, _default_restore, _json
        from vsr.evaluate_external import run_external_evaluation
        from vsr.reproducibility import configure_reproducibility, reproducibility_flags
        from vsr.native_qa_restore import _contexts_empty
        save = _json; need(not torch.cuda.is_initialized(), 'Fresh pre-CUDA process required')
        vae_module = importlib.import_module('diffusers.models.autoencoders.autoencoder_kl_wan')
        need(Path(vae_module.__file__).resolve() == Path(bindings['Wan_VAE_source']['path']).resolve(), 'Bound VAE file is not actually imported source')
        previews = {}
        for label, spec in protocol['checkpoints'].items():
            path = Path(spec['path']).resolve(); need(path == Path(members[label]['checkpoint_path']) and sha(path) == spec['checkpoint_sha256'], 'Accepted original task bytes changed')
            value = _default_preview(path, allowed_root=P/'runs'); cfg = read(path.parent.parent/'config.json'); entry = caption['entries'][label]
            need(value['checkpoint_sha256'] == spec['checkpoint_sha256'] and value['variant'] == spec['variant'] and value['step'] == 8192
                and value['config'] == cfg and bool(cfg.get('reference', {}).get('enabled', False)) is (spec['family'] == 'reference')
                and cfg['seed'] == cfg['training'].get('seed', cfg['seed']) == 20261102 and cfg['model'] == config['model'], 'Actual original payload/family/official model differs')
            for prefix in ('config', 'origin_config'): need(sha(entry[prefix+'_path']) == entry[prefix+'_sha256'], 'Caption configuration changed')
            need(entry['checkpoint_path'] == str(path) and entry['checkpoint_sha256'] == spec['checkpoint_sha256']
                and sha(entry['cache_path']) == entry['cache_file_sha256'], 'Actual caption cache/payload binding differs')
            previews[label] = value
        startup_guard.verify_bindings(seal['startup_gpu_guard'])
        guard = capture_startup_guard(startup_guard, args, run, save, startup_guard_observations, 'GPU_guard_before_protocol.json')
        need(args.unit in guard['own_diagnostic_service_units'], 'Register actual direct service before GPU load')
        runtime = configure_reproducibility(config); adapter = module_from(POLICY, 'vsr_external_public_policy')
        execution = {'schema': 'vsr_frozen_external_T1_policy_execution_v1', 'phase': args.phase,
            'worker_sha256': sha(__file__), 'root_protocol_path': str(args.protocol.resolve()),
            'root_protocol_sha256': FINAL_PROTOCOL_SHA, 'root_protocol_file_sha256': FINAL_PROTOCOL_SHA,
            'root_protocol': protocol, 'phase_code_seal': {'path': str(args.execution_seal.resolve()), 'sha256': args.expected_execution_seal_sha, 'value': seal},
            'predecessor_review': {'path': str(args.predecessor_review.resolve()), 'sha256': args.expected_predecessor_review_sha, 'value': review},
            'predecessor_closed_service': old_service, 'code_commit': common.COMMIT,
            'code106_sha256': environment['code_sha256'], 'policy_sha256': POLICY_SHA, 'lifecycle': LIFECYCLE,
            'native_metadata': metadata, 'ordered_original_frame_catalogue': catalogue,
            'actual_resource_proof': resource_proof, 'actual_QA_summary_sha256': bindings['QA']['summary_sha256'],
            'payload_preview': previews, 'runtime_before_GPU_load': runtime, 'startup_environment': ENVIRONMENT,
            'startup_gpu_guard': seal['startup_gpu_guard'], 'legacy_producer_guard_invoked': False,
            'factory_overrides': ['restore', 'vae'], 'factory_overrides_are_synthetic_fixtures': False,
            'fresh_only_no_resume': True, 'child_run': str(child), 'GPU_guard': guard,
            'training_updates': 0, 'test_used_for_selection': False, 'no_full_parameter_hash_per_restore': True}
        contract = digest(execution); save(run/'execution_policy.json', execution); save(run/'protocol.json', execution)
        save(run/'status.json', {'status': 'running', 'protocol_sha256': contract, 'phase': args.phase, 'child_run': str(child)})
        restore = RotatingExternalRestore(_default_restore, adapter.install, hq.validate_ledger, metadata,
            finite=lambda value: bool(torch.isfinite(value).all()), contexts_empty=_contexts_empty,
            progress=lambda roles: save(run/'policy_role_counters.json', {'phase': args.phase, 'roles': roles}))
        def vae_factory(cfg, *, device):
            vae = _default_vae(cfg, device=device)
            need(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in vae.parameters()), 'Original CUDA/FP32 real VAE required')
            return restore.register(vae, cfg)
        stage = 'unchanged_native_external_' + args.phase
        startup_guard.verify_bindings(seal['startup_gpu_guard'])
        capture_startup_guard(startup_guard, args, run, save, startup_guard_observations, 'GPU_guard_before_load.json')
        result = run_external_evaluation(P, {label: x['path'] for label, x in protocol['checkpoints'].items()},
            datasets=PHASE_DATASETS[args.phase], frames=17, seeds=(42, 43), steps=20, tile_hw=(256, 448), overlap=64,
            device='cuda', metric_device='cpu', run_dir=child, preview_fps=30.0, caption='A high quality video.',
            preview_max_side=None, selection_note=phase_selection_note(args.phase, contract),
            factories={'vae': vae_factory, 'restore': restore})
        stage = 'completion_native_budget_and_row_seals'
        rows = completed_rows(child, protocol, args.phase, metadata, catalogue)
        totals = restore.verify_totals(); save(run/'policy_role_counters.json', totals); save(run/'completed_row_seals.json', rows)
        need(sha(args.protocol) == FINAL_PROTOCOL_SHA and sha(args.execution_seal) == args.expected_execution_seal_sha
            and sha(args.predecessor_review) == args.expected_predecessor_review_sha and sha(POLICY) == POLICY_SHA
            and sha(__file__) == execution['worker_sha256'], 'Frozen phase/root/source proof changed during execution')
        startup_guard.verify_bindings(seal['startup_gpu_guard'])
        summary = {'status': 'passed', 'phase': args.phase, 'protocol_sha256': contract, 'child_run': str(child),
            'child_file_sha256': {name: sha(child/name) for name in ('protocol.json', 'status.json', 'summary.json')},
            'completed_row_seals_sha256': sha(run/'completed_row_seals.json'), 'policy_totals': totals,
            'runtime_flags': reproducibility_flags(), 'training_updates': 0, 'test_used_for_selection': False,
            'startup_guard_observations': startup_guard_observations, 'legacy_producer_guard_invoked': False,
            'quality_conclusion': None, 'wall_seconds': time.perf_counter()-started,
            'wall_scope': 'after initial CLI/raw-protocol/seal/review checks and run-directory creation through source/proof/archive metadata, own CPU previews, original evaluator and metadata seals; excludes interpreter startup/initial checks/final serialization'}
        save(run/'summary.json', summary); save(run/'status.json', {'status': 'passed', 'phase': args.phase,
            'protocol_sha256': contract, 'summary_sha256': sha(run/'summary.json'), 'child_run': str(child)})
        print(json.dumps({'status': 'passed', 'phase': args.phase, 'protocol_sha256': contract, 'child_run': str(child)}))
    except BaseException as error:
        failure = {'status': 'failed', 'phase': args.phase, 'protocol_sha256': contract, 'stage': stage,
            'startup_guard_observations': startup_guard_observations, 'legacy_producer_guard_invoked': False,
            'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
            'partial_roles': restore.roles if restore is not None else [], 'last_policy_audit': restore.last_audit if restore is not None else None,
            'child_run': str(child), 'child_file_sha256': {name: sha(child/name) for name in
                ('protocol.json', 'status.json', 'summary.json', 'progress.json') if (child/name).is_file()},
            'wall_seconds': time.perf_counter()-started, 'no_silent_fallback_or_retry': True, 'training_updates': 0}
        failure['wall_scope'] = 'after initial CLI/raw-protocol/seal/review checks and run-directory creation through failure observation; excludes interpreter startup/initial checks/final serialization'
        save(run/'failure_observations.json', failure); save(run/'status.json', {'status': 'failed', 'phase': args.phase,
            'protocol_sha256': contract, 'stage': stage, 'error': failure['error'], 'wall_seconds': failure['wall_seconds'],
            'failure_observations_sha256': sha(run/'failure_observations.json')})
        raise
    finally:
        restore = None; gc.collect()


if __name__ == '__main__':
    main()
