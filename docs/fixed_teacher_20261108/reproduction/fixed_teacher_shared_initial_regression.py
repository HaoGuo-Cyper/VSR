"""CPU-only exact shared-initial regression after a complete 1024-row validation.

No model imports, tensor deserialization, restoration, metric calculation, or
threshold adjustment. PNG bytes must match exactly. The 31 retained validation
windows are selected only by the prospective cohort metadata, never by scores.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback


class Mismatch(ValueError):
    def __init__(self, stage, **details):
        self.stage, self.details = stage, details
        super().__init__(stage + ': ' + json.dumps(details, sort_keys=True))


def need(value, stage, **details):
    if not value: raise Mismatch(stage, **details)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def fingerprint(value):
    """Compare declared bytes/SHA and metadata while allowing reviewed relocation."""
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            return {'bytes': value['bytes'], 'sha256': value['sha256']}
        return {k: fingerprint(v) for k, v in value.items()}
    if isinstance(value, list): return [fingerprint(v) for v in value]
    return value


class Verifier:
    def __init__(self): self.files = {}

    def verify(self, item, *, stage='artifact_binding', expected_path=None):
        path = Path(item['path']).resolve()
        if expected_path is not None:
            need(path == Path(expected_path).resolve(), stage, path=str(path), expected_path=str(expected_path))
        before = path.stat(); signature = (before.st_ino, before.st_size, before.st_mtime_ns)
        prior = self.files.get(str(path))
        if prior and prior['signature'] == signature:
            actual = prior['binding']
        else:
            actual = binding(path)
            after = path.stat()
            need(signature == (after.st_ino, after.st_size, after.st_mtime_ns), 'input_changed_during_read', path=str(path))
            self.files[str(path)] = {'signature': signature, 'binding': actual}
        need(actual['sha256'] == item['sha256'] and ('bytes' not in item or actual['bytes'] == item['bytes']),
             stage, expected=item, actual=actual)
        return actual

    def unchanged(self):
        for path, record in self.files.items():
            stat = Path(path).stat()
            need(record['signature'] == (stat.st_ino, stat.st_size, stat.st_mtime_ns), 'input_changed_during_comparison', path=path)


def mapped(path, protocol, seal):
    value = str(path)
    for source, target in sorted(protocol.get('path_map', {}).items(), key=lambda x: -len(x[0])):
        if value == source or value.startswith(source.rstrip('/') + '/'):
            value = str(target) + value[len(source):]; break
    result = Path(value)
    return result.resolve() if result.is_absolute() else (Path(seal['project_root']) / result).resolve()


def finished(run, *, historical=False):
    """The first and only read of an incomplete new run is its status.json."""
    run = Path(run).resolve(); state = read(run/'status.json')
    expected = 1152 if historical else 1024
    need(state.get('status') == 'completed' and state.get('restores') == expected and state.get('training_updates') == 0,
         'historical_not_complete' if historical else 'new_validation_not_complete', run=str(run), state=state)
    if historical: need(state.get('gate_status') == 'passed_G2', 'historical_G2_not_passed')
    protocol, summary = read(run/'protocol.json'), read(run/'summary.json')
    need(state.get('protocol_sha256') == sha(run/'protocol.json') and state.get('summary_sha256') == sha(run/'summary.json'),
         'completion_SHA_chain', run=str(run))
    if not historical:
        need(protocol.get('schema') == 'fixed_teacher_clean_evaluation_protocol_v1' and protocol.get('phase') == 'val'
             and summary.get('restores') == 1024 and summary.get('protocol_sha256') == state['protocol_sha256'], 'new_validation_scope')
        need(protocol['scope']['roles'] == 16 and protocol['scope']['clips'] == 32 and protocol['scope']['total_restores'] == 1024,
             'new_validation_matrix')
    else:
        need(protocol.get('schema') == 'vsr_evaluation_v2' and protocol.get('kind') == 'bases', 'historical_protocol_kind')
    return protocol, state


def reference_metadata(historical_run, new_seal_file, new_dataset_file):
    """Read completed historical metadata and prospective new inputs, no images."""
    old, _ = finished(historical_run, historical=True)
    verifier = Verifier()
    for key in ('cohort_seal', 'dataset'): verifier.verify(old[key])
    old_seal, old_data = read(old['cohort_seal']['path']), read(old['dataset']['path'])
    new_seal, new_data = read(new_seal_file), read(new_dataset_file)
    need(new_data.get('split') == 'val' and new_data.get('cohort_seal') == binding(new_seal_file)
         and new_data.get('cohort_seal_digest') == new_seal.get('seal_digest'), 'new_input_seal_chain')
    need(new_seal.get('original_cohort') == old['cohort_seal'] and len(new_seal['cohort']) == 32,
         'prospective_original_cohort')
    old_items = {x['record']['id']: x for x in old_seal['cohort']}
    old_assets = {x['record_id']: x for x in old_data['assets']}
    new_assets = {x['record_id']: x for x in new_data['assets']}
    retained = [x for x in new_seal['cohort'] if x.get('retained_original_window') is True]
    need(len(retained) == 31 and len(new_assets) == 32 and len(old_items) == len(old_assets) == 32
         and len({x['record']['id'] for x in new_seal['cohort']}) == 32
         and len({x['record']['source_id'] for x in new_seal['cohort']}) == 32, 'retained_window_count')
    comparisons = []
    source_keys = ('record_id', 'source_id', 'source_sha256', 'source_indices', 'temporal_stride', 'original_fps',
                   'output_fps', 'source_hw', 'resize_hw', 'crop_tlhw', 'degradation', 'valid_region')
    for item in retained:
        rid = item['record']['id']; need(rid in old_items and rid in new_assets, 'retained_record_missing', record_id=rid)
        prior, oa, na = old_items[rid], old_assets[rid], new_assets[rid]
        need((item['record']['source_id'], item['record']['split'], item['source_indices']) ==
             (prior['record']['source_id'], prior['record']['split'], prior['source_indices']), 'retained_source_window', record_id=rid)
        need(na.get('reused_original_RGB_bytes') is True, 'not_declared_original_RGB', record_id=rid)
        om, nm = oa['source_metadata'], na['metadata']
        need(all(om.get(k) == nm.get(k) for k in source_keys), 'retained_preprocessing_metadata', record_id=rid)
        assets = {}
        for kind in ('HR', 'clean'):
            left, right = oa[kind], na[kind]
            need({k:v for k,v in left.items() if k != 'path'} == {k:v for k,v in right.items() if k != 'path'},
                 'retained_NPY_declaration', record_id=rid, kind=kind)
            assets[kind] = {'historical': dict(left, path=str(mapped(left['path'], old, old_seal))),
                            'new': dict(right, path=str(mapped(right['path'], {}, new_seal)))}
        comparisons.append({'record_id': rid, 'source_id': item['record']['source_id'], 'source_indices': item['source_indices'], 'assets': assets})
    excluded = [x['record']['id'] for x in new_seal['cohort'] if x.get('retained_original_window') is not True]
    verifier.unchanged()
    return {'status':'ready_reference_only_not_output_comparison', 'historical_run':str(Path(historical_run).resolve()),
            'historical_protocol':binding(Path(historical_run)/'protocol.json'), 'new_cohort_seal':binding(new_seal_file),
            'new_dataset':binding(new_dataset_file), 'retained_clips':31, 'excluded_replaced_new_records':excluded,
            'sampling_seeds':[42,43], 'frames_per_restore':33, 'expected_restore_pairs':62, 'expected_PNG_pairs':2046,
            'comparison':'exact_PNG_file_SHA256; no tolerance', 'pairs':comparisons,
            'quality_comparison_performed':False, 'new_validation_results_read':False, 'GPU_work_performed':False}


def source_compatibility(old, new, verifier):
    left, right = old['source_bindings'], new['sources']
    expected = {'__init__.py','data.py','vae.py','model.py','teacher.py','losses.py','train.py','infer.py','evaluate.py','metrics.py','checkpoints.py'}
    need(set(right) == expected, 'new_source_inventory')
    for item in right.values(): verifier.verify(item)
    evidence = {}
    for name in ('infer', 'model', 'vae', 'metrics'):
        a, b = left[name], right[name+'.py']
        verifier.verify(a); verifier.verify(b)
        need(fingerprint(a) == fingerprint(b), 'numerical_source_changed', module=name)
        evidence[name] = {'historical':a, 'new':b, 'exact_file_bytes':True}
    for name, symbols in (('checkpoints', ('_sha256_file','_safe_checkpoint','_check_config','load_training_checkpoint')),
                          ('evaluate', ('_save_pngs','_crop_policy','_validate_restore'))):
        a, b = left[name], right[name+'.py']; verifier.verify(a); verifier.verify(b)
        def trees(path):
            return {node.name:ast.dump(node, include_attributes=False) for node in ast.parse(Path(path).read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        at, bt = trees(a['path']), trees(b['path'])
        need(all(symbol in at and at.get(symbol) == bt.get(symbol) for symbol in symbols), 'numerical_AST_changed', module=name, symbols=symbols)
        evidence[name] = {'historical':a, 'new':b, 'exact_symbol_AST':list(symbols)}
    for key in ('official_model', 'text_cache'):
        need(fingerprint(old[key]) == fingerprint(new[key]), 'model_or_text_declaration_changed', key=key)
    verifier.verify(old['text_cache']); verifier.verify(new['text_cache'])
    # Large official tensor files were fully verified by core execute/ft-verify.
    # Recheck their declarations and physical file sizes; do not deserialize them.
    for key, item in new['official_model'].items():
        need(Path(item['path']).is_file() and Path(item['path']).stat().st_size == item['bytes'], 'official_model_size_changed', key=key)
        if key.endswith('.json'): verifier.verify(item)
    for key in ('python','packages','native'):
        need(fingerprint(old['runtime'][key]) == fingerprint(new['runtime'][key]), 'runtime_declaration_changed', key=key)
    for item in new['runtime']['native'].values(): verifier.verify(item)
    need(old['strict_environment'] == new['strict_environment'], 'strict_environment_changed')
    return evidence


def load_audits(old_run, new_run, old_role, new_role, verifier):
    old_path = old_run/'roles'/old_role['role']/'load_audit.json'
    paths = sorted((new_run/'roles'/new_role['label']).glob('load_audit_*.json'))
    need(bool(paths), 'missing_new_load_audit')
    original = read(old_path); verifier.verify(binding(old_path))
    keys = ('checkpoint_sha256','format','step','variant','config','official_repo_id','official_revision',
            'restored_parameter_count','restored_parameter_names','checkpoint_tensor_dtypes','destination_tensor_dtypes',
            'optimizer_state_restored','scheduler_state_restored','rng_state_restored',
            'frozen_official_parameters_restored_from_task_state','inference_weights','training_only_state_restored',
            'EMA_weights_selected','training_only_fields_present')
    need(original.get('inference_weights') == 'student' and original.get('EMA_weights_selected') is False
         and original.get('training_only_state_restored') is False and original.get('training_only_fields_present') == [], 'historical_not_student_only')
    for path in paths:
        current = read(path); verifier.verify(binding(path))
        need(all(current.get(k) == original.get(k) for k in keys), 'student_load_audit_changed', path=str(path))
    verifier.verify(binding(old_run/'role_audits.json'))
    old_states = [x for x in read(old_run/'role_audits.json') if x['role'] == old_role['role']]
    need(len(old_states) == 1, 'missing_historical_state_audit'); prior = old_states[0]
    need(prior['trainable_before'] == prior['trainable_after'] and prior['training_updates'] == 0, 'historical_student_changed')
    states = sorted((new_run/'roles'/new_role['label']).glob('state_audit_*.json'))
    need(bool(states), 'missing_new_state_audit')
    for path in states:
        state = read(path); verifier.verify(binding(path))
        need(state.get('training_updates') == 0 and state.get('inference_weights') == 'student'
             and state.get('trainable_before') == state.get('trainable_after') == prior['trainable_before'], 'shared_initial_state_changed', path=str(path))
    return {'historical_load_audit':binding(old_path), 'new_load_audits':[binding(p) for p in paths],
            'new_state_audits':[binding(p) for p in states], 'student_heads_or_EMA_restored':False}


def row_payload(run, protocol_sha, directory, identity, verifier, observation):
    directory = Path(directory); metric_path = directory/'metrics.json'
    need(Path(observation['row_metrics_path']).resolve() == metric_path, 'observation_path_changed', expected=str(metric_path))
    metric_binding = binding(metric_path)
    need(observation['row_metrics_sha256'] == metric_binding['sha256'], 'observation_metrics_SHA', path=str(metric_path))
    verifier.verify(metric_binding)
    marker, row = read(directory/'complete.json'), read(metric_path)
    need(marker.get('protocol_sha256') == protocol_sha and marker.get('metrics_sha256') == metric_binding['sha256'] and marker.get('PNG_count') == 33,
         'row_marker_changed', path=str(directory))
    verifier.verify(binding(directory/'complete.json'))
    need(row.get('status') == 'completed' and row.get('training_updates') == 0
         and all(row['observation'].get(k) == v for k,v in identity.items()), 'row_identity_changed', identity=identity)
    pngs = row['PNG_artifacts']
    expected = [str((directory/'png'/f'{i:06d}.png').resolve()) for i in range(33)]
    need(len(pngs) == 33 and [str(Path(p['path']).resolve()) for p in pngs] == expected, 'PNG_order_or_count', path=str(directory))
    return row


def compare(historical_run, validation_run):
    old_run, new_run = Path(historical_run).resolve(), Path(validation_run).resolve()
    # Deliberately gate before reading any new protocol/results or historical images.
    new, new_state = finished(new_run)
    old, old_state = finished(old_run, historical=True)
    reference = reference_metadata(old_run, new['cohort_seal']['path'], new['dataset']['path'])
    verifier = Verifier()
    for run in (old_run,new_run):
        for name in ('protocol.json','status.json','summary.json','observations.json'): verifier.verify(binding(run/name))
    for key in ('cohort_seal','dataset'): verifier.verify(new[key])
    need(new['cohort_seal'] == reference['new_cohort_seal'] and new['dataset'] == reference['new_dataset'], 'new_protocol_input_bindings')
    old_roles = [r for r in old['roles'] if r.get('training_seed') == 20261102]
    new_roles = [r for r in new['roles'] if r.get('label') == 'shared_initial']
    need(len(old_roles) == len(new_roles) == 1, 'shared_initial_role_missing')
    a, b = old_roles[0], new_roles[0]
    need(a['variant'] == b['variant'] == 'Base' and a['expected_step'] == b['step'] == 8192
         and b.get('update') == 0 and b.get('weight') == 0 and b.get('arm') == 'shared'
         and b.get('family') == 'baseline' and a['config'] == b['config'], 'shared_initial_config_changed')
    need(fingerprint(a['checkpoint']) == fingerprint(b['checkpoint']), 'checkpoint_declaration_changed')
    verifier.verify(a['checkpoint']); verifier.verify(b['checkpoint'])
    for key, value in {'sampling_seeds':[42,43], 'frames':33, 'steps':20, 'CFG':1., 'tile_hw':[256,448], 'overlap':64, 'training_updates':0}.items():
        need(old['scope'].get(key) == new['scope'].get(key) == value, 'sampling_configuration_changed', key=key)
    need(new['scope'].get('conditions') == ['clean'] and 'clean' in old['scope'].get('conditions', []), 'clean_condition_scope')
    identity = lambda values: [(i['record']['id'], i['record']['source_id'], i['source_indices']) for i in values]
    need(identity(new['cohort']) == identity(read(new['cohort_seal']['path'])['cohort']), 'new_protocol_cohort_changed')
    evidence = source_compatibility(old, new, verifier)
    audits = load_audits(old_run, new_run, a, b, verifier)
    old_obs = read(old_run/'observations.json'); new_obs = read(new_run/'observations.json')
    need(old_obs.get('protocol_sha256') == old_state['protocol_sha256'] and new_obs.get('protocol_sha256') == new_state['protocol_sha256'], 'observations_protocol_changed')
    rows = new_obs['observations']; cohort = new['cohort']
    wanted = {(r['label'], c['record']['id'], s) for r in new['roles'] for c in cohort for s in (42,43)}
    need(len(new['roles']) == 16 and len(cohort) == 32 and len(wanted) == len(rows) == 1024
         and {(r['role'],r['record_id'],r['sampling_seed']) for r in rows} == wanted, 'incomplete_or_duplicate_new_matrix')
    old_cohort = read(old['cohort_seal']['path'])['cohort']
    old_wanted = {(r['training_seed'], c['record']['id'], condition, sample)
                  for r in old['roles'] for c in old_cohort for condition in old['scope']['conditions'] for sample in (42,43)}
    need(len(old_obs['observations']) == len(old_wanted) == 1152
         and {(r['training_seed'],r['record_id'],r['condition'],r['sampling_seed']) for r in old_obs['observations']} == old_wanted, 'incomplete_or_duplicate_historical_matrix')
    old_index = {(r['record_id'],r['sampling_seed']):r for r in old_obs['observations'] if r['training_seed'] == 20261102 and r['condition'] == 'clean'}
    new_index = {(r['record_id'],r['sampling_seed']):r for r in rows if r['role'] == 'shared_initial'}
    need(len(old_index) == len(new_index) == 64, 'shared_initial_observation_count')
    compared = []
    for pair in reference['pairs']:
        rid = pair['record_id']
        for kind, entries in pair['assets'].items():
            for entry in entries.values(): verifier.verify(entry, stage='retained_NPY_bytes')
        for sample in (42,43):
            identity = {'training_seed':20261102, 'record_id':rid, 'source_id':pair['source_id'], 'condition':'clean',
                        'sampling_seed':sample, 'checkpoint_sha256':a['checkpoint']['sha256']}
            left_dir = old_run/'results'/a['role']/rid/'clean'/f'seed_{sample}'
            right_dir = new_run/'results'/'shared_initial'/rid/f'seed_{sample}'
            left = row_payload(old_run, old_state['protocol_sha256'], left_dir, identity, verifier, old_index[(rid,sample)])
            right = row_payload(new_run, new_state['protocol_sha256'], right_dir, identity|{'role':'shared_initial'}, verifier, new_index[(rid,sample)])
            for side, payload in (('historical', left), ('new', right)):
                metadata = payload['source_metadata']
                need(metadata.get('source_frame_indices') == pair['source_indices']
                     and metadata.get('source_id') == pair['source_id'], 'executed_source_window_changed', record_id=rid, sample=sample, side=side)
            need(right.get('teacher_or_HR_input_to_inferrer') is False
                 and right.get('inference_inputs') == ['LRV','native text embedding','student checkpoint'], 'teacher_or_HR_used_at_inference', record_id=rid, sample=sample)
            need(left['restoration'] == right['restoration'] and left['VAE_ledger'] == right['VAE_ledger'], 'restoration_metadata_changed', record_id=rid, sample=sample)
            hashes = []
            for frame, (lp, rp) in enumerate(zip(left['PNG_artifacts'], right['PNG_artifacts'])):
                old_png, new_png = verifier.verify(lp), verifier.verify(rp)
                need(old_png['sha256'] == new_png['sha256'], 'prediction_PNG_bytes_changed', record_id=rid, sampling_seed=sample,
                     original_frame_index=frame, historical=old_png, new=new_png)
                hashes.append(old_png['sha256'])
            compared.append({'record_id':rid, 'sampling_seed':sample, 'frames':33, 'exact_PNG_SHA256':hashes})
    verifier.unchanged()
    need(len(compared) == 62 and sum(x['frames'] for x in compared) == 2046, 'comparison_budget')
    return {'schema':'fixed_teacher_shared_initial_regression_v1', 'status':'passed_exact_shared_initial_regression',
            'historical_protocol':binding(old_run/'protocol.json'), 'validation_protocol':binding(new_run/'protocol.json'),
            'checkpoint_SHA256':a['checkpoint']['sha256'], 'reference':reference, 'source_compatibility':evidence, 'student_audits':audits,
            'restore_pairs_compared':62, 'PNG_pairs_compared':2046, 'retained_NPY_bindings_verified':124,
            'compared':compared, 'official_model_tensor_bytes_rehashed_here':False,
            'official_model_policy':'Declared bytes/SHA match; current sizes/JSON rechecked. Full tensor-file hashes are the frozen core execute/ft-verify responsibility.',
            'metric_values_compared_or_reported':False, 'quality_conclusion':None,
            'GPU_work_performed':False, 'additional_restores':0, 'numerical_tolerance':0,
            'interpretation':'Compatibility regression for identical retained inputs and shared Base01 student only; no claim about trained method quality.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-run', type=Path, required=True)
    parser.add_argument('--validation-run', type=Path)
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--new-cohort-seal', type=Path)
    parser.add_argument('--new-dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    need(not args.output.exists(), 'output_exists', path=str(args.output))
    try:
        if args.reference_only:
            need(args.validation_run is None and args.new_cohort_seal and args.new_dataset, 'reference_only_arguments')
            value = reference_metadata(args.historical_run, args.new_cohort_seal, args.new_dataset)
        else:
            need(args.validation_run and not args.new_cohort_seal and not args.new_dataset, 'comparison_arguments')
            value = compare(args.historical_run, args.validation_run)
        code = 0
    except Exception as error:
        value = {'schema':'fixed_teacher_shared_initial_regression_failure_v1','status':'failed_strict_regression',
                 'stage':getattr(error,'stage','unexpected_error'), 'details':getattr(error,'details',{}),
                 'error':str(error),'traceback':traceback.format_exc(),'automatic_retry':False,
                 'tolerance_adjusted':False,'GPU_work_performed':False,'additional_restores':0}
        code = 1
    value['at_UTC'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':value['status'],'output':binding(args.output)},indent=2))
    return code


if __name__ == '__main__': raise SystemExit(main())
