"""Build fixed-teacher evaluation JSON configs only; never prepare or execute GPU work.

Validation inputs come exclusively from the immutable study.  A test config
requires the completed-validation selection receipt and its later RGB dataset.
The core ft-prepare command remains the authority for full source/artifact
verification and independently recomputes every test selection.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


def need(value, message):
    if not value:
        raise ValueError(message)


def load_evaluator(source_root, dependency_root=None):
    """Import only the explicitly requested checkout, without CUDA initialization."""
    root = Path(source_root).resolve()
    need((root/'vsr/evaluate.py').is_file(), 'Explicit source root has no evaluator')
    sys.dont_write_bytecode = True
    if dependency_root is not None:
        dependency_root = Path(dependency_root).resolve()
        need(dependency_root.is_dir(), 'Explicit dependency root is absent')
        sys.path.insert(0, str(dependency_root))
    sys.path.insert(0, str(root))
    evaluator = importlib.import_module('vsr.evaluate')
    need(Path(evaluator.__file__).resolve() == root/'vsr/evaluate.py',
         'A different vsr checkout is already imported; use a fresh Python process')
    return evaluator


def _bound(ev, value):
    return ev._hr_binding(value) if isinstance(value, dict) else ev.binding(value)


def build_config(ev, *, phase, study_file, training_run, run_root, dependency_root=None,
                 validation_run=None, selection_receipt=None, test_dataset=None):
    """Read finite completed training metadata, never model tensors or predictions."""
    need(phase in ('val', 'test'), 'Explicit val/test phase required')
    study_file = ev.binding(study_file)
    evidence = ev._hr_training_evidence(training_run, study_file, fixed_teacher=True)
    study = evidence['study']
    ev._ft_test_access_policy(study['evaluation'])
    training_run = Path(training_run).resolve()
    protocol = ev.read(training_run/'protocol.json')
    admission_file = _bound(ev, protocol['admission_file'])
    need(Path(study['gpu_admission']).resolve() == Path(admission_file['path'])
         and ev.read(admission_file['path']) == protocol['admission'], 'Training admission binding changed')
    need(protocol['device'] == study['device'] and protocol['gpu_lock'] == study['gpu_lock']
         and protocol['admission']['expected_gpu_uuid'] == study['gpu_uuid'],
         'Evaluation must preserve the declared training device, host UUID and shared GPU lock')
    need(Path(study['model_directory']).resolve().is_dir(), 'Official model directory is absent')
    roots = [Path(study['allowed_base_root']).resolve(), training_run]
    need(len(set(roots)) == 2, 'Base and new training checkpoint roots must be distinct')
    evaluation = study['evaluation']
    selection = None
    if phase == 'val':
        need(validation_run is None and selection_receipt is None and test_dataset is None,
             'Validation configuration does not consume a test selection or test dataset')
        seal = _bound(ev, evaluation['validation_seal'])
        dataset = _bound(ev, evaluation['validation_dataset'])
        roles = [{'label':'shared_initial', 'arm':'shared', 'family':'baseline', 'setting':'shared_initial',
                  'checkpoint':evidence['base']['checkpoint'], 'step':8192, 'update':0,
                  'weight':0., 'cumulative_training_seconds':0.}]
        for arm in study['arms']:
            record = evidence['arms'][arm['label']]
            for update in study['eval_updates']:
                cp = next(c for c in record['checkpoint_files'] if c['step'] == update)
                roles.append({'label':f'{arm["label"]}_u{update:06d}', 'arm':arm['label'],
                    'family':arm['family'], 'setting':arm['setting'], 'weight':arm['weight'],
                    'checkpoint':{k:cp[k] for k in ('path','bytes','sha256')},
                    'step':update, 'update':update, 'cumulative_training_seconds':cp['cumulative_training_seconds']})
    else:
        need(validation_run is not None and selection_receipt is not None and test_dataset is not None,
             'Test requires validation run, immutable selection receipt, and selected-after RGB dataset')
        selected_file = _bound(ev, selection_receipt)
        selection = ev.read(selected_file['path'])
        validation_run = Path(validation_run).resolve()
        need(selection.get('schema') == 'fixed_teacher_validation_selection_v1'
             and selection.get('study_file') == study_file
             and Path(selection['validation_run']).resolve() == validation_run
             and selection.get('selected_after_complete_validation') is True
             and selection.get('test_pixels_read_by_selection') is False,
             'Test selection receipt does not describe this completed validation/study')
        for key, name in [('validation_protocol','protocol.json'), ('validation_status','status.json'),
                          ('validation_summary','summary.json'), ('validation_observations','observations.json')]:
            bound = _bound(ev, selection[key])
            need(Path(bound['path']) == validation_run/name, 'Selection validation bindings escaped their run')
        seal = _bound(ev, evaluation['test_cohort_plan'])
        need(selection.get('test_cohort_plan') == seal, 'Selection changed the frozen test cohort')
        access = ev._ft_test_access_policy(evaluation)
        need(all(selection.get(k) == v for k,v in access.items()), 'Selection omitted the test access disclosure')
        dataset = _bound(ev, test_dataset)
        manifest = ev.read(dataset['path'])
        need(manifest.get('selection_receipt') == selected_file and manifest.get('cohort_seal') == seal
             and manifest.get('materialization',{}).get('selection_frozen_before_decode') is True,
             'Test RGB dataset was not materialized under this exact selection')
        roles = [{k:r[k] for k in ('label','arm','family','setting','weight','step','update',
                                   'cumulative_training_seconds','checkpoint')} for r in selection['selected_roles']]
        need([r['family'] for r in roles] == ['baseline','hr_vae','hr_dino'], 'Test must contain one selected role per family')
    weights = study['metric_weights']
    need(set(weights) == {'alexnet','linear'}, 'Use both explicitly frozen LPIPS weight files')
    text = _bound(ev, evidence['base']['config']['data']['text_cache'])
    cfg = {'schema':'fixed_teacher_evaluation_config_v1', 'phase':phase,
        'study_config':study_file, 'training_run':str(training_run), 'run_root':str(Path(run_root).resolve()),
        'allowed_checkpoint_roots':[str(p) for p in roots], 'roles':roles,
        'seal':seal['path'], 'dataset':dataset['path'], 'text_cache':text,
        'model_directory':str(Path(study['model_directory']).resolve()),
        'device':study['device'], 'gpu_lock':study['gpu_lock'], 'gpu_admission':protocol['admission'],
        'metric_weights':{key:_bound(ev,value) for key,value in weights.items()},
        'path_map':evaluation.get('path_map',{}),
        'configuration_generation':{'GPU_work_performed':False, 'protocol_created':False,
            'full_test_selection_recomputed_by':'vsr.evaluate ft-prepare, then execute/verify',
            'expected_roles':16 if phase == 'val' else 3,
            'expected_restores':1024 if phase == 'val' else 192}}
    if dependency_root is not None:
        root = Path(dependency_root).resolve()
        need(root.is_dir(), 'Explicit dependency root is absent')
        cfg['dependency_root'] = str(root)
    if phase == 'test':
        cfg.update(validation_run=str(validation_run), selection_receipt=selected_file)
    # Check role-to-training configuration/hash/timing links without loading CP tensors.
    ev._hr_role_specs(cfg, evidence, fixed_teacher=True, phase=phase, selection=selection)
    return cfg


def write_config(path, cfg):
    path = Path(path).resolve()
    need(not path.exists(), 'Evaluation configs are immutable; choose a new output path')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(cfg, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('val','test'))
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dependency-root', type=Path)
    parser.add_argument('--validation-run', type=Path)
    parser.add_argument('--selection-receipt', type=Path)
    parser.add_argument('--test-dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    ev = load_evaluator(args.source_root, args.dependency_root)
    cfg = build_config(ev, phase=args.phase, study_file=args.study, training_run=args.training_run,
        run_root=args.run_root, dependency_root=args.dependency_root, validation_run=args.validation_run,
        selection_receipt=args.selection_receipt, test_dataset=args.test_dataset)
    output = write_config(args.output, cfg)
    print(json.dumps({'status':'configuration_only_no_protocol_or_GPU_started', 'config':ev.binding(output),
                      'phase':cfg['phase'], **cfg['configuration_generation']}, indent=2))


if __name__ == '__main__':
    main()
