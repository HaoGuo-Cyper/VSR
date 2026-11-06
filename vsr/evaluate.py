"""One fresh-only evaluator for the three Bases and the three pilot arms.

prepare binds metadata and byte hashes, execute requires an explicit protocol
SHA and shared host GPU lease, verify rechecks complete row/artifact chains.
No partial matrix can pass a scientific gate; imports never start an experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import gc
import hashlib
import importlib.metadata
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

SEEDS = (20261102, 20261103, 20261104)
ARMS = ('flow_only', 'C1', 'A2')
CONDITIONS = ('clean', 'blur', 'noise', 'codec_allintra', 'drop_repeat', 'drop_black')
SAMPLES = (42, 43)
BASE_CHECKPOINT_SHAS = {
    '20261102':'28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f',
    '20261103':'87f60ce812797d73b27b9d2d0bcb41abb45ea4659ca2b6eb6092d02f3b58061a',
    '20261104':'c6f0aedcefac965334851e081d72916dccfa9559b469dd1ad819d7139df65edd'}
CORE_FILES = ('data', 'vae', 'model', 'teacher', 'losses', 'train', 'infer', 'evaluate', 'metrics', 'checkpoints')
COHORT_SEAL_DIGEST = '9c5e938d8b2c9b5e7d965fca2f6b3925557848235d6b9d404b6231bffcbc36e8'
G4_DEFINITION_SHA = 'd552dbb136223d246e05af202f25b072a2c822ad4eeaecead0aa0d6360b1ca58'
STRICT_ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1',
                      'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def binding(path, expected=None):
    path = Path(path).resolve()
    need(path.is_file(), 'Missing bound file: ' + str(path))
    value = sha(path)
    need(expected is None or value == expected, 'Changed bound file: ' + str(path))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': value}


def verify_binding(item):
    need(binding(item['path']) == item, 'Bound input changed: ' + item['path'])


def save(path, value, *, replace=False):
    from .data import write_json_new
    path = Path(path)
    if not replace:
        write_json_new(path, value)
        return
    need(path.name in ('status.json', 'progress.json'), 'Only live status/progress can be replaced')
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def source_bindings():
    root = Path(__file__).resolve().parent
    return {name: binding(root / (name + '.py')) for name in CORE_FILES}


def runtime_bindings(dependency_root=None):
    """Bind installed distribution records, native sources and metric deps."""
    before = list(sys.path)
    try:
        if dependency_root:
            sys.path.insert(0, str(Path(dependency_root).resolve()))
        packages = {}
        modules = {'torch':'torch','numpy':'numpy','diffusers':'diffusers','transformers':'transformers',
                   'peft':'peft','accelerate':'accelerate','safetensors':'safetensors',
                   'lpips':'lpips','scikit-image':'skimage','Pillow':'PIL','opencv-python-headless':'cv2',
                   'scipy':'scipy','torchvision':'torchvision'}
        for name, module in modules.items():
            dist = importlib.metadata.distribution(name)
            spec = importlib.machinery.PathFinder.find_spec(module, sys.path)
            need(spec is not None and spec.origin is not None, 'Missing dependency: ' + module)
            records = {kind:binding(dist.locate_file(item)) for item in (dist.files or [])
                       for kind in ('METADATA','RECORD') if str(item).endswith('.dist-info/' + kind)}
            need(set(records) == {'METADATA','RECORD'}, 'Distribution metadata incomplete: ' + name)
            packages[name] = {'version':dist.version,'entry':binding(spec.origin),'metadata':records}
        dist = importlib.metadata.distribution('diffusers')
        native = {name:binding(dist.locate_file(path)) for name,path in {
            'transformer':'diffusers/models/transformers/transformer_wan.py',
            'vae':'diffusers/models/autoencoders/autoencoder_kl_wan.py',
            'scheduler':'diffusers/schedulers/scheduling_unipc_multistep.py'}.items()}
        tree = {}
        if dependency_root:
            root = Path(dependency_root).resolve()
            need(root.is_dir(), 'Explicit dependency_root is absent')
            for path in sorted(root.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'):
                    need(path.resolve().is_relative_to(root), 'Dependency tree escapes declared root')
                    tree[str(path.relative_to(root))] = binding(path)
        return {'python':sys.version,'executable':str(Path(sys.executable).resolve()),'packages':packages,
                'native':native,'local_dependency_tree':tree}
    finally:
        sys.path[:] = before


def verify_imported_runtime(specification):
    """Reject cached/shadow imports before any model or CUDA initialization."""
    import importlib
    modules = {'torch':'torch','numpy':'numpy','diffusers':'diffusers','transformers':'transformers',
               'peft':'peft','accelerate':'accelerate','safetensors':'safetensors','lpips':'lpips',
               'scikit-image':'skimage','Pillow':'PIL','opencv-python-headless':'cv2',
               'scipy':'scipy','torchvision':'torchvision'}
    for name, declaration in specification['packages'].items():
        module = importlib.import_module(modules[name])
        need(binding(module.__file__) == declaration['entry']
             and importlib.metadata.version(name) == declaration['version'], 'Imported dependency shadowed bound source: '+name)
    native = {'transformer':'diffusers.models.transformers.transformer_wan',
              'vae':'diffusers.models.autoencoders.autoencoder_kl_wan',
              'scheduler':'diffusers.schedulers.scheduling_unipc_multistep'}
    for name, module in native.items():
        need(binding(importlib.import_module(module).__file__) == specification['native'][name],
             'Imported native implementation changed: '+name)


def _cleanup_preserving(primary, operations):
    """Attempt every cleanup; a secondary error never replaces the primary."""
    first = primary
    for operation in operations:
        try:
            operation()
        except BaseException as secondary:
            if first is None:
                first = secondary
            elif first is not secondary:
                first.add_note('Cleanup error: '+repr(secondary))
    if primary is None and first is not None:
        raise first


def _verify_nested_bindings(value):
    if isinstance(value,dict):
        if {'path','bytes','sha256'}.issubset(value):
            verify_binding({k:value[k] for k in ('path','bytes','sha256')})
        else:
            for child in value.values(): _verify_nested_bindings(child)
    elif isinstance(value,list):
        for child in value: _verify_nested_bindings(child)


def validate_completed_pilot_training(run_dir, roles, base_evidence, definition):
    """Bind the actual three2048 student checkpoints to their complete pilot."""
    run = Path(run_dir).resolve()
    protocol, status, summary = (read(run/name) for name in ('protocol.json','status.json','summary.json'))
    psha = sha(run/'protocol.json')
    need(protocol.get('schema') == 'vsr_training_protocol_v2' and protocol.get('mode') == 'pilot'
         and status.get('status') == summary.get('status') == 'completed' and summary.get('mode') == 'pilot',
         'A completed new-protocol scientific pilot is required')
    need(status['protocol_sha256'] == summary['protocol_sha256'] == psha
         and status['summary_sha256'] == sha(run/'summary.json'), 'Pilot completion hash chain differs')
    need(summary.get('scientific_training_updates') == 6144 and summary.get('diagnostic_optimizer_updates_applied',0) == 0
         and set(summary['roles']) == set(ARMS) and summary.get('G4_evaluated') is False,
         'Pilot must contain exactly three2048 arms, no diagnostics or prior G4 selection')
    need(protocol['G2_evidence'] == base_evidence, 'Pilot and evaluation use different Base evidence')
    frozen = protocol['G4_definition_binding']
    need(frozen['definition'] == definition and frozen['base_evaluation_protocol_sha256'] == base_evidence['protocol_sha256']
         and frozen['definitions_modified'] is False and frozen['binding_before_method_training'] is True,
         'G4 definitions were not bound unchanged before method training')
    current = {p.name:binding(p) for p in Path(__file__).parent.glob('*.py')}
    need(protocol['sources'] == current, 'Pilot was trained with a different numeric implementation')
    for key in ('cache','inputs','matched_Base','official_model_files','dependencies','G3_evidence'):
        _verify_nested_bindings(protocol[key])
    g3 = protocol['G3_evidence']
    need(g3['status'] == 'passed_G3', 'Pilot requires a passed resource gate')
    g3_summary = read(g3['summary']['path'])
    need(g3_summary.get('G3_passed') is True and g3_summary.get('diagnostic_optimizer_updates_applied') == 70
         and g3_summary['semantics']['status'] == 'passed_exact_GPU_semantics'
         and g3_summary['resource_gate']['status'] == 'passed_G3_resource_cost', 'Incomplete resource/semantic evidence')
    selected_base = next(r for r in base_evidence['checkpoint_hashes'] if str(r) == '20261102')
    need(protocol['matched_Base']['checkpoint']['sha256'] == base_evidence['checkpoint_hashes'][selected_base],
         'Pilot did not start from the evaluated matched Base01')
    ids = protocol['cache']['selection']['training_ids']
    need(len(ids) == 1641, 'Keep fixed training order')
    for role in roles:
        arm = role['role']; record = summary['roles'][arm]
        need(record['arm'] == arm and record['updates'] == 2048
             and record['optimizer_state_fresh_at_initialization'] is True and record['optimizer_foreach'] is False
             and record['diagnostic_updates'] == 0, 'Pilot arm was not fresh or budget differs')
        initialization = record['initialization']
        need(initialization['checkpoint_sha256'] == protocol['matched_Base']['checkpoint']['sha256']
             and initialization['step'] == 8192 and initialization['variant'] == 'Base'
             and all(initialization[k] is False for k in ('optimizer_state_restored','scheduler_state_restored','rng_state_restored')),
             'Pilot initialization does not match fresh Base continuation')
        checkpoints = record['checkpoint_files']
        need(len(checkpoints) == 4 and [Path(c['path']).name for c in checkpoints]
             == [f'checkpoint_{step:06d}.pt' for step in (512,1024,1536,2048)], 'Incomplete declared pilot checkpoint schedule')
        need(checkpoints[-1] == role['checkpoint'], 'Evaluation checkpoint is not this pilot final2048')
        for checkpoint in checkpoints:
            need(Path(checkpoint['path']).resolve().is_relative_to(run/'arm_runs'/arm), 'Pilot checkpoint escaped its arm')
            verify_binding(checkpoint)
        log = record['full_training_log']; verify_binding(log)
        need(Path(log['path']).resolve() == run/'arm_runs'/arm/'training.jsonl', 'Wrong arm training log')
        lines = [json.loads(line) for line in Path(log['path']).read_text().splitlines()]
        need(len(lines) == 2048 and [x['step'] for x in lines] == list(range(1,2049)), 'Incomplete pilot log')
        for ordinal,row in enumerate(lines):
            need(row['arm'] == arm and row['data_index'] == ordinal % 1641 and row['record_id'] == ids[ordinal % 1641],
                 'Pilot data order or role differs')
            need(all(type(row[k]) in (int,float) and __import__('math').isfinite(row[k])
                     for k in ('fm','sd','total','grad_norm_before_clip','sigma')), 'Nonfinite pilot training row')
    return {'files':{name:binding(run/name) for name in ('protocol.json','status.json','summary.json')},
            'mode':'pilot','scientific_training_updates':6144,'protocol_sha256':psha,
            'checkpoint_hashes':{r['role']:r['checkpoint']['sha256'] for r in roles}}


def model_bindings(directory):
    directory = Path(directory).resolve()
    result = {}
    for part in ('transformer','vae','scheduler'):
        root = directory / part
        need(root.is_dir() and (root/'config.json').is_file() if part != 'scheduler' else root.is_dir() and (root/'scheduler_config.json').is_file(),
             'Official model component/config missing: ' + part)
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                need(path.resolve().is_relative_to(directory), 'Official asset escapes model directory')
                result[str(path.relative_to(directory))] = binding(path)
    need(any(name.startswith('transformer/') and name.endswith('.safetensors') for name in result)
         and any(name.startswith('vae/') and name.endswith('.safetensors') for name in result), 'Official transformer and VAE safetensors required')
    return result


def _mapped(path, mapping, *, root=None):
    from .data import _mapped_path
    path = Path(path)
    if not path.is_absolute():
        need(root is not None, 'Relative asset requires explicit project root')
        path = Path(root) / path
    return _mapped_path(path, mapping)


def _role_specs(config):
    kind = config['kind']
    need(kind in ('bases','pilot'), 'Evaluation kind must be bases or pilot')
    wanted = list(map(str, SEEDS)) if kind == 'bases' else list(ARMS)
    items = config['checkpoints']
    need(isinstance(items, dict) and set(items) == set(wanted), 'Declare exactly the three required roles')
    allowed = Path(config['allowed_checkpoint_root']).resolve()
    result = []
    for role in wanted:
        declaration = items[role]
        path = Path(declaration['path']).resolve()
        need(path.is_relative_to(allowed) and path != allowed, 'Checkpoint escaped explicit allowed root')
        cp = binding(path, declaration['sha256'])
        if kind == 'bases':
            need(cp['sha256'] == BASE_CHECKPOINT_SHAS[role], 'Use the three already audited Base checkpoints')
        cfg = read(path.parent.parent / 'config.json')
        from .checkpoints import _check_config
        _check_config(cfg)
        variant = 'Base' if kind == 'bases' else role
        seed = int(role) if kind == 'bases' else 20261102
        expected_step = 8192 if kind == 'bases' else 2048
        need(cfg['training'].get('seed',cfg['seed']) == seed, 'Checkpoint source training seed differs')
        need(cfg.get('reproducibility',{}).get('mode') == 'strict', 'Original strict math precision is mandatory')
        need(variant in cfg['training']['variants'] and not cfg.get('reference',{}).get('enabled',False),
             'Wrong checkpoint family or variant')
        need([cfg['data'][k] for k in ('num_frames','temporal_stride','height','width','scale')] == [33,1,128,224,4],
             'Expected unchanged protocol33 checkpoint geometry')
        result.append({'role':role,'training_seed':seed,'variant':variant,'expected_step':expected_step,
                       'checkpoint':cp,'config':cfg,'config_binding':binding(path.parent.parent / 'config.json')})
    need(len({r['checkpoint']['sha256'] for r in result}) == 3, 'Distinct role checkpoints required')
    return result


def collect_protocol(config_path, run_dir):
    """Read and bind bytes only; never deserialize task tensors or load models."""
    from .data import validate_seal
    config_path = Path(config_path).resolve()
    cfg = read(config_path)
    roles = _role_specs(cfg)
    run = Path(run_dir).resolve(); run_root = Path(cfg['run_root']).resolve()
    need(run != run_root and run.is_relative_to(run_root), 'Output must be a fresh child of explicit run_root')
    seal_path, dataset_path = Path(cfg['seal']).resolve(), Path(cfg['dataset']).resolve()
    seal, dataset = read(seal_path), read(dataset_path)
    mapping = cfg.get('path_map', {})
    need(isinstance(mapping,dict), 'path_map must be explicit original->current paths')
    validate_seal(seal, path_map=mapping)
    need(seal['config']['conditions'] == list(CONDITIONS[1:]) and len(seal['cohort']) == 32,
         'Retain the unchanged sealed val32 and all five types')
    need(seal['seal_digest'] == COHORT_SEAL_DIGEST, 'The prospective original val32 cohort is fixed')
    need(dataset['cohort_seal_digest'] == seal['seal_digest']
         and dataset['status'] == 'completed_CPU_assets_no_predictions' and dataset['config'] == seal['config'], 'Dataset/cohort/config mismatch')
    ids = [item['record']['id'] for item in seal['cohort']]
    need(len(dataset['assets']) == 32 and {x['record_id'] for x in dataset['assets']} == set(ids), 'Dataset must retain all32 unique sealed clips')
    for asset in dataset['assets']:
        item = next(x for x in seal['cohort'] if x['record']['id'] == asset['record_id'])
        need(asset['source_id'] == item['record']['source_id'] and asset['mask'] == item['mask']
             and set(asset['conditions']) == set(CONDITIONS[1:]), 'Asset identity/mask/types differ')
    need(not run.is_relative_to(dataset_path.parent), 'Outputs cannot overwrite evaluation inputs')
    assets = []
    for item in dataset['assets']:
        for entry in (item['HR'],item['clean'],*[item['conditions'][c]['asset'] for c in CONDITIONS[1:]]):
            path = _mapped(entry['path'],mapping,root=seal['project_root'])
            assets.append(binding(path,entry['sha256']))
    need(len(assets) == 224, 'Bind all224 frozen NPY assets')
    text = cfg['text_cache']
    need(all(role['config']['data']['text_cache']['sha256'] == text['sha256'] for role in roles),
         'Use the same original fixed text embedding as each Base')
    need(all(role['config']['model']['repo_id'] == 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
             and role['config']['model']['revision'] == '0fad780a534b6463e45facd96134c9f345acfa5b'
             for role in roles), 'Official model revision changed')
    metric_weights = cfg['metric_weights']
    need(set(metric_weights) == {'alexnet','linear'}, 'Explicit local LPIPS weights required')
    protocol = {'schema':'vsr_evaluation_v2','kind':cfg['kind'],'config':cfg,'config_binding':binding(config_path),
        'roles':roles,'source_bindings':source_bindings(),'cohort_seal':binding(seal_path),
        'dataset':binding(dataset_path),'cohort_seal_digest':seal['seal_digest'],
        'assets':assets,'text_cache':binding(text['path'],text['sha256']),
        'metric_weights':{k:binding(v['path'],v['sha256']) for k,v in metric_weights.items()},
        'official_model':model_bindings(cfg['model_directory']),
        'runtime':runtime_bindings(cfg.get('dependency_root')),'path_map':mapping,
        'scope':{'roles':3,'clips':32,'conditions':list(CONDITIONS),'sampling_seeds':list(SAMPLES),
                 'total_restores':1152,'frames':33,'steps':20,'CFG':1.0,'tile_hw':[256,448],
                 'overlap':64,'training_updates':0,'fresh_only':True,'automatic_retry':False},
        'vae_policy':'per-restore original33-frame delegate, ledger checked, cache cleared and wrapper removed',
        'strict_environment':STRICT_ENVIRONMENT}
    if cfg['kind'] == 'pilot':
        base_protocol, base_evidence = validate_completed_base_evaluation(cfg['base_evaluation_run'])
        definition = binding(cfg['G4_definitions'],G4_DEFINITION_SHA)
        need(read(definition['path'])['schema'] == 'stage4_G4_definition_freeze_v1'
             and read(definition['path'])['cohort_seal_digest'] == seal['seal_digest'], 'Wrong G4 definitions/cohort')
        protocol['base_evaluation_evidence'] = base_evidence
        protocol['G4_definitions'] = definition
        training_run = Path(cfg['pilot_training_run']).resolve()
        protocol['pilot_training'] = validate_completed_pilot_training(training_run,roles,base_evidence,definition)
        need(base_protocol['cohort_seal_digest'] == seal['seal_digest'], 'Pilot/Base cohorts differ')
    return json.loads(json.dumps(protocol,ensure_ascii=False,allow_nan=False))


def prepare(config_path, run_dir):
    run = Path(run_dir).resolve()
    need(not run.exists(), 'Fresh-only output exists; do not retry or resume')
    protocol = collect_protocol(config_path,run)
    run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    result = {'status':'planned','protocol_sha256':sha(run/'protocol.json'),'run_dir':str(run),
              'created_utc':datetime.now(timezone.utc).isoformat(),'training_updates':0,'restores':0}
    save(run/'status.json',result)
    return result


def ordered_cases(protocol, seal):
    return [(role,item,condition,sample) for role in protocol['roles'] for item in seal['cohort']
            for condition in CONDITIONS for sample in SAMPLES]


def _verify_protocol_inputs(protocol):
    for item in protocol['source_bindings'].values(): verify_binding(item)
    for key in ('cohort_seal','dataset','text_cache','config_binding'): verify_binding(protocol[key])
    for item in protocol['assets']: verify_binding(item)
    for item in protocol['metric_weights'].values(): verify_binding(item)
    for item in protocol['official_model'].values(): verify_binding(item)
    for role in protocol['roles']:
        verify_binding(role['checkpoint']); verify_binding(role['config_binding'])
    need(protocol['runtime'] == runtime_bindings(protocol['config'].get('dependency_root')),
         'Prepared runtime/dependencies changed')
    if protocol['kind'] == 'pilot':
        verify_binding(protocol['G4_definitions'])
        for item in protocol['pilot_training']['files'].values(): verify_binding(item)


def _save_pngs(video, directory):
    from PIL import Image
    import torch
    directory.mkdir(parents=True)
    frames = torch.round(video.detach().float().clamp(0,1)*255).byte().cpu()[0].permute(1,2,3,0).numpy()
    result = []
    for index, frame in enumerate(frames):
        path = directory/f'{index:06d}.png'
        Image.fromarray(frame,mode='RGB').save(path)
        result.append(binding(path))
    return result


def _contexts_empty(model):
    return all(getattr(model,name,None) is None for name in ('_captured','_condition_tokens','_reference_context','reference_context')) and not bool(getattr(model,'_capture_enabled',False))


@contextmanager
def _crop_policy(vae):
    from .vae import install_cache_policy, validate_cache_ledger
    policy = install_cache_policy(vae,max_records=2)
    try:
        yield policy
        need(policy.dropped_records == 0, 'VAE ledger overflow')
        validate_cache_ledger(policy.ledger,False,1)
        need(policy.ledger[0]['input_shape'] == [1,3,33,128,224], 'Expected original complete33-frame VAE encode')
        for key in ('_feat_map','_enc_feat_map'):
            value = getattr(vae,key,None)
            need(isinstance(value,list) and all(x is None for x in value), 'VAE temporal cache survived restoration')
    finally:
        primary = sys.exc_info()[1]
        def remove():
            policy.remove()
            need('encode' not in vae.__dict__ and getattr(vae,'_vsr_t1_cache_policy',None) is None,
                 'Native VAE encode wrapper not removed')
        _cleanup_preserving(primary,[remove])


def _validate_restore(video, meta, model, sample):
    import torch
    need(list(video.shape) == meta['output_shape'] == [1,3,33,128,224] and bool(torch.isfinite(video).all()),
         'Bad restored shape or nonfinite pixels')
    need(meta['tile_count'] == 1 and len(meta['tiles']) == 1 and meta['reference']['enabled'] is False
         and meta['reference']['encode_count'] == 0 and meta['temporal_windowing'] is False
         and meta['padding']['time_end'] == 0 and meta['inference_inputs'] == ['LRV','native text embedding'],
         'Changed native crop/reference/input geometry')
    sampling = meta['sampling']
    need(sampling['steps'] == 20 and sampling['seed'] == sample and sampling['cfg'] == 1.0
         and sampling['global_noise_shape'] == [1,16,9,16,28] and sampling['noise_device'] == 'cpu'
         and sampling['sampler'] == 'local official-config UniPC' and _contexts_empty(model), 'Changed native sampler or retained model context')


def _execute_loaded(protocol, run, protocol_sha):
    cfg = protocol['config']
    if cfg.get('dependency_root'): sys.path.insert(0,cfg['dependency_root'])
    verify_imported_runtime(protocol['runtime'])
    import torch
    from .checkpoints import load_training_checkpoint
    from .model import configure_reproducibility,tensor_state_hashes
    from .vae import load_frozen_vae
    from .data import FramewiseReader,CAPTION_PLACEHOLDER
    from .infer import restore_video
    from .metrics import quantize_rgb,paired_metrics,score_masked,summarize_g2,summarize_pilot
    cfg = protocol['config']
    runtime = configure_reproducibility(protocol['roles'][0]['config'])
    flags = runtime['flags']
    need(flags['deterministic_algorithms'] and not flags['deterministic_warn_only']
         and flags['sdp_math'] and not flags['sdp_flash'] and not flags['sdp_memory_efficient']
         and not flags['sdp_cudnn'] and not flags['matmul_allow_tf32'] and not flags['cudnn_allow_tf32'],
         'Original strict deterministic math/TF32 flags required')
    save(run/'runtime.json',runtime)
    cache = torch.load(protocol['text_cache']['path'],map_location='cpu',weights_only=True)
    text = cache.get(CAPTION_PLACEHOLDER) if isinstance(cache,dict) else None
    need(isinstance(text,torch.Tensor) and list(text.shape) == [1,512,4096] and text.dtype == torch.bfloat16
         and not text.requires_grad and bool(torch.isfinite(text).all()), 'Fixed text cache changed; no T5 fallback')
    from .metrics import PerceptualMetrics
    seal = read(protocol['cohort_seal']['path'])
    reader = perceptual = None
    observations, role_audits = [], []
    try:
        perceptual = PerceptualMetrics(protocol['metric_weights']['alexnet']['path'],protocol['metric_weights']['linear']['path'],device='cpu')
        reader = FramewiseReader(protocol['dataset']['path'],seal,path_map=protocol['path_map'])
        torch.cuda.set_device(cfg['device'])
        for role in protocol['roles']:
            model = vae = None
            try:
                model, audit = load_training_checkpoint(role['checkpoint']['path'],device=cfg['device'],
                    allowed_root=cfg['allowed_checkpoint_root'],variant=role['variant'],model_directory=cfg['model_directory'])
                need(audit['step'] == role['expected_step'] and audit['checkpoint_sha256'] == role['checkpoint']['sha256']
                     and audit['config'] == role['config'] and model.projector is None,'Loaded role differs from declared checkpoint')
                inference_config = copy.deepcopy(role['config'])
                inference_config['model']['local_dir'] = cfg['model_directory']
                model.eval(); vae = load_frozen_vae(inference_config,device=cfg['device'])
                need(all(p.dtype == torch.float32 and not p.requires_grad for p in vae.parameters()), 'Original frozen FP32 VAE required')
                before = tensor_state_hashes(model)
                save(run/'roles'/role['role']/'load_audit.json',audit)
                for item in seal['cohort']:
                    record = item['record']['id']
                    for condition in CONDITIONS:
                        source = reader.load('Stage2-val-'+condition,record,num_frames=33,start=0)
                        target = quantize_rgb(source['hr'])
                        for sample in SAMPLES:
                            directory = run/'results'/role['role']/record/condition/f'seed_{sample}'
                            need(not directory.exists(),'Fresh-only row exists')
                            directory.mkdir(parents=True)
                            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                            started = time.perf_counter()
                            with _crop_policy(vae) as policy:
                                video, meta = restore_video(model,vae,source['lr'],text,inference_config,
                                    steps=20,seed=sample,tile_hw=(256,448),overlap=64)
                                _validate_restore(video,meta,model,sample)
                                policy_ledger = copy.deepcopy(policy.ledger)
                            policy = None
                            torch.cuda.synchronize()
                            seconds = time.perf_counter()-started
                            measured = quantize_rgb(video)
                            pngs = _save_pngs(measured,directory/'png')
                            full = paired_metrics(measured,target,perceptual,excluded_reference=0)
                            full['primary_pool'] = 'all_frames'
                            score = score_masked(full,item['mask'])
                            observation = {'training_seed':role['training_seed'],'record_id':record,
                                'source_id':item['record']['source_id'],'condition':condition,'sampling_seed':sample,
                                'checkpoint_sha256':role['checkpoint']['sha256'],'cohort_seal_digest':seal['seal_digest'],'score':score}
                            if protocol['kind'] == 'pilot': observation['arm'] = role['role']
                            row = {'status':'completed','observation':observation,'full_frame_pair_metrics':full,
                                'source_metadata':source['metadata'],'restoration':meta,'VAE_ledger':policy_ledger,
                                'PNG_artifacts':pngs,'restore_seconds':seconds,
                                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'training_updates':0}
                            save(directory/'metrics.json',row)
                            save(directory/'complete.json',{'protocol_sha256':protocol_sha,'metrics_sha256':sha(directory/'metrics.json'),
                                                           'PNG_count':33,'resume_allowed':False})
                            observation = observation | {'row_metrics_path':str(directory/'metrics.json'),
                                                         'row_metrics_sha256':sha(directory/'metrics.json')}
                            observations.append(observation)
                            save(run/'progress.json',{'status':'running','completed_restores':len(observations),
                                 'last':{k:observation[k] for k in ('training_seed','record_id','condition','sampling_seed')},'training_updates':0},replace=True)
                            del video, measured, full
                        del source, target
                after = tensor_state_hashes(model)
                need(before == after and all(p.grad is None for p in model.parameters()) and _contexts_empty(model),
                     'Evaluation changed model state or retained gradients')
                role_audits.append({'role':role['role'],'restores':384,'trainable_before':before,'trainable_after':after,
                                    'inference_weights':'student','training_updates':0})
            finally:
                primary = sys.exc_info()[1]
                _cleanup_preserving(primary, [lambda: model.clear_features() if model is not None else None,
                                              lambda: vae.clear_cache() if vae is not None else None])
                model = vae = policy = None
                _cleanup_preserving(primary, [gc.collect, torch.cuda.empty_cache])
    finally:
        primary = sys.exc_info()[1]
        _cleanup_preserving(primary, [lambda: reader.close() if reader is not None else None])
        perceptual = None
        _cleanup_preserving(primary, [gc.collect, torch.cuda.empty_cache])
    need(len(observations) == 1152 and len(role_audits) == 3, 'Incomplete matrix')
    save(run/'observations.json',{'observations':observations,'protocol_sha256':protocol_sha})
    save(run/'role_audits.json',role_audits)
    if protocol['kind'] == 'bases':
        summary = summarize_g2(observations,seal)
    else:
        _, evidence = validate_completed_base_evaluation(cfg['base_evaluation_run'])
        need(evidence == protocol['base_evaluation_evidence'],'Bound Base evaluation changed')
        base_rows = read(Path(cfg['base_evaluation_run'])/'observations.json')['observations']
        summary = summarize_pilot(observations,base_rows,seal,
             {r['role']:r['checkpoint']['sha256'] for r in protocol['roles']},
             {str(s):h for s,h in evidence['checkpoint_hashes'].items()})
    save(run/'summary.json',summary)
    return summary


def execute(run_dir, expected_protocol_sha):
    run = Path(run_dir).resolve(); protocol = read(run/'protocol.json')
    need(sha(run/'protocol.json') == expected_protocol_sha, 'Explicit exact protocol SHA required')
    state = read(run/'status.json')
    need(state['status'] == 'planned' and state['protocol_sha256'] == expected_protocol_sha,'No resume or retry')
    cfg = protocol['config']
    need(str(cfg['device']).startswith('cuda'), 'Formal evaluator requires the explicitly selected GPU')
    need(all(os.environ.get(k) == v for k,v in STRICT_ENVIRONMENT.items()),'Strict/offline launch environment missing')
    _verify_protocol_inputs(protocol)
    need(protocol == collect_protocol(protocol['config_binding']['path'],run),'Prepared effective protocol changed')
    with (run/'execute.lock').open('x') as stream: stream.write(str(os.getpid()))
    save(run/'status.json',{'status':'running','protocol_sha256':expected_protocol_sha,'training_updates':0},replace=True)
    from .checkpoints import gpu_lease
    def record_failure(error):
        failure = {'status':'failed_stop','error':str(error),'traceback':traceback.format_exc(),
                   'protocol_sha256':expected_protocol_sha,'automatic_retry':False,'training_updates':0}
        if not (run/'failure.json').exists(): save(run/'failure.json',failure)
        save(run/'status.json',failure,replace=True)
    try:
        with gpu_lease(cfg['gpu_lock'],cfg['device'],cfg['gpu_admission']) as receipt:
            try:
                save(run/'GPU_idle_before_load.json',receipt)
                summary = _execute_loaded(protocol,run,expected_protocol_sha)
                result = {'status':'completed','protocol_sha256':expected_protocol_sha,'restores':1152,'training_updates':0,
                          'gate_status':summary['status'],'summary_sha256':sha(run/'summary.json')}
                save(run/'status.json',result,replace=True)
            except BaseException as error:
                record_failure(error)
                def cleanup():
                    import torch
                    if torch.cuda.is_initialized(): torch.cuda.empty_cache()
                _cleanup_preserving(error,[gc.collect,cleanup])
                raise
        return result
    except BaseException as error:
        if not (run/'failure.json').exists(): record_failure(error)
        raise


def validate_completed_evaluation(run_dir, *, expected_kind=None):
    """Recheck all1152 row/PNG/source chains, then recompute the original gate."""
    from .metrics import index_observations,summarize_g2,summarize_pilot,score_masked
    run = Path(run_dir).resolve()
    protocol, state = read(run/'protocol.json'),read(run/'status.json')
    need(protocol['schema'] == 'vsr_evaluation_v2' and (expected_kind is None or protocol['kind'] == expected_kind), 'Wrong evaluation schema/kind')
    raw_sha = sha(run/'protocol.json')
    need(state['status'] == 'completed' and state['protocol_sha256'] == raw_sha and state['restores'] == 1152
         and state['training_updates'] == 0, 'Evaluation is not a complete zero-update1152 run')
    _verify_protocol_inputs(protocol)
    need(protocol == collect_protocol(protocol['config_binding']['path'],run), 'Completed evaluation protocol inputs no longer reconstruct')
    bundle = read(run/'observations.json')
    need(bundle['protocol_sha256'] == raw_sha,'Observation protocol mismatch')
    rows = bundle['observations']; seal = read(protocol['cohort_seal']['path'])
    hashes = {r['role']:r['checkpoint']['sha256'] for r in protocol['roles']}
    index_observations(rows,seal,hashes,kind=protocol['kind'])
    for obs in rows:
        path = Path(obs['row_metrics_path']).resolve()
        need(path.is_relative_to(run/'results') and path.name == 'metrics.json','Row path escaped results')
        need(sha(path) == obs['row_metrics_sha256'],'Row metric SHA changed')
        row, marker = read(path),read(path.parent/'complete.json')
        need(marker == {'protocol_sha256':raw_sha,'metrics_sha256':sha(path),'PNG_count':33,'resume_allowed':False},
             'Row completion seal changed')
        declared = {k:v for k,v in obs.items() if not k.startswith('row_metrics_')}
        need(row['status'] == 'completed' and row['training_updates'] == 0 and row['observation'] == declared,
             'Row observation differs from canonical matrix')
        need(row['observation']['score'] == score_masked(row['full_frame_pair_metrics'],row['observation']['score']['mask']),
             'Frozen pool score no longer matches full frame/pair metrics')
        pngs = row['PNG_artifacts']
        need(len(pngs) == 33 and len({p['path'] for p in pngs}) == 33,'Missing or duplicate prediction PNGs')
        for png in pngs:
            need(Path(png['path']).resolve().is_relative_to(path.parent/'png'),'PNG escaped its row')
            verify_binding(png)
    role_audits = read(run/'role_audits.json')
    need(len(role_audits) == 3 and {r['role'] for r in role_audits} == set(hashes)
         and all(r['restores'] == 384 and r['training_updates'] == 0 and r['inference_weights'] == 'student'
                 and r['trainable_before'] == r['trainable_after'] for r in role_audits), 'Incomplete or mutated role audit')
    if protocol['kind'] == 'bases':
        recomputed = summarize_g2(rows,seal)
    else:
        base_protocol, evidence = validate_completed_base_evaluation(protocol['config']['base_evaluation_run'])
        need(evidence == protocol['base_evaluation_evidence'],'Base evidence changed')
        base_rows = read(Path(evidence['run_dir'])/'observations.json')['observations']
        recomputed = summarize_pilot(rows,base_rows,seal,hashes,
            {r['role']:r['checkpoint']['sha256'] for r in base_protocol['roles']})
    recomputed = json.loads(json.dumps(recomputed,ensure_ascii=False,allow_nan=False))
    need(read(run/'summary.json') == recomputed and state['summary_sha256'] == sha(run/'summary.json')
         and state['gate_status'] == recomputed['status'],'Summary/status not reproducible from full matrix')
    evidence = {'status':recomputed['status'],'run_dir':str(run),'protocol_sha256':raw_sha,
                'summary_sha256':sha(run/'summary.json'),'observations_sha256':sha(run/'observations.json'),
                'cohort_seal_digest':seal['seal_digest'],'checkpoint_hashes':hashes,
                'rows':1152,'G2':recomputed if protocol['kind'] == 'bases' else None}
    return protocol,evidence


def validate_completed_base_evaluation(run_dir):
    protocol,evidence = validate_completed_evaluation(run_dir,expected_kind='bases')
    need(evidence['status'] == 'passed_G2','Real recomputed G2 has not passed; stop dependent work')
    return protocol,evidence


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare',help='metadata and byte hashes only; create a fresh prospective protocol')
    p.add_argument('--config',type=Path,required=True); p.add_argument('--run-dir',type=Path,required=True)
    p = sub.add_parser('execute',help='explicit GPU execution of a reviewed planned protocol')
    p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--expected-protocol-sha',required=True)
    p = sub.add_parser('verify',help='verify all row/PNG/source seals and recompute the complete gate')
    p.add_argument('--run-dir',type=Path,required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'prepare': result = prepare(args.config,args.run_dir)
    elif args.command == 'execute': result = execute(args.run_dir,args.expected_protocol_sha)
    else:
        _, evidence = validate_completed_evaluation(args.run_dir)
        result = {k:v for k,v in evidence.items() if k != 'G2'}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__': main()
