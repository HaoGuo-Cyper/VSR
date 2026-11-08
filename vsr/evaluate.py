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


# Independent clean-x4 HR-latent study evaluation.  No old G2/G3/G4 dependency.
HR_EVALUATION_SCHEMA = 'hr_vae_clean_evaluation_protocol_v1'
FT_EVALUATION_SCHEMA = 'fixed_teacher_clean_evaluation_protocol_v1'
TRD_EVALUATION_SCHEMA = 'wan_trd_evaluation_protocol_v1'
VMAE_EVALUATION_SCHEMA = 'vsr_videomaev2_evaluation_protocol_v1'
HR_SCALAR_METRICS = ('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1',
                     'tLP_alex_v0_1_TecoGAN_crop','tLP100_alex_v0_1_TecoGAN_crop',
                     'tOF_Farneback_TecoGAN_crop')


def _hr_sources():
    return {p.name:binding(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def _hr_binding(item):
    need(isinstance(item,dict) and {'path','sha256'}.issubset(item),'Explicit file/SHA binding required')
    actual=binding(item['path'],item['sha256'])
    need('bytes' not in item or item['bytes']==actual['bytes'],'Bound file size changed')
    return actual


def _hr_finite(value):
    import math
    return type(value) in (int,float) and math.isfinite(value)


def _hr_training_evidence(directory, study_file, *, fixed_teacher=False, trd=False, videomaev2=False):
    """Check the declared completed trajectories, fresh state and checkpoint chain."""
    directory=Path(directory).resolve()
    protocol,status,summary=(read(directory/name) for name in ('protocol.json','status.json','summary.json'))
    psha=sha(directory/'protocol.json')
    need(sum((fixed_teacher,trd,videomaev2))<=1,'One declared study family required')
    prefix='vsr_videomaev2' if videomaev2 else 'wan_trd' if trd else 'fixed_teacher' if fixed_teacher else 'hr_vae'
    need(protocol.get('schema')==prefix+'_training_protocol_v1'
         and summary.get('schema')==prefix+'_training_summary_v1','Expected the declared fixed-teacher training schemas')
    need(status.get('status')==summary.get('status')=='completed'
         and status['protocol_sha256']==summary['protocol_sha256']==psha
         and status['summary_sha256']==sha(directory/'summary.json'),'Training completion hash chain differs')
    need(protocol['sources']==_hr_sources(),'Training/evaluation source files differ')
    need(protocol['study_file']==study_file and read(study_file['path'])==protocol['study'],'Study changed after training')
    study=protocol['study'];arms=study['arms']
    expected_arm_count=2 if videomaev2 else 5 if fixed_teacher else 3
    if fixed_teacher:
        need(study.get('schema')=='fixed_teacher_study_v1'
             and [a['family'] for a in arms]==['baseline','hr_vae','hr_vae','hr_dino','hr_dino']
             and [a['setting'] for a in arms]==['baseline','low','high','low','high']
             and arms[0]['weight']==0 and all(_hr_finite(a['weight']) and a['weight']>0 for a in arms[1:])
             and arms[1]['weight']<arms[2]['weight'] and arms[3]['weight']<arms[4]['weight'],
             'Require baseline plus two calibrated settings per frozen teacher family')
        _hr_binding(study['calibration'])
    elif trd:
        need(study.get('schema')=='wan_trd_study_v1'
             and [a['label'] for a in arms]==['baseline','frozen','ema']
             and arms[0]['weight']==0 and _hr_finite(arms[1]['weight']) and arms[1]['weight']>0
             and arms[1]['weight']==arms[2]['weight'],
             'Require baseline/frozen/EMA with one common positive alignment weight')
        calibration=protocol['calibration']
        need(set(calibration)=={'protocol','summary','status_file'},'Bind the completed calibration protocol, summary and status')
        for value in calibration.values():_hr_binding(value)
        cal=read(calibration['summary']['path']);calstatus=read(calibration['status_file']['path'])
        need(Path(study['calibration_report']).resolve()==Path(calibration['summary']['path'])
             and cal.get('status')=='passed_trd_preflight' and calstatus.get('status')=='completed'
             and cal['protocol_sha256']==calstatus['protocol_sha256']==calibration['protocol']['sha256']
             and calstatus['summary_sha256']==calibration['summary']['sha256']
             and cal['scientific_updates']==0 and cal['diagnostic_optimizer_updates']==41
             and cal['sources']==protocol['sources'] and cal['calibration']['weight']==arms[1]['weight'],
             'TRD common weight must come from the completed matching-source41-update preflight')
    elif videomaev2:
        need(study.get('schema')=='vsr_videomaev2_study_v1'
             and [a['label'] for a in arms]==['baseline','videomaev2']
             and [a['weight'] for a in arms]==[0.,.5],
             'Require the fixed baseline0 and online VideoMAEv2 weight0.5 without selection')
        calibration=protocol['calibration']
        need(set(calibration)=={'protocol','summary','status_file'},'Bind completed VideoMAEv2 preflight')
        for value in calibration.values():_hr_binding(value)
        cal=read(calibration['summary']['path']);calstatus=read(calibration['status_file']['path'])
        need(cal.get('status')=='passed_videomaev2_preflight' and calstatus.get('status')=='completed'
             and cal['protocol_sha256']==calstatus['protocol_sha256']==calibration['protocol']['sha256']
             and calstatus['summary_sha256']==calibration['summary']['sha256']
             and cal['scientific_updates']==0 and cal['diagnostic_optimizer_updates']==26
             and cal['sources']==protocol['sources'],'Require complete matching-source26-update VideoMAEv2 preflight')
        need(isinstance(protocol.get('teacher_binding'),dict) and protocol['teacher_binding']
             and isinstance(protocol.get('rgb_assets'),dict)
             and protocol['rgb_assets'].get('schema')=='vsr_online_HR_asset_audit_v1'
             and protocol['rgb_assets'].get('training_count')==1641
             and len(protocol['rgb_assets'].get('records',[]))==1641
             and protocol['rgb_assets'].get('teacher_features_cached') is False,
             'Bind the official frozen online teacher and all1641 paired HR inputs')
    else:need([a['weight'] for a in arms]==[0.,.01,.1],'Original HR-only weights changed')
    need(len(arms)==expected_arm_count
         and len({a['label'] for a in arms})==expected_arm_count and study['updates']==2048
         and study['eval_updates']==[512,1024,2048]
         and study['checkpoint_updates']==[512,1024,1536,2048],'Unchanged prospective HR study budget required')
    expected_updates=len(arms)*study['updates']
    need(status.get('scientific_updates')==summary.get('scientific_updates')==expected_updates,
         'Require every declared scientific training update')
    need(set(summary['arms'])=={a['label'] for a in arms},'Training summary must retain all three declared arms')
    if fixed_teacher or trd or videomaev2:_verify_nested_bindings(protocol)
    else:
        for key in ('base','cache','inputs','official_model_files','dependencies'):
            _verify_nested_bindings(protocol[key])
    ids=protocol['cache']['selection']['training_ids']
    need(len(ids)==1641 and len(set(ids))==1641,'Use the fixed complete training order')
    if videomaev2:
        need([r['record_id'] for r in protocol['rgb_assets']['records']]==ids
             and [r['index'] for r in protocol['rgb_assets']['records']]==list(range(1641))
             and all(r.get('split')=='train' for r in protocol['rgb_assets']['records']),
             'Online HR inputs must preserve the exact FM training record order')
    files={name:binding(directory/name) for name in ('protocol.json','status.json','summary.json')}
    initial=None;checked={};common_random_sequence=None;teacher_initial=None;teacher_sequence=None
    for arm in arms:
        label,weight=arm['label'],arm['weight'];record=summary['arms'][label]
        need(record['weight']==weight and record['updates']==2048,'Wrong arm budget/weight')
        if trd:
            need(record.get('teacher_kind')==label and record.get('EMA_updates')==(2048 if label=='ema' else 0),
                 'TRD teacher identity/update count differs')
            if label!='baseline':
                current_teacher=record['teacher_initial_hashes']
                need(isinstance(current_teacher,dict) and current_teacher
                     and all(isinstance(k,str) and isinstance(v,str) and len(v)==64
                             and all(c in '0123456789abcdef' for c in v) for k,v in current_teacher.items())
                     and current_teacher==cal['teacher_initial_hashes'],
                     'Teacher per-parameter initialization hashes must match successful preflight')
                if teacher_initial is None:teacher_initial=current_teacher
                need(current_teacher==teacher_initial,'Frozen and EMA teachers must share the same initial Base')
        elif videomaev2:
            need(record.get('family')==label and record.get('setting')=='common'
                 and record.get('EMA_updates')==0 and record.get('teacher_parameter_updates')==0
                 and record.get('teacher_forward_count')==(2048 if label=='videomaev2' else 0)
                 and record.get('teacher_is_frozen_cached_HR') is False
                 and record.get('teacher_binding')==(protocol['teacher_binding'] if label=='videomaev2' else None),
                 'VideoMAEv2 must remain the declared frozen online teacher, without EMA or feature cache')
        else:
            need(record.get('teacher_is_frozen_cached_HR',record.get('teacher_is_frozen_cached')) is True,
                 'Wrong frozen teacher source')
        if fixed_teacher:
            need(record['family']==arm['family'] and record['setting']==arm['setting'],
                 'Training family/setting differs from the prospective study')
        initialization=record['initialization']
        need(initialization['checkpoint_sha256']==protocol['base']['checkpoint']['sha256']
             and initialization['step']==8192 and initialization['variant']=='Base'
             and all(initialization[k] is False for k in ('optimizer_state_restored','scheduler_state_restored','rng_state_restored')),
             'Every arm must start from the same fresh Base student state')
        current=record['initial_student_hashes']
        need(isinstance(current,dict) and current.get('all_trainable'),'Initial student hashes are required')
        if initial is None: initial=current
        need(current==initial,'Arms did not share the exact initial student')
        log=_hr_binding(record['full_training_log'])
        need(Path(log['path'])==directory/'arm_runs'/label/'training.jsonl','Wrong arm training log path')
        rows=[json.loads(line) for line in Path(log['path']).read_text().splitlines()]
        need(len(rows)==2048,'Incomplete HR-latent training log')
        previous=0.;previous_hr_read=0.
        for i,row in enumerate(rows,1):
            need(row['step']==i and row['arm']==label and row['weight']==weight
                 and row['data_index']==(i-1)%1641 and row['record_id']==ids[(i-1)%1641],
                 'Training arm/step/weight/data order changed')
            need(all(_hr_finite(row[k]) for k in ('fm','align','total','grad_norm_before_clip','sigma','seconds',
                                                 'cumulative_training_seconds','gpu_event_seconds')),
                 'Nonfinite HR training scalar')
            need(row['seconds']>=0 and row['gpu_event_seconds']>=0
                 and row['cumulative_training_seconds']>=previous,'Training time must be nonnegative and monotonic')
            if trd:
                need(row.get('teacher_forward') is (label!='baseline')
                     and row.get('EMA_updates')==(i if label=='ema' else 0)
                     and type(row.get('timestep_index')) is int
                     and _hr_finite(row.get('native_timestep')) and 0<=row['sigma']<=1,
                     'TRD row teacher/update/native timestep differs')
                if label!='baseline':
                    need(type(row.get('teacher_timestep_index')) is int
                         and _hr_finite(row.get('teacher_native_timestep'))
                         and _hr_finite(row.get('teacher_sigma')) and 0<=row['teacher_sigma']<=row['sigma'],
                         'TRD teacher noise must be a finite native lower-noise timestep')
            if videomaev2:
                need(row.get('teacher_forward') is (label=='videomaev2')
                     and row.get('teacher_forward_count')==(i if label=='videomaev2' else 0)
                     and row.get('EMA_updates')==0 and type(row.get('timestep_index')) is int
                     and _hr_finite(row.get('native_timestep')) and 0<=row['sigma']<=1
                     and all(_hr_finite(row.get(k)) and row[k]>=0 for k in ('hr_read_seconds','cumulative_hr_read_seconds'))
                     and row['cumulative_hr_read_seconds']>=previous_hr_read,
                     'VideoMAEv2 row teacher/count/native timestep/HR read cost differs')
                previous_hr_read=row['cumulative_hr_read_seconds']
            previous=row['cumulative_training_seconds']
        if trd or videomaev2:
            common=[tuple(row[k] for k in ('data_index','record_id','sigma','timestep_index','native_timestep')) for row in rows]
            if common_random_sequence is None:common_random_sequence=common
            need(common==common_random_sequence,'TRD arms must share data and native timestep sequence')
            if trd and label!='baseline':
                target=[tuple(row[k] for k in ('teacher_sigma','teacher_timestep_index','teacher_native_timestep')) for row in rows]
                if teacher_sequence is None:teacher_sequence=target
                need(target==teacher_sequence,'Frozen/EMA teacher noise schedules differ')
        need(_hr_finite(record['cumulative_training_seconds'])
             and record['cumulative_training_seconds']==previous,'Final cumulative training time differs')
        if videomaev2:
            need(record.get('cumulative_hr_read_seconds')==previous_hr_read,'Final HR-read cost differs from all committed rows')
        checkpoints=record['checkpoint_files']
        need(len(checkpoints)==4 and [c.get('step',c.get('update')) for c in checkpoints]==[512,1024,1536,2048],
             'Require all four intermediate/final training checkpoints')
        cp_records=[]
        for cp,step in zip(checkpoints,(512,1024,1536,2048)):
            actual=_hr_binding(cp);path=Path(actual['path'])
            need(path==directory/'arm_runs'/label/'Base'/f'checkpoint_{step:06d}.pt',
                 'Student checkpoint escaped its actual training arm')
            need(cp.get('variant')=='Base' and cp.get('update',step)==step
                 and cp['cumulative_training_seconds']==rows[step-1]['cumulative_training_seconds'],
                 'Checkpoint timing/variant/update differs from committed log')
            cp_records.append(actual|{'step':step,'update':step,'cumulative_training_seconds':cp['cumulative_training_seconds']})
            files[f'{label}/checkpoint_{step:06d}']=actual
        for name in ('config.json','initialization.json','completion.json'):
            path=directory/'arm_runs'/label/name
            read(path);files[f'{label}/{name}']=binding(path)
        need(read(directory/'arm_runs'/label/'initialization.json')==initialization,'Initialization sidecar differs')
        if trd or videomaev2:need(read(directory/'arm_runs'/label/'completion.json')==record,'Arm completion/summary differ')
        files[f'{label}/training.jsonl']=log
        checked[label]={'weight':weight,'updates':2048,'checkpoint_files':cp_records,
                        'initial_student_hashes':current,'cumulative_training_seconds':previous}
        if fixed_teacher:checked[label].update(family=arm['family'],setting=arm['setting'])
        if trd:checked[label].update(family=label,setting='common',teacher_kind=label,
                                   EMA_updates=record['EMA_updates'],teacher_initial_hashes=record.get('teacher_initial_hashes'))
        if videomaev2:checked[label].update(family=label,setting='common',EMA_updates=0,
            teacher_forward_count=record['teacher_forward_count'],teacher_parameter_updates=0,
            teacher_binding=record['teacher_binding'],cumulative_hr_read_seconds=previous_hr_read)
    return {'schema':'vsr_videomaev2_completed_training_evidence_v1' if videomaev2 else 'wan_trd_completed_training_evidence_v1' if trd else 'hr_vae_completed_training_evidence_v1','run':str(directory),
            'protocol_sha256':psha,'files':files,'arms':checked,'base':protocol['base'],
            'study_file':study_file,'study':study,'scientific_updates':expected_updates,
            'all_training_rows_finite_contiguous':True,
            'all6144_rows_finite_contiguous':expected_updates==6144,
            'all_arms_same_initial_student':True}


def _hr_role_specs(config, evidence, *, fixed_teacher=False, trd=False, videomaev2=False, phase='val', selection=None):
    import re
    roots=[Path(p).resolve() for p in config['allowed_checkpoint_roots']]
    need(roots and len(set(roots))==len(roots),'Declare explicit distinct allowed checkpoint roots')
    count=((2 if videomaev2 else 3) if phase=='test' else 1+3*len(evidence['arms'])) if (fixed_teacher or trd or videomaev2) else 10
    wanted=config['roles'];need(isinstance(wanted,list) and len(wanted)==count,'Missing declared evaluated checkpoints')
    labels=set();updates=set();roles=[]
    base_config=evidence['base']['config']
    for item in wanted:
        label=item['label'];need(isinstance(label,str) and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}',label)
                                and label not in labels,'Unsafe or duplicate role label')
        labels.add(label);checkpoint=_hr_binding(item['checkpoint']);path=Path(checkpoint['path'])
        need(any(path!=r and path.is_relative_to(r) for r in roots),'Checkpoint outside explicit allowed roots')
        step,update,weight=item['step'],item['update'],item['weight'];arm=item['arm']
        need(type(step) is int and type(update) is int and _hr_finite(weight)
             and _hr_finite(item['cumulative_training_seconds']) and item['cumulative_training_seconds']>=0,
             'Invalid role update/step/weight/time metadata')
        need(path.parent.name=='Base' and path.name==f'checkpoint_{step:06d}.pt','Expected a declared Base student checkpoint')
        cfg=read(path.parent.parent/'config.json')
        from .checkpoints import _check_config
        _check_config(cfg)
        need(cfg['model']==base_config['model'] and cfg.get('conditioning')==base_config.get('conditioning')
             and cfg['data']==base_config['data'] and not cfg.get('reference',{}).get('enabled',False)
             and cfg.get('reproducibility',{}).get('mode')=='strict','Student model/conditioning/data/precision changed')
        if update==0:
            need(phase!='test' and step==8192 and checkpoint==evidence['base']['checkpoint'] and weight==0
                 and item['cumulative_training_seconds']==0 and ('shared',0) not in updates,
                 'Sharedstep0 must be the exact common starting Base')
            updates.add(('shared',0))
        else:
            need(arm in evidence['arms'] and step==update and update in (512,1024,2048),
                 'Use declared intermediate/final study checkpoints only')
            record=evidence['arms'][arm];cp=next(c for c in record['checkpoint_files'] if c['step']==step)
            need(weight==record['weight'] and checkpoint=={k:cp[k] for k in ('path','bytes','sha256')}
                 and item['cumulative_training_seconds']==cp['cumulative_training_seconds']
                 and (arm,update) not in updates,'Role is not this training arm/time/checkpoint')
            updates.add((arm,update))
        role=({k:item[k] for k in ('label','arm','weight','update','step','cumulative_training_seconds')}|
              {'variant':'Base','checkpoint':checkpoint,'config':cfg,'config_binding':binding(path.parent.parent/'config.json')})
        if fixed_teacher or trd or videomaev2:
            family,setting=('baseline','shared_initial') if update==0 else (evidence['arms'][arm]['family'],evidence['arms'][arm]['setting'])
            need(('family' not in item or item['family']==family) and ('setting' not in item or item['setting']==setting),
                 'Role family/setting metadata differs from its actual trained arm')
            role.update(family=family,setting=setting)
        roles.append(role)
    expected={('shared',0)}|{(a,u) for a in evidence['arms'] for u in (512,1024,2048)}
    if fixed_teacher and phase=='test':
        need(selection is not None,'Test roles require a verified validation-selection receipt')
        expected={(r['arm'],2048) for r in selection['selected_roles']}
        selected={r['arm']:r for r in selection['selected_roles']}
        for role in roles:
            need(role['checkpoint']==selected[role['arm']]['checkpoint'],'Test checkpoint is not the selected final student')
    if (trd or videomaev2) and phase=='test':
        need(selection is not None,'TRD test requires a complete validation final-checkpoint lock')
        expected={(arm,2048) for arm in evidence['arms']}
        locked={r['arm']:r for r in selection['final_roles']}
        need(set(locked)==set(evidence['arms']),'Final lock must retain every declared arm without selection')
        for role in roles:
            need(role['checkpoint']==locked[role['arm']]['checkpoint'],'Test checkpoint differs from locked final2048')
    need(updates==expected,'All study checkpoints must be evaluated exactly once')
    if trd or videomaev2:
        arm_order=('baseline','videomaev2') if videomaev2 else ('baseline','frozen','ema')
        ordered=([('shared',0)]+[(a,u) for a in arm_order for u in (512,1024,2048)]
                 if phase=='val' else [(a,2048) for a in arm_order])
        need([(r['arm'],r['update']) for r in roles]==ordered,'TRD role order must retain the prospective canonical matrix')
    return roles


def hr_collect_protocol(config_path, run_dir):
    from .data import validate_seal
    config_path=Path(config_path).resolve();cfg=read(config_path);run=Path(run_dir).resolve()
    run_root=Path(cfg['run_root']).resolve()
    need(run!=run_root and run.is_relative_to(run_root),'New evaluation must be a child of its output root')
    need(str(cfg['device']).startswith('cuda:') and str(cfg['device'])[5:].isdigit()
         and Path(cfg['gpu_lock']).is_absolute()
         and isinstance(cfg['gpu_admission'].get('expected_gpu_uuid'),str)
         and cfg['gpu_admission']['expected_gpu_uuid'],'Declare an explicit CUDA device, shared lock and pinned host UUID')
    study_file=_hr_binding(cfg['study_config'])
    training=_hr_training_evidence(cfg['training_run'],study_file)
    roles=_hr_role_specs(cfg,training)
    seal_path,dataset_path=Path(cfg['seal']).resolve(),Path(cfg['dataset']).resolve()
    seal,dataset=read(seal_path),read(dataset_path);mapping=cfg.get('path_map',{})
    validate_seal(seal,path_map=mapping)
    need(seal['seal_digest']==COHORT_SEAL_DIGEST and len(seal['cohort'])==32
         and dataset['cohort_seal_digest']==seal['seal_digest'] and dataset['config']==seal['config']
         and dataset['status']=='completed_CPU_assets_no_predictions','Unchanged fullval32 provenance required')
    cohort={x['record']['id']:x for x in seal['cohort']}
    need(len(cohort)==32 and len(dataset['assets'])==32 and {x['record_id'] for x in dataset['assets']}==set(cohort),
         'Require all32 unique clips, without type/mask selection')
    assets=[]
    for item in dataset['assets']:
        fixed=cohort[item['record_id']]
        need(item['source_id']==fixed['record']['source_id'] and item['mask']==fixed['mask'],'Source/mask provenance changed')
        for entry in (item['HR'],item['clean']):
            assets.append(binding(_mapped(entry['path'],mapping,root=seal['project_root']),entry['sha256']))
    need(len(assets)==64 and not run.is_relative_to(dataset_path.parent),'Clean assets/output overlap or incomplete')
    text=_hr_binding(cfg['text_cache'])
    need(all(r['config']['data']['text_cache']['sha256']==text['sha256'] for r in roles),'Use original cached native text')
    weights=cfg['metric_weights'];need(set(weights)=={'alexnet','linear'},'Explicit pinned LPIPS weights required')
    fixed=sorted(seal['cohort'],key=lambda x:x['record']['source_id'])
    need(len({x['record']['source_id'] for x in fixed})==32,'Expected independent source clips')
    video_ids=[x['record']['id'] for x in fixed[:3]]
    need('fixed_video_clip_ids' not in cfg or cfg['fixed_video_clip_ids']==video_ids,
         'Video clips must be prespecified first3 sourceIDs, never selected from scores')
    return {'schema':HR_EVALUATION_SCHEMA,'config':cfg,'config_binding':binding(config_path),
            'sources':_hr_sources(),'study_file':study_file,'training_evidence':training,'roles':roles,
            'cohort_seal':binding(seal_path),'cohort_seal_digest':seal['seal_digest'],'dataset':binding(dataset_path),
            'assets':assets,'path_map':mapping,'text_cache':text,
            'metric_weights':{k:_hr_binding(v) for k,v in weights.items()},
            'official_model':model_bindings(cfg['model_directory']),
            'runtime':runtime_bindings(cfg.get('dependency_root')),'strict_environment':STRICT_ENVIRONMENT,
            'fixed_video_clip_ids':video_ids,
            'scope':{'roles':10,'clips':32,'sampling_seeds':[42,43],'conditions':['clean'],
                     'total_restores':640,'frames':33,'HR_hw':[128,224],'LR_hw':[32,56],
                     'steps':20,'CFG':1.,'tile_hw':[256,448],'overlap':64,'primary_pool':'all33_frames',
                     'primary_metric':'lpips_alex_v0_1','training_updates':0,
                     'source_windows':'complete frozen33-frame clips, not full source videos',
                     'row_resume_same_protocol_allowed':True,'automatic_protocol_change':False}}


def hr_prepare(config_path, run_dir):
    run=Path(run_dir).resolve();need(not run.exists(),'Use a new HR study evaluation directory')
    protocol=hr_collect_protocol(config_path,run);run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    result={'status':'planned','protocol_sha256':sha(run/'protocol.json'),'run_dir':str(run),
            'restores':0,'training_updates':0}
    save(run/'status.json',result);return result


def _hr_is_ft(protocol):
    # Both independent studies use source-tagged clean assets and every-row MP4s.
    return protocol['schema'] in (FT_EVALUATION_SCHEMA,TRD_EVALUATION_SCHEMA,VMAE_EVALUATION_SCHEMA)


def _hr_total(protocol):return protocol.get('scope',{}).get('total_restores',640)


def _hr_seal_view(protocol):
    return {'cohort':protocol['cohort']} if _hr_is_ft(protocol) else read(protocol['cohort_seal']['path'])


def _hr_video_required(protocol, record):
    return _hr_is_ft(protocol) or record in protocol['fixed_video_clip_ids']


def _hr_gt_ids(protocol):
    return [x['record']['id'] for x in protocol['cohort']] if _hr_is_ft(protocol) else protocol['fixed_video_clip_ids']


def _hr_collector(protocol):
    if protocol['schema']==VMAE_EVALUATION_SCHEMA:return vmae_collect_protocol
    if protocol['schema']==TRD_EVALUATION_SCHEMA:return trd_collect_protocol
    return ft_collect_protocol if _hr_is_ft(protocol) else hr_collect_protocol


def _hr_cases(protocol, seal):
    return [(role,item,sample) for role in protocol['roles'] for item in seal['cohort'] for sample in (42,43)]


def _hr_identity(protocol, role, item, sample):
    result={'role':role['label'],'arm':role['arm'],'weight':role['weight'],'update':role['update'],
            'step':role['step'],'cumulative_training_seconds':role['cumulative_training_seconds'],
            'training_seed':20261102,'record_id':item['record']['id'],'source_id':item['record']['source_id'],
            'sampling_seed':sample,'condition':'clean','checkpoint_sha256':role['checkpoint']['sha256'],
            'cohort_seal_digest':protocol['cohort_seal_digest']}
    if _hr_is_ft(protocol):result.update(family=role['family'],setting=role['setting'],split=protocol['scope']['source_split'])
    return result


def _hr_scalars(full):
    pool=full['all_frames']
    need(pool['frame_count']==33 and len(pool['per_frame'])==33
         and pool['temporal_pair_count']==28,'Expected all33 frames and28 original temporal pairs')
    need({tuple(x['indices']) for x in pool['per_temporal_pair']}=={(i,i+1) for i in range(2,30)},
         'Original cutfr2 temporal pairs changed')
    output={key:pool[key] for key in HR_SCALAR_METRICS}
    need(all(_hr_finite(x) for x in output.values()),'Nonfinite clean-evaluation scalar')
    return output


def _hr_directory(run, role, item, sample):
    return run/'results'/role['label']/item['record']['id']/f'seed_{sample}'


def _hr_verify_row(protocol, run, psha, role, item, sample):
    directory=_hr_directory(run,role,item,sample)
    row,marker=read(directory/'metrics.json'),read(directory/'complete.json')
    need(marker.get('protocol_sha256')==psha and marker.get('metrics_sha256')==sha(directory/'metrics.json')
         and marker.get('PNG_count')==33,'Completed row protocol/metric marker changed')
    identity=_hr_identity(protocol,role,item,sample)
    need(row['status']=='completed' and row['training_updates']==0
         and row['observation']==identity|{'metrics':_hr_scalars(row['full_frame_pair_metrics'])},
         'Completed clean row identity/metric pool differs')
    need(row['source_metadata']['split']==protocol.get('scope',{}).get('source_split','val')
         and row['source_metadata']['source_frame_indices']==item['source_indices'],'Frozen validation window changed')
    need(row['teacher_or_HR_input_to_inferrer'] is False
         and row['inference_inputs']==['LRV','native text embedding','student checkpoint']
         and row['restoration']['inference_inputs']==['LRV','native text embedding'],
         'Evaluation inference must use LR/text/student only')
    pngs=row['PNG_artifacts']
    need(len(pngs)==33 and len({x['path'] for x in pngs})==33,'Missing/duplicate prediction PNGs')
    for artifact in pngs:
        need(Path(artifact['path']).resolve().is_relative_to(directory/'png'),'Prediction PNG escaped its row')
        verify_binding(artifact)
    video=row.get('video_artifact')
    required=_hr_video_required(protocol,item['record']['id'])
    need((video is not None)==required and marker.get('video_artifact')==video,'Fixed video artifact missing or changed')
    if video:
        need(Path(video['path']).resolve()==directory/'prediction.mp4','Video escaped its row')
        verify_binding(video)
        from .infer import verify_output_video
        verify_output_video(video['path'],frames=33,height=128,width=224,fps=row['source_metadata']['output_fps'])
    return row['observation']|{'row_metrics_path':str(directory/'metrics.json'),
                               'row_metrics_sha256':sha(directory/'metrics.json')}


def _hr_completed_prefix(protocol, run, psha, seal):
    rows=[];gap=False
    for role,item,sample in _hr_cases(protocol,seal):
        marker=_hr_directory(run,role,item,sample)/'complete.json'
        if not marker.exists():gap=True;continue
        need(not gap,'Committed clean rows must form a complete canonical prefix')
        rows.append(_hr_verify_row(protocol,run,psha,role,item,sample))
    return rows


def _hr_gt(protocol, run, psha, item, target, fps):
    from .data import write_video
    from .infer import verify_output_video
    directory=run/'ground_truth'/item['record']['id'];marker=directory/'complete.json'
    dataset={'assets':protocol['clean_assets']} if _hr_is_ft(protocol) else read(protocol['dataset']['path'])
    expected_HR=next(x['HR']['sha256'] for x in dataset['assets'] if x['record_id']==item['record']['id'])
    if marker.exists():
        value=read(marker);need(value['protocol_sha256']==psha and value['source_HR_sha256']==expected_HR
             and value['record_id']==item['record']['id'] and value['source_id']==item['record']['source_id']
             and value['frames']==33 and value['FPS']==fps,'GT marker provenance changed')
        for artifact in [*value['PNG_artifacts'],value['video_artifact']]:
            need(Path(artifact['path']).resolve().is_relative_to(directory),'GT artifact escaped its directory')
            verify_binding(artifact)
        need(len(value['PNG_artifacts'])==33 and len({p['path'] for p in value['PNG_artifacts']})==33,
             'GT requires all33 unique PNGs')
        verify_output_video(value['video_artifact']['path'],frames=33,height=128,width=224,fps=fps)
        return value
    if directory.exists():_hr_quarantine(run,directory)
    pngs=_save_pngs(target,directory/'png');video=directory/'GT.mp4'
    write_video(video,target,fps);verify_output_video(video,frames=33,height=128,width=224,fps=fps)
    result={'protocol_sha256':psha,'record_id':item['record']['id'],'source_id':item['record']['source_id'],
            'PNG_artifacts':pngs,'video_artifact':binding(video),'frames':33,'FPS':fps,'not_an_inference_input':True}
    result['source_HR_sha256']=expected_HR
    save(marker,result);return result


def _hr_quarantine(run, directory):
    """Preserve an interrupted uncommitted row before recomputing that row."""
    relative=directory.relative_to(run);parent=run/'incomplete_attempts'/relative.parent
    parent.mkdir(parents=True,exist_ok=True);ordinal=1
    target=parent/(relative.name+f'.attempt_{ordinal:03d}')
    while target.exists():
        ordinal+=1;target=parent/(relative.name+f'.attempt_{ordinal:03d}')
    os.rename(directory,target)


def _hr_save_final(path, value):
    """Publish once, or accept byte-equivalent aggregate after an interruption."""
    if Path(path).exists():need(read(path)==value,'An existing final aggregate changed during resume')
    else:save(path,value)


def _hr_execute_loaded(protocol, run, psha, completed):
    cfg=protocol['config']
    if cfg.get('dependency_root'):sys.path.insert(0,cfg['dependency_root'])
    verify_imported_runtime(protocol['runtime'])
    import torch
    from .data import FramewiseReader,CAPTION_PLACEHOLDER,write_video
    from .model import configure_reproducibility,tensor_state_hashes
    from .checkpoints import load_training_checkpoint
    from .vae import load_frozen_vae
    from .infer import restore_video,verify_output_video
    from .metrics import quantize_rgb,paired_metrics,PerceptualMetrics
    runtime=configure_reproducibility(protocol['roles'][0]['config'])
    need(not runtime['flags']['cuda_initialized'],'Use a fresh process before the CUDA context')
    _hr_save_final(run/'runtime.json',runtime)
    text=torch.load(protocol['text_cache']['path'],map_location='cpu',weights_only=True)[CAPTION_PLACEHOLDER]
    need(list(text.shape)==[1,512,4096] and text.dtype==torch.bfloat16 and bool(torch.isfinite(text).all()),
         'Original native text cache required, without T5 fallback')
    seal=_hr_seal_view(protocol);reader=perceptual=None
    observations=list(completed);done=len(completed);ordinal=0;audits=[];gt={}
    try:
        perceptual=PerceptualMetrics(protocol['metric_weights']['alexnet']['path'],protocol['metric_weights']['linear']['path'],device='cpu')
        if _hr_is_ft(protocol):
            from .data import FixedCleanReader
            reader=FixedCleanReader(protocol['dataset']['path'],expected_split=protocol['scope']['source_split'],
                 cohort_seal=protocol['cohort_seal']['path'],path_map=protocol['path_map'],
                 expected_dataset_sha256=protocol['dataset']['sha256'])
        else:reader=FramewiseReader(protocol['dataset']['path'],seal,path_map=protocol['path_map'])
        torch.cuda.set_device(cfg['device'])
        for role in protocol['roles']:
            model=vae=None
            existing_audits=list((run/'roles'/role['label']).glob('state_audit_*.json'))
            remaining=(ordinal+64)>done or not existing_audits
            try:
                if remaining:
                    model,audit=load_training_checkpoint(role['checkpoint']['path'],device=cfg['device'],
                        allowed_root=next(str(Path(r).resolve()) for r in cfg['allowed_checkpoint_roots']
                                          if Path(role['checkpoint']['path']).is_relative_to(Path(r).resolve())),
                        variant='Base',model_directory=cfg['model_directory'])
                    need(audit['step']==role['step'] and audit['checkpoint_sha256']==role['checkpoint']['sha256']
                         and audit['config']==role['config'] and model.projector is None,'Wrong student-only restore payload')
                    if protocol['schema']==VMAE_EVALUATION_SCHEMA:
                        _vmae_inference_checkpoint_audit(protocol,role,audit)
                    if protocol['schema']==TRD_EVALUATION_SCHEMA:
                        _trd_inference_checkpoint_audit(protocol,role,audit)
                    model.eval();inference=copy.deepcopy(role['config']);inference['model']['local_dir']=cfg['model_directory']
                    vae=load_frozen_vae(inference,device=cfg['device'])
                    need(all(p.dtype==torch.float32 and not p.requires_grad for p in vae.parameters()),'Frozen FP32 VAE required')
                    before=tensor_state_hashes(model)
                    attempt=sum(1 for _ in (run/'roles'/role['label']).glob('load_audit_*.json'))+1
                    save(run/'roles'/role['label']/f'load_audit_{attempt:03d}.json',audit)
                for item in seal['cohort']:
                    source=target=None
                    for sample in (42,43):
                        if ordinal<done:ordinal+=1;continue
                        if source is None:
                            source=(reader.load(item['record']['id'],condition='clean',num_frames=33,start=0) if _hr_is_ft(protocol)
                                    else reader.load('Stage2-val-clean',item['record']['id'],num_frames=33,start=0))
                            target=quantize_rgb(source['hr'])
                            if _hr_video_required(protocol,item['record']['id']):
                                gt[item['record']['id']]=_hr_gt(protocol,run,psha,item,target,source['fps'])
                        directory=_hr_directory(run,role,item,sample)
                        if directory.exists():_hr_quarantine(run,directory)
                        directory.mkdir(parents=True)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                        with _crop_policy(vae) as policy:
                            video,meta=restore_video(model,vae,source['lr'],text,inference,steps=20,seed=sample,
                                                     tile_hw=(256,448),overlap=64)
                            _validate_restore(video,meta,model,sample);ledger=copy.deepcopy(policy.ledger)
                        policy=None;torch.cuda.synchronize();seconds=time.perf_counter()-started
                        measured=quantize_rgb(video);pngs=_save_pngs(measured,directory/'png')
                        full=paired_metrics(measured,target,perceptual,excluded_reference=0);full['primary_pool']='all_frames'
                        video_artifact=None
                        if _hr_video_required(protocol,item['record']['id']):
                            path=directory/'prediction.mp4';write_video(path,measured,source['fps'])
                            verify_output_video(path,frames=33,height=128,width=224,fps=source['fps']);video_artifact=binding(path)
                        observation=_hr_identity(protocol,role,item,sample)|{'metrics':_hr_scalars(full)}
                        row={'status':'completed','observation':observation,'full_frame_pair_metrics':full,
                             'source_metadata':source['metadata'],'restoration':meta,'VAE_ledger':ledger,
                             'PNG_artifacts':pngs,'video_artifact':video_artifact,'restore_seconds':seconds,
                             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'training_updates':0,
                             'inference_inputs':['LRV','native text embedding','student checkpoint'],
                             'teacher_or_HR_input_to_inferrer':False}
                        save(directory/'metrics.json',row)
                        save(directory/'complete.json',{'protocol_sha256':psha,'metrics_sha256':sha(directory/'metrics.json'),
                                                        'PNG_count':33,'video_artifact':video_artifact})
                        observations.append(observation|{'row_metrics_path':str(directory/'metrics.json'),
                                                        'row_metrics_sha256':sha(directory/'metrics.json')})
                        ordinal+=1
                        save(run/'progress.json',{'status':'running','protocol_sha256':psha,'completed_restores':len(observations),
                             'total_restores':_hr_total(protocol),'last_role':role['label'],'training_updates':0},replace=True)
                        del video,measured,full
                    source=target=None
                if remaining:
                    after=tensor_state_hashes(model)
                    need(before==after and all(p.grad is None for p in model.parameters()) and _contexts_empty(model),
                         'Evaluation changed student state')
                    role_audit={'role':role['label'],'trainable_before':before,'trainable_after':after,
                                'training_updates':0,'inference_weights':'student','restores_expected':64}
                    save(run/'roles'/role['label']/f'state_audit_{attempt:03d}.json',role_audit)
                audits.append({'role':role['label'],'restores':64,'training_updates':0,'inference_weights':'student'})
            finally:
                primary=sys.exc_info()[1]
                _cleanup_preserving(primary,[lambda:model.clear_features() if model is not None else None,
                                             lambda:vae.clear_cache() if vae is not None else None])
                model=vae=policy=None;_cleanup_preserving(primary,[gc.collect,torch.cuda.empty_cache])
    finally:
        primary=sys.exc_info()[1];_cleanup_preserving(primary,[lambda:reader.close() if reader is not None else None])
        perceptual=None;_cleanup_preserving(primary,[gc.collect,torch.cuda.empty_cache])
    total=_hr_total(protocol)
    need(len(observations)==total and ordinal==total,'Incomplete clean study matrix')
    _hr_save_final(run/'observations.json',{'observations':observations,'protocol_sha256':psha})
    _hr_save_final(run/'role_audits.json',audits)
    summary={'status':'completed_clean_evaluation','protocol_sha256':psha,'restores':total,'roles':len(protocol['roles']),'training_updates':0,
             'primary_metric':'all33_frame_LPIPS','quality_gate_applied':False,'old_EMA_G2_G3_G4_applied':False,
             'fixed_video_clip_ids':protocol['fixed_video_clip_ids'],'prediction_PNGs':total*33,
             'prediction_MP4s':total if _hr_is_ft(protocol) else 60,'GT_clips':len(_hr_gt_ids(protocol)),
             'source_windows':'complete33frameclips_not_source_full_length'}
    if _hr_is_ft(protocol):summary.update(phase=protocol['phase'],source_split=protocol['scope']['source_split'],
                                         historical_test_exposure_disclosed=protocol['data_independence'])
    _hr_save_final(run/'summary.json',summary);return summary


def hr_execute(run_dir, expected_protocol_sha, *, resume=False, _schema=HR_EVALUATION_SCHEMA):
    import fcntl
    run=Path(run_dir).resolve();protocol=read(run/'protocol.json');psha=sha(run/'protocol.json')
    need(protocol['schema']==_schema and psha==expected_protocol_sha,'Exact clean-study protocol SHA required')
    state=read(run/'status.json')
    need(state['protocol_sha256']==psha and state['status']!='completed','Completed runs use hr-verify, never re-execute')
    need(state['status']=='planned' or resume,'Explicit --resume required after an interruption/failure')
    need(all(os.environ.get(k)==v for k,v in STRICT_ENVIRONMENT.items()),'Strict offline environment required')
    need(protocol==_hr_collector(protocol)(protocol['config_binding']['path'],run),'Prepared sources/data/training chain changed')
    seal=_hr_seal_view(protocol)
    with (run/'hr_execute.lock').open('a+b') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('This HR evaluation already has an active executor') from error
        completed=_hr_completed_prefix(protocol,run,psha,seal)
        attempt=len(list((run/'attempts').glob('attempt_*.json')))+1
        save(run/'attempts'/f'attempt_{attempt:03d}.json',{'protocol_sha256':psha,'resume_requested':resume,
             'verified_rows_reused':len(completed),'previous_status':state['status'],'automatic_protocol_change':False})
        save(run/'status.json',{'status':'running','protocol_sha256':psha,'restores':len(completed),'training_updates':0},replace=True)
        try:
            from .checkpoints import gpu_lease
            cfg=protocol['config']
            with gpu_lease(cfg['gpu_lock'],cfg['device'],cfg['gpu_admission']) as receipt:
                save(run/'attempts'/f'GPU_admission_{attempt:03d}.json',receipt)
                summary=_hr_execute_loaded(protocol,run,psha,completed)
            result={'status':'completed','protocol_sha256':psha,'restores':_hr_total(protocol),'training_updates':0,
                    'summary_sha256':sha(run/'summary.json'),'quality_gate_applied':False}
            save(run/'status.json',result,replace=True);return result
        except BaseException as error:
            failure={'status':'failed_resumable_same_protocol','protocol_sha256':psha,
                     'error':str(error),'traceback':traceback.format_exc(),'training_updates':0,
                     'no_automatic_protocol_change':True}
            save(run/'attempts'/f'failure_{attempt:03d}.json',failure)
            save(run/'status.json',failure,replace=True);raise
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def hr_verify(run_dir, *, _schema=HR_EVALUATION_SCHEMA):
    run=Path(run_dir).resolve();protocol,state=read(run/'protocol.json'),read(run/'status.json')
    psha=sha(run/'protocol.json')
    total=_hr_total(protocol)
    need(protocol['schema']==_schema and state['status']=='completed'
         and state['protocol_sha256']==psha and state['restores']==total and state['training_updates']==0,
         'A completed same-protocol clean matrix is required')
    need(protocol==_hr_collector(protocol)(protocol['config_binding']['path'],run),'Completed clean protocol inputs changed')
    seal=_hr_seal_view(protocol);rows=_hr_completed_prefix(protocol,run,psha,seal)
    need(len(rows)==total and read(run/'observations.json')=={'observations':rows,'protocol_sha256':psha},
         'Missing, duplicate or changed clean observations')
    audits=read(run/'role_audits.json')
    need(audits==[{'role':r['label'],'restores':64,'training_updates':0,'inference_weights':'student'}
                  for r in protocol['roles']],'Incomplete role audit')
    for role in protocol['roles']:
        records=list((run/'roles'/role['label']).glob('state_audit_*.json'))
        need(records,'Missing executed student state audit')
        for path in records:
            value=read(path);need(value['trainable_before']==value['trainable_after'] and value['training_updates']==0,
                                  'Student state changed during a restore attempt')
    for record in _hr_gt_ids(protocol):
        item=next(x for x in seal['cohort'] if x['record']['id']==record)
        need((run/'ground_truth'/record/'complete.json').is_file(),'Missing GT fixed clip')
        _hr_gt(protocol,run,psha,item,None,item['cached_header']['fps'])
    summary=read(run/'summary.json')
    need(sha(run/'summary.json')==state['summary_sha256'] and summary['restores']==total
         and summary['protocol_sha256']==psha and summary['quality_gate_applied'] is False,'Summary completion chain differs')
    return {'status':'verified_complete_clean_evaluation','run':str(run),'protocol_sha256':psha,
            'restores':total,'prediction_PNGs_verified':total*33,'prediction_MP4s_verified':total if _hr_is_ft(protocol) else 60,
            'GT_PNGs_verified':len(_hr_gt_ids(protocol))*33,'GT_MP4s_verified':len(_hr_gt_ids(protocol)),'training_updates':0,
            'observations_sha256':sha(run/'observations.json'),'summary_sha256':sha(run/'summary.json'),
            'quality_gate_applied':False,
            'validation_not_heldout_test':protocol.get('scope',{}).get('source_split','val')=='val'}


def _ft_test_access_policy(evaluation):
    """Separate formal test RGB/scoring from a bound integrity-only exception."""
    need(evaluation['test_pixels_after_selection'] is True
         and evaluation['test_prior_exposure']=='historical_models_previously_evaluated',
         'Disclose historical test use and freeze this-round selection-before-test policy')
    exception=evaluation.get('test_integrity_thumbnail_exception')
    if exception is not None:
        need(isinstance(exception,dict) and exception.get('scope')=='integrity_only'
             and exception.get('automated') is True and exception.get('human_quality_review') is False
             and exception.get('model_scores_used') is False,
             'Preselection test thumbnails must remain automated integrity-only, without human quality review or model scores')
        _hr_binding(exception['audit_receipt'])
    return {'test_pixels_after_selection_scope':'formal_RGB_evaluation_assets_and_model_scores',
            'test_integrity_thumbnail_exception':exception,
            'no_preselection_test_pixel_access_claimed':False}


def ft_collect_protocol(config_path, run_dir):
    """Independent val/test metadata, without old EMA gates or test-as-val aliases."""
    from .data import FixedCleanReader
    config_path=Path(config_path).resolve();cfg=read(config_path);run=Path(run_dir).resolve()
    phase=cfg['phase'];need(phase in ('val','test'),'Explicit val/test phase required')
    run_root=Path(cfg['run_root']).resolve()
    need(run!=run_root and run.is_relative_to(run_root),'Evaluation must be a new child of its output root')
    need(str(cfg['device']).startswith('cuda:') and str(cfg['device'])[5:].isdigit()
         and Path(cfg['gpu_lock']).is_absolute() and cfg['gpu_admission'].get('expected_gpu_uuid'),
         'Explicit CUDA device, shared lock and pinned host UUID required')
    study_file=_hr_binding(cfg['study_config'])
    training=_hr_training_evidence(cfg['training_run'],study_file,fixed_teacher=True)
    study=training['study'];evaluation=study['evaluation']
    access_policy=_ft_test_access_policy(evaluation)
    seal_binding=binding(cfg['seal']);dataset_binding=binding(cfg['dataset'])
    selection=None;selection_binding=None
    if phase=='val':
        need(seal_binding==_hr_binding(evaluation['validation_seal'])
             and dataset_binding==_hr_binding(evaluation['validation_dataset']),
             'Validation inputs changed after prospective study freeze')
    else:
        selection_binding=_hr_binding(cfg['selection_receipt'])
        selection=read(selection_binding['path'])
        expected=_ft_selection_payload(cfg['validation_run'])
        need(selection==expected and selection['study_file']==study_file,
             'Test selection must reproduce complete validation results and the exact study')
        need(seal_binding==_hr_binding(evaluation['test_cohort_plan'])
             and selection['test_cohort_plan']==seal_binding,'Test cohort changed after selection')
        dataset=read(cfg['dataset'])
        need(dataset.get('selection_receipt')==selection_binding
             and dataset.get('cohort_seal')==seal_binding
             and dataset.get('materialization',{}).get('selection_frozen_before_decode') is True,
             'Test pixels must be materialized only after the same frozen selection receipt')
    roles=_hr_role_specs(cfg,training,fixed_teacher=True,phase=phase,selection=selection)
    mapping=cfg.get('path_map',{})
    reader=FixedCleanReader(cfg['dataset'],expected_split=phase,cohort_seal=cfg['seal'],
                           path_map=mapping,expected_dataset_sha256=dataset_binding['sha256'])
    try:description=reader.describe()
    finally:reader.close()
    cohort,assets=description['cohort'],description['assets']
    need(description['source_split']==phase and description['dataset_binding']==dataset_binding
         and len(cohort)==len(assets)==32,'Complete32 source-tagged clean clips required')
    ids=[x['record']['id'] for x in cohort]
    need(len(set(ids))==len({x['record']['source_id'] for x in cohort})==32
         and {x['record_id'] for x in assets}==set(ids),'Require32 distinct sources without row selection')
    bindings=[]
    for item in assets:
        bindings.extend([_hr_binding(item['HR']),_hr_binding(item['clean'])])
    need(not run.is_relative_to(Path(cfg['dataset']).resolve().parent),'Output cannot overwrite clean inputs')
    text=_hr_binding(cfg['text_cache'])
    need(all(r['config']['data']['text_cache']['sha256']==text['sha256'] for r in roles),'Fixed native text cache changed')
    weights=cfg['metric_weights'];need(set(weights)=={'alexnet','linear'},'Explicit frozen LPIPS weights required')
    videos=[x['record']['id'] for x in sorted(cohort,key=lambda r:r['record']['source_id'])[:3]]
    if 'fixed_video_clip_ids' in cfg:need(cfg['fixed_video_clip_ids']==videos,'Predeclared comparison clips changed')
    protocol={'schema':FT_EVALUATION_SCHEMA,'phase':phase,'config':cfg,'config_binding':binding(config_path),
        'sources':_hr_sources(),'study_file':study_file,'training_evidence':training,'roles':roles,
        'cohort_seal':seal_binding,'cohort_seal_digest':description['seal_digest'],'cohort':cohort,
        'dataset':dataset_binding,'clean_assets':assets,'assets':bindings,'path_map':mapping,
        'text_cache':text,'metric_weights':{k:_hr_binding(v) for k,v in weights.items()},
        'official_model':model_bindings(cfg['model_directory']),
        'runtime':runtime_bindings(cfg.get('dependency_root')),'strict_environment':STRICT_ENVIRONMENT,
        'fixed_video_clip_ids':videos,
        'data_independence':{'reader':description['data_independence'],'source_split':phase,
                            'test_prior_exposure':evaluation['test_prior_exposure'],
                            'historically_pristine_test':False,
                            'test_materialization_requires_validation_selection':True,**access_policy},
        'scope':{'roles':len(roles),'clips':32,'sampling_seeds':[42,43],'conditions':['clean'],
                 'total_restores':len(roles)*64,'source_split':phase,'frames':33,
                 'HR_hw':[128,224],'LR_hw':[32,56],'steps':20,'CFG':1.,'tile_hw':[256,448],
                 'overlap':64,'primary_pool':'all33_frames','primary_metric':'lpips_alex_v0_1',
                 'video_policy':'all_clips','training_updates':0,'row_resume_same_protocol_allowed':True,
                 'source_windows':'complete frozen33-frame clips, not full source videos'}}
    if phase=='test':protocol.update(selection_receipt=selection_binding,validation_selection=selection)
    return protocol


def ft_prepare(config_path, run_dir):
    run=Path(run_dir).resolve();need(not run.exists(),'Use a new fixed-teacher evaluation directory')
    protocol=ft_collect_protocol(config_path,run);run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    result={'status':'planned','protocol_sha256':sha(run/'protocol.json'),'run_dir':str(run),
            'phase':protocol['phase'],'restores':0,'training_updates':0}
    save(run/'status.json',result);return result


def ft_execute(run_dir, expected_protocol_sha, *, resume=False):
    return hr_execute(run_dir,expected_protocol_sha,resume=resume,_schema=FT_EVALUATION_SCHEMA)


def ft_verify(run_dir):
    return hr_verify(run_dir,_schema=FT_EVALUATION_SCHEMA)


def _ft_selection_payload(validation_run):
    """Recompute final-checkpoint choices from the complete val set, never test."""
    import statistics
    run=Path(validation_run).resolve()
    verification=ft_verify(run)
    protocol=read(run/'protocol.json')
    need(protocol['phase']=='val' and protocol['scope']['total_restores']==1024,
         'Selection requires the complete five-trajectory validation phase')
    study=read(protocol['study_file']['path'])
    access_policy=_ft_test_access_policy(study['evaluation'])
    rows=read(run/'observations.json')['observations'];records=[x['record']['id'] for x in protocol['cohort']]
    index={(r['role'],r['record_id'],r['sampling_seed']):r for r in rows}
    need(len(index)==1024,'Validation selection cannot drop or duplicate rows')
    finals={r['arm']:r for r in protocol['roles'] if r['update']==2048}
    need(set(finals)=={a['label'] for a in study['arms']},'Missing prospective final checkpoint')
    means={arm:statistics.mean(statistics.mean(index[role['label'],record,s]['metrics']['lpips_alex_v0_1']
                    for s in (42,43)) for record in records) for arm,role in finals.items()}
    selected=[];candidates={}
    for family in ('baseline','hr_vae','hr_dino'):
        arms=[a for a in study['arms'] if a['family']==family]
        need(len(arms)==(1 if family=='baseline' else 2),'Keep both calibrated candidates per method family')
        chosen=min(arms,key=lambda a:(means[a['label']],a['weight']))
        role=finals[chosen['label']]
        selected.append({k:role[k] for k in ('label','arm','family','setting','weight','step','update',
                                            'cumulative_training_seconds','checkpoint','config_binding')})
        candidates[family]={a['label']:{'weight':a['weight'],'mean_val_LPIPS':means[a['label']]} for a in arms}
    return {'schema':'fixed_teacher_validation_selection_v1','study_file':protocol['study_file'],
            'validation_run':str(run),'validation_protocol':binding(run/'protocol.json'),
            'validation_status':binding(run/'status.json'),'validation_summary':binding(run/'summary.json'),
            'validation_observations':binding(run/'observations.json'),'validation_verification':verification,
            'runtime_sources':protocol['sources'],'test_cohort_plan':study['evaluation']['test_cohort_plan'],
            'candidate_final_means':candidates,'selected_roles':selected,
            'selection_rule':'per-family final2048 all33 LPIPS mean(sampling42/43 then32sources); minimum; exact tie lower weight',
            'all_candidates_reported':True,'selected_after_complete_validation':True,
            'test_pixels_read_by_selection':False,'test_pixel_read_flag_scope':'this selection computation only',
            'test_prior_exposure':study['evaluation']['test_prior_exposure'],**access_policy,
            'pristine_test_claimed':False,'training_seed_repetitions':1}


def ft_select(validation_run, output):
    output=Path(output).resolve();need(not output.exists(),'Selection receipts are immutable; use a new path')
    payload=_ft_selection_payload(validation_run);save(output,payload)
    return {'status':'selected_from_complete_validation','receipt':binding(output),
            'selected_roles':payload['selected_roles'],'test_pixels_read':False}



def _trd_inference_checkpoint_audit(protocol, role, audit):
    """A training-only teacher/head is never selected as inference weights."""
    need(audit.get('training_only_state_restored') is False and audit.get('EMA_weights_selected') is False,
         'TRD inference must restore only the current student')
    if role['update']==0:return
    state=audit.get('trd_training_metadata',{})
    need(state.get('schema')=='vsr_trd_training_state_v1'
         and state.get('protocol_sha256')==protocol['training_evidence']['protocol_sha256']
         and state.get('arm')==role['arm'] and state.get('completed_updates')==role['update']
         and state.get('training_seed')==20261102 and state.get('weight')==role['weight']
         and state.get('student_layer')==9
         and state.get('EMA_updates')==(role['update'] if role['arm']=='ema' else 0),
         'TRD checkpoint training-only identity differs from its declared role')


def trd_collect_protocol(config_path, run_dir):
    """Bind three prospective teacher-target arms; no metric-driven selection."""
    from .data import FixedCleanReader
    config_path=Path(config_path).resolve();cfg=read(config_path);run=Path(run_dir).resolve()
    phase=cfg['phase'];need(phase in ('val','test'),'Explicit TRD val/test phase required')
    run_root=Path(cfg['run_root']).resolve()
    need(run!=run_root and run.is_relative_to(run_root),'Evaluation must be a new child of its output root')
    need(str(cfg['device']).startswith('cuda:') and str(cfg['device'])[5:].isdigit()
         and Path(cfg['gpu_lock']).is_absolute() and cfg['gpu_admission'].get('expected_gpu_uuid'),
         'Explicit CUDA device, shared lock and pinned host UUID required')
    study_file=_hr_binding(cfg['study_config'])
    training=_hr_training_evidence(cfg['training_run'],study_file,trd=True)
    study=training['study'];evaluation=study['evaluation']
    need(evaluation['test_prior_exposure']=='historical_models_previously_evaluated',
         'Disclose historical model and pixel access to these reused test sources')
    seal_binding=binding(cfg['seal']);dataset_binding=binding(cfg['dataset'])
    lock=None;lock_binding=None
    if phase=='val':
        need(seal_binding==_hr_binding(evaluation['validation_seal'])
             and dataset_binding==_hr_binding(evaluation['validation_dataset']),
             'Validation inputs changed after prospective study freeze')
    else:
        lock_binding=_hr_binding(cfg['finals_receipt']);lock=read(lock_binding['path'])
        expected=_trd_finals_payload(cfg['validation_run'])
        need(lock==expected and lock['study_file']==study_file,
             'Test final lock must reproduce complete validation and the exact study')
        need(seal_binding==_hr_binding(evaluation['test_cohort_plan'])
             and dataset_binding==_hr_binding(evaluation['test_dataset'])
             and lock['test_cohort_plan']==seal_binding and lock['test_dataset']==dataset_binding,
             'Previously materialized test inputs changed after prospective freeze')
    roles=_hr_role_specs(cfg,training,trd=True,phase=phase,selection=lock)
    mapping=cfg.get('path_map',{})
    reader=FixedCleanReader(cfg['dataset'],expected_split=phase,cohort_seal=cfg['seal'],
                           path_map=mapping,expected_dataset_sha256=dataset_binding['sha256'])
    try:description=reader.describe()
    finally:reader.close()
    cohort,assets=description['cohort'],description['assets']
    need(description['source_split']==phase and description['dataset_binding']==dataset_binding
         and len(cohort)==len(assets)==32,'Complete32 source-tagged clean clips required')
    ids=[x['record']['id'] for x in cohort]
    need(len(set(ids))==len({x['record']['source_id'] for x in cohort})==32
         and {x['record_id'] for x in assets}==set(ids),'Require32 distinct sources without row selection')
    bindings=[]
    for item in assets:bindings.extend([_hr_binding(item['HR']),_hr_binding(item['clean'])])
    need(not run.is_relative_to(Path(cfg['dataset']).resolve().parent),'Output cannot overwrite clean inputs')
    text=_hr_binding(cfg['text_cache'])
    need(all(r['config']['data']['text_cache']['sha256']==text['sha256'] for r in roles),'Fixed native text cache changed')
    weights=cfg['metric_weights'];need(set(weights)=={'alexnet','linear'},'Explicit frozen LPIPS weights required')
    videos=[x['record']['id'] for x in sorted(cohort,key=lambda r:r['record']['source_id'])[:3]]
    if 'fixed_video_clip_ids' in cfg:need(cfg['fixed_video_clip_ids']==videos,'Predeclared comparison clips changed')
    protocol={'schema':TRD_EVALUATION_SCHEMA,'phase':phase,'config':cfg,'config_binding':binding(config_path),
        'sources':_hr_sources(),'study_file':study_file,'training_evidence':training,'roles':roles,
        'cohort_seal':seal_binding,'cohort_seal_digest':description['seal_digest'],'cohort':cohort,
        'dataset':dataset_binding,'clean_assets':assets,'assets':bindings,'path_map':mapping,
        'text_cache':text,'metric_weights':{k:_hr_binding(v) for k,v in weights.items()},
        'official_model':model_bindings(cfg['model_directory']),
        'runtime':runtime_bindings(cfg.get('dependency_root')),'strict_environment':STRICT_ENVIRONMENT,
        'fixed_video_clip_ids':videos,
        'data_independence':{'reader':description['data_independence'],'source_split':phase,
            'test_prior_exposure':evaluation['test_prior_exposure'],'historically_pristine_test':False,
            'test_assets_previously_materialized':True,'this_study_weight_selection':False,
            'test_scoring_requires_complete_validation_lock':True},
        'scope':{'roles':len(roles),'clips':32,'sampling_seeds':[42,43],'conditions':['clean'],
            'total_restores':len(roles)*64,'source_split':phase,'frames':33,
            'HR_hw':[128,224],'LR_hw':[32,56],'steps':20,'CFG':1.,'tile_hw':[256,448],
            'overlap':64,'primary_pool':'all33_frames','primary_metric':'lpips_alex_v0_1',
            'video_policy':'all_clips','training_updates':0,'row_resume_same_protocol_allowed':True,
            'weight_or_checkpoint_selection':False,'old_EMA_G2_G3_G4_applied':False,
            'source_windows':'complete frozen33-frame clips, not full source videos'}}
    need(protocol['scope']['total_restores']==(640 if phase=='val' else 192),'TRD matrix budget changed')
    if phase=='test':protocol.update(finals_receipt=lock_binding,validation_final_lock=lock)
    return protocol


def trd_prepare(config_path, run_dir):
    run=Path(run_dir).resolve();need(not run.exists(),'Use a new TRD evaluation directory')
    protocol=trd_collect_protocol(config_path,run);run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    result={'status':'planned','protocol_sha256':sha(run/'protocol.json'),'run_dir':str(run),
            'phase':protocol['phase'],'restores':0,'training_updates':0}
    save(run/'status.json',result);return result


def trd_execute(run_dir, expected_protocol_sha, *, resume=False):
    return hr_execute(run_dir,expected_protocol_sha,resume=resume,_schema=TRD_EVALUATION_SCHEMA)


def trd_verify(run_dir):
    result=hr_verify(run_dir,_schema=TRD_EVALUATION_SCHEMA)
    run=Path(run_dir).resolve();protocol=read(run/'protocol.json')
    need(protocol['phase'] in ('val','test')
         and result['restores']==(640 if protocol['phase']=='val' else 192),'TRD complete matrix required')
    for role in protocol['roles']:
        audits=list((run/'roles'/role['label']).glob('load_audit_*.json'))
        need(audits,'Missing TRD student checkpoint load audit')
        for path in audits:
            audit=read(path)
            need(audit['step']==role['step'] and audit['checkpoint_sha256']==role['checkpoint']['sha256']
                 and audit['config']==role['config'],'TRD persisted load audit differs from its role')
            _trd_inference_checkpoint_audit(protocol,role,audit)
    return result|{'training_only_checkpoint_roles_verified':True,'all_three_arms_retained':True,
                   'weight_or_checkpoint_selection':False}


def _trd_finals_payload(validation_run):
    """Completion lock only: keep every predefined final, without reading scores."""
    run=Path(validation_run).resolve();verification=trd_verify(run)
    protocol=read(run/'protocol.json')
    need(protocol['phase']=='val' and verification['restores']==640,
         'Final lock requires the entire640-row validation matrix')
    study=read(protocol['study_file']['path']);evaluation=study['evaluation']
    finals=[r for r in protocol['roles'] if r['update']==2048]
    need([r['arm'] for r in finals]==['baseline','frozen','ema']
         and len(finals)==3 and all(r['step']==2048 for r in finals),
         'Retain baseline/frozen/EMA final2048 in prospective order, without selection')
    return {'schema':'wan_trd_validation_final_lock_v1','study_file':protocol['study_file'],
        'validation_run':str(run),'validation_protocol':binding(run/'protocol.json'),
        'validation_status':binding(run/'status.json'),'validation_summary':binding(run/'summary.json'),
        'validation_observations':binding(run/'observations.json'),'validation_verification':verification,
        'runtime_sources':protocol['sources'],'test_cohort_plan':_hr_binding(evaluation['test_cohort_plan']),
        'test_dataset':_hr_binding(evaluation['test_dataset']),
        'final_roles':[{k:r[k] for k in ('label','arm','family','setting','weight','step','update',
                       'cumulative_training_seconds','checkpoint','config_binding')} for r in finals],
        'selection_rule':'none; all three prospective final2048 checkpoints',
        'metric_values_used_to_select':False,'test_pixels_read_by_lock':False,
        'test_prior_exposure':evaluation['test_prior_exposure'],'pristine_test_claimed':False,
        'training_seed_repetitions':1}


def trd_lock_finals(validation_run, output):
    output=Path(output).resolve();need(not output.exists(),'Final lock receipts are immutable; use a new path')
    payload=_trd_finals_payload(validation_run);save(output,payload)
    return {'status':'all_finals_locked_after_complete_validation','receipt':binding(output),
            'final_roles':payload['final_roles'],'metric_values_used_to_select':False}


def trd_build_config(*, training_run, phase, output, run_root, validation_run=None, finals_receipt=None):
    """Create a metadata config from complete training; never execute or select scores."""
    need(phase in ('val','test'),'Explicit val/test phase required')
    destination=Path(output).resolve();need(not destination.exists(),'Config is immutable; use a new output path')
    training=Path(training_run).resolve();protocol=read(training/'protocol.json')
    study_file=_hr_binding(protocol['study_file']);evidence=_hr_training_evidence(training,study_file,trd=True)
    study=evidence['study'];evaluation=study['evaluation']
    roles=[]
    if phase=='val':
        roles.append({'label':'shared_initial','arm':'shared','weight':0.,'step':8192,'update':0,
            'checkpoint':evidence['base']['checkpoint'],'cumulative_training_seconds':0.})
        for arm in ('baseline','frozen','ema'):
            item=evidence['arms'][arm]
            for cp in item['checkpoint_files']:
                if cp['step'] not in (512,1024,2048):continue
                roles.append({'label':f"{arm}_{cp['step']:06d}",'arm':arm,'weight':item['weight'],
                    'step':cp['step'],'update':cp['step'],'checkpoint':{k:cp[k] for k in ('path','bytes','sha256')},
                    'cumulative_training_seconds':cp['cumulative_training_seconds']})
    else:
        need(validation_run is not None and finals_receipt is not None,'Test requires verified complete validation lock')
        lock_binding=_hr_binding({'path':str(Path(finals_receipt).resolve()),'sha256':sha(finals_receipt)})
        receipt=read(lock_binding['path']);expected=_trd_finals_payload(validation_run)
        need(receipt==expected and receipt['study_file']==study_file,'Final receipt differs from full validation')
        roles=[{k:r[k] for k in ('label','arm','family','setting','weight','step','update','checkpoint','cumulative_training_seconds')}
               for r in receipt['final_roles']]
    roots=list(dict.fromkeys([str(Path(study['allowed_base_root']).resolve()),str(training)]))
    cfg={'schema':'wan_trd_evaluation_config_v1','phase':phase,'run_root':str(Path(run_root).resolve()),
        'study_config':study_file,'training_run':str(training),'roles':roles,'allowed_checkpoint_roots':roots,
        'seal':_hr_binding(evaluation['validation_seal' if phase=='val' else 'test_cohort_plan'])['path'],
        'dataset':_hr_binding(evaluation['validation_dataset' if phase=='val' else 'test_dataset'])['path'],
        'text_cache':_hr_binding(evidence['base']['config']['data']['text_cache']),
        'metric_weights':{k:_hr_binding(v) for k,v in study['metric_weights'].items()},
        'model_directory':study['model_directory'],'device':study['device'],'gpu_lock':study['gpu_lock'],
        'gpu_admission':protocol['admission'],'dependency_root':study.get('dependency_root','/home/hugo/Projects/VSR/.deps'),
        'path_map':{}}
    if phase=='test':cfg.update(validation_run=str(Path(validation_run).resolve()),finals_receipt=lock_binding)
    _hr_role_specs(cfg,evidence,trd=True,phase=phase,selection=receipt if phase=='test' else None)
    save(destination,cfg)
    return {'status':'metadata_config_created_no_execution','config':binding(destination),'phase':phase,
            'roles':len(roles),'restores':len(roles)*64,'training_updates':0,'metrics_used_for_selection':False}



def _vmae_inference_checkpoint_audit(protocol, role, audit):
    """A training-only teacher/head is never selected as inference weights."""
    need(audit.get('training_only_state_restored') is False and audit.get('EMA_weights_selected') is False,
         'VideoMAEv2 inference must restore only the current student')
    if role['update']==0:return
    state=audit.get('videomaev2_training_metadata',{})
    need(state.get('schema')=='vsr_videomaev2_training_state_v1'
         and state.get('protocol_sha256')==protocol['training_evidence']['protocol_sha256']
         and state.get('arm')==role['arm'] and state.get('completed_updates')==role['update']
         and state.get('training_seed')==20261102 and state.get('weight')==role['weight']
         and state.get('student_layer')==17
         and state.get('EMA_updates')==0
         and state.get('teacher_parameter_updates')==0
         and state.get('teacher_forward_count')==(role['update'] if role['arm']=='videomaev2' else 0),
         'VideoMAEv2 checkpoint training-only identity differs from its declared role')


def vmae_collect_protocol(config_path, run_dir):
    """Bind two prospective online-teacher arms; no metric-driven selection."""
    from .data import FixedCleanReader
    config_path=Path(config_path).resolve();cfg=read(config_path);run=Path(run_dir).resolve()
    need(cfg.get('schema')=='vsr_videomaev2_evaluation_config_v1','Explicit VideoMAEv2 evaluation config schema required')
    phase=cfg['phase'];need(phase in ('val','test'),'Explicit VideoMAEv2 val/test phase required')
    run_root=Path(cfg['run_root']).resolve()
    need(run!=run_root and run.is_relative_to(run_root),'Evaluation must be a new child of its output root')
    need(str(cfg['device']).startswith('cuda:') and str(cfg['device'])[5:].isdigit()
         and Path(cfg['gpu_lock']).is_absolute() and cfg['gpu_admission'].get('expected_gpu_uuid'),
         'Explicit CUDA device, shared lock and pinned host UUID required')
    study_file=_hr_binding(cfg['study_config'])
    training=_hr_training_evidence(cfg['training_run'],study_file,videomaev2=True)
    study=training['study'];evaluation=study['evaluation']
    need(evaluation['test_prior_exposure']=='historical_models_previously_evaluated',
         'Disclose historical model and pixel access to these reused test sources')
    seal_binding=binding(cfg['seal']);dataset_binding=binding(cfg['dataset'])
    lock=None;lock_binding=None
    if phase=='val':
        need(seal_binding==_hr_binding(evaluation['validation_seal'])
             and dataset_binding==_hr_binding(evaluation['validation_dataset']),
             'Validation inputs changed after prospective study freeze')
    else:
        lock_binding=_hr_binding(cfg['finals_receipt']);lock=read(lock_binding['path'])
        expected=_vmae_finals_payload(cfg['validation_run'])
        need(lock==expected and lock['study_file']==study_file,
             'Test final lock must reproduce complete validation and the exact study')
        need(seal_binding==_hr_binding(evaluation['test_cohort_plan'])
             and dataset_binding==_hr_binding(evaluation['test_dataset'])
             and lock['test_cohort_plan']==seal_binding and lock['test_dataset']==dataset_binding,
             'Previously materialized test inputs changed after prospective freeze')
    roles=_hr_role_specs(cfg,training,videomaev2=True,phase=phase,selection=lock)
    mapping=cfg.get('path_map',{})
    reader=FixedCleanReader(cfg['dataset'],expected_split=phase,cohort_seal=cfg['seal'],
                           path_map=mapping,expected_dataset_sha256=dataset_binding['sha256'])
    try:description=reader.describe()
    finally:reader.close()
    cohort,assets=description['cohort'],description['assets']
    need(description['source_split']==phase and description['dataset_binding']==dataset_binding
         and len(cohort)==len(assets)==32,'Complete32 source-tagged clean clips required')
    ids=[x['record']['id'] for x in cohort]
    need(len(set(ids))==len({x['record']['source_id'] for x in cohort})==32
         and {x['record_id'] for x in assets}==set(ids),'Require32 distinct sources without row selection')
    bindings=[]
    for item in assets:bindings.extend([_hr_binding(item['HR']),_hr_binding(item['clean'])])
    need(not run.is_relative_to(Path(cfg['dataset']).resolve().parent),'Output cannot overwrite clean inputs')
    text=_hr_binding(cfg['text_cache'])
    need(all(r['config']['data']['text_cache']['sha256']==text['sha256'] for r in roles),'Fixed native text cache changed')
    weights=cfg['metric_weights'];need(set(weights)=={'alexnet','linear'},'Explicit frozen LPIPS weights required')
    videos=[x['record']['id'] for x in sorted(cohort,key=lambda r:r['record']['source_id'])[:3]]
    if 'fixed_video_clip_ids' in cfg:need(cfg['fixed_video_clip_ids']==videos,'Predeclared comparison clips changed')
    protocol={'schema':VMAE_EVALUATION_SCHEMA,'phase':phase,'config':cfg,'config_binding':binding(config_path),
        'sources':_hr_sources(),'study_file':study_file,'training_evidence':training,'roles':roles,
        'cohort_seal':seal_binding,'cohort_seal_digest':description['seal_digest'],'cohort':cohort,
        'dataset':dataset_binding,'clean_assets':assets,'assets':bindings,'path_map':mapping,
        'text_cache':text,'metric_weights':{k:_hr_binding(v) for k,v in weights.items()},
        'official_model':model_bindings(cfg['model_directory']),
        'runtime':runtime_bindings(cfg.get('dependency_root')),'strict_environment':STRICT_ENVIRONMENT,
        'fixed_video_clip_ids':videos,
        'data_independence':{'reader':description['data_independence'],'source_split':phase,
            'test_prior_exposure':evaluation['test_prior_exposure'],'historically_pristine_test':False,
            'test_assets_previously_materialized':True,'this_study_weight_selection':False,
            'test_scoring_requires_complete_validation_lock':True},
        'scope':{'roles':len(roles),'clips':32,'sampling_seeds':[42,43],'conditions':['clean'],
            'total_restores':len(roles)*64,'source_split':phase,'frames':33,
            'HR_hw':[128,224],'LR_hw':[32,56],'steps':20,'CFG':1.,'tile_hw':[256,448],
            'overlap':64,'primary_pool':'all33_frames','primary_metric':'lpips_alex_v0_1',
            'video_policy':'all_clips','training_updates':0,'row_resume_same_protocol_allowed':True,
            'weight_or_checkpoint_selection':False,'old_EMA_G2_G3_G4_applied':False,
            'source_windows':'complete frozen33-frame clips, not full source videos'}}
    need(protocol['scope']['total_restores']==(448 if phase=='val' else 128),'VideoMAEv2 matrix budget changed')
    if phase=='test':protocol.update(finals_receipt=lock_binding,validation_final_lock=lock)
    return protocol


def vmae_prepare(config_path, run_dir):
    run=Path(run_dir).resolve();need(not run.exists(),'Use a new VideoMAEv2 evaluation directory')
    protocol=vmae_collect_protocol(config_path,run);run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    result={'status':'planned','protocol_sha256':sha(run/'protocol.json'),'run_dir':str(run),
            'phase':protocol['phase'],'restores':0,'training_updates':0}
    save(run/'status.json',result);return result


def vmae_execute(run_dir, expected_protocol_sha, *, resume=False):
    return hr_execute(run_dir,expected_protocol_sha,resume=resume,_schema=VMAE_EVALUATION_SCHEMA)


def vmae_verify(run_dir):
    result=hr_verify(run_dir,_schema=VMAE_EVALUATION_SCHEMA)
    run=Path(run_dir).resolve();protocol=read(run/'protocol.json')
    need(protocol['phase'] in ('val','test')
         and result['restores']==(448 if protocol['phase']=='val' else 128),'VideoMAEv2 complete matrix required')
    for role in protocol['roles']:
        audits=list((run/'roles'/role['label']).glob('load_audit_*.json'))
        need(audits,'Missing VideoMAEv2 student checkpoint load audit')
        for path in audits:
            audit=read(path)
            need(audit['step']==role['step'] and audit['checkpoint_sha256']==role['checkpoint']['sha256']
                 and audit['config']==role['config'],'VideoMAEv2 persisted load audit differs from its role')
            _vmae_inference_checkpoint_audit(protocol,role,audit)
    return result|{'training_only_checkpoint_roles_verified':True,'both_arms_retained':True,
                   'weight_or_checkpoint_selection':False}


def _vmae_finals_payload(validation_run):
    """Completion lock only: keep every predefined final, without reading scores."""
    run=Path(validation_run).resolve();verification=vmae_verify(run)
    protocol=read(run/'protocol.json')
    need(protocol['phase']=='val' and verification['restores']==448,
         'Final lock requires the entire448-row validation matrix')
    study=read(protocol['study_file']['path']);evaluation=study['evaluation']
    finals=[r for r in protocol['roles'] if r['update']==2048]
    need([r['arm'] for r in finals]==['baseline','videomaev2']
         and len(finals)==2 and all(r['step']==2048 for r in finals),
         'Retain baseline/VideoMAEv2 final2048 in prospective order, without selection')
    return {'schema':'vsr_videomaev2_validation_final_lock_v1','study_file':protocol['study_file'],
        'validation_run':str(run),'validation_protocol':binding(run/'protocol.json'),
        'validation_status':binding(run/'status.json'),'validation_summary':binding(run/'summary.json'),
        'validation_observations':binding(run/'observations.json'),'validation_verification':verification,
        'runtime_sources':protocol['sources'],'test_cohort_plan':_hr_binding(evaluation['test_cohort_plan']),
        'test_dataset':_hr_binding(evaluation['test_dataset']),
        'final_roles':[{k:r[k] for k in ('label','arm','family','setting','weight','step','update',
                       'cumulative_training_seconds','checkpoint','config_binding')} for r in finals],
        'selection_rule':'none; both prospective final2048 checkpoints',
        'metric_values_used_to_select':False,'test_pixels_read_by_lock':False,
        'test_prior_exposure':evaluation['test_prior_exposure'],'pristine_test_claimed':False,
        'training_seed_repetitions':1}


def vmae_lock_finals(validation_run, output):
    output=Path(output).resolve();need(not output.exists(),'Final lock receipts are immutable; use a new path')
    payload=_vmae_finals_payload(validation_run);save(output,payload)
    return {'status':'all_finals_locked_after_complete_validation','receipt':binding(output),
            'final_roles':payload['final_roles'],'metric_values_used_to_select':False}


def vmae_build_config(*, training_run, phase, output, run_root, validation_run=None, finals_receipt=None):
    """Create a metadata config from complete training; never execute or select scores."""
    need(phase in ('val','test'),'Explicit val/test phase required')
    destination=Path(output).resolve();need(not destination.exists(),'Config is immutable; use a new output path')
    training=Path(training_run).resolve();protocol=read(training/'protocol.json')
    study_file=_hr_binding(protocol['study_file']);evidence=_hr_training_evidence(training,study_file,videomaev2=True)
    study=evidence['study'];evaluation=study['evaluation']
    roles=[]
    if phase=='val':
        roles.append({'label':'shared_initial','arm':'shared','weight':0.,'step':8192,'update':0,
            'checkpoint':evidence['base']['checkpoint'],'cumulative_training_seconds':0.})
        for arm in ('baseline','videomaev2'):
            item=evidence['arms'][arm]
            for cp in item['checkpoint_files']:
                if cp['step'] not in (512,1024,2048):continue
                roles.append({'label':f"{arm}_{cp['step']:06d}",'arm':arm,'weight':item['weight'],
                    'step':cp['step'],'update':cp['step'],'checkpoint':{k:cp[k] for k in ('path','bytes','sha256')},
                    'cumulative_training_seconds':cp['cumulative_training_seconds']})
    else:
        need(validation_run is not None and finals_receipt is not None,'Test requires verified complete validation lock')
        lock_binding=_hr_binding({'path':str(Path(finals_receipt).resolve()),'sha256':sha(finals_receipt)})
        receipt=read(lock_binding['path']);expected=_vmae_finals_payload(validation_run)
        need(receipt==expected and receipt['study_file']==study_file,'Final receipt differs from full validation')
        roles=[{k:r[k] for k in ('label','arm','family','setting','weight','step','update','checkpoint','cumulative_training_seconds')}
               for r in receipt['final_roles']]
    roots=list(dict.fromkeys([str(Path(study['allowed_base_root']).resolve()),str(training)]))
    cfg={'schema':'vsr_videomaev2_evaluation_config_v1','phase':phase,'run_root':str(Path(run_root).resolve()),
        'study_config':study_file,'training_run':str(training),'roles':roles,'allowed_checkpoint_roots':roots,
        'seal':_hr_binding(evaluation['validation_seal' if phase=='val' else 'test_cohort_plan'])['path'],
        'dataset':_hr_binding(evaluation['validation_dataset' if phase=='val' else 'test_dataset'])['path'],
        'text_cache':_hr_binding(evidence['base']['config']['data']['text_cache']),
        'metric_weights':{k:_hr_binding(v) for k,v in study['metric_weights'].items()},
        'model_directory':study['model_directory'],'device':study['device'],'gpu_lock':study['gpu_lock'],
        'gpu_admission':protocol['admission'],'dependency_root':study.get('dependency_root','/home/hugo/Projects/VSR/.deps'),
        'path_map':{}}
    if phase=='test':cfg.update(validation_run=str(Path(validation_run).resolve()),finals_receipt=lock_binding)
    _hr_role_specs(cfg,evidence,videomaev2=True,phase=phase,selection=receipt if phase=='test' else None)
    save(destination,cfg)
    return {'status':'metadata_config_created_no_execution','config':binding(destination),'phase':phase,
            'roles':len(roles),'restores':len(roles)*64,'training_updates':0,'metrics_used_for_selection':False}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare',help='metadata and byte hashes only; create a fresh prospective protocol')
    p.add_argument('--config',type=Path,required=True); p.add_argument('--run-dir',type=Path,required=True)
    p = sub.add_parser('execute',help='explicit GPU execution of a reviewed planned protocol')
    p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--expected-protocol-sha',required=True)
    p = sub.add_parser('verify',help='verify all row/PNG/source seals and recompute the complete gate')
    p.add_argument('--run-dir',type=Path,required=True)
    p = sub.add_parser('hr-prepare',help='bind the independent clean-x4 HR-latent study; no GPU load')
    p.add_argument('--config',type=Path,required=True); p.add_argument('--run-dir',type=Path,required=True)
    p = sub.add_parser('hr-execute',help='execute640 clean restores, optionally resume verified same-protocol rows')
    p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--expected-protocol-sha',required=True)
    p.add_argument('--resume',action='store_true')
    p = sub.add_parser('hr-verify',help='verify the complete640-row clean evaluation and all prediction artifacts')
    p.add_argument('--run-dir',type=Path,required=True)
    p = sub.add_parser('ft-prepare',help='bind fixed-teacher val1024 or selected test192 metadata')
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p = sub.add_parser('ft-execute',help='execute or resume same-protocol clean fixed-teacher rows')
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--expected-protocol-sha',required=True)
    p.add_argument('--resume',action='store_true')
    p = sub.add_parser('ft-verify',help='verify the complete val/test rows, source chain, PNG and MP4 artifacts')
    p.add_argument('--run-dir',type=Path,required=True)
    p = sub.add_parser('ft-select',help='verify fullval and compute immutable per-family final-weight choices before test decode')
    p.add_argument('--validation-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('trd-build-config',help='metadata-only val640/test192 config from completed paired training')
    p.add_argument('--training-run',type=Path,required=True);p.add_argument('--phase',choices=('val','test'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--validation-run',type=Path);p.add_argument('--finals-receipt',type=Path)
    p=sub.add_parser('trd-prepare',help='bind baseline/frozen/EMA val640 or all-final test192 metadata')
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p=sub.add_parser('trd-execute',help='execute or resume verified same-protocol Wan TRD rows')
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--expected-protocol-sha',required=True)
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('trd-verify',help='verify complete Wan TRD sources, all rows, PNGs, MP4s and student identities')
    p.add_argument('--run-dir',type=Path,required=True)
    p=sub.add_parser('trd-lock-finals',help='lock all three final2048 students after full validation, without score selection')
    p.add_argument('--validation-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('vmae-build-config',help='metadata-only val448/test128 config from completed online VideoMAEv2 study')
    p.add_argument('--training-run',type=Path,required=True);p.add_argument('--phase',choices=('val','test'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--validation-run',type=Path);p.add_argument('--finals-receipt',type=Path)
    p=sub.add_parser('vmae-prepare',help='bind baseline/online-VideoMAEv2 val448 or all-final test128 metadata')
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p=sub.add_parser('vmae-execute',help='execute or resume verified same-protocol VideoMAEv2 evaluation rows')
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--expected-protocol-sha',required=True)
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('vmae-verify',help='verify all VideoMAEv2-study rows, PNGs, MP4s and student-only identities')
    p.add_argument('--run-dir',type=Path,required=True)
    p=sub.add_parser('vmae-lock-finals',help='lock both final2048 students after full validation, without score selection')
    p.add_argument('--validation-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'vmae-build-config': result = vmae_build_config(training_run=args.training_run,phase=args.phase,
        output=args.output,run_root=args.run_root,validation_run=args.validation_run,finals_receipt=args.finals_receipt)
    elif args.command == 'vmae-prepare': result = vmae_prepare(args.config,args.run_dir)
    elif args.command == 'vmae-execute': result = vmae_execute(args.run_dir,args.expected_protocol_sha,resume=args.resume)
    elif args.command == 'vmae-verify': result = vmae_verify(args.run_dir)
    elif args.command == 'vmae-lock-finals': result = vmae_lock_finals(args.validation_run,args.output)
    elif args.command == 'trd-build-config': result = trd_build_config(training_run=args.training_run,phase=args.phase,
        output=args.output,run_root=args.run_root,validation_run=args.validation_run,finals_receipt=args.finals_receipt)
    elif args.command == 'trd-prepare': result = trd_prepare(args.config,args.run_dir)
    elif args.command == 'trd-execute': result = trd_execute(args.run_dir,args.expected_protocol_sha,resume=args.resume)
    elif args.command == 'trd-verify': result = trd_verify(args.run_dir)
    elif args.command == 'trd-lock-finals': result = trd_lock_finals(args.validation_run,args.output)
    elif args.command == 'ft-prepare': result = ft_prepare(args.config,args.run_dir)
    elif args.command == 'ft-execute': result = ft_execute(args.run_dir,args.expected_protocol_sha,resume=args.resume)
    elif args.command == 'ft-verify': result = ft_verify(args.run_dir)
    elif args.command == 'ft-select': result = ft_select(args.validation_run,args.output)
    elif args.command == 'hr-prepare': result = hr_prepare(args.config,args.run_dir)
    elif args.command == 'hr-execute': result = hr_execute(args.run_dir,args.expected_protocol_sha,resume=args.resume)
    elif args.command == 'hr-verify': result = hr_verify(args.run_dir)
    elif args.command == 'prepare': result = prepare(args.config,args.run_dir)
    elif args.command == 'execute': result = execute(args.run_dir,args.expected_protocol_sha)
    else:
        _, evidence = validate_completed_evaluation(args.run_dir)
        result = {k:v for k,v in evidence.items() if k != 'G2'}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__': main()
