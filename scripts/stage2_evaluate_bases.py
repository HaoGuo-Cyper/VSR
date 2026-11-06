#!/usr/bin/env python3
"""Prepare/execute the complete frozen Stage2 val32 Base evaluation matrix.

prepare: JSON, hashes and package metadata only; no torch or pixel decoding.
execute: original native restore_video/UniPC and LPIPS/TecoGAN, zero updates.
Requires all three new completed Base8192 checkpoints and a fresh staging run.
No resume, row limits, codec regeneration, test/external data or score selection.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import weakref

sys.dont_write_bytecode = True
from stage2_framewise import FramewiseReader, digest, file_sha, read, validate_seal
from stage2_metrics import need, score_pools, summarize_g2

OLD = Path('/home/hugo/Projects/VSR')
SEEDS = [20261102, 20261103, 20261104]
CONDITIONS = ['clean', 'blur', 'noise', 'codec_allintra', 'drop_repeat', 'drop_black']
ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    'PYTORCH_ALLOC_CONF': 'expandable_segments:True', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false'}
PACKAGE_MODULES = {'torch':'torch','numpy':'numpy','diffusers':'diffusers','transformers':'transformers',
    'peft':'peft','accelerate':'accelerate','safetensors':'safetensors','Pillow':'PIL',
    'opencv-python-headless':'cv2','lpips':'lpips','scikit-image':'skimage','scipy':'scipy',
    'torchvision':'torchvision','lazy-loader':'lazy_loader','tifffile':'tifffile'}
PACKAGES = list(PACKAGE_MODULES)
POLICY_SHA = 'd94340b33868912f43633c71ef4a34a97b2dd7eb7b74e9d40ded363b09417270'
HQ_HELPER_SHA = '6304f49d5a30ad0a26fd8ae5e8a52d782dd12ee5dd6148b87b099f03d93c50c7'
ALEX_SHA = '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
LINEAR_SHA = 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'
NATIVE_SOURCES = {'WanTransformer3DModel':'diffusers/models/transformers/transformer_wan.py',
    'AutoencoderKLWan':'diffusers/models/autoencoders/autoencoder_kl_wan.py',
    'UniPCMultistepScheduler':'diffusers/schedulers/scheduling_unipc_multistep.py'}


def save(path, value, *, replace=False):
    path = Path(path)
    need(replace or not path.exists(), 'Refusing to overwrite: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def binding(path, expected=None):
    path = Path(path).resolve()
    value = file_sha(path)
    need(expected is None or value == expected, 'Changed sealed file: '+str(path))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': value}


def dependency_spec(dependency_root):
    """Resolve and bind explicit local metric dependencies without importing them.

    The full 41MiB local dependency tree is bound (except disposable bytecode).
    Installed runtime distributions bind their METADATA/RECORD and entry source;
    this does not claim to rehash multi-GB installed PyTorch/CUDA distributions.
    """
    root = Path(dependency_root).resolve()
    need(root.is_dir(), 'Explicit dependency_root is missing; installing/falling back is forbidden')
    before = list(sys.path)
    try:
        sys.path.insert(0,str(root))
        packages = {}
        for name,module in PACKAGE_MODULES.items():
            distribution = importlib.metadata.distribution(name)
            spec = importlib.machinery.PathFinder.find_spec(module,sys.path)
            need(spec is not None and spec.origin is not None, 'Missing dependency source: '+module)
            source = Path(spec.origin).resolve()
            if name in ('lpips','scikit-image','lazy-loader','tifffile'):
                need(source.is_relative_to(root), 'Metric package must resolve inside explicit dependency_root: '+module)
            metadata = {kind:binding(distribution.locate_file(item)) for item in (distribution.files or [])
                        for kind in ('METADATA','RECORD') if str(item).endswith('.dist-info/'+kind)}
            need(set(metadata)=={'METADATA','RECORD'}, 'Missing auditable distribution metadata: '+name)
            packages[name] = {'version':distribution.version,'module':module,'source':binding(source),'metadata':metadata}
        files = {str(path.relative_to(root)):binding(path) for path in sorted(root.rglob('*'))
                 if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo')}
        need(files and all(Path(b['path']).is_relative_to(root) for b in files.values()), 'Dependency tree contains external file links')
        return {'root':str(root),'packages':packages,'local_tree':files,
            'binding_scope':'All non-bytecode files in explicit local dependency tree; runtime distribution METADATA/RECORD and package entry sources. No installs or alternate package fallback.'}
    finally:
        sys.path[:] = before


def verify_imported_dependencies(dependencies):
    """Called after strict setup, before model loading; verify actual imports."""
    for name,item in dependencies['packages'].items():
        module = importlib.import_module(item['module'])
        need(binding(module.__file__) == item['source'], 'Imported dependency source shadowed pinned package: '+name)
        need(importlib.metadata.version(name)==item['version'], 'Imported distribution version changed: '+name)


def native_source_bindings(dependency_root):
    """Locate three numerical implementations through distribution metadata."""
    before = list(sys.path)
    try:
        sys.path.insert(0,str(Path(dependency_root).resolve()))
        distribution = importlib.metadata.distribution('diffusers')
        return {name:binding(distribution.locate_file(relative)) for name,relative in NATIVE_SOURCES.items()}
    finally:
        sys.path[:] = before


def verify_native_sources(sources,classes):
    need(set(sources)==set(classes)==set(NATIVE_SOURCES), 'Require all three native numerical implementations')
    for name,native_class in classes.items():
        need(binding(inspect.getfile(native_class)) == sources[name], 'Imported native numerical implementation changed: '+name)


def safe_output(path, project, asset_root):
    path, project, asset_root = Path(path).resolve(), Path(project).resolve(), Path(asset_root).resolve()
    need(project != OLD and not project.is_relative_to(OLD), 'Historical project is read-only')
    need(path != project/'runs' and path.is_relative_to(project/'runs') and not path.is_relative_to(asset_root),
         'Evaluation output must be a fresh staging runs/ child, outside input assets')
    return path


def parse_checkpoints(values):
    result = {}
    for value in values:
        key, separator, path = value.partition('=')
        need(separator == '=' and key.isdigit(), 'Use TRAINING_SEED=/absolute/Base/checkpoint_008192.pt')
        seed = int(key)
        need(seed in SEEDS and seed not in result and Path(path).is_absolute(), 'Duplicate/unknown seed or relative checkpoint path')
        result[seed] = Path(path).resolve()
    need(sorted(result) == SEEDS, 'Exactly the three prespecified new Base seeds are required')
    return result


def expected_effective_config(source, *, cache_from=None):
    """Exact stdlib mirror of stage2_base_queue.effective_config, no probe."""
    config = copy.deepcopy(read(source))
    need('_sha256' not in config and '_source_config_sha256' not in config
         and 'cache_from' not in config['data'], 'Expected original sealed raw Base config')
    config['_source_config_sha256'] = file_sha(source)
    if cache_from is not None:
        config['data']['cache_from'] = str(Path(cache_from).resolve())
    config['_sha256'] = hashlib.sha256((json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    return config


def cache_producer_spec(source, allowed_root, seal):
    """Bind the completed seed02 cache producer, without reading tensor data."""
    source, allowed_root = Path(source).resolve(), Path(allowed_root).resolve()
    need(source != allowed_root and source.is_relative_to(allowed_root), 'Producer escaped new actual run root')
    config_path = source/'config.json'
    expected = expected_effective_config(seal['base_configs'][str(SEEDS[0])]['path'])
    need(read(config_path) == expected, 'Cache producer is not the exact seed02 effective config')
    status, selection, geometry, environment = (read(source/name) for name in
        ('status.json','selection.json','correspondence.json','environment.json'))
    need(status['status'] == 'completed' and status.get('seed') == SEEDS[0] and status.get('profile') is False
         and status.get('test_accessed') is False and status.get('run') == str(source), 'Seed02 cache producer must be completed formal Base')
    manifest = read(seal['manifest'])
    train_ids = [r['id'] for r in sorted((r for r in manifest['records'] if r['split']=='train'),
                 key=lambda r:hashlib.sha256(f"train:20261102:{r['id']}".encode()).hexdigest())]
    expected_selection = {'manifest_sha256':seal['manifest_digest'],'training_ids':train_ids,
        'validation_ids':[r['record']['id'] for r in seal['cohort']],'test_accessed':False,'caption_policy':expected['caption_policy']}
    need(len(train_ids) == 1641 and selection == expected_selection, 'Cache producer train1641/val32 selection differs')
    need(geometry['latent_grid'] == [9,16,28] and geometry['token_grid'] == [9,8,14], 'Producer latent/token geometry differs')
    cache_seal_path = source/'cache_asset_seal.json';cache_seal = read(cache_seal_path)
    need(cache_seal.get('source_run') == str(source) and cache_seal.get('contains_task_initialization') is False,
         'Cache seal source/task-state exclusion differs')
    required = {'config.json','selection.json','correspondence.json','fresh_training_cache.pt','fresh_text_embeddings.pt'}
    required.update(f'clip_{i}/{name}' for i in range(1673) for name in ('metadata.json','paired_rgb.pt','fresh_latents.pt'))
    artifact_map = cache_seal.get('artifact_sha256',{})
    need(set(artifact_map) == required and all(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) for h in artifact_map.values()),
         'Producer seal must cover exactly the full frozen1673-clip data, not task checkpoints')
    for name in ('config.json','selection.json','correspondence.json'):
        need(file_sha(source/name) == artifact_map[name], 'Producer metadata changed after cache seal')
    need(artifact_map['fresh_text_embeddings.pt'] == expected['data']['text_cache']['sha256'], 'Producer sealed text differs')
    encoder_names = ['vsr/data.py','vsr/preflight.py','vsr/stage2_preparation.py']
    encoders = {name:environment['code_sha256'][name] for name in encoder_names}
    need(all(re.fullmatch('[0-9a-f]{64}',value) for value in encoders.values()), 'Missing producer encoder provenance')
    return {'source_run':str(source),'selection':selection,'geometry':geometry,'encoder_hashes':encoders,
        'seal':binding(cache_seal_path),'metadata':{name:binding(source/name) for name in
            ('config.json','status.json','selection.json','correspondence.json','environment.json')},
        'payloads_rehashed_or_loaded_by_evaluation_prepare':False,
        'scope':'Producer/consumer cache provenance is checked; independent evaluation consumes the sealed val32 NPY assets, not these training-cache tensors.'}


def checkpoint_spec(path, seed, allowed_root, seal, *, expected_cache_source):
    path, allowed_root = Path(path).resolve(), Path(allowed_root).resolve()
    need(allowed_root != OLD/'runs' and not allowed_root.is_relative_to(OLD)
         and not allowed_root.is_relative_to(Path('/home/hugo/Projects/ReprVSR-pretrained')),
         'Explicit allowed_root must be the new actual checkpoint run root')
    need(path.is_relative_to(allowed_root) and path.name == 'checkpoint_008192.pt' and path.parent.name == 'Base',
         'Checkpoint escaped new allowed_root or is not final Base8192')
    run = path.parent.parent
    config_path, summary_path, init_path, selection_path = (run/'config.json', path.parent/'summary.json',
                                                          path.parent/'initialization.json', run/'selection.json')
    config, summary, init, selection = (read(p) for p in (config_path, summary_path, init_path, selection_path))
    need(file_sha(seal['base_configs'][str(seed)]['path']) == seal['base_configs'][str(seed)]['sha256'],
         'Base source config changed after cohort seal')
    expected = expected_effective_config(seal['base_configs'][str(seed)]['path'],
        cache_from=None if seed==SEEDS[0] else expected_cache_source)
    need(config == expected, 'Actual Base must match the exact queue effective config, including runtime provenance and canonical SHA')
    run_status = read(run/'status.json')
    need(run_status['status']=='completed' and run_status.get('seed')==seed and run_status.get('profile') is False
         and run_status.get('test_accessed') is False and run_status.get('run')==str(run), 'Formal Base run has not completed')
    need(config['seed'] == 20261102 and config['training'].get('seed',config['seed']) == seed
         and config['training']['updates'] == 8192 and config['training']['variants'] == ['Base']
         and config['data']['num_frames'] == 33 and config['data']['temporal_stride'] == 1
         and [config['data']['height'],config['data']['width']] == [128,224]
         and config['reproducibility']['mode'] == 'strict'
         and not config.get('reference',{}).get('enabled',False), 'Wrong training seed/geometry/family or backend')
    need(summary['status'] == 'completed' and summary['updates'] == 8192, 'Base training is not completed8192')
    need(init['fresh_official_initialization'] is True and init['legacy_state_loaded'] is False
         and init.get('training_seed',init.get('seed')) == seed and init['data_seed'] == 20261102
         and init['official_revision'] == config['model']['revision'], 'Fresh Base initialization identity differs')
    need(selection['test_accessed'] is False and selection['validation_ids'] == [r['record']['id'] for r in seal['cohort']],
         'Base validation cohort or heldout isolation differs')
    origin = {name:binding(p) for name,p in [('config',config_path),('summary',summary_path),('initialization',init_path),
                                           ('selection',selection_path),('status',run/'status.json')]}
    if seed != SEEDS[0]:
        producer = cache_producer_spec(expected_cache_source,allowed_root,seal)
        reuse_path = run/'cache_reuse.json';reuse = read(reuse_path)
        expected_reuse = {'status':'verified','source_run':producer['source_run'],'clip_count':1673,
            'training_clip_count':1641,'validation_clip_count':32,'manifest_sha256':seal['manifest_digest'],
            'official_revision':config['model']['revision'],'encoder_source_hashes':producer['encoder_hashes'],
            'legacy_state_read':False,'temporal_stride':1,'processing_data_version':'hq-vsr-paired-v2',
            'tensor_loading':'lazy LRU<=2; weights_only=True CPU','manifest_records_checked':True,
            'source_asset_seal_sha256':producer['seal']['sha256'],'source_encoder_hashes':producer['encoder_hashes'],
            'task_checkpoint_loaded':False,'T5_loaded':False,'DINO_loaded':False}
        need(reuse == expected_reuse, 'Consumer cache_reuse provenance differs from the completed seed02 producer')
        need(selection == producer['selection'] and read(run/'correspondence.json') == producer['geometry'],
             'Consumer cache selection/geometry differs from producer')
        environment = read(run/'environment.json')
        need(all(environment['code_sha256'].get(k)==v for k,v in producer['encoder_hashes'].items()),
             'Consumer encoder code differs from producer cache provenance')
        origin.update(cache_reuse=binding(reuse_path),correspondence=binding(run/'correspondence.json'),environment=binding(run/'environment.json'))
    else:
        need(Path(expected_cache_source).resolve()==run and not (run/'cache_reuse.json').exists(),
             'Seed02 must be the exact fresh cache producer, not a cache consumer')
    return {'training_seed':seed, 'checkpoint':binding(path), 'config':config,'origin':origin}


def official_bindings(config, manifest_path):
    manifest = read(manifest_path)
    need(manifest['sha'] == config['model']['revision'], 'Official model revision changed')
    root = Path(config['model']['local_dir'])
    result = []
    for item in manifest['siblings']:
        relative = item['rfilename']
        if not relative.startswith(('transformer/','vae/','scheduler/')):
            continue
        path = root/relative
        if 'lfs' in item:
            result.append(binding(path,item['lfs']['sha256']))
        else:
            raw = path.read_bytes()
            need(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == item['blobId'], 'Official configuration/blob changed')
            result.append(binding(path))
    return result


def collect_protocol(args):
    root = args.project_root.resolve()
    need(Path.cwd().resolve() == root, 'FramewiseReader requires staging project root as CWD')
    seal = read(args.seal)
    validate_seal(seal)
    need(seal['project_root'] == str(root), 'Use the cohort-sealed staging project root')
    dataset = read(args.dataset)
    need(dataset['status'] == 'completed_CPU_assets_no_predictions' and dataset['cohort_seal_digest'] == seal['seal_digest']
         and dataset['config'] == seal['config'], 'Wrong/incomplete CPU asset dataset')
    need(sorted(x['record_id'] for x in dataset['assets']) == sorted(x['record']['id'] for x in seal['cohort']), 'Asset cohort differs')
    assets = {}
    asset_root = args.dataset.resolve().parent
    for row in dataset['assets']:
        need(set(row['conditions']) == set(CONDITIONS[1:]), 'Missing/substituted degradation type')
        specs = [row['HR'],row['clean'],*[row['conditions'][c]['asset'] for c in CONDITIONS[1:]]]
        for spec in specs:
            path = Path(spec['path']).resolve()
            need(path.is_relative_to(asset_root) and str(path) not in assets, 'Asset path escaped/duplicated')
            assets[str(path)] = binding(path,spec['sha256'])
    need(len(assets) == 224, 'Require all224 authoritative NPY files')
    parsed = parse_checkpoints(args.checkpoint)
    producer_run = parsed[SEEDS[0]].parent.parent
    producer = cache_producer_spec(producer_run,args.allowed_checkpoint_root,seal)
    checkpoints = {str(seed):checkpoint_spec(path,seed,args.allowed_checkpoint_root,seal,expected_cache_source=producer_run)
                   for seed,path in sorted(parsed.items())}
    cfg = checkpoints[str(SEEDS[0])]['config']
    text_cache = cfg['data']['text_cache']
    for spec in checkpoints.values():
        need(spec['config']['model'] == cfg['model'] and spec['config']['data']['text_cache'] == text_cache,
             'Model/text input differs across new Base seeds')
    caption_preflight = root/'scripts/frozen_external/outputs/VSR_final_evaluation_text_cache_preflight.json'
    old_caption = read(caption_preflight)['entries']['NoRef_Base_seed20261102_step8192']
    need(Path(text_cache['path']).resolve() == Path(old_caption['cache_path']).resolve()
         and text_cache['sha256'] == old_caption['cache_file_sha256'], 'Cached native caption binding changed; T5 fallback forbidden')
    dependencies = dependency_spec(args.dependency_root)
    native_sources = native_source_bindings(dependencies['root'])
    versions = {name:item['version'] for name,item in dependencies['packages'].items()}
    helpers = {'policy':binding(root/'scripts/frozen_external/work/t1_vae_cache_policy.py',POLICY_SHA),
               'ledger':binding(root/'scripts/frozen_external/work/run_frozen_HQ_test_T1_policy.py',HQ_HELPER_SHA),
               'gpu_classifier':binding(root/'scripts/frozen_external/work/gpu_process_classification.py')}
    protocol = {'schema':'stage2_complete_Base_val32_evaluation_v1', 'project_root':str(root),
        'allowed_checkpoint_root':str(args.allowed_checkpoint_root.resolve()), 'worker':binding(__file__),
        'cohort_seal':binding(args.seal), 'cohort_seal_digest':seal['seal_digest'], 'dataset_manifest':binding(args.dataset),
        'assets':assets, 'checkpoints':checkpoints,'cache_producer':producer, 'official_model_manifest':binding(args.official_manifest),
        'official_files':official_bindings(cfg,args.official_manifest), 'installed_Wan_VAE_source':native_sources['AutoencoderKLWan'],
        'native_numerical_sources':native_sources,
        'metric_weights':[binding(args.metric_root/'alexnet-owt-7be5be79.pth',ALEX_SHA),binding(args.metric_root/'lpips_v0.1_alex.pth',LINEAR_SHA)],
        'caption':{'text':seal['config']['caption'],'cache':binding(text_cache['path'],text_cache['sha256']),
                   'preflight':binding(caption_preflight),'shape':old_caption['shape'],'dtype':old_caption['dtype'],
                   'embedding_sha256':old_caption['embedding_sha256'],'T5_fallback':False},
        'code':{str(p):file_sha(p) for p in [*sorted((root/'vsr').glob('*.py')),root/'scripts/stage2_framewise.py',root/'scripts/stage2_metrics.py']},
        'helpers':helpers,'desktop_binding':desktop_binding(args.desktop_pid,args.desktop_name,helpers['gpu_classifier']['path']),
        'dependencies':dependencies,
        'runtime':{'packages':versions,'python':sys.version,'executable':str(Path(sys.executable).resolve()),'environment':ENVIRONMENT},
        'scope':{'training_seeds':SEEDS,'record_ids':[r['record']['id'] for r in seal['cohort']], 'conditions':CONDITIONS,
            'sampling_seeds':[42,43],'frames':33,'temporal_stride':1,'HR_hw':[128,224],'LR_hw':[32,56],
            'steps':20,'cfg':1.0,'sampler':'original restore_video/sample_tile_latent/official-config UniPC',
            'tile_hw':[256,448],'overlap':64,'actual_tiles':1,'HRI':False,'temporal_windowing':False,
            'views_per_checkpoint':192,'restores_per_checkpoint':384,'total_restores_and_observations':1152,'prediction_PNG_count':38016,
            'no_row_limits_or_selection':True,'test_and_external_pixels_allowed':False},
        'VAE_policy':{'lifecycle':'public remove/reinstall same VAE after every successful restore; remove finally',
            'numerical_delegate':'unchanged original restore_video only','expected_shape':[1,3,33,128,224],
            'per_checkpoint':{'restores':384,'rotations':384,'original_delegate':384,'events':384,'T1':0},
            'max_records_per_restore':2,'cache_check':'native _feat_map/_enc_feat_map must be lists of None after restore; no replacement encode/decode'},
        'output':{'fresh_only':True,'resume':False,'automatic_retry':False,'save_prediction_PNGs':True,
                  'scoring':'original paired_metrics over common quantize_rgb PNG-equivalent output/HR; full33 per-frame/original-pair metrics then frozen mask pools'},
        'quality_gate':seal['config']['G2'], 'training_updates':0,
        'cost_note':'1152 restores. Existing synthetic33 extra-forward difference ~0.090s suggests sampler-only ~0.58h at20 steps; complete VAE/CPUmetrics/PNG/I/O duration is unmeasured, provisionally1–3h, not a guarantee.'}
    # Normalize before freezing so tuple-versus-JSON-list metadata can never
    # invalidate an otherwise unchanged prepared run after JSON round-trip.
    return json.loads(json.dumps(protocol,ensure_ascii=False,allow_nan=False))


def import_helper(path,name):
    before = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = before


def vae_cache_state(vae):
    result = {}
    for name in ('_feat_map','_enc_feat_map'):
        value = getattr(vae,name,None)
        result[name] = {'slots':len(value) if isinstance(value,list) else None,
                       'all_None':isinstance(value,list) and all(x is None for x in value)}
    return result


class RotatingCropRestore:
    """Shape33 observer around the sole original numerical restore delegate."""
    def __init__(self, restore, install, validate_ledger, finite, contexts_empty):
        self.restore,self.install,self.validate_ledger = restore,install,validate_ledger
        self.finite,self.contexts_empty = finite,contexts_empty
        self.role = {'restores':0,'rotations':0,'original_delegate':0,'events':0,'T1':0}
        self.last_audit = None

    @staticmethod
    def removed(vae):
        return 'encode' not in vae.__dict__ and getattr(vae,'_vsr_t1_cache_policy',None) is None

    def register(self,vae,config):
        need(self.removed(vae) and not config.get('reference',{}).get('enabled',False), 'Expected unwrapped NoRef VAE')
        self.config_digest = digest(config)
        self.install(vae,max_records=2)

    def __call__(self,model,vae,lr,text,config,*,seed):
        active = getattr(vae,'_vsr_t1_cache_policy',None)
        audit = {'status':'running','numeric_restore_calls':0,'cleanup_errors':[]}
        self.last_audit = audit
        try:
            need(active is not None and active.ledger == [] and active.dropped_records == 0
                 and self.config_digest == digest(config) and list(lr.shape) == [1,3,33,32,56] and seed in (42,43),
                 'Dirty policy, changed config or out-of-scope input')
            audit['numeric_restore_calls'] = 1
            video,meta = self.restore(model,vae,lr,text,config,steps=20,seed=seed,tile_hw=(256,448),overlap=64,
                                      reference_image=None,reference_index=None)
            audit['ledger'] = copy.deepcopy(active.ledger)
            need(active.dropped_records == 0 and meta['tile_count'] == len(meta['tiles']) == 1
                 and meta['reference']['enabled'] is False and meta['reference']['encode_count'] == 0,
                 'Changed native crop/reference geometry')
            self.validate_ledger(audit['ledger'],False,1)
            need(audit['ledger'][0]['input_shape'] == [1,3,33,128,224], 'VAE did not encode the complete original33-frame crop')
            need(list(video.shape) == meta['output_shape'] == [1,3,33,128,224]
                 and meta['sampling']['steps'] == 20 and meta['sampling']['seed'] == seed and meta['sampling']['cfg'] == 1.0
                 and meta['sampling']['global_noise_shape'] == [1,16,9,16,28]
                 and meta['sampling']['noise_device'] == 'cpu' and meta['sampling']['sampler'] == 'local official-config UniPC'
                 and meta['temporal_windowing'] is False and meta['padding']['time_end'] == 0
                 and meta['inference_inputs'] == ['LRV','native text embedding']
                 and self.finite(video) and self.contexts_empty(model), 'Native output/sampler/context contract failed')
            audit['native_VAE_end_cache'] = vae_cache_state(vae)
            need(all(x['all_None'] for x in audit['native_VAE_end_cache'].values()), 'Native VAE temporal cache leaked across restores')
            active.remove()
            need(self.removed(vae), 'Policy removal did not restore original class encode')
            self.install(vae,max_records=2)
            for key in ('restores','rotations','original_delegate','events'):
                self.role[key] += 1
            audit.update(status='passed',original_encode_restored_before_reinstall=True,next_policy_installed=True,
                         no_parameter_hash_per_restore=True)
            meta['stage2_policy_audit'] = copy.deepcopy(audit)
            return video,meta
        except BaseException as error:
            audit.update(status='failed',error=str(error))
            for policy in (getattr(vae,'_vsr_t1_cache_policy',None),active):
                if policy is not None:
                    try:policy.remove()
                    except BaseException as cleanup:audit['cleanup_errors'].append(str(cleanup))
            try:vae.clear_cache()
            except BaseException as cleanup:audit['cleanup_errors'].append(str(cleanup))
            audit['original_encode_restored_after_failure'] = self.removed(vae)
            raise

    def finish(self,vae,expected=384):
        need(self.role == {'restores':expected,'rotations':expected,'original_delegate':expected,'events':expected,'T1':0},
             'Incomplete per-checkpoint native restore budget')
        vae._vsr_t1_cache_policy.remove()
        need(self.removed(vae), 'Final policy removal failed')
        return copy.deepcopy(self.role)


def process_start_ticks(pid):
    return int((Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19])


def desktop_binding(pid,name,classifier_path):
    classifier = import_helper(classifier_path,'stage2_desktop_prepare')
    info = classifier.read_process_info(pid)
    classification = classifier.classify_inventory([f'{pid}, {name}, 0 MiB'],own_pid=None,own_worker_verified=False,
                                                    expected_uid=os.getuid(),proc_reader=lambda _:info)
    need(classification['all_inventory_processes_verified_allowed'] and classification['observations'][0]['classification'] == 'verified_desktop_gnome_shell_gjs_console',
         'Only the independently verified same-user system GNOME gjs can be pinned as desktop')
    return {'PID':pid,'reported_name':name,'start_ticks':process_start_ticks(pid),'UID':os.getuid(),
            'process_metadata':info,'executable':binding(info['exe']),'memory_limit_MiB':256}


def gpu_idle_guard(desktop,classifier,*,runner=subprocess.run,proc_reader=None,start_reader=process_start_ticks):
    result = runner(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                     '--format=csv,noheader'],capture_output=True,text=True,timeout=20)
    classification = classifier.classify_inventory(result.stdout.splitlines(),own_pid=None,own_worker_verified=False,
        expected_uid=desktop['UID'],proc_reader=proc_reader or classifier.read_process_info,query_success=result.returncode==0)
    pinned=[]
    for row in classification['observations']:
        try:
            matches = (row.get('classification')=='verified_desktop_gnome_shell_gjs_console'
                and row.get('pid')==desktop['PID'] and row.get('reported_process_name')==desktop['reported_name']
                and start_reader(row['pid'])==desktop['start_ticks']
                and row.get('process_metadata')==desktop['process_metadata'])
        except (OSError,ValueError):matches=False
        pinned.append(matches)
    return {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'classification':classification,
            'pinned_desktop_matches':pinned,'passed':classification['all_inventory_processes_verified_allowed'] and all(pinned)}


@contextmanager
def gpu_lease(allowed_root):
    """Same independent GPU-owner lease as Base worker; not parent queue lock."""
    with (Path(allowed_root).resolve()/'stage2_gpu.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(lease,fcntl.LOCK_UN)


def score_masked(full,mask):
    pool = full['all_frames']
    return score_pools(pool,{'lpips_per_frame':pool['lpips_per_frame'],
                            'tecogan':{'per_pair':pool['per_temporal_pair']}},mask)


def ordered_cases(seal):
    return [(seed,item['record']['id'],condition,sample)
            for seed in SEEDS for item in seal['cohort'] for condition in CONDITIONS for sample in (42,43)]


def prepare(args):
    root = args.project_root.resolve()
    run = safe_output(args.run_dir,root,args.dataset.resolve().parent)
    need(not run.exists(), 'Prepare requires a new absent output run')
    protocol = collect_protocol(args)
    need('torch' not in sys.modules, 'Metadata prepare must not import torch')
    run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    save(run/'status.json',{'status':'planned','protocol_sha256':file_sha(run/'protocol.json'),
                           'created_utc':datetime.now(timezone.utc).isoformat(),'training_updates':0,'torch_imported':False})
    print(json.dumps({'status':'planned','run':str(run),'protocol_sha256':file_sha(run/'protocol.json')}))


def execute(args):
    root = args.project_root.resolve()
    run = safe_output(args.run_dir,root,args.dataset.resolve().parent)
    protocol = read(run/'protocol.json')
    need(args.expected_protocol_sha and file_sha(run/'protocol.json') == args.expected_protocol_sha,
         'Exact prepared raw protocol SHA required')
    state = read(run/'status.json')
    need(state['status'] == 'planned' and state['protocol_sha256'] == args.expected_protocol_sha,
         'Only never-started planned runs execute; no resume/retry')
    with (run/'execute.lock').open('x') as stream:stream.write(str(os.getpid())+'\n')
    save(run/'status.json',{'status':'running','protocol_sha256':args.expected_protocol_sha,'training_updates':0},replace=True)
    torch = model = vae = wrapper = reader = None
    observations, roles, stable_inputs, stages = [], [], {}, {}
    current = 'CPU_bindings'
    started = time.perf_counter()
    try:
        need(protocol == collect_protocol(args), 'Prepared files/packages/config changed; make a fresh reviewed plan')
        need(all(os.environ.get(k) == v for k,v in ENVIRONMENT.items()), 'Launch with exact strict/offline environment')
        classifier = import_helper(protocol['helpers']['gpu_classifier']['path'],'stage2_gpu_guard')
        guard = gpu_idle_guard(protocol['desktop_binding'],classifier);save(run/'GPU_idle_before_load.json',guard)
        need(guard['passed'], 'GPU is busy or its compute-process state cannot be inspected; no overlap with training')
        sys.path[:0] = [str(root),protocol['dependencies']['root']]
        import torch as torch_module
        torch = torch_module
        need(not torch.cuda.is_initialized(), 'Strict flags must precede CUDA initialization')
        torch.set_num_threads(2)
        from vsr.reproducibility import configure_reproducibility, tensor_state_hashes
        runtime = configure_reproducibility(protocol['checkpoints'][str(SEEDS[0])]['config'])
        need(runtime['flags']['deterministic_algorithms'] and not runtime['flags']['deterministic_warn_only']
             and runtime['flags']['sdp_math'] and not runtime['flags']['sdp_flash']
             and not runtime['flags']['sdp_memory_efficient'] and not runtime['flags']['sdp_cudnn'], 'Strict native math backend required')
        save(run/'runtime.json',runtime)
        need(not torch.cuda.is_initialized(), 'Strict setup must preserve uninitialized CUDA before CPU imports')
        verify_imported_dependencies(protocol['dependencies'])
        from vsr.checkpoints import load_training_checkpoint
        from vsr.infer_video import load_frozen_vae
        from vsr.inference import restore_video
        from vsr.frozen_evaluation import quantize_rgb, _save_pngs, stable_hq_input_contract
        from vsr.metric_pools import paired_metrics
        from vsr.perceptual_metrics import PerceptualMetrics
        from vsr.native_qa_restore import _contexts_empty
        from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
        verify_native_sources(protocol['native_numerical_sources'],{'AutoencoderKLWan':AutoencoderKLWan,
            'WanTransformer3DModel':WanTransformer3DModel,'UniPCMultistepScheduler':UniPCMultistepScheduler})
        for name,module in list(sys.modules.items()):
            if name == 'vsr' or name.startswith('vsr.'):
                need(Path(module.__file__).resolve().is_relative_to(root/'vsr'),'Core package shadowed staging source')
        cache = torch.load(protocol['caption']['cache']['path'],map_location='cpu',weights_only=True)
        text = cache.get(protocol['caption']['text']) if isinstance(cache,dict) else None
        need(isinstance(text,torch.Tensor) and list(text.shape) == protocol['caption']['shape']
             and str(text.dtype) == protocol['caption']['dtype'] and bool(torch.isfinite(text).all()),'Invalid cached caption; no T5 fallback')
        header = json.dumps([str(text.dtype),list(text.shape)],separators=(',',':')).encode()+b'\n'
        need(hashlib.sha256(header+text.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == protocol['caption']['embedding_sha256'],
             'Native caption tensor digest differs')
        text = text.detach().cpu();del cache
        helper = {k:import_helper(v['path'],'stage2_'+k) for k,v in protocol['helpers'].items()}
        seal = read(args.seal)
        reader = FramewiseReader(args.dataset,seal)
        perceptual = PerceptualMetrics(protocol['metric_weights'][0]['path'],protocol['metric_weights'][1]['path'],device='cpu')
        need(not torch.cuda.is_initialized(), 'CPU imports/text/metric setup initialized CUDA before model load')
        for seed in SEEDS:
            current = {'training_seed':seed,'phase':'Base_model_VAE_load'}
            cp = protocol['checkpoints'][str(seed)]
            model,audit = load_training_checkpoint(cp['checkpoint']['path'],device='cuda',allowed_root=protocol['allowed_checkpoint_root'],variant='Base')
            need(audit['checkpoint_sha256'] == cp['checkpoint']['sha256'] and audit['step'] == 8192
                 and audit['config'] == cp['config'] and model.projector is None, 'Actual loaded Base differs')
            vae = load_frozen_vae(cp['config'],device='cuda')
            need(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in vae.parameters()),'Original CUDA FP32 VAE required')
            model.eval();vae.requires_grad_(False).eval()
            vae_ref,model_ref = weakref.ref(vae),weakref.ref(model)
            wrapper = RotatingCropRestore(restore_video,helper['policy'].install,helper['ledger'].validate_ledger,
                                         lambda x:bool(torch.isfinite(x).all()),_contexts_empty)
            wrapper.register(vae,cp['config'])
            trainable_before = tensor_state_hashes(model)['all_trainable']['sha256']
            save(run/f'checkpoints/{seed}/load_audit.json',audit)
            seed_started = time.perf_counter()
            for item in seal['cohort']:
                record = item['record']['id']
                for condition in CONDITIONS:
                    source = reader.load('Stage2-val-'+condition,record,num_frames=33,start=0)
                    lr,hr,metadata = source['lr'],source['hr'],source['metadata']
                    contract = stable_hq_input_contract(metadata,lr,hr,fps=source['fps'],frames=33)
                    key = condition+'/'+record
                    need(key not in stable_inputs or stable_inputs[key] == contract, 'Candidates used different exact target/condition inputs')
                    stable_inputs[key] = contract
                    _,target = quantize_rgb(hr)
                    for sample in (42,43):
                        current = {'training_seed':seed,'record_id':record,'condition':condition,'sampling_seed':sample}
                        directory = run/f'results/{seed}/{record}/{condition}/seed_{sample}'
                        need(not directory.exists(),'Fresh-only row path already exists')
                        directory.mkdir(parents=True)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before = time.perf_counter()
                        video,restoration = wrapper(model,vae,lr,text,cp['config'],seed=sample)
                        torch.cuda.synchronize()
                        restore_seconds = time.perf_counter()-before
                        peak = torch.cuda.max_memory_allocated()
                        quantized,measured = quantize_rgb(video)
                        pngs = _save_pngs(quantized,directory/'png',run)
                        need(len(pngs) == 33,'Every restore must save all33 PNG frames')
                        metric_started = time.perf_counter()
                        full = paired_metrics(measured,target,perceptual,excluded_reference=0)
                        full['primary_pool'] = 'all_frames'
                        score = score_masked(full,item['mask'])
                        observation = {'training_seed':seed,'record_id':record,'source_id':item['record']['source_id'],
                            'sampling_seed':sample,'condition':condition,'checkpoint_sha256':cp['checkpoint']['sha256'],
                            'cohort_seal_digest':seal['seal_digest'],'score':score}
                        row = {'status':'completed','observation':observation,'source_metadata':metadata,'stable_input_contract':contract,
                            'full_frame_pair_metrics':full,'restoration':restoration,'PNG_artifacts':pngs,
                            'restore_seconds':restore_seconds,'metric_seconds':time.perf_counter()-metric_started,
                            'cuda_peak_allocated_bytes':peak,'training_updates':0}
                        save(directory/'metrics.json',row)
                        save(directory/'complete.json',{'protocol_sha256':args.expected_protocol_sha,'metrics_sha256':file_sha(directory/'metrics.json'),
                                                       'PNG_count':33,'resume_allowed':False})
                        observation['row_metrics_path'] = str(directory/'metrics.json')
                        observation['row_metrics_sha256'] = file_sha(directory/'metrics.json')
                        observations.append(observation)
                        save(run/'progress.json',{'status':'running','completed_restores':len(observations),'last':current,'training_updates':0},replace=True)
                        del video,quantized,measured,full
                    del source,lr,hr,target
            trainable_after = tensor_state_hashes(model)['all_trainable']['sha256']
            need(trainable_before == trainable_after and all(p.grad is None for p in model.parameters()) and _contexts_empty(model),
                 'Evaluation changed trainable bytes or left gradients/features')
            role = wrapper.finish(vae)
            roles.append({'training_seed':seed,'policy_counters':role,'trainable_sha256_before':trainable_before,
                          'trainable_sha256_after':trainable_after,'wall_seconds':time.perf_counter()-seed_started})
            model = vae = wrapper = None
            gc.collect();torch.cuda.empty_cache()
            need(vae_ref() is None and model_ref() is None,'Previous Base/model VAE still referenced across checkpoint roles')
            save(run/'completed_checkpoint_roles.json',{'roles':roles,'training_updates':0},replace=True)
        need(len(observations) == 1152 and len(roles) == 3 and len(stable_inputs) == 192,'Evaluation matrix is incomplete')
        current = 'full_matrix_G2_aggregation'
        gate = summarize_g2(observations,seal)
        save(run/'observations.json',{'protocol_sha256':args.expected_protocol_sha,'observations':observations,'training_updates':0})
        save(run/'G2.json',gate)
        summary = {'status':'completed_full_matrix','G2_status':gate['status'],'protocol_sha256':args.expected_protocol_sha,
            'training_updates':0,'observations':1152,'prediction_PNGs':38016,'roles':roles,
            'stable_input_contracts_digest':digest(stable_inputs),'wall_seconds':time.perf_counter()-started,
            'G2_sha256':file_sha(run/'G2.json'),'observations_sha256':file_sha(run/'observations.json'),
            'test_or_external_pixels_used':False,'next_action':'report_and_stop' if gate['status'] != 'passed_G2' else 'report_G2_before_next_stage'}
        save(run/'summary.json',summary)
        save(run/'status.json',{'status':'completed_full_matrix','G2_status':gate['status'],'summary_sha256':file_sha(run/'summary.json'),
                               'protocol_sha256':args.expected_protocol_sha,'training_updates':0},replace=True)
        print(json.dumps({'status':summary['status'],'G2_status':gate['status'],'run':str(run)}))
    except BaseException as error:
        failure = {'status':'failed_stop_no_retry','case':current,'error':f'{type(error).__name__}: {error}',
            'traceback':traceback.format_exc(),'completed_restores':len(observations),'completed_roles':roles,
            'last_policy_audit':wrapper.last_audit if wrapper else None,'training_updates':0,'automatic_retry':False,
            'partial_results_not_a_G2_pass':True,'wall_seconds':time.perf_counter()-started}
        save(run/'failure.json',failure)
        save(run/'status.json',{'status':'failed_stop_no_retry','failure_sha256':file_sha(run/'failure.json'),
                               'protocol_sha256':args.expected_protocol_sha,'training_updates':0},replace=True)
        raise
    finally:
        if vae is not None:
            policy = getattr(vae,'_vsr_t1_cache_policy',None)
            if policy is not None:policy.remove()
        if model is not None:model.clear_features()
        if reader is not None:reader.close()
        model = vae = wrapper = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--seal',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',action='append',required=True)
    parser.add_argument('--allowed-checkpoint-root',type=Path,required=True)
    parser.add_argument('--official-manifest',type=Path,default=OLD/'docs/official-model-manifest.json')
    parser.add_argument('--metric-root',type=Path,default=OLD/'weights')
    parser.add_argument('--dependency-root',type=Path,default=OLD/'.deps')
    parser.add_argument('--desktop-pid',type=int,default=4816)
    parser.add_argument('--desktop-name',default='gjs')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--expected-protocol-sha')
    args = parser.parse_args()
    if args.mode == 'prepare':
        need(args.expected_protocol_sha is None,'prepare creates its own protocol SHA')
        prepare(args)
    else:
        with gpu_lease(args.allowed_checkpoint_root):execute(args)


if __name__ == '__main__':main()
