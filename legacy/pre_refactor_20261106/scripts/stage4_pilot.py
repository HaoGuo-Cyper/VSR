#!/usr/bin/env python3
"""Sealed future T4.1: three fresh matched-Base arms, 2048 updates each.

prepare reads JSON/byte hashes only. execute waits for real passed T2/G2/G3,
uses the shared exclusive GPU lifecycle and unchanged continuation Step, and
writes student checkpoints. No quality inference, G4 verdict, sweep, resume or
automatic retry is implemented. Import never loads torch or initializes CUDA.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import gc
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace
import weakref

sys.dont_write_bytecode = True
import stage3_resource_gate as g3

need,read,save,binding,file_sha = g3.need,g3.read,g3.save,g3.binding,g3.file_sha
ROLES = g3.ROLES
UPDATES,TRAIN_CLIPS = 2048,1641
CHECKPOINT_AT = (512,1024,1536,2048)
BUDGET = {'arms':3,'updates_per_arm':UPDATES,'formal_pilot_updates':6144,
          'run_equivalents':.75,'RE_denominator_updates':8192,'warmup_updates':0,
          'additional_semantic_or_diagnostic_updates':0,'checkpoints_per_arm':4,'total_checkpoints':12}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _memory_record(value):
    keys = ('peak_allocated_bytes','peak_reserved_bytes','physical_device_bytes')
    need(isinstance(value,dict) and all(type(value.get(k)) is int and value[k]>=0 for k in keys)
         and value['physical_device_bytes']>0 and value['peak_allocated_bytes']>0
         and value['peak_reserved_bytes']>=value['peak_allocated_bytes']
         and max(value[k] for k in keys[:2]) <= min(g3.MEMORY_CAP,value['physical_device_bytes']),
         'G3/pilot memory evidence exceeds16GiB/physical memory or is incomplete')


def _diagnostic_counts(value):
    expected = {'semantic_direct_core':2,'semantic_A2_rho0':2,'flow_only':22,'C1':22,'A2':22}
    need(value.get('diagnostic_optimizer_updates_applied') == 70
         and value.get('diagnostic_optimizer_updates_attempted') == 70 and value.get('completed_step_returns') == 70
         and value.get('scientific_training_updates') == 0
         and value.get('optimizer_calls_not_returned_partial_effect_unknown') == 0
         and value.get('cleanup_events') == [] and set(value.get('by_role',{})) == set(expected),
         'G3 requires66 benchmark rows plus4 semantic updates, with no unknown partial calls/cleanup events')
    for role,count in expected.items():
        need(value['by_role'][role] == {'attempted':count,'applied':count,'returned_steps':count}, 'G3 per-role update counts differ')


def completed_g3(directory,root):
    """Metadata-only G3 completion validation, including its real T2/G2 origin."""
    directory,root = Path(directory).resolve(),Path(root).resolve()
    need(not (directory/'failure.json').exists() and not (directory/'cleanup_failure.json').exists(), 'Failed/cleanup-failed G3 cannot authorize a pilot')
    status,summary,protocol,semantics,observations,gate = (read(directory/name) for name in
        ('status.json','summary.json','protocol.json','semantics.json','observations.json','G3.json'))
    psha = file_sha(directory/'protocol.json')
    need(status['status'] == summary['status'] == gate['status'] == 'passed_G3_resource_cost'
         and status['G3_passed'] is True and summary['G3_passed'] is True, 'A real passed G3 resource/semantic gate is required')
    need(status['summary_sha256'] == file_sha(directory/'summary.json')
         and summary['semantics_sha256'] == file_sha(directory/'semantics.json')
         and summary['observations_sha256'] == file_sha(directory/'observations.json')
         and summary['G3_sha256'] == file_sha(directory/'G3.json')
         and all(item['protocol_sha256'] == psha for item in (status,summary,observations)), 'G3 completion SHA chain differs')
    need(protocol['schema'] == 'stage3_G3_full_optimizer_diagnostic_v1' and protocol['project_root'] == str(root)
         and protocol['budget'] == g3.BUDGET and summary['budget'] == g3.BUDGET
         and protocol['roles'] == list(ROLES) and protocol['warmup_per_role'] == 2 and protocol['timed_per_role'] == 20,
         'G3 fixed66+4 protocol or role order differs')
    for item in (status,summary,observations):_diagnostic_counts(item)
    rows = observations['rows']
    recalculated = json.loads(json.dumps(g3.summarize_roles(rows),allow_nan=False))
    need(len(rows) == 66 and recalculated == gate and gate['primary_ratio_A2_over_flow_all20'] <= 1.5,
         'G3 all20-timed cost gate does not reproduce from its66 raw benchmark rows')
    checks = semantics['checks']
    need(semantics['status'] == 'passed_exact_GPU_semantics' and semantics['diagnostic_optimizer_updates'] == 4
         and semantics['historical_GPU_foreach_None_equivalence_claimed'] is False and semantics['tolerance_fallback'] is False
         and checks['same_layer'] == {'layer':9,'exact_equal':True,'teacher_detached':True,'student_restored':True,'optimizer_updates':0}
         and checks['semantic_direct_core']['updates'] == checks['semantic_A2_rho0']['updates'] == 2
         and checks['semantic_A2_rho0']['exact_trajectory_equal'] is True
         and checks['semantic_direct_core']['optimizer_foreach'] is False
         and checks['semantic_A2_rho0']['optimizer_foreach'] is False, 'G3 same-layer/rho0 exact GPU semantics are incomplete')
    for row in rows:_memory_record(row['memory'])
    for role in ('semantic_direct_core','semantic_A2_rho0'):_memory_record(checks[role]['memory'])
    roles = summary['roles']
    need(len(roles) == 3 and [item['role'] for item in roles] == list(ROLES), 'G3 summary roles are incomplete/reordered')
    initial = {item['trainable_sha256_before'] for item in roles}
    need(len(initial) == 1, 'G3 roles did not start from identical matched-Base student weights')
    for item in roles:
        need(item['returned_updates'] == item['optimizer_updates_applied'] == 22
             and item['EMA_updates'] == (22 if item['role']=='A2' else 0)
             and item['fresh_Base_SHA256'] == protocol['matched_Base']['checkpoint']['sha256']
             and item['frozen_identity_versions_unchanged'] is True and item['mode_and_gradient_cleanup_passed'] is True,
             'G3 role initialization/update/frozen contracts differ')
        _memory_record(item['memory'])
    need(summary['resident_VAE_unchanged_FP32'] is True and summary['quality_inference'] is False
         and summary['reusable_checkpoints_written'] is False, 'G3 was not the independent no-quality diagnostic')
    # This calls completed_t2, which recomputes the real1152-row G2 and checks
    # all3 completed Base8192/config/cache/asset/source/runtime bindings.
    current = g3.collect_protocol(SimpleNamespace(project_root=root,evaluation_run=Path(protocol['completed_T2']['directory'])))
    need(current == protocol, 'G3 shared numerical source/weights/cache/runtime no longer matches its sealed protocol')
    return protocol,{'directory':str(directory),'files':{name:binding(directory/name) for name in
        ('status.json','summary.json','protocol.json','semantics.json','observations.json','G3.json')},
        'passed_real_G2_and_G3':True,'benchmark_rows':66,'semantic_updates':4,'total_diagnostic_updates':70,
        'matched_Base_initial_trainable_sha256':next(iter(initial))}


def method_config(base,arm,base_origin=None):
    need(arm in ROLES,'Unknown pilot arm')
    value = copy.deepcopy(base)
    value.pop('_sha256',None);value.pop('_source_config_sha256',None)
    value.pop('formal_training',None);value.pop('stage2_contract',None)
    value['stage'] = 'stage4_pilot_noref_self_distillation'
    value['restoration_architecture'] = 'noref_base_v1'
    value['training'].update(variants=[arm],updates=UPDATES,checkpoint_at=list(CHECKPOINT_AT),evaluate_at=[])
    value['matched_Base_origin'] = copy.deepcopy(base_origin or {'training_seed':g3.METHOD['training_seed'],
        'step':8192,'source_config_sha256':base.get('_sha256')})
    value['self_distillation'] = copy.deepcopy(g3.METHOD) | {'arm':arm,'corruption_enabled':arm!='flow_only',
        'alignment_enabled':arm=='A2','effective_rho':0. if arm=='flow_only' else g3.METHOD['rho'],
        'effective_gamma':g3.METHOD['gamma'] if arm=='A2' else 0.}
    value['stage4_contract'] = {'training_clip_count':TRAIN_CLIPS,'validation_training_access':False,
        'updates_per_arm':UPDATES,'warmup_updates':0,'fresh_matched_Base_each_arm':True,'fresh_AdamW':True,
        'T5_load_allowed':False,'DINO_load_allowed':False,'RGB_decode_allowed':False,'quality_inference':False}
    value['_sha256'] = hashlib.sha256((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    need(value['model'] == base['model'] and value['conditioning'] == base['conditioning'], 'Method architecture drifted from Base')
    return value


DEGRADATION = {'scale':4,'kernel':'torch_bicubic','antialias':True,'align_corners':False,
               'clamp_01':True,'temporal_parameters':'shared_for_all_frames'}


def _metadata(meta,spec,contract):
    record = spec['record']
    need(meta['record_id'] == record['id'] and meta['source_id'] == record['source_id']
         and meta['source_sha256'] == record['sha256'] and meta['split'] == record['split'] == 'train'
         and Path(meta['source_path']).resolve() == Path(record['path']).resolve()
         and meta['caption'] == contract['caption'] and meta['data_version'] == 'hq-vsr-paired-v2'
         and meta['seed'] == contract['data_seed'], 'Latent metadata clip/source/split/caption identity differs')
    indices = meta['source_indices']
    need(len(indices)==33 and all(type(i) is int for i in indices) and indices[0]>=0
         and indices == list(range(indices[0],indices[0]+33)) and type(meta['temporal_stride']) is int
         and meta['temporal_stride']==1 and meta['valid_region']=='entire_crop_no_padding'
         and meta['degradation']==DEGRADATION, 'Latent metadata temporal/degradation/padding geometry differs')
    source = meta['source_hw']
    need(len(source)==2 and all(type(x) is int for x in source) and source[0]>=128 and source[1]>=224,
         'Invalid original HR geometry')
    ratio = max(128/source[0],224/source[1])
    resized = [max(128,round(source[0]*ratio)),max(224,round(source[1]*ratio))]
    need(meta['resize_hw']==resized and meta['crop_tlhw']==[(resized[0]-128)//2,(resized[1]-224)//2,128,224],
         'Latent metadata crop/resize differs from the original accepted preprocessing')
    fps = meta['original_fps']
    need(type(fps) in (int,float) and math.isfinite(fps) and fps>0 and meta['output_fps']==fps
         and meta['fps_policy']=='preserve_original_no_temporal_resampling'
         and meta['window_seed']==int(digest(['window',contract['data_seed'],record['id']])[:16],16), 'Latent FPS/window policy differs')
    b = meta['binding']
    need(b['manifest_sha256']==contract['manifest_sha256'] and b['official_revision']==contract['official_revision']
         and b['config_sha256']==contract['producer_config_sha256']
         and all(b['code_sha256'].get(k)==v for k,v in contract['encoder_hashes'].items())
         and b['data_cache_key']==digest({k:v for k,v in meta.items() if k!='binding'}), 'Latent metadata cache/source/encoder binding differs')
    need(spec.get('binding_sha256',digest(b)) == digest(b), 'Prepared clip binding changed')
    return b


def training_inputs(protocol):
    evaluation = protocol['evaluation_protocol'];producer = evaluation['cache_producer']
    source = Path(producer['source_run']).resolve()
    need(source != g3.ACTUAL_RUNS.resolve() and source.is_relative_to(g3.ACTUAL_RUNS.resolve()), 'Use only the completed actual Base cache producer')
    selection = producer['selection']
    ids = selection['training_ids']
    need(len(ids)==TRAIN_CLIPS and len(set(ids))==TRAIN_CLIPS and len(selection['validation_ids'])==32
         and not set(ids).intersection(selection['validation_ids']) and selection['test_accessed'] is False, 'Require fixed train1641/val32 isolation')
    cohort = read(evaluation['cohort_seal']['path'])
    manifest = read(cohort['manifest'])
    records = {item['id']:item for item in manifest['records']}
    need(set(ids)=={item['id'] for item in manifest['records'] if item['split']=='train'}, 'Training selection differs from original manifest split')
    artifact_map = read(producer['seal']['path'])['artifact_sha256']
    cfg = protocol['matched_Base']['config']
    contract = {'producer_run':str(source),'manifest_sha256':selection['manifest_sha256'],
        'manifest':binding(cohort['manifest'],cohort['manifest_file_sha256']),
        'official_revision':cfg['model']['revision'],'data_seed':cfg['seed'],'caption':evaluation['caption']['text'],
        'producer_config_sha256':cfg['_sha256'],'encoder_hashes':producer['encoder_hashes'],
        'training_ids':ids,'validation_ids_not_loaded':selection['validation_ids'],'max_cached':2}
    items = []
    for index,identifier in enumerate(ids):
        directory = source/f'clip_{index}'
        spec = {'index':index,'record':records[identifier],
            'metadata':binding(directory/'metadata.json',artifact_map[f'clip_{index}/metadata.json']),
            'latents':binding(directory/'fresh_latents.pt',artifact_map[f'clip_{index}/fresh_latents.pt'])}
        meta = read(directory/'metadata.json');b = _metadata(meta,spec,contract)
        spec['binding_sha256'] = digest(b)
        items.append(spec)
    return {'contract':contract,'clips':items,'latent_shape':g3.METHOD['latent_shape'],'tensor_payloads_loaded_by_prepare':False}


class LazyLatentReader:
    """Frozen CPU normalized latents only; strict per-access seals and LRU<=2."""
    def __init__(self,inputs,*,max_cached=2):
        need(type(max_cached) is int and 1<=max_cached<=2,'Latent LRU must retain at most2 clips')
        self.inputs,self.max_cached,self._cache = inputs,max_cached,OrderedDict()
        need(len(inputs['clips'])==TRAIN_CLIPS,'Require all1641 fixed training clips')
    def __len__(self):return len(self.inputs['clips'])
    @property
    def cached_indices(self):return tuple(self._cache)
    def clear(self):self._cache.clear()
    def __getitem__(self,index):
        need(type(index) is int and 0<=index<len(self),'Training latent index out of range')
        import torch
        spec = self.inputs['clips'][index]
        producer = Path(self.inputs['contract']['producer_run']).resolve()
        need(spec['index']==index and all(Path(spec[key]['path']).resolve()==producer/f'clip_{index}'/name
             for key,name in (('metadata','metadata.json'),('latents','fresh_latents.pt'))), 'Clip path/order escaped the fixed producer')
        need(binding(spec['metadata']['path'])==spec['metadata'] and binding(spec['latents']['path'])==spec['latents'],
             'Training metadata/latent SHA changed after prepare')
        meta = read(spec['metadata']['path']);b = _metadata(meta,spec,self.inputs['contract'])
        if index in self._cache:
            self._cache.move_to_end(index);payload = self._cache[index]
        else:
            payload = torch.load(spec['latents']['path'],map_location='cpu',weights_only=True)
            need(isinstance(payload,dict) and set(payload)=={'clean','low','binding'} and payload['binding']==b,
                 'Latent payload binding/key set differs from sealed clip metadata')
            self._cache[index] = payload
            while len(self._cache)>self.max_cached:self._cache.popitem(last=False)
        need(payload['binding']==b and all(isinstance(payload[key],torch.Tensor)
             and list(payload[key].shape)==g3.METHOD['latent_shape'] and payload[key].dtype==torch.float32
             and payload[key].device.type=='cpu' and not payload[key].requires_grad and bool(torch.isfinite(payload[key]).all())
             for key in ('clean','low')), 'Latents must be finite frozen CPU FP32 with the accepted33/9 geometry')
        return {'clean':payload['clean'],'low':payload['low']}


def source_bindings(root):
    root = Path(root).resolve()
    return {**g3.source_bindings(root),**{name:binding(root/name) for name in
        ('scripts/stage4_pilot.py','tests/test_stage4_pilot.py')}}


def collect_protocol(args):
    root = Path(args.project_root).resolve()
    need(Path.cwd().resolve()==root and Path(__file__).resolve()==root/'scripts/stage4_pilot.py', 'Use the pilot staging source as CWD')
    need(root!=g3.ACTUAL_RUNS.parent and not root.is_relative_to(g3.t2.OLD)
         and not root.is_relative_to(Path('/home/hugo/Projects/ReprVSR-pretrained')), 'Historical/actual assets are read-only; use staging source')
    previous,evidence = completed_g3(args.g3_run,root)
    base = previous['matched_Base']
    origin = {'checkpoint':base['checkpoint'],'training_seed':g3.METHOD['training_seed'],'step':8192,
              'source_config_sha256':base['config']['_sha256']}
    criteria = binding(args.g4_criteria) if getattr(args,'g4_criteria',None) is not None else None
    if criteria is not None:need(isinstance(read(criteria['path']),dict),'G4 criteria, if provided, must be a JSON object; this worker does not approve or evaluate it')
    result = {'schema':'stage4_pilot_training_v1','project_root':str(root),'worker':binding(__file__),
        'allowed_pilot_checkpoint_root':str(root/'runs'),'completed_T2':previous['completed_T2'],'completed_G3':evidence,
        'g3_protocol':previous,'matched_Base':base,'method_configs':{arm:method_config(base['config'],arm,origin) for arm in ROLES},
        'training_inputs':training_inputs(previous),'source':source_bindings(root),'runtime':previous['runtime'],
        'roles':list(ROLES),'budget':BUDGET,'checkpoint_at':list(CHECKPOINT_AT),
        'optimizer':previous['optimizer'],'resident_VAE':previous['resident_VAE'],
        'training_seeds':[g3.METHOD['training_seed']],'comparison_axis':'method arm, not training seed',
        'timing':{'step':'H2D normalized latent pair + unchanged complete Step.step + synchronize; includes scheduled owned CPU snapshot copy',
            'outside_step':'CPU latent loading/hashes, RNG hashes, feature-statistics/SVD, JSON/IO, checkpoints, role loads/cleanup',
            'wall_clock_includes_all_phases':True},
        'quality_evaluation':{'performed':False,'G4_evaluated':False,'criteria':criteria,
            'criteria_approval_validated_by_this_training_worker':False,'quality_evaluation_blocked':criteria is None,
            'next_stage':'Independent T4.3 evaluator must require approved predeclared nonplaceholder criteria before scoring'},
        'output':{'fresh_only':True,'resume':False,'automatic_retry':False,'quality_inference':False,'reuse_diagnostic_weights':False}}
    return json.loads(json.dumps(result,ensure_ascii=False,allow_nan=False))


def safe_run(args):
    run,root = Path(args.run_dir).resolve(),Path(args.project_root).resolve()
    need(run!=root/'runs' and run.is_relative_to(root/'runs')
         and not run.is_relative_to(Path(args.g3_run).resolve())
         and not run.is_relative_to(g3.ACTUAL_RUNS.resolve())
         and not run.is_relative_to(g3.t2.OLD.resolve()), 'Pilot outputs must be a fresh independent staging ROOT/runs child')
    return run


def prepare(args):
    run = safe_run(args)
    need(not run.exists(),'Fresh-only pilot output; existing directories are forbidden')
    protocol = collect_protocol(args)
    run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    for arm in ROLES:save(run/'arm_runs'/arm/'config.json',protocol['method_configs'][arm])
    save(run/'status.json',{'status':'planned','protocol_sha256':file_sha(run/'protocol.json'),
        'scientific_training_updates':0,'pilot_optimizer_updates_applied':0,'completed_arms':[],
        'G4_evaluated':False,'quality_evaluation_blocked':protocol['quality_evaluation']['quality_evaluation_blocked']})
    return {'status':'planned','run':str(run),'protocol_sha256':file_sha(run/'protocol.json'),'budget':BUDGET}


class PilotCounter(g3.UpdateCounter):
    def snapshot(self):
        value = super().snapshot()
        attempted = value.pop('diagnostic_optimizer_updates_attempted')
        applied = value.pop('diagnostic_optimizer_updates_applied')
        value.update(pilot_optimizer_updates_attempted=attempted,pilot_optimizer_updates_applied=applied,
                     scientific_training_updates=applied)
        return value


def _row_contract(row,arm,ordinal,protocol,reference_FM,reference_corruption):
    inputs = protocol['training_inputs']['clips'];index = ordinal%TRAIN_CLIPS
    need(row['step']==ordinal+1 and row['arm']==arm and row['data_index']==index
         and row['matched_FM']['data_index']==index
         and row['matched_FM']['record_id']==inputs[index]['record']['id']
         and row['matched_FM']['latent_file_sha256']==inputs[index]['latents']['sha256'], 'Pilot step/data order differs')
    need(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('fm','sd','total','sigma','lr','seconds'))
         and row['seconds']>0 and 0<=row['sigma']<=1
         and row['teacher_forward']==(arm=='A2' and row['sigma']>g3.SIGMA_BOUNDARY)
         and (row['cosine_mean'] is None if not row['teacher_forward'] else
              type(row['cosine_mean']) in (int,float) and math.isfinite(row['cosine_mean'])), 'Nonfinite/mismatched pilot objective/window log')
    need(row['sigma']==row['matched_FM']['sigma'] and (row['sd']==0 if not row['teacher_forward'] else True), 'Pilot flow/sigma/disabled-SD mismatch')
    fm = digest(row['matched_FM'])
    if arm=='flow_only':reference_FM.append(fm)
    else:need(reference_FM[ordinal]==fm,'Pilot FM data/noise/sigma stream differs across arms')
    corruption = digest({'rng':row['matched_corruption_RNG'],'mask_spans':row['mask_spans']})
    if arm=='C1':reference_corruption.append(corruption)
    elif arm=='A2':need(reference_corruption[ordinal]==corruption,'C1/A2 corruption RNG or mask differs')


def _feature_report(row,target):
    report = row['feature_diagnostics']
    need(type(row.get('feature_diagnostics_snapshot_collected')) is bool
         and row['feature_diagnostics_snapshot_collected'] == (report is not None), 'Collected feature snapshots require their own persisted report')
    if report is not None:
        expected = Path(target).resolve()/'feature_diagnostics'/f"step_{row['step']:06d}.json"
        need(Path(report['path']).resolve()==expected and binding(expected)==report,
             'Feature diagnostic report escaped its arm/step path or changed')


def _load_text(torch,protocol):
    cap = protocol['g3_protocol']['evaluation_protocol']['caption']
    need(binding(cap['cache']['path'])==cap['cache'],'Cached text file changed')
    cache = torch.load(cap['cache']['path'],map_location='cpu',weights_only=True)
    text = cache.get(cap['text']) if isinstance(cache,dict) else None
    need(isinstance(text,torch.Tensor) and list(text.shape)==[1,512,4096]==cap['shape']
         and str(text.dtype)==cap['dtype']=='torch.bfloat16' and not text.requires_grad and bool(torch.isfinite(text).all()),
         'Exact finite BF16 cached text required; no T5 fallback')
    header = json.dumps([str(text.dtype),list(text.shape)],separators=(',',':')).encode()+b'\n'
    need(hashlib.sha256(header+text.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==cap['embedding_sha256'],'Cached text tensor digest differs')
    return text.detach().to('cuda')


def _run_arm(torch,protocol,arm,run,reader,text,scheduler,counter,reference_FM,reference_corruption,phase):
    from vsr.self_distillation import _require_finite
    from vsr.method_checkpoint import save_method_checkpoint
    from vsr.reproducibility import tensor_state_hashes
    arm_run = run/'arm_runs'/arm;target = arm_run/arm
    need(not target.exists(),'Pilot arm output must never have started')
    target.mkdir()
    cfg = protocol['method_configs'][arm]
    need(read(arm_run/'config.json')==cfg,'Prepared per-arm canonical config changed')
    engine = pair = clean = low = None
    reference = None;flags = [];receipts = [];objectives = [];step_seconds = 0.;stats_seconds = 0.;checkpoint_seconds = 0.
    started = time.perf_counter();g3_protocol = protocol['g3_protocol']
    torch.cuda.reset_peak_memory_stats()
    try:
        phase.update(arm=arm,step=0,phase='fresh_matched_Base_and_fresh_AdamW')
        engine = g3._new_engine(g3_protocol,arm)
        reference = weakref.ref(engine.model)
        options = protocol['optimizer']
        need(engine.steps==0 and not engine.optimizer.state
             and engine.max_grad_norm==options['max_grad_norm']
             and all(group['foreach'] is False and group['lr']==options['learning_rate']
                     and group['weight_decay']==options['weight_decay'] for group in engine.optimizer.param_groups),
             'Pilot optimizer must start fresh with the exact matched Base lr/wd/maxnorm and foreach=False')
        counter.attach(engine.optimizer,arm)
        initial = tensor_state_hashes(engine.model)['all_trainable']['sha256']
        need(initial==protocol['completed_G3']['matched_Base_initial_trainable_sha256']
             and engine.base_load_audit['checkpoint_sha256']==protocol['matched_Base']['checkpoint']['sha256'],
             'Pilot must freshly reload matched Base student; diagnostic/previous-arm parameters are forbidden')
        frozen = g3._frozen_signature(engine.model)
        flags = [(module,module.training) for module in engine.model.modules()]
        _require_finite(('Nonfinite loaded pilot parameter: '+name,p) for name,p in engine.model.named_parameters())
        save(arm_run/'initialization.json',{'matched_Base':engine.base_load_audit,'initial_trainable_sha256':initial,
            'fresh_AdamW':True,'foreach':False,'diagnostic_or_previous_arm_weights_loaded':False})
        save(arm_run/'status.json',{'status':'running','variant':arm,'updates':0,'scientific_training_updates':0,
            'protocol_sha256':file_sha(run/'protocol.json')})
        with (target/'training.jsonl').open('x') as log:
            for ordinal in range(UPDATES):
                index = engine.data_index(TRAIN_CLIPS)
                need(index==ordinal%TRAIN_CLIPS,'Pilot fixed training order changed')
                phase.update(arm=arm,step=ordinal+1,phase='optimizer_step')
                pair = reader[index]
                item = protocol['training_inputs']['clips'][index]
                before_FM = g3._streams(torch,engine)
                before_aux = g3._streams(torch,engine,('mask','corruption'))
                torch.cuda.synchronize();tick = time.perf_counter()
                clean,low = (pair[key].to('cuda') for key in ('clean','low'))
                result = engine.step(clean,low,text,scheduler)
                torch.cuda.synchronize();elapsed = time.perf_counter()-tick
                counter.by_role[arm]['returned_steps'] += 1
                need(engine.steps==ordinal+1 and result['step']==ordinal+1,'Pilot returned step differs from its fixed budget')
                phase.update(phase='CPU_feature_statistics_and_logging')
                tick = time.perf_counter();features = engine.drain_feature_diagnostics()
                stats_seconds += time.perf_counter()-tick
                feature_file = None
                if features is not None:
                    feature_path = target/'feature_diagnostics'/f'step_{ordinal+1:06d}.json'
                    save(feature_path,features);feature_file = binding(feature_path)
                after_FM = g3._streams(torch,engine)
                after_aux = g3._streams(torch,engine,('mask','corruption'))
                need(frozen==g3._frozen_signature(engine.model)
                     and all(module.training==flag for module,flag in flags)
                     and all(p.grad is None for p in engine.parameters)
                     and engine.model._condition_tokens is None and engine.model._captured is None,
                     'Pilot frozen/mode/gradient/feature cleanup contract changed')
                row = {**result,'role':arm,'data_index':index,'seconds':elapsed,
                    'lr':engine.optimizer.param_groups[0]['lr'],'memory':g3._memory(torch),
                    'matched_FM':{'data_index':index,'record_id':item['record']['id'],'latent_file_sha256':item['latents']['sha256'],
                        'before_rng':before_FM,'after_rng':after_FM,
                        **{key:result[key] for key in ('timestep_index','native_timestep','sigma')}},
                    'matched_corruption_RNG':{'before':before_aux,'after':after_aux},
                    'feature_diagnostics':feature_file,'feature_statistics_timing':'CPU std/SVD outside Step; detached copy inside Step'}
                _row_contract(row,arm,ordinal,protocol,reference_FM,reference_corruption)
                _feature_report(row,target)
                need(row['lr']==options['learning_rate'],'Pilot learning rate must remain constant')
                log.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
                log.flush()
                objectives.append({key:row[key] for key in ('fm','sd','total')});step_seconds += elapsed
                save(run/'progress.json',{'phase':dict(phase),'completed_arms':[name for name in ROLES if name in counter.by_role and counter.by_role[name]['returned_steps']==UPDATES],
                    **counter.snapshot()},replace=True)
                if ordinal+1 in CHECKPOINT_AT:
                    phase.update(phase='new_student_checkpoint')
                    tick = time.perf_counter()
                    receipts.append(save_method_checkpoint(target/f'checkpoint_{ordinal+1:06d}.pt',engine,cfg,
                        allowed_root=Path(protocol['allowed_pilot_checkpoint_root']),
                        initialization={'matched_Base':protocol['matched_Base']['checkpoint'],'initial_trainable_sha256':initial}))
                    checkpoint_seconds += time.perf_counter()-tick
                    save(arm_run/'checkpoint_receipts.json',{'checkpoints':receipts,'protocol_sha256':file_sha(run/'protocol.json')},replace=True)
            os.fsync(log.fileno())
        _require_finite(('Nonfinite final pilot parameter: '+name,p) for name,p in engine.model.named_parameters())
        if engine.ema is not None:engine.ema._validate(engine.model)
        need(engine.steps==UPDATES and counter.by_role[arm]=={'attempted':UPDATES,'applied':UPDATES,'returned_steps':UPDATES}
             and [item['step'] for item in receipts]==list(CHECKPOINT_AT), 'Pilot arm update/checkpoint matrix is incomplete')
        summary = {'status':'completed','arm':arm,'variant':arm,'run':str(arm_run),'training_seed':g3.METHOD['training_seed'],
            'updates':UPDATES,'scientific_training_updates':UPDATES,'optimizer_counts':copy.deepcopy(counter.by_role[arm]),
            'config_sha256':cfg['_sha256'],'final_checkpoint':receipts[-1],'checkpoints':receipts,
            'training_jsonl':binding(target/'training.jsonl'),'training_jsonl_sha256':file_sha(target/'training.jsonl'),
            'initial_trainable_sha256':initial,'final_trainable_sha256':tensor_state_hashes(engine.model)['all_trainable']['sha256'],
            'matched_Base':protocol['matched_Base']['checkpoint'],
            'T2_protocol_sha256':protocol['completed_T2']['files']['protocol.json']['sha256'],
            'G3_protocol_sha256':protocol['completed_G3']['files']['protocol.json']['sha256'],
            'step_seconds':step_seconds,'CPU_feature_statistics_seconds':stats_seconds,'checkpoint_seconds':checkpoint_seconds,
            'arm_wall_seconds_before_cleanup':time.perf_counter()-started,'memory':g3._memory(torch),
            'objectives':{key:{'mean':statistics.mean(x[key] for x in objectives),'sample_SD':statistics.stdev(x[key] for x in objectives),
                               'final':objectives[-1][key]} for key in ('fm','sd','total')},
            'EMA_updates':engine.ema.num_updates if engine.ema else 0,'foreach':False,
            'frozen_identity_versions_unchanged':True,'mode_gradient_feature_cleanup_passed':True,'quality_inference':False,'G4_evaluated':False}
        save(arm_run/'summary.json',summary)
        save(arm_run/'status.json',{'status':'completed','variant':arm,'updates':UPDATES,'scientific_training_updates':UPDATES,
            'summary_sha256':file_sha(arm_run/'summary.json'),'final_checkpoint':receipts[-1],
            'protocol_sha256':file_sha(run/'protocol.json')},replace=True)
        return summary
    finally:
        successful = sys.exc_info()[0] is None
        if successful:phase.update(phase='arm_GPU_cleanup')
        g3._cleanup_engine(torch,engine,successful=successful,counter=counter)
        flags = [];engine = pair = clean = low = None
        reader.clear()
        g3._check_release(torch,reference,successful,counter,arm)


def _save_failure(run,error,details,phase,protocol_sha,counter,arms,started):
    failure = {'status':'failed_stop_no_retry','error':f'{type(error).__name__}: {error}','traceback':details,
        'phase':dict(phase),'protocol_sha256':protocol_sha,'completed_arms':list(arms),**counter.snapshot(),
        'wall_seconds':time.perf_counter()-started,'automatic_retry':False,'quality_inference':False,'G4_evaluated':False}
    save(run/'failure.json',failure)
    state = {'status':'failed_stop_no_retry','protocol_sha256':protocol_sha,'failure_sha256':file_sha(run/'failure.json'),
             'completed_arms':list(arms),'G4_evaluated':False,**counter.snapshot()}
    if (run/'summary.json').exists():
        previous = read(run/'summary.json')
        previous.update(status='failed_stop_no_retry',completed_training_before_execution_failure=previous['status'],
            failure_sha256=state['failure_sha256'],next_action='stop_report_no_fallback_retry')
        save(run/'summary.json',previous,replace=True);state['summary_sha256'] = file_sha(run/'summary.json')
    arm = phase.get('arm')
    if arm in ROLES and arm not in arms and (run/'arm_runs'/arm/'status.json').exists():
        arm_state = read(run/'arm_runs'/arm/'status.json')
        arm_state.update(status='failed_stop_no_retry',failure_sha256=state['failure_sha256'],
            updates=counter.by_role.get(arm,{}).get('returned_steps',0),
            scientific_training_updates=counter.by_role.get(arm,{}).get('applied',0))
        save(run/'arm_runs'/arm/'status.json',arm_state,replace=True)
    save(run/'status.json',state,replace=True)


def execute(args):
    run = safe_run(args)
    need(args.expected_protocol_sha and file_sha(run/'protocol.json')==args.expected_protocol_sha,'Exact prepared pilot protocol SHA required')
    initial_status = read(run/'status.json')
    need(initial_status['status']=='planned' and initial_status['protocol_sha256']==args.expected_protocol_sha,
         'Only a never-started pilot can execute; no resume/retry')
    with (run/'execute.lock').open('x') as stream:stream.write(str(os.getpid())+'\n')
    protocol = read(run/'protocol.json')
    save(run/'status.json',{'status':'running','protocol_sha256':args.expected_protocol_sha,'completed_arms':[],
        'scientific_training_updates':0,'pilot_optimizer_updates_applied':0,'G4_evaluated':False},replace=True)
    torch = vae = text = scheduler = reader = summary = None
    counter = PilotCounter();arms = {};phase = {'phase':'CPU_prerequisites_and_GPU_lease'};started = time.perf_counter()
    def failure(error,details):_save_failure(run,error,details,phase,args.expected_protocol_sha,counter,arms,started)
    def secondary(primary,error,details):
        save(run/'cleanup_failure.json',{'primary_error_preserved':f'{type(primary).__name__}: {primary}',
            'primary_failure_sha256':file_sha(run/'failure.json'),'secondary_cleanup_error':f'{type(error).__name__}: {error}',
            'secondary_traceback':details,'G4_evaluated':False,'automatic_retry':False})
        state=read(run/'status.json');state['cleanup_failure_sha256']=file_sha(run/'cleanup_failure.json')
        save(run/'status.json',state,replace=True)
    def cleanup():
        nonlocal vae,text,scheduler,reader
        phase.update(phase='final_GPU_cleanup')
        if reader is not None:reader.clear()
        vae = text = scheduler = reader = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():torch.cuda.empty_cache()
        if summary is not None and not (run/'failure.json').exists():
            summary.update(wall_seconds=time.perf_counter()-started,final_GPU_cleanup_completed=True)
            save(run/'summary.json',summary,replace=True)
            state=read(run/'status.json');state['summary_sha256']=file_sha(run/'summary.json')
            save(run/'status.json',state,replace=True)
    with g3._leased_lifecycle(g3.ACTUAL_RUNS,cleanup,failure,secondary):
        need(protocol==collect_protocol(args),'Prepared pilot source/runtime/prerequisite/input files changed')
        need(all(os.environ.get(key)==value for key,value in g3.t2.ENVIRONMENT.items()),'Exact strict/offline launch environment required')
        evaluation = protocol['g3_protocol']['evaluation_protocol']
        classifier = g3.t2.import_helper(evaluation['helpers']['gpu_classifier']['path'],'stage4_GPU_guard')
        guard = g3.t2.gpu_idle_guard(evaluation['desktop_binding'],classifier)
        save(run/'GPU_idle_before_load.json',guard)
        need(guard['passed'],'GPU busy/unverified; pilot must never overlap another training/evaluation worker')
        sys.path[:0] = [protocol['project_root'],evaluation['dependencies']['root']]
        import torch as torch_module
        torch = torch_module
        need(not torch.cuda.is_initialized(),'Strict configuration must precede CUDA initialization')
        torch.set_num_threads(2)
        from vsr.reproducibility import configure_reproducibility
        from vsr.checkpoints import _check_config
        runtime = configure_reproducibility(protocol['matched_Base']['config']);flags = runtime['flags']
        need(flags['deterministic_algorithms'] and not flags['deterministic_warn_only'] and flags['sdp_math']
             and not any(flags[key] for key in ('sdp_flash','sdp_memory_efficient','sdp_cudnn')),'Strict native math backend required')
        for cfg in protocol['method_configs'].values():_check_config(cfg)
        save(run/'runtime.json',runtime)
        g3.t2.verify_imported_dependencies(evaluation['dependencies'])
        from diffusers import AutoencoderKLWan,WanTransformer3DModel,UniPCMultistepScheduler
        from vsr.infer_video import load_frozen_vae
        from vsr.self_distillation import _require_finite
        g3.t2.verify_native_sources(evaluation['native_numerical_sources'],{'AutoencoderKLWan':AutoencoderKLWan,
            'WanTransformer3DModel':WanTransformer3DModel,'UniPCMultistepScheduler':UniPCMultistepScheduler})
        for name,module in list(sys.modules.items()):
            if name=='vsr' or name.startswith('vsr.'):
                need(Path(module.__file__).resolve().is_relative_to(Path(protocol['project_root'])/'vsr'),'Imported vsr module shadows sealed staging')
        need(not torch.cuda.is_initialized(),'CPU imports initialized CUDA before validated model loading')
        phase.update(phase='resident_FP32_VAE_and_text')
        vae = load_frozen_vae(protocol['matched_Base']['config'],device='cuda')
        need(not vae.training and all(not p.requires_grad and p.dtype==torch.float32 and p.device.type=='cuda' for p in vae.parameters()),
             'One frozen eval FP32 VAE must remain resident across all pilot arms')
        _require_finite(('Nonfinite resident VAE: '+name,p) for name,p in vae.named_parameters())
        vae_signature = g3._state_signature(vae)
        reader = LazyLatentReader(protocol['training_inputs'],max_cached=2);text = _load_text(torch,protocol)
        scheduler = UniPCMultistepScheduler.from_pretrained(protocol['matched_Base']['config']['model']['local_dir'],
            subfolder='scheduler',local_files_only=True)
        scheduler.set_timesteps(1000,device='cuda')
        reference_FM,reference_corruption = [],[]
        for arm in ROLES:
            arms[arm] = _run_arm(torch,protocol,arm,run,reader,text,scheduler,counter,reference_FM,reference_corruption,phase)
            need(vae_signature==g3._state_signature(vae),'Resident FP32 VAE changed during pilot training')
            save(run/'completed_arms.json',{'completed_arms':list(arms),'arms':arms,**counter.snapshot()},replace=True)
        need(counter.snapshot()['scientific_training_updates']==6144
             and counter.snapshot()['completed_step_returns']==6144 and not counter.cleanup_events,
             'Pilot6144-step budget or cleanup contract differs')
        need(source_bindings(protocol['project_root'])==protocol['source'],'Sealed source changed during pilot training')
        summary = {'status':'completed_pilot_training','protocol_sha256':args.expected_protocol_sha,'completed_arms':list(arms),
            'arms':arms,'budget':BUDGET,**counter.snapshot(),'matched_Base':protocol['matched_Base']['checkpoint'],
            'completed_T2':protocol['completed_T2'],'completed_G3':protocol['completed_G3'],
            'training_seeds':[g3.METHOD['training_seed']],'comparison_axis':'method arm, not training seed',
            'wall_seconds_before_final_cleanup':time.perf_counter()-started,'resident_VAE_unchanged_FP32':True,
            'quality_inference':False,'G4_evaluated':False,'quality_evaluation_blocked':protocol['quality_evaluation']['quality_evaluation_blocked'],
            'next_action':'Independent T4.3 with approved predeclared criteria; no quality/G4 conclusion from training losses'}
        save(run/'summary.json',summary)
        save(run/'status.json',{'status':'completed_pilot_training','protocol_sha256':args.expected_protocol_sha,
            'summary_sha256':file_sha(run/'summary.json'),'completed_arms':list(arms),'G4_evaluated':False,**counter.snapshot()},replace=True)
        return summary


def completed_pilot(directory,root):
    """Metadata-only consumer contract; no checkpoint tensor or pixels loaded."""
    directory,root = Path(directory).resolve(),Path(root).resolve()
    need(not (directory/'failure.json').exists() and not (directory/'cleanup_failure.json').exists(),'Failed pilot cannot be evaluated')
    status,summary,protocol = (read(directory/name) for name in ('status.json','summary.json','protocol.json'))
    psha = file_sha(directory/'protocol.json')
    need(protocol['schema']=='stage4_pilot_training_v1' and protocol['project_root']==str(root)
         and protocol['allowed_pilot_checkpoint_root']==str(root/'runs')
         and status['status']==summary['status']=='completed_pilot_training'
         and status['summary_sha256']==file_sha(directory/'summary.json')
         and status['protocol_sha256']==summary['protocol_sha256']==psha,'Pilot completion/protocol SHA chain differs')
    criteria = protocol['quality_evaluation']['criteria']
    args = SimpleNamespace(project_root=root,g3_run=Path(protocol['completed_G3']['directory']),
                           g4_criteria=Path(criteria['path']) if criteria else None)
    need(collect_protocol(args)==protocol,'Pilot shared source/config/runtime/G2/G3/cache bindings changed')
    for item in (status,summary):
        need(item['completed_arms']==list(ROLES) and item['scientific_training_updates']==6144
             and item['pilot_optimizer_updates_applied']==item['pilot_optimizer_updates_attempted']==item['completed_step_returns']==6144
             and item['optimizer_calls_not_returned_partial_effect_unknown']==0 and item['cleanup_events']==[]
             and item['G4_evaluated'] is False and set(item['by_role'])==set(ROLES), 'Pilot must contain3 completed2048-step arms, not3 seeds')
        need(all(item['by_role'][arm]=={'attempted':UPDATES,'applied':UPDATES,'returned_steps':UPDATES} for arm in ROLES), 'Pilot per-arm count chain differs')
    need(summary['budget']==BUDGET and summary['training_seeds']==[g3.METHOD['training_seed']]
         and summary['comparison_axis']=='method arm, not training seed'
         and summary['quality_inference'] is False and summary['resident_VAE_unchanged_FP32'] is True,'Pilot scope/evidence differs')
    expected_initial = protocol['completed_G3']['matched_Base_initial_trainable_sha256']
    reference_FM,reference_corruption = [],[];receipts = {}
    for arm in ROLES:
        arm_run = directory/'arm_runs'/arm
        state,report,cfg = (read(arm_run/name) for name in ('status.json','summary.json','config.json'))
        need(state['status']==report['status']=='completed' and state['variant']==report['variant']==report['arm']==arm
             and state['updates']==report['updates']==state['scientific_training_updates']==report['scientific_training_updates']==UPDATES
             and state['summary_sha256']==file_sha(arm_run/'summary.json') and state['protocol_sha256']==psha
             and cfg==protocol['method_configs'][arm] and report==summary['arms'][arm]
             and report['run']==str(arm_run) and report['config_sha256']==cfg['_sha256'], 'Pilot arm config/completion chain differs')
        need(report['initial_trainable_sha256']==expected_initial and report['matched_Base']==protocol['matched_Base']['checkpoint']
             and report['T2_protocol_sha256']==protocol['completed_T2']['files']['protocol.json']['sha256']
             and report['G3_protocol_sha256']==protocol['completed_G3']['files']['protocol.json']['sha256']
             and report['EMA_updates']==(UPDATES if arm=='A2' else 0) and report['foreach'] is False
             and report['frozen_identity_versions_unchanged'] is True and report['mode_gradient_feature_cleanup_passed'] is True,
             'Pilot arm did not preserve its fresh matched-Base/frozen/optimizer origin')
        _memory_record(report['memory'])
        cps = report['checkpoints']
        need(len(cps)==4 and [cp['step'] for cp in cps]==list(CHECKPOINT_AT)
             and state['final_checkpoint']==report['final_checkpoint']==cps[-1], 'Pilot checkpoint schedule or final receipt differs')
        for cp in cps:
            expected = arm_run/arm/f"checkpoint_{cp['step']:06d}.pt"
            need(Path(cp['path']).resolve()==expected and cp['variant']==arm
                 and cp['restoration_architecture']=='noref_base_v1' and cp['inference_parameters']=='student'
                 and cp['config_sha256']==cfg['_sha256'] and cp['existing_artifacts_overwritten'] is False
                 and binding(expected)['sha256']==cp['sha256'], 'Pilot student checkpoint receipt changed')
        log = arm_run/arm/'training.jsonl'
        need(binding(log)==report['training_jsonl'] and file_sha(log)==report['training_jsonl_sha256'],'Pilot training log changed')
        with log.open() as stream:
            count = 0
            for ordinal,line in enumerate(stream):
                row = json.loads(line);_row_contract(row,arm,ordinal,protocol,reference_FM,reference_corruption)
                _memory_record(row['memory'])
                need(row['lr']==protocol['optimizer']['learning_rate'],'Pilot learning-rate log differs')
                _feature_report(row,arm_run/arm)
                count += 1
        need(count==UPDATES,'Pilot log must contain exactly2048 returned optimizer steps')
        receipts[arm] = {'run':str(arm_run),'config':binding(arm_run/'config.json'),'status':binding(arm_run/'status.json'),
            'summary':binding(arm_run/'summary.json'),'final_checkpoint':cps[-1],'training_jsonl':binding(log)}
    return protocol,{'directory':str(directory),'files':{name:binding(directory/name) for name in ('status.json','summary.json','protocol.json')},
        'arms':receipts,'scientific_training_updates':6144,'training_seeds':[g3.METHOD['training_seed']],
        'comparison_axis':'method arm, not training seed','quality_inference':False,'G4_evaluated':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--g3-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--g4-criteria',type=Path)
    parser.add_argument('--expected-protocol-sha')
    args = parser.parse_args()
    if args.mode=='prepare':
        need(args.expected_protocol_sha is None,'prepare creates its own protocol SHA')
        result = prepare(args)
    else:result = execute(args)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
