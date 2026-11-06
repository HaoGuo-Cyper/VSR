#!/usr/bin/env python3
"""Future G3: metadata-only prepare, then an exclusive, sealed GPU diagnostic.

Requires completed three-Base/1152-observation T2 and recomputed passed G2.
Runs four semantic AdamW updates separately from 66 benchmark updates. Every
role reloads the same final seed20261102 Base and starts a fresh AdamW. Nothing
is saved as a reusable task checkpoint or used for quality inference. Prepare
never imports torch. Developing this file is not a GPU authorization or G3 pass.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, ExitStack
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import statistics
import struct
import sys
import time
import traceback
import weakref

sys.dont_write_bytecode = True
import stage2_evaluate_bases as t2

need, read, save, binding, file_sha = t2.need, t2.read, t2.save, t2.binding, t2.file_sha
ACTUAL_RUNS = Path('/home/hugo/Projects/VSR_/runs')
ROLES = ('flow_only', 'C1', 'A2')
WARMUP, TIMED, SEMANTIC_STEPS = 2, 20, 2
MEMORY_CAP = 16 * 1024**3
SIGMA_BOUNDARY = struct.unpack('f',struct.pack('f',.3))[0]
METHOD = {'training_seed':20261102, 'frames':33, 'temporal_stride':1, 'latent_frames':9,
          'latent_shape':[1,16,9,16,28], 'rho':.25, 'span':2, 'exclude_latent0':True,
          'gamma':.8, 'ema_decay':.999, 'student_layer':9, 'teacher_layer':21,
          'hidden_dim':1024, 'teacher_window':'native scheduler sigma > 0.3',
          'teacher_same_noised_HR_and_sigma':True, 'optimizer_foreach':False,
          'feature_diagnostics_first':True,'feature_diagnostics_every':512}
BUDGET = {'same_layer_forward_checks':2, 'semantic_reference_updates':2,
          'semantic_A2_rho0_updates':2, 'benchmark_warmup_updates':6,
          'benchmark_timed_updates':60, 'benchmark_updates':66,
          'total_diagnostic_updates':70, 'scientific_training_updates':0}


def evaluation_args(protocol):
    """Reconstruct the exact completed T2 prepare contract, without imports."""
    return SimpleNamespace(project_root=Path(protocol['project_root']),
        seal=Path(protocol['cohort_seal']['path']), dataset=Path(protocol['dataset_manifest']['path']),
        checkpoint=[f"{s}={protocol['checkpoints'][str(s)]['checkpoint']['path']}" for s in t2.SEEDS],
        allowed_checkpoint_root=Path(protocol['allowed_checkpoint_root']),
        official_manifest=Path(protocol['official_model_manifest']['path']),
        metric_root=Path(protocol['metric_weights'][0]['path']).parent,
        dependency_root=Path(protocol['dependencies']['root']),
        desktop_pid=protocol['desktop_binding']['PID'], desktop_name=protocol['desktop_binding']['reported_name'])


def completed_t2(directory, root):
    """Validate raw seals, recompute G2 and revalidate all three formal Bases."""
    directory, root = Path(directory).resolve(), Path(root).resolve()
    status, summary, protocol, gate, observations = (read(directory/name) for name in
        ('status.json','summary.json','protocol.json','G2.json','observations.json'))
    psha = file_sha(directory/'protocol.json')
    need(status['status'] == summary['status'] == 'completed_full_matrix'
         and status['G2_status'] == summary['G2_status'] == gate['status'] == 'passed_G2',
         'T2 must be complete and G2 passed; partial/failed evaluations cannot authorize G3')
    need(status['summary_sha256'] == file_sha(directory/'summary.json')
         and summary['G2_sha256'] == file_sha(directory/'G2.json')
         and summary['observations_sha256'] == file_sha(directory/'observations.json'), 'T2 completion seal mismatch')
    need(all(x['protocol_sha256'] == psha for x in (status,summary,observations))
         and all(x['training_updates'] == 0 for x in (status,summary,observations)), 'T2 protocol/update declaration mismatch')
    need(protocol['schema'] == 'stage2_complete_Base_val32_evaluation_v1'
         and protocol['project_root'] == str(root) and protocol['training_updates'] == 0
         and Path(protocol['allowed_checkpoint_root']).resolve() == ACTUAL_RUNS.resolve(), 'Wrong T2 root/schema/actual checkpoint tree')
    need(summary['observations'] == len(observations['observations']) == 1152
         and summary['prediction_PNGs'] == 38016 and summary['test_or_external_pixels_used'] is False,
         'Require all1152 restores and heldout isolation')
    roles = summary['roles']
    need(len(roles) == 3 and sorted(x['training_seed'] for x in roles) == t2.SEEDS
         and all(x['trainable_sha256_before'] == x['trainable_sha256_after'] for x in roles),
         'Three completed T2 checkpoint roles must preserve their trainables')
    # Reuses all strict Base8192/init/config/cache/package/native-code/asset
    # validation. This hashes bytes but never loads tensors, pixels or torch.
    need(t2.collect_protocol(evaluation_args(protocol)) == protocol,
         'Completed T2 source/input/config/runtime no longer matches its seal')
    seal = read(protocol['cohort_seal']['path'])
    recalculated = json.loads(json.dumps(t2.summarize_g2(observations['observations'],seal),allow_nan=False))
    need(recalculated == gate and recalculated['status'] == 'passed_G2', 'G2 does not reproduce from all1152 observations')
    declarations = {str(seed):value for seed,value in gate['checkpoint_digest_declarations'].items()}
    need(declarations == {key:value['checkpoint']['sha256'] for key,value in protocol['checkpoints'].items()},
         'G2 checkpoint declarations differ from the three completed Bases')
    return protocol, {'directory':str(directory),'files':{name:binding(directory/name) for name in
        ('status.json','summary.json','protocol.json','G2.json','observations.json')},
        'recomputed_G2':True,'observations':1152,'all3_completed_Base8192':True}


def latent_inputs(evaluation):
    producer = Path(evaluation['cache_producer']['source_run']).resolve()
    need(producer.is_relative_to(ACTUAL_RUNS.resolve()) and producer != ACTUAL_RUNS.resolve(), 'Use the completed new seed02 cache')
    selection = evaluation['cache_producer']['selection']
    need(len(selection['training_ids']) == 1641 and len(selection['validation_ids']) == 32
         and selection['test_accessed'] is False, 'Use fixed full training selection, never validation/test inputs')
    sealed = read(evaluation['cache_producer']['seal']['path'])['artifact_sha256']
    result = []
    for index in range(WARMUP + TIMED):
        directory = producer/f'clip_{index}'
        meta = read(directory/'metadata.json')
        need(meta['record_id'] == selection['training_ids'][index] and meta['split'] == 'train'
             and meta['caption'] == evaluation['caption']['text']
             and len(meta['source_indices']) == 33
             and meta['source_indices'] == list(range(meta['source_indices'][0],meta['source_indices'][0]+33)),
             'Diagnostic data must be the first22 original stride1 training clips')
        result.append({'index':index,'record_id':meta['record_id'],'binding':meta['binding'],
            'metadata':binding(directory/'metadata.json',sealed[f'clip_{index}/metadata.json']),
            'latents':binding(directory/'fresh_latents.pt',sealed[f'clip_{index}/fresh_latents.pt'])})
    return result


def source_bindings(root):
    root = Path(root).resolve()
    paths = [*sorted((root/'vsr').glob('*.py')), root/'scripts/stage3_resource_gate.py',
             root/'scripts/stage2_evaluate_bases.py',root/'scripts/stage2_framewise.py',root/'scripts/stage2_metrics.py',
             root/'tests/test_stage3_resource_gate.py',root/'tests/test_self_distillation.py',root/'tests/test_self_distillation_step.py',
             root/'tests/test_method_checkpoint.py',root/'tests/test_method_diagnostics.py',root/'tests/test_restoration_architecture.py']
    return {str(p.relative_to(root)):binding(p) for p in paths}


def collect_protocol(args):
    root = Path(args.project_root).resolve()
    need(Path.cwd().resolve() == root and Path(__file__).resolve() == root/'scripts/stage3_resource_gate.py',
         'Use this staging source root as CWD; source shadowing is forbidden')
    need(root != t2.OLD and not root.is_relative_to(t2.OLD) and root != ACTUAL_RUNS.parent,
         'Develop/run this future worker from frozen staging, never the active actual or historical source')
    evaluation, evidence = completed_t2(args.evaluation_run,root)
    base = evaluation['checkpoints'][str(METHOD['training_seed'])]
    options = base['config']['training']
    need(options['updates'] == 8192 and options['variants'] == ['Base']
         and base['config'].get('stage') == 'stage2_protocol33_noref_base', 'Diagnostic/probe checkpoints are forbidden')
    return json.loads(json.dumps({'schema':'stage3_G3_full_optimizer_diagnostic_v1',
        'project_root':str(root),'worker':binding(__file__),'completed_T2':evidence,
        'evaluation_protocol':evaluation,'matched_Base':base,'inputs':latent_inputs(evaluation),
        'source':source_bindings(root),'runtime':evaluation['runtime'],
        'method':METHOD,'budget':BUDGET,'roles':list(ROLES),'warmup_per_role':WARMUP,'timed_per_role':TIMED,
        'optimizer':{'type':'fresh AdamW','learning_rate':options['learning_rate'],
            'weight_decay':options['weight_decay'],'max_grad_norm':options['max_grad_norm'],
            'foreach':False,'resume_optimizer_scheduler_RNG':False},
        'resident_VAE':{'FP32':True,'frozen_eval':True,'encode_decode_calls':0,'same_VAE_all_roles':True},
        'semantics':{'same_layer':9,'same_layer_exact':True,'rho0_exact_parameter_and_optimizer_trajectory':True,
            'reference':'direct original core FM forward/backward/clip with explicit AdamW foreach=False',
            'historical_GPU_foreach_None_bitwise_equivalence_claimed':False,'tolerance_fallback':False},
        'gate':{'primary':'mean(A2 all20 timed full steps)/mean(flow_only all20 timed full steps)',
            'maximum_ratio':1.5,'memory_cap_bytes':MEMORY_CAP,'outlier_removal':False,
            'secondary':['active-only A2/flow','A2/C1','C1/flow','teacher active count','all per-step FM/SD'],
            'all_timed_sigma_window':'default sigma>.3; no forced active timestep or resampling',
            'sample_SD_ddof':1},
        'timing':{'includes':'H2D clean/LR + entire unchanged NoRefContinuationStep.step + final CUDA synchronize',
            'Step_contracts_included':['finite values/gradients','frozen checks','teacher restore','AdamW','EMA','cosine scalar','scheduled detached CPU snapshot copy'],
            'excludes':'role load, independent full-state hashes, RNG-state hashes, CPU JSON I/O, semantic checks, explicitly drained CPU feature std/SVD',
            'not_training_wall_clock_or_RE_cost':True},
        'output':{'fresh_only':True,'resume':False,'automatic_retry':False,'quality_inference':False,
            'save_task_or_optimizer_checkpoint':False,'reuse_diagnostic_weights':False},
        'source_freeze':'Shared numeric vsr sources must still match the completed T2 protocol. This independent G3 script/tests are sealed at G3 prepare after passed G2. Subsequent development belongs in scratch, not the sealed source tree.'},
        ensure_ascii=False,allow_nan=False))


def safe_run(args):
    run, root = Path(args.run_dir).resolve(), Path(args.project_root).resolve()
    need(run.is_relative_to(root/'runs') and run != root/'runs'
         and not run.is_relative_to(Path(args.evaluation_run).resolve()), 'Output must be a fresh independent staging runs child')
    return run


def prepare(args):
    run = safe_run(args)
    need(not run.exists(), 'Fresh-only G3 output; existing directories are forbidden')
    protocol = collect_protocol(args)
    run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    save(run/'status.json',{'status':'planned','protocol_sha256':file_sha(run/'protocol.json'),
        'diagnostic_optimizer_updates_applied':0,'scientific_training_updates':0,'G3_passed':False})
    return {'status':'planned','run':str(run),'protocol_sha256':file_sha(run/'protocol.json'),'budget':BUDGET}


def summarize_roles(rows):
    """Pure-stdlib predeclared ratio over every timed row, never selected scores."""
    groups = {}
    for role in ROLES:
        values = [x for x in rows if x['role'] == role]
        need(len(values) == WARMUP+TIMED and [x['ordinal'] for x in values] == list(range(WARMUP+TIMED))
             and [x['phase'] for x in values] == ['warmup']*WARMUP+['timed']*TIMED, 'Incomplete/reordered role timings')
        need(all(type(x['seconds']) in (int,float) and math.isfinite(x['seconds']) and x['seconds'] > 0
                 and all(type(x[k]) in (int,float) and math.isfinite(x[k]) for k in ('fm','sd','total','sigma'))
                 for x in values), 'Nonfinite timing/objective observation')
        groups[role] = values
    need(len(rows) == 66, 'Exactly66 benchmark updates required')
    need([x['role'] for x in rows] == [role for role in ROLES for _ in range(WARMUP+TIMED)], 'Roles must run sequentially flow_only/C1/A2')
    for ordinal in range(WARMUP+TIMED):
        first = groups['flow_only'][ordinal]['matched_FM']
        need(all(groups[role][ordinal]['matched_FM'] == first for role in ROLES), 'Data/noise/sigma streams differ across roles')
        need(groups['C1'][ordinal]['mask_spans'] == groups['A2'][ordinal]['mask_spans'], 'C1/A2 corruption masks differ')
        need(groups['C1'][ordinal]['matched_corruption_RNG'] == groups['A2'][ordinal]['matched_corruption_RNG'],
             'C1/A2 corruption RNG streams differ')
        for role in ROLES:
            row = groups[role][ordinal]
            need(row['sigma'] == row['matched_FM']['sigma'] and 0 <= row['sigma'] <= 1
                 and row['teacher_forward'] == (role == 'A2' and row['sigma'] > SIGMA_BOUNDARY)
                 and (row['sd'] == 0 if not row['teacher_forward'] else True), 'Teacher window or disabled SD mismatch')
    def stats(values):
        return {'n':len(values),'mean_seconds':statistics.mean(values) if values else None,
                'sample_SD_seconds':statistics.stdev(values) if len(values)>1 else None}
    timed = {role:groups[role][WARMUP:] for role in ROLES}
    summaries = {role:stats([x['seconds'] for x in values]) for role,values in timed.items()}
    objectives = {role:{key:{'mean':statistics.mean(x[key] for x in values),
                            'sample_SD':statistics.stdev(x[key] for x in values)}
                        for key in ('fm','sd','total')} for role,values in timed.items()}
    active_indices = [i for i,x in enumerate(timed['A2']) if x['teacher_forward']]
    def ratio(numerator,denominator,indices=None):
        a,b = timed[numerator],timed[denominator]
        indices = list(range(TIMED)) if indices is None else indices
        return statistics.mean(a[i]['seconds'] for i in indices)/statistics.mean(b[i]['seconds'] for i in indices) if indices else None
    primary = ratio('A2','flow_only')
    return {'status':'passed_G3_resource_cost' if primary <= 1.5 else 'failed_G3_overhead_stop',
        'primary_ratio_A2_over_flow_all20':primary,'maximum_ratio':1.5,'role_timing':summaries,'timed_objectives':objectives,
        'secondary':{'A2_over_C1_all20':ratio('A2','C1'),'C1_over_flow_all20':ratio('C1','flow_only'),
            'A2_over_flow_same_active_indices':ratio('A2','flow_only',active_indices),
            'A2_over_C1_same_active_indices':ratio('A2','C1',active_indices),
            'teacher_active_count_timed':len(active_indices),'active_timed_indices':active_indices,
            'teacher_active_count_including_warmup':sum(x['teacher_forward'] for x in groups['A2']),
            'active_A2_timing':stats([timed['A2'][i]['seconds'] for i in active_indices])},
        'observations':66,'timed_observations':60,'warmup_observations':6,'outliers_removed':0,
        'full_rows_retained':True,'scope':'single-seed resource diagnostic on real cached training latents; no quality or RE conclusion'}


class UpdateCounter:
    """Count successful AdamW calls even if later EMA/contract checks fail."""
    def __init__(self):
        self.role = None
        self.by_role = {}
        self.cleanup_events = []
    def attach(self,optimizer,role):
        self.role = role
        bucket = self.by_role.setdefault(role,{'attempted':0,'applied':0,'returned_steps':0})
        original = optimizer.step
        def counted(*args,**kwargs):
            bucket['attempted'] += 1
            result = original(*args,**kwargs)
            bucket['applied'] += 1
            return result
        optimizer.step = counted
    def snapshot(self):
        return {'by_role':copy.deepcopy(self.by_role),
            'diagnostic_optimizer_updates_attempted':sum(x['attempted'] for x in self.by_role.values()),
            'diagnostic_optimizer_updates_applied':sum(x['applied'] for x in self.by_role.values()),
            'completed_step_returns':sum(x['returned_steps'] for x in self.by_role.values()),
            'optimizer_calls_not_returned_partial_effect_unknown':sum(x['attempted']-x['applied'] for x in self.by_role.values()),
            'applied_count_scope':'AdamW calls that successfully returned; a throwing AdamW call may have partial effects and is never resumed',
            'cleanup_events':copy.deepcopy(self.cleanup_events),'scientific_training_updates':0}


def _state_signature(model):
    return [(name,id(value),value.data_ptr(),value._version,tuple(value.shape),str(value.dtype),str(value.device),value.requires_grad)
            for name,value in list(model.named_parameters())+list(model.named_buffers())]


def _frozen_signature(model):
    return [(name,id(p),p.data_ptr(),p._version,tuple(p.shape),str(p.dtype),str(p.device))
            for name,p in model.named_parameters() if not p.requires_grad]


def _tensor_digest(torch,value):
    cpu = value.detach().cpu().contiguous()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(json.dumps([str(cpu.dtype),list(cpu.shape)]).encode()+b'\n'+raw).hexdigest()


def _streams(torch,engine,keys=('noise','timestep')):
    return {key:_tensor_digest(torch,getattr(engine.rng,key).get_state()) for key in keys}


def _memory(torch):
    result = {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
              'physical_device_bytes':torch.cuda.get_device_properties(0).total_memory}
    need(max(result['peak_allocated_bytes'],result['peak_reserved_bytes']) <= min(MEMORY_CAP,result['physical_device_bytes']),
         'G3 exceeded the fixed16GiB/physical-device memory ceiling')
    return result


def _load_inputs(torch,protocol):
    pairs = []
    for item in protocol['inputs']:
        need(binding(item['latents']['path']) == item['latents'], 'Sealed diagnostic latent file changed')
        payload = torch.load(item['latents']['path'],map_location='cpu',weights_only=True)
        need(isinstance(payload,dict) and set(payload) == {'clean','low','binding'}
             and payload['binding'] == item['binding'], 'Latent data binding differs')
        pair = {key:payload[key] for key in ('clean','low')}
        need(all(isinstance(x,torch.Tensor) and list(x.shape) == METHOD['latent_shape'] and x.dtype == torch.float32
                 and x.device.type == 'cpu' and not x.requires_grad and bool(torch.isfinite(x).all()) for x in pair.values()),
             'Use original finite frozen FP32 normalized33-frame/9-position latents')
        pairs.append(pair)
    cap = protocol['evaluation_protocol']['caption']
    cache = torch.load(cap['cache']['path'],map_location='cpu',weights_only=True)
    text = cache.get(cap['text']) if isinstance(cache,dict) else None
    need(isinstance(text,torch.Tensor) and list(text.shape) == [1,512,4096] == cap['shape']
         and str(text.dtype) == cap['dtype'] == 'torch.bfloat16' and bool(torch.isfinite(text).all()),
         'Exact finite BF16 cached text required; T5 fallback forbidden')
    header = json.dumps([str(text.dtype),list(text.shape)],separators=(',',':')).encode()+b'\n'
    need(hashlib.sha256(header+text.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == cap['embedding_sha256'],
         'Cached text tensor digest differs')
    return pairs,text.detach().to('cuda')


def _new_engine(protocol,arm,*,rho=None):
    from vsr.self_distillation_step import load_matched_base_continuation
    base = protocol['matched_Base']
    options = {key:METHOD[key] for key in ('rho','span','gamma','student_layer','teacher_layer','hidden_dim','ema_decay',
                                        'feature_diagnostics_first','feature_diagnostics_every')}
    if rho is not None:options['rho'] = rho
    engine = load_matched_base_continuation(base['checkpoint']['path'],expected_checkpoint_sha256=base['checkpoint']['sha256'],
        expected_training_seed=METHOD['training_seed'],allowed_root=protocol['evaluation_protocol']['allowed_checkpoint_root'],
        arm=arm,device='cuda',**options)
    need(engine.base_load_audit['config'] == base['config'] and not engine.optimizer.state
         and all(group['foreach'] is False for group in engine.optimizer.param_groups), 'Fresh matched Base/AdamW contract differs')
    return engine


def _cleanup_engine(torch,engine,*,successful=True,counter=None):
    try:
        if engine is not None:
            engine.optimizer.zero_grad(set_to_none=True)
            engine.model.clear_features()
    except BaseException as error:
        if successful:raise
        counter.cleanup_events.append({'cleanup_error':f'{type(error).__name__}: {error}',
            'original_exception_preserved':True})


def _check_release(torch,reference,successful,counter,role):
    """Propagating traceback frames legitimately keep a failed engine alive."""
    try:
        gc.collect();torch.cuda.empty_cache()
        released = reference is None or reference() is None
        if successful:need(released,'Diagnostic model/optimizer remains resident across roles: '+role)
        else:counter.cleanup_events.append({'role':role,'release_assert_skipped_during_original_exception':True,
            'model_reference_alive_via_possible_traceback':not released,'process_must_stop':True})
    except BaseException as error:
        if successful:raise
        counter.cleanup_events.append({'role':role,'cleanup_error':f'{type(error).__name__}: {error}',
            'original_exception_preserved':True})


def _snapshot_core(engine):
    result = {}
    for name,p in engine.model.named_parameters():
        if p.requires_grad:
            result[name] = {'parameter':p.detach().cpu().clone(),
                'optimizer':{key:value.detach().cpu().clone() if hasattr(value,'detach') else value
                             for key,value in engine.optimizer.state[p].items()}}
    return result


def _equal_snapshots(torch,expected,actual):
    need(set(expected) == set(actual), 'rho0 trainable names differ')
    for name,left in expected.items():
        right = actual[name]
        need(torch.equal(left['parameter'],right['parameter']) and set(left['optimizer']) == set(right['optimizer']),
             'rho0 exact parameter/optimizer mismatch: '+name)
        for key,value in left['optimizer'].items():
            other = right['optimizer'][key]
            need(torch.equal(value,other) if isinstance(value,torch.Tensor) else value == other,
                 'rho0 exact AdamW moment/count mismatch: '+name+'/'+key)


def run_semantics(torch,protocol,pairs,text,scheduler,counter):
    """Independent fresh models; no diagnostic state enters the timed roles."""
    from vsr.self_distillation_step import capture_block_output
    from vsr.self_distillation import _require_finite
    from vsr.reproducibility import tensor_state_hashes
    reports, trajectories, scalar_rows = {}, [], []
    engine = clean = low = teacher = student = flow = target = noisy = loss = None
    reference = None
    try:
        engine = _new_engine(protocol,'A2',rho=0.)
        reference = weakref.ref(engine.model)
        before = tensor_state_hashes(engine.model)['all_trainable']['sha256']
        frozen = _frozen_signature(engine.model)
        flags = [(module,module.training) for module in engine.model.modules()]
        clean,low = (pairs[0][key].to('cuda') for key in ('clean','low'))
        noisy,target,timestep,sigma,_ = engine.rng.flow_inputs(clean,scheduler)
        teacher = engine.teacher_features(noisy,timestep,text,low,layer=9)
        engine.model.train()
        flow,student = capture_block_output(engine.model,9,noisy,timestep,text,low)
        need(not teacher.requires_grad and student.requires_grad and torch.equal(teacher,student),
             'GPU same-layer9 EMA=student exact equality failed; stop without relaxing tolerance')
        engine.model.clear_features()
        for module,flag in flags:module.training = flag
        need(before == tensor_state_hashes(engine.model)['all_trainable']['sha256']
             and frozen == _frozen_signature(engine.model)
             and all(p.grad is None for p in engine.model.parameters()), 'Teacher swap did not restore the student/frozen state')
        reports['same_layer'] = {'layer':9,'exact_equal':True,'teacher_detached':True,'student_restored':True,'optimizer_updates':0}
    finally:
        successful = sys.exc_info()[0] is None
        _cleanup_engine(torch,engine,successful=successful,counter=counter)
        flags = [];module = None
        engine = clean = low = teacher = student = flow = target = noisy = loss = None
        _check_release(torch,reference,successful,counter,'semantic_same_layer')
    for role in ('semantic_direct_core','semantic_A2_rho0'):
        reference = None
        try:
            engine = _new_engine(protocol,'flow_only' if role == 'semantic_direct_core' else 'A2',rho=0.)
            reference = weakref.ref(engine.model)
            counter.attach(engine.optimizer,role)
            frozen = _frozen_signature(engine.model)
            for ordinal in range(SEMANTIC_STEPS):
                clean,low = (pairs[ordinal][key].to('cuda') for key in ('clean','low'))
                if role == 'semantic_direct_core':
                    engine.model.train()
                    noisy,target,timestep,sigma,index = engine.rng.flow_inputs(clean,scheduler)
                    engine.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):flow,projection = engine.model(noisy,timestep,text,low)
                    need(projection is None,'Direct core must be Base-only')
                    loss = torch.nn.functional.mse_loss(flow.float(),target)
                    need(bool(torch.isfinite(loss)), 'Nonfinite semantic direct-core objective')
                    loss.backward()
                    _require_finite(('Nonfinite semantic core gradient: '+name,p.grad) for name,p in engine.model.named_parameters() if p.grad is not None)
                    need(all(p.grad is None for p in engine.model.parameters() if not p.requires_grad),'Frozen semantic reference received gradients')
                    torch.nn.utils.clip_grad_norm_(engine.parameters,engine.max_grad_norm,error_if_nonfinite=True)
                    engine.optimizer.step();engine.model.clear_features();engine.optimizer.zero_grad(set_to_none=True)
                    row = {'fm':float(loss.detach()),'sigma':float(sigma),'timestep_index':index}
                    trajectories.append(_snapshot_core(engine));scalar_rows.append(row)
                else:
                    result = engine.step(clean,low,text,scheduler)
                    row = {key:result[key] for key in ('fm','sigma','timestep_index')}
                    need(row == scalar_rows[ordinal] and result['sd'] == 0 and not result['teacher_forward'],
                         'GPU rho0 direct-core flow trajectory differs')
                    _equal_snapshots(torch,trajectories[ordinal],_snapshot_core(engine))
                counter.by_role[role]['returned_steps'] += 1
                need(frozen == _frozen_signature(engine.model), 'Frozen semantic parameters changed')
            reports[role] = {'updates':2,'exact_trajectory_equal':True if role.endswith('rho0') else None,
                'optimizer_foreach':False,'memory':_memory(torch)}
        finally:
            successful = sys.exc_info()[0] is None
            _cleanup_engine(torch,engine,successful=successful,counter=counter)
            engine = clean = low = teacher = student = flow = target = noisy = loss = None
            _check_release(torch,reference,successful,counter,role)
    trajectories.clear()
    return {'status':'passed_exact_GPU_semantics','checks':reports,'diagnostic_optimizer_updates':4,
        'historical_GPU_foreach_None_equivalence_claimed':False,'tolerance_fallback':False}


def run_benchmark_role(torch,protocol,role,pairs,text,scheduler,counter,rows,run):
    from vsr.self_distillation import _require_finite
    from vsr.reproducibility import tensor_state_hashes
    engine = clean = low = None
    reference = None
    torch.cuda.reset_peak_memory_stats()
    try:
        engine = _new_engine(protocol,role)
        reference = weakref.ref(engine.model)
        counter.attach(engine.optimizer,role)
        frozen = _frozen_signature(engine.model)
        flags = [(module,module.training) for module in engine.model.modules()]
        before = tensor_state_hashes(engine.model)['all_trainable']['sha256']
        _require_finite(('Nonfinite loaded parameter: '+name,p) for name,p in engine.model.named_parameters())
        for ordinal in range(WARMUP+TIMED):
            item = protocol['inputs'][ordinal]
            before_rng = _streams(torch,engine)
            before_aux = _streams(torch,engine,('mask','corruption'))
            torch.cuda.synchronize()
            started = time.perf_counter()
            clean,low = (pairs[ordinal][key].to('cuda') for key in ('clean','low'))
            result = engine.step(clean,low,text,scheduler)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            counter.by_role[role]['returned_steps'] += 1
            feature_diagnostics = engine.drain_feature_diagnostics()
            after_rng = _streams(torch,engine)
            after_aux = _streams(torch,engine,('mask','corruption'))
            need(all(module.training == flag for module,flag in flags)
                 and frozen == _frozen_signature(engine.model)
                 and all(p.grad is None for p in engine.parameters)
                 and engine.model._condition_tokens is None and engine.model._captured is None,
                 'Role flags/frozen/gradient/feature cleanup contract changed')
            memory = _memory(torch)
            row = {'role':role,'ordinal':ordinal,'phase':'warmup' if ordinal<WARMUP else 'timed','seconds':elapsed,
                **result,'memory':memory,'matched_FM':{'data_index':ordinal,'record_id':item['record_id'],
                    'latent_file_sha256':item['latents']['sha256'],'before_rng':before_rng,'after_rng':after_rng,
                    **{k:result[k] for k in ('timestep_index','native_timestep','sigma')}},
                'matched_corruption_RNG':{'before':before_aux,'after':after_aux},
                'feature_diagnostics_CPU_outside_step_timing':feature_diagnostics}
            rows.append(row)
            save(run/'progress.json',{'role':role,'completed_rows':len(rows),**counter.snapshot()},replace=True)
        _require_finite(('Nonfinite final parameter: '+name,p) for name,p in engine.model.named_parameters())
        if engine.ema is not None:engine.ema._validate(engine.model)
        return {'role':role,'fresh_Base_SHA256':protocol['matched_Base']['checkpoint']['sha256'],
            'trainable_sha256_before':before,'trainable_sha256_after':tensor_state_hashes(engine.model)['all_trainable']['sha256'],
            'returned_updates':engine.steps,'optimizer_updates_applied':counter.by_role[role]['applied'],
            'EMA_updates':engine.ema.num_updates if engine.ema else 0,'memory':_memory(torch),
            'frozen_identity_versions_unchanged':True,'mode_and_gradient_cleanup_passed':True}
    finally:
        successful = sys.exc_info()[0] is None
        _cleanup_engine(torch,engine,successful=successful,counter=counter)
        flags = []
        engine = clean = low = None
        _check_release(torch,reference,successful,counter,role)


@contextmanager
def _leased_lifecycle(allowed_root,cleanup,record_failure,record_secondary):
    """Hold the GPU lease through failure records AND the final cleanup.

    An error while cleaning an already-failed execution is secondary evidence;
    it must not replace the original numerical/OOM exception. A cleanup-only
    error instead fails the run even if its cost gate was already calculated.
    ExitStack releases the lease only after this try/except/finally finishes.
    """
    with ExitStack() as lease:
        primary = None
        try:
            lease.enter_context(t2.gpu_lease(allowed_root))
            yield
        except BaseException as error:
            primary = error
            try:record_failure(error,traceback.format_exc())
            except BaseException as receipt_error:
                error.add_note('Writing the primary failure receipt also failed: '+repr(receipt_error))
            raise
        finally:
            try:
                cleanup()
            except BaseException as secondary:
                details = traceback.format_exc()
                if primary is None:
                    try:record_failure(secondary,details)
                    except BaseException as receipt_error:
                        secondary.add_note('Writing the cleanup failure receipt also failed: '+repr(receipt_error))
                    raise
                # Even a failed secondary receipt must not replace the primary.
                try:record_secondary(primary,secondary,details)
                except BaseException as receipt_error:
                    primary.add_note('Secondary cleanup error: '+repr(secondary)
                                     +'; writing its receipt also failed: '+repr(receipt_error))


def _save_execution_failure(run,failure,counts):
    """Canonical status cannot stay passed when final cleanup fails."""
    save(run/'failure.json',failure)
    state = {'status':'failed_stop_no_retry','G3_passed':False,'protocol_sha256':failure['protocol_sha256'],
        'failure_sha256':file_sha(run/'failure.json'),**counts}
    if (run/'summary.json').exists():
        previous = read(run/'summary.json')
        previous.update(status='failed_stop_no_retry',G3_passed=False,
            computed_cost_gate_status_before_execution_failure=previous['status'],
            execution_failure_sha256=state['failure_sha256'],next_action='stop_report_no_fallback_retry')
        save(run/'summary.json',previous,replace=True)
        state['summary_sha256'] = file_sha(run/'summary.json')
    save(run/'status.json',state,replace=True)


def execute(args):
    run = safe_run(args)
    need(args.expected_protocol_sha and file_sha(run/'protocol.json') == args.expected_protocol_sha,'Exact prepared protocol SHA required')
    status = read(run/'status.json')
    need(status['status'] == 'planned' and status['protocol_sha256'] == args.expected_protocol_sha,
         'Only a never-started planned run may execute; no resume/retry')
    with (run/'execute.lock').open('x') as stream:stream.write(str(os.getpid())+'\n')
    save(run/'status.json',{'status':'running','protocol_sha256':args.expected_protocol_sha,
        'diagnostic_optimizer_updates_applied':0,'scientific_training_updates':0,'G3_passed':False},replace=True)
    protocol = read(run/'protocol.json')
    torch = vae = text = scheduler = pairs = None
    rows, roles, counter, semantics = [], [], UpdateCounter(), None
    started, phase = time.perf_counter(), 'CPU_bindings_and_exclusive_GPU_lease'
    def record_failure(error,details):
        failure = {'status':'failed_stop_no_retry','phase':phase,'error':f'{type(error).__name__}: {error}',
            'traceback':details,'protocol_sha256':args.expected_protocol_sha,'partial_rows':rows,
            'completed_roles':roles,'semantics':semantics,**counter.snapshot(),'wall_seconds':time.perf_counter()-started,
            'G3_passed':False,'automatic_retry':False,'quality_inference':False,'reusable_checkpoints_written':False}
        _save_execution_failure(run,failure,counter.snapshot())

    def record_secondary(primary,secondary,details):
        receipt = {'phase':'final_GPU_cleanup','primary_error_preserved':f'{type(primary).__name__}: {primary}',
            'primary_failure_sha256':file_sha(run/'failure.json'),
            'secondary_cleanup_error':f'{type(secondary).__name__}: {secondary}',
            'secondary_traceback':details,'G3_passed':False,'automatic_retry':False}
        save(run/'cleanup_failure.json',receipt)
        state = read(run/'status.json')
        state['cleanup_failure_sha256'] = file_sha(run/'cleanup_failure.json')
        save(run/'status.json',state,replace=True)

    def final_cleanup():
        nonlocal vae,text,scheduler,pairs,phase
        phase = 'final_GPU_cleanup'
        vae = text = scheduler = pairs = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():torch.cuda.empty_cache()

    with _leased_lifecycle(ACTUAL_RUNS,final_cleanup,record_failure,record_secondary):
        need(protocol == collect_protocol(args),'Prepared G3 source/prerequisites/runtime changed; stop and create a fresh reviewed run')
        need(all(os.environ.get(k) == v for k,v in t2.ENVIRONMENT.items()), 'Exact strict/offline launch environment required')
        evaluation = protocol['evaluation_protocol']
        classifier = t2.import_helper(evaluation['helpers']['gpu_classifier']['path'],'stage3_GPU_guard')
        guard = t2.gpu_idle_guard(evaluation['desktop_binding'],classifier)
        save(run/'GPU_idle_before_load.json',guard)
        need(guard['passed'],'GPU busy/unverified; never overlap Base/evaluation or retry automatically')
        sys.path[:0] = [protocol['project_root'],evaluation['dependencies']['root']]
        import torch as torch_module
        torch = torch_module
        need(not torch.cuda.is_initialized(),'Strict configuration must precede CUDA initialization')
        torch.set_num_threads(2)
        from vsr.reproducibility import configure_reproducibility
        runtime = configure_reproducibility(protocol['matched_Base']['config'])
        flags = runtime['flags']
        need(flags['deterministic_algorithms'] and not flags['deterministic_warn_only']
             and flags['sdp_math'] and not any(flags[k] for k in ('sdp_flash','sdp_memory_efficient','sdp_cudnn')),
             'Strict native math backend required')
        save(run/'runtime.json',runtime)
        t2.verify_imported_dependencies(evaluation['dependencies'])
        from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
        from vsr.infer_video import load_frozen_vae
        from vsr.self_distillation import _require_finite
        t2.verify_native_sources(evaluation['native_numerical_sources'],{'AutoencoderKLWan':AutoencoderKLWan,
            'WanTransformer3DModel':WanTransformer3DModel,'UniPCMultistepScheduler':UniPCMultistepScheduler})
        for name,module in list(sys.modules.items()):
            if name == 'vsr' or name.startswith('vsr.'):
                need(Path(module.__file__).resolve().is_relative_to(Path(protocol['project_root'])/'vsr'),'Imported vsr module shadows sealed staging')
        need(not torch.cuda.is_initialized(),'CPU setup/dependency imports unexpectedly initialized CUDA')
        phase = 'resident_FP32_VAE_and_training_latents'
        vae = load_frozen_vae(protocol['matched_Base']['config'],device='cuda')
        need(not vae.training and all(not p.requires_grad and p.dtype == torch.float32 and p.device.type == 'cuda'
                                     for p in vae.parameters()),'One frozen eval FP32 resident VAE required')
        _require_finite(('Nonfinite frozen VAE parameter: '+name,p) for name,p in vae.named_parameters())
        vae_signature = _state_signature(vae)
        pairs,text = _load_inputs(torch,protocol)
        scheduler = UniPCMultistepScheduler.from_pretrained(protocol['matched_Base']['config']['model']['local_dir'],
            subfolder='scheduler',local_files_only=True)
        scheduler.set_timesteps(1000,device='cuda')
        torch.cuda.reset_peak_memory_stats()
        phase = 'independent_GPU_semantics'
        semantics = run_semantics(torch,protocol,pairs,text,scheduler,counter)
        need(vae_signature == _state_signature(vae),'Resident VAE changed during semantic checks')
        save(run/'semantics.json',semantics)
        for role in ROLES:
            phase = 'benchmark_'+role
            roles.append(run_benchmark_role(torch,protocol,role,pairs,text,scheduler,counter,rows,run))
            need(vae_signature == _state_signature(vae),'Resident VAE changed during benchmark')
            save(run/'completed_roles.json',{'roles':roles,**counter.snapshot()},replace=True)
        phase = 'complete_cost_gate'
        gate = summarize_roles(rows)
        need(counter.snapshot()['diagnostic_optimizer_updates_applied'] == 70
             and all(x['returned_updates'] == x['optimizer_updates_applied'] == 22 for x in roles), 'Diagnostic update budget/count differs')
        need(source_bindings(protocol['project_root']) == protocol['source'],'Sealed source changed during G3')
        save(run/'observations.json',{'rows':rows,'protocol_sha256':args.expected_protocol_sha,**counter.snapshot()})
        save(run/'G3.json',gate)
        summary = {'status':gate['status'],'G3_passed':gate['status'] == 'passed_G3_resource_cost',
            'protocol_sha256':args.expected_protocol_sha,'semantics_sha256':file_sha(run/'semantics.json'),
            'observations_sha256':file_sha(run/'observations.json'),'G3_sha256':file_sha(run/'G3.json'),
            'roles':roles,'budget':BUDGET,**counter.snapshot(),'wall_seconds':time.perf_counter()-started,
            'resident_VAE_unchanged_FP32':True,'quality_inference':False,'reusable_checkpoints_written':False,
            'next_action':'report_cost_and_semantics_before_pilot' if gate['status'] == 'passed_G3_resource_cost' else 'stop_report_no_fallback_retry',
            'scope':'G3 semantics/resource measurement only; T3.6 logging and research/quality gates are separately reviewed'}
        save(run/'summary.json',summary)
        save(run/'status.json',{'status':summary['status'],'G3_passed':summary['G3_passed'],
            'protocol_sha256':args.expected_protocol_sha,'summary_sha256':file_sha(run/'summary.json'),**counter.snapshot()},replace=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--evaluation-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--expected-protocol-sha')
    args = parser.parse_args()
    if args.mode == 'prepare':
        need(args.expected_protocol_sha is None,'prepare creates its own protocol SHA')
        result = prepare(args)
    else:result = execute(args)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':main()
