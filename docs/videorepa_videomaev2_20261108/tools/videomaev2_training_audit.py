"""Completed two-arm online VideoMAEv2 audit: CPU only, no quality/media reads.

The completed4096 status is checked before protocol/log/checkpoint access.
Eight full CP states are independently checked; public output contains only
metadata/hashes, never teacher/head/student tensors or original RNG bytes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ARMS=('baseline','videomaev2')
STEPS=(512,1024,1536,2048)
STREAMS=('noise','timestep','mask','corruption')
CORE={'__init__.py','data.py','vae.py','model.py','teacher.py','losses.py','train.py','infer.py','evaluate.py','metrics.py','checkpoints.py'}
WORKSPACE=Path(__file__).resolve().parent.parent

def need(value,message):
    if not value:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def binding(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}

class Verifier:
    def __init__(self):self.files={}
    def verify(self,value):
        item={k:value[k] for k in ('path','bytes','sha256')};path=Path(item['path']).resolve()
        before=path.stat();signature=(before.st_ino,before.st_size,before.st_mtime_ns)
        need(binding(path)==item,'Changed input: '+str(path));after=path.stat()
        need(signature==(after.st_ino,after.st_size,after.st_mtime_ns),'Input changed while hashed')
        self.files[str(path)]=signature;return path
    def unchanged(self):
        for path,signature in self.files.items():
            value=Path(path).stat();need((value.st_ino,value.st_size,value.st_mtime_ns)==signature,'Input changed during audit: '+path)

def finite(value):return type(value) in (int,float) and math.isfinite(value)

def tensor_description(torch,value):
    need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and not value.requires_grad,'CPU-owned tensor required')
    raw=value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'dtype':str(value.dtype),'shape':list(value.shape),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def parameters(torch,state):
    need(isinstance(state,dict) and state,'Nonempty parameter mapping required')
    need(all(isinstance(k,str) and isinstance(v,torch.Tensor) and v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for k,v in state.items()),
         'Expected finite FP32 parameter state')
    return {name:tensor_description(torch,value) for name,value in state.items()}

def trainable_digest(torch,state):
    result=hashlib.sha256();numel=0
    for name,value in sorted(state.items()):
        header=json.dumps({'name':name,'dtype':str(value.dtype),'shape':list(value.shape)},sort_keys=True,separators=(',',':')).encode()
        raw=value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header,raw):result.update(len(part).to_bytes(8,'big'));result.update(part)
        numel+=value.numel()
    return {'sha256':result.hexdigest(),'tensor_count':len(state),'numel':numel}

def origin(value):return {k:value[k] for k in ('checkpoint_path','checkpoint_sha256','step','variant','config')}

def summarize(rows,weight):
    mean=lambda key:statistics.mean(x[key] for x in rows)
    return {'rows':len(rows),'fm_mean':mean('fm'),'unit_TRD_mean':mean('align'),'weighted_TRD_mean':weight*mean('align'),
            'total_mean':mean('total'),'gradient_norm_mean':mean('grad_norm_before_clip'),
            'gradient_norm_max':max(x['grad_norm_before_clip'] for x in rows),
            'active_pair_fraction_mean':statistics.mean(x['alignment']['active_pair_fraction'] for x in rows) if weight else None}

def csv_file(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key,value in row.items() if not isinstance(value,(dict,list))))
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def metadata(run,source_root,expected_protocol_sha=None):
    run=Path(run).resolve();state=read(run/'status.json')
    need(state.get('status')=='completed' and state.get('scientific_updates')==4096,
         'Reject incomplete training before reading protocol, logs or checkpoints')
    verify=Verifier();verify.verify(binding(run/'status.json'))
    protocol,summary=read(run/'protocol.json'),read(run/'summary.json');psha=sha(run/'protocol.json')
    need(expected_protocol_sha is None or psha==expected_protocol_sha,'Unexpected training protocol SHA')
    need(state.get('protocol_sha256')==psha and state.get('summary_sha256')==sha(run/'summary.json'), 'Completion chain differs')
    need(protocol.get('schema')=='vsr_videomaev2_training_protocol_v1' and protocol.get('scientific_update_budget')==4096
         and summary.get('schema')=='vsr_videomaev2_training_summary_v1' and summary.get('status')=='completed'
         and summary.get('protocol_sha256')==psha and summary.get('scientific_updates')==4096
         and summary.get('diagnostic_updates_in_this_training')==0,'Require complete two-arm4096 VideoMAEv2 training')
    for name in ('protocol.json','summary.json'):verify.verify(binding(run/name))
    study=protocol['study'];verify.verify(protocol['study_file']);need(read(protocol['study_file']['path'])==study,'Study changed')
    need(study.get('schema')=='vsr_videomaev2_study_v1' and study.get('mode')=='training'
         and study['updates']==2048 and study['training_seed']==20261102 and study['student_layer']==17
         and study['margin']==.1 and study['normalization_eps']==1e-12 and study['online_teacher'] is True
         and study['checkpoint_updates']==list(STEPS)
         and study['arms']==[{'label':'baseline','weight':0.},{'label':'videomaev2','weight':.5}]
         and set(summary['arms'])==set(ARMS),'Prospective method/budget differs')
    source_root=Path(source_root).resolve();need(set(protocol['sources'])==CORE,'Require all11 numerical sources')
    for name,item in protocol['sources'].items():
        need(Path(item['path']).resolve()==source_root/'vsr'/name,'Source escaped declared checkout');verify.verify(item)
    calbindings=protocol['calibration'];need(set(calbindings)=={'protocol','summary','status_file'},'Missing preflight bindings')
    for item in calbindings.values():verify.verify(item)
    calp,cal,calstate=(read(calbindings[k]['path']) for k in ('protocol','summary','status_file'))
    need(calp['schema']=='vsr_videomaev2_preflight_protocol_v1' and calp['sources']==protocol['sources']==cal['sources']
         and cal['status']=='passed_videomaev2_preflight' and cal['scientific_updates']==0 and cal['diagnostic_optimizer_updates']==26
         and cal['teacher_forwards']==31 and cal['gradient_probe_cases']==15 and cal['gradient_VJP_calls']==30
         and cal['target_viability']['eligible'] is True and cal['zero_weight_exact'] is True
         and cal['resume_reports']['videomaev2']['exact_full_engine_state'] is True
         and cal['protocol_sha256']==calbindings['protocol']['sha256']==calstate['protocol_sha256']
         and calstate['status']=='completed' and calstate['summary_sha256']==calbindings['summary']['sha256'], 'Passed26-update preflight chain differs')
    teacher=protocol['teacher_binding']
    need(teacher==calp['teacher_binding'] and teacher['schema']=='vsr_videomaev2_teacher_v1'
         and teacher['weights']['sha256']=='8141a6955e0700d11bf15928fe6d61e5cfe482606fed8cfdddb1b922c0fd88ec'
         and teacher['encoder_load_contract']['encoder_state_keys']==158
         and teacher['encoder_load_contract']['encoder_parameters']==86225664
         and teacher['feature_spec']['input_range']=='RGB_float32_0_1'
         and teacher['feature_spec']['input_frame_indices']==list(range(1,33))
         and teacher['feature_spec']['output_shape']==[1,1792,768] and teacher['feature_spec']['target_cached'] is False
         and teacher['position_embedding']['shape']==[1,1792,768],'Frozen online teacher metadata differs')
    # This audit loads only Base and eight task CPs. The real preflight/core
    # source-chain checks own official teacher weights/positions and RGB payloads.
    need(protocol['rgb_assets']==calp['rgb_assets'],'Sealed online RGB cohort changed')
    verify.verify(protocol['base']['checkpoint']);verify.verify(protocol['base']['config_file'])
    need(read(protocol['base']['config_file']['path'])==protocol['base']['config'],'Base config sidecar differs')
    ids=protocol['cache']['selection']['training_ids'];need(len(ids)==len(set(ids))==1641,'Training cohort differs')
    logs={};initial=common=None;costs=[]
    fields=('record_id','data_index','sigma','timestep_index','native_timestep')
    for arm in study['arms']:
        label,weight=arm['label'],arm['weight'];record=summary['arms'][label];directory=run/'arm_runs'/label
        need(record['updates']==2048 and record['weight']==weight and record['family']==record['teacher_kind']==label
             and record['setting']=='common' and record['EMA_updates']==record['teacher_parameter_updates']==0
             and record['teacher_forward_count']==(2048 if weight else 0)
             and record['teacher_binding']==(teacher if weight else None)
             and record['teacher_is_frozen_cached_HR'] is False
             and [c['step'] for c in record['checkpoint_files']]==list(STEPS),'Arm identity/count differs')
        for name in ('config.json','initialization.json','completion.json'):verify.verify(binding(directory/name))
        need(read(directory/'completion.json')==record and read(directory/'initialization.json')==record['initialization'],'Arm sidecar differs')
        init=record['initialization'];need(init['checkpoint_sha256']==protocol['base']['checkpoint']['sha256']
             and init['step']==8192 and init['variant']=='Base'
             and all(init[k] is False for k in ('optimizer_state_restored','scheduler_state_restored','rng_state_restored')),'Not a fresh matched Base initialization')
        if initial is None:initial=record['initial_student_hashes']
        need(record['initial_student_hashes']==initial,'Student initialization differs across arms')
        path=verify.verify(record['full_training_log']);need(path==directory/'training.jsonl','Wrong canonical training log')
        raw=path.read_bytes();need(raw.endswith(b'\n'),'Incomplete trailing training row')
        rows=[json.loads(line) for line in raw.splitlines()]
        need(len(rows)==2048 and [r['step'] for r in rows]==list(range(1,2049)),'Missing/noncontiguous training step')
        elapsed=hr_seconds=0.
        for row in rows:
            i=(row['step']-1)%1641
            need(row['arm']==row['family']==label and row['setting']=='common' and row['weight']==weight
                 and row['data_index']==i and row['record_id']==ids[i],'Wrong arm/data order')
            need(all(finite(row[k]) for k in ('sigma','native_timestep','fm','align','total','grad_norm_before_clip',
                'student_total_grad_norm_before_clip','head_total_grad_norm_before_clip','seconds','gpu_event_seconds',
                'cumulative_training_seconds','hr_read_seconds','cumulative_hr_read_seconds')),'Nonfinite training scalar')
            need(0<=row['sigma']<=1 and type(row['timestep_index']) is int and 0<=row['timestep_index']<1000
                 and all(row[k]>=0 for k in ('grad_norm_before_clip','seconds','gpu_event_seconds','hr_read_seconds')),'Invalid native input/time')
            need(row['student_layer']==17 and row['teacher_forward'] is bool(weight)
                 and row['teacher_parameter_updates']==row['EMA_updates']==0
                 and row['teacher_forward_count']==(row['step'] if weight else 0),'Teacher forward/update accounting differs')
            need(row['noise_bin']==('low' if row['sigma']<.33 else 'mid' if row['sigma']<.67 else 'high'),'Noise bin differs')
            if weight:
                d=row['alignment'];need(d['margin']==.1 and d['normalize_eps']==1e-12 and d['tokens_per_example']==1792
                    and d['all_pairs']==3211264 and d['spatial_pairs']==200704 and d['temporal_pairs']==3010560
                    and d['all_tokens_supervised'] is True and d['diagonal_included'] is True and d['ordered_pairs'] is True
                    and d['projected_all_latent_positions']==9 and d['dropped_latent_positions']==[0] and d['target_grid']==[16,8,14]
                    and d['student_normalized_dtype']==d['student_gram_dtype']=='torch.bfloat16'
                    and d['teacher_normalized_dtype']==d['teacher_gram_dtype']==d['difference_dtype']==d['loss_dtype']=='torch.float32'
                    and finite(d['active_pair_fraction']) and 0<=d['active_pair_fraction']<=1,'Target mapping/full-Gram/dtypes differ')
            else:need(row['align']==0 and row['hr_read_seconds']==0,'Baseline used teacher target or extra RGB')
            elapsed+=row['seconds'];hr_seconds+=row['hr_read_seconds']
            need(row['cumulative_training_seconds']==elapsed and row['cumulative_hr_read_seconds']==hr_seconds,'Committed timers changed')
        need(record['cumulative_training_seconds']==elapsed and record['cumulative_hr_read_seconds']==hr_seconds,'Completion times differ')
        current=[[r[k] for k in fields] for r in rows]
        if common is None:common=current
        need(common==current,'Shared data/native sigma/timestep sequence differs')
        for cp in record['checkpoint_files']:
            cp_path=verify.verify(cp);step=cp['step']
            need(cp_path==directory/'Base'/f'checkpoint_{step:06d}.pt' and cp['update']==step
                 and cp['cumulative_training_seconds']==rows[step-1]['cumulative_training_seconds'],'CP path/time differs')
        costs.append({'arm':label,'updates':2048,'teacher_forwards':2048 if weight else 0,
            'committed_synchronized_step_seconds':elapsed,'separate_CPU_HR_read_seconds':hr_seconds,
            'sum_step_plus_extra_HR_seconds':elapsed+hr_seconds,'step_contains_teacher':bool(weight),
            'HR_read_is_contained_in_attempt_wall_not_step':True})
        logs[label]=rows
    cost=summary['compute_cost']
    need(cost['committed_training_step_seconds']==sum(r['cumulative_training_seconds'] for r in summary['arms'].values())
         and cost['committed_hr_read_seconds']==sum(r['cumulative_hr_read_seconds'] for r in summary['arms'].values())
         and cost['hr_read_time_scope']=='Outside synchronized Step; contained in whole attempt wall, never added again to attempt wall',
         'Step/HR-read cost scopes differ')
    verify.unchanged()
    return run,state,protocol,summary,logs,verify,{'fields':list(fields),'steps':2048,'digest':digest(common),'both_arms_exactly_equal':True},costs


def global_rng_audit(torch,state,device):
    import random
    import numpy as np
    need(isinstance(state,dict) and set(state)=={'python','numpy','torch_cpu','torch_cuda'},'Incomplete global RNG')
    random.Random().setstate(state['python']);n=state['numpy']
    need(set(n)=={'kind','keys','position','has_gauss','cached_gaussian'} and n['kind']=='MT19937'
         and len(n['keys'])==624 and all(type(v) is int and 0<=v<2**32 for v in n['keys'])
         and type(n['position']) is int and 0<=n['position']<=624 and n['has_gauss'] in (0,1)
         and finite(n['cached_gaussian']),'Invalid NumPy global RNG')
    np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    cpu=state['torch_cpu'];desc=tensor_description(torch,cpu)
    need(cpu.dtype==torch.uint8 and cpu.ndim==1,'Invalid global CPU RNG');torch.Generator(device='cpu').set_state(cpu)
    cuda=state['torch_cuda'];need(isinstance(cuda,list) and len(cuda)==(1 if str(device).startswith('cuda') else 0),'Global CUDA RNG count differs')
    cuda_desc=[]
    for value in cuda:
        need(value.dtype==torch.uint8 and value.ndim==1 and value.numel()>0,'Invalid global CUDA RNG bytes')
        cuda_desc.append(tensor_description(torch,value))
    return digest({'python':state['python'],'numpy':n,'torch_cpu':desc,'torch_cuda':cuda_desc})


def checkpoint_audit(protocol,summary,verify,psha):
    import torch
    from torch import nn
    need(not torch.cuda.is_initialized(),'CPU-only audit required')
    need(torch.__version__==protocol['dependencies']['torch']['version'],'CPU torch version differs')
    base=torch.load(protocol['base']['checkpoint']['path'],map_location='cpu',weights_only=True)
    need(base['format']=='vsr_training_v1' and base.get('variant') in (None,'Base') and base['step']==8192
         and base['config']==protocol['base']['config'],'Original Base payload differs')
    initial=base['trainable_state'];initial_desc=parameters(torch,initial);base_digest=trainable_digest(torch,initial)
    rng_reference={};rng_rows=[];update_rows=[];teacher_rows=[];flags=[];head_initial=None
    fields={'schema','protocol_sha256','arm','weight','student_layer','completed_updates','training_seed','teacher_binding',
        'teacher_forward_count','teacher_parameter_updates','alignment_state','alignment_spec','optimizer','optimizer_parameter_names',
        'explicit_RNG','RNG_devices','global_RNG','data_cursor','matched_Base','cumulative_training_seconds','learning_rate_schedule',
        'restored_by_generic_inference_loader'}
    for arm in protocol['study']['arms']:
        label,weight=arm['label'],arm['weight'];record=summary['arms'][label]
        need(record['initial_student_hashes']['all_trainable']==base_digest,'Student did not initialize from exact Base')
        for item in record['checkpoint_files']:
            value=torch.load(item['path'],map_location='cpu',weights_only=True);step=item['step'];state=value['training_only']
            need(value['format']=='vsr_training_v1' and value['variant']=='Base' and value['step']==step
                and value['inference_parameters']=='student' and set(state)==fields
                and state['schema']=='vsr_videomaev2_training_state_v1' and state['protocol_sha256']==psha
                and state['arm']==label and state['completed_updates']==step and state['training_seed']==20261102
                and state['weight']==weight and state['student_layer']==17
                and state['teacher_forward_count']==(step if weight else 0) and state['teacher_parameter_updates']==0
                and state['cumulative_training_seconds']==item['cumulative_training_seconds'],'Complete CP provenance/count differs')
            cfg=value['config'];need(cfg==read(Path(item['path']).parent.parent/'config.json')
                and all(cfg.get(k)==base['config'].get(k) for k in ('model','conditioning','data','seed')),'CP inference/config differs')
            aux=cfg['auxiliary_supervision'];need(aux['arm']==label and aux['weight']==weight and aux['student_layer']==17
                and aux['margin']==.1 and aux['normalization_eps']==1e-12 and aux['target_grid']==[16,8,14]
                and aux['EMA'] is False and aux['inference_parameters']=='student','Auxiliary config differs')
            expected_teacher=protocol['teacher_binding'] if weight else None
            need(state['teacher_binding']==aux['teacher_binding']==record['teacher_binding']==expected_teacher
                and 'teacher_state' not in state and 'teacher' not in value,'External teacher tensors must not enter CP')
            need(origin(value['initialization'])==origin(state['matched_Base']) and state['matched_Base']['config']==base['config']
                and state['matched_Base']['checkpoint_sha256']==protocol['base']['checkpoint']['sha256']
                and Path(state['matched_Base']['checkpoint_path']).resolve()==Path(protocol['base']['checkpoint']['path']).resolve(),'CP Base source differs')
            student=value['trainable_state'];desc=parameters(torch,student)
            need(set(desc)==set(initial_desc) and all(desc[n]['shape']==initial_desc[n]['shape'] for n in desc),'Student inventory differs')
            student_hash=trainable_digest(torch,student);changed=sum(desc[n]['sha256']!=initial_desc[n]['sha256'] for n in desc)
            if step==2048:need(student_hash==record['final_student_hashes']['all_trainable'],'Final student hash differs')
            need(state['data_cursor']=={'completed_updates':step,'training_clip_count':1641,'next_training_index':step%1641}
                and state['learning_rate_schedule']=={'type':'constant','separate_scheduler':None,'lr_saved_in_optimizer_groups':True},'Cursor/LR contract differs')
            need(set(state['explicit_RNG'])==set(state['RNG_devices'])==set(STREAMS)
                and state['RNG_devices']=={'noise':protocol['device'],'timestep':'cpu','mask':'cpu','corruption':protocol['device']},'Explicit RNG device contract differs')
            streams={}
            for name in STREAMS:
                tensor=state['explicit_RNG'][name];need(tensor.dtype==torch.uint8 and tensor.ndim==1,'RNG must be CPU bytevector')
                streams[name]=tensor_description(torch,tensor)
                saved=rng_reference.setdefault(step,{}).setdefault(name,tensor.clone())
                need(torch.equal(saved,tensor),'Four explicit RNG streams differ bytewise across arms: '+name)
            rng_rows.append({'arm':label,'step':step,'all_four_exactly_equal':True,
                **{n+'_sha256':streams[n]['sha256'] for n in STREAMS},'global_rng_sha256':global_rng_audit(torch,state['global_RNG'],protocol['device'])})
            head=state['alignment_state'];spec=state['alignment_spec'];head_count=head_changed=head_nonzero=0
            optimizer_parameters={'restoration.'+n:t for n,t in student.items()}
            if not weight:need(head is spec is None and record['head_parameters']==0,'Baseline acquired an auxiliary head')
            else:
                need(spec['schema']=='vsr_videomaev2_alignment_v1' and spec['student_dim']==1536 and spec['hidden_dim']==2048
                    and spec['teacher_dim']==768 and spec['linear_count']==3 and spec['activation']=='SiLU'
                    and spec['parameter_tensors']==6 and spec['parameter_count']==record['head_parameters']==8917760
                    and spec['init_seed']==20261102 and spec['project_before_drop0'] is True
                    and spec['drop_latent_positions']==[0] and spec['target_grid']==[16,8,14]
                    and spec['interpolation']=='trilinear8_to16_align_cornersFalse'
                    and spec['student_normalize_and_Gram']=='torch.bfloat16' and spec['teacher_normalize_and_Gram']=='torch.float32'
                    and spec['difference_and_loss']=='torch.float32' and spec['margin']==.1 and spec['normalize_eps']==1e-12,'Six-parameter MLP spec differs')
                head_desc=parameters(torch,head)
                shapes={'projector.0.weight':[2048,1536],'projector.0.bias':[2048],
                        'projector.2.weight':[2048,2048],'projector.2.bias':[2048],
                        'projector.4.weight':[768,2048],'projector.4.bias':[768]}
                need(set(head)==set(shapes) and all(head_desc[n]['shape']==shape for n,shape in shapes.items()),'Head inventory differs')
                if head_initial is None:
                    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
                        torch.random.default_generator.manual_seed(spec['init_seed'])
                        projector=nn.Sequential(nn.Linear(1536,2048,dtype=torch.float32),nn.SiLU(),
                            nn.Linear(2048,2048,dtype=torch.float32),nn.SiLU(),nn.Linear(2048,768,dtype=torch.float32))
                    head_initial={'projector.'+n:v.detach().clone() for n,v in projector.state_dict().items()};del projector
                head_count=len(head);head_changed=sum(not torch.equal(head[n],head_initial[n]) for n in head)
                optimizer_parameters.update({'student_projector.'+n.removeprefix('projector.'):t for n,t in head.items()})
            optimizer=state['optimizer'];groups=state['optimizer_parameter_names'];moments={};names=[];identifiers=[]
            need(len(groups)==len(optimizer['param_groups']) and all(g['foreach'] is False for g in optimizer['param_groups']),'AdamW groups differ')
            for group,group_names in zip(optimizer['param_groups'],groups):
                need(len(group['params'])==len(group_names),'AdamW names incomplete')
                for identifier,name in zip(group['params'],group_names):
                    moments[name]=optimizer['state'][identifier];names.append(name);identifiers.append(identifier)
            need(len(names)==len(set(names))==len(optimizer_parameters) and set(names)==set(optimizer_parameters)
                and len(identifiers)==len(set(identifiers))==len(optimizer_parameters) and set(identifiers)==set(optimizer['state']), 'Optimizer names/IDs/state must be bijective')
            for name,parameter in optimizer_parameters.items():
                moment=moments[name];counter=moment['step'];need(set(moment)=={'step','exp_avg','exp_avg_sq'}
                    and counter.ndim==0 and bool(torch.isfinite(counter)) and float(counter)==step,'AdamW step counter differs')
                for key in ('exp_avg','exp_avg_sq'):
                    tensor=moment[key];need(tensor.device.type=='cpu' and tensor.dtype==torch.float32 and tensor.shape==parameter.shape
                        and not tensor.requires_grad and bool(torch.isfinite(tensor).all()),'Invalid complete AdamW moment')
                    if key=='exp_avg_sq':need(bool((tensor>=0).all()),'Negative second moment')
                if name.startswith('student_projector.') and bool(moment['exp_avg_sq'].ne(0).any()):head_nonzero+=1
            if not changed:flags.append({'arm':label,'step':step,'flag':'no_student_parameter_change'})
            if weight and (not head_changed or not head_nonzero):flags.append({'arm':label,'step':step,'flag':'no_head_update_or_nonzero_moment_evidence'})
            update_rows.append({'arm':label,'step':step,'student_tensors':len(desc),'student_changed_tensors':changed,'student_digest':student_hash['sha256'],
                'head_tensors':head_count,'head_changed_tensors':head_changed,'head_nonzero_second_moment_tensors':head_nonzero,'all_optimizer_step_counts_match':True})
            teacher_rows.append({'arm':label,'step':step,'teacher_binding_sha256':digest(expected_teacher),
                'teacher_forward_count':state['teacher_forward_count'],'teacher_parameter_updates':0,'external_teacher_tensors_in_CP':False})
            del value,state,student,optimizer,moments,optimizer_parameters
    verify.unchanged();need(not torch.cuda.is_initialized(),'CUDA initialized in CPU audit')
    return {'rng_rows':rng_rows,'update_rows':update_rows,'teacher_rows':teacher_rows,'flags':flags,'Base_digest':base_digest,
        'explicit_four_RNG_streams_bytewise_identical_at_each_checkpoint':True,'global_RNG_complete_and_CPU_validated':True,
        'CUDA_RNG_byte_metadata_checked_without_GPU_replay':True,'all8_student_and_auxiliary_states_checked':True,'GPU_initialized':False}


def audit(run,output,*,source_root,expected_protocol_sha=None,allowed_output_root=None):
    run,state,protocol,summary,logs,verify,sequence,costs=metadata(run,source_root,expected_protocol_sha)
    output=Path(output).resolve();allowed=Path(allowed_output_root).resolve() if allowed_output_root else WORKSPACE/'outputs'
    need(output!=allowed and output.is_relative_to(allowed) and not output.exists(),'Use a new child of workspace outputs')
    checks=checkpoint_audit(protocol,summary,verify,state['protocol_sha256']);windows=[];bins=[]
    for arm in protocol['study']['arms']:
        label,weight=arm['label'],arm['weight'];rows=logs[label]
        for name,part in [('first256',rows[:256]),('last256',rows[-256:])]:windows.append({'arm':label,'window':name}|summarize(part,weight))
        for name in ('low','mid','high'):
            part=[row for row in rows if row['noise_bin']==name]
            if part:bins.append({'arm':label,'noise_bin':name}|summarize(part,weight))
    report={'schema':'vsr_videomaev2_completed_training_audit_v1','status':'completed_with_diagnostic_flags' if checks['flags'] else 'passed_videomaev2_training_audit',
        'run':str(run),'producer':binding(__file__),'protocol':binding(run/'protocol.json'),'status_file':binding(run/'status.json'),
        'summary':binding(run/'summary.json'),'sources':protocol['sources'],'study_file':protocol['study_file'],
        'scientific_updates':4096,'arms':2,'updates_each':2048,'checkpoint_count':8,'training_seed_repetitions':1,
        'matched_sequence':sequence,'checkpoint_checks':checks,'cost_rows':costs,'containing_training_cost':summary['compute_cost'],
        'step_time_contains_online_teacher':True,'separate_HR_read_time_already_inside_attempt_wall':True,
        'all_inputs_unchanged':True,'CPU_only':True,'Wan_or_teacher_model_reconstructed':False,'head_initialization_only_reconstructed':True,
        'quality_scores_read':False,'new_training_or_restores':0,'public_tensor_or_raw_RNG_bytes':False,
        'large_official_model_files_rehashed':False,'external_teacher_tensor_payload_loaded':False,
        'large_model_provenance_scope':'Base and eight task CPs are independently hashed/loaded; all11 sources/logs/preflight JSON chain checked. Official VideoMAEv2/Wan weights and RGB payload provenance relies on bound successful preflight and core prepare/verify. Teacher CP state is immutable metadata, no86M tensors.',
        'limitations':['Parameter participation and decreasing auxiliaryloss do not prove inference improvement.',
            'Teacher source/weight/position bindings and zero-update counters are preserved; frozen teacher tensors intentionally are not duplicated in CPs.',
            'CUDA RNG states are validated as saved CPU bytevectors without creating CUDA generators; GPU exact replay belongs to preflight.',
            'First/last windows and noise bins use different samples/noise and are descriptive optimization diagnostics only.']}
    verify.unchanged();output.mkdir(parents=True)
    (output/'training_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,rows in [('checkpoint_rng_digests.csv',checks['rng_rows']),('checkpoint_update_evidence.csv',checks['update_rows']),
        ('teacher_checkpoint_evidence.csv',checks['teacher_rows']),('arm_windows.csv',windows),('arm_noise_bins.csv',bins),('arm_costs.csv',costs)]:csv_file(output/name,rows)
    (output/'README.md').write_text('# VideoMAEv2完整训练CPU审计\n\n只在两组各2048、共4096已完成后读取完整训练/8CP。核公共数据/σ/时间步、四显式RNG逐字配对、全部global RNG存在及CPU可验证格式、6tensor MLP与AdamW、源Base、固定教师binding和0参数更新、Step内在线教师与Step外HR读取成本。仅输出摘要/hash，不输出tensor或原RNG。\n\n教师模型未构造/载入，官方大权重与RGB依赖已绑定预飞/原生核验；原head初始化在CPU单独复建以检查更新。无GPU、无新推理、无质量分数。\n')
    return {'status':report['status'],'report':binding(output/'training_audit.json'),'diagnostic_flags':checks['flags'],'output_dir':str(output)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--expected-protocol-sha')
    args=parser.parse_args(argv);need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run completed training audit with CUDA_VISIBLE_DEVICES empty')
    print(json.dumps(audit(args.run,args.output_dir,source_root=args.source_root,expected_protocol_sha=args.expected_protocol_sha),indent=2))


if __name__=='__main__':main()
