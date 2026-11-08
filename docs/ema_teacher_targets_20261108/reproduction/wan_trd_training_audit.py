"""Completed Wan TRD training audit; CPU tensors only, no model or quality scores.

Only status.json may be read before the complete6144 gate. Public reports carry
hashes and counts, never weights, optimizer tensors or raw RNG state bytes.
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
import traceback

ARMS=('baseline','frozen','ema')
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


def metadata(run,source_root,expected_protocol_sha=None):
    run=Path(run).resolve();state=read(run/'status.json')
    need(state.get('status')=='completed' and state.get('scientific_updates')==6144,
         'Reject incomplete training before reading protocol, logs or checkpoints')
    verify=Verifier();verify.verify(binding(run/'status.json'))
    protocol=read(run/'protocol.json');summary=read(run/'summary.json');psha=sha(run/'protocol.json')
    need(expected_protocol_sha is None or psha==expected_protocol_sha,'Unexpected training protocol SHA')
    need(state.get('protocol_sha256')==psha and state.get('summary_sha256')==sha(run/'summary.json'), 'Completion hash chain differs')
    need(protocol.get('schema')=='wan_trd_training_protocol_v1' and protocol.get('scientific_update_budget')==6144
         and summary.get('schema')=='wan_trd_training_summary_v1' and summary.get('status')=='completed'
         and summary.get('protocol_sha256')==psha and summary.get('scientific_updates')==6144
         and summary.get('diagnostic_updates_in_this_training')==0,'Require the exact completed three-arm TRD study')
    for name in ('protocol.json','summary.json'):verify.verify(binding(run/name))
    study=protocol['study'];verify.verify(protocol['study_file']);need(read(protocol['study_file']['path'])==study,'Study file changed')
    need(study.get('schema')=='wan_trd_study_v1' and study.get('mode')=='training'
         and study['updates']==2048 and study['training_seed']==20261102 and study['student_layer']==9
         and study['margin']==.1 and study['normalization_eps']==1e-6
         and study['teacher_noise_ratio']==.5 and study['ema_decay']==.999
         and study['checkpoint_updates']==list(STEPS),'TRD prospective science/budget differs')
    arms=study['arms'];need([a['label'] for a in arms]==list(ARMS) and arms[0]['weight']==0
         and finite(arms[1]['weight']) and arms[1]['weight']>0 and arms[1]['weight']==arms[2]['weight']
         and set(summary['arms'])==set(ARMS),'Require all three arms and one common positive auxiliary weight')
    source_root=Path(source_root).resolve();need(set(protocol['sources'])==CORE,'Require all11 numerical sources')
    for name,item in protocol['sources'].items():
        need(Path(item['path']).resolve()==source_root/'vsr'/name,'Source escaped the declared checkout');verify.verify(item)
    calibration=protocol['calibration']
    need(set(calibration)=={'protocol','summary','status_file'},'Incomplete calibration bindings')
    for item in calibration.values():verify.verify(item)
    calp=read(calibration['protocol']['path']);cal=read(calibration['summary']['path']);calstate=read(calibration['status_file']['path'])
    need(calp['schema']=='wan_trd_calibration_protocol_v1' and calp['sources']==protocol['sources']==cal['sources']
         and cal['status']=='passed_trd_preflight' and cal['scientific_updates']==0 and cal['diagnostic_optimizer_updates']==41
         and cal['protocol_sha256']==calibration['protocol']['sha256']==calstate['protocol_sha256']
         and calstate['status']=='completed' and calstate['summary_sha256']==calibration['summary']['sha256']
         and cal['calibration']['weight']==arms[1]['weight'],'Passed41-update preflight/common weight chain differs')
    verify.verify(protocol['base']['checkpoint']);verify.verify(protocol['base']['config_file'])
    need(read(protocol['base']['config_file']['path'])==protocol['base']['config'],'Base config sidecar changed')
    ids=protocol['cache']['selection']['training_ids'];need(len(ids)==len(set(ids))==1641,'Fixed training cohort differs')
    logs={};common=None;target_sequence=None;initial=None
    fields=('record_id','data_index','sigma','timestep_index','native_timestep')
    for arm in arms:
        label=arm['label'];record=summary['arms'][label];directory=run/'arm_runs'/label
        need(record['updates']==2048 and record['weight']==arm['weight'] and record['teacher_kind']==label
             and record['family']==label and record['setting']=='common'
             and record['EMA_updates']==(2048 if label=='ema' else 0)
             and [cp['step'] for cp in record['checkpoint_files']]==list(STEPS),'Arm identity/count differs')
        for name in ('config.json','initialization.json','completion.json'):verify.verify(binding(directory/name))
        need(read(directory/'completion.json')==record and read(directory/'initialization.json')==record['initialization'],
             'Completion/initialization sidecar differs')
        origin=record['initialization']
        need(origin['checkpoint_sha256']==protocol['base']['checkpoint']['sha256'] and origin['step']==8192
             and origin['variant']=='Base' and all(origin[k] is False for k in
                 ('optimizer_state_restored','scheduler_state_restored','rng_state_restored')),'Arm did not fresh-load the matched Base')
        if initial is None:initial=record['initial_student_hashes']
        need(record['initial_student_hashes']==initial,'Student initial hashes differ')
        if label!='baseline':
            need(record['teacher_initial_hashes']==cal['teacher_initial_hashes'],'Teacher initial hashes differ from preflight')
            if label=='frozen':need(record['teacher_final_hashes']==record['teacher_initial_hashes'],'Frozen teacher changed')
        path=verify.verify(record['full_training_log']);need(path==directory/'training.jsonl','Log escaped its exact arm')
        raw=path.read_bytes();need(raw.endswith(b'\n'),'Incomplete trailing training row')
        rows=[json.loads(line) for line in raw.splitlines()]
        need(len(rows)==2048 and [r['step'] for r in rows]==list(range(1,2049)),'Incomplete/noncontiguous training log')
        seconds=0.
        for row in rows:
            index=(row['step']-1)%1641
            need(row['arm']==label and row['family']==label and row['setting']=='common' and row['weight']==arm['weight']
                 and row['record_id']==ids[index] and row['data_index']==index,'Matched arm/data order differs')
            need(all(finite(row[k]) for k in ('sigma','native_timestep','fm','align','total','grad_norm_before_clip',
                                             'seconds','gpu_event_seconds','cumulative_training_seconds')),
                 'Nonfinite loss/gradient/time/native timestep')
            need(type(row['timestep_index']) is int and 0<=row['timestep_index']<1000 and 0<=row['sigma']<=1
                 and row['grad_norm_before_clip']>=0 and row['seconds']>=0 and row['gpu_event_seconds']>=0,
                 'Invalid timestep/sigma/gradient/time')
            need(row['noise_bin']==('low' if row['sigma']<.33 else 'mid' if row['sigma']<.67 else 'high'), 'Noise-bin contract differs')
            need(row['teacher_kind']==label and row['teacher_forward'] is (label!='baseline')
                 and row['EMA_updates']==(row['step'] if label=='ema' else 0), 'Teacher/update count differs')
            if label!='baseline':
                need(finite(row['teacher_sigma']) and 0<=row['teacher_sigma']<=row['sigma']
                     and type(row['teacher_timestep_index']) is int and 0<=row['teacher_timestep_index']<1000
                     and finite(row['teacher_native_timestep']),'Teacher native input differs')
                relation=row['alignment']
                need(relation['margin']==.1 and relation['normalize_eps']==1e-6 and relation['all_tokens_supervised'] is True
                     and relation['tokens_per_example']==1008 and relation['all_pairs']==1016064
                     and relation['spatial_pairs']==112896 and relation['temporal_pairs']==903168
                     and finite(relation['active_pair_fraction']) and 0<=relation['active_pair_fraction']<=1,
                     'Full native TRD token/grid/reduction differs')
            seconds+=row['seconds'];need(row['cumulative_training_seconds']==seconds,'Committed training time changed')
        need(record['cumulative_training_seconds']==seconds,'Completion committed time differs')
        current=[[r[k] for k in fields] for r in rows]
        if common is None:common=current
        need(current==common,'Public data/noise-time sequence differs across arms')
        if label!='baseline':
            targets=[[r[k] for k in ('teacher_sigma','teacher_timestep_index','teacher_native_timestep')] for r in rows]
            if target_sequence is None:target_sequence=targets
            need(targets==target_sequence,'Frozen/EMA teacher inputs differ')
        for item in record['checkpoint_files']:
            path=verify.verify(item);step=item['step']
            need(path==directory/'Base'/f'checkpoint_{step:06d}.pt' and item['cumulative_training_seconds']==rows[step-1]['cumulative_training_seconds'],
                 'Checkpoint source/time differs')
        logs[label]=rows
    verify.unchanged()
    return run,state,protocol,summary,logs,verify,{'fields':list(fields),'steps':2048,'digest':digest(common),
        'all_three_exactly_equal':True,'teacher_input_sequences_F_E_exactly_equal':True,'teacher_sequence_digest':digest(target_sequence)}


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


def checkpoint_audit(protocol,summary,verify,psha):
    import torch
    need(not torch.cuda.is_initialized(),'CPU audit must start without CUDA')
    need(torch.__version__==protocol['dependencies']['torch']['version'],'CPU PyTorch version differs')
    base=torch.load(protocol['base']['checkpoint']['path'],map_location='cpu',weights_only=True)
    need(base['format']=='vsr_training_v1' and base.get('variant') in (None,'Base') and base['step']==8192
         and base['config']==protocol['base']['config'],'Unexpected Base payload/config')
    base_state=base['trainable_state'];base_desc=parameters(torch,base_state);base_digest=trainable_digest(torch,base_state)
    base_hashes={name:item['sha256'] for name,item in base_desc.items()}
    # Teacher hashes use raw tensor bytes rather than the composite student hash.
    rng_reference={};rng_rows=[];update_rows=[];flags=[];teacher_rows=[]
    for arm in protocol['study']['arms']:
        label=arm['label'];record=summary['arms'][label]
        need(record['initial_student_hashes']['all_trainable']==base_digest,'Initial student is not exact Base state')
        need(record['initialization']['config']==base['config'],'Initialization config is not Base config')
        for item in record['checkpoint_files']:
            value=torch.load(item['path'],map_location='cpu',weights_only=True);step=item['step'];state=value['training_only']
            need(value['format']=='vsr_training_v1' and value['variant']=='Base' and value['step']==step
                 and value['inference_parameters']=='student' and state['schema']=='vsr_trd_training_state_v1'
                 and state['protocol_sha256']==psha and state['arm']==label and state['completed_updates']==step
                 and state['training_seed']==20261102 and state['weight']==arm['weight'] and state['student_layer']==9
                 and state['cumulative_training_seconds']==item['cumulative_training_seconds'],'TRD checkpoint provenance differs')
            need(value['config']==read(Path(item['path']).parent.parent/'config.json')
                 and all(value['config'].get(k)==base['config'].get(k) for k in ('model','conditioning','data','seed')),
                 'Checkpoint inference architecture/data differs')
            need(origin(value['initialization'])==origin(state['matched_Base'])
                 and state['matched_Base']['checkpoint_sha256']==protocol['base']['checkpoint']['sha256']
                 and Path(state['matched_Base']['checkpoint_path']).resolve()==Path(protocol['base']['checkpoint']['path']).resolve(), 'Checkpoint Base origin differs')
            student=value['trainable_state'];desc=parameters(torch,student)
            need(set(desc)==set(base_desc) and all((desc[n]['dtype'],desc[n]['shape'])==(base_desc[n]['dtype'],base_desc[n]['shape']) for n in desc),'Student parameter inventory differs')
            student_hash=trainable_digest(torch,student);changed=sum(desc[n]['sha256']!=base_desc[n]['sha256'] for n in desc)
            if step==2048:need(student_hash==record['final_student_hashes']['all_trainable'],'Final student digest differs')
            need(state['data_cursor']=={'completed_updates':step,'training_clip_count':1641,'next_training_index':step%1641},'Checkpoint data cursor differs')
            need(set(state['explicit_RNG'])==set(state['RNG_devices'])==set(STREAMS),'Incomplete explicit RNG state')
            need(state['RNG_devices']=={'noise':protocol['device'],'timestep':'cpu','mask':'cpu','corruption':protocol['device']},'RNG device declarations differ')
            streams={}
            for name in STREAMS:
                tensor=state['explicit_RNG'][name];need(tensor.dtype==torch.uint8 and tensor.ndim==1,'RNG must be CPU byte vector')
                streams[name]=tensor_description(torch,tensor)
                reference=rng_reference.setdefault(step,{}).setdefault(name,tensor.clone())
                need(torch.equal(tensor,reference),'Bytewise explicit RNG differs across arms: '+name)
            rng_rows.append({'arm':label,'step':step,'all_four_exactly_equal':True,**{n+'_sha256':streams[n]['sha256'] for n in STREAMS}})
            head=state['alignment_state'];spec=state['alignment_spec'];teacher=state['teacher_state'];head_count=head_changed=head_nonzero=0
            optimizer_parameters={'restoration.'+n:t for n,t in student.items()}
            if label=='baseline':need(head is spec is teacher is None,'Baseline acquired a head/teacher')
            else:
                need(spec['class']=='vsr.losses.FullGramTRD' and spec['input_dim']==spec['output_dim']
                     and spec['margin']==.1 and spec['eps']==1e-6 and spec['bias'] is True
                     and spec['relation_reduction']=='all_ordered_token_pairs_mean','Head spec differs')
                head_desc=parameters(torch,head);dimension=spec['input_dim']
                need(set(head)=={'projector.weight','projector.bias'}
                     and head['projector.weight'].shape==(dimension,dimension) and head['projector.bias'].shape==(dimension,), 'Head parameter inventory differs')
                initial={'projector.weight':torch.eye(dimension),'projector.bias':torch.zeros(dimension)}
                head_count=len(head);head_changed=sum(not torch.equal(head[n],initial[n]) for n in head)
                optimizer_parameters.update({'student_projector.'+n.removeprefix('projector.'):t for n,t in head.items()})
                need(teacher['schema']=='vsr_fp32_trainable_teacher_v1' and teacher['mode']==label and teacher['decay']==.999
                     and teacher['num_updates']==(step if label=='ema' else 0)
                     and set(teacher['trainable_names'])==set(teacher['shadow'])==set(base_state),'Teacher shadow identity/count differs')
                teacher_desc=parameters(torch,teacher['shadow'])
                need(all(teacher_desc[n]['shape']==base_desc[n]['shape'] for n in teacher_desc),'Teacher shadow geometry differs')
                teacher_hashes={n:d['sha256'] for n,d in teacher_desc.items()}
                if label=='frozen':need(teacher_hashes==base_hashes,'Frozen teacher has drifted from Base')
                if step==2048:need(teacher_hashes==record['teacher_final_hashes'],'Final teacher digest differs')
                need(record['teacher_initial_hashes']==base_hashes,'Teacher did not initialize exactly from Base')
                teacher_rows.append({'arm':label,'step':step,'teacher_updates':teacher['num_updates'],
                    'teacher_equals_original_Base':teacher_hashes==base_hashes,'teacher_digest':digest(teacher_hashes),
                    'teacher_changed_tensors_from_Base':sum(teacher_hashes[n]!=base_hashes[n] for n in base_hashes)})
            optimizer=state['optimizer'];groups=state['optimizer_parameter_names'];moments={};names=[];identifiers=[]
            need(len(groups)==len(optimizer['param_groups']) and all(g['foreach'] is False for g in optimizer['param_groups']),'AdamW group policy differs')
            for group,group_names in zip(optimizer['param_groups'],groups):
                need(len(group['params'])==len(group_names),'Incomplete optimizer name mapping')
                for identifier,name in zip(group['params'],group_names):
                    moments[name]=optimizer['state'][identifier];names.append(name);identifiers.append(identifier)
            need(len(names)==len(set(names))==len(optimizer_parameters) and set(names)==set(optimizer_parameters)
                 and len(identifiers)==len(set(identifiers))==len(optimizer_parameters) and set(identifiers)==set(optimizer['state']),
                 'Optimizer parameter names/IDs/state are not a complete bijection')
            for name,parameter in optimizer_parameters.items():
                moment=moments[name];counter=moment['step']
                need(set(moment)=={'step','exp_avg','exp_avg_sq'} and isinstance(counter,torch.Tensor)
                     and counter.ndim==0 and bool(torch.isfinite(counter)) and float(counter)==step,'AdamW step count differs')
                for key in ('exp_avg','exp_avg_sq'):
                    tensor=moment[key];need(tensor.device.type=='cpu' and tensor.dtype==torch.float32 and tensor.shape==parameter.shape
                         and not tensor.requires_grad and bool(torch.isfinite(tensor).all()),'Invalid complete AdamW moments')
                    if key=='exp_avg_sq':need(bool((tensor>=0).all()),'Negative AdamW second moment')
                if name.startswith('student_projector.') and bool(moment['exp_avg_sq'].ne(0).any()):head_nonzero+=1
            if not changed:flags.append({'arm':label,'step':step,'flag':'no_student_parameter_change'})
            if label!='baseline' and (head_changed!=2 or head_nonzero!=2):
                flags.append({'arm':label,'step':step,'flag':'incomplete_head_update_or_nonzero_second_moment_evidence',
                              'head_changed_tensors':head_changed,'head_nonzero_second_moment_tensors':head_nonzero})
            update_rows.append({'arm':label,'step':step,'student_tensors':len(desc),'student_changed_tensors':changed,
                'student_digest':student_hash['sha256'],'head_tensors':head_count,'head_changed_tensors':head_changed,
                'head_nonzero_second_moment_tensors':head_nonzero,'all_optimizer_step_counts_match':True})
            del value,state,student,optimizer,moments,optimizer_parameters
    verify.unchanged();need(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    return {'rng_rows':rng_rows,'update_rows':update_rows,'teacher_rows':teacher_rows,'flags':flags,
            'Base_digest':base_digest,'explicit_four_RNG_streams_bytewise_identical_at_each_checkpoint':True,
            'all12_student_and_auxiliary_states_checked':True,'GPU_initialized':False}


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


def audit(run,output,*,source_root,expected_protocol_sha=None,allowed_output_root=None):
    run,state,protocol,summary,logs,verify,sequence=metadata(run,source_root,expected_protocol_sha)
    output=Path(output).resolve();allowed=Path(allowed_output_root).resolve() if allowed_output_root else WORKSPACE/'outputs'
    need(output!=allowed and output.is_relative_to(allowed) and not output.exists(),'Use a new child of workspace outputs/')
    checks=checkpoint_audit(protocol,summary,verify,state['protocol_sha256'])
    windows=[];bins=[]
    for arm in protocol['study']['arms']:
        label,weight=arm['label'],arm['weight'];rows=logs[label]
        for name,part in [('first256',rows[:256]),('last256',rows[-256:])]:windows.append({'arm':label,'window':name}|summarize(part,weight))
        for name in ('low','mid','high'):
            part=[row for row in rows if row['noise_bin']==name]
            if part:bins.append({'arm':label,'noise_bin':name}|summarize(part,weight))
    report={'schema':'wan_trd_completed_training_audit_v1','status':'completed_with_diagnostic_flags' if checks['flags'] else 'passed_wan_trd_training_audit',
        'run':str(run),'producer':binding(__file__),'protocol':binding(run/'protocol.json'),'status_file':binding(run/'status.json'),
        'summary':binding(run/'summary.json'),'sources':protocol['sources'],'study_file':protocol['study_file'],
        'scientific_updates':6144,'arms':3,'updates_each':2048,'checkpoint_count':12,'training_seed_repetitions':1,
        'matched_sequence':sequence,'checkpoint_checks':checks,'all_inputs_unchanged':True,
        'CPU_only':True,'model_reconstruction':False,'quality_scores_read':False,'new_training_or_restores':0,
        'public_tensor_or_raw_RNG_bytes':False,'large_official_model_files_rehashed':False,
        'large_model_provenance_scope':'Official tensor bindings are preserved in the core protocol; this audit independently hashes Base and all12 task checkpoints, sources and calibration chain.',
        'limitations':['Parameter and AdamW participation do not prove useful features or final inference improvements.',
                      'Different student trajectories are expected despite identical sampled data/noise streams.',
                      'Window/noise-bin training summaries are descriptive optimization diagnostics, not causal convergence or quality conclusions.']}
    verify.unchanged();output.mkdir(parents=True)
    (output/'training_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,rows in [('checkpoint_rng_digests.csv',checks['rng_rows']),('checkpoint_update_evidence.csv',checks['update_rows']),
                      ('teacher_checkpoint_evidence.csv',checks['teacher_rows']),('arm_windows.csv',windows),('arm_noise_bins.csv',bins)]:csv_file(output/name,rows)
    (output/'README.md').write_text('# Wan TRD 完整训练审计\n\n核对三组各2048步、共6144步和全部12个检查点。检查源Base、逐行数据与噪声时间、四条显式RNG逐字节配对、学生与投影头更新、AdamW状态，以及冻结教师保持原Base和EMA更新次数。仅输出摘要和digest，不导出权重或原始RNG。\n\n首末窗口和噪声分箱为训练诊断，不是推理质量结论；本工具不读取验证/测试分数，不构建模型，也不运行GPU。完整官方大权重未在此重复hash，其绑定由核心协议负责。\n')
    return {'status':report['status'],'report':binding(output/'training_audit.json'),'diagnostic_flags':checks['flags'],'output_dir':str(output)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--expected-protocol-sha')
    args=parser.parse_args(argv);os.environ['CUDA_VISIBLE_DEVICES']=''
    try:result=audit(args.run,args.output_dir,source_root=args.source_root,expected_protocol_sha=args.expected_protocol_sha)
    except Exception as error:
        print(json.dumps({'status':'rejected_or_failed_wan_trd_training_audit','error':str(error),'traceback':traceback.format_exc(),
                          'GPU_work':False,'automatic_retry':False},indent=2));return 1
    print(json.dumps(result,indent=2));return 0 if result['status']=='passed_wan_trd_training_audit' else 2


if __name__=='__main__':raise SystemExit(main())
