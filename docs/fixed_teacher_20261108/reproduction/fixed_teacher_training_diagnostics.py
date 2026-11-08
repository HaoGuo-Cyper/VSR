"""Independent completed-training diagnostics, CPU only and no teacher model.

An incomplete run is rejected after reading status.json alone. Only an exactly
completed five-arm10240-update study unlocks logs and checkpoint CPU tensors.
Public outputs contain digests and summaries, never RNG bytes or model weights.
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

STREAMS=('noise','timestep','mask','corruption')
STEPS=(512,1024,1536,2048)
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


def exact_binding(value):return {k:value[k] for k in ('path','bytes','sha256')}


class Verifier:
    def __init__(self):self.files={}
    def verify(self,value):
        item=exact_binding(value);path=Path(item['path']).resolve();before=path.stat();signature=(before.st_ino,before.st_size,before.st_mtime_ns)
        need(binding(path)==item,'Changed input: '+str(path));after=path.stat()
        need(signature==(after.st_ino,after.st_size,after.st_mtime_ns),'Input changed while hashed')
        self.files[str(path)]=signature;return path
    def unchanged(self):
        for path,expected in self.files.items():
            s=Path(path).stat();need((s.st_ino,s.st_size,s.st_mtime_ns)==expected,'Input changed during diagnostics: '+path)


def metadata(run,*,source_root=None):
    """The only read allowed before the completion gate is status.json."""
    run=Path(run).resolve();state=read(run/'status.json')
    need(state.get('status')=='completed' and state.get('scientific_updates')==10240,
         'Reject incomplete training before reading protocol, logs, scores or checkpoints')
    verify=Verifier();status_binding=binding(run/'status.json');verify.verify(status_binding)
    protocol=read(run/'protocol.json');summary=read(run/'summary.json')
    need(state.get('protocol_sha256')==sha(run/'protocol.json') and state.get('summary_sha256')==sha(run/'summary.json'), 'Training completion SHA chain differs')
    need(protocol.get('schema')=='fixed_teacher_training_protocol_v1' and protocol.get('scientific_update_budget')==10240
         and summary.get('schema')=='fixed_teacher_training_summary_v1' and summary.get('status')=='completed'
         and summary.get('protocol_sha256')==state['protocol_sha256'] and summary.get('scientific_updates')==10240
         and summary.get('diagnostic_updates_in_this_training')==0,'Require the exact completed fixed-teacher study')
    for name in ('protocol.json','summary.json'):verify.verify(binding(run/name))
    study=protocol['study'];verify.verify(protocol['study_file']);need(read(protocol['study_file']['path'])==study,'Study file/protocol changed')
    need(study.get('schema')=='fixed_teacher_study_v1' and study['updates']==2048 and study['training_seed']==20261102
         and study['checkpoint_updates']==list(STEPS) and study['EMA'] is False and study['condition_corruption'] is False,
         'Unexpected training budget or method setting')
    arms=study['arms'];need(len(arms)==5 and len({a['label'] for a in arms})==5
         and {f:sum(a['family']==f for a in arms) for f in ('baseline','hr_vae','hr_dino')}=={'baseline':1,'hr_vae':2,'hr_dino':2}
         and set(summary['arms'])=={a['label'] for a in arms},'Require all five original trajectories')
    sources=protocol['sources'];need(set(sources)==CORE,'Require the full11-source inventory')
    expected_root=Path(source_root).resolve() if source_root else Path(sources['train.py']['path']).resolve().parent.parent
    for name,item in sources.items():
        need(Path(item['path']).resolve()==expected_root/'vsr'/name,'Source inventory escaped the explicit checkout');verify.verify(item)
    def verify_nested(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}.issubset(value):verify.verify(value)
            else:
                for child in value.values():verify_nested(child)
        elif isinstance(value,list):
            for child in value:verify_nested(child)
    verify_nested(protocol['dependencies'])
    cal=protocol['calibration']
    for item in cal.values():verify.verify(item)
    calstate=read(cal['status_file']['path']);calp=read(cal['protocol']['path']);calreport=read(cal['summary']['path'])
    need(calstate.get('status')=='completed' and calstate.get('protocol_sha256')==cal['protocol']['sha256']
         and calstate.get('summary_sha256')==cal['summary']['sha256'] and calreport.get('status')=='passed_fixed_teacher_preflight'
         and calreport.get('scientific_updates')==0 and calreport.get('diagnostic_optimizer_updates')==27
         and calreport.get('protocol_sha256')==cal['protocol']['sha256'] and calreport.get('sources')==calp.get('sources')==sources,
         'Completed calibration/source chain failed')
    ids=protocol['cache']['selection']['training_ids'];need(len(ids)==len(set(ids))==1641,'Training cohort changed')
    logs={};sequence=None;initial=None
    sequence_fields=('record_id','data_index','sigma','timestep_index','native_timestep')
    for arm in arms:
        label=arm['label'];record=summary['arms'][label];directory=run/'arm_runs'/label
        need(record['updates']==2048 and record['family']==arm['family'] and record['setting']==arm['setting']
             and record['weight']==arm['weight'] and [c['step'] for c in record['checkpoint_files']]==list(STEPS), 'Arm declaration/checkpoint coverage changed')
        verify.verify(binding(directory/'completion.json'));need(read(directory/'completion.json')==record,'Arm completion differs from summary')
        path=verify.verify(record['full_training_log']);need(path==directory/'training.jsonl','Log escaped its exact arm')
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        need(len(rows)==2048 and [r['step'] for r in rows]==list(range(1,2049)),'Log is partial, duplicated or noncontiguous')
        cumulative=0.
        for row in rows:
            index=(row['step']-1)%1641
            need(row['arm']==label and row['family']==arm['family'] and row['setting']==arm['setting'] and row['weight']==arm['weight']
                 and row['record_id']==ids[index] and row['data_index']==index,'Matched data schedule/arm identity differs')
            need(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in
                     ('sigma','timestep_index','native_timestep','fm','align','total','grad_norm_before_clip','seconds','cumulative_training_seconds')),
                 'Nonfinite training metadata/loss/gradient/time')
            need(type(row['timestep_index']) is int and 0<=row['timestep_index']<1000 and 0<=row['sigma']<=1
                 and row['grad_norm_before_clip']>=0 and row['seconds']>=0,'Invalid native timestep/sigma/gradient/time')
            expected_bin='low' if row['sigma']<.33 else 'mid' if row['sigma']<.67 else 'high'
            need(row['noise_bin']==expected_bin,'Original logged noise-bin/sigma contract differs')
            need(row.get('teacher_forward') is False and row.get('EMA_updates')==0,'Unexpected online teacher or EMA update')
            cumulative+=row['seconds'];need(row['cumulative_training_seconds']==cumulative,'Committed step-time prefix changed')
        need(record['cumulative_training_seconds']==cumulative,'Arm committed time differs')
        current=[[r[k] for k in sequence_fields] for r in rows]
        if sequence is None:sequence=current;initial=record['initial_student_hashes']
        need(current==sequence,'Five trajectories do not share the entire2048-step data/sigma/timestep sequence')
        need(record['initial_student_hashes']==initial,'Five trajectories did not share one initial student')
        for cp in record['checkpoint_files']:
            path=verify.verify(cp);need(path==directory/'Base'/f'checkpoint_{cp["step"]:06d}.pt','Checkpoint escaped its exact arm')
            need(cp['cumulative_training_seconds']==rows[cp['step']-1]['cumulative_training_seconds'],'Checkpoint time differs from its logged step')
        logs[label]=rows
    verify.verify(protocol['base']['checkpoint'])
    return run,state,protocol,summary,logs,verify,{'fields':list(sequence_fields),'steps':2048,'digest':digest(sequence),'all_arms_exactly_equal':True}


def tensor_description(torch,tensor):
    need(isinstance(tensor,torch.Tensor) and tensor.device.type=='cpu' and not tensor.requires_grad,'CPU-owned frozen tensor required')
    raw=tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'dtype':str(tensor.dtype),'shape':list(tensor.shape),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def parameter_descriptions(torch,state):
    need(isinstance(state,dict) and state,'Nonempty student/head parameter dictionary required')
    result={}
    for name,tensor in state.items():
        need(isinstance(name,str) and tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all()),'Parameters must be finite FP32')
        result[name]=tensor_description(torch,tensor)
    return result


def model_trainable_digest(torch,state):
    """Exact name/dtype/shape/raw-byte algorithm of frozen model.tensor_state_hashes."""
    all_digest=hashlib.sha256();numel=0
    for name,tensor in sorted(state.items()):
        header=json.dumps({'name':name,'dtype':str(tensor.dtype),'shape':list(tensor.shape)},sort_keys=True,separators=(',',':')).encode()
        raw=tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        for value in (header,raw):all_digest.update(len(value).to_bytes(8,'big'));all_digest.update(value)
        numel+=tensor.numel()
    return {'sha256':all_digest.hexdigest(),'tensor_count':len(state),'numel':numel}


def initial_head(torch,spec,seed):
    """Recreate only one CPU Linear, never a restoration/teacher model."""
    need(spec['class'] in ('vsr.losses.HRLatentRegression','vsr.losses.HRDINOCosineRegression')
         and type(spec['student_dim']) is int and type(spec['target_dim']) is int
         and spec['student_dim']>0 and spec['target_dim']>0,'Unknown training-only head contract')
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed+5).get_state())
        layer=torch.nn.Linear(spec['student_dim'],spec['target_dim'],dtype=torch.float32,device='cpu')
    result={'projector.'+name:value.detach().clone() for name,value in layer.state_dict().items()}
    return result


def origin(value):return {k:value[k] for k in ('checkpoint_path','checkpoint_sha256','step','variant','config')}


def checkpoint_diagnostics(protocol,summary,verify):
    import torch  # Deliberately imported only after the completed10240 metadata gate.
    need(not torch.cuda.is_initialized(),'Diagnostics must run in a CPU-only fresh process')
    need(torch.__version__==protocol['dependencies']['torch']['version'],'CPU diagnostic PyTorch version differs from training')
    base=protocol['base']['checkpoint'];payload=torch.load(base['path'],map_location='cpu',weights_only=True)
    need(payload['format']=='vsr_training_v1' and payload.get('variant') in (None,'Base') and payload['step']==8192
         and Path(base['path']).parent.name=='Base' and Path(base['path']).name=='checkpoint_008192.pt'
         and payload['config']['training']['variants']==['Base'] and payload['config']['training']['updates']==8192,
         'Unexpected original Base payload/path/config (legacy variant may be absent)')
    base_state=payload['trainable_state'];base_params=parameter_descriptions(torch,base_state);base_digest=model_trainable_digest(torch,base_state)
    reference_rng={};rng_rows=[];update_rows=[];head_initials={};head_specs={};flags=[]
    for arm in protocol['study']['arms']:
        record=summary['arms'][arm['label']]
        need(record['initial_student_hashes']['all_trainable']==base_digest,'Declared initial student differs from exact source Base parameters')
        for cp in record['checkpoint_files']:
            value=torch.load(cp['path'],map_location='cpu',weights_only=True);step=cp['step'];state=value['training_only']
            need(value['format']=='vsr_training_v1' and value['variant']=='Base' and value['step']==step
                 and value['inference_parameters']=='student' and state['schema']=='vsr_hr_latent_training_state_v1'
                 and state['protocol_sha256']==protocol['_diagnostic_protocol_sha256'] and state['completed_updates']==step
                 and state['training_seed']==20261102 and state['weight']==arm['weight'] and state['student_layer']==7
                 and state['cumulative_training_seconds']==cp['cumulative_training_seconds'], 'Checkpoint same-study/resume metadata differs')
            expected_kind='dino' if arm['family']=='hr_dino' else 'vae'
            need(state['teacher_kind']==expected_kind and state['restored_by_generic_inference_loader'] is False
                 and value['config']['model']==payload['config']['model']
                 and value['config'].get('conditioning')==payload['config'].get('conditioning')
                 and value['config']['data']==payload['config']['data'],'Checkpoint teacher/inference/config contract differs')
            need(origin(value['initialization'])==origin(state['matched_Base'])
                 and state['matched_Base']['checkpoint_sha256']==base['sha256']
                 and Path(state['matched_Base']['checkpoint_path']).resolve()==Path(base['path']).resolve(), 'Checkpoint matched Base origin changed')
            need(set(state['explicit_RNG'])==set(state['RNG_devices'])==set(STREAMS),'Missing explicit RNG stream')
            devices={'noise':protocol['device'],'timestep':'cpu','mask':'cpu','corruption':protocol['device']}
            need(state['RNG_devices']==devices,'Explicit RNG device policy differs')
            streams={}
            for name in STREAMS:
                tensor=state['explicit_RNG'][name]
                need(tensor.dtype==torch.uint8 and tensor.ndim==1,'RNG must be an owned CPU byte vector')
                streams[name]=tensor_description(torch,tensor)
                if step not in reference_rng:reference_rng[step]={}
                if name not in reference_rng[step]:reference_rng[step][name]=tensor.clone()
                need(torch.equal(tensor,reference_rng[step][name]),'Checkpoint RNG stream differs across matched arms: '+name)
            rng_rows.append({'arm':arm['label'],'step':step,'all_four_streams_match_reference':True,
                             **{name+'_sha256':streams[name]['sha256'] for name in STREAMS},'stream_descriptions':streams})
            student=value['trainable_state'];params=parameter_descriptions(torch,student)
            need(params.keys()==base_params.keys() and all((params[n]['dtype'],params[n]['shape'])==(base_params[n]['dtype'],base_params[n]['shape']) for n in params),
                 'Student parameter name/dtype/shape inventory differs from original Base')
            student_digest=model_trainable_digest(torch,student);changed=sum(params[n]['sha256']!=base_params[n]['sha256'] for n in params)
            if step==2048:need(student_digest==record['final_student_hashes']['all_trainable'],'Final student digest differs from completion record')
            head=state['alignment_state'];head_changed=head_count=0;head_digest=head_initial_digest=None
            parameters={'restoration.'+n:t for n,t in student.items()}
            if arm['family']=='baseline':need(head is None and state['alignment_spec'] is None,'Baseline must have no auxiliary head')
            else:
                expected_class='vsr.losses.HRDINOCosineRegression' if expected_kind=='dino' else 'vsr.losses.HRLatentRegression'
                need(state['alignment_spec']['class']==expected_class,'Training-only head class differs from its teacher family')
                descriptions=parameter_descriptions(torch,head)
                if arm['label'] not in head_initials:
                    head_specs[arm['label']]=state['alignment_spec'];head_initials[arm['label']]=initial_head(torch,state['alignment_spec'],20261102)
                need(state['alignment_spec']==head_specs[arm['label']],'Training-only head spec changed across checkpoints')
                initial=parameter_descriptions(torch,head_initials[arm['label']])
                need(descriptions.keys()==initial.keys() and all((descriptions[n]['dtype'],descriptions[n]['shape'])==(initial[n]['dtype'],initial[n]['shape']) for n in initial),
                     'Training-only head name/dtype/shape differs from seeded initialization')
                head_count=len(descriptions);head_changed=sum(descriptions[n]['sha256']!=initial[n]['sha256'] for n in initial)
                head_digest=digest(descriptions);head_initial_digest=digest(initial)
                parameters.update({'student_projector.'+n.removeprefix('projector.'):t for n,t in head.items()})
            opt=state['optimizer'];groups=state['optimizer_parameter_names']
            need(len(groups)==len(opt['param_groups']) and all(g['foreach'] is False for g in opt['param_groups']),'AdamW group policy differs')
            moments={};names=[];identifiers=[]
            for group,group_names in zip(opt['param_groups'],groups):
                need(len(group['params'])==len(group_names),'Optimizer parameter mapping incomplete')
                for identifier,name in zip(group['params'],group_names):
                    moments[name]=opt['state'][identifier];names.append(name);identifiers.append(identifier)
            need(len(names)==len(set(names))==len(parameters) and set(names)==set(parameters)
                 and len(identifiers)==len(set(identifiers))==len(parameters) and set(identifiers)==set(opt['state']),
                 'Incomplete optimizer parameter-ID/name/state bijection')
            head_moments_nonzero=0
            for name,parameter in parameters.items():
                moment=moments[name]
                counter=moment['step'];need(isinstance(counter,torch.Tensor) and counter.device.type=='cpu' and counter.ndim==0
                     and not counter.requires_grad and bool(torch.isfinite(counter)) and float(counter)==step,'AdamW parameter step counter differs')
                for key in ('exp_avg','exp_avg_sq'):
                    tensor=moment[key];need(tensor.device.type=='cpu' and tensor.dtype==torch.float32 and tensor.shape==parameter.shape
                                            and not tensor.requires_grad and bool(torch.isfinite(tensor).all()),'Invalid complete AdamW moments')
                    if key=='exp_avg_sq':need(bool((tensor>=0).all()),'Negative AdamW second moment')
                if name.startswith('student_projector.') and bool(moment['exp_avg_sq'].ne(0).any()):head_moments_nonzero+=1
            if head_count and head_moments_nonzero==0:flags.append({'arm':arm['label'],'step':step,'flag':'no_nonzero_head_second_moment_gradient_evidence; parameter changes alone may be weight_decay'})
            if not changed:flags.append({'arm':arm['label'],'step':step,'flag':'no_student_parameter_bytes_changed_from_Base'})
            if head_count and not head_changed:flags.append({'arm':arm['label'],'step':step,'flag':'no_head_parameter_bytes_changed_from_seeded_initial'})
            update_rows.append({'arm':arm['label'],'family':arm['family'],'weight':arm['weight'],'step':step,
                'student_parameter_tensors':len(params),'student_changed_from_Base_tensors':changed,'student_all_trainable_sha256':student_digest['sha256'],
                'initial_Base_all_trainable_sha256':base_digest['sha256'],'head_parameter_tensors':head_count,
                'head_changed_from_seeded_initial_tensors':head_changed,'head_sha256':head_digest,'head_initial_sha256':head_initial_digest,
                'head_nonzero_second_moment_tensors':head_moments_nonzero,'all_optimizer_parameter_step_counters_equal_checkpoint':True})
            del value,state,student,opt,moments,parameters
    verify.unchanged();need(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    return {'rng_rows':rng_rows,'update_rows':update_rows,'flags':flags,'Base_digest':base_digest,
            'four_streams_bytewise_equal_across_all_arms_at_each_checkpoint':True,'GPU_initialized':False,
            'head_initialization':'Only CPU Linear(student_dim,target_dim), same isolated training_seed+5 initialization; no Wan/VAE/DINO model constructed'}


def summarize(rows,weight):
    count=len(rows);mean=lambda key:statistics.mean(r[key] for r in rows) if count else None
    fm=mean('fm');align=mean('align')
    return {'count':count,'fm_mean':fm,'align_mean':align,'weighted_aux_mean':None if align is None else weight*align,
        'total_mean':mean('total'),'grad_norm_before_clip_mean':mean('grad_norm_before_clip'),
        'grad_norm_before_clip_max':max((r['grad_norm_before_clip'] for r in rows),default=None),
        'unweighted_aux_over_FM_ratio_of_means':align/fm if fm else None,
        'weighted_aux_over_FM_ratio_of_means':weight*align/fm if fm else None,
        'sigma_min':min((r['sigma'] for r in rows),default=None),'sigma_max':max((r['sigma'] for r in rows),default=None)}


def csv_file(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row if not isinstance(row[key],(dict,list))))
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def diagnose(run,output,*,source_root=None,allowed_output_root=None):
    run,state,protocol,summary,logs,verify,sequence=metadata(run,source_root=source_root)
    output=Path(output).resolve();allowed=Path(allowed_output_root).resolve() if allowed_output_root else WORKSPACE/'outputs'
    need(output!=allowed and output.is_relative_to(allowed) and not output.exists(),'Use a fresh child of the original workspace outputs/')
    protocol=protocol|{'_diagnostic_protocol_sha256':state['protocol_sha256']}
    checks=checkpoint_diagnostics(protocol,summary,verify)
    windows=[];bins=[];by_arm={}
    for arm in protocol['study']['arms']:
        label=arm['label'];rows=logs[label];prefix={k:arm[k] for k in ('label','family','setting','weight')}
        window={name:summarize(part,arm['weight']) for name,part in (('first256',rows[:256]),('last256',rows[-256:]))}
        noise={name:summarize([r for r in rows if r['noise_bin']==name],arm['weight']) for name in ('low','mid','high')}
        for name,stats in window.items():windows.append(prefix|{'window':name}|stats)
        for name,stats in noise.items():bins.append(prefix|{'noise_bin':name}|stats)
        by_arm[label]={'family':arm['family'],'setting':arm['setting'],'weight':arm['weight'],'windows':window,'noise_bins':noise,
                       'max_grad_norm_before_clip':max(r['grad_norm_before_clip'] for r in rows),'sequence_digest':sequence['digest']}
    report={'schema':'fixed_teacher_completed_training_diagnostics_v1','status':'completed_with_diagnostic_flags' if checks['flags'] else 'passed_completed_training_diagnostics',
        'run':str(run),'producer':binding(__file__),'protocol':binding(run/'protocol.json'),'status_file':binding(run/'status.json'),
        'summary':binding(run/'summary.json'),'sources':protocol['sources'],'study_file':protocol['study_file'],
        'scientific_updates':10240,'training_arms':5,'updates_each':2048,'checkpoint_count':20,'training_seed_repetitions':1,
        'matched_sequence':sequence,'initial_student_hash_identical':True,'checkpoint_checks':checks,'arms':by_arm,
        'noise_bin_contract':'Original log label, independently checked: sigma<.33 low; .33<=sigma<.67 mid; sigma>=.67 high',
        'aux_over_FM_contract':'Both unit-aux/FΜ and loss-weighted-aux/FM ratios of means; null if mean FM is zero. VAE SmoothL1 and DINO cosine units differ.',
        'public_raw_RNG_or_weight_bytes':False,'restoration_or_teacher_model_reconstruction':False,'training_only_Linear_head_recreated_on_CPU':True,'GPU_work':False,'new_training_or_restoration_updates':0,
        'quality_conclusion':None,'limitations':['Training-loss/gradient summaries are optimization diagnostics, not super-resolution quality or evidence for choosing method weights.',
            'One trained seed; identical explicit RNG and timestep schedules do not mean student features/gradients remain identical after different auxiliary losses.',
            'Changed parameter byte counts and complete AdamW moments prove update participation, not useful representation learning or improved held-out quality.',
            'No validation/test predictions or quality scores are read by this helper.']}
    verify.unchanged();output.mkdir(parents=True)
    (output/'training_diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    csv_file(output/'arm_windows.csv',windows);csv_file(output/'arm_noise_bins.csv',bins)
    csv_file(output/'checkpoint_rng_digests.csv',checks['rng_rows']);csv_file(output/'checkpoint_update_evidence.csv',checks['update_rows'])
    note='''# 完整训练诊断\n\n已核对5条轨迹各2048步、共10240步及20个checkpoint；完整数据/σ/timestep序列与四条显式RNG在相同checkpoint逐字节一致。这里只公开digest，不导出RNG状态或权重。\n\narm_windows.csv列出首/末256步FM、辅助损失、总损失及裁剪前梯度；arm_noise_bins.csv按原σ分箱汇总。aux/FM同时列单位辅助量与乘权重后的比值，不能跨VAE和DINO直接当作同一物理量。checkpoint_update_evidence.csv以源Base参数及隔离seed+5的单层CPU投影头初始化为参照。没有构建Wan、VAE或DINO teacher，没有GPU调用。\n\n这些是优化与公平性检查，不能据此宣称超分效果变好或选参。质量结论仍依赖前瞻冻结的完整验证/测试结果；单训练种子的局限保持不变。\n'''
    (output/'README.md').write_text(note)
    return {'status':report['status'],'output_dir':str(output),'report':binding(output/'training_diagnostics.json'),'diagnostic_flags':checks['flags']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source-root',type=Path);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:result=diagnose(args.run,args.output_dir,source_root=args.source_root)
    except Exception as error:
        print(json.dumps({'status':'rejected_or_failed_completed_training_diagnostics','error':str(error),'traceback':traceback.format_exc(),
                          'GPU_work':False,'automatic_retry':False},indent=2));return 1
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
