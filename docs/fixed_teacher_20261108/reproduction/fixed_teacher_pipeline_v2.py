"""Serial fixed-teacher handoff controller, outside the frozen core.

Default invocation only inspects the planned inputs. Explicit --execute starts
one synchronous child at a time. A failed command or metadata review stops the
pipeline and preserves its logs; this controller has no retry/resume policy.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

CORE_NAMES={'__init__.py','data.py','vae.py','model.py','teacher.py','losses.py','train.py','infer.py','evaluate.py','metrics.py','checkpoints.py'}
HELPERS=('fixed_teacher_pipeline_v2.py','fixed_teacher_eval_configs.py','fixed_teacher_data.py',
         'analyze_fixed_teacher_study.py','analyze_hr_vae_study.py')
OFFLINE={'CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONUNBUFFERED':'1','PYTHONHASHSEED':'0','PYTORCH_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MPLBACKEND':'Agg','XFORMERS_DISABLED':'1'}


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def binding(path):
    path=Path(path).resolve();need(path.is_file(),'Missing bound input: '+str(path))
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def verify(item):
    need(binding(item['path'])==item,'Bound input changed: '+item['path'])


def write(path,value,*,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not replace:
        with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        return
    fd,temp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:Path(temp).unlink(missing_ok=True)


def now():return datetime.now(timezone.utc).isoformat()


def bound(value):
    """Canonical file binding; retain provenance only in its original config."""
    if not isinstance(value,dict):return binding(value)
    need({'path','sha256'}.issubset(value),'Explicit file path/SHA binding required')
    actual=binding(value['path'])
    need(actual['sha256']==value['sha256'],'Bound file SHA changed: '+str(value['path']))
    need('bytes' not in value or value['bytes']==actual['bytes'],'Bound file size changed: '+str(value['path']))
    return actual


@dataclass(frozen=True)
class Options:
    source_root:Path
    study:Path
    training_run:Path
    run_root:Path
    python:Path
    output_dir:Path
    dependency_root:Path|None=None
    after_training:bool=True
    helper_root:Path=Path(__file__).resolve().parent


def completed_training(run,protocol):
    run=Path(run);state=read(run/'status.json');summary=read(run/'summary.json')
    need(state.get('status')=='completed' and state.get('scientific_updates')==10240
         and state.get('protocol_sha256')==sha(run/'protocol.json') and state.get('summary_sha256')==sha(run/'summary.json')
         and summary.get('schema')=='fixed_teacher_training_summary_v1' and summary.get('protocol_sha256')==sha(run/'protocol.json')
         and summary.get('status')=='completed' and summary.get('scientific_updates')==10240
         and summary.get('diagnostic_updates_in_this_training')==0,'Training must complete exactly10240 committed scientific updates')
    need(set(summary['arms'])=={a['label'] for a in protocol['study']['arms']},'Training lost a declared trajectory')
    initial=[];row_count=0;cp_count=0
    for arm in protocol['study']['arms']:
        record=summary['arms'][arm['label']]
        need(record['updates']==2048 and record['family']==arm['family'] and record['setting']==arm['setting'] and record['weight']==arm['weight']
             and [c['step'] for c in record['checkpoint_files']]==[512,1024,1536,2048],'Training checkpoint budget/family changed')
        directory=run/'arm_runs'/arm['label'];need(read(directory/'completion.json')==record,'Arm completion and summary differ')
        initial.append(record['initial_student_hashes'])
        log=record['full_training_log'];verify(log);need(Path(log['path']).resolve()==directory/'training.jsonl','Training log escaped its exact arm')
        rows=[json.loads(line) for line in Path(log['path']).read_text().splitlines() if line.strip()]
        need(len(rows)==2048 and [r['step'] for r in rows]==list(range(1,2049)),'Training log is partial/duplicate/noncontiguous')
        cumulative=0.;training_ids=protocol['cache']['selection']['training_ids']
        need(len(training_ids)==1641 and len(set(training_ids))==1641,'Expected unchanged1641 training records')
        for row in rows:
            index=(row['step']-1)%1641
            need(row['data_index']==index and row['record_id']==training_ids[index],'Training data order differs from the matched schedule')
            need(row['arm']==arm['label'] and row['family']==arm['family'] and row['weight']==arm['weight']
                 and all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('fm','align','total','grad_norm_before_clip','sigma','seconds','cumulative_training_seconds')),
                 'Training row identity or finite loss/gradient/time failed')
            need(row['seconds']>=0,'Negative synchronized step time')
            cumulative+=row['seconds'];need(row['cumulative_training_seconds']==cumulative,'Committed cumulative step time differs from its log prefix')
        need(record['cumulative_training_seconds']==cumulative,'Arm training-time total changed')
        row_count+=len(rows)
        for cp in record['checkpoint_files']:
            path=Path(cp['path']).resolve();need(path==directory/'Base'/f'checkpoint_{cp["step"]:06d}.pt' and path.parent.name=='Base'
                 and path.name==f'checkpoint_{cp["step"]:06d}.pt','Final training checkpoint escaped its declared student path')
            need(cp['cumulative_training_seconds']==rows[cp['step']-1]['cumulative_training_seconds'],'Checkpoint time differs from the committed log prefix')
            verify({k:cp[k] for k in ('path','bytes','sha256')});cp_count+=1
    need(all(v==initial[0] for v in initial),'Trajectories did not share the same initial student')
    return {'status':binding(run/'status.json'),'summary':binding(run/'summary.json'),'scientific_updates':10240,
            'finite_contiguous_log_rows':row_count,'checkpoint_SHA_verified':cp_count,'same_initial_student':True}


def inspect_inputs(options):
    o=options;source=o.source_root.resolve();root=o.run_root.resolve();train=o.training_run.resolve();helpers=o.helper_root.resolve()
    outputs=helpers.parent/'outputs';output=o.output_dir.resolve()
    need(source.is_dir() and root.is_dir() and train!=root and train.is_relative_to(root),'Explicit training/run/source paths required')
    need(output!=outputs and output.is_relative_to(outputs),'Deliveries must be a new child of this workspace outputs/')
    need(o.python.is_absolute() and o.python.is_file() and os.access(o.python,os.X_OK),'Explicit executable Python required')
    sources={p.name:binding(p) for p in sorted((source/'vsr').glob('*.py'))}
    need(set(sources)==CORE_NAMES,'Require exactly the frozen11 runtime sources')
    helper_bindings={name:binding(helpers/name) for name in HELPERS}
    study=read(o.study);protocol=read(train/'protocol.json');state=read(train/'status.json')
    need(study.get('schema')=='fixed_teacher_study_v1','Final calibrated fixed_teacher_study_v1 required')
    need(protocol.get('schema')=='fixed_teacher_training_protocol_v1' and protocol.get('study')==study
         and protocol.get('study_file')==binding(o.study),'Planned training is not bound to this final study')
    need(protocol.get('sources')==sources,'Planned runtime sources differ from the requested checkout')
    need(state.get('protocol_sha256')==sha(train/'protocol.json'),'Training protocol/status SHA differs')
    if o.after_training:
        need(state.get('status')=='completed','After-training mode refuses planned/running/failed training; no training child is started')
    else:
        need(state.get('status')=='planned' and state.get('scientific_updates')==0 and not (train/'failure.json').exists()
             and not list(train.glob('attempt_*.json')) and not (train/'arm_runs').exists(),'Include-training requires a never-started planned run')
    need(Path(study['run_root']).resolve()==root,'Run root differs from the frozen study')
    arms=study['arms'];need(len(arms)==5 and len({a['label'] for a in arms})==5,'Retain exactly5 prospective trajectories')
    need({f:sum(a['family']==f for a in arms) for f in ('baseline','hr_vae','hr_dino')}=={'baseline':1,'hr_vae':2,'hr_dino':2},'Family budget changed')
    need(study['updates']==2048 and study['training_seed']==20261102 and study['student_layer']==7
         and study['eval_updates']==[512,1024,2048] and study['checkpoint_updates']==[512,1024,1536,2048]
         and study['head_depth']==1 and study['smooth_l1_beta']==1. and protocol['scientific_update_budget']==10240,
         'Fixed training/update/checkpoint budget changed')
    need(all(isinstance(a['weight'],(int,float)) and not isinstance(a['weight'],bool) and math.isfinite(a['weight'])
             and (a['weight']==0 if a['family']=='baseline' else a['weight']>0) for a in arms),'Invalid calibrated loss weights')
    need(study['EMA'] is False and study['condition_corruption'] is False,'Round1 must keep fixed clean HR teachers without EMA or corruption')
    need(protocol['ordinary_inference_weight_EMA'] is False and protocol['memory_cap_bytes']==16*1024**3,'Inference EMA or16GiB cap changed')
    need(study['device']==protocol['device']=='cuda:0' and study['gpu_lock']==protocol['gpu_lock']
         and Path(study['gpu_lock']).is_absolute() and study['gpu_uuid']==protocol['admission']['expected_gpu_uuid'],
         'GPU/device/shared-lock binding changed')
    calibration=protocol['calibration']
    for item in calibration.values():verify(item)
    calp=read(calibration['protocol']['path']);cal=read(calibration['summary']['path']);calstate=read(calibration['status_file']['path'])
    need(calstate.get('status')=='completed' and calstate.get('protocol_sha256')==calibration['protocol']['sha256']
         and calstate.get('summary_sha256')==calibration['summary']['sha256']
         and cal.get('status')=='passed_fixed_teacher_preflight' and cal.get('scientific_updates')==0
         and cal.get('diagnostic_optimizer_updates')==27 and cal.get('protocol_sha256')==calibration['protocol']['sha256']
         and cal.get('sources')==calp.get('sources')==sources
         and not (Path(calibration['summary']['path']).parent/'failure.json').exists(),
         'Calibration must pass through cleanup with the exact27 diagnostic/source/status chain')
    need(Path(study['calibration_report']).resolve()==Path(calibration['summary']['path']),'Study changed calibrated report')
    expected_arms=[{'label':'baseline','family':'baseline','setting':'baseline','weight':0.}]
    for family,short,kind in (('hr_vae','vae','vae'),('hr_dino','dino','dino')):
        expected_arms += [{'label':short+'_'+strength,'family':family,'setting':strength,'weight':weight}
                          for strength,weight in zip(('low','high'),cal['calibrations'][kind]['weights'])]
    need(arms==expected_arms,'Study loss weights are not the passed train-gradient calibration candidates')
    if o.after_training:completed_training(train,protocol)
    evaluation=study['evaluation'];need(evaluation['test_pixels_after_selection'] is True
         and evaluation['test_prior_exposure']=='historical_models_previously_evaluated','Test access/history policy changed')
    immutable={'study':binding(o.study),'training_protocol':binding(train/'protocol.json'),'python':binding(o.python)}
    immutable.update({'calibration_'+k:v for k,v in calibration.items()})
    if o.after_training:immutable.update(training_completed_status=binding(train/'status.json'),training_summary=binding(train/'summary.json'))
    for key in ('validation_seal','validation_dataset','test_cohort_plan'):
        immutable[key]=bound(evaluation[key]);verify(immutable[key])
    exception=evaluation.get('test_integrity_thumbnail_exception')
    if exception is not None:
        need(exception.get('scope')=='integrity_only' and exception.get('automated') is True
             and exception.get('human_quality_review') is False and exception.get('model_scores_used') is False,'Integrity exception changed')
        immutable['integrity_audit']=bound(exception['audit_receipt']);verify(immutable['integrity_audit'])
    for key in ('alexnet','linear'):
        immutable['metric_'+key]=bound(study['metric_weights'][key]);verify(immutable['metric_'+key])
    pipeline=root/('pipeline_'+train.name)
    need(not pipeline.exists() and not output.exists(),'Pipeline/output directories are immutable; existing paths refuse duplicate work')
    need(not pipeline.is_relative_to(source) and not pipeline.is_relative_to(train)
         and not output.is_relative_to(source),'Pipeline/delivery paths cannot overlap runtime checkout or training inputs')
    if o.dependency_root is not None:need(o.dependency_root.resolve().is_dir(),'Explicit dependency root missing')
    return {'schema':'fixed_teacher_serial_pipeline_v1','created_at_UTC':now(),'source_root':str(source),
        'training_run':str(train),'mode':'after_training' if o.after_training else 'include_training','pipeline_dir':str(pipeline),'run_root':str(root),'output_dir':str(output),
        'sources':sources,'helpers':helper_bindings,'immutable_inputs':immutable,'study':study,
        'initial_training_status':binding(train/'status.json'),'python':str(o.python.resolve()),
        'dependency_root':str(o.dependency_root.resolve()) if o.dependency_root else None,
        'budget':{'scientific_updates':10240,'scientific_RE':1.25,'validation_roles':16,'validation_restores':1024,
                  'test_roles':3,'test_restores':192,'evaluation_training_updates':0},
        'no_automatic_retry':True,'no_automatic_scientific_adjustment':True,'test_formal_RGB_after_selection':True,
        'final_delivery_status':'awaiting_root_delivery'}


class SubprocessRunner:
    def run(self,command,*,cwd,env,log):
        with Path(log).open('xb') as stream:
            return subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode


class Pipeline:
    def __init__(self,options,runner=None):self.options=options;self.runner=runner or SubprocessRunner();self.plan=None;self.command=None;self.exit_code=None;self.phase='initial_input_review';self.generated_bindings={}

    def frozen(self):
        for section in ('sources','helpers','immutable_inputs'):
            for item in self.plan[section].values():verify(item)
        for item in self.generated_bindings.values():verify(item)

    def register(self,value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}.issubset(value):
                item={k:value[k] for k in ('path','bytes','sha256')};verify(item)
                previous=self.generated_bindings.get(item['path'])
                need(previous is None or previous==item,'Sealed pipeline output was rebound')
                self.generated_bindings[item['path']]=item
            else:
                for child in value.values():self.register(child)
        elif isinstance(value,list):
            for child in value:self.register(child)

    def reviewed_sha(self,path):
        path=str(Path(path).resolve());need(path in self.generated_bindings,'Protocol was not reviewed before execute')
        verify(self.generated_bindings[path]);return self.generated_bindings[path]['sha256']

    def event(self,event,**fields):
        record={'at_UTC':now(),'phase':self.phase,'event':event,**fields}
        with (self.root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

    def status(self,status,**fields):write(self.root/'status.json',{'status':status,'phase':self.phase,'at_UTC':now(),**fields},replace=True)

    def child(self,phase,command,review=None):
        self.phase=phase;self.command=list(map(str,command));self.exit_code=None;self.frozen()
        self.event('starting',command=self.command);self.status('running',command=self.command)
        started=time.perf_counter()
        code=self.runner.run(self.command,cwd=self.plan['source_root'],env=dict(self.environment),log=self.root/'logs'/f'{phase}.log')
        seconds=time.perf_counter()-started;self.exit_code=code
        self.event('child_exited',exit_code=code,subprocess_wall_seconds=seconds)
        need(code==0,'Child failed with exit '+str(code))
        self.frozen();evidence=review() if review else {'exit_code':0}
        if phase in ('04_val_execute','10_test_execute'):
            which='val' if phase=='04_val_execute' else 'test';run=self.root/('validation' if which=='val' else 'test')
            timing={'schema':'fixed_teacher_execution_wall_v1','status':'completed','stage':'evaluation_'+which,
                    'run':str(run),'protocol_sha256':sha(run/'protocol.json'),'wall_seconds':seconds,
                    'total_wall_seconds':seconds,'attempts':[{'attempt':0,'status':'completed','wall_seconds':seconds}],
                    'scope':'Monotonic child process start through normal exit; includes execute model loading, restore, metrics, PNG/MP4 and cleanup; excludes prepare/verify; not pure GPU kernel time'}
            write(run/'execution_wall_time.json',timing);evidence=evidence|{'execution_wall_time':binding(run/'execution_wall_time.json')}
        self.register(evidence)
        write(self.root/'phases'/f'{phase}.json',{'status':'completed','command':self.command,'evidence':evidence,'at_UTC':now()})
        self.event('completed');self.status('phase_completed')
        return evidence

    def review_training(self):
        train=Path(self.plan['training_run']);return completed_training(train,read(train/'protocol.json'))

    def review_config(self,phase):
        config=read(self.root/f'{phase}_config.json');study=self.plan['study'];training=read(Path(self.plan['training_run'])/'protocol.json')
        count=16 if phase=='val' else 3
        need(config.get('schema')=='fixed_teacher_evaluation_config_v1' and config.get('phase')==phase
             and config.get('study_config')==self.plan['immutable_inputs']['study']
             and Path(config['training_run']).resolve()==Path(self.plan['training_run'])
             and Path(config['run_root']).resolve()==self.root,'Builder source/study/run scope changed')
        need(len(config['roles'])==count and config['device']==training['device'] and config['gpu_lock']==training['gpu_lock']
             and config['gpu_admission']==training['admission'],'Evaluation roles/device/lock changed')
        target='validation_seal' if phase=='val' else 'test_cohort_plan'
        need(binding(config['seal'])==self.plan['immutable_inputs'][target],'Evaluation cohort changed')
        if phase=='val':need(binding(config['dataset'])==self.plan['immutable_inputs']['validation_dataset'],'Validation dataset changed')
        else:
            need(config['selection_receipt']==binding(self.root/'selection.json')
                 and binding(config['dataset'])==binding(self.root/'test_assets/dataset.json'),'Test receipt/assets changed')
        need(Path(config['model_directory']).resolve()==Path(study['model_directory']).resolve()
             and config['metric_weights']=={k:bound(v) for k,v in study['metric_weights'].items()},'Official model or metrics changed')
        return {'config':binding(self.root/f'{phase}_config.json'),'expected_roles':count,'expected_restores':count*64}

    def review_protocol(self,phase):
        run=self.root/('validation' if phase=='val' else 'test');p=read(run/'protocol.json');state=read(run/'status.json')
        t=read(Path(self.plan['training_run'])/'protocol.json');scope=p['scope'];count=16 if phase=='val' else 3
        need(p['schema']=='fixed_teacher_clean_evaluation_protocol_v1' and p['phase']==phase
             and p['sources']==self.plan['sources'] and p['study_file']==self.plan['immutable_inputs']['study'],
             'Prepared evaluation source/study/phase changed')
        need(state.get('status')=='planned' and state.get('restores')==0 and state.get('training_updates')==0
             and state.get('protocol_sha256')==sha(run/'protocol.json'),'Prepared evaluation was already started')
        expected={'roles':count,'clips':32,'sampling_seeds':[42,43],'conditions':['clean'],'total_restores':count*64,
            'source_split':phase,'frames':33,'HR_hw':[128,224],'LR_hw':[32,56],'steps':20,'CFG':1.,
            'tile_hw':[256,448],'overlap':64,'primary_pool':'all33_frames','primary_metric':'lpips_alex_v0_1',
            'video_policy':'all_clips','training_updates':0}
        need(all(scope.get(k)==v for k,v in expected.items()),'Prepared sampling/metric/geometry/budget changed')
        need(len(p['roles'])==count and len({r['label'] for r in p['roles']})==count,'Prepared roles are missing/duplicated')
        need(len(p['cohort'])==len({i['record']['id'] for i in p['cohort']})==len({i['record']['source_id'] for i in p['cohort']})==32,
             'Prepared cohort must retain all32 distinct sources')
        need(all(i['record']['split']==phase and len(i['source_indices'])==33
                 and i['source_indices']==list(range(i['source_indices'][0],i['source_indices'][0]+33)) for i in p['cohort']),
             'Prepared source windows/splits changed')
        target='validation_seal' if phase=='val' else 'test_cohort_plan'
        sealed=read(self.plan['immutable_inputs'][target]['path'])['cohort']
        identity=lambda items:[(x['record']['id'],x['record']['source_id'],x['record']['split'],x['source_indices']) for x in items]
        need(identity(p['cohort'])==identity(sealed),'Prepared cohort does not match exact source/window order in its seal')
        need(p['cohort_seal']==self.plan['immutable_inputs'][target]
             and p['dataset']==binding(read(self.root/f'{phase}_config.json')['dataset']),'Prepared data bindings changed')
        official={k:v for k,v in t['official_model_files'].items() if '__pycache__' not in Path(k).parts}
        need(p['official_model']==official,'Evaluation changed official model/VAE/scheduler bytes')
        natives={'transformer':'models/transformers/transformer_wan.py','vae':'models/autoencoders/autoencoder_kl_wan.py',
                 'scheduler':'schedulers/scheduling_unipc_multistep.py'}
        need(p['runtime']['native']=={k:t['dependencies']['diffusers_native_sources'][v] for k,v in natives.items()},'Evaluation changed native implementation')
        need(p['metric_weights']=={k:bound(v) for k,v in self.plan['study']['metric_weights'].items()},'Prepared LPIPS weights changed')
        need(p['text_cache']==bound(t['base']['config']['data']['text_cache']),'Prepared native text cache changed')
        need(p['config']['device']==t['device'] and p['config']['gpu_lock']==t['gpu_lock']
             and p['config']['gpu_admission']==t['admission'],'Prepared GPU admission changed')
        if phase=='val':
            wanted={('shared',0)}|{(a['label'],u) for a in self.plan['study']['arms'] for u in (512,1024,2048)}
        else:
            selection=read(self.root/'selection.json');wanted={(r['arm'],2048) for r in selection['selected_roles']}
            need(p['selection_receipt']==binding(self.root/'selection.json') and p['validation_selection']==selection,'Prepared test selection changed')
        need({(r['arm'],r['update']) for r in p['roles']}==wanted,'Prepared role/update matrix changed')
        for role in p['roles']:
            cfg=role['config'];base=t['base']['config']
            need(role['variant']=='Base' and cfg['model']==base['model'] and cfg['data']==base['data']
                 and cfg.get('conditioning')==base.get('conditioning') and not cfg.get('reference',{}).get('enabled',False),
                 'Prepared inference model/conditioning/data changed')
        return {'protocol':binding(run/'protocol.json'),'scope':scope,'native_model_metric_source_chain':'matched_training',
                'new_scientific_thresholds':False,'evaluation_training_updates':0}

    def review_completed_evaluation(self,phase):
        run=self.root/('validation' if phase=='val' else 'test');p=read(run/'protocol.json');state=read(run/'status.json');summary=read(run/'summary.json')
        count=1024 if phase=='val' else 192
        need(state.get('status')=='completed' and state.get('training_updates')==0 and state.get('restores')==count
             and state.get('protocol_sha256')==sha(run/'protocol.json') and state.get('summary_sha256')==sha(run/'summary.json')
             and summary.get('restores')==count and summary.get('protocol_sha256')==sha(run/'protocol.json'), 'Incomplete evaluation completion chain')
        rows=read(run/'observations.json')['observations']
        wanted={(r['label'],i['record']['id'],s) for r in p['roles'] for i in p['cohort'] for s in (42,43)}
        need(len(rows)==count and {(r['role'],r['record_id'],r['sampling_seed']) for r in rows}==wanted,'Incomplete or duplicate evaluation row matrix')
        return {'status':binding(run/'status.json'),'summary':binding(run/'summary.json'),
                'observations':binding(run/'observations.json'),'restores':count,'training_updates':0}

    def review_selection(self):
        s=read(self.root/'selection.json');val=self.root/'validation';study=self.plan['immutable_inputs']['study']
        need(s.get('schema')=='fixed_teacher_validation_selection_v1' and s['study_file']==study
             and s['selected_after_complete_validation'] is True and s['test_pixels_read_by_selection'] is False
             and s['test_cohort_plan']==self.plan['immutable_inputs']['test_cohort_plan'],'Invalid validation-only selection receipt')
        for key,name in (('validation_protocol','protocol.json'),('validation_status','status.json'),('validation_summary','summary.json'),('validation_observations','observations.json')):
            need(s[key]==binding(val/name),'Selection validation chain changed')
        need([r['family'] for r in s['selected_roles']]==['baseline','hr_vae','hr_dino'],'Test must retain baseline and one setting per teacher')
        finals={r['arm']:r for r in read(val/'protocol.json')['roles'] if r['update']==2048}
        for role in s['selected_roles']:
            need(role['update']==role['step']==2048 and role['arm'] in finals
                 and role['checkpoint']==finals[role['arm']]['checkpoint'],'Selection changed trained final checkpoint')
        need(not (self.root/'test_assets').exists(),'Formal test RGB existed before selection')
        return {'receipt':binding(self.root/'selection.json'),'formal_test_RGB_absent_before_selection':True}

    def review_test_assets(self):
        d=read(self.root/'test_assets/dataset.json')
        need(d['schema']=='vsr_fixed_clean_dataset_v1' and d['status']=='completed_CPU_clean_assets' and d['split']=='test'
             and d['selection_receipt']==binding(self.root/'selection.json')
             and d['cohort_seal']==self.plan['immutable_inputs']['test_cohort_plan']
             and d['materialization']['selection_frozen_before_decode'] is True and len(d['assets'])==32,
             'Test RGB producer bypassed immutable selection/cohort')
        return {'dataset':binding(self.root/'test_assets/dataset.json'),'formal_RGB_after_selection':True}

    def review_analysis(self,phase):
        destination=Path(self.plan['output_dir'])/phase;report=read(destination/'analysis.json')
        run=self.root/('validation' if phase=='val' else 'test')
        need(report.get('schema')=='fixed_teacher_descriptive_analysis_v1' and report.get('phase')==phase
             and report.get('study_file')==self.plan['immutable_inputs']['study'] and report.get('runtime_sources')==self.plan['sources']
             and report.get('protocol_sha256')==sha(run/'protocol.json') and report.get('matrix_rows')==(1024 if phase=='val' else 192),
             'Analysis does not describe this complete frozen phase')
        artifacts={name:binding(destination/name) for name in ('analysis.json','report.md','quality_vs_updates.png','quality_vs_training_seconds.png')}
        return {'output_dir':str(destination),'artifacts':artifacts,'scientific_reinterpretation_performed_by_controller':False}

    def execute(self):
        self.plan=inspect_inputs(self.options);self.root=Path(self.plan['pipeline_dir'])
        self.root.mkdir()  # Atomic fresh path is the sole parent-controller claim.
        try:
            (self.root/'logs').mkdir();(self.root/'phases').mkdir();Path(self.plan['output_dir']).mkdir(parents=True)
            write(self.root/'plan.json',self.plan);self.status('planned')
            self.environment=dict(os.environ)|OFFLINE
            self.environment['CUDA_VISIBLE_DEVICES']=self.plan['study']['gpu_uuid']
            paths=[self.plan['source_root'],str(self.options.helper_root.resolve())]
            if self.plan['dependency_root']:paths.insert(1,self.plan['dependency_root'])
            self.environment['PYTHONPATH']=os.pathsep.join(paths)
            python=self.plan['python'];helpers=self.options.helper_root.resolve();train=Path(self.plan['training_run'])
            core=lambda *a:[python,'-m','vsr.evaluate',*map(str,a)]
            builder=lambda phase:[python,str(helpers/'fixed_teacher_eval_configs.py'),phase,'--source-root',self.plan['source_root'],
                '--study',str(self.options.study.resolve()),'--training-run',str(train),'--run-root',str(self.root),
                *(['--dependency-root',self.plan['dependency_root']] if self.plan['dependency_root'] else []),'--output',str(self.root/f'{phase}_config.json')]
            if self.plan['mode']=='include_training':
                self.child('01_train',[python,'-m','vsr.train','ft-execute','--run-dir',str(train),'--expected-protocol-sha',sha(train/'protocol.json')],self.review_training)
            else:
                self.phase='01_completed_training_review';self.frozen();evidence=self.review_training()
                write(self.root/'phases'/'01_completed_training_review.json',{'status':'completed','training_child_started':False,'evidence':evidence});self.event('completed',training_child_started=False)
            self.child('02_val_config',builder('val'),lambda:self.review_config('val'))
            self.child('03_val_prepare',core('ft-prepare','--config',self.root/'val_config.json','--run-dir',self.root/'validation'),lambda:self.review_protocol('val'))
            self.child('04_val_execute',core('ft-execute','--run-dir',self.root/'validation','--expected-protocol-sha',self.reviewed_sha(self.root/'validation/protocol.json')),lambda:self.review_completed_evaluation('val'))
            self.child('05_val_verify',core('ft-verify','--run-dir',self.root/'validation'),lambda:self.review_completed_evaluation('val'))
            self.child('06_val_select',core('ft-select','--validation-run',self.root/'validation','--output',self.root/'selection.json'),self.review_selection)
            self.child('07_test_materialize',[python,str(helpers/'fixed_teacher_data.py'),'test-materialize','--plan',self.plan['immutable_inputs']['test_cohort_plan']['path'],
                '--study',str(self.options.study.resolve()),'--selection-receipt',str(self.root/'selection.json'),'--output',str(self.root/'test_assets')],self.review_test_assets)
            test=builder('test')+['--validation-run',str(self.root/'validation'),'--selection-receipt',str(self.root/'selection.json'),
                '--test-dataset',str(self.root/'test_assets/dataset.json')]
            self.child('08_test_config',test,lambda:self.review_config('test'))
            self.child('09_test_prepare',core('ft-prepare','--config',self.root/'test_config.json','--run-dir',self.root/'test'),lambda:self.review_protocol('test'))
            self.child('10_test_execute',core('ft-execute','--run-dir',self.root/'test','--expected-protocol-sha',self.reviewed_sha(self.root/'test/protocol.json')),lambda:self.review_completed_evaluation('test'))
            self.child('11_test_verify',core('ft-verify','--run-dir',self.root/'test'),lambda:self.review_completed_evaluation('test'))
            for number,phase,run in ((12,'val','validation'),(13,'test','test')):
                self.child(f'{number:02d}_{phase}_analyze',[python,str(helpers/'analyze_fixed_teacher_study.py'),'--run',str(self.root/run),
                    '--output-dir',str(Path(self.plan['output_dir'])/phase)],lambda phase=phase:self.review_analysis(phase))
            self.frozen();self.phase='awaiting_root_delivery'
            result={'status':'awaiting_root_delivery','plan':binding(self.root/'plan.json'),'training_run':str(train),
                    'validation_run':str(self.root/'validation'),'test_run':str(self.root/'test'),
                    'selection_receipt':binding(self.root/'selection.json'),'output_dir':self.plan['output_dir'],
                    'scientific_updates':10240,'validation_restores':1024,'test_restores':192,'automatic_next_scientific_round':False,
                    'sealed_generated_bindings':self.generated_bindings}
            write(self.root/'completion.json',result);self.event('completed');self.status('awaiting_root_delivery',completion=binding(self.root/'completion.json'))
            return result
        except BaseException as error:
            record={'status':'failed_stopped','phase':self.phase,'command':self.command,'exit_code':self.exit_code,'error':str(error),'traceback':traceback.format_exc(),
                    'automatic_retry':False,'scientific_adjustment':False,'at_UTC':now()}
            if not (self.root/'failure.json').exists():write(self.root/'failure.json',record)
            self.event('failed',error=str(error));self.status('failed_stopped',failure=binding(self.root/'failure.json'))
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','study','training-run','run-root','python','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dependency-root',type=Path);p.add_argument('--execute',action='store_true',help='Explicitly run the reviewed serial pipeline; default only inspects')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--after-training',dest='after_training',action='store_true',default=True,help='Default: require complete training; never launch training')
    mode.add_argument('--include-training',dest='after_training',action='store_false',help='Explicit opt-in to launch one never-started planned training run')
    a=p.parse_args(argv);o=Options(a.source_root,a.study,a.training_run,a.run_root,a.python,a.output_dir,a.dependency_root,a.after_training)
    result=Pipeline(o).execute() if a.execute else inspect_inputs(o)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
