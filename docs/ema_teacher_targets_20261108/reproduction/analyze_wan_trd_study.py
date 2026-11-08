"""CPU-only, complete-matrix analysis of baseline / frozen Base / EMA Wan TRD.

Requires a bound native trd-verify receipt before reading scores.  Does not train,
restore, decode images, alter a protocol, select weights/checkpoints or run gates
from the historical EMA/G4 experiment.  Reused test sources remain exploratory.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import numpy as np
import analyze_hr_vae_study as common

read,sha,need,finite,verify_file=common.read,common.sha,common.need,common.finite,common.verify_file
METRICS=common.METRICS
ARMS=('baseline','frozen','ema')
UPDATES=(0,512,1024,2048)
CONTRASTS=(('ema_vs_frozen','ema','frozen'),('frozen_vs_baseline','frozen','baseline'),('ema_vs_baseline','ema','baseline'))
HIGHER_BETTER=common.HIGHER_BETTER


def binding(path):
    p=Path(path).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}


def stats(values):
    v=list(map(float,values));need(len(v)==32 and all(finite(x) for x in v),'Require32 finite source/clip values')
    return {'mean':statistics.mean(v),'clip_sample_SD_ddof1':statistics.stdev(v),'clips':32}


def _complete_header(run,verification):
    """Read status first, and fail before opening observations for incomplete runs."""
    run=Path(run).resolve();state=read(run/'status.json')
    need(state.get('status')=='completed' and state.get('training_updates')==0
         and state.get('restores') in (640,192),'Require completed TRD evaluation before any quality read')
    protocol=read(run/'protocol.json');phase=protocol.get('phase');count={'val':640,'test':192}.get(phase)
    need(protocol.get('schema')=='wan_trd_evaluation_protocol_v1' and count is not None
         and state['restores']==protocol['scope']['total_restores']==count
         and state['protocol_sha256']==sha(run/'protocol.json'),'Protocol/phase/full-matrix mismatch')
    summary=read(run/'summary.json')
    need(state['summary_sha256']==sha(run/'summary.json') and summary['protocol_sha256']==state['protocol_sha256']
         and summary['restores']==count and summary.get('quality_gate_applied') is False,'Completion summary chain changed')
    need(verification is not None,'Native trd-verify receipt required before analysis')
    vbind=binding(verification);verified=read(verify_file(vbind))
    need(verified.get('status')=='verified_complete_clean_evaluation' and Path(verified['run']).resolve()==run
         and verified['protocol_sha256']==state['protocol_sha256'] and verified['summary_sha256']==state['summary_sha256']
         and verified['observations_sha256']==sha(run/'observations.json') and verified['restores']==count
         and verified['prediction_PNGs_verified']==count*33 and verified['prediction_MP4s_verified']==count
         and verified['GT_PNGs_verified']==1056 and verified['GT_MP4s_verified']==32
         and verified.get('training_updates')==0 and verified.get('training_only_checkpoint_roles_verified') is True
         and verified.get('all_three_arms_retained') is True and verified.get('weight_or_checkpoint_selection') is False,
         'Native verification must cover this exact completed matrix and all student/PNG/MP4 identities')
    return run,phase,count,state,protocol,vbind


def _verify_final_lock(protocol,study):
    lock=read(verify_file(protocol['finals_receipt']))
    need(lock==protocol['validation_final_lock'] and lock['schema']=='wan_trd_validation_final_lock_v1'
         and lock['study_file']==protocol['study_file'] and lock['metric_values_used_to_select'] is False
         and lock['test_cohort_plan']==study['evaluation']['test_cohort_plan']
         and lock['test_dataset']==study['evaluation']['test_dataset'],'TRD all-final lock differs')
    for name in ('validation_protocol','validation_status','validation_summary','validation_observations'):verify_file(lock[name])
    vp=read(lock['validation_protocol']['path']);vs=read(lock['validation_status']['path']);vsum=read(lock['validation_summary']['path'])
    need(vp.get('schema')=='wan_trd_evaluation_protocol_v1' and vp['phase']=='val'
         and vp['scope']['total_restores']==vs['restores']==vsum['restores']==640
         and vs['status']=='completed' and vs['training_updates']==0
         and vs['protocol_sha256']==vsum['protocol_sha256']==lock['validation_protocol']['sha256']
         and vs['summary_sha256']==lock['validation_summary']['sha256'],'Final lock does not bind complete validation')
    vv=lock['validation_verification']
    need(vv['status']=='verified_complete_clean_evaluation' and vv['restores']==640
         and vv['protocol_sha256']==vs['protocol_sha256']
         and vv['observations_sha256']==lock['validation_observations']['sha256'],
         'Final lock validation verification changed')
    finals={r['arm']:r for r in vp['roles'] if r['update']==2048}
    need(set(finals)==set(ARMS) and [r['arm'] for r in lock['final_roles']]==list(ARMS),
         'Final lock must retain every arm without metric selection')
    for role,locked in zip(protocol['roles'],lock['final_roles']):
        need(role['arm']==locked['arm'] and role['update']==role['step']==2048
             and role['checkpoint']==locked['checkpoint']==finals[role['arm']]['checkpoint']
             and role['weight']==locked['weight']==finals[role['arm']]['weight'],
             'Test changed a prospectively locked final student')
    return lock


def _costs(run,phase,protocol,training,restore_seconds):
    files=training['files'];tp=read(verify_file(files['protocol.json']));ts=read(verify_file(files['summary.json']))
    need(tp['schema']=='wan_trd_training_protocol_v1' and ts['schema']=='wan_trd_training_summary_v1'
         and ts['status']=='completed' and ts['scientific_updates']==6144,'Training cost provenance differs')
    cal_binding=tp['calibration']['summary'];cal=read(verify_file(cal_binding))
    need(cal['status']=='passed_trd_preflight' and cal['scientific_updates']==0
         and cal['diagnostic_optimizer_updates']==41,'Calibration cost scope differs')
    ledger=ts.get('compute_cost')
    if ledger is not None:
        need(all(not isinstance(v,(int,float)) or finite(v) and v>=0 for v in ledger.values()),'Invalid training cost ledger')
    per_arm={};cost_rows=[]
    def cost(category,name,value,source,scope,unit='seconds',arm=''):
        need(value is None or finite(value) and value>=0,'Invalid cost value')
        cost_rows.append({'phase':phase,'category':category,'arm':arm,'measurement':name,'value':value,
            'unit':unit,'available':value is not None,'pure_GPU_kernel_time':False,'scope':scope,
            'source_path':source['path'] if source else '', 'source_sha256':source['sha256'] if source else ''})
    for arm in ARMS:
        record=ts['arms'][arm];log=read_jsonl(verify_file(record['full_training_log']))
        need(record['updates']==len(log)==2048 and [r['step'] for r in log]==list(range(1,2049)),
             'Cost rows require all2048 canonical updates per arm')
        forwards=sum(r['teacher_forward'] is True for r in log)
        need(forwards==(0 if arm=='baseline' else 2048) and record['EMA_updates']==(2048 if arm=='ema' else 0),
             'Unexpected teacher/update count in canonical training logs')
        seconds=record['cumulative_training_seconds'];need(finite(seconds) and seconds>0,'Invalid committed arm time')
        per_arm[arm]={'committed_step_wall_seconds':seconds,'updates':2048,'online_teacher_forwards':forwards,
                     'EMA_updates':record['EMA_updates'],'head_parameters':record.get('head_parameters'),'memory':record.get('memory')}
        cost('training','committed_step_wall_seconds',seconds,record['full_training_log'],
             'Retained synchronized student/head/teacher forward-backward, checks, AdamW and EMA; subset of attempt wall.',arm=arm)
        cost('training','online_teacher_forwards',forwards,record['full_training_log'],'Included in step timer; teacher-only elapsed is not isolated.',unit='forwards',arm=arm)
        cost('training','EMA_updates',record['EMA_updates'],files['summary.json'],'Included in step timer; not another training trajectory.',unit='updates',arm=arm)
    for key in ('committed_training_step_seconds','known_discarded_completed_step_seconds','wall_seconds_all_training_attempts'):
        cost('training',key,(ledger or {}).get(key),files['summary.json'],
             'Containing attempt wall and contained committed/discarded step timers overlap; do not add them together.')
    cost('training','known_discarded_completed_step_returns',(ledger or {}).get('known_discarded_completed_step_returns'),files['summary.json'],
         'Discarded optimizer returns, not retained scientific progress.',unit='updates')
    cost('calibration','preflight_wall_seconds',cal.get('wall_seconds'),cal_binding,
         'Preflight worker timer; includes15 train-only gradient probes and41 diagnostic updates within its declared scope; not formal training.')
    cost('calibration','diagnostic_optimizer_updates',41,cal_binding,'Separate from6144 scientific updates.',unit='updates')
    for arm,resource in cal.get('resources',{}).items():
        for key in ('timed6_mean_seconds','timed6_SD_seconds'):
            cost('calibration_resource',key,resource.get(key),cal_binding,'Six timed diagnostics after two warmups; subset of preflight wall; includes all slow values.',arm=arm)
    wall_path=run/'execution_wall_time.json';wall_binding=wall=None
    if wall_path.exists():
        wall_binding=binding(wall_path);wall=read(wall_path)
        need(wall.get('status')=='completed' and wall.get('stage')=='evaluation_'+phase
             and Path(wall['run']).resolve()==run and wall['protocol_sha256']==sha(run/'protocol.json')
             and finite(wall['wall_seconds']) and wall['wall_seconds']>=0 and isinstance(wall.get('scope'),str),
             'Evaluation wall receipt differs from completed phase')
    cost('evaluation','execution_wall_seconds',wall.get('wall_seconds') if wall else None,wall_binding,
         wall['scope'] if wall else 'Unrecorded: unknown, not estimated from timestamps or throughput.')
    restores=math.fsum(restore_seconds) if len(restore_seconds)==protocol['scope']['total_restores'] else None
    cost('evaluation','canonical_restore_seconds_sum',restores,binding(run/'observations.json'),
         'Successful canonical restore_video calls only; excludes model loading, metrics/media I/O and discarded attempts; contained in execute wall.')
    return {'training_attempts_source':files['summary.json'],'training_attempts':ledger,'per_arm':per_arm,
        'train_only_preflight':{'source':cal_binding,'wall_seconds':cal.get('wall_seconds'),'diagnostic_optimizer_updates':41,
                              'calibration':cal.get('calibration'),'resources':cal.get('resources')},
        'evaluation_execution':{'source':wall_binding,'receipt':wall,'canonical_restore_seconds_sum':restores},
        'offline_teacher_cache_this_study':{'used':False,'new_feature_extraction_seconds':0,
            'reason':'Frozen and EMA Wan teachers run online; teacher work is included in training steps.'},
        'common_Base_and_VAE_cache':'Existing shared sunk work, excluded from continuation axis; not claimed zero total cost.',
        'teacher_only_seconds':None,'teacher_only_seconds_reason':'Not isolated by the training timer; report matched arm overhead and teacher-forward counts.',
        'total_experiment_seconds':None,'missing_costs_are_unknown_not_zero':True,
        'aggregation_rule':'Do not sum containing attempt/preflight/execute timers with their step/probe/restore subcomponents.'},cost_rows


def read_jsonl(path):
    # No GPU/tensor loads; only complete canonical JSONL from completed training.
    lines=Path(path).read_text().splitlines()
    return [json.loads(x,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite log'))) for x in lines]


def _difference(left,right):
    result=(left-right).copy()
    for j,m in enumerate(METRICS):
        if m not in HIGHER_BETTER:result[:,j]*=-1
    return result


def _lpips_status(item):
    low,high=item['paired_clip_95pct_CI']
    if low==high==0:return 'no_observed_difference'
    return 'single_seed_descriptive_LPIPS_gain' if low>0 else 'no_observed_LPIPS_gain' if high<0 else 'insufficient_evidence'


def _quality_verdict(comparisons):
    main=comparisons['ema_vs_frozen'][2048];protect=comparisons['ema_vs_baseline'][2048]
    lpips_both=all(c['lpips_alex_v0_1']['paired_clip_95pct_CI'][0]>0 for c in (main,protect))
    fidelity={name:{m:comparisons[name][2048][m]['mean']>=0 for m in ('psnr_rgb_db','ssim_rgb')}
              for name in ('ema_vs_frozen','ema_vs_baseline')}
    point_pass=all(v for values in fidelity.values() for v in values.values())
    if lpips_both:status='candidate_for_multiseed_confirmation' if point_pass else 'LPIPS_gain_with_fidelity_tradeoff'
    elif any(c['lpips_alex_v0_1']['paired_clip_95pct_CI'][1]<0 for c in (main,protect)):status='no_observed_overall_EMA_advantage'
    elif any(c['lpips_alex_v0_1']['paired_clip_95pct_CI']==[0.,0.] for c in (main,protect)):status='no_observed_difference'
    else:status='insufficient_evidence'
    return {'classification':status,'EMA_over_frozen_and_baseline_LPIPS_CI_lower_strictly_above_zero':lpips_both,
        'raw_mean_PSNR_SSIM_nonnegative_guards':fidelity,'fidelity_point_guards_passed':point_pass,
        'temporal_metrics_additional_AND_gate':False,'old_G4_or_Base_seed_SD_gate_applied':False,
        'tolerance':0.,'single_training_seed_conditional_exploration':True,
        'scientific_success_claimed':False,'automatically_authorizes_next_experiment':False}


def compute(run,verification):
    run,phase,count,state,protocol,vbind=_complete_header(run,verification)
    need(len(protocol['sources'])==11,'Require all11 frozen package sources')
    for value in protocol['sources'].values():verify_file(value)
    study=read(verify_file(protocol['study_file']));arms=study['arms']
    need(study.get('schema')=='wan_trd_study_v1' and [a['label'] for a in arms]==list(ARMS)
         and arms[0]['weight']==0 and finite(arms[1]['weight']) and arms[1]['weight']>0
         and arms[1]['weight']==arms[2]['weight'] and study['updates']==2048
         and study['eval_updates']==[512,1024,2048],'Require unchanged three-arm common-weight study')
    training=protocol['training_evidence']
    need(training['scientific_updates']==6144 and training['study_file']==protocol['study_file'],'Complete paired training evidence changed')
    for value in training['files'].values():verify_file(value)
    verify_file(protocol['cohort_seal']);verify_file(protocol['dataset'])
    cohort=protocol['cohort'];records=[x['record']['id'] for x in cohort]
    sources={x['record']['id']:x['record']['source_id'] for x in cohort}
    need(len(records)==len(set(records))==len(set(sources.values()))==32,'Require32 distinct paired sources')
    scope=protocol['scope']
    need(scope['source_split']==phase and scope['conditions']==['clean'] and scope['sampling_seeds']==[42,43]
         and scope['primary_pool']=='all33_frames' and scope['training_updates']==0,'Clean metric scope changed')
    need(protocol['data_independence']['test_prior_exposure']=='historical_models_previously_evaluated'
         and protocol['data_independence']['historically_pristine_test'] is False,'Historical test exposure must remain disclosed')
    roles={r['label']:r for r in protocol['roles']}
    need(len(roles)==(10 if phase=='val' else 3),'Missing or duplicate role labels')
    points={(r['arm'],r['update']):r for r in roles.values()}
    wanted=([('shared',0)]+[(a,u) for a in ARMS for u in (512,1024,2048)] if phase=='val' else [(a,2048) for a in ARMS])
    need(list(points)==wanted,'Canonical shared/three-arm checkpoint matrix changed')
    weight={a['label']:a['weight'] for a in arms}
    for (arm,update),role in points.items():
        need(role['weight']==(0 if update==0 else weight[arm]) and role['step']==(8192 if update==0 else update),
             'Role weight/checkpoint update changed')
    lock=_verify_final_lock(protocol,study) if phase=='test' else None
    bundle=read(run/'observations.json');rows=bundle['observations']
    need(bundle['protocol_sha256']==state['protocol_sha256'] and len(rows)==count,'Incomplete observations')
    index={};restore_seconds=[]
    for row in rows:
        key=row['role'],row['record_id'],row['sampling_seed']
        need(key not in index and key[0] in roles and key[1] in sources and key[2] in (42,43),'Duplicate or foreign observation')
        role=roles[key[0]]
        need(row['split']==phase and row['source_id']==sources[key[1]] and row['condition']=='clean'
             and row['training_seed']==20261102 and row['cohort_seal_digest']==protocol['cohort_seal_digest']
             and row['checkpoint_sha256']==role['checkpoint']['sha256']
             and all(row[k]==role[k] for k in ('arm','family','setting','weight','step','update','cumulative_training_seconds')),
             'Observation role/source/weight/time/checkpoint identity differs')
        need(all(finite(row['metrics'][m]) for m in METRICS),'Nonfinite metric')
        path=Path(row['row_metrics_path']).resolve()
        need(path.is_relative_to(run/'results') and sha(path)==row['row_metrics_sha256'],'Canonical row metric SHA changed')
        raw=read(path);marker=read(path.parent/'complete.json')
        need(raw['observation']=={k:v for k,v in row.items() if not k.startswith('row_metrics_')}
             and marker['protocol_sha256']==state['protocol_sha256'] and marker['metrics_sha256']==row['row_metrics_sha256']
             and marker['PNG_count']==33,'Completed row marker/identity differs')
        if 'restore_seconds' in raw:
            need(finite(raw['restore_seconds']) and raw['restore_seconds']>=0,'Invalid restore timer')
            restore_seconds.append(raw['restore_seconds'])
        index[key]=row['metrics']
    need(set(index)=={(r,c,s) for r in roles for c in records for s in (42,43)},'Full matrix has missing observations')
    updates=UPDATES if phase=='val' else (2048,);vectors={};curves={}
    for arm in ARMS:
        curves[arm]={};previous=-1.
        for update in updates:
            role=points['shared',0] if update==0 else points[arm,update]
            seconds=role['cumulative_training_seconds']
            need(finite(seconds) and seconds>=0 and seconds>previous,'Training point times must increase')
            if update==0:need(seconds==0,'Shared initial has no additional training time')
            previous=seconds
            values=np.asarray([[statistics.mean(index[role['label'],rec,s][m] for s in (42,43)) for m in METRICS]
                               for rec in records],dtype=np.float64)
            vectors[arm,update]=values
            curves[arm][update]={'role':role['label'],'arm':arm,'weight':weight[arm],'update':update,
                'cumulative_training_seconds':seconds,'metrics':{m:stats(values[:,j]) for j,m in enumerate(METRICS)}}
    draws=np.random.Generator(np.random.PCG64(20261102)).integers(0,32,size=(10000,32),dtype=np.int64)
    comparisons={};paired=[]
    for name,method,reference in CONTRASTS:
        comparisons[name]={}
        for update in updates:
            difference=_difference(vectors[method,update],vectors[reference,update])
            intervals=np.quantile(difference[draws].mean(axis=1),[.025,.975],axis=0,method='linear')
            comparisons[name][update]={m:stats(difference[:,j])|{'paired_clip_95pct_CI':list(map(float,intervals[:,j])),
                'difference':method+'-minus-'+reference if m in HIGHER_BETTER else reference+'-minus-'+method,
                'positive_favors':method} for j,m in enumerate(METRICS)}
            for i,record in enumerate(records):
                paired.append({'phase':phase,'comparison':name,'method_arm':method,'reference_arm':reference,
                    'update':update,'record_id':record,'source_id':sources[record],'positive_favors':method,
                    **{m:float(difference[i,j]) for j,m in enumerate(METRICS)}})
    per_clip=[]
    for role in roles.values():
        for record in records:
            per_clip.append({'phase':phase,**{k:role[k] for k in ('label','arm','weight','update','step','cumulative_training_seconds')},
                'record_id':record,'source_id':sources[record],'training_seed':20261102,'sampling_seeds':'42,43',
                'checkpoint_sha256':role['checkpoint']['sha256'],
                **{m:statistics.mean(index[role['label'],record,s][m] for s in (42,43)) for m in METRICS}})
    final={name:{'method_arm':method,'reference_arm':reference,'LPIPS_status':_lpips_status(comparisons[name][2048]['lpips_alex_v0_1']),
                'metrics':comparisons[name][2048]} for name,method,reference in CONTRASTS}
    if phase=='val':
        target=curves['baseline'][2048]['metrics']['lpips_alex_v0_1']['mean']
        efficiency=common.first_target_crossings(curves,'baseline',target)
        efficiency['final2048_seconds_ratio_to_baseline']={a:curves[a][2048]['cumulative_training_seconds']/curves['baseline'][2048]['cumulative_training_seconds'] for a in ARMS}
        budgets=[]
        for base_update in UPDATES[1:]:
            cap=curves['baseline'][base_update]['cumulative_training_seconds'];measured={}
            for arm in ARMS:
                update=max(u for u in UPDATES if curves[arm][u]['cumulative_training_seconds']<=cap)
                point=curves[arm][update]
                measured[arm]={'latest_observed_update_within_budget':update,'actual_seconds':point['cumulative_training_seconds'],
                    'unused_budget_seconds':cap-point['cumulative_training_seconds'],'metrics':point['metrics']}
            budgets.append({'baseline_update_defining_budget':base_update,'budget_seconds':cap,'arms':measured})
        efficiency['equal_time_budget_observed_points']=budgets
        efficiency['equal_time_caveat']='Latest measured checkpoint within each common time budget; unequal residual unused time, no interpolation or exact equal-time claim.'
    else:efficiency={'status':'not_estimated_from_test','reason':'Only prospectively fixed final2048 points are tested; use val curves for efficiency.'}
    costs,cost_rows=_costs(run,phase,protocol,training,restore_seconds)
    return {'schema':'wan_trd_descriptive_analysis_v1','created_UTC':datetime.now(timezone.utc).isoformat(),
        'phase':phase,'run':str(run),'protocol_sha256':state['protocol_sha256'],'summary_sha256':state['summary_sha256'],
        'observations_sha256':sha(run/'observations.json'),'native_verification':vbind,'study_file':protocol['study_file'],
        'analysis_sources':{'main':binding(__file__),'common_statistics':binding(common.__file__)},
        'runtime_sources':protocol['sources'],'python_packages':{x:importlib.metadata.version(x) for x in ('numpy','matplotlib')},
        'training_seed':20261102,'training_seed_count':1,'sampling_seeds':[42,43],'scientific_updates':6144,
        'matrix_rows':count,'row_metrics_SHA_reverified':count,'metric_names':list(METRICS),
        'paired_clips':records,'paired_source_ids':sources,'curves':curves,'paired_comparisons':comparisons,
        'final_comparisons':final,'quality_verdict':_quality_verdict(comparisons),'per_clip_metrics':per_clip,
        'paired_clip_differences':paired,'efficiency':efficiency,'cost_accounting':costs,'cost_rows':cost_rows,
        'aggregation':'Mean sampling seeds42/43 within clip, then equal32distinct sources. SD is clip sampleSD ddof1, not training-seed SD.',
        'primary_comparison':'ema_vs_frozen','secondary_comparisons':['frozen_vs_baseline','ema_vs_baseline'],
        'bootstrap':{'draws':10000,'seed':20261102,'generator':'PCG64','draw_indices_sha256':hashlib.sha256(draws.astype('<i8',copy=False).tobytes()).hexdigest(),
            'interval':'95% linear-percentile','paired_resampling_unit':'clip after sampler mean','common_draws_all_metrics_contrasts':True,
            'scope':'descriptive conditional on this single training seed','p_values_computed':False},
        'time_axis':{'field':'cumulative_training_seconds','meaning':'Canonical committed synchronized training-step wall seconds',
            'pure_GPU_time':False,'total_experiment_time':False,'interpolation_used':False,
            'includes':'student/head/online-teacher forward/backward, finite checks, optimizer, EMA and cleanup inside timer',
            'excludes':'Base/VAE sunk work, preflight, outside-timer data/model loads, checkpoint IO, evaluation and discarded/replayed steps'},
        'data_independence':protocol['data_independence'],'validation_final_lock':protocol.get('finals_receipt'),
        'weight_or_checkpoint_selection':False,'GPU_work_performed':False,'auxiliary_loss_used_to_claim_quality':False,
        'limitations':['One training seed: clip intervals and two sampling repeats do not establish reproducibility over training seeds or statistical significance.',
            'Both val and test cohorts have historical model exposure. Test is a reused, source-separated exploratory evaluation, not a pristine/blind confirmatory holdout.',
            'All three final2048 checkpoints are prospectively retained; no weights or intermediate checkpoints are selected using these metrics.',
            'Fixed33-frame128x224 crops with clean4x bicubic degradation; not full source duration, full resolution or an external benchmark.',
            'Full TRD supervision configurations are compared. EMA differs from a frozen Base teacher by evolving teacher weights; this is not an original CogVideoX VideoREPA replication.',
            'LPIPS improvements, PSNR/SSIM point guards and temporal tradeoffs are reported separately. No inherited G4 threshold, extra temporal AND gate or tolerance tuning.',
            'Efficiency uses sparse measured checkpoints and synchronized step wall time. Common preparation and overlapping attempt/calibration/evaluation costs are separate; unknown cost is not zero.']}


def markdown(report):
    lines=['# Wan TRD：原始基线、冻结 Base 教师与 EMA 教师','',
        f'阶段：{report["phase"]}；完整{report["matrix_rows"]}次恢复。三个训练arm各2048步，共6144步（0.75RE），仅一个训练seed。',
        '每clip先平均采样42/43，再对32个不同source等权平均。表内SD为clip sample SD，不是训练seed SD。','',
        '| Arm / λ | 更新 | 已提交step墙钟秒 | LPIPS↓ | PSNR↑ dB | SSIM↑ | tLP100↓ | tOF↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,points in report['curves'].items():
        for update,p in points.items():
            cells=[f'{p["metrics"][m]["mean"]:.6f} ± {p["metrics"][m]["clip_sample_SD_ddof1"]:.6f}' for m in METRICS]
            lines.append(f'| {arm} / {p["weight"]} | {update} | {p["cumulative_training_seconds"]:.3f} | '+' | '.join(cells)+' |')
    lines+=['','## 最终2048步配对结果','',
        'LPIPS/tLP100/tOF取reference−method，PSNR/SSIM取method−reference；正值有利于method。10000次PCG64配对clip的95%linear-percentile区间仅描述当前训练seed。','',
        '| 对比（method vs reference） | LPIPS收益 [95%CI] | PSNR差 [95%CI] | SSIM差 [95%CI] | LPIPS判断 |',
        '| --- | ---: | ---: | ---: | --- |']
    for name,result in report['final_comparisons'].items():
        cells=[]
        for m in METRICS[:3]:
            x=result['metrics'][m];lo,hi=x['paired_clip_95pct_CI'];cells.append(f'{x["mean"]:.8f} [{lo:.8f}, {hi:.8f}]')
        lines.append(f'| {name} | '+' | '.join(cells)+f' | {result["LPIPS_status"]} |')
    verdict=report['quality_verdict'];lines+=['','## 预先声明的判定','',
        f'分类：**{verdict["classification"]}**。',
        '进入多种子确认的候选条件：EMA相对冻结教师及baseline的LPIPS收益区间下界均严格大于0，并且相对两者的未取整PSNR、SSIM平均差均≥0。区间跨0表示证据不足；保真代价单列。时序指标不额外加入AND门槛。',
        '本分析不会自动授权下一轮实验，也不把单seed描述性区间称作显著性证明。','', '## 训练效率与成本','']
    efficiency=report['efficiency']
    if efficiency['status']=='not_estimated_from_test':lines.append('Test只测试三组固定final2048，用val中间检查点曲线判断效率。')
    elif efficiency['status']=='uninterpretable_initial_already_meets_target':lines.append('共享step0已达到baseline@2048的LPIPS目标，不能据此声称训练加速。')
    else:
        lines.append(f'目标为baseline@2048 LPIPS={efficiency["target_LPIPS"]:.8f}；只使用实际测量点，不插值。')
        for arm,value in efficiency['crossings'].items():lines.append(f'- {arm}：'+('已测点未达到目标。' if value['update'] is None else f'最早观测{value["update"]}步，{value["cumulative_training_seconds"]:.3f}同步step秒。'))
    lines+=['','时间轴是成功保留步骤的同步墙钟时间，含在线教师与EMA，既非纯GPU时间也非整体实验时间。完整attempt墙钟、预检41步及评价开销另列；总量与子项有重叠，不能重复相加。',
        '冻结/EMA教师均在线运行，没有本轮离线特征缓存。Base及VAE公共缓存是已有成本，并非总成本为零。',
        '', '[逐clip五指标](per_clip_metrics.csv) · [逐clip配对差](paired_clip_differences.csv) · [全部汇总](summary.csv) · [配对区间](paired_summary.csv) · [成本与范围](costs.csv) · [完整JSON](analysis.json)',
        '', '![质量随更新变化](quality_vs_updates.png)','','![质量随同步训练时间变化](quality_vs_training_seconds.png)','','## 限制','']
    lines+=['- '+x for x in report['limitations']]
    return '\n'.join(lines)+'\n'


def _csv(path,rows):
    rows=list(rows);need(rows,'Cannot deliver an empty table')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def plots(report,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    palette={'baseline':'#222222','frozen':'#0072B2','ema':'#D55E00'}
    for timed,name in ((False,'quality_vs_updates.png'),(True,'quality_vs_training_seconds.png')):
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
        for arm,points in report['curves'].items():
            updates=sorted(points);x=[points[u]['cumulative_training_seconds'] if timed else u for u in updates]
            for axis,(metric,title) in zip(axes,[(METRICS[0],'LPIPS, lower better'),(METRICS[1],'PSNR dB, higher better'),(METRICS[3],'tLP100, lower better')]):
                axis.plot(x,[points[u]['metrics'][metric]['mean'] for u in updates],marker='o',color=palette[arm],
                          linestyle='-' if len(updates)>1 else 'None',label=arm)
                axis.set_title(title);axis.grid(alpha=.25);axis.set_xlabel('Committed synchronized step wall seconds' if timed else 'Additional optimizer updates')
                if not timed:axis.set_xticks(updates)
        axes[0].legend();fig.suptitle('Wan TRD | '+report['phase']+'32 | one training seed | '+('lines guide the eye; no interpolation' if report['phase']=='val' else 'final points only; no convergence inference'))
        fig.savefig(destination/name,dpi=180);plt.close(fig)


def deliver(run,verification,output_dir):
    destination=Path(output_dir).resolve();need(not destination.exists(),'Analysis deliverables are immutable; use a fresh directory')
    report=compute(run,verification);destination.mkdir(parents=True)
    (destination/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (destination/'README.md').write_text(markdown(report))
    _csv(destination/'per_clip_metrics.csv',report['per_clip_metrics'])
    _csv(destination/'paired_clip_differences.csv',report['paired_clip_differences'])
    summary=[];paired=[]
    for arm,points in report['curves'].items():
        for update,point in points.items():
            for metric,values in point['metrics'].items():summary.append({'arm':arm,'weight':point['weight'],'update':update,'cumulative_training_seconds':point['cumulative_training_seconds'],'metric':metric,**values})
    for name,points in report['paired_comparisons'].items():
        for update,metrics in points.items():
            for metric,values in metrics.items():paired.append({'comparison':name,'update':update,'metric':metric,**{k:v for k,v in values.items() if k!='paired_clip_95pct_CI'},'CI_low':values['paired_clip_95pct_CI'][0],'CI_high':values['paired_clip_95pct_CI'][1]})
    _csv(destination/'summary.csv',summary);_csv(destination/'paired_summary.csv',paired);_csv(destination/'costs.csv',report['cost_rows'])
    plots(report,destination)
    manifest={'schema':'wan_trd_analysis_delivery_v1','phase':report['phase'],'protocol_sha256':report['protocol_sha256'],
              'files':[binding(p) for p in sorted(destination.iterdir()) if p.is_file()],
              'media_are_local_only':True,'text_CSV_JSON_publication_allowed':True,'GPU_work_performed':False}
    (destination/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return {'status':'complete_CPU_analysis_delivery','phase':report['phase'],'output_dir':str(destination),
            'matrix_rows':report['matrix_rows'],'classification':report['quality_verdict']['classification'],'manifest':binding(destination/'delivery_manifest.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--verification',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(deliver(args.run,args.verification,args.output_dir),ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
