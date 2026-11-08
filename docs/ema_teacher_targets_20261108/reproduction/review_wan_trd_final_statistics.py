"""Independent CPU/text-only audit; rejects either incomplete matrix before scores.

Imports no project or analysis helper.  Recomputes from 640+192 canonical rows,
with manual linear-percentile bootstrap endpoints and fsum sample statistics.
Native media/source verification is required and remains a separate operation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

METRICS=('lpips_alex_v0_1','psnr_rgb_db','ssim_rgb','tLP100_alex_v0_1_TecoGAN_crop','tOF_Farneback_TecoGAN_crop')
ARMS=('baseline','frozen','ema')
POINTS=(0,512,1024,2048)
CONTRASTS=(('ema_vs_frozen','ema','frozen'),('frozen_vs_baseline','frozen','baseline'),('ema_vs_baseline','ema','baseline'))
HIGHER={'psnr_rgb_db','ssim_rgb'}
# Comparison of two floating-point implementations only; never applied to quality guards.
PARITY_ATOL=2e-12


def require(ok,message):
    if not ok: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def bind(path):
    p=Path(path).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)}


def bound(item):
    p=Path(item['path']).resolve()
    require(p.is_file() and digest(p)==item['sha256'],'Changed bound file '+str(p))
    require('bytes' not in item or p.stat().st_size==item['bytes'],'Changed bound file size')
    return p


def finite(v):return type(v) in (int,float) and math.isfinite(v)


def same_number(a,b,label,atol=PARITY_ATOL):
    require(finite(a) and finite(b) and math.isclose(a,b,rel_tol=0,abs_tol=atol),'Numerical parity mismatch: '+label)


def sample_stats(v):
    xs=list(map(float,v));require(len(xs)==32 and all(map(math.isfinite,xs)),'Expected 32 finite clip values')
    mean=math.fsum(xs)/32
    sd=math.sqrt(math.fsum((x-mean)**2 for x in xs)/31)
    return {'mean':mean,'clip_sample_SD_ddof1':sd,'clips':32}


def percentile_linear(values,p):
    ordered=np.sort(np.asarray(values,dtype=np.float64),axis=0)
    location=(len(ordered)-1)*p;lo=math.floor(location);hi=math.ceil(location)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(location-lo)


def lpips_class(interval):
    low,high=interval
    if low==high==0:return 'no_observed_difference'
    if low>0:return 'single_seed_descriptive_LPIPS_gain'
    if high<0:return 'no_observed_LPIPS_gain'
    return 'insufficient_evidence'


def verdict(comparisons):
    names=('ema_vs_frozen','ema_vs_baseline')
    checks={n:{m:comparisons[n][2048][m]['mean']>=0 for m in ('psnr_rgb_db','ssim_rgb')} for n in names}
    fidelity=all(v for row in checks.values() for v in row.values())
    ci=[comparisons[n][2048][METRICS[0]]['paired_clip_95pct_CI'] for n in names]
    positive=all(x[0]>0 for x in ci)
    if positive:status='candidate_for_multiseed_confirmation' if fidelity else 'LPIPS_gain_with_fidelity_tradeoff'
    elif any(x[1]<0 for x in ci):status='no_observed_overall_EMA_advantage'
    elif any(x==[0.,0.] for x in ci):status='no_observed_difference'
    else:status='insufficient_evidence'
    return {'classification':status,'EMA_over_frozen_and_baseline_LPIPS_CI_lower_strictly_above_zero':positive,
            'raw_mean_PSNR_SSIM_nonnegative_guards':checks,'fidelity_point_guards_passed':fidelity}


def complete_statuses(val_run,test_run):
    # No protocol, observation, metric or analysis file is opened before BOTH pass.
    states={}
    for phase,root,count in [('val',val_run,640),('test',test_run,192)]:
        state=read(root/'status.json')
        require(state.get('status')=='completed' and state.get('training_updates')==0 and state.get('restores')==count,
                'Both matrices must be completed before any quality read: '+phase)
        states[phase]=state
    return states


def training_bindings_metadata_only(evidence,roles):
    """Hash every text binding; stat exactly12 bound CPs, relying on native verify.

    No checkpoint content is opened here.  Protocol/receipt SHA and explicit
    arm/checkpoint metadata must agree before this narrower check is accepted.
    """
    training_root=Path(evidence['run']).resolve()
    files=evidence['files'];expected={f'{arm}/checkpoint_{step:06d}' for arm in ARMS for step in (512,1024,1536,2048)}
    seen=set();cps={}
    for key,item in files.items():
        path=Path(item['path']).resolve()
        if key in expected:
            arm,filename=key.split('/');step=int(filename.rsplit('_',1)[1])
            wanted=training_root/'arm_runs'/arm/'Base'/(filename+'.pt')
            require(path==wanted and path.is_file() and type(item.get('bytes')) is int and item['bytes']>0
                and path.stat().st_size==item['bytes'],'Checkpoint path/size metadata differs: '+key)
            require(isinstance(item.get('sha256'),str) and len(item['sha256'])==64
                and all(c in '0123456789abcdef' for c in item['sha256']),'Malformed native-bound checkpoint SHA')
            matches=[x for x in evidence['arms'][arm]['checkpoint_files'] if x['step']==step and x['update']==step]
            require(len(matches)==1 and all(matches[0][k]==item[k] for k in ('path','bytes','sha256')),'Arm/checkpoint metadata binding differs')
            seen.add(key);cps[arm,step]=item
        else:
            require(path.suffix in ('.json','.jsonl','.log','.txt','.md','.py'),'Unexpected non-text training binding: '+key)
            bound(item)
    require(seen==expected,'Expected all12 native-verified training checkpoint metadata entries')
    for role in roles:
        if role['update']>0:
            require(role['checkpoint']==cps[role['arm'],role['update']],'Role checkpoint does not match native-bound training metadata')
    return len(seen)


def header(root,phase,state,receipt):
    p=read(root/'protocol.json');count={'val':640,'test':192}[phase]
    require(p.get('schema')=='wan_trd_evaluation_protocol_v1' and p['phase']==phase,'Wrong protocol')
    require(state['protocol_sha256']==digest(root/'protocol.json') and p['scope']['total_restores']==count,'Protocol/count hash mismatch')
    summary=read(root/'summary.json');require(state['summary_sha256']==digest(root/'summary.json'),'Summary hash mismatch')
    require(summary['protocol_sha256']==state['protocol_sha256'] and summary['restores']==count and summary.get('quality_gate_applied') is False,'Completion summary differs')
    v=read(receipt)
    expected={'status':'verified_complete_clean_evaluation','protocol_sha256':state['protocol_sha256'],
        'summary_sha256':state['summary_sha256'],'observations_sha256':digest(root/'observations.json'),
        'restores':count,'prediction_PNGs_verified':count*33,'prediction_MP4s_verified':count,
        'GT_PNGs_verified':1056,'GT_MP4s_verified':32,'training_updates':0,
        'training_only_checkpoint_roles_verified':True,'all_three_arms_retained':True,'weight_or_checkpoint_selection':False}
    require(Path(v['run']).resolve()==root and all(v.get(k)==value for k,value in expected.items()),'Native verification receipt mismatch')
    require(len(p['sources'])==11,'Expected 11 frozen sources')
    for b in p['sources'].values():bound(b)
    training_bindings_metadata_only(p['training_evidence'],p['roles'])
    for k in ('study_file','cohort_seal','dataset'):bound(p[k])
    require(p['scope']['source_split']==phase and p['scope']['sampling_seeds']==[42,43]
        and p['scope']['conditions']==['clean'] and p['scope']['primary_pool']=='all33_frames'
        and p['scope']['training_updates']==0,'Metric pool differs')
    require(p['data_independence']['test_prior_exposure']=='historical_models_previously_evaluated'
        and p['data_independence']['historically_pristine_test'] is False,'Historical exposure disclosure changed')
    return p,v


def pairing_lock(valp,testp,val_run,test_run,valverify):
    require(valp['study_file']==testp['study_file'] and valp['training_evidence']==testp['training_evidence'],'Different training study across phases')
    require(valp['sources']==testp['sources'],'Different source binding across phases')
    study=read(bound(valp['study_file']))
    require(study.get('schema')=='wan_trd_study_v1' and [x['label'] for x in study['arms']]==list(ARMS)
        and study['arms'][0]['weight']==0 and study['arms'][1]['weight']==study['arms'][2]['weight']>0
        and study['updates']==2048 and study['eval_updates']==[512,1024,2048],'Study settings changed')
    lock=read(bound(testp['finals_receipt']))
    require(lock==testp['validation_final_lock'] and lock['schema']=='wan_trd_validation_final_lock_v1'
        and lock['study_file']==valp['study_file'] and lock['metric_values_used_to_select'] is False,'Final lock differs')
    for key in ('protocol','status','summary','observations'):
        b=lock['validation_'+key];bound(b)
        require(Path(b['path']).resolve()==val_run/(key+'.json'),'Final lock refers to another validation')
    require(lock['validation_verification']==valverify,'Final lock verification differs')
    require(lock['test_cohort_plan']==study['evaluation']['test_cohort_plan'] and lock['test_dataset']==study['evaluation']['test_dataset'],'Test cohort final lock differs')
    finals={r['arm']:r for r in valp['roles'] if r['update']==2048}
    require([r['arm'] for r in testp['roles']]==list(ARMS)==[r['arm'] for r in lock['final_roles']],'All finals not retained')
    for actual,locked in zip(testp['roles'],lock['final_roles']):
        initial=finals[actual['arm']]
        require(actual['checkpoint']==locked['checkpoint']==initial['checkpoint'] and actual['step']==actual['update']==2048
            and actual['weight']==locked['weight']==initial['weight'],'Final checkpoint/weight changed')
    return study


def training_costs(protocol):
    e=protocol['training_evidence'];require(e['scientific_updates']==6144,'Wrong scientific budget')
    files=e['files'];tp=read(bound(files['protocol.json']));s=read(bound(files['summary.json']))
    require(tp['schema']=='wan_trd_training_protocol_v1' and s['schema']=='wan_trd_training_summary_v1'
        and s['status']=='completed' and s['scientific_updates']==6144,'Incomplete training cost source')
    cal=read(bound(tp['calibration']['summary']))
    require(cal['status']=='passed_trd_preflight' and cal['scientific_updates']==0 and cal['diagnostic_optimizer_updates']==41,'Preflight source changed')
    result={};times={}
    for arm in ARMS:
        record=s['arms'][arm];log_path=bound(record['full_training_log'])
        rows=[json.loads(line) for line in log_path.read_text().splitlines()]
        require(record['updates']==len(rows)==2048 and [r['step'] for r in rows]==list(range(1,2049)),'Incomplete canonical training steps')
        forward=sum(r['teacher_forward'] is True for r in rows)
        require(forward==(0 if arm=='baseline' else 2048) and record['EMA_updates']==(2048 if arm=='ema' else 0),'Teacher/EMA counts changed')
        require(all(finite(r['seconds']) and r['seconds']>=0 and finite(r['cumulative_training_seconds']) for r in rows),'Invalid step timer')
        elapsed=math.fsum(r['seconds'] for r in rows)
        same_number(elapsed,record['cumulative_training_seconds'],'sum of committed '+arm+' seconds',1e-7)
        require(rows[-1]['cumulative_training_seconds']==record['cumulative_training_seconds'],'Final time not canonical')
        times[arm]={r['step']:r['cumulative_training_seconds'] for r in rows}
        result[arm]={'committed_step_wall_seconds':record['cumulative_training_seconds'],'recomputed_sum_step_seconds':elapsed,
            'updates':2048,'online_teacher_forwards':forward,'EMA_updates':record['EMA_updates']}
    ledger=s.get('compute_cost')
    if ledger is not None:
        require(all(not isinstance(v,(int,float)) or finite(v) and v>=0 for v in ledger.values()),'Invalid attempt cost')
        if ledger.get('committed_training_step_seconds') is not None:
            same_number(math.fsum(x['committed_step_wall_seconds'] for x in result.values()),ledger['committed_training_step_seconds'],'total committed seconds',1e-7)
    return {'per_arm':result,'training_attempts':ledger,'preflight_wall_seconds':cal.get('wall_seconds'),
        'preflight_diagnostic_updates':41,'scientific_updates':6144,'offline_teacher_cache_used':False,
        'teacher_only_seconds':None,'total_experiment_seconds':None,'overlapping_timers_added':False},times


def phase_rows(root,p,state,times):
    records=[x['record']['id'] for x in p['cohort']];source={x['record']['id']:x['record']['source_id'] for x in p['cohort']}
    require(len(records)==len(set(records))==len(set(source.values()))==32,'Expected distinct 32 paired sources')
    roles={r['label']:r for r in p['roles']};phase=p['phase']
    wanted=[('shared',0)]+[(a,u) for a in ARMS for u in POINTS[1:]] if phase=='val' else [(a,2048) for a in ARMS]
    require([(r['arm'],r['update']) for r in p['roles']]==wanted and len(roles)==len(wanted),'Missing/duplicate/canonical role changed')
    for r in roles.values():
        if r['update']==0:require(r['step']==8192 and r['cumulative_training_seconds']==0,'Wrong shared initial')
        else:require(r['step']==r['update'] and r['cumulative_training_seconds']==times[r['arm']][r['update']],'Checkpoint time differs from training log')
    bundle=read(root/'observations.json');rows=bundle['observations']
    require(bundle['protocol_sha256']==state['protocol_sha256'] and len(rows)==p['scope']['total_restores'],'Incomplete observations')
    values={};seconds=[]
    for row in rows:
        k=row['role'],row['record_id'],row['sampling_seed']
        require(k not in values and k[0] in roles and k[1] in source and k[2] in (42,43),'Duplicate/foreign observation')
        role=roles[k[0]]
        require(row['condition']=='clean' and row['training_seed']==20261102 and row['split']==phase
            and row['source_id']==source[k[1]] and row['cohort_seal_digest']==p['cohort_seal_digest']
            and row['checkpoint_sha256']==role['checkpoint']['sha256']
            and all(row[x]==role[x] for x in ('arm','family','setting','weight','step','update','cumulative_training_seconds')),'Observation identity mismatch')
        require(all(finite(row['metrics'][m]) for m in METRICS),'Nonfinite metric')
        mp=Path(row['row_metrics_path']).resolve();require(mp.is_relative_to(root/'results') and digest(mp)==row['row_metrics_sha256'],'Row metric SHA mismatch')
        raw=read(mp);marker=read(mp.parent/'complete.json')
        require(raw['observation']=={k:v for k,v in row.items() if not k.startswith('row_metrics_')}
            and marker['protocol_sha256']==state['protocol_sha256'] and marker['metrics_sha256']==row['row_metrics_sha256'] and marker['PNG_count']==33,'Row completion mismatch')
        require(finite(raw['restore_seconds']) and raw['restore_seconds']>=0,'Missing canonical restore time')
        seconds.append(raw['restore_seconds']);values[k]=row['metrics']
    require(set(values)=={(r,c,s) for r in roles for c in records for s in (42,43)},'Missing matrix cell')
    lookup={(r['arm'],r['update']):r for r in roles.values()};vectors={};curves={};points=POINTS if phase=='val' else (2048,)
    for arm in ARMS:
        curves[arm]={}
        for update in points:
            role=lookup['shared',0] if update==0 else lookup[arm,update]
            vector=np.array([[math.fsum(values[role['label'],r,s][m] for s in (42,43))/2 for m in METRICS] for r in records],dtype=np.float64)
            vectors[arm,update]=vector
            curves[arm][update]={'metrics':{m:sample_stats(vector[:,j]) for j,m in enumerate(METRICS)},'cumulative_training_seconds':role['cumulative_training_seconds']}
    cost={'canonical_restore_seconds_sum':math.fsum(seconds),'execution_wall_seconds':None}
    wp=root/'execution_wall_time.json'
    if wp.exists():
        w=read(wp);require(w['status']=='completed' and w['stage']=='evaluation_'+phase and Path(w['run']).resolve()==root
            and w['protocol_sha256']==state['protocol_sha256'] and finite(w['wall_seconds']) and w['wall_seconds']>=0,'Execute wall receipt invalid')
        cost['execution_wall_seconds']=w['wall_seconds']
    return records,source,vectors,curves,cost


def differences(vectors,points,draws):
    result={}
    for name,method,reference in CONTRASTS:
        result[name]={}
        for u in points:
            delta=vectors[method,u]-vectors[reference,u]
            for j,m in enumerate(METRICS):
                if m not in HIGHER:delta[:,j]=-delta[:,j]
            replicated=np.sum(delta[draws],axis=1,dtype=np.float64)/32
            low=percentile_linear(replicated,.025);high=percentile_linear(replicated,.975)
            result[name][u]={m:sample_stats(delta[:,j])|{'paired_clip_95pct_CI':[float(low[j]),float(high[j])]} for j,m in enumerate(METRICS)}
    return result


def efficiency(curves):
    target=curves['baseline'][2048]['metrics'][METRICS[0]]['mean']
    if curves['baseline'][0]['metrics'][METRICS[0]]['mean']<=target:
        result={'status':'uninterpretable_initial_already_meets_target','crossings':{a:None for a in ARMS},'speedups_vs_baseline':{a:None for a in ARMS}}
    else:
        crossings={}
        for a in ARMS:
            candidates=[u for u in POINTS if curves[a][u]['metrics'][METRICS[0]]['mean']<=target]
            first=candidates[0] if candidates else None
            crossings[a]={'status':'reached_at_observed_checkpoint' if first is not None else 'not_reached_at_measured_checkpoints','update':first,'cumulative_training_seconds':curves[a][first]['cumulative_training_seconds'] if first is not None else None}
        base=crossings['baseline']['cumulative_training_seconds']
        result={'status':'discrete_checkpoint_comparison','crossings':crossings,'speedups_vs_baseline':{a:base/v['cumulative_training_seconds'] if v['cumulative_training_seconds'] else None for a,v in crossings.items()}}
    result['target_LPIPS']=target
    result['final2048_seconds_ratio_to_baseline']={a:curves[a][2048]['cumulative_training_seconds']/curves['baseline'][2048]['cumulative_training_seconds'] for a in ARMS}
    result['equal_time_updates']={str(u):{a:max(x for x in POINTS if curves[a][x]['cumulative_training_seconds']<=curves['baseline'][u]['cumulative_training_seconds']) for a in ARMS} for u in POINTS[1:]}
    return result


def compare_statistics(calc,existing,label):
    for field in ('mean','clip_sample_SD_ddof1'):same_number(calc[field],existing[field],label+' '+field)
    require(existing['clips']==32,label+' clip count')
    if 'paired_clip_95pct_CI' in calc:
        for i,x in enumerate(calc['paired_clip_95pct_CI']):same_number(x,existing['paired_clip_95pct_CI'][i],label+' CI')


def compare_phase(phase,records,curves,comparisons,quality,eff,cost,training,analysis_path,protocol_sha,observations_sha,draw_sha):
    a=read(analysis_path);require(a['schema']=='wan_trd_descriptive_analysis_v1' and a['phase']==phase
        and a['protocol_sha256']==protocol_sha and a['observations_sha256']==observations_sha,'Analysis source identity mismatch')
    require(a['paired_clips']==records and a['matrix_rows']==(640 if phase=='val' else 192),'Analysis cohort differs')
    require(a['bootstrap']['draw_indices_sha256']==draw_sha and a['bootstrap']['draws']==10000 and a['bootstrap']['seed']==20261102
        and a['bootstrap']['generator']=='PCG64' and a['bootstrap']['p_values_computed'] is False,'Bootstrap declaration differs')
    for b in a['analysis_sources'].values():bound(b)
    points=POINTS if phase=='val' else (2048,)
    for arm in ARMS:
        for u in points:
            old=a['curves'][arm][str(u)];same_number(curves[arm][u]['cumulative_training_seconds'],old['cumulative_training_seconds'],'Curve time')
            for m in METRICS:compare_statistics(curves[arm][u]['metrics'][m],old['metrics'][m],phase+' '+arm+' '+str(u)+' '+m)
    for name,_,_ in CONTRASTS:
        for u in points:
            for m in METRICS:compare_statistics(comparisons[name][u][m],a['paired_comparisons'][name][str(u)][m],phase+' '+name+' '+str(u)+' '+m)
        require(lpips_class(comparisons[name][2048][METRICS[0]]['paired_clip_95pct_CI'])==a['final_comparisons'][name]['LPIPS_status'],'LPIPS classification differs')
    for k,v in quality.items():require(a['quality_verdict'][k]==v,'Quality classification/raw guard mismatch: '+k)
    require(a['quality_verdict']['tolerance']==0 and a['quality_verdict']['old_G4_or_Base_seed_SD_gate_applied'] is False
        and a['quality_verdict']['temporal_metrics_additional_AND_gate'] is False and a['quality_verdict']['scientific_success_claimed'] is False,'Scientific guard/claim metadata changed')
    c=a['cost_accounting']
    require(c['training_attempts']==training['training_attempts'] and c['train_only_preflight']['diagnostic_optimizer_updates']==41,'Cost ledger differs')
    require(c['train_only_preflight']['wall_seconds']==training['preflight_wall_seconds'],'Preflight wall cost differs')
    for arm,r in training['per_arm'].items():
        old=c['per_arm'][arm]
        for k in ('updates','online_teacher_forwards','EMA_updates'):require(old[k]==r[k],'Cost count mismatch '+arm+' '+k)
        same_number(old['committed_step_wall_seconds'],r['committed_step_wall_seconds'],'Arm committed cost')
    same_number(c['evaluation_execution']['canonical_restore_seconds_sum'],cost['canonical_restore_seconds_sum'],'Canonical restore time',1e-8)
    recorded=c['evaluation_execution']['receipt'];require((recorded['wall_seconds'] if recorded else None)==cost['execution_wall_seconds'],'Eval wall cost differs')
    require(c['total_experiment_seconds'] is None and c['teacher_only_seconds'] is None and c['offline_teacher_cache_this_study']['used'] is False,'Overclaimed total/teacher/cache cost')
    if phase=='val':
        old=a['efficiency'];require(old['status']==eff['status'] and old['crossings']==eff['crossings'],'Observed crossing differs')
        for arm in ARMS:
            x=eff['speedups_vs_baseline'][arm];y=old['speedups_vs_baseline'][arm]
            if x is None:require(y is None,'Unsupported acceleration claim')
            else:same_number(x,y,'Observed speedup')
            same_number(eff['final2048_seconds_ratio_to_baseline'][arm],old['final2048_seconds_ratio_to_baseline'][arm],'Time ratio')
        for b in old['equal_time_budget_observed_points']:
            u=str(b['baseline_update_defining_budget'])
            require({arm:b['arms'][arm]['latest_observed_update_within_budget'] for arm in ARMS}==eff['equal_time_updates'][u],'Equal time point differs')
    else:require(a['efficiency']['status']=='not_estimated_from_test','Test used to estimate acceleration')


def audit(val_run,val_verification,test_run,test_verification,val_analysis,test_analysis):
    val_run=Path(val_run).resolve();test_run=Path(test_run).resolve()
    states=complete_statuses(val_run,test_run)
    vp,vv=header(val_run,'val',states['val'],val_verification);tp,tv=header(test_run,'test',states['test'],test_verification)
    pairing_lock(vp,tp,val_run,test_run,vv)
    training,times=training_costs(vp)
    draws=np.random.Generator(np.random.PCG64(20261102)).integers(0,32,size=(10000,32),dtype=np.int64)
    dh=hashlib.sha256(draws.astype('<i8',copy=False).tobytes()).hexdigest();phases={};all_sources=[]
    for phase,root,p,ap in [('val',val_run,vp,val_analysis),('test',test_run,tp,test_analysis)]:
        rec,src,vectors,curves,cost=phase_rows(root,p,states[phase],times)
        comparisons=differences(vectors,POINTS if phase=='val' else (2048,),draws);quality=verdict(comparisons)
        eff=efficiency(curves) if phase=='val' else {'status':'not_estimated_from_test'}
        compare_phase(phase,rec,curves,comparisons,quality,eff,cost,training,ap,states[phase]['protocol_sha256'],digest(root/'observations.json'),dh)
        phases[phase]={'records':rec,'sources':src,'curves':curves,'paired_comparisons':comparisons,'quality_verdict':quality,'efficiency':eff,'evaluation_cost':cost,
            'protocol':bind(root/'protocol.json'),'observations':bind(root/'observations.json'),'analysis_compared':bind(ap)}
        all_sources.append(set(src.values()))
    require(not (all_sources[0]&all_sources[1]),'Val/test source overlap')
    return {'schema':'wan_trd_independent_final_statistics_review_v1','status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'review_source':bind(__file__),'numpy_version':np.__version__,'observations_recomputed':832,'sampler_averaged_clip_role_rows':416,
        'metrics':list(METRICS),'contrasts':[x[0] for x in CONTRASTS],'paired_metric_intervals_checked':75,'bootstrap_draws_per_contrast_point':10000,
        'draw_indices_sha256':dh,'statistical_implementation':'Independent fsum mean/sample SD; manual sorted linear percentile; no analysis-helper imports or statistical calls.',
        'numeric_cross_implementation_parity_atol':PARITY_ATOL,'scientific_guard_tolerance':0.,'single_training_seed':20261102,
        'GPU_work_performed':False,'PNG_or_checkpoint_tensor_reread':False,'large_checkpoint_content_rehashed':False,'tensor_payload_loaded':False,
        'unique_trained_checkpoints_metadata_checked':12,'checkpoint_verification_scope':'Independent path/size plus exact arm/role/protocol metadata matching; contents/SHA and tensor identities rely on immediately preceding bound native trd-verify receipts. All source/log/metric/observation text SHA independently checked.',
        'native_media_verifications':{'val':bind(val_verification),'test':bind(test_verification)},
        'scientific_interpretation':'Exploratory single training seed, paired source-clip bootstrap conditional on this model; historical test exposure; no p-values.',
        'training_cost':training,'phases':phases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('val-run','val-verification','test-run','test-verification','val-analysis','test-analysis','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Refuse to overwrite existing audit')
    result=audit(a.val_run,a.val_verification,a.test_run,a.test_verification,a.val_analysis,a.test_analysis)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'output':str(a.output.resolve()),'sha256':digest(a.output),'observations':832,'intervals':75}))

if __name__=='__main__':main()
