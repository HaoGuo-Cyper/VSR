"""Completed fixed-teacher val/test analysis; preserves the HR-only prototype.

This unbound helper never trains, infers, decodes heldout RGB or changes a
selection receipt.  Eval's ft-select is the authority for opening test assets.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import statistics
from pathlib import Path
import numpy as np
import analyze_hr_vae_study as common

METRICS = common.METRICS
UPDATES = common.UPDATES
read, sha, need, finite = common.read, common.sha, common.need, common.finite
verify_file = common.verify_file


def stats(vector):
    values=list(map(float,vector))
    need(len(values)==32 and all(finite(v) for v in values),'Need32 finite paired clip values')
    return {'mean':statistics.mean(values),'clip_sample_SD_ddof1':statistics.stdev(values),'clips':32}


def _selection_choices(study, role_values):
    result = {}
    for family in ('hr_vae', 'hr_dino'):
        candidates = [a for a in study['arms'] if a['family'] == family]
        need(len(candidates) == 2, 'Both predeclared candidates per family must remain reported')
        selected = min(candidates, key=lambda a: (role_values[a['label']], a['weight']))
        result[family] = {'arm': selected['label'], 'setting': selected['setting'], 'weight': selected['weight'],
                          'mean_val_final_LPIPS': role_values[selected['label']],
                          'all_candidates': {a['label']: {'weight': a['weight'], 'LPIPS': role_values[a['label']]}
                                             for a in candidates}}
    return result


def _test_access_policy(study):
    exception=study['evaluation'].get('test_integrity_thumbnail_exception')
    if exception is not None:
        need(exception.get('scope')=='integrity_only' and exception.get('automated') is True
             and exception.get('human_quality_review') is False and exception.get('model_scores_used') is False,
             'Test thumbnail exception changed from automated integrity-only access')
        verify_file(exception['audit_receipt'])
    return {'test_pixels_after_selection_scope':'formal_RGB_evaluation_assets_and_model_scores',
            'test_integrity_thumbnail_exception':exception,
            'no_preselection_test_pixel_access_claimed':False}


def _verify_test_selection(protocol, study):
    receipt = read(verify_file(protocol['selection_receipt']))
    need(receipt == protocol['validation_selection'] and receipt['study_file'] == protocol['study_file']
         and receipt['selected_after_complete_validation'] is True
         and receipt['test_pixels_read_by_selection'] is False, 'Test selection provenance changed')
    if study['evaluation'].get('test_integrity_thumbnail_exception') is not None:
        need(all(receipt.get(k)==v for k,v in _test_access_policy(study).items()),
             'Test selection omitted or changed the bound integrity-only access disclosure')
    for key in ('validation_protocol', 'validation_status', 'validation_summary', 'validation_observations'):
        verify_file(receipt[key])
    vp = read(receipt['validation_protocol']['path']); vb = read(receipt['validation_observations']['path'])
    need(vp['phase'] == 'val' and vp['scope']['total_restores'] == 1024 and len(vb['observations']) == 1024,
         'Test requires the full prospective validation study')
    records = [x['record']['id'] for x in vp['cohort']]
    index = {(r['role'], r['record_id'], r['sampling_seed']): r for r in vb['observations']}
    need(len(index) == 1024, 'Validation selection has duplicate observations')
    final_roles = {r['arm']: r for r in vp['roles'] if r['update'] == 2048}
    means = {arm: statistics.mean(statistics.mean(index[role['label'], rec, s]['metrics']['lpips_alex_v0_1']
                                         for s in (42, 43)) for rec in records)
             for arm, role in final_roles.items()}
    choices = _selection_choices(study, means)
    selected = {r['family']: r for r in receipt['selected_roles']}
    baseline = next(a['label'] for a in study['arms'] if a['family'] == 'baseline')
    need(set(selected) == {'baseline', 'hr_vae', 'hr_dino'} and selected['baseline']['arm'] == baseline,
         'Test must retain baseline and one selected student per teacher family')
    for family, choice in choices.items():
        need(selected[family]['arm'] == choice['arm']
             and selected[family]['checkpoint'] == final_roles[choice['arm']]['checkpoint'],
             'Test weight/CP was not selected solely from final validation LPIPS')
    for role in protocol['roles']:
        need(role['checkpoint'] == selected[role['family']]['checkpoint']
             and role['arm'] == selected[role['family']]['arm'] and role['update'] == role['step'] == 2048,
             'Test changed the validation-selected final checkpoint')
    return choices, means



def _cost_accounting(training, study, run, protocol_sha, phase):
    """Keep the convergence axis separate from overlapping end-to-end costs."""
    def bound_json(item):
        need(Path(item['path']).suffix=='.json','Cost receipts must be explicitly bound JSON')
        return read(verify_file(item))
    files=training.get('files',{})
    summary_binding=files.get('summary.json')
    training_summary=bound_json(summary_binding) if summary_binding else {}
    protocol_binding=files.get('protocol.json')
    training_protocol=bound_json(protocol_binding) if protocol_binding else {}
    calibration=bound_json(study['calibration'])
    cache_binding=training_protocol.get('dino_cache',{}).get('completion_receipt')
    cache=bound_json(cache_binding) if cache_binding else {}
    wall_path=Path(run)/'execution_wall_time.json'
    wall_binding=wall_receipt=None
    if wall_path.exists():
        wall_binding={'path':str(wall_path.resolve()),'bytes':wall_path.stat().st_size,'sha256':sha(wall_path)}
        wall_receipt=bound_json(wall_binding)
        need(wall_receipt.get('schema')=='fixed_teacher_execution_wall_v1'
             and wall_receipt.get('status')=='completed'
             and wall_receipt.get('stage')=='evaluation_'+phase
             and Path(wall_receipt['run']).resolve()==Path(run).resolve()
             and wall_receipt['protocol_sha256']==protocol_sha
             and finite(wall_receipt['wall_seconds']) and wall_receipt['wall_seconds']>=0
             and isinstance(wall_receipt.get('scope'),str) and wall_receipt['scope'],
             'Evaluation execution wall receipt changed its run/protocol/scope')
    ledger=training_summary.get('compute_cost')
    if ledger is not None:
        need(all(not isinstance(v,(int,float)) or finite(v) and v>=0 for v in ledger.values()),
             'Training cost ledger contains negative/nonfinite durations or counts')
    return {
        'training_attempts':{'source':summary_binding,'compute_cost':ledger,
            'available':ledger is not None,
            'per_arm':{arm:{k:record.get(k) for k in ('cumulative_training_seconds','updates','head_parameters','memory')}
                       for arm,record in training_summary.get('arms',{}).items()}},
        'train_only_calibration':{'source':study['calibration'],
            'preflight_wall_seconds_including_metadata_io':calibration.get('preflight_wall_seconds_including_metadata_io'),
            'gradient_calibration_wall_seconds_including_fresh_model_load':calibration.get('gradient_calibration_wall_seconds_including_fresh_model_load'),
            'gradient_wall_is_a_subcomponent_of_preflight_wall':True,
            'diagnostics':{key:calibration.get(key) for key in ('calibrations','resources',
                'relative_timed6_seconds_vs_baseline','online_cache_checks','HR_VAE_reconstruction_diagnostics',
                'all_step_rows','diagnostic_optimizer_updates','both_heads_full_resume_exact',
                'original_baseline_exact_GPU_updates','diagnostic_weights_reused',
                'validation_or_test_scores_used_for_weights')}},
        'offline_DINO_cache':{'source':cache_binding,'timings':cache.get('timings'),
            'VAE_cache_cost':cache.get('VAE_cache_cost'),'available':bool(cache),
            'receipt_metadata':{key:cache.get(key) for key in ('training_count','validation_count',
                'teacher_frames','optimizer_updates','test_pixels_accessed','feature_tensor_bytes',
                'serialized_feature_bytes','teacher_weight_bytes','teacher_precision','GPU_memory',
                'new_VAE_feature_extraction_seconds_this_round')}},
        'evaluation_execution':{'source':wall_binding,'receipt':wall_receipt,'available':wall_receipt is not None,
            'not_inferred_from':'file timestamps, prediction speed or training curves'},
        'common_Base_and_original_VAE_cache':'Existing shared sunk preparation; excluded from the continuation curve, not zero-cost claims.',
        'checkpoint_and_evaluation_IO':'Not on the convergence axis. Included only where a bound wall receipt explicitly covers it; no separate duration is invented.',
        'overall_total_seconds':None,
        'aggregation_rule':'Scopes overlap: gradient calibration is inside preflight wall; committed/wasted steps are inside training attempt wall. Do not sum containing totals and their subcomponents.',
        'missing_costs_are_unknown_not_zero':True}


TIME_AXIS_CONTRACT={
    'field':'cumulative_training_seconds','unit':'seconds',
    'meaning':'Committed synchronized training-step wall time from the current canonical log; successful returned steps only.',
    'includes':['H2D inside the step timer','student/head forward and backward','finite checks','gradient clipping','AdamW','step cleanup'],
    'excludes':['Common Base8192 initialization','offline teacher/cache preparation','train-only calibration',
        'model loading','CPU cache/data loading before the timer','checkpoint save/load and export IO',
        'evaluation/metric/media IO','discarded/replayed work outside the committed curve','service and admission overhead'],
    'pure_GPU_elapsed':False,'pure_kernel_time':False,'total_experiment_cost':False,
    'checkpoint_crossings_are_committed_step_wall_comparisons_not_end_to_end_speedups':True}


def compute(run):
    run = Path(run).resolve(); state = read(run / 'status.json')
    need(state.get('status') == 'completed' and state.get('training_updates') == 0,
         'Only a completed fixed-teacher phase can be analyzed')
    protocol = read(run / 'protocol.json'); phase = protocol.get('phase')
    expected = 1024 if phase == 'val' else 192 if phase == 'test' else None
    need(expected is not None and protocol.get('schema') == 'fixed_teacher_clean_evaluation_protocol_v1'
         and state['restores'] == protocol['scope']['total_restores'] == expected
         and state['protocol_sha256'] == sha(run / 'protocol.json'), 'Phase/schema/full matrix mismatch')
    summary = read(run / 'summary.json')
    need(state['summary_sha256'] == sha(run / 'summary.json')
         and summary['protocol_sha256'] == state['protocol_sha256'] and summary['restores'] == expected,
         'Completion summary chain changed')
    need(len(protocol['sources']) == 11, 'Bind all11 runtime source files')
    for item in protocol['sources'].values(): verify_file(item)
    study = read(verify_file(protocol['study_file'])); arms = study['arms']
    need(study['schema'] == 'fixed_teacher_study_v1' and len(arms) == 5
         and [a['family'] for a in arms] == ['baseline','hr_vae','hr_vae','hr_dino','hr_dino'],
         'Report baseline and both frozen settings of each teacher family')
    verify_file(study['calibration'])
    access_policy=_test_access_policy(study)
    if access_policy['test_integrity_thumbnail_exception'] is not None:
        need(all(protocol['data_independence'].get(k)==v for k,v in access_policy.items()),
             'Evaluation omitted or changed the bound integrity-only access disclosure')
    training = protocol['training_evidence']
    need(training['scientific_updates'] == len(arms) * study['updates'] == 10240
         and training['study_file'] == protocol['study_file'], 'Study training budget/evidence differs')
    for item in training['files'].values(): verify_file(item)
    verify_file(protocol['cohort_seal'])
    cohort = protocol['cohort']; records = [x['record']['id'] for x in cohort]
    source = {x['record']['id']: x['record']['source_id'] for x in cohort}
    need(len(records) == len(set(records)) == len(set(source.values())) == 32,
         'Require32 distinct clips/sources')
    need(protocol['scope']['source_split'] == phase and protocol['scope']['conditions'] == ['clean']
         and protocol['scope']['sampling_seeds'] == [42,43]
         and protocol['scope']['primary_pool'] == 'all33_frames', 'Clean source/metric scope changed')
    roles = {r['label']: r for r in protocol['roles']}
    need(len(roles) == (16 if phase == 'val' else 3), 'Missing checkpoint roles')
    selected, validation_means = _verify_test_selection(protocol,study) if phase == 'test' else (None,None)
    bundle = read(run / 'observations.json'); rows = bundle['observations']
    need(bundle['protocol_sha256'] == state['protocol_sha256'] and len(rows) == expected, 'Incomplete observations')
    index = {};restore_seconds=[]
    for row in rows:
        key = row['role'], row['record_id'], row['sampling_seed']
        need(key not in index and key[0] in roles and key[1] in source and key[2] in (42,43), 'Duplicate/foreign phase row')
        role = roles[key[0]]
        need(row['split'] == phase and row['condition'] == 'clean' and row['source_id'] == source[key[1]]
             and row['training_seed'] == 20261102 and row['cohort_seal_digest'] == protocol['cohort_seal_digest']
             and row['checkpoint_sha256'] == role['checkpoint']['sha256']
             and all(row[k] == role[k] for k in ('arm','family','setting','weight','update','step','cumulative_training_seconds')),
             'Row family/split/checkpoint identity differs')
        need(all(finite(row['metrics'][m]) for m in METRICS), 'Nonfinite phase metric')
        path = Path(row['row_metrics_path']).resolve()
        need(path.is_relative_to(run/'results') and sha(path) == row['row_metrics_sha256'], 'Bound row JSON changed')
        raw=read(path)
        need(raw['observation'] == {k:v for k,v in row.items() if not k.startswith('row_metrics_')},
             'Canonical row differs from raw row')
        if 'restore_seconds' in raw:
            need(finite(raw['restore_seconds']) and raw['restore_seconds']>=0,'Invalid measured restore duration')
            restore_seconds.append(raw['restore_seconds'])
        index[key] = row['metrics']
    need(set(index) == {(r,rec,s) for r in roles for rec in records for s in (42,43)}, 'Missing full phase observations')
    baseline = arms[0]['label']; arm_info = {a['label']: a for a in arms}
    present = [a['label'] for a in arms if a['label'] in {r['arm'] for r in roles.values()}]
    if phase == 'val': present = [a['label'] for a in arms]
    updates = UPDATES if phase == 'val' else (2048,)
    shared = [r for r in roles.values() if r['update'] == 0]
    need((len(shared)==1) if phase=='val' else not shared, 'Shared initial role belongs only to validation')
    by_point = {(r['arm'],r['update']):r for r in roles.values() if r['update']}
    need(set(by_point)=={(a,u) for a in present for u in updates if u}, 'Missing phase checkpoint point')
    curves, vectors = {}, {}
    for arm in present:
        curves[arm] = {}; previous=-1.
        for update in updates:
            role=shared[0] if update==0 else by_point[arm,update]
            seconds=role['cumulative_training_seconds']
            need(finite(seconds) and seconds>=0 and seconds>previous, 'Invalid cumulative training seconds')
            previous=seconds
            values=np.asarray([[statistics.mean(index[role['label'],rec,s][m] for s in (42,43))
                                for m in METRICS] for rec in records],dtype=np.float64)
            vectors[arm,update]=values
            curves[arm][update]={'role':role['label'],'family':arm_info[arm]['family'],'setting':arm_info[arm]['setting'],
                 'weight':arm_info[arm]['weight'],'update':update,'cumulative_training_seconds':seconds,
                 'metrics':{m:stats(values[:,j]) for j,m in enumerate(METRICS)}}
    indices=np.random.Generator(np.random.PCG64(20261102)).integers(0,32,size=(10000,32),dtype=np.int64)
    comparisons={}
    methods=[a for a in present if a!=baseline]
    def contrast(left,right,*,method_label='method',reference_label='baseline'):
        difference=left-right
        for j,m in enumerate(METRICS):
            if m not in common.HIGHER_BETTER:difference[:,j]*=-1
        ci=np.quantile(difference[indices].mean(axis=1),[.025,.975],axis=0,method='linear')
        return {m:stats(difference[:,j])|{'paired_clip_95pct_CI':list(map(float,ci[:,j])),
                 'difference':method_label+'-minus-'+reference_label if m in common.HIGHER_BETTER else reference_label+'-minus-'+method_label,
                 'positive_favors_method':True} for j,m in enumerate(METRICS)}
    for arm in methods:
        comparisons[arm]={u:contrast(vectors[arm,u],vectors[baseline,u]) for u in updates}
    final={}
    for arm in methods:
        result=comparisons[arm][2048];lo,hi=result['lpips_alex_v0_1']['paired_clip_95pct_CI']
        final[arm]={'family':arm_info[arm]['family'],'setting':arm_info[arm]['setting'],'weight':arm_info[arm]['weight'],
            'LPIPS_status':'single_training_seed_LPIPS_improvement' if lo>0 else 'no_observed_gain' if hi<=0 else 'insufficient_evidence',
            'LPIPS_baseline_minus_method':result['lpips_alex_v0_1'],
            'PSNR_method_minus_baseline':result['psnr_rgb_db'],'SSIM_method_minus_baseline':result['ssim_rgb'],
            'PSNR_fidelity_tradeoff':result['psnr_rgb_db']['mean']<0,'SSIM_fidelity_tradeoff':result['ssim_rgb']['mean']<0}
    if phase=='val':
        means={arm:curves[arm][2048]['metrics']['lpips_alex_v0_1']['mean'] for arm in present}
        validation_means=means
        selected=_selection_choices(study,means)
        target=curves[baseline][2048]['metrics']['lpips_alex_v0_1']['mean']
        efficiency=common.first_target_crossings(curves,baseline,target)
        efficiency['final2048_training_seconds_ratio_method_over_baseline']={
            a:curves[a][2048]['cumulative_training_seconds']/curves[baseline][2048]['cumulative_training_seconds'] for a in methods}
    else:efficiency={'status':'not_estimated_from_test','reason':'Only frozen finalCPs were tested. Use validation checkpoint curves for training efficiency.'}
    vae,dino=selected['hr_vae']['arm'],selected['hr_dino']['arm']
    teacher_contrast=contrast(vectors[vae,2048],vectors[dino,2048],method_label='hr_vae',reference_label='hr_dino')
    per_clip=[]
    for role in roles.values():
        for rec in records:
            per_clip.append({'phase':phase,'condition':'clean','role':role['label'],
                **{k:role[k] for k in ('arm','family','setting','weight','update','step','cumulative_training_seconds')},
                'record_id':rec,'source_id':source[rec],'training_seed':20261102,'sampling_seeds':'42,43',
                'sampler_count':2,'checkpoint_sha256':role['checkpoint']['sha256'],
                **{m:statistics.mean(index[role['label'],rec,s][m] for s in (42,43)) for m in METRICS}})
    paired=[]
    def append_paired(method,reference,update,comparison):
        difference=vectors[method,update]-vectors[reference,update]
        for j,m in enumerate(METRICS):
            if m not in common.HIGHER_BETTER:difference[:,j]*=-1
        left=curves[method][update];right=curves[reference][update]
        for i,rec in enumerate(records):
            paired.append({'phase':phase,'comparison':comparison,'method_arm':method,'reference_arm':reference,
                'method_role':left['role'],'reference_role':right['role'],'update':update,
                'method_family':arm_info[method]['family'],'method_setting':arm_info[method]['setting'],
                'method_weight':arm_info[method]['weight'],'reference_weight':arm_info[reference]['weight'],
                'record_id':rec,'source_id':source[rec],'positive_favors_method':True,
                **{m:float(difference[i,j]) for j,m in enumerate(METRICS)}})
    for arm in methods:
        for update in updates:append_paired(arm,baseline,update,'method_vs_baseline')
    append_paired(vae,dino,2048,'selected_hr_vae_vs_selected_hr_dino')
    candidate_settings=[dict(arm=a['label'],family=a['family'],setting=a['setting'],weight=a['weight'],
        selected_for_test=(a['family']=='baseline' or selected[a['family']]['arm']==a['label']),
        evaluated_in_this_phase=a['label'] in present,selection_update=2048,
        mean_val_final_LPIPS=validation_means[a['label']]) for a in arms]
    costs=_cost_accounting(training,study,run,sha(run/'protocol.json'),phase)
    costs['evaluation_execution']['canonical_successful_restore_seconds_sum']=math.fsum(restore_seconds) if len(restore_seconds)==expected else None
    costs['evaluation_execution']['restore_duration_rows']=len(restore_seconds)
    costs['evaluation_execution']['restore_sum_scope']='Canonical successful restore_video calls only; excludes model loads, metrics, PNG/MP4 I/O, verification and failed/discarded attempts; not pure GPU time.'
    return {'schema':'fixed_teacher_descriptive_analysis_v1','created_UTC':datetime.now(timezone.utc).isoformat(),
        'phase':phase,'run':str(run),'protocol_sha256':sha(run/'protocol.json'),
        'observations_sha256':sha(run/'observations.json'),'summary_sha256':sha(run/'summary.json'),
        'study_file':protocol['study_file'],'calibration':study['calibration'],'runtime_sources':protocol['sources'],
        'analysis_sources':{'generic':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
                            'shared_HR_prototype':{'path':str(Path(common.__file__).resolve()),'sha256':sha(common.__file__)}},
        'method_training_seed':20261102,'sampling_seeds':[42,43],'matrix_rows':expected,'scientific_training_updates':10240,
        'metric_names':list(METRICS),'paired_clips':records,'paired_source_ids':source,
        'aggregation':'Withinclip sampler42/43 mean, then equal32distinct sources; SD is clipSD, not trainingseedSD.',
        'time_axis_contract':TIME_AXIS_CONTRACT,'cost_accounting':costs,
        'curves':curves,'paired_comparisons':comparisons,'final_comparisons':final,'selected_by_family_from_validation':selected,
        'per_clip_sampler_means':per_clip,'paired_clip_differences':paired,'candidate_settings':candidate_settings,
        'paired_difference_signs':{m:'method-minus-reference' if m in common.HIGHER_BETTER else 'reference-minus-method' for m in METRICS},
        'selected_VAE_versus_selected_DINO':{'positive_favors':'HR-VAE','metrics':teacher_contrast,
            'sign':'VAE-minus-DINO for PSNR/SSIM; DINO-minus-VAE for lower-is-better metrics',
            'postselection_descriptive':phase=='val'},'efficiency':efficiency,
        'data_independence':protocol['data_independence'],'test_access_policy':access_policy,
        'historically_pristine_test':False,
        'selection_recomputed_from_validation_only':True,'selection_receipt_generated_by_this_helper':False,
        'bootstrap':{'draws':10000,'seed':20261102,'generator':'PCG64',
          'draw_indices_sha256':__import__('hashlib').sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest(),
          'interval':'95%linear-percentile','common_paired_clip_draws':True,'p_values_computed':False,
          'scope':'descriptive conditional on one trained seed; no selection/multiplicity correction'},
        'limitations':['Weights are chosen only on the study-bound validation cohort. Evaluation sources excluded or replaced before training/scoring are determined by the bound data-integrity audit, not model quality. Formal test RGB evaluation assets and model scores are opened after this-round selection. A bound preselection exception permits automated integrity-only thumbnails without human quality review or model scores; this is not a claim of no earlier test-pixel access. Old models previously evaluated this test split, so it is not a historically pristine holdout.',
          'One training seed; paired clip intervals and sampler repetitions are not independent training repeats or a multi-seed significance test.',
          'Full selected33-frame128x224 crops,x4 clean only, not entire original source videos or full-resolution/external evaluation.',
          'Validation efficiency uses only observed checkpoints, no interpolation. Shared initial already meeting the target makes acceleration uninterpretable.',
          'Time axis is committed synchronized training-step wall seconds, not pure GPU elapsed/kernel time or total experiment cost. Offline DINO/cache preparation, calibration, model/data loading, checkpoint/evaluation IO and discarded/replayed work are reported separately where a bound receipt covers them; missing costs are unknown, not zero.',
          'LPIPS gains and PSNR/SSIM fidelity tradeoffs are separately reported; no old EMA gate is inherited.'],
        'GPU_work_performed_by_analysis':False,'heldout_test_used_to_choose_weights':False}


def markdown(report):
    phase=report['phase'];lines=['# 固定teacher对照：baseline / HR-VAE / HR-DINO', '',
        f'阶段：**{phase}**；完整{report["matrix_rows"]}恢复，5条训练轨迹共10240次更新=1.25RE。方法训练seed只有20261102。', '',
        '**主指标：all33 LPIPS；表内SD为32个clip sample SD，不是训练seed SD。** 每clip先平均采样42/43，再等权平均不同source。', '',
        '| Family / setting / λ | 更新 | 已提交step墙钟秒 | LPIPS↓ | PSNR↑ | SSIM↑ | tLP100↓ | tOF↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,points in report['curves'].items():
        for update,point in points.items():
            cells=[f'{point["metrics"][m]["mean"]:.5f} ± {point["metrics"][m]["clip_sample_SD_ddof1"]:.5f}' for m in METRICS]
            lines.append(f'| {point["family"]}/{point["setting"]}/{point["weight"]} | {update} | {point["cumulative_training_seconds"]:.3f} | '+' | '.join(cells)+' |')
    lines+=['', '## 各family的validation选择', '']
    for family,choice in report['selected_by_family_from_validation'].items():
        lines.append(f'- {family}：{choice["arm"]}，λ={choice["weight"]}；最终2048 val LPIPS={choice["mean_val_final_LPIPS"]:.7f}。各family独立选最低值，精确并列选小weight。')
    lines+=['','**test阶段不重新选择权重或checkpoint。原test来源被历史模型评过，不能称pristine/blind历史test。**',
            '正式test RGB评价资产与模型评分在val选择后开启；选择前允许绑定记录的自动完整性缩略图筛查，无人工质量阅图、无模型分数。不能声称本轮此前从未访问test像素。','',
            '## Final2048相对baseline的配对描述区间','',
            'LPIPS取baseline−method，PSNR/SSIM取method−baseline；负PSNR/SSIM明确表示保真tradeoff。10000次PCG64配对clip、95%linear-percentile CI，仅描述本训练seed。','',
            '| Arm | LPIPS收益 [95%CI] | PSNR差 [95%CI] | SSIM差 [95%CI] | 结论 |',
            '| --- | ---: | ---: | ---: | --- |']
    labels={'single_training_seed_LPIPS_improvement':'本训练seed有LPIPS改善','no_observed_gain':'未见增益','insufficient_evidence':'证据不足'}
    for arm,result in report['final_comparisons'].items():
        cells=[]
        for key in ('LPIPS_baseline_minus_method','PSNR_method_minus_baseline','SSIM_method_minus_baseline'):
            item=result[key];lo,hi=item['paired_clip_95pct_CI'];cells.append(f'{item["mean"]:.7f} [{lo:.7f}, {hi:.7f}]')
        lines.append(f'| {arm} | '+' | '.join(cells)+f' | {labels[result["LPIPS_status"]]} |')
    lines+=['','## 质量与效率分开解释','']
    eff=report['efficiency']
    if eff['status']=='not_estimated_from_test':lines.append('Test仅测冻结后的最终3个checkpoint，不用test推断收敛速度或重新调参；训练效率见val checkpoint曲线。')
    elif eff['status']=='uninterpretable_initial_already_meets_target':lines.append('共享step0已达到baseline@2048的LPIPS目标，无法据此判断方法加速。')
    else:
        lines.append(f'LPIPS固定目标=baseline@2048={eff["target_LPIPS"]:.7f}；只认最早实际checkpoint，不插值。')
        for arm,item in eff['crossings'].items():
            lines.append(f'- {arm}：'+('已测点均未达到目标。' if item['update'] is None else f'最早实测{item["update"]}步，{item["cumulative_training_seconds"]:.3f}同步训练秒。'))
    lines+=['', '**时间轴只表示已提交日志中的同步训练step墙钟秒，既不是纯GPU elapsed/kernel时间，也不是总实验花费。**',
            '离线teacher/cache、校准、模型/数据载入、checkpoint与评价I/O、丢弃或重放attempt另列；未知不记为0。以下wall总量有包含关系，不能与其子项相加。', '']
    costs=report['cost_accounting'];training_cost=costs['training_attempts']['compute_cost']
    if training_cost is not None:
        lines.append('- 绑定训练attempt成本：`'+json.dumps(training_cost,ensure_ascii=False,sort_keys=True)+'`。')
    else:lines.append('- 训练attempt/浪费成本：绑定记录未提供独立ledger，无法由step曲线推算。')
    calibration_cost=costs['train_only_calibration']
    lines.append('- 校准preflight墙钟秒：'+str(calibration_cost['preflight_wall_seconds_including_metadata_io'])+'；各method梯度校准子项：`'+json.dumps(calibration_cost['gradient_calibration_wall_seconds_including_fresh_model_load'],ensure_ascii=False)+'`。')
    cache_cost=costs['offline_DINO_cache']
    lines.append('- 离线DINO/cache时序：`'+json.dumps(cache_cost['timings'],ensure_ascii=False,sort_keys=True)+'`；receipt中的scope决定覆盖范围。')
    execution=costs['evaluation_execution']
    if execution['available']:
        lines.append('- 本阶段评价execute墙钟秒：'+str(execution['receipt']['wall_seconds'])+'；'+execution['receipt']['scope'])
    else:lines.append('- 本阶段评价execute总墙钟未提供绑定receipt，不能从文件时间戳或逐次恢复耗时推算；未知不记为0。')
    lines+=['','## 逐视频表与完整交付','',
        '- [逐clip五指标CSV](per_clip_metrics.csv)：每个实际role每clip先平均采样42/43；共享step0仅一份，不复制为五条独立结果。',
        '- [逐clip配对差CSV](paired_clip_differences.csv)：保留全部候选/更新及所选VAE对DINO；PSNR/SSIM为method−reference，其余为reference−method，正值均有利于method。',
        '- [role汇总CSV](summary.csv)、[全部配对区间CSV](paired_summary.csv)、[全部候选设置CSV](candidate_settings.csv)。',
        '- [成本CSV](costs.csv)与[校准/梯度/资源/重建诊断CSV](calibration_diagnostics.csv)：保留原绑定来源和覆盖范围；包含关系的总量与子项不能相加。',
        '- [JSON](analysis.json)保留完整数组与统计；[交付hash清单](delivery_manifest.json)绑定全部CSV、JSON、报告及曲线。']
    lines+=['','![质量vs更新](quality_vs_updates.png)','','![质量vs已提交step墙钟时间](quality_vs_training_seconds.png)','','## 局限','']
    lines+=['- '+x for x in report['limitations']]
    lines+=['',f'Protocol SHA：`{report["protocol_sha256"]}`；study SHA：`{report["study_file"]["sha256"]}`。完整逐step五指标区间、全部候选和source记录见[JSON](analysis.json)。']
    return '\n'.join(lines)+'\n'


def plots(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    palette=['#222222','#0072B2','#56B4E9','#D55E00','#E69F00'];markers=['o','s','^','D','v']
    panel=(('lpips_alex_v0_1','All33 LPIPS, lower better'),('psnr_rgb_db','All33 RGB PSNR, higher better'),
           ('tLP100_alex_v0_1_TecoGAN_crop','Original-pair tLP100, lower better'))
    for time_axis,name in ((False,'quality_vs_updates.png'),(True,'quality_vs_training_seconds.png')):
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
        for i,(arm,points) in enumerate(report['curves'].items()):
            updates=sorted(points);xs=[points[u]['cumulative_training_seconds'] if time_axis else u for u in updates]
            for axis,(metric,title) in zip(axes,panel):
                axis.plot(xs,[points[u]['metrics'][metric]['mean'] for u in updates],marker=markers[i],color=palette[i],
                    linestyle='-' if len(updates)>1 else 'None',label=f'{arm} lambda={points[updates[-1]]["weight"]}')
                axis.set_title(title,fontsize=10);axis.grid(alpha=.25)
                axis.set_xlabel('Committed synchronized step wall seconds (not total cost)' if time_axis else 'Additional optimizer updates')
                if not time_axis:axis.set_xticks(updates)
        axes[0].legend(fontsize=7)
        note='Lines guide the eye; no efficiency interpolation' if report['phase']=='val' else 'Final-checkpoint test points only; no convergence inference'
        fig.suptitle(f'Fixed teachers | {report["phase"]}32 | 1 training seed | {note}',fontsize=10)
        fig.savefig(destination/name,dpi=180);plt.close(fig)


def _cost_rows(report):
    costs=report['cost_accounting'];rows=[]
    def add(category,measurement,value,source,scope,*,arm='',unit='seconds',overlap=''):
        if value is not None:
            need(finite(value) and value>=0,'Cost table contains a nonfinite/negative value')
        rows.append({'phase':report['phase'],'category':category,'arm':arm,'measurement':measurement,
            'value':value,'unit':unit,'available':value is not None,'pure_GPU_time':False,
            'scope':scope,'overlap_or_contains':overlap,'source_path':source['path'] if source else '',
            'source_sha256':source['sha256'] if source else ''})
    training=costs['training_attempts'];ledger=training['compute_cost'] or {};source=training['source']
    for key,scope,unit in [
        ('committed_training_step_seconds','Canonical retained successful synchronized training steps, across all five arms.','seconds'),
        ('known_discarded_completed_step_seconds','Recorded successful returned steps discarded by checkpoint rollback; not convergence progress.','seconds'),
        ('known_discarded_completed_step_returns','Known discarded completed optimizer step returns, not additional retained progress.','updates'),
        ('wall_seconds_all_training_attempts','Worker-recorded training attempt timers through outcome/summary creation, including failed/redo steps and loads/I/O inside those timers. Final cleanup, earlier protocol validation and launcher/service overhead are not necessarily covered.','seconds')]:
        add('training',key,ledger.get(key),source,scope,unit=unit,
            overlap='Committed and wasted step times are subcomponents of training attempt wall; never sum them into that wall total.')
    for arm,record in training['per_arm'].items():
        add('training','committed_training_step_seconds',record.get('cumulative_training_seconds'),source,
            'This arm retained synchronized training-step wall time.',arm=arm,
            overlap='Subcomponent of all-arm committed total and all-attempt wall.')
    calibration=costs['train_only_calibration'];source=calibration['source']
    add('calibration','preflight_wall_seconds_including_metadata_io',
        calibration['preflight_wall_seconds_including_metadata_io'],source,
        'Calibration worker timer after protocol revalidation through summary creation, including model/metadata I/O inside that timer; excludes earlier prepare/revalidation and final cleanup.',
        overlap='Contains gradient-calibration fresh-model-load subcomponents and resource diagnostics.')
    for kind,value in (calibration['gradient_calibration_wall_seconds_including_fresh_model_load'] or {}).items():
        add('calibration','gradient_calibration_wall_seconds_including_fresh_model_load',value,source,
            'Train-only gradient calibration for this teacher, including fresh model loading.',arm=kind,
            overlap='Subcomponent of calibration preflight wall; not an additional cost to add to it.')
    for arm,record in (calibration['diagnostics'].get('resources') or {}).items():
        for name in ('timed6_mean_seconds','timed6_sample_SD_seconds'):
            add('calibration_resource',name,record.get(name),source,
                'Six measured synchronized diagnostic steps after two warmup steps; diagnostic weights are discarded.',arm=arm,
                overlap='Diagnostic subset of calibration preflight; not formal training or pure GPU time.')
        for name in ('peak_allocated_bytes','peak_reserved_bytes'):
            add('calibration_resource',name,(record.get('memory') or {}).get(name),source,
                'Recorded diagnostic CUDA memory high-water mark.',arm=arm,unit='bytes')
    cache=costs['offline_DINO_cache'];timings=cache['timings'] or {};source=cache['source']
    for name,scope_key,fallback in [
        ('total_cache_build_seconds','total_scope','Offline DINO cache builder wall through completed seal.'),
        ('teacher_feature_extraction_seconds','feature_scope','Encode wall including resize/normalize/transfers, DINO forward and grouping.'),
        ('other_io_binding_seconds','other_scope','Other offline loading, hash, binding and publication work.'),
        ('pure_GPU_encoder_seconds',None,'Pure GPU encoder duration was not measured; unknown is not zero.')]:
        add('offline_DINO_cache',name,timings.get(name),source,timings.get(scope_key,fallback) if scope_key else fallback,
            overlap='Feature and other I/O durations partition cache builder wall; do not add both to their containing total.')
    metadata=cache['receipt_metadata']
    for name,unit in [('teacher_frames','frames'),('training_count','clips'),('validation_count','clips'),
                      ('feature_tensor_bytes','bytes'),('serialized_feature_bytes','bytes')]:
        add('offline_DINO_cache',name,metadata.get(name),source,'Actual offline cache receipt scope, including historical unused val extras.',unit=unit)
    add('common_VAE_cache','new_VAE_feature_extraction_seconds_this_round',
        metadata.get('new_VAE_feature_extraction_seconds_this_round'),source,
        'New extraction only; existing common VAE/Base preparation is sunk shared cost, not a zero-cost preparation claim.')
    evaluation=costs['evaluation_execution'];receipt=evaluation['receipt'] or {};source=evaluation['source']
    add('evaluation','execution_wall_seconds',receipt.get('wall_seconds'),source,
        receipt.get('scope','No completed bound execution wall receipt; not inferred from file timestamps or restore rates.'),
        overlap='Contains this execute attempt restore, metrics and media I/O; verify/prepare coverage only if explicitly recorded.')
    add('evaluation','canonical_successful_restore_seconds_sum',evaluation['canonical_successful_restore_seconds_sum'],
        {'path':str(Path(report['run'])/'observations.json'),'sha256':report['observations_sha256']},
        evaluation['restore_sum_scope'],overlap='Successful canonical restore calls are a subset of execute wall, not total evaluation cost.')
    return rows


def _write_csv(path, rows, fields=None):
    rows=list(rows)
    if fields is None:
        fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return {'file':Path(path).name,'rows':len(rows),'columns':fields}


def tables(report, destination):
    """Lossless sampler-mean per-video tables, all settings and separate costs."""
    outputs=[]
    outputs.append(_write_csv(destination/'per_clip_metrics.csv',report['per_clip_sampler_means']))
    outputs.append(_write_csv(destination/'paired_clip_differences.csv',report['paired_clip_differences']))
    outputs.append(_write_csv(destination/'candidate_settings.csv',report['candidate_settings']))
    grouped={}
    for row in report['per_clip_sampler_means']:grouped.setdefault(row['role'],[]).append(row)
    summaries=[]
    for role,clips in grouped.items():
        first=clips[0];item={key:first[key] for key in ('phase','role','arm','family','setting','weight','update',
            'step','cumulative_training_seconds','checkpoint_sha256')}
        item.update(clips=32,samplers_per_clip=2,SD_scope='clip sample SD, not training seed SD')
        for metric in METRICS:
            value=stats([c[metric] for c in clips]);item[metric+'_mean']=value['mean']
            item[metric+'_clip_SD']=value['clip_sample_SD_ddof1']
        summaries.append(item)
    outputs.append(_write_csv(destination/'summary.csv',summaries))
    pairs=[]
    for arm,points in report['paired_comparisons'].items():
        for update,metrics in points.items():
            for metric,value in metrics.items():
                pairs.append(dict(comparison='method_vs_baseline',method_arm=arm,update=update,metric=metric,
                    difference=value['difference'],mean=value['mean'],clip_SD=value['clip_sample_SD_ddof1'],
                    CI95_low=value['paired_clip_95pct_CI'][0],CI95_high=value['paired_clip_95pct_CI'][1],
                    interval_scope='descriptive pairedclip single training seed, no pvalue or multiplicity correction'))
    for metric,value in report['selected_VAE_versus_selected_DINO']['metrics'].items():
        pairs.append(dict(comparison='selected_hr_vae_vs_selected_hr_dino',method_arm=report['selected_by_family_from_validation']['hr_vae']['arm'],
            update=2048,metric=metric,difference=value['difference'],mean=value['mean'],clip_SD=value['clip_sample_SD_ddof1'],
            CI95_low=value['paired_clip_95pct_CI'][0],CI95_high=value['paired_clip_95pct_CI'][1],
            interval_scope='descriptive pairedclip single training seed, no pvalue or multiplicity correction'))
    outputs.append(_write_csv(destination/'paired_summary.csv',pairs))
    outputs.append(_write_csv(destination/'costs.csv',_cost_rows(report)))
    diagnostics=[];source=report['cost_accounting']['train_only_calibration']['source']
    def flatten(value,path):
        if isinstance(value,dict):
            for key,child in value.items():flatten(child,path+[str(key)])
        elif isinstance(value,list):
            for index,child in enumerate(value):flatten(child,path+[str(index)])
        else:diagnostics.append({'field_path':'.'.join(path),'value':value,
            'source_path':source['path'],'source_sha256':source['sha256'],
            'scope':'Train-only calibration diagnostic; VAE reconstruction is not an SR quality upper bound.'})
    flatten(report['cost_accounting']['train_only_calibration']['diagnostics'],[])
    outputs.append(_write_csv(destination/'calibration_diagnostics.csv',diagnostics))
    return outputs


def analyze(run, output_dir, *, make_plots=True):
    report=compute(run);destination=Path(output_dir).resolve()
    need(not destination.exists(),'Preserve previous reports; use a new output directory');destination.mkdir(parents=True)
    report['CSV_deliverables']=tables(report,destination)
    (destination/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (destination/'report.md').write_text(markdown(report))
    if make_plots:plots(report,destination)
    artifacts=[{'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(destination.iterdir()) if path.is_file()]
    (destination/'delivery_manifest.json').write_text(json.dumps({'protocol_sha256':report['protocol_sha256'],
        'study_file':report['study_file'],'artifacts':artifacts},indent=2,ensure_ascii=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv);dest=args.output_dir or common.WORKSPACE/'outputs'/('FixedTeacher_'+args.run.name)
    result=analyze(args.run,dest,make_plots=not args.no_plots)
    print(json.dumps({'output':str(dest),'phase':result['phase'],'selection':result['selected_by_family_from_validation']},indent=2))
    return result


if __name__=='__main__':main()
