#!/usr/bin/env python3
"""Independent stdlib-only review of a completed VideoMAEv2 preflight.

Reads text evidence and small source bindings, never imports torch/the model,
loads checkpoints, or rehashes tensor assets. Refuses running/failed runs and
existing output. Runtime equality claims are verified against bound receipts;
they are not new tensor comparisons or extra model forwards.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from datetime import datetime,timezone


def need(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):
    if isinstance(value,float):need(math.isfinite(value),'Nonfinite JSON number')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)
def binding(path):
    p=Path(path).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
def alignment(d,unit):
    need(d['margin']==.1 and d['normalize_eps']==1e-12,'Original objective changed')
    need(d['target_grid']==[16,8,14] and d['tokens_per_example']==1792 and d['all_pairs']==1792**2,'Full target grid/pairs changed')
    need(d['spatial_pairs']==16*112**2 and d['temporal_pairs']==15*16*112**2,'Pair reduction changed')
    need(d['diagonal_included'] and d['ordered_pairs'] and d['all_tokens_supervised'],'Pairs masked or excluded')
    for k in ('student_normalized_dtype','student_gram_dtype','projected_dtype','interpolated_dtype'):
        need(d[k]=='torch.bfloat16','Student original dtype changed: '+k)
    for k in ('teacher_normalized_dtype','teacher_gram_dtype','difference_dtype','loss_dtype'):
        need(d[k]=='torch.float32','Teacher/promotion dtype changed: '+k)
    need(d['projected_all_latent_positions']==9 and d['dropped_latent_positions']==[0],'Time mapping changed')
    need(0<=d['active_pairs']<=d['all_pairs'] and abs(d['active_pair_fraction']-d['active_pairs']/d['all_pairs'])<1e-7,'Active pairs inconsistent')
    # Only a floating-reduction accounting check; this tolerance is not used by
    # the positive-gradient gate, any scientific score, margin, or training.
    need(abs(unit-(d['spatial_loss_mean']+15*d['temporal_loss_mean'])/16)<2e-7,'Full-mean diagnostic reduction inconsistent')


def audit_documents(p,s,rows,bench,zero,resume,repeat):
    for value in (p,s,rows,bench,zero,resume,repeat):finite(value)
    need(p['schema']=='vsr_videomaev2_preflight_protocol_v1' and s['status']=='passed_videomaev2_preflight','Not a passed independent preflight')
    need(s['sources']==p['sources'],'Summary source chain differs')
    need(p['scientific_update_budget']==s['scientific_updates']==0 and p['diagnostic_optimizer_budget']==s['diagnostic_optimizer_updates']==26,'Optimizer budget changed')
    need(p['gradient_probe_cases']==s['gradient_probe_cases']==15 and p['gradient_VJP_calls']==s['gradient_VJP_calls']==30,'Probe/VJP budget changed')
    need(p['teacher_forward_budget']==s['teacher_forwards']==31,'Teacher budget changed')
    need(p['study']['arms']==[{'label':'baseline','weight':0.},{'label':'videomaev2','weight':.5}] and p['study']['student_layer']==17,'Arms/absolute depth changed')
    need(s['weight']==.5 and s['weight_calibrated'] is False and p['study']['margin']==.1 and p['study']['normalization_eps']==1e-12,'Fixed original defaults changed')
    need(p['online_teacher_forward'] and not p['feature_cache_created'] and not p['ordinary_inference_weight_EMA'],'Online fixed teacher changed')
    counters=s['optimizer_counter'];expected=[0,3,3,2,1,1,8,8]
    need(list(counters['by_role'])==['01_videomaev2','02_baseline','03_videomaev2','04_videomaev2','05_videomaev2','06_videomaev2','07_baseline','08_videomaev2'],'Diagnostic order changed')
    for role,n in zip(counters['by_role'].values(),expected):need(role==dict(attempted=n,applied=n,returned_steps=n),'Incomplete optimizer attempt')
    need(all(counters[k]==26 for k in ('diagnostic_optimizer_updates_attempted','diagnostic_optimizer_updates_applied','completed_step_returns')),'Counter totals differ')
    need(counters['optimizer_calls_not_returned_partial_effect_unknown']==0 and counters['scientific_training_updates']==0 and not counters['cleanup_events'],'Partial optimizer or cleanup failure')
    need(len(rows)==15 and rows==s['target_viability']['rows'],'Incomplete/different probe rows')
    need([(r['data_index'],r['nominal_sigma']) for r in rows]==[(i,t) for i in range(5) for t in (.2,.5,.8)],'Probe matrix reordered/dropped')
    for ordinal,r in enumerate(rows):
        need(all(r[k]>=0 for k in ('fm','unit_aux','fm_grad_norm','aux_grad_norm','head_grad_norm')),'Negative probe observation')
        need(r['VJP_calls']==2 and r['forced_inputs'] and r['noise_seed']==p['study']['training_seed']+101+ordinal,'Forced independent probe stream differs')
        need(r['teacher_forward'] and r['teacher_parameter_updates']==r['EMA_updates']==0 and not r['condition_corruption'] and r['student_layer']==17,'Teacher/student probe contract changed')
        need(0<=r['sigma']<=1 and 0<=r['timestep_index']<1000,'Native sigma/index invalid')
        cosine=r['FM_aux_gradient_cosine'];need(cosine is None or -1.0000001<=cosine<=1.0000001,'Gradient cosine invalid')
        alignment(r['alignment'],r['unit_aux'])
    positive=[r for r in rows if r['fm_grad_norm']>0 and r['aux_grad_norm']>0];levels=sorted({r['nominal_sigma'] for r in positive})
    gate=s['target_viability'];need(len(positive)>=8 and len(levels)>=2,'Independent positive-gradient viability failed')
    need(gate['eligible'] and gate['positive_gradient_cases']==len(positive) and gate['noise_levels_with_positive_gradients']==levels and gate['weight']==.5 and not gate['weight_calibrated'],'Gate summary differs from recomputation')
    need(gate['cases']==15 and gate['VJP_calls']==30 and gate['optimizer_updates']==0 and gate['minimum_positive_cases']==8 and gate['minimum_noise_levels']==2,'Gate definition changed')
    need(s['zero_weight_exact'] and zero['steps_per_arm']==3 and zero['exact_student_optimizer_4RNG_globalRNG_cursor'] and zero['teacher_forwards']==0 and zero['head_constructed'] is False,'Missing exact-zero proof')
    need(set(zero['rows'])=={'baseline','videomaev2'} and len(zero['rows']['baseline'])==3 and zero['rows']['baseline']==zero['rows']['videomaev2'],'Zero objective rows differ')
    need(resume==s['resume_reports']['videomaev2'] and resume['completed_steps']==2 and resume['exact_full_engine_state'] and resume['native_student_reload_exact'] and resume['native_restore_count']==0,'Missing complete-state/native reload proof')
    need(close(resume['loaded_prefix_seconds']+resume['new_step_seconds'],resume['resumed_cumulative_seconds']) and resume['independent_execution_wall_times_compared_for_equality'] is False,'Resume timing conflated')
    need(repeat['teacher_forwards']==4 and repeat['optimizer_updates']==0 and repeat['target_is_clean_HR'] and repeat['mapping']==p['teacher_binding'],'Teacher repeat scope changed')
    need(repeat['cases']==s['teacher_repeat'] and [r['data_index'] for r in repeat['cases']]==[0,1],'Repeat cases differ')
    for r in repeat['cases']:
        need(r['repeated_exact'] and r['teacher_frozen'] and r['shape']==[1,1792,768] and r['dtype']=='torch.float32' and len(r['target_sha256'])==64,'Teacher repetition/mapping missing')
        binding=r['hr_binding'];need(binding['index']==r['data_index'] and binding['record_id']==r['record_id'] and binding['split']=='train','Wrong RGB target correspondence')
        need(len(binding['source_indices'])==33 and all(b-a==1 for a,b in zip(binding['source_indices'],binding['source_indices'][1:])),'Teacher HR window not contiguous')
    need(s['initial_student_hashes']['baseline']==s['initial_student_hashes']['videomaev2'],'Initial Base mismatch')
    need(len(bench)==16,'Incomplete benchmark rows');resources={}
    for ordinal,arm in enumerate(('baseline','videomaev2')):
        values=bench[ordinal*8:ordinal*8+8];resource=s['resources'][arm]
        need(values==resource['all8_rows'] and [r['step'] for r in values]==list(range(1,9)) and [r['warmup'] for r in values]==[True,True]+[False]*6,'Benchmark ordering differs')
        for i,r in enumerate(values):
            need(r['arm']==arm and r['weight']==(0. if arm=='baseline' else .5) and r['seconds']>0 and r['gpu_event_seconds']>0,'Invalid full Step timing')
            need(r['teacher_forward_count']==(0 if arm=='baseline' else i+1) and r['EMA_updates']==0 and r['teacher_parameter_updates']==0,'Teacher update/count differs')
            need(abs(r['total']-(r['fm']+r['weight']*r['align']))<2e-7,'Weighted objective bookkeeping differs')
            if arm=='videomaev2':alignment(r['alignment'],r['align'])
        times=[r['seconds'] for r in values[2:]]
        need(close(statistics.mean(times),resource['timed6_mean_seconds']) and close(statistics.stdev(times),resource['timed6_SD_seconds']),'Timing summary omitted or changed observations')
        need(resource['teacher_forwards']==(0 if arm=='baseline' else 8) and resource['outliers_removed']==0 and resource['VAE_resident'],'Benchmark teacher/VAE residency changed')
        memory=resource['memory'];need(0<memory['peak_allocated_bytes']<=memory['peak_reserved_bytes']<=min(p['memory_cap_bytes'],memory['physical_device_bytes']),'Memory ceiling failed')
        resources[arm]={'timed6_mean_seconds':statistics.mean(times),'timed6_SD_seconds':statistics.stdev(times),'all8_seconds':[r['seconds'] for r in values],'memory':memory}
    for b,r in zip(bench[:8],bench[8:]):need(all(b[k]==r[k] for k in ('sigma','timestep_index','native_timestep')),'Benchmark FM streams differ')
    return {'counts':{'diagnostic_updates':26,'scientific_updates':0,'teacher_forwards':31,'probe_cases':15,'VJP_calls':30,'benchmark_rows':16},
        'viability':{'positive_cases':len(positive),'noise_levels':levels,'fixed_weight':.5,'weight_calibrated':False},'resources':resources,
        'online_R_over_B_timed6_ratio':resources['videomaev2']['timed6_mean_seconds']/resources['baseline']['timed6_mean_seconds'],
        'zero_weight_receipt_exact':True,'resume_receipt_exact':True,'repeat_target_hashes':[r['target_sha256'] for r in repeat['cases']]}


def review(run,output,cpu_receipt):
    run=Path(run).resolve();output=Path(output).resolve();need(not output.exists(),'Refuse to overwrite existing review')
    state=read(run/'status.json');need(state['status']=='completed' and not (run/'failure.json').exists(),'Refuse incomplete/failed preflight before diagnostic reads')
    need(state['protocol_sha256']==sha(run/'protocol.json') and state['summary_sha256']==sha(run/'summary.json'),'Completed receipt hashes differ')
    p=read(run/'protocol.json');s=read(run/'summary.json');need(s['protocol_sha256']==state['protocol_sha256'],'Summary protocol mismatch')
    rows=[json.loads(x) for x in (run/'gradient_probe_rows.jsonl').read_text().splitlines()];bench=[json.loads(x) for x in (run/'benchmark.jsonl').read_text().splitlines()]
    zero,resume,repeat=[read(run/n) for n in ('zero_weight.json','resume_report.json','teacher_repeat.json')]
    need(read(run/'target_viability.json')==s['target_viability'],'Target sidecar differs')
    evidence=audit_documents(p,s,rows,bench,zero,resume,repeat)
    need(len(p['sources'])==11,'Expected 11 bound package sources')
    for name,b in p['sources'].items():need(binding(b['path'])==b,'Bound source changed: '+name)
    cpu=read(cpu_receipt);need(cpu['status']=='passed' and cpu['failures']==cpu['errors']==0 and not cpu['CUDA_initialized'],'CPU regression receipt failed')
    for name,digest in cpu['sources'].items():
        if name.startswith('vsr/'):need(p['sources'][Path(name).name]['sha256']==digest,'CPU source chain mismatch: '+name)
    for tree in (p['teacher_dependencies'],p['dependencies']):
        def check(value):
            if isinstance(value,dict):
                if {'path','sha256','bytes'}<=value.keys():need(binding(value['path'])==value,'Bound numeric dependency changed')
                else:
                    for child in value.values():check(child)
            elif isinstance(value,list):
                for child in value:check(child)
        check(tree)
    need(binding(p['teacher_binding']['source']['path'])==p['teacher_binding']['source'],'Official encoder source changed')
    cp=Path(resume['checkpoint']['path']).resolve();need(cp.is_relative_to(run) and cp.is_file() and cp.stat().st_size==resume['checkpoint']['bytes'],'Resume checkpoint metadata missing')
    admission=read(run/'GPU_admission.json');need(not admission['unknown_processes_admitted'] and admission['lease_held_through_cleanup'] and admission['lock_path']==p['gpu_lock'],'GPU lease evidence failed')
    runtime=read(run/'runtime.json');need(runtime['mode']=='strict' and runtime['flags']['deterministic_algorithms'] and not runtime['flags']['deterministic_warn_only'] and not runtime['rng_state_modified'],'Strict runtime evidence failed')
    result={'schema':'independent_videomaev2_preflight_review_v1','status':'passed_independent_videomaev2_preflight_review',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':state['protocol_sha256'],'diagnostic_flags':[],
        'run_dir':str(run),'evidence':evidence,'summary':binding(run/'summary.json'),'protocol':binding(run/'protocol.json'),
        'sidecars':{n:binding(run/n) for n in ('status.json','gradient_probe_rows.jsonl','benchmark.jsonl','target_viability.json','zero_weight.json','resume_report.json','teacher_repeat.json','runtime.json','GPU_admission.json')},
        'CPU_regression_receipt':binding(cpu_receipt),'source_bindings_checked':11,'helper':binding(__file__),
        'scope':'Independent arithmetic/count/source/receipt review. Runtime tensor equality is attested by bound native GPU receipts, not repeated. No task quality inference.',
        'large_model_cache_checkpoint_payloads_opened':False,'extra_model_forwards':0,'extra_optimizer_updates':0,'GPU_used':False,
        'service_exit_and_current_lock_state':'Parent host completion check required; this helper verifies recorded lease/cleanup evidence only.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--output',required=True);a.add_argument('--cpu-receipt',required=True);v=a.parse_args()
    result=review(v.run,v.output,v.cpu_receipt);print(json.dumps({'status':result['status'],'protocol_sha256':result['protocol_sha256'],'diagnostic_flags':result['diagnostic_flags'],'evidence':result['evidence']},indent=2))
