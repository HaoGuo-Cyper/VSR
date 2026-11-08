"""Text-only second review; does not import torch or open checkpoint payloads."""
from pathlib import Path
import json,hashlib,math,statistics
from datetime import datetime,timezone

W=Path('/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work')
R=W/'videomaev2_runs/paired_training_20261108T150511Z'
A=W.parent/'outputs/VideoREPA_VideoMAEv2_20261108/training_diagnostics/training_audit.json'
O=W/'scratch/videomaev2_training_completion_independent_review_20261108.json'
def need(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def finite(v):
 if type(v) is float:need(math.isfinite(v),'Nonfinite numeric evidence')
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
need(not O.exists(),'Preserve existing independent review')
state=read(R/'status.json');need(state['status']=='completed' and state['scientific_updates']==4096,'Incomplete training')
need(not list(R.glob('failure*')) and not list(R.glob('**/rollback*')),'Unexpected interrupted attempt')
need(sha(A)=='25d8437000b4712ebf9a016bbc9dfc5a59c0b3f6df7b127237ebadbd3bb9ca3f','Training audit identity changed')
need(sha(R/'summary.json')==state['summary_sha256']=='a7b5a572e923ebd2c04aff4c6a4db952b4f97f938207db9b2f105d1645bad8bf','Training summary identity changed')
a=read(A);p=read(R/'protocol.json');s=read(R/'summary.json')
need(sha(R/'protocol.json')==state['protocol_sha256']==s['protocol_sha256']=='5292bbd39d21303ac4f57541d80c2e897b02782291b4e73817eb01bd97676bdf','Protocol/status chain differs')
need(a['status']=='passed_videomaev2_training_audit' and a['scientific_updates']==4096 and a['updates_each']==2048 and a['arms']==2 and a['checkpoint_count']==8 and a['training_seed_repetitions']==1,'First audit incomplete')
for k,where in [('protocol',R/'protocol.json'),('summary',R/'summary.json'),('status_file',R/'status.json'),('producer',W/'videomaev2_training_audit.py')]:need(a[k]==binding(where),'Audit binding changed: '+k)
need(a['sources']==p['sources'] and len(p['sources'])==11,'11-source chain differs')
for name,b in p['sources'].items():need(b==binding(b['path']),'Frozen source changed: '+name)
need(binding(p['study_file']['path'])==p['study_file']==a['study_file'],'Study changed')
need(s['status']=='completed' and s['scientific_updates']==p['scientific_update_budget']==4096 and s['diagnostic_updates_in_this_training']==0 and s['selection_or_evaluation_done'] is False,'Scientific budget/task boundary differs')
need(p['study']['arms']==[{'label':'baseline','weight':0.},{'label':'videomaev2','weight':.5}] and p['study']['training_seed']==20261102 and p['study']['updates']==2048,'Arms/seed/budget changed')
need(set(s['arms'])=={'baseline','videomaev2'},'Wrong roles')
checks=a['checkpoint_checks'];need(checks['flags']==[] and all(checks[k] for k in ('explicit_four_RNG_streams_bytewise_identical_at_each_checkpoint','global_RNG_complete_and_CPU_validated','CUDA_RNG_byte_metadata_checked_without_GPU_replay','all8_student_and_auxiliary_states_checked')) and not checks['GPU_initialized'],'Checkpoint audit flags/incomplete')
need(a['CPU_only'] and not a['Wan_or_teacher_model_reconstructed'] and not a['quality_scores_read'] and not a['external_teacher_tensor_payload_loaded'],'Audit scope differs')
cal=p['calibration']
for b in cal.values():need(binding(b['path'])==b,'Preflight receipt changed')
calp=read(cal['protocol']['path']);cals=read(cal['summary']['path'])
need(calp['sources']==p['sources']==cals['sources'] and cals['status']=='passed_videomaev2_preflight' and cals['diagnostic_optimizer_updates']==26 and cals['teacher_forwards']==31,'Preflight source/count chain differs')
need(calp['teacher_binding']==p['teacher_binding'],'Teacher binding changed between preflight/training')
fields=['record_id','data_index','sigma','timestep_index','native_timestep'];logs={};costs=[];checkpoints=[];sourceids=[x['record_id'] for x in p['rgb_assets']['records']]
need(len(sourceids)==1641 and len(set(sourceids))==1641,'Wrong training source count')
for arm,weight in [('baseline',0.),('videomaev2',.5)]:
 d=s['arms'][arm];base=R/'arm_runs'/arm
 need(d==read(base/'completion.json') and d['initialization']==read(base/'initialization.json'),'Completion/initialization sidecar differs')
 need(d['weight']==weight and d['updates']==2048 and d['teacher_binding']==(p['teacher_binding'] if weight else None) and d['teacher_forward_count']==(2048 if weight else 0) and d['teacher_parameter_updates']==d['EMA_updates']==0,'Teacher contract differs')
 need(d['head_parameters']==(8917760 if weight else 0) and not d['teacher_is_frozen_cached_HR'],'Wrong head/cache')
 init=d['initialization'];need(init['checkpoint_sha256']==p['base']['checkpoint']['sha256'] and init['step']==8192 and init['variant']=='Base','Wrong shared initial student')
 need(all(init[k] is False for k in ('optimizer_state_restored','scheduler_state_restored','rng_state_restored','EMA_weights_selected','training_only_state_restored')),'Unmatched optimizer/RNG/EMA initialization')
 logfile=Path(d['full_training_log']['path']);need(logfile==base/'training.jsonl' and binding(logfile)==d['full_training_log'],'Canonical log binding differs')
 raw=logfile.read_bytes();need(raw.endswith(b'\n'),'Partial log tail');rows=[json.loads(v) for v in raw.splitlines()];finite(rows)
 need(len(rows)==2048 and [r['step'] for r in rows]==list(range(1,2049)),'Incomplete/noncontiguous log')
 elapsed=hrtime=0.;positivehead=0;positivestudent=0
 for i,row in enumerate(rows):
  need(row['arm']==row['family']==arm and row['setting']=='common' and row['weight']==weight,'Wrong row role/weight')
  need(row['data_index']==i%1641 and row['record_id']==sourceids[i%1641],'Wrong data traversal')
  need(row['student_layer']==17 and not row['forced_inputs'] and not row['condition_corruption'] and row['teacher_parameter_updates']==row['EMA_updates']==0,'Changed training feature/noise contract')
  need(row['teacher_forward'] is bool(weight) and row['teacher_forward_count']==(i+1 if weight else 0),'Teacher step accounting differs')
  need(0<=row['sigma']<=1 and type(row['timestep_index']) is int and 0<=row['timestep_index']<1000 and row['noise_bin']==('low' if row['sigma']<.33 else 'mid' if row['sigma']<.67 else 'high'),'Native sigma/index/bin invalid')
  need(all(row[k]>=0 for k in ('seconds','gpu_event_seconds','hr_read_seconds','fm','align','grad_norm_before_clip','student_total_grad_norm_before_clip','head_total_grad_norm_before_clip')),'Negative training observation')
  need(abs(row['total']-row['fm']-weight*row['align'])<2e-7,'Weighted-loss accounting mismatch')
  elapsed+=row['seconds'];hrtime+=row['hr_read_seconds'];need(row['cumulative_training_seconds']==elapsed and row['cumulative_hr_read_seconds']==hrtime,'Cumulative timing inconsistent')
  positivestudent+=row['student_total_grad_norm_before_clip']>0;positivehead+=row['head_total_grad_norm_before_clip']>0
  if weight:
   z=row['alignment'];need(z['all_pairs']==1792**2 and z['target_grid']==[16,8,14] and z['margin']==.1 and z['normalize_eps']==1e-12 and z['projected_all_latent_positions']==9 and z['dropped_latent_positions']==[0],'Changed fullGram/mapping')
   need(z['student_gram_dtype']=='torch.bfloat16' and z['teacher_gram_dtype']==z['difference_dtype']==z['loss_dtype']=='torch.float32' and z['diagonal_included'] and z['ordered_pairs'],'Changed dtype/pair reduction')
  else:need(row['align']==row['head_total_grad_norm_before_clip']==row['hr_read_seconds']==0,'Baseline used auxiliary work')
 need(d['cumulative_training_seconds']==elapsed and d['cumulative_hr_read_seconds']==hrtime,'Final time mismatch')
 memory=d['memory'];need(memory['peak_allocated_bytes']<=memory['peak_reserved_bytes']<=min(p['memory_cap_bytes'],memory['physical_device_bytes']),'Memory limit exceeded')
 need([c['step'] for c in d['checkpoint_files']]==[512,1024,1536,2048],'Missing checkpoint metadata')
 for cp in d['checkpoint_files']:
  path=Path(cp['path']);need(path==base/'Base'/f"checkpoint_{cp['step']:06d}.pt" and path.is_file() and path.stat().st_size==cp['bytes'] and len(cp['sha256'])==64,'Checkpoint metadata absent/wrong path')
  need(cp['step']==cp['update'] and cp['variant']=='Base' and cp['cumulative_training_seconds']==rows[cp['step']-1]['cumulative_training_seconds'],'Checkpoint step timing differs')
  checkpoints.append(cp)
 logs[arm]=rows
 costs.append({'arm':arm,'steps':2048,'synchronized_step_seconds':elapsed,'CPU_HR_read_seconds':hrtime,'step_plus_HR_read_seconds':elapsed+hrtime,'positive_student_gradient_steps':positivestudent,'positive_head_gradient_steps':positivehead,'peak_reserved_bytes':memory['peak_reserved_bytes']})
need(s['arms']['baseline']['initial_student_hashes']==s['arms']['videomaev2']['initial_student_hashes'],'Initial student hashes differ')
need(s['arms']['baseline']['initial_student_hashes']['all_trainable']==checks['Base_digest'],'First-audit Base digest differs')
seq=[[r[k] for k in fields] for r in logs['baseline']]
need(seq==[[r[k] for k in fields] for r in logs['videomaev2']],'Paired sequence differs')
need(a['matched_sequence']=={'fields':fields,'steps':2048,'digest':digest(seq),'both_arms_exactly_equal':True},'Sequence digest mismatch')
expectedorder=[(arm,step) for arm in ('baseline','videomaev2') for step in (512,1024,1536,2048)]
for kind in ('rng_rows','update_rows','teacher_rows'):need([(v['arm'],v['step']) for v in checks[kind]]==expectedorder,'Incomplete checkpoint audit table: '+kind)
for b,r in zip(checks['rng_rows'][:4],checks['rng_rows'][4:]):
 need(b['all_four_exactly_equal'] and r['all_four_exactly_equal'] and all(b[k]==r[k] for k in ('noise_sha256','timestep_sha256','mask_sha256','corruption_sha256','global_rng_sha256')),'Checkpoint RNG digests differ')
for v in checks['update_rows']:
 need(v['student_tensors']==v['student_changed_tensors']==488 and v['all_optimizer_step_counts_match'],'Student/AdamW evidence incomplete')
 n=6 if v['arm']=='videomaev2' else 0;need(v['head_tensors']==v['head_changed_tensors']==v['head_nonzero_second_moment_tensors']==n,'Head update evidence incomplete')
for v in checks['teacher_rows']:
 teacher=p['teacher_binding'] if v['arm']=='videomaev2' else None
 need(v['teacher_binding_sha256']==digest(teacher) and v['teacher_forward_count']==(v['step'] if teacher else 0) and v['teacher_parameter_updates']==0 and not v['external_teacher_tensors_in_CP'],'Teacher provenance/counter audit differs')
for ours,theirs in zip(costs,a['cost_rows']):
 need(ours['arm']==theirs['arm'] and ours['synchronized_step_seconds']==theirs['committed_synchronized_step_seconds'] and ours['CPU_HR_read_seconds']==theirs['separate_CPU_HR_read_seconds'] and ours['step_plus_HR_read_seconds']==theirs['sum_step_plus_extra_HR_seconds'],'Cost audit arithmetic differs')
cost=s['compute_cost'];need(cost==a['containing_training_cost'],'Containing cost receipt changed')
need(cost['committed_training_step_seconds']==sum(c['synchronized_step_seconds'] for c in costs) and cost['committed_hr_read_seconds']==sum(c['CPU_HR_read_seconds'] for c in costs),'Total cost sums differ')
need(cost['known_discarded_completed_step_seconds']==cost['known_discarded_completed_step_returns']==0 and cost['successful_step_returns_this_attempt']==4096,'Discarded/returned accounting differs')
need(cost['wall_seconds_all_training_attempts']>=sum(c['step_plus_HR_read_seconds'] for c in costs),'Nested costs exceed containing attempt time')
need(len(list(R.glob('attempt_*.json')))==1 and len(list(R.glob('completion_attempt_*.json')))==1,'Unexpected extra attempt')
result={'schema':'videomaev2_training_completion_independent_review_v1','status':'passed_independent_videomaev2_training_review','created_UTC':datetime.now(timezone.utc).isoformat(),
 'protocol_sha256':state['protocol_sha256'],'diagnostic_flags':[],'audit':binding(A),'summary':binding(R/'summary.json'),'protocol':binding(R/'protocol.json'),'status_file':binding(R/'status.json'),
 'counts':{'scientific_updates':4096,'training_trajectories':2,'updates_each':2048,'checkpoint_metadata_count':8,'R_online_teacher_forwards':2048,'teacher_parameter_updates':0,'new_diagnostic_updates':0},
 'matched_sequence_digest':digest(seq),'checkpoint_rng_digest_pairs_verified':4,'student_updated_tensors_per_checkpoint':488,'R_head_updated_tensors_per_checkpoint':6,
 'costs_recomputed_from_all4096_rows':costs,'containing_cost':cost,
 'online_R_over_B_synchronized_step_ratio':costs[1]['synchronized_step_seconds']/costs[0]['synchronized_step_seconds'],
 'R_over_B_including_extra_HR_read_ratio':costs[1]['step_plus_HR_read_seconds']/costs[0]['step_plus_HR_read_seconds'],
 'all11_current_sources_match':True,'teacher_source_weight_position_binding_unchanged':True,'preflight_counts_and_source_chain_valid':True,
 'scope':'Independent text/log arithmetic plus existing checkpoint-audit digests; no repeated tensor comparison, model/checkpoint loading, large weight hash, GPU or validation quality read. Teacher freezing rests on runtime source checks and bound metadata/count evidence.',
 'quality_conclusion':None,'partial_validation_scores_read':False,'checkpoint_payloads_opened':0,'GPU_used':False,'extra_optimizer_updates':0,'extra_model_forwards':0,'review_script':binding(__file__)}
with O.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':result['status'],'protocol_sha256':result['protocol_sha256'],'diagnostic_flags':result['diagnostic_flags'],'costs':costs,'ratios':[result['online_R_over_B_synchronized_step_ratio'],result['R_over_B_including_extra_HR_read_ratio']],'receipt':binding(O)},indent=2))
