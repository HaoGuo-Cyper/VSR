"""Run only after the original queue completes; no tensor/model/GPU loading."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import math
import statistics
import subprocess
import os
import tempfile

work = Path(__file__).resolve().parent
staging = work / 'VSR-sync'
actual = Path('/home/hugo/Projects/VSR_')
seeds = (20261102,20261103,20261104)
runtime = '41407b0e62748d879d2d9e1d8c5b58fa5202d09f'

def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def canonical(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return not isinstance(value,(int,float)) or math.isfinite(value)
def stats(values):
    return {'values':values,'mean':statistics.mean(values),'sample_SD':statistics.stdev(values),'training_seeds':list(seeds)}

def require(ok,message):
    if not ok:raise ValueError(message)

def numeric(value,minimum=None):
    return type(value) in (int,float) and math.isfinite(value) and (minimum is None or value>=minimum)

def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def runtime_sources(root):
    # Exact file scope of runtime414 audit.environment; never use staging here.
    return {str(p.relative_to(root)):sha(p) for folder in ('vsr','configs','tests')
            for p in sorted((root/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}

def validate_environment(value,expected_code,expected_native):
    require(value['git_commit']==runtime and value['git_status']=='', 'Runtime environment must record a clean414 checkout (git_status is a string)')
    require(value['code_sha256']==expected_code, 'Recorded119-file runtime source tree differs from actual runtime414')
    require(value['dependency_source_sha256']==expected_native and len(expected_native)==3,
            'Native Wan/VAE/UniPC implementation source differs across runs or from installed files')

def validate_initialization(initial):
    state=initial['trainable_initialization_hashes'];all_state=state['all_trainable'];groups=state['groups']
    require(state['schema']=='vsr_trainable_state_sha256_v1' and state['frozen_tensors_copied'] is False
            and state['includes_names_shapes_dtypes_raw_bytes'] is True, 'Initialization fingerprint semantics differ')
    require(valid_sha(all_state['sha256']) and all_state['tensor_count']==488 and all_state['numel']==6500992
            and len(initial['trainable_parameters'])==488 and initial['trainable_count']==6500992
            and sum(initial['trainable_parameters'].values())==6500992, 'Base LoRA/LR trainable counts differ')
    require(set(groups)=={'lrv','lora','projector','hri','other_trainable'}, 'Unexpected initialization parameter groups')
    for name,count,numel in (('lrv',8,602752),('lora',480,5898240),('projector',0,0),('hri',0,0),('other_trainable',0,0)):
        group=groups[name]
        require(valid_sha(group['sha256']) and group['tensor_count']==count and group['numel']==numel
                and len(group['names'])==count and len(set(group['names']))==count
                and group['dtypes']==(['torch.float32'] if count else []), 'Initialization group shape/dtype/count differs: '+name)
    names=[name for group in groups.values() for name in group['names']]
    require(set(names)==set(initial['trainable_parameters']) and len(names)==488
            and state['frozen_tensor_count']==825 and state['frozen_numel']==1418996800,
            'Initialization frozen/trainable identity differs')
    require(all(name.startswith(('lrv_stem.','lrv_residuals.')) for name in groups['lrv']['names'])
            and all('.lora_' in name for name in groups['lora']['names']), 'Unexpected non-LoRA/LR restoration trainable names')
    return all_state

def validate_training_rows(rows,selection,config,summary=None):
    require(len(rows)==8192 and [r['step'] for r in rows]==list(range(1,8193)) and finite(rows), 'Incomplete/nonfinite formal8192 log')
    required=('fm','loss','alignment','relation','grad_norm_before_clip','update_seconds','optimizer_updates_seconds','sigma','timestep')
    require(all(all(numeric(row.get(key),0) for key in required) for row in rows),
            'Every update needs explicit finite nonnegative loss and scalar pre-clip gradient norm (not None/bool/missing)')
    require(all(row['update_seconds']>0 and row['sigma']<=1 and row['timestep']<=1000
                and type(row['timestep_index']) is int and 0<=row['timestep_index']<1000
                and row['alignment']==row['relation']==0 and row['relation_diagnostics'] is None
                and row['loss']==row['fm'] for row in rows), 'Base FM-only/native-timestep log contract differs')
    ids=selection['training_ids']
    require([row['clip_id'] for row in rows]==[ids[i%1641] for i in range(8192)],
            'Actual optimizer clip order differs from the same fixed train1641 sequence; validation must never enter training')
    require(config['training']['gradient_log_every']==512, 'Expected the predeclared512-step detailed gradient cadence')
    detailed=[row for row in rows if row.get('gradient_groups') is not None]
    require([row['step'] for row in detailed]==[1,2,3,*range(512,8193,512)], 'Expected19 detailed group checks, separate from8192 scalar norms')
    group_names={'lrv_stem','lrv_residuals','lora','projector','hri','frozen'}
    for row in detailed:
        groups=row['gradient_groups']
        require(set(groups)==group_names and all(group['finite'] is True and numeric(group['l2'],0)
                and type(group['elements_with_grad']) is int and group['elements_with_grad']>=0 for group in groups.values()),
                'Detailed gradient group checks are incomplete/nonfinite')
        require(all(groups[name]['elements_with_grad']==0 and groups[name]['l2']==0 for name in ('projector','hri','frozen'))
                and sum(groups[name]['elements_with_grad'] for name in ('lrv_stem','lrv_residuals','lora'))==6500992,
                'Detailed Base gradients must cover all LoRA/LR parameters and no frozen/HRI/projector values')
    elapsed=0.
    for row in rows:
        elapsed+=row['update_seconds']
        require(math.isclose(row['optimizer_updates_seconds'],elapsed,rel_tol=1e-10,abs_tol=1e-7), 'Cumulative optimizer timing differs from per-update observations')
    if summary is not None:
        require(math.isclose(summary['optimizer_updates_seconds'],elapsed,rel_tol=1e-10,abs_tol=1e-7), 'Summary optimizer timing differs from the complete log')
    return {'scalar_clip_norm_rows':8192,'detailed_group_and_frozen_checks':len(detailed),
            'actual_training_clip_order_verified':True,'optimizer_seconds_recomputed':elapsed}

def validate_reuse_receipt(receipt,producer,manifest_sha,revision,encoders,seal_sha):
    expected={'status':'verified','source_run':str(producer),'clip_count':1673,'training_clip_count':1641,
        'validation_clip_count':32,'manifest_sha256':manifest_sha,'official_revision':revision,
        'encoder_source_hashes':encoders,'legacy_state_read':False,'temporal_stride':1,
        'processing_data_version':'hq-vsr-paired-v2','tensor_loading':'lazy LRU<=2; weights_only=True CPU',
        'manifest_records_checked':True,'source_asset_seal_sha256':seal_sha,'source_encoder_hashes':encoders,
        'task_checkpoint_loaded':False,'T5_loaded':False,'DINO_loaded':False}
    require(receipt==expected,'Consumer cache receipt must match the entire producer/split/encoder/no-task/no-T5/no-DINO contract')

def publish_new_json(output,value):
    # A complete new report is published atomically; racing invocations cannot
    # overwrite an existing immutable report or leave a half-written final file.
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=output.parent,prefix='.'+output.name+'.',delete=False) as stream:
            temporary=Path(stream.name);stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
        os.link(temporary,output)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)

def validate_monitoring(summary,config,selection,records):
    evaluation=summary['evaluation'];metrics=evaluation['metrics'];videos=metrics['per_video'];aggregate=metrics['aggregate']
    require(evaluation['step']==8192 and metrics['status']=='completed' and metrics['quality_hypotheses_verified'] is False
            and metrics['scope']==config['evaluation']['scope'] and aggregate['video_count']==32,
            'Monitoring must be the completed final float-RGB val32 protocol, not a quality/G2 verdict')
    require([row['record_id'] for row in videos]==selection['validation_ids']
            and all(row['clip_index']==1641+i and row['source_id']==records[row['record_id']]['source_id']
                    and row['split']=='val' and row['frame_count']==33 for i,row in enumerate(videos)),
            'Final monitoring cohort/source/frame identity differs')
    sampling=metrics['sampling'];modes=config['evaluation']['modes'];samples=config['evaluation']['seeds']
    require(samples==[42,43] and modes==['correct','disabled','shuffled']
            and sampling['seeds']==samples and sampling['steps']==20 and sampling['cfg']==1.
            and sampling['matched_noise_across_modes'] is True and sampling['hri'] is False
            and metrics['metric_protocol']['measurement_source']=='lossless float32 RGB tensors; MP4 only for playback',
            'Final monitoring sampling/measurement protocol differs')
    keys=('psnr_rgb_db','ssim_rgb','mse_rgb_01')
    def metric_domain(key,value):
        if not numeric(value):return False
        if key=='ssim_rgb':return -1<=value<=1
        if key=='mse_rgb_01':return 0<=value<=1
        return value>=0
    for row in videos:
        require(set(row['modes'])==set(modes), 'Final monitoring mode coverage differs')
        for mode in modes:
            value=row['modes'][mode]
            require([sample['seed'] for sample in value['seeds']]==samples
                    and all(sample['noise_seed']==sample['seed'] and sample['frame_count']==33 for sample in value['seeds']),
                    'Final monitoring sampler-seed coverage differs')
            for key in keys:
                values=[sample['metrics'][key] for sample in value['seeds']]
                require(all(metric_domain(key,number) for number in values) and metric_domain(key,value['mean'][key])
                        and math.isclose(value['mean'][key],sum(values)/len(values),rel_tol=1e-10,abs_tol=1e-10),
                        'Per-video monitoring seed aggregation differs')
    for mode in modes:
        for key in keys:
            values=[row['modes'][mode]['mean'][key] for row in videos]
            require(metric_domain(key,aggregate['modes'][mode][key])
                    and math.isclose(aggregate['modes'][mode][key],sum(values)/len(values),rel_tol=1e-10,abs_tol=1e-10),
                    'Monitoring equal-video aggregation differs')
    return aggregate

def main():
    if not __debug__:raise RuntimeError('Do not run this assertion-based audit with Python optimization enabled')
    queue = read(actual/'runs/active_stage2_queue.json')
    require(queue['status']=='completed', 'Wait for full queue completion, not just final training rows')
    assert queue['automatic_retry'] is False
    assert [row['name'] for row in queue['completed']]==['resource_probe',*[f'seed{seed}' for seed in seeds]]
    runs = {int(row['name'][4:]):Path(row['run']) for row in queue['completed'] if row['name'].startswith('seed')}
    assert set(runs) == set(seeds)
    assert subprocess.check_output(['git','-C',str(actual),'rev-parse','HEAD']).decode().strip() == runtime
    assert not subprocess.check_output(['git','-C',str(actual),'status','--porcelain']).strip()
    producer = runs[seeds[0]]
    code=runtime_sources(actual)
    assert len(code)==119 and {folder:sum(name.startswith(folder+'/') for name in code) for folder in ('vsr','configs','tests')}=={'vsr':49,'configs':24,'tests':46}
    producer_environment=read(producer/'environment.json')
    native={path:sha(Path(path)) for path in producer_environment['dependency_source_sha256']}
    probe=read(Path(queue['completed'][0]['run'])/'status.json')
    assert probe['status']=='completed' and probe['profile'] is True and probe['seed']==seeds[0]
    assert probe['summary']['status']=='passed' and probe['summary']['updates']==8 and probe['summary']['formal_training_updates']==0
    assert probe['summary']['trained_parameters_reused'] is False and probe['summary']['actual_EMA_A2_profile'] is False
    seal_path = producer/'cache_asset_seal.json'
    seal = read(seal_path)
    assert seal['source_run'] == str(producer) and seal['contains_task_initialization'] is False
    required = {'config.json','selection.json','correspondence.json','fresh_training_cache.pt','fresh_text_embeddings.pt'}
    required.update(f'clip_{i}/{name}' for i in range(1673) for name in ('metadata.json','paired_rgb.pt','fresh_latents.pt'))
    assert set(seal['artifact_sha256']) == required and len(required) == 5024
    # Rehash every producer artifact once after training releases its resource.
    total_bytes = 0
    for relative,expected in seal['artifact_sha256'].items():
        path = producer/relative
        assert path.resolve().is_relative_to(producer.resolve()) and path.is_file()
        assert sha(path) == expected, 'Sealed cache artifact changed: '+relative
        total_bytes += path.stat().st_size
    producer_selection = read(producer/'selection.json')
    geometry = read(producer/'correspondence.json')
    assert len(producer_selection['training_ids']) == 1641 and len(producer_selection['validation_ids']) == 32
    assert not set(producer_selection['training_ids']) & set(producer_selection['validation_ids'])
    assert geometry['latent_grid'] == [9,16,28] and geometry['token_grid'] == [9,8,14]
    producer_config=read(producer/'config.json')
    manifest_path=actual/producer_config['data']['manifest'];manifest=read(manifest_path)
    payload={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    assert manifest['manifest_sha256']==hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    records={record['id']:record for record in manifest['records']}
    assert len(records)==len(manifest['records'])
    source_splits={};byte_splits={}
    for record in records.values():
        assert record['split'] in ('train','val','test') and valid_sha(record['sha256'])
        assert source_splits.setdefault(record['source_id'],record['split'])==record['split'], 'Raw source leakage across splits'
        assert byte_splits.setdefault(record['sha256'],record['split'])==record['split'], 'Byte-identical video leakage across splits'
    expected_ids={split:[record['id'] for record in sorted((r for r in manifest['records'] if r['split']==split),
                  key=lambda r:hashlib.sha256(f"{split}:20261102:{r['id']}".encode()).hexdigest())] for split in ('train','val')}
    assert producer_selection=={'manifest_sha256':manifest['manifest_sha256'],'training_ids':expected_ids['train'],
        'validation_ids':expected_ids['val'][:32],'test_accessed':False,'caption_policy':producer_config['caption_policy']}
    assert len(expected_ids['train'])==1641 and producer_config['seed']==20261102
    assert not (producer/'cache_reuse.json').exists() and 'cache_from' not in producer_config['data']
    encoders={name:code[name] for name in ('vsr/data.py','vsr/preflight.py','vsr/stage2_preparation.py')}
    preparation=read(producer/'preparation_complete.json')
    assert preparation['status']=='completed' and preparation['clip_count']==1673 and preparation['training_clips']==1641 and preparation['validation_clips']==32
    assert preparation['encoder_source_hashes']==encoders and preparation['T5_loaded'] is preparation['DINO_loaded'] is preparation['task_model_or_optimizer_loaded'] is False
    assert seal['artifact_sha256']['fresh_text_embeddings.pt']==producer_config['data']['text_cache']['sha256']
    reports = []
    initial_hashes = []
    common_flags=None;common_official=None;common_initial_structure=None
    for seed in seeds:
        run = runs[seed]
        assert run.resolve().is_relative_to((actual/'runs').resolve())
        raw_path = actual/f'configs/stage2_20261105/base_seed{seed}.json'
        expected = copy.deepcopy(read(raw_path))
        assert '_sha256' not in expected and '_source_config_sha256' not in expected and 'cache_from' not in expected['data']
        expected['_source_config_sha256'] = sha(raw_path)
        if seed != seeds[0]:expected['data']['cache_from'] = str(producer.resolve())
        expected['_sha256'] = hashlib.sha256(canonical(expected)).hexdigest()
        config = read(run/'config.json')
        assert config == expected
        comparable=copy.deepcopy(config);producer_comparable=copy.deepcopy(producer_config)
        for value in (comparable,producer_comparable):
            for key in ('_sha256','_source_config_sha256','experiment_name'):value.pop(key,None)
            value['training'].pop('seed',None);value['data'].pop('cache_from',None)
        assert comparable==producer_comparable, 'Only training-seed/run-name/cache-consumption fields may differ'
        status,summary,initial,environment = (read(run/name) for name in
            ('status.json','Base/summary.json','Base/initialization.json','environment.json'))
        assert status['status'] == summary['status'] == 'completed' and status['summary'] == summary
        assert status['run'] == str(run) and status['seed'] == seed and status['profile'] is False
        assert summary['updates'] == 8192 and summary['variant'] == 'Base'
        assert summary['training_clip_count'] == 1641 and summary['validation_clip_count'] == 32
        assert status['test_accessed'] is False and summary['test_accessed'] is False
        assert read(run/'selection.json') == producer_selection and read(run/'correspondence.json') == geometry
        validate_environment(environment,code,native)
        reproducibility=read(run/'reproducibility.json');flags=reproducibility['flags']
        assert reproducibility['mode']=='strict' and reproducibility['rng_state_modified'] is False
        assert all(flags[key] is True for key in ('deterministic_algorithms','cudnn_deterministic','sdp_math'))
        assert all(flags[key] is False for key in ('deterministic_warn_only','cudnn_benchmark','cudnn_allow_tf32','matmul_allow_tf32',
            'matmul_allow_fp16_reduced_precision_reduction','matmul_allow_bf16_reduced_precision_reduction',
            'sdp_flash','sdp_memory_efficient','sdp_cudnn','sdp_math_allow_fp16_bf16_reduction'))
        assert flags['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and flags['PYTHONHASHSEED']=='20261102' and flags['cpu_threads']==2
        if common_flags is None:common_flags=flags
        assert flags==common_flags
        official=read(run/'official_artifacts.json')
        assert official['status']=='passed' and official['repo_id']==config['model']['repo_id'] and official['revision']==config['model']['revision']
        assert official['legacy_state_loaded'] is False and official['official_weights_only'] is True
        if common_official is None:common_official=official
        assert official==common_official
        assert initial['fresh_official_initialization'] is True and initial['optimizer_fresh'] is True
        assert initial['legacy_state_loaded'] is False and initial['reference_enabled'] is False
        assert initial['official_revision'] == config['model']['revision']
        assert initial['seed'] == initial['training_seed'] == seed and initial['data_seed'] == 20261102
        assert set(initial['loading_info'])=={'missing_keys','unexpected_keys','mismatched_keys','error_msgs'} and all(values==[] for values in initial['loading_info'].values())
        initialization_hash = validate_initialization(initial)
        structure={'parameters':initial['trainable_parameters'],
            'groups':{name:{key:value for key,value in group.items() if key!='sha256'}
                      for name,group in initial['trainable_initialization_hashes']['groups'].items()}}
        if common_initial_structure is None:common_initial_structure=structure
        assert structure==common_initial_structure, 'Only initialization bytes may differ between training seeds, not parameter structure'
        initial_hashes.append(initialization_hash['sha256'])
        rows = [json.loads(line) for line in (run/'Base/training.jsonl').read_text().splitlines()]
        log_audit=validate_training_rows(rows,producer_selection,config,summary)
        final_monitoring=validate_monitoring(summary,config,producer_selection,records)
        assert numeric(summary['phase_wall_seconds'],summary['optimizer_updates_seconds'])
        assert numeric(status['entire_wall_seconds'],summary['phase_wall_seconds']) and numeric(summary['training_gpu_hours'],0)
        assert math.isclose(summary['training_gpu_hours'],summary['optimizer_updates_seconds']/3600,rel_tol=1e-10,abs_tol=1e-10)
        assert type(summary['peak_allocated_bytes']) is int and 0<summary['peak_allocated_bytes']<=min(16*1024**3,environment['free_total_bytes'][1])
        receipt = None
        if seed != seeds[0]:
            receipt_path = run/'cache_reuse.json'
            receipt = read(receipt_path)
            validate_reuse_receipt(receipt,producer,manifest['manifest_sha256'],config['model']['revision'],encoders,sha(seal_path))
        checkpoint = run/'Base/checkpoint_008192.pt'
        reports.append({'seed':seed,'run':str(run),'config_sha256':sha(run/'config.json'),
            'status_sha256':sha(run/'status.json'),'summary_sha256':sha(run/'Base/summary.json'),
            'training_log_sha256':sha(run/'Base/training.jsonl'),
            'initialization_sha256':sha(run/'Base/initialization.json'),
            'environment_sha256':sha(run/'environment.json'),'reproducibility_sha256':sha(run/'reproducibility.json'),
            'official_artifacts_sha256':sha(run/'official_artifacts.json'),
            'initial_trainable_state':initialization_hash,'fresh_official_and_optimizer':True,
            'checkpoint':{'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':sha(checkpoint),'exported':False},
            'updates':8192,'all8192_losses_and_gradient_norms_finite':True,
            'detailed_finite_and_frozen_absent_gradient_rows':19,
            'gradient_and_data_log_audit':log_audit,
            'frozen_gradient_evidence_scope':'19 explicitly logged group checks;8192 scalar clip norms are separately checked',
            'final_loss':rows[-1]['loss'],'last1024_loss_mean':statistics.mean(r['loss'] for r in rows[-1024:]),
            'optimizer_seconds':summary['optimizer_updates_seconds'],
            'phase_wall_seconds':summary['phase_wall_seconds'],'entire_wall_seconds':status['entire_wall_seconds'],
            'peak_allocated_bytes':summary['peak_allocated_bytes'],
            'final_float_RGB_monitoring':final_monitoring,
            'cache_reuse_receipt':receipt})
    assert len(set(initial_hashes)) == 3, 'Training seeds must have independent initial trainable states'
    assert initial_hashes[0]==probe['summary']['trainable_sha256_before'] and initial_hashes[0]!=probe['summary']['trainable_sha256_after'], 'Base seed02 must start from fresh initialization, not the trained8-update probe'
    assert all(read(runs[seed]/'environment.json')['code_sha256'] == read(producer/'environment.json')['code_sha256'] for seed in seeds)
    report = {'observed_at_UTC':datetime.now(timezone.utc).isoformat(),'status':'three_Bases_completed_and_audited',
        'audit_helper_sha256':sha(Path(__file__)),
        'runtime_git_commit':runtime,'runtime_clean':True,'formal_updates':24576,'formal_RE':3,
        'resource_diagnostic_updates_separate':8,'runs':reports,
        'same_train1641_val32_geometry_encoder_sources':True,'test_accessed':False,
        'cache_seal':{'path':str(seal_path),'sha256':sha(seal_path),'entries_rehashed':5024,'artifact_bytes_rehashed':total_bytes},
        'runtime_source_verification':{'file_count':len(code),'code_sha256':code,'native_implementation_sha256':native,
            'manifest_file_sha256':sha(manifest_path),'strict_flags':common_flags},
        'training_seed_statistics':{key:stats([r[key] for r in reports]) for key in
            ('final_loss','last1024_loss_mean','optimizer_seconds','phase_wall_seconds','entire_wall_seconds','peak_allocated_bytes')},
        'float_RGB_monitoring_training_seed_statistics':{key:stats([r['final_float_RGB_monitoring']['modes']['correct'][key] for r in reports])
            for key in ('psnr_rgb_db','ssim_rgb','mse_rgb_01')},
        'monitoring_scope':'unquantized float RGB monitoring; single fixed val32 cohort, not T2.4 quantized comparison or G2',
        'G2':'pending all1152 frozen restores','G3':'not executed','binary_artifacts_exported':False}
    report['checkpoint_validation_scope']='Final-file byte SHA256/size plus completed config/log/init/summary; no tensor deserialization. The task loader must validate payload before T2 GPU execution.'
    output = staging/'docs/stage2_20261106/three_base_completion_audit.json'
    publish_new_json(output,report)
    print(json.dumps({'status':report['status'],'report':str(output),'checkpoint_sha256':[r['checkpoint']['sha256'] for r in reports]}))

if __name__ == '__main__':main()
