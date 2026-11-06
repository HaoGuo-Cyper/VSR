#!/usr/bin/env python3
"""Future T4.3: all three method students on the unchanged sealed val32 assets.

prepare is stdlib metadata/hashing only. execute requires prospectively frozen
G4 criteria and completed real G2/G3/three-arm2048 training. It produces1152
observations and descriptive paired statistics, never an automatic G4 verdict.
No Base inference, quality tuning, test/external data, training or retry occurs.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import traceback
import weakref

sys.dont_write_bytecode = True
import stage2_evaluate_bases as t2
import stage3_resource_gate as g3

need,read,save,binding,file_sha = t2.need,t2.read,t2.save,t2.binding,t2.file_sha
ARMS = ['flow_only','C1','A2']
TRAINING_SEED = 20261102
UPDATES = 2048
CONDITIONS = list(t2.CONDITIONS)
SAMPLES = [42,43]
METRICS = [f'{pool}.{metric}' for pool in ('masked','neighbors','all_frames')
           for metric in ('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1')] + [
    'temporal.mask_touch_tLP100','temporal.all28pairs_tLP100',
    'raw_harm_gap.psnr_rgb_db','raw_harm_gap.lpips_alex_v0_1']
CRITERIA_DEFINITIONS = ['primary_recovery','base_seed_spread','clean_fidelity','aggregation','decision_rule']
PLACEHOLDER = re.compile(r'\b(?:TODO|TBD|PENDING|PLACEHOLDER)\b|待定|待补|待确认|未决定|占位|<[^>]+>',re.I)


def now():return datetime.now(timezone.utc).isoformat()


def json_record(value):
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))


def _defined(value):
    if isinstance(value,str):return len(value.strip()) >= 4 and not PLACEHOLDER.search(value)
    if type(value) is bool:return True
    if type(value) in (int,float):return math.isfinite(value)
    if isinstance(value,list):return bool(value) and all(_defined(x) for x in value)
    if isinstance(value,dict):return bool(value) and all(isinstance(k,str) and _defined(v) for k,v in value.items())
    return False


def criteria_spec(path,expected_sha,evaluation,*,prepared_utc=None):
    """Bind prospective operational definitions to the existing authorized plan.

    The task owner records who froze the file and the decision/authorization
    reference. This is not a new human-approval flow or an automatic G4 rule.
    """
    need(isinstance(expected_sha,str) and re.fullmatch('[0-9a-f]{64}',expected_sha), 'Exact reviewed G4 criteria raw SHA is mandatory')
    receipt = binding(path,expected_sha);criteria = read(path)
    need(criteria.get('schema')=='stage4_G4_prospective_criteria_v1'
         and criteria.get('status')=='frozen' and _defined(criteria.get('frozen_by'))
         and criteria.get('score_access_before_freeze') is False
         and _defined(criteria.get('decision_reference') or criteria.get('authorization_reference')),
         'Prospectively frozen G4 definitions and existing decision reference required; placeholders are forbidden')
    expected_scope = {'training_seed':TRAINING_SEED,'arms':ARMS,'updates':UPDATES,
        'cohort_seal_digest':evaluation['cohort_seal_digest'],
        'Base_evaluation_protocol_sha256':evaluation['protocol_sha256']}
    need(criteria.get('scope')==expected_scope, 'G4 criteria scope differs from the fixed pilot/Base/cohort')
    definitions = criteria.get('definitions')
    need(isinstance(definitions,dict) and set(definitions)==set(CRITERIA_DEFINITIONS)
         and all(isinstance(definitions[k],(str,dict)) and _defined(definitions[k]) for k in CRITERIA_DEFINITIONS),
         'All G4 definitions must be present and nonplaceholder before scoring')
    try:
        frozen = datetime.fromisoformat(criteria['frozen_utc'].replace('Z','+00:00'))
        prepared = datetime.fromisoformat((prepared_utc or now()).replace('Z','+00:00'))
        need(frozen.tzinfo is not None and prepared.tzinfo is not None and frozen<=prepared,
             'G4 criteria must already be frozen before evaluation prepare')
    except (KeyError,TypeError,AttributeError,ValueError) as error:
        raise ValueError('Valid prospective UTC G4 freeze timestamp required') from error
    return {'file':receipt,'criteria':criteria,'automatic_G4_verdict':False,
        'authorization_scope':'Existing task decision/authorization reference and exact frozen file SHA; this worker neither adds an approval step nor defines criteria.'}


def canonical_method_config(config,arm,base):
    value = copy.deepcopy(config);expected = value.pop('_sha256',None)
    need(isinstance(expected,str) and hashlib.sha256((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()==expected,
         'Method canonical config SHA differs')
    need(config.get('restoration_architecture')=='noref_base_v1'
         and config['training']['variants']==[arm] and config['training']['updates']==UPDATES
         and config['training'].get('seed',config.get('seed'))==TRAINING_SEED and config.get('seed')==20261102,
         'Require the true method name,2048 updates and matched training/data seed')
    need(config.get('model')==base['config']['model'] and config.get('conditioning')==base['config']['conditioning']
         and config.get('reference',{}).get('enabled',False) is False
         and config['conditioning'].get('hri_enabled',False) is False,
         'Method restoration model/conditioning must equal its NoRef matched Base')
    need([config['data'][k] for k in ('num_frames','temporal_stride','height','width','scale')]==[33,1,128,224,4]
         and config['reproducibility']['mode']=='strict','Method geometry/backend differs')
    return config


def pilot_helper(root):
    return t2.import_helper(Path(root)/'scripts/stage4_pilot.py','stage4_pilot_prerequisites')


def method_checkpoints(directory,pilot_protocol,pilot_summary,root):
    directory,root = Path(directory).resolve(),Path(root).resolve()
    base = pilot_protocol['matched_Base'];result = {}
    for arm in ARMS:
        run = directory/'arm_runs'/arm
        config = canonical_method_config(read(run/'config.json'),arm,base)
        need(config==pilot_protocol['method_configs'][arm],'Method arm config differs from its frozen pilot protocol')
        summary,status = read(run/'summary.json'),read(run/'status.json')
        need(summary==pilot_summary['arms'][arm] and status['status']==summary['status']=='completed'
             and summary['updates']==status['updates']==UPDATES
             and summary['training_seed']==TRAINING_SEED and summary['arm']==arm,
             'Missing/wrong method arm completion')
        need(status['summary_sha256']==file_sha(run/'summary.json'), 'Method arm summary seal differs')
        need(isinstance(summary['final_trainable_sha256'],str)
             and re.fullmatch('[0-9a-f]{64}',summary['final_trainable_sha256']), 'Completed method student parameter SHA is required')
        receipt = summary['final_checkpoint'];path = Path(receipt['path']).resolve()
        need(path==run/arm/f'checkpoint_{UPDATES:06d}.pt' and path.is_relative_to(root/'runs')
             and receipt['variant']==arm and receipt['step']==UPDATES
             and receipt['inference_parameters']=='student' and receipt['restoration_architecture']=='noref_base_v1',
             'Final checkpoint must be the completed2048 student under staging runs, never the Base actual tree')
        result[arm] = {'arm':arm,'training_seed':TRAINING_SEED,'checkpoint':binding(path,receipt['sha256']),
            'config':config,'config_file':binding(run/'config.json'),'summary':binding(run/'summary.json'),
            'status':binding(run/'status.json'),'writer_receipt':receipt,
            'expected_final_student_trainable_sha256':summary['final_trainable_sha256']}
    return result


def collect_protocol(args):
    root = args.project_root.resolve();directory = args.pilot_run.resolve()
    need(Path.cwd().resolve()==root and Path(__file__).resolve()==root/'scripts/stage4_evaluate_pilot.py',
         'Use this frozen staging source root/CWD, required by FramewiseReader')
    need(root!=t2.OLD and not root.is_relative_to(t2.OLD) and root!=g3.ACTUAL_RUNS.parent,
         'Historical and active actual source roots are not evaluator outputs')
    helper = pilot_helper(root)
    pilot_protocol,pilot_evidence = helper.completed_pilot(directory,root)
    pilot_summary,pilot_status = read(directory/'summary.json'),read(directory/'status.json')
    need(pilot_protocol['schema']=='stage4_pilot_training_v1'
         and pilot_status['status']==pilot_summary['status']=='completed_pilot_training'
         and pilot_status['summary_sha256']==file_sha(directory/'summary.json')
         and pilot_status['protocol_sha256']==pilot_summary['protocol_sha256']==file_sha(directory/'protocol.json')
         and pilot_summary['completed_arms']==ARMS and pilot_summary['scientific_training_updates']==6144
         and pilot_summary['G4_evaluated'] is False,'All three completed2048 pilot arms required; quality scores must still be prospective')
    evaluation = pilot_protocol['g3_protocol']['evaluation_protocol']
    # Re-run the original complete three-Base/G2 validator, not a summary flag.
    base_directory = Path(pilot_protocol['g3_protocol']['completed_T2']['directory'])
    base_protocol,base_evidence = g3.completed_t2(base_directory,root)
    need(base_protocol==evaluation,'Pilot and current completed Base evaluation protocols differ')
    base_psha = file_sha(base_directory/'protocol.json')
    criteria_evaluation = {'cohort_seal_digest':evaluation['cohort_seal_digest'],'protocol_sha256':base_psha}
    criteria = criteria_spec(args.g4_criteria,args.expected_g4_criteria_sha256,criteria_evaluation)
    training_criteria = pilot_protocol['quality_evaluation']['criteria']
    if training_criteria is not None:
        need(training_criteria['sha256']==criteria['file']['sha256'], 'Evaluation criteria differ from criteria already bound at pilot training prepare')
    checkpoints = method_checkpoints(directory,pilot_protocol,pilot_summary,root)
    return json_record({'schema':'stage4_pilot_student_val32_evaluation_v1','project_root':str(root),
        'worker':binding(__file__),'tests':binding(root/'tests/test_stage4_evaluate_pilot.py'),
        'pilot_prerequisite_worker':binding(root/'scripts/stage4_pilot.py'),
        'completed_pilot':pilot_evidence,'pilot_files':{name:binding(directory/name) for name in ('protocol.json','summary.json','status.json')},
        'prerequisite_directories':[str(directory),str(base_directory),pilot_protocol['completed_G3']['directory'],
            str(Path(evaluation['dataset_manifest']['path']).parent)],
        'evaluation_protocol':evaluation,'completed_Base_evaluation':base_evidence,
        'Base_observations':binding(base_directory/'observations.json'),'Base_summary':binding(base_directory/'summary.json'),
        'checkpoints':checkpoints,'allowed_method_checkpoint_root':str(root/'runs'),
        'G4_criteria':criteria,'scope':{'arms':ARMS,'method_training_seeds':[TRAINING_SEED],'method_updates':UPDATES,
            'conditions':CONDITIONS,'sampling_seeds':SAMPLES,'records':32,'restores_per_arm':384,
            'total_restores_and_observations':1152,'prediction_PNGs':38016,
            'Base_restores':0,'training_updates':0,'test_or_external_pixels':False,'row_limits_or_selection':False},
        'numerics':'Unchanged completed T2 NPY/reader/native restore/UniPC20 CFG1/quantize_rgb/LPIPS/TecoGAN/pools; student only.',
        'statistical_scope':{'method_training_seed_count':1,'sampler_seeds_are_training_repeats':False,
            'all_five_types_equal_weight':True,'pooling':'two sampler seeds within clip, equal clips, equal types',
            'Base_seed_spread':'Descriptive sample SD across the three original Base training-seed means; no automatic threshold interpretation.',
            'G4_verdict_computed':False,'p_values_computed':False,'Stage5_significance_decision':False},
        'output':{'fresh_only':True,'resume':False,'automatic_retry':False,'automatic_training':False,
            'cleanup_failure_revokes_completion':True,'G4_decision':'not_computed_requires_review'}})


def flattened(score):
    result = {f'{pool}.{metric}':score['pools'][pool][metric]
              for pool in ('masked','neighbors','all_frames') for metric in ('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1')}
    result.update({f'temporal.{k}':score['temporal'][k] for k in ('mask_touch_tLP100','all28pairs_tLP100')})
    result.update({f'raw_harm_gap.{k}':score['raw_harm_gap'][k] for k in ('psnr_rgb_db','lpips_alex_v0_1')})
    need(all(type(v) in (int,float) and math.isfinite(v) for v in result.values()), 'Nonfinite/missing common metric pool')
    return result


def index_rows(rows,seal,checkpoints,*,method):
    cohort = {item['record']['id']:item for item in seal['cohort']};seen = {}
    roles = ARMS if method else t2.SEEDS
    for row in rows:
        role = row['arm'] if method else row['training_seed']
        key = role,row['record_id'],row['condition'],row['sampling_seed']
        need(key not in seen and role in roles and key[1] in cohort and key[2] in CONDITIONS and key[3] in SAMPLES,
             'Duplicate/out-of-cohort/selected evaluation observation')
        item = cohort[key[1]]
        need(row['source_id']==item['record']['source_id'] and row['cohort_seal_digest']==seal['seal_digest']
             and row['score']['mask']==item['mask'] and row['checkpoint_sha256']==checkpoints[str(role)],
             'Score source/mask/seal/checkpoint identity differs')
        if method:need(row['training_seed']==TRAINING_SEED and row['method_updates']==UPDATES,'Method rows must remain single matched training seed2048')
        seen[key] = flattened(row['score'])
    need(len(seen)==1152 and len(cohort)==32,'Require all1152 rows, including clean and all five types')
    return seen


def _stats(values):
    return {'mean':statistics.mean(values),'between_clip_sample_SD':statistics.stdev(values),
            'n_clips':len(values),'n_sampling_seeds_averaged_within_clip':2}


def summarize(rows,base_rows,seal,protocol):
    methods = index_rows(rows,seal,{arm:protocol['checkpoints'][arm]['checkpoint']['sha256'] for arm in ARMS},method=True)
    bases = index_rows(base_rows,seal,{seed:value['checkpoint']['sha256'] for seed,value in protocol['evaluation_protocol']['checkpoints'].items()},method=False)
    records = [item['record']['id'] for item in seal['cohort']]
    def clips(index,role,condition):
        return {record:{metric:statistics.mean(index[role,record,condition,sample][metric] for sample in SAMPLES)
                       for metric in METRICS} for record in records}
    def table(index,role):
        result = {}
        selected_by_condition = {c:clips(index,role,c) for c in CONDITIONS}
        for condition in CONDITIONS:
            selected = selected_by_condition[condition]
            result[condition] = {metric:_stats([selected[r][metric] for r in records]) for metric in METRICS}
        result['equal_five_type_macro'] = {metric:_stats([statistics.mean(selected_by_condition[c][r][metric] for c in CONDITIONS[1:])
            for r in records]) for metric in METRICS}
        return result
    per_arm = {arm:table(methods,arm) for arm in ARMS}
    per_base = {str(seed):table(bases,seed) for seed in t2.SEEDS}
    base_spread = {condition:{metric:{'mean_of_three_Base_training_seed_means':statistics.mean(per_base[str(s)][condition][metric]['mean'] for s in t2.SEEDS),
        'sample_SD_across_three_Base_training_seed_means_ddof1':statistics.stdev(per_base[str(s)][condition][metric]['mean'] for s in t2.SEEDS),
        'per_training_seed_mean':{str(s):per_base[str(s)][condition][metric]['mean'] for s in t2.SEEDS},
        'n_training_seeds':3,'automatic_G4_threshold':False}
        for metric in METRICS} for condition in [*CONDITIONS,'equal_five_type_macro']}
    pairs = [('A2','C1'),('A2','flow_only'),('C1','flow_only'),*[(arm,'matched_Base') for arm in ARMS]]
    comparisons,appendix = {},[]
    for candidate,reference in pairs:
        label = candidate+'_minus_'+reference;per_condition = {};per_clip = {}
        for condition in CONDITIONS:
            left = clips(methods,candidate,condition)
            right = clips(bases,TRAINING_SEED,condition) if reference=='matched_Base' else clips(methods,reference,condition)
            per_clip[condition] = {r:{m:left[r][m]-right[r][m] for m in METRICS} for r in records}
            per_condition[condition] = {m:_stats([per_clip[condition][r][m] for r in records]) for m in METRICS}
            for r in records:appendix.append({'comparison':label,'condition':condition,'record_id':r,
                'candidate_minus_reference':per_clip[condition][r],
                'clean_subtracted_candidate_minus_reference':None if condition=='clean' else {
                    m:per_clip[condition][r][m]-per_clip['clean'][r][m] for m in METRICS}})
        per_condition['equal_five_type_macro'] = {m:_stats([statistics.mean(per_clip[c][r][m] for c in CONDITIONS[1:]) for r in records]) for m in METRICS}
        comparisons[label] = per_condition
    return {'status':'complete_descriptive_statistics_G4_not_decided','method_observations':1152,'Base_observations_reused':1152,
        'method_training_seed_count':1,'method_training_seed':TRAINING_SEED,'Base_training_seed_count':3,
        'per_arm':per_arm,'original_Base_per_training_seed':per_base,'descriptive_Base_seed_spread':base_spread,
        'paired_comparisons':comparisons,'paired_clip_appendix':appendix,
        'G4_criteria':protocol['G4_criteria']['file'],'G4_decision':'not_computed_requires_review',
        'p_values_computed':False,'Stage5_significance_decision':False,'confidence_intervals_computed':False,
        'aggregation':'Mean sampler42/43 within each clip; equal32clips; equal all5degradation types. Clean reported separately.',
        'delta_sign':'All deltas are candidate minus reference in original units. Higher PSNR/SSIM means higher fidelity; higher LPIPS/tLP100 means larger error. Harm-gap signs retain the frozen T2 definition.',
        'uncertainty_scope':'Between-clip SD is descriptive dispersion, not training-seed uncertainty. Three Base seed means/SD are reported without selecting or interpreting a G4 threshold.'}


def compact_csv(report):
    stream = io.StringIO();writer = csv.writer(stream)
    writer.writerow(['section','label','condition','metric','mean','sample_SD','SD_unit','n'])
    for section,key in [('method','per_arm'),('paired_delta','paired_comparisons')]:
        for label,conditions in report[key].items():
            for condition,metrics in conditions.items():
                for metric,value in metrics.items():writer.writerow([section,label,condition,metric,value['mean'],value['between_clip_sample_SD'],'clips_after_sampler_mean',32])
    for condition,metrics in report['descriptive_Base_seed_spread'].items():
        for metric,value in metrics.items():writer.writerow(['Base_reference','three_training_seeds',condition,metric,
            value['mean_of_three_Base_training_seed_means'],value['sample_SD_across_three_Base_training_seed_means_ddof1'],'training_seed_means',3])
    return stream.getvalue()


def safe_run(args):
    root = args.project_root.resolve();run = args.run_dir.resolve()
    need(run!=root/'runs' and run.is_relative_to(root/'runs')
         and not run.is_relative_to(args.pilot_run.resolve()),'Fresh evaluation output must be an independent staging runs child')
    return run


def prepare(args):
    run = safe_run(args)
    need(not run.exists(),'Fresh evaluation run must not exist')
    protocol = collect_protocol(args)
    need(all(not run.is_relative_to(Path(directory).resolve()) for directory in protocol['prerequisite_directories']),
         'Evaluation cannot write inside prerequisite runs or sealed input assets')
    need('torch' not in sys.modules,'Metadata-only evaluation prepare must not import torch; use a clean CLI process')
    run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    save(run/'status.json',{'status':'planned','created_utc':now(),'protocol_sha256':file_sha(run/'protocol.json'),
        'G4_criteria_sha256':protocol['G4_criteria']['file']['sha256'],'G4_decision':'not_computed_requires_review',
        'training_updates':0,'torch_imported':False})
    return {'status':'planned','run':str(run),'protocol_sha256':file_sha(run/'protocol.json'),
        'G4_criteria_sha256':protocol['G4_criteria']['file']['sha256'],'G4_decision':'not_computed_requires_review'}


def _report_text(report):
    lines = ['# Complete pilot validation scores; G4 remains undecided','',
        'All three method students use training seed20261102 and2048 continuation updates. Each has384 restores; total1152.',
        'Original three-Base observations are reused without Base inference. All32 clips, two sampler seeds and five types are retained.',
        '', 'Deltas below are candidate minus reference. The SD is between clips after averaging sampler42/43; it is not training-seed uncertainty.',
        'The JSON/CSV include every common pool/type and descriptive three-Base training-seed SD. No p-value, significance or G4 threshold is calculated.',
        '', '| Comparison | Masked PSNR delta dB | Masked LPIPS delta | Mask-touch tLP100 delta |',
        '| --- | ---: | ---: | ---: |']
    for label,conditions in report['paired_comparisons'].items():
        metrics = conditions['equal_five_type_macro']
        values = [metrics[k]['mean'] for k in ('masked.psnr_rgb_db','masked.lpips_alex_v0_1','temporal.mask_touch_tLP100')]
        lines.append('| '+label+' | '+' | '.join(f'{v:.8f}' for v in values)+' |')
    lines.extend(['','G4 criteria raw SHA: '+report['G4_criteria']['sha256'],
        'G4 decision: `not_computed_requires_review`. Review the prospectively frozen criteria separately.',
        'No test211/external pixels, extra training, sweep or quality-based checkpoint selection was performed.',''])
    return '\n'.join(lines)


def _failure(run,error,details,protocol_sha,current,observations,roles,started):
    failure = {'status':'failed_stop_no_retry','case':current,'error':f'{type(error).__name__}: {error}',
        'traceback':details,'protocol_sha256':protocol_sha,'completed_restores':len(observations),
        'completed_roles':roles,'partial_observations':observations,'training_updates':0,'automatic_retry':False,
        'G4_decision':'not_computed_requires_review','wall_seconds':time.perf_counter()-started}
    save(run/'failure.json',failure)
    state = {'status':'failed_stop_no_retry','failure_sha256':file_sha(run/'failure.json'),
        'protocol_sha256':protocol_sha,'G4_decision':'not_computed_requires_review','training_updates':0}
    if (run/'summary.json').exists():
        summary = read(run/'summary.json');summary.update(status='failed_stop_no_retry',
            completion_revoked_by_execution_failure=True,execution_failure_sha256=state['failure_sha256'])
        save(run/'summary.json',summary,replace=True);state['summary_sha256']=file_sha(run/'summary.json')
    save(run/'status.json',state,replace=True)


def execute(args):
    root = args.project_root.resolve();run = safe_run(args)
    need(args.expected_protocol_sha and file_sha(run/'protocol.json')==args.expected_protocol_sha,'Exact prepared protocol raw SHA is required')
    protocol = read(run/'protocol.json');state = read(run/'status.json')
    need(all(not run.is_relative_to(Path(directory).resolve()) for directory in protocol['prerequisite_directories']),
         'Evaluation output escaped its independent run boundary')
    need(state['status']=='planned' and state['protocol_sha256']==args.expected_protocol_sha,
         'Only never-started planned evaluations execute; no retry/resume')
    with (run/'execute.lock').open('x') as stream:stream.write(str(__import__('os').getpid())+'\n')
    save(run/'status.json',{'status':'running','protocol_sha256':args.expected_protocol_sha,
        'G4_decision':'not_computed_requires_review','training_updates':0},replace=True)
    torch = model = vae = wrapper = reader = None
    observations,roles,stable_inputs = [],[],{}
    started = time.perf_counter();current = 'CPU_bindings'
    def record_failure(error,details):
        _failure(run,error,details,args.expected_protocol_sha,current,observations,roles,started)
    def record_secondary(primary,secondary,details):
        save(run/'cleanup_failure.json',{'primary_error_preserved':f'{type(primary).__name__}: {primary}',
            'primary_failure_sha256':file_sha(run/'failure.json'),'secondary_cleanup_error':f'{type(secondary).__name__}: {secondary}',
            'traceback':details,'training_updates':0})
        status = read(run/'status.json');status['cleanup_failure_sha256']=file_sha(run/'cleanup_failure.json')
        save(run/'status.json',status,replace=True)
    def final_cleanup():
        nonlocal model,vae,wrapper,reader,current
        current = 'final_GPU_cleanup';errors = []
        for name,action in [('policy',lambda:getattr(vae,'_vsr_t1_cache_policy',None).remove() if vae is not None and getattr(vae,'_vsr_t1_cache_policy',None) is not None else None),
                            ('features',lambda:model.clear_features() if model is not None else None),
                            ('reader',lambda:reader.close() if reader is not None else None)]:
            try:action()
            except BaseException as error:errors.append(name+': '+repr(error))
        model = vae = wrapper = reader = None
        try:
            gc.collect()
            if torch is not None and torch.cuda.is_initialized():torch.cuda.empty_cache()
        except BaseException as error:errors.append('allocator: '+repr(error))
        need(not errors,'Final cleanup errors: '+'; '.join(errors))
    # Uses the independent actual Base GPU-owner lease, including cleanup and
    # primary/secondary receipts. This is separate from every run's O_EXCL file.
    with g3._leased_lifecycle(g3.ACTUAL_RUNS,final_cleanup,record_failure,record_secondary):
        need(protocol==collect_protocol(args),'Prepared pilot/criteria/config/input/runtime sources changed; stop')
        import os
        need(all(os.environ.get(k)==v for k,v in t2.ENVIRONMENT.items()),'Exact strict/offline environment is required')
        evaluation = protocol['evaluation_protocol']
        classifier = t2.import_helper(evaluation['helpers']['gpu_classifier']['path'],'stage4_evaluation_GPU_guard')
        guard = t2.gpu_idle_guard(evaluation['desktop_binding'],classifier);save(run/'GPU_idle_before_load.json',guard)
        need(guard['passed'],'GPU is busy or unverified; never overlap training or another evaluation')
        sys.path[:0] = [str(root),evaluation['dependencies']['root']]
        import torch as torch_module
        torch = torch_module
        need(not torch.cuda.is_initialized(),'Strict setup must precede CUDA initialization')
        torch.set_num_threads(2)
        from vsr.reproducibility import configure_reproducibility,tensor_state_hashes
        runtime = configure_reproducibility(protocol['checkpoints'][ARMS[0]]['config']);flags = runtime['flags']
        need(flags['deterministic_algorithms'] and not flags['deterministic_warn_only'] and flags['sdp_math']
             and not any(flags[k] for k in ('sdp_flash','sdp_memory_efficient','sdp_cudnn')),'Strict native math backend required')
        save(run/'runtime.json',runtime)
        t2.verify_imported_dependencies(evaluation['dependencies'])
        from diffusers import AutoencoderKLWan,WanTransformer3DModel,UniPCMultistepScheduler
        t2.verify_native_sources(evaluation['native_numerical_sources'],{'AutoencoderKLWan':AutoencoderKLWan,
            'WanTransformer3DModel':WanTransformer3DModel,'UniPCMultistepScheduler':UniPCMultistepScheduler})
        from vsr.checkpoints import load_training_checkpoint
        from vsr.infer_video import load_frozen_vae
        from vsr.inference import restore_video
        from vsr.frozen_evaluation import quantize_rgb,_save_pngs,stable_hq_input_contract
        from vsr.metric_pools import paired_metrics
        from vsr.perceptual_metrics import PerceptualMetrics
        from vsr.native_qa_restore import _contexts_empty
        for name,module in list(sys.modules.items()):
            if name=='vsr' or name.startswith('vsr.'):
                need(Path(module.__file__).resolve().is_relative_to(root/'vsr'),'Imported core source shadows frozen staging')
        cap = evaluation['caption'];cache = torch.load(cap['cache']['path'],map_location='cpu',weights_only=True)
        text = cache.get(cap['text']) if isinstance(cache,dict) else None
        need(isinstance(text,torch.Tensor) and list(text.shape)==cap['shape']==[1,512,4096]
             and str(text.dtype)==cap['dtype']=='torch.bfloat16' and bool(torch.isfinite(text).all()),'Exact cached native caption required; no T5 fallback')
        header = json.dumps([str(text.dtype),list(text.shape)],separators=(',',':')).encode()+b'\n'
        need(hashlib.sha256(header+text.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==cap['embedding_sha256'],'Cached caption tensor SHA differs')
        text = text.detach().cpu();del cache
        helper = {key:t2.import_helper(value['path'],'stage4_eval_'+key) for key,value in evaluation['helpers'].items()}
        seal = read(evaluation['cohort_seal']['path'])
        reader = t2.FramewiseReader(evaluation['dataset_manifest']['path'],seal)
        perceptual = PerceptualMetrics(evaluation['metric_weights'][0]['path'],evaluation['metric_weights'][1]['path'],device='cpu')
        need(not torch.cuda.is_initialized(),'CPU source/text/metric setup initialized CUDA unexpectedly')
        base_rows = read(protocol['Base_observations']['path'])['observations']
        for arm in ARMS:
            current = {'arm':arm,'phase':'student_VAE_load'};cp = protocol['checkpoints'][arm]
            model,audit = load_training_checkpoint(cp['checkpoint']['path'],device='cuda',
                allowed_root=protocol['allowed_method_checkpoint_root'],variant=arm)
            need(audit['checkpoint_sha256']==cp['checkpoint']['sha256'] and audit['step']==UPDATES
                 and audit['variant']==arm and audit['config']==cp['config'] and audit['inference_weights']=='student'
                 and audit['training_only_state_restored'] is False and audit['EMA_weights_selected'] is False
                 and model.projector is None and not hasattr(model,'reference_adapter'),'Loaded model is not the exact method student')
            model.eval();vae = load_frozen_vae(cp['config'],device='cuda')
            need(not vae.training and all(not p.requires_grad and p.dtype==torch.float32 and p.device.type=='cuda' for p in vae.parameters()),
                 'Original frozen FP32 CUDA VAE required')
            model_ref,vae_ref = weakref.ref(model),weakref.ref(vae)
            wrapper = t2.RotatingCropRestore(restore_video,helper['policy'].install,helper['ledger'].validate_ledger,
                lambda x:bool(torch.isfinite(x).all()),_contexts_empty)
            wrapper.register(vae,cp['config'])
            before = tensor_state_hashes(model)['all_trainable']['sha256'];role_started = time.perf_counter()
            need(before==cp['expected_final_student_trainable_sha256'],'Loaded student parameters differ from the completed training final-state SHA')
            save(run/f'roles/{arm}/load_audit.json',audit)
            for item in seal['cohort']:
                record = item['record']['id']
                for condition in CONDITIONS:
                    source = reader.load('Stage2-val-'+condition,record,num_frames=33,start=0)
                    lr,hr,metadata = source['lr'],source['hr'],source['metadata']
                    contract = stable_hq_input_contract(metadata,lr,hr,fps=source['fps'],frames=33)
                    key = condition+'/'+record
                    need(key not in stable_inputs or stable_inputs[key]==contract,'Student candidates used different exact target/condition inputs')
                    stable_inputs[key] = contract;_,target = quantize_rgb(hr)
                    for sample in SAMPLES:
                        current = {'arm':arm,'record_id':record,'condition':condition,'sampling_seed':sample}
                        directory = run/f'results/{arm}/{record}/{condition}/seed_{sample}'
                        need(not directory.exists(),'Fresh-only row exists');directory.mkdir(parents=True)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();restore_started = time.perf_counter()
                        video,restoration = wrapper(model,vae,lr,text,cp['config'],seed=sample)
                        torch.cuda.synchronize();restore_seconds = time.perf_counter()-restore_started
                        peak = {'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
                        need(max(peak.values())<=min(16*1024**3,torch.cuda.get_device_properties(0).total_memory),'Evaluation exceeded16GiB/physical-device cap')
                        quantized,measured = quantize_rgb(video);pngs = _save_pngs(quantized,directory/'png',run)
                        need(len(pngs)==33,'Every restore must save all33 original frames')
                        metric_started = time.perf_counter();full = paired_metrics(measured,target,perceptual,excluded_reference=0)
                        full['primary_pool']='all_frames';score = t2.score_masked(full,item['mask'])
                        observation = {'arm':arm,'training_seed':TRAINING_SEED,'method_updates':UPDATES,
                            'record_id':record,'source_id':item['record']['source_id'],'condition':condition,'sampling_seed':sample,
                            'checkpoint_sha256':cp['checkpoint']['sha256'],'cohort_seal_digest':seal['seal_digest'],'score':score}
                        row = {'status':'completed','observation':observation,'source_metadata':metadata,'stable_input_contract':contract,
                            'full_frame_pair_metrics':full,'restoration':restoration,'PNG_artifacts':pngs,
                            'restore_seconds':restore_seconds,'metric_seconds':time.perf_counter()-metric_started,
                            'cuda_peak_bytes':peak,'training_updates':0,'G4_decision':'not_computed_requires_review'}
                        save(directory/'metrics.json',row)
                        save(directory/'complete.json',{'protocol_sha256':args.expected_protocol_sha,
                            'metrics_sha256':file_sha(directory/'metrics.json'),'PNG_count':33,'resume_allowed':False})
                        observation.update(row_metrics_path=str(directory/'metrics.json'),row_metrics_sha256=file_sha(directory/'metrics.json'))
                        observations.append(observation)
                        save(run/'progress.json',{'completed_restores':len(observations),'last':current,'training_updates':0},replace=True)
                        del video,quantized,measured,full
                    del source,lr,hr,target
            after = tensor_state_hashes(model)['all_trainable']['sha256']
            need(before==after and all(p.grad is None for p in model.parameters()) and _contexts_empty(model),'Inference changed student weights or left gradients/features')
            counters = wrapper.finish(vae)
            roles.append({'arm':arm,'training_seed':TRAINING_SEED,'method_updates':UPDATES,'policy_counters':counters,
                'trainable_sha256_before':before,'trainable_sha256_after':after,'wall_seconds':time.perf_counter()-role_started})
            model = vae = wrapper = None;gc.collect();torch.cuda.empty_cache()
            need(model_ref() is None and vae_ref() is None,'Previous method model/VAE remains resident across roles')
            save(run/'completed_roles.json',{'roles':roles,'training_updates':0},replace=True)
        need(len(observations)==1152 and len(roles)==3 and len(stable_inputs)==192,'Complete method matrix required')
        current = 'paired_descriptive_statistics'
        need(t2.digest(stable_inputs)==read(protocol['Base_summary']['path'])['stable_input_contracts_digest'],
             'Method and original Base matrices did not consume byte-identical bound crop inputs')
        report = summarize(observations,base_rows,seal,protocol)
        save(run/'observations.json',{'protocol_sha256':args.expected_protocol_sha,'observations':observations,
            'method_training_seed_count':1,'training_updates':0})
        save(run/'descriptive_statistics.json',report)
        for name,value in [('descriptive_statistics.csv',compact_csv(report)),('descriptive_statistics.md',_report_text(report))]:
            with (run/name).open('x') as stream:stream.write(value)
        need(protocol==collect_protocol(args),'Sources/criteria/prerequisite artifacts changed during evaluation')
        summary = {'status':'completed_full_method_matrix_descriptive_only','protocol_sha256':args.expected_protocol_sha,
            'observations':1152,'prediction_PNGs':38016,'method_training_seed_count':1,'Base_inference_restores':0,
            'roles':roles,'observations_sha256':file_sha(run/'observations.json'),
            'statistics_sha256':file_sha(run/'descriptive_statistics.json'),'G4_criteria_sha256':protocol['G4_criteria']['file']['sha256'],
            'G4_decision':'not_computed_requires_review','training_updates':0,'test_or_external_pixels_used':False,
            'wall_seconds':time.perf_counter()-started,'next_action':'review_complete_statistics_against_prospectively_frozen_G4_criteria'}
        save(run/'summary.json',summary)
        save(run/'status.json',{'status':summary['status'],'protocol_sha256':args.expected_protocol_sha,
            'summary_sha256':file_sha(run/'summary.json'),'G4_decision':summary['G4_decision'],'training_updates':0},replace=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pilot-run',type=Path,required=True)
    parser.add_argument('--g4-criteria',type=Path,required=True)
    parser.add_argument('--expected-g4-criteria-sha256',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--expected-protocol-sha')
    args = parser.parse_args()
    if args.mode=='prepare':
        need(args.expected_protocol_sha is None,'prepare creates its protocol SHA');result = prepare(args)
    else:result = execute(args)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
