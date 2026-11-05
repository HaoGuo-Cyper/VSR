"""Fixed HQ-val native QA with verified T1 policy and post-hoc exact event join.

Workspace driver. It never changes the production reader/restore/sampler.
Root must freeze plans, resource proof and metric device before launching.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import weakref

P = Path('/home/hugo/Projects/VSR')
W = Path('/home/hugo/Documents/Codex/2026-11-02/files-pasted-by-the-user-wan2')
sys.path.insert(0, str(P))
UNIT = 'vsr-native-qa-t1-policy.service'
COMMON = W/'work/probe_global_HRI_encode_allocator.py'
COMMON_SHA = 'f6f1da551bdeff048253ec86aeb5679e7a5cfa609b0023dc92029664f8446514'
POLICY = W/'work/t1_vae_cache_policy.py'
POLICY_SHA = 'd94340b33868912f43633c71ef4a34a97b2dd7eb7b74e9d40ded363b09417270'
RESOURCE_DRIVER_SHA = '0206bd698134a016c93b4e5c33848ad0d8a7876b2972f4c7f67b68ec6bde754a'
PARITY = P/'runs/T1_VAE_cache_policy_parity_resource_20261103T011701.294518Z'
PARITY_SUMMARY_SHA = '65e4f538ac120dcf28c1e29fa022f8fe3eef42f77cb71896ccc33aae42adae73'
CPU_PARITY = W/'outputs/VSR_T1_VAE_cache_policy_CPU_parity.json'
CPU_PARITY_SHA = 'd2b2e5c4d5baaced491abd9fd9759e4a84172da7b1d5b9cb75089c42fb60bb30'
ACCEPTED = W/'outputs/VSR_evaluation_primary_checkpoint_cohort_acceptance_v1.json'
ACCEPTED_SHA = 'c2e054cd00c2eca25a8c190bee4282ea1af7c0d272c36453bbe20f2d3a3a3557'
BASES = {
    'NoRef_Base': {'path': str(P/'runs/formal_hq_v2_noref_20261102T142214.588571Z/Base/checkpoint_008192.pt'),
                   'sha256': '8f3d1e19edc24b8ad72219ef207b85c4c798f241f6f204fac57e57c11960806c', 'reference': False},
    'HRI_Base': {'path': str(P/'runs/formal_hq_v2_reference_20261102T171545.936007Z/Base/checkpoint_008192.pt'),
                 'sha256': 'b9664949b02270a8240b792d28521f8c7b4c9db9c7170ee6f1b9ef349ea5d5ac', 'reference': True},
}
PROPOSED = {
    'vsr_native_crop_operator_qa_predeclaration_v1': ('native_crop_qa_predeclaration.v2.proposed.json',
        '9bcd143df9256ba48839cc84a9917ce9a957832be7e8bc849b49a1c28b47c6fd', 'HQ-val-nativecrop', 3),
    'vsr_native_fullframe_operator_qa_predeclaration_v1': ('native_fullframe_qa_predeclaration.v2.proposed.json',
        'c4db814c6c8b20e453312e18d8e295712169b11110b782fc1a16c56ae4f00f05', 'HQ-val-nativefull', 16),
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    need(isinstance(value, dict), 'Expected JSON object: '+str(path))
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound_inside(path, root):
    path = Path(path).resolve()
    need(path.is_relative_to(Path(root).resolve()), 'Path escaped its declared run: '+str(path))
    return path


def verify_resource(run, expected_summary_sha, code):
    run = bound_inside(run, P/'runs')
    need(re.fullmatch('[0-9a-f]{64}', expected_summary_sha) and sha(run/'summary.json') == expected_summary_sha,
         'Bind the real full resource summary SHA explicitly')
    status, summary, protocol = (read(run/name) for name in ('status.json','summary.json','execution_policy.json'))
    need(read(run/'protocol.json') == protocol and status['status'] == summary['status'] == 'passed'
         and summary['full_resource_probe_passed'] is True
         and status['summary_sha256'] == expected_summary_sha
         and status['protocol_sha256'] == summary['protocol_sha256'] == digest(protocol)
         and protocol['schema'] == 'vsr_native_resource_T1_cache_policy_execution_v1'
         and protocol['worker_sha256'] == RESOURCE_DRIVER_SHA and protocol['policy_sha256'] == POLICY_SHA
         and protocol['code106_sha256'] == code and protocol['code_commit'] == '04228644605f5a3d96834ba05dde636636400817'
         and protocol['metric_devices'] == ['cpu','cuda:0']
         and protocol['factory_overrides'] == ['restore','vae']
         and protocol['factory_overrides_are_synthetic_fixtures'] is False,
         'Need actual matching full resource pass under the verified policy')
    inner = bound_inside(protocol['inner_run'], run)
    need(inner == run/'native_probe' and summary['inner_run'] == str(inner), 'Unexpected inner resource run')
    need(set(summary['inner_file_sha256']) == {'protocol.json','status.json','summary.json','phase_observations.json'},
         'Complete inner resource file bindings required')
    for name, expected in summary['inner_file_sha256'].items():
        need(sha(inner/name) == expected, 'Resource inner binding changed: '+name)
    inner_summary, inner_status = read(inner/'summary.json'), read(inner/'status.json')
    need(inner_summary['protocol_sha256'] == inner_status['protocol_sha256'] == digest(read(inner/'protocol.json'))
         and inner_status['summary_sha256'] == sha(inner/'summary.json'), 'Inner resource protocol/summary seal changed')
    cases = {'no_reference_maximum_native_off': {'T1':0,'delegate':96},
             'reference_maximum_paired_on': {'T1':1,'delegate':30},
             'reference_maximum_native_off': {'T1':0,'delegate':96}}
    need(inner_summary['status'] == inner_status['status'] == 'passed'
         and set(inner_summary['native_cases']) == set(cases)
         and all(value['status'] == 'completed' for value in inner_summary['native_cases'].values())
         and set(inner_summary['MUSIQ_resource_forwards']) == {'cpu','cuda:0'}
         and all(value['status'] == 'completed' and value['forward_completed'] is True
                 and value['synthetic_quality_value_discarded'] is True for value in inner_summary['MUSIQ_resource_forwards'].values())
         and set(summary['case_policy_audit']) == set(cases)
         and all(summary['case_policy_audit'][key]['observed_routes'] == value for key,value in cases.items())
         and summary['all_role_VAEs_collected'] is True, 'Full resource cases/metrics/policy evidence incomplete')
    return {'run': str(run), 'summary_sha256': expected_summary_sha,
            'outer_file_sha256': {name:sha(run/name) for name in ('execution_policy.json','protocol.json','status.json','summary.json')},
            'inner_run': str(inner), 'inner_file_sha256': summary['inner_file_sha256']}


def validate_event(event, expected_role):
    need(event['status'] == 'completed' and len(event['policy_ledger']) == 1, 'Incomplete observed encode/policy event')
    row = event['policy_ledger'][0]
    if expected_role == 'whole_HRI_T1':
        need(row['route'] == 'T1_no_future_cache_retention' and row['status'] == 'completed'
             and row['encoder_pre_calls'] == row['encoder_post_calls'] == 1
             and row['feat_idx_argument_identity_preserved'] is True and not row['cleanup_errors']
             and row['parameter_buffer_names_devices_dtypes_identities_and_flags_unchanged'] is True
             and all(value['tensor_bytes'] == 0 for value in row['encoder_end_cache'].values()), 'T1 policy semantics failed')
    else:
        need(expected_role == 'LR_tile' and row['route'] == 'original_delegate'
             and row['encoder_hooks_installed'] is False, 'Multi-frame encode did not delegate unchanged')


def join_events(child, qa_summary, instances, plans, configs):
    """Pure metadata/SHA join; no model, reader, decode or quality decision."""
    joined = []; used = set(); all_events = 0; t1_count = 0
    need(set(qa_summary['cases']) == {value[2] for value in PROPOSED.values()}, 'Exactly two native QA geometries required')
    for dataset, outcome in qa_summary['cases'].items():
        case = bound_inside(outcome['run'], child)
        need(sha(case/'summary.json') == outcome['summary_sha256']
             and sha(case/'native_operator_case_audit.json') == outcome['structural_audit_sha256'], 'Original QA case seal changed')
        case_summary, case_status, case_protocol = (read(case/name) for name in ('summary.json','status.json','protocol.json'))
        need(case_summary['status'] == case_status['status'] == 'completed'
             and case_summary['protocol_sha256'] == case_status['protocol_sha256'] == outcome['protocol_sha256'] == digest(case_protocol)
             and case_summary['resumed_complete_sequences'] == []
             and read(case/'native_operator_case_audit.json')['status'] == 'passed', 'Original QA artifact/playback audit failed')
        tile_count = plans[dataset]['restoration']['tile_count']
        files = list((case/'results').glob('*/*/*/metrics.json'))
        need(len(files) == 2, 'Expected exactly one NoRef and one HRI row per geometry')
        seen_labels = set()
        for path in files:
            row = read(path); label = row['label']
            need(label in BASES and label not in seen_labels, 'Unexpected or duplicate checkpoint label')
            seen_labels.add(label)
            base = BASES[label]; modes = ('hri_on','hri_off') if base['reference'] else ('no_reference',)
            actual_spec = case_protocol['checkpoints'][label]
            need(actual_spec['path'] == base['path'] and actual_spec['checkpoint_sha256'] == base['sha256']
                 and actual_spec['variant'] == 'Base' and actual_spec['step'] == 8192
                 and actual_spec['config'] == configs[label], 'Actual evaluated checkpoint/config differs from outer binding')
            need(row['status'] == 'completed' and row['checkpoint_sha256'] == base['sha256'] and row['dataset'] == dataset
                 and row['sequence_id'] == plans[dataset]['selection']['record_id']
                 and row['evaluated_frames'] == 17 and set(row['modes']) == set(modes), 'Actual QA row differs from fixed cohort/modes/data')
            seal = read(path.parent/'complete.json')
            need(seal['metrics_sha256'] == sha(path) and seal['contract_sha256'] == row['contract_sha256'], 'Original metric completion seal changed')
            for artifact in row['artifacts']:
                need(sha(bound_inside(case/artifact['path'], case)) == artifact['sha256'], 'Original PNG/playback/input artifact changed')
            expected = []; segments = []
            for mode in modes:  # Actual frozen evaluator execution order; not file/glob/factory order.
                entries = row['modes'][mode]['seeds']
                need(len(entries) == 1 and entries[0]['seed'] == 42
                     and entries[0]['hri_supplied'] is (mode == 'hri_on')
                     and entries[0]['reference_index'] == (0 if mode == 'hri_on' else None)
                     and entries[0]['png_count'] == 17, 'Wrong actual QA seed/reference/frame mode')
                entry = entries[0]; audit = entry['restoration']['native_operator_audit']; calls = audit['VAE_encode_calls']
                png_directory = bound_inside(case/entry['png_directory'],case)
                need(sorted(path.name for path in png_directory.iterdir()) == [f'frame_{i:08d}.png' for i in range(17)],
                     'Actual authoritative output PNG sequence incomplete')
                need(audit['status'] == 'passed' and not audit['audit_errors'] and all(audit['checks'].values())
                     and len(calls) == tile_count+int(mode == 'hri_on')
                     and [call['ordinal'] for call in calls] == list(range(len(calls))), 'Original native encode audit incomplete')
                start = len(expected)
                for index, call in enumerate(calls):
                    role = 'whole_HRI_T1' if mode == 'hri_on' and index == 0 else 'LR_tile'
                    need(call['role'] == role, 'Original reference/tile order differs')
                    expected.append((call['shape'], call['pixels_sha256'], role))
                segments.append({'dataset':dataset,'sequence':row['sequence_id'],'checkpoint_label':label,
                    'checkpoint_sha256':base['sha256'],'mode':mode,'seed':42,'start':start,'stop':len(expected),
                    'metrics_file':str(path),'metrics_sha256':sha(path)})
            candidates = []
            for instance in instances:
                actual = instance['events']
                if (instance['config_sha256'] == digest(actual_spec['config']) and instance['reference_enabled'] is base['reference']
                        and len(actual) == len(expected)
                        and all(event['local_ordinal'] == i and event['shape'] == item[0]
                                and event['pixels_sha256'] == item[1] and event['dtype'] == 'torch.float32'
                                for i,(event,item) in enumerate(zip(actual,expected)))):
                    candidates.append(instance)
            need(len(candidates) == 1 and candidates[0]['instance_id'] not in used, 'Encode-to-real-QA-row join ambiguous, missing or reused')
            instance = candidates[0]; used.add(instance['instance_id'])
            for event, item in zip(instance['events'], expected):
                validate_event(event, item[2]); t1_count += int(item[2] == 'whole_HRI_T1'); all_events += 1
            for segment in segments:
                segment.update(instance_id=instance['instance_id'],
                    global_event_ordinals=[event['global_ordinal'] for event in instance['events'][segment['start']:segment['stop']]])
                joined.append(segment)
        need(seen_labels == set(BASES), 'Missing actual Base family row')
    need(len(instances) == 4 and used == {item['instance_id'] for item in instances}
         and len(joined) == 6 and all_events == 59 and t1_count == 2
         and all_events == sum(len(item['events']) for item in instances)
         and sorted(event['global_ordinal'] for item in instances for event in item['events']) == list(range(59)),
         'All four VAE streams/six restores/59 events must be consumed exactly once')
    return {'status':'passed','joins':joined,'VAE_instances':4,'restores':6,'events':59,'T1':2,'original_delegate':57,
            'rule':'unique complete same-config ordered exact normalized encode-input shape/hash streams; no label inferred at construction'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, action='append', required=True)
    parser.add_argument('--metric-device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--resource-run', type=Path, required=True)
    parser.add_argument('--expected-resource-summary-sha', required=True)
    args = parser.parse_args()
    need(sha(COMMON) == COMMON_SHA and sha(POLICY) == POLICY_SHA, 'Frozen shared worker/policy changed')
    common = load_module('vsr_QA_T1_shared', COMMON)
    run = bound_inside(args.run_dir, P/'runs')
    need(run != P/'runs' and (not run.exists() or not any(run.iterdir())), 'Use a new empty own run; no automatic resume/retry')
    run.mkdir(parents=True, exist_ok=True)
    def save(path,value):
        temporary = path.with_name(path.name+'.partial')
        temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        temporary.replace(path)
    started = time.perf_counter(); contract = None; stage = 'CPU_preflight'
    instances = []; refs = []; event_ordinal = 0; child = run/'native_qa'
    try:
        config, previous_failure, environment = common.metadata_preflight()
        for path, expected in ((CPU_PARITY,CPU_PARITY_SHA),(PARITY/'summary.json',PARITY_SUMMARY_SHA),(ACCEPTED,ACCEPTED_SHA)):
            need(sha(path) == expected, 'Frozen accepted/parity evidence changed: '+str(path))
        need(read(CPU_PARITY)['status'] == read(PARITY/'summary.json')['status'] == 'passed', 'Require actual CPU/CUDA parity')
        resource = verify_resource(args.resource_run,args.expected_resource_summary_sha,environment['code_sha256'])
        configs = {}
        for label, spec in BASES.items():
            path = Path(spec['path']); need(sha(path) == spec['sha256'], 'Accepted original Base bytes changed')
            configs[label] = read(path.parent.parent/'config.json')
            need(bool(configs[label].get('reference',{}).get('enabled',False)) is spec['reference'], 'Own Base family mismatch')
        need(len(args.plan) == 2, 'Exactly two final raw plans required')
        plans = {}; plan_bindings = []; plan_objects = []
        for path in args.plan:
            path = path.expanduser().resolve(); plan = read(path)
            need(plan.get('schema') in PROPOSED, 'Unknown native QA plan schema')
            name, proposed_sha, dataset, tiles = PROPOSED[plan['schema']]
            proposed_path = W/'work/native_qa_plan'/name
            need(sha(proposed_path) == proposed_sha and dataset not in plans, 'Proposed reference plan changed or geometry duplicated')
            proposed = read(proposed_path)
            for field in ('selection','input_preparation','restoration','required_modes'):
                need(plan.get(field) == proposed[field], 'Final QA numeric/data/mode declaration differs from v2: '+field)
            need(plan['checkpoint_policy']['no_reference_SHA256'] == BASES['NoRef_Base']['sha256']
                 and plan['checkpoint_policy']['reference_SHA256'] == BASES['HRI_Base']['sha256']
                 and plan['restoration']['tile_count'] == tiles, 'Final QA Base/checkpoint/geometry binding differs')
            plans[dataset] = plan; plan_objects.append(plan)
            plan_bindings.append({'path':str(path),'sha256':sha(path),'canonical_sha256':digest(plan),
                'dataset':dataset,'proposed_reference_path':str(proposed_path),'proposed_reference_sha256':proposed_sha})
        import torch
        from vsr.frozen_evaluation import _default_vae, _json
        from vsr.native_operator_qa import run_native_operator_qa
        from vsr.native_qa_restore import _tensor_sha
        from vsr.reference_diagnostics import assert_gpu_idle
        from vsr.reproducibility import configure_reproducibility, reproducibility_flags
        save = _json
        need(not torch.cuda.is_initialized(), 'Fresh pre-CUDA process required')
        guard = assert_gpu_idle(P)
        need(UNIT in guard['own_diagnostic_service_units'], 'Register this direct QA policy service before worker')
        runtime = configure_reproducibility(config)
        adapter = load_module('vsr_native_QA_T1_policy', POLICY)
        execution = {'schema':'vsr_native_QA_T1_policy_execution_v1','worker_sha256':sha(Path(__file__)),
            'policy_sha256':POLICY_SHA,'code_commit':common.COMMIT,'code106_sha256':environment['code_sha256'],
            'shared_worker_sha256':COMMON_SHA,'CPU_parity_source':{'path':str(CPU_PARITY),'sha256':CPU_PARITY_SHA},
            'CUDA_parity_source':{'run':str(PARITY),'summary_sha256':PARITY_SUMMARY_SHA},
            'accepted_cohort':{'path':str(ACCEPTED),'sha256':ACCEPTED_SHA},'checkpoints':BASES,
            'configs':configs,'resource_proof':resource,'raw_plan_bindings':plan_bindings,
            'plans':plan_objects,'inference_device':'cuda:0','metric_device':args.metric_device,
            'metric_device_selection':'explicit root CLI choice after resource feasibility; no QA-quality selection',
            'runtime_before_GPU_load':runtime,'startup_environment':{key:os.environ.get(key) for key in common.ENV_REQUIRED},
            'factory_overrides':['vae'],'factory_overrides_are_synthetic_fixtures':False,
            'reader_restore_sampler':'unchanged NativeQAReader, AuditedNativeRestore and original sampler/noise/operator',
            'memory_policy':'verified instance-only T1 fresh feat_cache=None; non-T1 exact original delegation; FP32 VAE/original devices/full math SDPA; no offload',
            'event_join':'post-QA unique complete per-VAE same-config ordered actual encode shape+pixels SHA streams to original native audit modes; ambiguity fails',
            'expected_totals':{'VAE_instances':4,'restores':6,'events':59,'T1':2,'original_delegate':57},
            'child_run':str(child),'training_updates':0,'test_or_external_data_used':False,
            'quality_scores_used_for_acceptance':False,'quality_hypotheses_verified':False,
            'prior_failure_provenance':previous_failure,'GPU_guard':guard}
        contract = digest(execution)
        save(run/'execution_policy.json',execution); save(run/'protocol.json',execution)
        save(run/'status.json',{'status':'running','protocol_sha256':contract,'child_run':str(child)})

        def vae_factory(cfg, *, device):
            nonlocal event_ordinal
            need(all(ref() is None for ref in refs), 'Previous role VAE retained across QA roles/geometries')
            vae = _default_vae(cfg,device=device)
            need(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in vae.parameters()), 'Original CUDA FP32 VAE required')
            active = adapter.install(vae); original_encode = vae.encode
            instance = {'instance_id':len(instances),'config_sha256':digest(cfg),
                        'reference_enabled':bool(cfg.get('reference',{}).get('enabled',False)),
                        'events':[],'no_dataset_mode_assumed_at_construction':True}
            instances.append(instance); refs.append(weakref.ref(vae))
            def observed_encode(*positional,**keywords):
                nonlocal event_ordinal
                pixels = positional[0] if positional else keywords.get('x')
                event = {'global_ordinal':event_ordinal,'local_ordinal':len(instance['events']),
                    'shape':list(pixels.shape),'dtype':str(pixels.dtype),'pixels_sha256':_tensor_sha(pixels),'status':'running'}
                event_ordinal += 1; instance['events'].append(event)
                start = len(active.ledger); error = None
                try:
                    result = original_encode(*positional,**keywords)
                    event['status'] = 'completed'
                    return result
                except BaseException as caught:
                    error = caught; event.update(status='failed',error=f'{type(caught).__name__}: {caught}')
                    raise
                finally:
                    try:
                        event['policy_ledger'] = active.ledger[start:]
                        need(active.dropped_records == 0, 'Policy ledger overflow')
                        save(run/'encode_observations.json',{'instances':instances})
                    except BaseException as observation_error:
                        if error is None:
                            raise
                        error.add_note('Encode observer failed without replacing original: '+str(observation_error))
            vae.encode = observed_encode
            return vae

        stage = 'unchanged_native_operator_QA'
        result = run_native_operator_qa(P,{label:spec['path'] for label,spec in BASES.items()},plan_objects,
            run_dir=child,device='cuda',metric_device=args.metric_device,factories={'vae':vae_factory})
        stage = 'post_QA_unique_policy_event_join'
        qa_status = read(child/'qa_status.json')
        need(result['status'] == qa_status['status'] == 'passed'
             and result['protocol_sha256'] == qa_status['protocol_sha256'] == digest(read(child/'qa_protocol.json'))
             and qa_status['summary_sha256'] == sha(child/'qa_summary.json'), 'Original native QA did not pass/seal')
        joined = join_events(child,result,instances,plans,configs)
        need(all(ref() is None for ref in refs), 'QA VAE instances leaked after original lifecycle cleanup')
        need(sha(POLICY) == POLICY_SHA and sha(Path(__file__)) == execution['worker_sha256']
             and all(sha(item['path']) == item['sha256'] for item in plan_bindings), 'Execution source/final raw plan changed')
        save(run/'policy_event_join.json',joined)
        summary = {'status':'passed','protocol_sha256':contract,'child_run':str(child),
            'child_file_sha256':{name:sha(child/name) for name in ('qa_protocol.json','qa_status.json','qa_summary.json')},
            'encode_observations_sha256':sha(run/'encode_observations.json'),'policy_event_join_sha256':sha(run/'policy_event_join.json'),
            'all_role_VAEs_collected':True,'native_20step_operator_QA_passed':True,
            'training_updates':0,'test_or_external_data_used':False,'quality_scores_used_for_acceptance':False,
            'quality_hypotheses_verified':False,'runtime_flags':reproducibility_flags(),
            'wall_seconds':time.perf_counter()-started,'wall_scope':'outer worker entry including source/protocol/resource checks, original QA and metadata/artifact event-join audits; excludes interpreter startup/final serialization'}
        save(run/'summary.json',summary)
        save(run/'status.json',{'status':'passed','protocol_sha256':contract,'summary_sha256':sha(run/'summary.json'),
            'child_run':str(child),'native_20step_operator_QA_passed':True})
        print(json.dumps({'status':'passed','protocol_sha256':contract,'child_run':str(child)}))
    except BaseException as error:
        failure = {'status':'failed','protocol_sha256':contract,'stage':stage,
            'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc(),
            'child_run':str(child),'child_file_sha256':{name:sha(child/name) for name in
                ('qa_protocol.json','qa_status.json','qa_summary.json','native_restore_failure_observations.json') if (child/name).is_file()},
            'partial_encode_observations':instances,'wall_seconds':time.perf_counter()-started,
            'training_updates':0,'test_or_external_data_used':False,'quality_scores_used_for_acceptance':False,
            'native_20step_operator_QA_passed':False,'no_silent_fallback_or_retry':True}
        save(run/'failure_observations.json',failure)
        save(run/'status.json',{'status':'failed','protocol_sha256':contract,'stage':stage,'error':failure['error'],
            'failure_observations_sha256':sha(run/'failure_observations.json'),'wall_seconds':failure['wall_seconds']})
        raise
    finally:
        gc.collect()


if __name__ == '__main__':
    main()
