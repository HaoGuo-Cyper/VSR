"""Fresh-only frozen HQ internal test, delegating the production evaluator.

Only the verified instance T1 cache policy and small public-ledger lifecycle
audits are added. No producer/reader/sampler/metric arithmetic is replaced.
"""
import argparse
import copy
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
POLICY = W/'work/t1_vae_cache_policy.py'
POLICY_SHA = 'd94340b33868912f43633c71ef4a34a97b2dd7eb7b74e9d40ded363b09417270'
COMMON = W/'work/probe_global_HRI_encode_allocator.py'
COMMON_SHA = 'f6f1da551bdeff048253ec86aeb5679e7a5cfa609b0023dc92029664f8446514'
QA_HELPER = W/'work/run_native_QA_T1_policy.py'
QA_HELPER_SHA = 'dd99144dace5f9c7543881ad6c6daf2718936ea17d6ab639729a09fb525100e0'
ACCEPTED = W/'outputs/VSR_evaluation_primary_checkpoint_cohort_acceptance_v1.json'
ACCEPTED_SHA = 'c2e054cd00c2eca25a8c190bee4282ea1af7c0d272c36453bbe20f2d3a3a3557'
CAPTION = W/'outputs/VSR_final_evaluation_text_cache_preflight.json'
CAPTION_SHA = 'c903115c195e5f8f818c46743e82d8233309c92400fb8cdad964308263a1568e'
ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG':':4096:8',
    'PYTORCH_ALLOC_CONF':'expandable_segments:True','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
    'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
HQ_FIXED = {'schema':'vsr_final_HQ_internal_test_protocol_v1','split':'test',
    'frames':9,'target_hw':[128,224],'temporal_stride':4,'data_seed':20261102,
    'modes':['identity','blur','noise','compression','combined'],'sampling_seeds':[42,43],
    'steps':20,'tile_hw':[256,448],'overlap':64,'record_count':211,'source_count':180}
LIFECYCLE = 'public_remove_reinstall_same_VAE_after_each_successful_restore'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    need(isinstance(value,dict),'Expected JSON object: '+str(path))
    return value


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_protocol(protocol, accepted, test_records):
    """Metadata only; deliberately callable in CPU fixture tests."""
    need(protocol.get('status') == 'root_finalized_before_test_pixels' and protocol.get('proposal_only') is False,
         'A root-finalized protocol is required; proposals cannot execute')
    hq = protocol['hq_internal_test']
    need(all(hq.get(key) == value for key,value in HQ_FIXED.items()),'HQ numerical/data protocol changed')
    records = sorted(test_records,key=lambda row:row['id'])
    need(len(records) == 211 and len({row['source_id'] for row in records}) == 180
         and hq['record_ids'] == [row['id'] for row in records]
         and hq['source_by_record_id'] == {row['id']:row['source_id'] for row in records},'Require all211/180 sealed test identities')
    members = {row['label']:row for row in accepted['members']}
    need(len(members) == 8 and set(protocol['checkpoints']) == set(members),'All8 accepted original checkpoints required')
    for label, member in members.items():
        spec = protocol['checkpoints'][label]
        for key, expected in {'path':member['checkpoint_path'],'checkpoint_sha256':member['checkpoint_sha256'],
                'family':member['family'],'variant':member['variant'],'step':8192,'data_seed':20261102,'training_seed':20261102}.items():
            need(spec.get(key) == expected,'Accepted checkpoint cohort mismatch: '+label+'/'+key)
    need({(row['family'],row['variant']) for row in protocol['checkpoints'].values()} ==
         {(family,variant) for family in ('no_reference','reference') for variant in ('Base','Align','Relation','Both')},
         'Need each original family/variant once')
    execution = protocol['execution']; policy = execution['T1_policy']
    need(execution['inference_device'] == 'cuda' and execution['metric_device'] == 'cpu'
         and execution['caption'] == 'A high quality video.' and execution['startup_environment'] == ENVIRONMENT,
         'Frozen devices/caption/startup environment changed')
    need(policy['path'] == str(POLICY) and policy['sha256'] == POLICY_SHA
         and isinstance(policy['max_records'],int) and not isinstance(policy['max_records'],bool)
         and policy['max_records'] >= 6330 and policy['lifecycle'] == LIFECYCLE,
         'Declare sufficient public policy capacity and the reviewed rotation lifecycle')
    return members


def validate_ledger(rows, enabled, tile_count):
    need(isinstance(rows,list) and len(rows) == tile_count+int(enabled),'Missing or extra per-restore policy records')
    for index,row in enumerate(rows):
        if enabled and index == 0:
            need(row['route'] == 'T1_no_future_cache_retention' and row['status'] == 'completed'
                 and row['encoder_pre_calls'] == row['encoder_post_calls'] == 1
                 and row['feat_idx_argument_identity_preserved'] is True and not row['cleanup_errors']
                 and row['parameter_buffer_names_devices_dtypes_identities_and_flags_unchanged'] is True
                 and all(item['tensor_bytes'] == 0 for item in row['encoder_end_cache'].values()),'T1 route/cleanup audit failed')
        else:
            need(row['route'] == 'original_delegate' and row['encoder_hooks_installed'] is False,
                 'Every multi-frame tile must delegate the unchanged original encode')


class RotatingPolicyRestore:
    """No VAE/model/policy strong references persist outside each real call."""
    def __init__(self, restore, install, *, max_records, finite, contexts_empty, progress=None):
        self.restore, self.install = restore, install
        self.max_records, self.finite, self.contexts_empty = max_records, finite, contexts_empty
        self.progress = progress; self.roles = []; self.refs = []; self.last_audit = None

    def register(self, vae, config):
        need(all(ref() is None for ref in self.refs),'A prior role VAE leaked across checkpoints')
        need('encode' not in vae.__dict__ and getattr(vae,'_vsr_t1_cache_policy',None) is None,
             'Default VAE must start with its unwrapped official class encode')
        role = {'role_id':len(self.roles),'config_sha256':digest(config),
            'reference_enabled':bool(config.get('reference',{}).get('enabled',False)),
            'restores':0,'T1':0,'original_delegate':0,'events':0,'rotations':0}
        self.roles.append(role); self.refs.append(weakref.ref(vae))
        vae._vsr_HQ_policy_role_id = role['role_id']
        self.install(vae,max_records=self.max_records)
        return vae

    @staticmethod
    def removed(vae):
        return ('encode' not in vae.__dict__ and getattr(vae,'_vsr_t1_cache_policy',None) is None)

    def __call__(self, model, vae, lrv, text, config, **options):
        active = getattr(vae,'_vsr_t1_cache_policy',None)
        role = self.roles[vae._vsr_HQ_policy_role_id]
        audit = {'policy_sha256':POLICY_SHA,'lifecycle':LIFECYCLE,'role_id':role['role_id'],
            'max_records':self.max_records,'status':'running','numeric_restore_calls':0,'cleanup_errors':[]}
        self.last_audit = audit
        enabled = options.get('reference_image') is not None
        try:
            need(active is not None and active.dropped_records == 0 and active.ledger == [],'Expected a fresh public per-restore ledger')
            need(role['config_sha256'] == digest(config) and (role['reference_enabled'] or not enabled),'VAE family/config changed')
            audit['numeric_restore_calls'] = 1
            video, metadata = self.restore(model,vae,lrv,text,config,**options)  # Sole numerical delegate.
            audit['ledger'] = active.ledger  # At most this restore's two HQ records; never cumulative O(N^2).
            tile_count = metadata['tile_count']
            need(isinstance(tile_count,int) and not isinstance(tile_count,bool) and tile_count > 0
                 and len(metadata['tiles']) == tile_count and metadata['reference']['enabled'] is enabled
                 and metadata['reference']['encode_count'] == int(enabled),'Actual tile/reference metadata mismatch')
            need(active.dropped_records == 0,'Policy ledger overflow')
            validate_ledger(audit['ledger'],enabled,tile_count)
            need(list(video.shape) == [1,3,9,128,224] and metadata['output_shape'] == [1,3,9,128,224]
                 and self.finite(video) and self.contexts_empty(model),'Output finite/geometry/context cleanup failure')
            active.remove()
            need(self.removed(vae),'Public remove did not restore original encode before reinstall')
            audit['original_encode_restored_before_reinstall'] = True
            self.install(vae,max_records=self.max_records)
            need(getattr(vae,'_vsr_t1_cache_policy',None) is not None,'Next policy was not installed')
            role['restores'] += 1; role['T1'] += int(enabled); role['original_delegate'] += tile_count
            role['events'] += len(audit['ledger']); role['rotations'] += 1
            audit.update(status='passed',observed_T1=int(enabled),observed_original_delegate=tile_count,
                next_policy_installed=True,no_parameter_hash_per_restore=True)
            metadata['T1_cache_policy_audit'] = copy.deepcopy(audit)
            if self.progress is not None and sum(item['restores'] for item in self.roles) % 100 == 0:
                self.progress(self.roles)
            return video,metadata
        except BaseException as error:
            audit.update(status='failed',error=f'{type(error).__name__}: {error}')
            try:
                if 'ledger' not in audit and active is not None:
                    audit['ledger'] = active.ledger
            except BaseException as cleanup_error:
                audit['cleanup_errors'].append('partial public ledger: '+str(cleanup_error))
            # No reinstall/retry after any failure, including metadata/rotation failure.
            current = getattr(vae,'_vsr_t1_cache_policy',None)
            for candidate in (current,active):
                if candidate is not None:
                    try: candidate.remove()
                    except BaseException as cleanup_error: audit['cleanup_errors'].append('remove: '+str(cleanup_error))
            try: vae.clear_cache()
            except BaseException as cleanup_error: audit['cleanup_errors'].append('official clear_cache: '+str(cleanup_error))
            audit['original_encode_restored_after_failure'] = self.removed(vae)
            for note in audit['cleanup_errors']:
                error.add_note(note)
            raise

    def verify_totals(self):
        need(len(self.roles) == 8 and sum(item['reference_enabled'] for item in self.roles) == 4
             and all(ref() is None for ref in self.refs),'All eight role VAE lifetimes must finish without strong-reference leaks')
        for role in self.roles:
            expected = {'restores':4220,'T1':2110,'original_delegate':4220,'events':6330,'rotations':4220} if role['reference_enabled'] else {
                'restores':2110,'T1':0,'original_delegate':2110,'events':2110,'rotations':2110}
            need(all(role[key] == value for key,value in expected.items()),'Full role restore/event budget incomplete')
        return {'status':'passed','roles':self.roles,'restores':25320,'events':33760,
                'T1':8440,'original_delegate':25320,'all_role_VAEs_collected':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--expected-protocol-sha',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--unit',required=True)
    args = parser.parse_args()
    need(re.fullmatch('[a-zA-Z0-9_.@-]+[.]service',args.unit),'Explicit direct service unit required')
    need(re.fullmatch('[0-9a-f]{64}',args.expected_protocol_sha) and sha(args.protocol) == args.expected_protocol_sha,'Root raw protocol SHA mismatch')
    run = args.run_dir.expanduser().resolve(); child = run/'hq_test'
    need(run.is_relative_to(P/'runs') and run != P/'runs' and (not run.exists() or not any(run.iterdir())),
         'Fresh own outer run required; preserve failures and do not resume naked child')
    run.mkdir(parents=True,exist_ok=True)
    def save(path,value):
        temporary = path.with_name(path.name+'.partial')
        temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); temporary.replace(path)
    started = time.perf_counter(); contract = None; stage = 'preGPU_metadata'; restore = None
    try:
        protocol = read(args.protocol)
        need(sha(ACCEPTED) == ACCEPTED_SHA and sha(CAPTION) == CAPTION_SHA
             and sha(COMMON) == COMMON_SHA and sha(QA_HELPER) == QA_HELPER_SHA and sha(POLICY) == POLICY_SHA,'Frozen evidence/adapter/helper changed')
        accepted = read(ACCEPTED); manifest = read(P/'manifests/hq_vsr.json')
        members = validate_protocol(protocol,accepted,[row for row in manifest['records'] if row['split'] == 'test'])
        hq = protocol['hq_internal_test']
        for key,expected_path in (('manifest',P/'manifests/hq_vsr.json'),
                                  ('temporal_inventory',P/'manifests/hq_temporal_stride4_inventory.json')):
            need(hq[key]['path'] == str(expected_path) and sha(expected_path) == hq[key]['sha256'],
                 'Frozen HQ source catalogue/inventory changed')
        need(hq['embedded_manifest_sha256'] == manifest['manifest_sha256']
             and hq['record_ids_sha256'] == digest(hq['record_ids'])
             and hq['source_by_record_id_sha256'] == digest(hq['source_by_record_id']),
             'Frozen HQ record/source identity digests changed')
        for key,value in ENVIRONMENT.items(): need(os.environ.get(key) == value,'Startup environment changed: '+key)
        common = module_from(COMMON,'vsr_HQ_frozen_shared'); qa_helper = module_from(QA_HELPER,'vsr_HQ_resource_proof')
        config,old_failure,environment = common.metadata_preflight()
        bindings = protocol['bindings']; producer = bindings['producer']
        need(producer['commit'] == common.COMMIT and producer['code_sha256'] == environment['code_sha256']
             and producer['environment_path'] == str(common.SOURCE_ENV)
             and sha(common.SOURCE_ENV) == producer['environment_sha256'],'Frozen producer106/environment binding changed')
        resource = qa_helper.verify_resource(bindings['resource']['run'],bindings['resource']['summary_sha256'],environment['code_sha256'])
        qa_run = Path(bindings['QA']['run']).resolve()
        need(qa_run.is_relative_to(P/'runs') and sha(qa_run/'summary.json') == bindings['QA']['summary_sha256'],'Actual QA summary SHA changed')
        qa_summary,qa_status,qa_protocol = (read(qa_run/name) for name in ('summary.json','status.json','execution_policy.json'))
        need(qa_summary['status'] == qa_status['status'] == 'passed' and qa_summary['native_20step_operator_QA_passed'] is True
             and qa_status['summary_sha256'] == bindings['QA']['summary_sha256']
             and qa_status['protocol_sha256'] == qa_summary['protocol_sha256'] == digest(qa_protocol)
             and qa_protocol['policy_sha256'] == POLICY_SHA and qa_protocol['code106_sha256'] == environment['code_sha256'],
             'Actual policy QA did not pass with matching source')
        for name,value in qa_summary['child_file_sha256'].items(): need(sha(Path(qa_summary['child_run'])/name) == value,'QA child binding changed')
        need(sha(qa_run/'policy_event_join.json') == qa_summary['policy_event_join_sha256']
             and read(qa_run/'policy_event_join.json')['status'] == 'passed','QA exact policy join proof missing')
        for key in ('CPU_parity','caption_preflight','Wan_VAE_source'):
            need(sha(bindings[key]['path']) == bindings[key]['sha256'],'Bound asset/source changed: '+key)
        need(bindings['CPU_parity']['sha256'] == qa_helper.CPU_PARITY_SHA
             and bindings['caption_preflight'] == {'path':str(CAPTION),'sha256':CAPTION_SHA}
             and bindings['CUDA_parity']['run'] == str(qa_helper.PARITY)
             and bindings['CUDA_parity']['summary_sha256'] == qa_helper.PARITY_SUMMARY_SHA
             and sha(qa_helper.PARITY/'summary.json') == qa_helper.PARITY_SUMMARY_SHA,'Real parity/caption evidence differs')
        caption = read(CAPTION); need(caption['status'] == 'passed' and set(caption['entries']) == set(members),'All8 caption evidence required')
        import torch
        from vsr.frozen_evaluation import _default_preview,_default_vae,_default_restore,_json
        from vsr.evaluate_hq import run_hq_evaluation
        from vsr.reference_diagnostics import assert_gpu_idle
        from vsr.reproducibility import configure_reproducibility,reproducibility_flags
        from vsr.native_qa_restore import _contexts_empty
        from vsr.hq_evaluation_data import _video_noise_seed
        save = _json
        need(not torch.cuda.is_initialized(),'Fresh pre-CUDA process required')
        need(hq['noise_seed_by_record_id'] == {record:_video_noise_seed(20261102,record) for record in hq['record_ids']},
             'Frozen actual per-video robustness noise seeds changed')
        import importlib
        installed_VAE = importlib.import_module('diffusers.models.autoencoders.autoencoder_kl_wan')
        need(Path(installed_VAE.__file__).resolve() == Path(bindings['Wan_VAE_source']['path']).resolve(),
             'Bound Wan VAE file is not the actually installed imported source')
        previews = {}
        for label,spec in protocol['checkpoints'].items():
            path = Path(spec['path']).resolve()
            need(path == Path(members[label]['checkpoint_path']) and sha(path) == spec['checkpoint_sha256'],'Accepted task bytes changed')
            value = _default_preview(path,allowed_root=P/'runs')  # CPU weights_only preview, no model/optimizer restore.
            cfg = read(path.parent.parent/'config.json'); entry = caption['entries'][label]
            need(value['checkpoint_sha256'] == spec['checkpoint_sha256'] and value['step'] == 8192
                 and value['variant'] == spec['variant'] and value['config'] == cfg
                 and bool(cfg.get('reference',{}).get('enabled',False)) is (spec['family'] == 'reference')
                 and cfg['seed'] == cfg['training'].get('seed',cfg['seed']) == 20261102
                 and cfg['model'] == config['model'],'Actual payload8192/config/family/official base changed')
            for prefix in ('config','origin_config'):
                need(sha(entry[prefix+'_path']) == entry[prefix+'_sha256'],'Caption config changed')
            need(entry['checkpoint_path'] == str(path) and entry['checkpoint_sha256'] == spec['checkpoint_sha256']
                 and sha(entry['cache_path']) == entry['cache_file_sha256'],'Actual caption asset identity changed')
            previews[label] = value
        guard = assert_gpu_idle(P); need(args.unit in guard['own_diagnostic_service_units'],'Register actual direct worker unit before GPU load')
        runtime = configure_reproducibility(config)
        adapter = module_from(POLICY,'vsr_HQ_public_rotation_policy')
        execution = {'schema':'vsr_frozen_HQ_test_T1_policy_execution_v1','worker_sha256':sha(Path(__file__)),
            'root_protocol_path':str(args.protocol.resolve()),'root_protocol_sha256':args.expected_protocol_sha,
            'root_protocol_file_sha256':args.expected_protocol_sha,
            'code_commit':common.COMMIT,'code106_sha256':environment['code_sha256'],
            'root_protocol':protocol,'policy_sha256':POLICY_SHA,'lifecycle':LIFECYCLE,
            'source_helpers_sha256':{str(COMMON):COMMON_SHA,str(QA_HELPER):QA_HELPER_SHA},
            'payload_preview':previews,'payload_preview_scope':'CPU weights_only own8 payloads; optimizer tensors may deserialize but no optimizer/model/RNG restoration',
            'actual_resource_proof':resource,'actual_QA_summary_sha256':bindings['QA']['summary_sha256'],
            'runtime_before_GPU_load':runtime,'startup_environment':ENVIRONMENT,'GPU_guard':guard,
            'factory_overrides':['restore','vae'],'factory_overrides_are_synthetic_fixtures':False,
            'no_full_parameter_hash_per_restore':True,'fresh_only_no_resume':True,
            'child_run':str(child),'training_updates':0,'test_used_for_selection':False,
            'memory_policy_boundary':'QA kept one policy per VAE; here public remove/reinstall rotates after every successful restore on same VAE. Same original per-encode mathematics; no parameter/device/dtype/backend/sampler/reader/metric changes.'}
        contract = digest(execution)
        save(run/'execution_policy.json',execution); save(run/'protocol.json',execution)
        save(run/'status.json',{'status':'running','protocol_sha256':contract,'child_run':str(child)})
        restore = RotatingPolicyRestore(_default_restore,adapter.install,
            max_records=protocol['execution']['T1_policy']['max_records'],
            finite=lambda value: bool(torch.isfinite(value).all()),contexts_empty=_contexts_empty,
            progress=lambda roles:save(run/'policy_role_counters.json',{'roles':roles}))
        def vae_factory(cfg,*,device):
            vae = _default_vae(cfg,device=device)
            need(all(value.device.type == 'cuda' and value.dtype == torch.float32 for value in vae.parameters()),'Original CUDA FP32 VAE required')
            return restore.register(vae,cfg)
        stage = 'unchanged_full_HQ_internal_test'
        save(run/'GPU_guard_before_load.json',assert_gpu_idle(P))
        result = run_hq_evaluation(P,{label:spec['path'] for label,spec in protocol['checkpoints'].items()},
            split='test',modes=HQ_FIXED['modes'],frames=9,height=128,width=224,temporal_stride=4,seed=20261102,
            seeds=(42,43),steps=20,tile_hw=(256,448),overlap=64,device='cuda',metric_device='cpu',
            run_dir=child,caption='A high quality video.',preview_max_side=None,
            selection_note='accepted original8; frozen outer execution_policy_sha256='+contract,
            factories={'vae':vae_factory,'restore':restore})
        stage = 'completion_budget_and_seal_audit'
        need(result['status'] == 'completed' and result['resumed_complete_sequences'] == []
             and set(result['results']) == set(members)
             and all(set(groups) == {'HQ-test-'+mode for mode in HQ_FIXED['modes']}
                     and all(value['video_count'] == 211 for value in groups.values()) for groups in result['results'].values()),
             'Every original8/full211/all5 dataset must complete in fresh run')
        totals = restore.verify_totals(); save(run/'policy_role_counters.json',totals)
        need(sha(args.protocol) == args.expected_protocol_sha and sha(POLICY) == POLICY_SHA
             and sha(Path(__file__)) == execution['worker_sha256'],'Execution protocol/source changed')
        summary = {'status':'passed','protocol_sha256':contract,'child_run':str(child),
            'child_file_sha256':{name:sha(child/name) for name in ('protocol.json','status.json','summary.json')},
            'policy_totals':totals,'runtime_flags':reproducibility_flags(),'training_updates':0,
            'test_used_for_selection':False,'wall_seconds':time.perf_counter()-started,
            'wall_scope':'worker entry incl metadata, own CPU payload previews, production full test and wrapper counters; excludes interpreter startup/final serialization',
            'interpretation':'actual test measurements only; no checkpoint/parameter/seed selection or claimed module benefit here'}
        save(run/'summary.json',summary); save(run/'status.json',{'status':'passed','protocol_sha256':contract,
            'summary_sha256':sha(run/'summary.json'),'child_run':str(child)})
        print(json.dumps({'status':'passed','protocol_sha256':contract,'child_run':str(child)}))
    except BaseException as error:
        failure = {'status':'failed','protocol_sha256':contract,'stage':stage,
            'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc(),
            'partial_roles':restore.roles if restore is not None else [],
            'last_policy_audit':restore.last_audit if restore is not None else None,
            'child_run':str(child),'child_file_sha256':{name:sha(child/name) for name in
                ('protocol.json','status.json','summary.json','progress.json') if (child/name).is_file()},
            'wall_seconds':time.perf_counter()-started,'no_silent_fallback_or_retry':True,'training_updates':0}
        save(run/'failure_observations.json',failure); save(run/'status.json',{'status':'failed',
            'protocol_sha256':contract,'stage':stage,'error':failure['error'],'wall_seconds':failure['wall_seconds'],
            'failure_observations_sha256':sha(run/'failure_observations.json')})
        raise
    finally:
        restore = None; gc.collect()


if __name__ == '__main__':
    main()
