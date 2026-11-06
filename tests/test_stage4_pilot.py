"""CPU fixtures for sealed pilot metadata, frozen latent reads and lifecycle.

No real checkpoint, source video, CUDA context or scientific training is used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'scripts'))
import stage4_pilot as worker
from tests.test_stage3_resource_gate import timing_rows, competing_worker_lock_probe


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def canonical(value):
    value.pop('_sha256',None)
    value['_sha256'] = hashlib.sha256((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    return value


def base_config():
    return canonical({'stage':'stage2_protocol33_noref_base','seed':20261102,'_source_config_sha256':'c'*64,
        'model':{'repo_id':'Wan-AI/Wan2.1-T2V-1.3B-Diffusers','revision':'0fad780a534b6463e45facd96134c9f345acfa5b','local_dir':'/unused/official'},
        'conditioning':{'hri_enabled':False,'rank':8,'alpha':8,'injection_layers':[0,5,9],'alignment_layer':9,'stem_channels':128,'projector_hidden':1024,'teacher_dim':768},
        'data':{'num_frames':33,'temporal_stride':1,'height':128,'width':224,'scale':4},'reference':{'enabled':False},
        'formal_training':{'updates':8192,'base_only':True},'stage2_contract':{'T5_load_allowed':False,'expected_training_clips':1641},
        'evaluation':{'sampling_steps':20,'seeds':[42,43]},
        'training':{'seed':20261102,'variants':['Base'],'updates':8192,'checkpoint_at':[2048,4096,8192],
            'evaluate_at':[0,2048,4096,8192],'learning_rate':1e-4,'weight_decay':.01,'max_grad_norm':1.}})


def memory():return {'peak_allocated_bytes':1024,'peak_reserved_bytes':2048,'physical_device_bytes':16*1024**3}


def g3_fixture(root):
    directory=root/'runs/G3';directory.mkdir(parents=True)
    protocol={'schema':'stage3_G3_full_optimizer_diagnostic_v1','project_root':str(root),
        'budget':worker.g3.BUDGET,'roles':list(worker.ROLES),'warmup_per_role':2,'timed_per_role':20,
        'completed_T2':{'directory':str(root/'runs/T2')},
        'matched_Base':{'checkpoint':{'path':'/unused/Base/checkpoint_008192.pt','sha256':'b'*64},'config':base_config()}}
    write(directory/'protocol.json',protocol);psha=worker.file_sha(directory/'protocol.json')
    rows=timing_rows(a2_seconds=1.25)
    for row in rows:row['memory']=memory()
    count=worker.g3.UpdateCounter()
    count.by_role={name:{'attempted':n,'applied':n,'returned_steps':n} for name,n in
        {'semantic_direct_core':2,'semantic_A2_rho0':2,'flow_only':22,'C1':22,'A2':22}.items()}
    observations={'protocol_sha256':psha,'rows':rows,**count.snapshot()}
    gate=worker.g3.summarize_roles(rows)
    checks={'same_layer':{'layer':9,'exact_equal':True,'teacher_detached':True,'student_restored':True,'optimizer_updates':0},
        'semantic_direct_core':{'updates':2,'exact_trajectory_equal':None,'optimizer_foreach':False,'memory':memory()},
        'semantic_A2_rho0':{'updates':2,'exact_trajectory_equal':True,'optimizer_foreach':False,'memory':memory()}}
    semantics={'status':'passed_exact_GPU_semantics','checks':checks,'diagnostic_optimizer_updates':4,
        'historical_GPU_foreach_None_equivalence_claimed':False,'tolerance_fallback':False}
    for name,item in (('observations.json',observations),('G3.json',gate),('semantics.json',semantics)):write(directory/name,item)
    roles=[{'role':arm,'trainable_sha256_before':'a'*64,'trainable_sha256_after':'f'*64,'returned_updates':22,
        'optimizer_updates_applied':22,'EMA_updates':22 if arm=='A2' else 0,'fresh_Base_SHA256':'b'*64,
        'frozen_identity_versions_unchanged':True,'mode_and_gradient_cleanup_passed':True,'memory':memory()} for arm in worker.ROLES]
    summary={'status':'passed_G3_resource_cost','G3_passed':True,'protocol_sha256':psha,'budget':worker.g3.BUDGET,
        'roles':roles,'resident_VAE_unchanged_FP32':True,'quality_inference':False,'reusable_checkpoints_written':False,
        **count.snapshot(),**{name+'_sha256':worker.file_sha(directory/filename) for name,filename in
            (('semantics','semantics.json'),('observations','observations.json'),('G3','G3.json'))}}
    write(directory/'summary.json',summary)
    write(directory/'status.json',{'status':'passed_G3_resource_cost','G3_passed':True,'protocol_sha256':psha,
        'summary_sha256':worker.file_sha(directory/'summary.json'),**count.snapshot()})
    return directory,protocol


def reseal_g3(directory):
    summary=worker.read(directory/'summary.json')
    for name in ('semantics','observations','G3'):summary[name+'_sha256']=worker.file_sha(directory/(name+'.json'))
    write(directory/'summary.json',summary)
    status=worker.read(directory/'status.json');status['summary_sha256']=worker.file_sha(directory/'summary.json')
    write(directory/'status.json',status)


def latent_fixture(root,count=3):
    import torch
    source=root/'cache';source.mkdir()
    contract={'producer_run':str(source),'manifest_sha256':'e'*64,'official_revision':'f'*40,
        'data_seed':20261102,'caption':'A high quality video.','producer_config_sha256':'c'*64,
        'encoder_hashes':{'vsr/data.py':'d'*64,'vsr/preflight.py':'d'*64,'vsr/stage2_preparation.py':'d'*64}}
    specs=[]
    for index in range(count):
        identifier=f'clip{index}'
        record={'id':identifier,'source_id':f'source{index}','sha256':str(index)*64,'split':'train','path':str(root/f'raw{index}.mp4')}
        meta={'data_version':'hq-vsr-paired-v2','record_id':identifier,'source_id':record['source_id'],
            'source_sha256':record['sha256'],'source_path':record['path'],'split':'train','seed':20261102,
            'window_seed':int(worker.digest(['window',20261102,identifier])[:16],16),
            'source_indices':list(range(33)),'temporal_stride':1,'source_hw':[256,448],'resize_hw':[128,224],
            'crop_tlhw':[0,0,128,224],'valid_region':'entire_crop_no_padding','degradation':worker.DEGRADATION,
            'original_fps':24.,'output_fps':24.,'fps_policy':'preserve_original_no_temporal_resampling','caption':contract['caption']}
        b={'data_cache_key':worker.digest(meta),'manifest_sha256':contract['manifest_sha256'],
            'official_revision':contract['official_revision'],'config_sha256':contract['producer_config_sha256'],'code_sha256':contract['encoder_hashes']}
        meta['binding']=b
        directory=source/f'clip_{index}';write(directory/'metadata.json',meta)
        pair={'clean':torch.full(worker.g3.METHOD['latent_shape'],float(index)),
              'low':torch.zeros(worker.g3.METHOD['latent_shape']),'binding':b}
        torch.save(pair,directory/'fresh_latents.pt')
        specs.append({'index':index,'record':record,'metadata':worker.binding(directory/'metadata.json'),
                      'latents':worker.binding(directory/'fresh_latents.pt'),'binding_sha256':worker.digest(b)})
    return {'contract':contract,'clips':specs,'latent_shape':worker.g3.METHOD['latent_shape']}


def pilot_fixture(root):
    directory=root/'runs/pilot';directory.mkdir(parents=True)
    base={'path':'/unused/actual/Base/checkpoint_008192.pt','sha256':'b'*64}
    protocol={'schema':'stage4_pilot_training_v1','project_root':str(root),'allowed_pilot_checkpoint_root':str(root/'runs'),
        'matched_Base':{'checkpoint':base},'completed_T2':{'files':{'protocol.json':{'sha256':'e'*64}}},
        'completed_G3':{'directory':str(root/'runs/G3'),'files':{'protocol.json':{'sha256':'c'*64}},'matched_Base_initial_trainable_sha256':'a'*64},
        'method_configs':{arm:worker.method_config(base_config(),arm) for arm in worker.ROLES},
        'training_inputs':{'clips':[{'record':{'id':f'clip{i}'},'latents':{'sha256':hashlib.sha256(str(i).encode()).hexdigest()}}
                                   for i in range(worker.TRAIN_CLIPS)]},
        'optimizer':{'learning_rate':1e-4},'quality_evaluation':{'criteria':None,'quality_evaluation_blocked':True}}
    write(directory/'protocol.json',protocol);psha=worker.file_sha(directory/'protocol.json')
    reports={}
    for arm in worker.ROLES:
        arm_run=directory/'arm_runs'/arm;target=arm_run/arm;target.mkdir(parents=True)
        cfg=protocol['method_configs'][arm];write(arm_run/'config.json',cfg)
        cps=[]
        for step in worker.CHECKPOINT_AT:
            path=target/f'checkpoint_{step:06d}.pt';path.write_bytes(f'CPU metadata fixture {arm}/{step}'.encode())
            cps.append({'path':str(path),'sha256':worker.file_sha(path),'step':step,'variant':arm,
                'restoration_architecture':'noref_base_v1','inference_parameters':'student',
                'config_sha256':cfg['_sha256'],'existing_artifacts_overwritten':False})
        log=target/'training.jsonl'
        with log.open('w') as stream:
            for ordinal in range(worker.UPDATES):
                index=ordinal%worker.TRAIN_CLIPS;active=arm=='A2'
                row={'step':ordinal+1,'arm':arm,'data_index':index,'seconds':.1,'lr':1e-4,
                    'fm':2.,'sd':.2 if active else 0.,'total':2.16 if active else 2.,'sigma':.8,
                    'teacher_forward':active,'cosine_mean':.8 if active else None,'mask_spans':[] if arm=='flow_only' else [[[2,4]]],
                    'matched_FM':{'data_index':index,'record_id':f'clip{index}',
                        'latent_file_sha256':protocol['training_inputs']['clips'][index]['latents']['sha256'],
                        'sigma':.8,'noise':f'noise{ordinal}','timestep':f'time{ordinal}'},
                    'matched_corruption_RNG':{'mask':'unused' if arm=='flow_only' else f'mask{ordinal}',
                                              'corruption':'unused' if arm=='flow_only' else f'corruption{ordinal}'},
                    'memory':memory(),'feature_diagnostics':None,'feature_diagnostics_snapshot_collected':False}
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
        report={'status':'completed','arm':arm,'variant':arm,'run':str(arm_run),'training_seed':20261102,
            'updates':worker.UPDATES,'scientific_training_updates':worker.UPDATES,'config_sha256':cfg['_sha256'],
            'final_checkpoint':cps[-1],'checkpoints':cps,'training_jsonl':worker.binding(log),'training_jsonl_sha256':worker.file_sha(log),
            'initial_trainable_sha256':'a'*64,'final_trainable_sha256':'f'*64,'matched_Base':base,
            'T2_protocol_sha256':'e'*64,'G3_protocol_sha256':'c'*64,'EMA_updates':worker.UPDATES if arm=='A2' else 0,
            'foreach':False,'frozen_identity_versions_unchanged':True,'mode_gradient_feature_cleanup_passed':True,'memory':memory()}
        write(arm_run/'summary.json',report)
        write(arm_run/'status.json',{'status':'completed','variant':arm,'updates':worker.UPDATES,
            'scientific_training_updates':worker.UPDATES,'summary_sha256':worker.file_sha(arm_run/'summary.json'),
            'final_checkpoint':cps[-1],'protocol_sha256':psha})
        reports[arm]=report
    counter=worker.PilotCounter();counter.by_role={arm:{'attempted':2048,'applied':2048,'returned_steps':2048} for arm in worker.ROLES}
    summary={'status':'completed_pilot_training','protocol_sha256':psha,'completed_arms':list(worker.ROLES),
        'arms':reports,'budget':worker.BUDGET,**counter.snapshot(),'training_seeds':[20261102],
        'comparison_axis':'method arm, not training seed','quality_inference':False,'resident_VAE_unchanged_FP32':True,'G4_evaluated':False}
    write(directory/'summary.json',summary)
    write(directory/'status.json',{'status':'completed_pilot_training','protocol_sha256':psha,
        'summary_sha256':worker.file_sha(directory/'summary.json'),'completed_arms':list(worker.ROLES),'G4_evaluated':False,**counter.snapshot()})
    return directory,protocol


def reseal_pilot(directory,arm=None):
    summary=worker.read(directory/'summary.json')
    if arm is not None:
        arm_run=directory/'arm_runs'/arm;report=worker.read(arm_run/'summary.json')
        summary['arms'][arm]=report
        state=worker.read(arm_run/'status.json');state['summary_sha256']=worker.file_sha(arm_run/'summary.json')
        write(arm_run/'status.json',state)
    write(directory/'summary.json',summary)
    state=worker.read(directory/'status.json');state['summary_sha256']=worker.file_sha(directory/'summary.json')
    write(directory/'status.json',state)


class ImportAndConfigTests(unittest.TestCase):
    def test_import_help_metadata_only_and_budget(self):
        code=f"import sys;sys.dont_write_bytecode=True;sys.path.insert(0,{str(ROOT/'scripts')!r});import stage4_pilot;assert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/stage4_pilot.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(worker.BUDGET['formal_pilot_updates'],3*2048)
        self.assertEqual(worker.BUDGET['run_equivalents'],.75)
        self.assertEqual(worker.BUDGET['warmup_updates'],0)
        self.assertEqual(worker.CHECKPOINT_AT,(512,1024,1536,2048))

    def test_per_arm_config_keeps_Base_architecture_and_explicit_true_arm(self):
        from vsr.checkpoints import _check_config
        base=base_config();before=copy.deepcopy(base)
        for arm in worker.ROLES:
            cfg=worker.method_config(base,arm)
            self.assertEqual(cfg['training']['variants'],[arm]);self.assertEqual(cfg['training']['updates'],2048)
            self.assertEqual(cfg['training']['checkpoint_at'],[512,1024,1536,2048])
            self.assertEqual(cfg['training']['evaluate_at'],[])
            self.assertNotIn('formal_training',cfg);self.assertNotIn('stage2_contract',cfg)
            self.assertEqual(cfg['model'],base['model']);self.assertEqual(cfg['conditioning'],base['conditioning'])
            self.assertEqual(cfg['data'],base['data']);self.assertEqual(cfg['evaluation'],base['evaluation'])
            self.assertEqual(cfg['restoration_architecture'],'noref_base_v1');self.assertNotIn('_source_config_sha256',cfg)
            self.assertEqual(_check_config(cfg),cfg['_sha256'])
            self.assertEqual(cfg['self_distillation']['corruption_enabled'],arm!='flow_only')
            self.assertEqual(cfg['self_distillation']['alignment_enabled'],arm=='A2')
        self.assertEqual(base,before)

    def test_formal_counter_does_not_reuse_diagnostic_zero(self):
        counter=worker.PilotCounter();optimizer=SimpleNamespace(step=lambda:None)
        counter.attach(optimizer,'A2');optimizer.step()
        report=counter.snapshot()
        self.assertEqual(report['scientific_training_updates'],1)
        self.assertEqual(report['pilot_optimizer_updates_applied'],1)
        self.assertEqual(report['completed_step_returns'],0)
        self.assertNotIn('diagnostic_optimizer_updates_applied',report)

    def test_prepare_fresh_per_arm_configs_and_no_placeholder_criteria(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);args=SimpleNamespace(project_root=root,g3_run=root/'runs/G3',run_dir=root/'runs/pilot')
            protocol={'method_configs':{arm:worker.method_config(base_config(),arm) for arm in worker.ROLES},
                'quality_evaluation':{'quality_evaluation_blocked':True,'criteria':None}}
            with patch.object(worker,'collect_protocol',return_value=protocol):worker.prepare(args)
            self.assertEqual(worker.read(args.run_dir/'status.json')['scientific_training_updates'],0)
            self.assertTrue(worker.read(args.run_dir/'status.json')['quality_evaluation_blocked'])
            for arm in worker.ROLES:self.assertEqual(worker.read(args.run_dir/'arm_runs'/arm/'config.json'),protocol['method_configs'][arm])
            self.assertEqual(list(args.run_dir.rglob('*criteria*')),[])
            with self.assertRaisesRegex(ValueError,'Fresh-only'):worker.prepare(args)


class G3PrerequisiteTests(unittest.TestCase):
    def test_valid66_rows_plus4_semantic_counter_and_real_T2_delegation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory,protocol=g3_fixture(root)
            with patch.object(worker.g3,'collect_protocol',return_value=protocol) as validate:
                result,evidence=worker.completed_g3(directory,root)
            self.assertEqual(result,protocol);self.assertEqual(evidence['benchmark_rows'],66)
            self.assertEqual(evidence['semantic_updates'],4);self.assertEqual(evidence['total_diagnostic_updates'],70)
            self.assertEqual(validate.call_args.args[0].evaluation_run,root/'runs/T2')

    def test_resealed_missing_row_counts_semantics_memory_or_initial_mismatch_rejected(self):
        for kind in ('row','count','semantics','row_memory','role_memory','semantic_memory','initial','cleanup','source'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);directory,protocol=g3_fixture(root)
                if kind in ('row','count','row_memory'):
                    item=worker.read(directory/'observations.json')
                    if kind=='row':item['rows'].pop()
                    elif kind=='count':item['by_role']['A2']['returned_steps']=21
                    else:item['rows'][-1]['memory']['peak_reserved_bytes']=17*1024**3
                    write(directory/'observations.json',item)
                elif kind in ('semantics','semantic_memory'):
                    item=worker.read(directory/'semantics.json')
                    if kind=='semantics':item['checks']['semantic_A2_rho0']['exact_trajectory_equal']=False
                    else:item['checks']['semantic_A2_rho0']['memory']['peak_allocated_bytes']=17*1024**3
                    write(directory/'semantics.json',item)
                elif kind in ('role_memory','initial'):
                    item=worker.read(directory/'summary.json')
                    if kind=='initial':item['roles'][-1]['trainable_sha256_before']='c'*64
                    else:item['roles'][-1]['memory']['peak_reserved_bytes']=17*1024**3
                    write(directory/'summary.json',item)
                elif kind=='cleanup':write(directory/'cleanup_failure.json',{'failure':'never accepted'})
                reseal_g3(directory)
                with patch.object(worker.g3,'collect_protocol',return_value=protocol,
                    side_effect=ValueError('underlying T2 incomplete') if kind=='source' else None):
                    with self.assertRaises(ValueError):worker.completed_g3(directory,root)

    def test_70_raw_rows_and_stale_SHA_never_masquerade_as_valid_G3(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory,protocol=g3_fixture(root)
            item=worker.read(directory/'observations.json');item['rows']+=copy.deepcopy(item['rows'][:4])
            write(directory/'observations.json',item);reseal_g3(directory)
            with patch.object(worker.g3,'collect_protocol',return_value=protocol):
                with self.assertRaises(ValueError):worker.completed_g3(directory,root)


class LazyLatentTests(unittest.TestCase):
    def test_training_prepare_binds_fixed_order_metadata_without_loading_tensors(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary,patch.object(worker,'TRAIN_CLIPS',3):
            root=Path(temporary);inputs=latent_fixture(root);source=Path(inputs['contract']['producer_run']);cfg=base_config()
            artifacts={}
            for spec in inputs['clips']:
                meta=worker.read(spec['metadata']['path']);meta['binding']['official_revision']=cfg['model']['revision']
                meta['binding']['config_sha256']=cfg['_sha256'];write(Path(spec['metadata']['path']),meta)
                payload=torch.load(spec['latents']['path'],map_location='cpu',weights_only=True)
                payload['binding']=meta['binding'];torch.save(payload,spec['latents']['path'])
                index=spec['index']
                artifacts[f'clip_{index}/metadata.json']=worker.file_sha(spec['metadata']['path'])
                artifacts[f'clip_{index}/fresh_latents.pt']=worker.file_sha(spec['latents']['path'])
            manifest=root/'manifest.json';write(manifest,{'records':[spec['record'] for spec in inputs['clips']]})
            cohort=root/'cohort.json';write(cohort,{'manifest':str(manifest),'manifest_file_sha256':worker.file_sha(manifest)})
            seal=source/'cache_asset_seal.json';write(seal,{'artifact_sha256':artifacts})
            identifiers=[spec['record']['id'] for spec in inputs['clips']]
            protocol={'matched_Base':{'config':cfg},'evaluation_protocol':{'cohort_seal':{'path':str(cohort)},
                'caption':{'text':inputs['contract']['caption']},'cache_producer':{'source_run':str(source),
                    'seal':{'path':str(seal)},'encoder_hashes':inputs['contract']['encoder_hashes'],
                    'selection':{'training_ids':identifiers,'validation_ids':[f'val{i}' for i in range(32)],
                                 'test_accessed':False,'manifest_sha256':inputs['contract']['manifest_sha256']}}}}
            with patch.object(worker.g3,'ACTUAL_RUNS',root),patch.dict(sys.modules,{'torch':None}):
                result=worker.training_inputs(protocol)
            self.assertEqual([spec['record']['id'] for spec in result['clips']],identifiers)
            self.assertEqual(result['contract']['training_ids'],identifiers)
            self.assertFalse(result['tensor_payloads_loaded_by_prepare']);self.assertEqual(result['contract']['max_cached'],2)

    def test_LRU_at_most2_weights_only_rehash_and_no_source_video_needed(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary,patch.object(worker,'TRAIN_CLIPS',3):
            inputs=latent_fixture(Path(temporary));reader=worker.LazyLatentReader(inputs)
            original=torch.load
            with patch('torch.load',wraps=original) as load:
                reader[0];reader[1];reader[0];reader[2]
                self.assertEqual(load.call_count,3)
                self.assertTrue(all(call.kwargs=={'map_location':'cpu','weights_only':True} for call in load.call_args_list))
            self.assertEqual(reader.cached_indices,(0,2));reader.clear();self.assertEqual(reader.cached_indices,())
            self.assertFalse(torch.cuda.is_initialized())

    def test_cached_hit_still_rejects_changed_file_or_cached_nonfinite_value(self):
        import torch
        for kind in ('file','tensor'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary,patch.object(worker,'TRAIN_CLIPS',3):
                inputs=latent_fixture(Path(temporary));reader=worker.LazyLatentReader(inputs);pair=reader[0]
                if kind=='file':Path(inputs['clips'][0]['latents']['path']).write_bytes(b'changed')
                else:pair['clean'].view(-1)[-1]=float('nan')
                with self.assertRaises(ValueError):reader[0]

    def test_resealed_metadata_source_geometry_FPS_and_cache_key_fail_before_tensor_use(self):
        changes=[lambda m:m.update(source_id='wrong'),lambda m:m.update(source_sha256='a'*64),
            lambda m:m.update(split='val'),lambda m:m.update(temporal_stride=4),
            lambda m:m.update(crop_tlhw=[0,0,64,224]),lambda m:m['degradation'].update(scale=2),
            lambda m:m.update(output_fps=12.),lambda m:m['binding'].update(data_cache_key='wrong')]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary,patch.object(worker,'TRAIN_CLIPS',3):
                inputs=latent_fixture(Path(temporary));spec=inputs['clips'][0]
                meta=worker.read(spec['metadata']['path']);change(meta);write(Path(spec['metadata']['path']),meta)
                spec['metadata']=worker.binding(spec['metadata']['path']);spec['binding_sha256']=worker.digest(meta['binding'])
                with self.assertRaises(ValueError):worker.LazyLatentReader(inputs)[0]

    def test_wrong_shape_dtype_or_payloadbinding_rejected(self):
        import torch
        for kind in ('shape','dtype','binding'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary,patch.object(worker,'TRAIN_CLIPS',3):
                inputs=latent_fixture(Path(temporary));spec=inputs['clips'][0]
                payload=torch.load(spec['latents']['path'],weights_only=True)
                if kind=='shape':payload['clean']=payload['clean'][:,:,:8]
                elif kind=='dtype':payload['clean']=payload['clean'].to(torch.bfloat16)
                else:payload['binding']['data_cache_key']='wrong'
                torch.save(payload,spec['latents']['path']);spec['latents']=worker.binding(spec['latents']['path'])
                with self.assertRaises(ValueError):worker.LazyLatentReader(inputs)[0]


class PilotLifecycleTests(unittest.TestCase):
    def test_failed_prerequisite_and_secondary_cleanup_preserve_primary_and_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);actual=root/'actual/runs';actual.mkdir(parents=True)
            args=SimpleNamespace(project_root=root,g3_run=root/'runs/G3',run_dir=root/'runs/pilot',expected_protocol_sha=None)
            protocol={'method_configs':{arm:{} for arm in worker.ROLES},'quality_evaluation':{'quality_evaluation_blocked':True}}
            with patch.object(worker,'collect_protocol',return_value=protocol):worker.prepare(args)
            args.expected_protocol_sha=worker.file_sha(args.run_dir/'protocol.json')
            held=[]
            def cleanup():
                held.append(competing_worker_lock_probe(actual));raise RuntimeError('secondary cleanup fixture')
            with patch.object(worker.g3,'ACTUAL_RUNS',actual),patch.object(worker,'collect_protocol',side_effect=RuntimeError('primary prerequisite fixture')),\
                 patch.object(worker.gc,'collect',side_effect=cleanup):
                with self.assertRaisesRegex(RuntimeError,'primary prerequisite fixture'):worker.execute(args)
            self.assertEqual(held,[73]);self.assertEqual(competing_worker_lock_probe(actual),0)
            failure=worker.read(args.run_dir/'failure.json');status=worker.read(args.run_dir/'status.json')
            self.assertEqual(failure['scientific_training_updates'],0);self.assertFalse(failure['G4_evaluated'])
            self.assertEqual(status['status'],'failed_stop_no_retry')
            self.assertEqual(worker.read(args.run_dir/'cleanup_failure.json')['primary_error_preserved'],'RuntimeError: primary prerequisite fixture')
            with self.assertRaisesRegex(ValueError,'never-started'):worker.execute(args)


class CompletedPilotTests(unittest.TestCase):
    def test_reads_all6144_rows_three_arms_and_staging_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory,protocol=pilot_fixture(root)
            with patch.object(worker,'collect_protocol',return_value=protocol):
                result,evidence=worker.completed_pilot(directory,root)
            self.assertEqual(result,protocol);self.assertEqual(evidence['scientific_training_updates'],6144)
            self.assertEqual(evidence['training_seeds'],[20261102]);self.assertEqual(set(evidence['arms']),set(worker.ROLES))
            for arm in worker.ROLES:
                self.assertEqual(evidence['arms'][arm]['final_checkpoint']['step'],2048)
                self.assertTrue(Path(evidence['arms'][arm]['final_checkpoint']['path']).is_relative_to(root/'runs'))

    def test_changed_checkpoint_missing_row_and_resealed_RNG_mismatch_fail(self):
        for kind in ('checkpoint','missing_row','RNG','science_zero','false_initial'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);directory,protocol=pilot_fixture(root);arm='A2'
                arm_run=directory/'arm_runs'/arm
                if kind=='checkpoint':
                    (arm_run/arm/'checkpoint_002048.pt').write_bytes(b'changed checkpoint bytes')
                elif kind in ('missing_row','RNG'):
                    path=arm_run/arm/'training.jsonl';lines=path.read_text().splitlines()
                    if kind=='missing_row':lines.pop()
                    else:
                        row=json.loads(lines[-1]);row['matched_FM']['noise']='unmatched';lines[-1]=json.dumps(row)
                    path.write_text('\n'.join(lines)+'\n')
                    report=worker.read(arm_run/'summary.json');report['training_jsonl']=worker.binding(path)
                    report['training_jsonl_sha256']=worker.file_sha(path);write(arm_run/'summary.json',report)
                    reseal_pilot(directory,arm)
                elif kind=='science_zero':
                    state=worker.read(directory/'status.json');state['scientific_training_updates']=0;write(directory/'status.json',state)
                else:
                    report=worker.read(arm_run/'summary.json');report['initial_trainable_sha256']='c'*64
                    write(arm_run/'summary.json',report);reseal_pilot(directory,arm)
                with patch.object(worker,'collect_protocol',return_value=protocol):
                    with self.assertRaises(ValueError):worker.completed_pilot(directory,root)

    def test_feature_report_must_match_collection_flag_and_own_arm_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);target=root/'arm/A2';path=target/'feature_diagnostics/step_000512.json'
            write(path,{'CPU_fixture':True})
            row={'step':512,'feature_diagnostics_snapshot_collected':True,'feature_diagnostics':worker.binding(path)}
            worker._feature_report(row,target)
            row['feature_diagnostics_snapshot_collected']=False
            with self.assertRaises(ValueError):worker._feature_report(row,target)
            row['feature_diagnostics_snapshot_collected']=True
            outside=root/'other.json';write(outside,{'external':'not allowed as placeholder'})
            row['feature_diagnostics']=worker.binding(outside)
            with self.assertRaises(ValueError):worker._feature_report(row,target)

    def test_memory_zero_or_reserved_less_than_allocated_is_not_evidence(self):
        for item in ({'peak_allocated_bytes':0,'peak_reserved_bytes':0,'physical_device_bytes':16*1024**3},
                     {'peak_allocated_bytes':2048,'peak_reserved_bytes':1024,'physical_device_bytes':16*1024**3}):
            with self.assertRaises(ValueError):worker._memory_record(item)


if __name__=='__main__':unittest.main()
