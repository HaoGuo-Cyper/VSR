"""CPU-only metadata/lifecycle fixtures; no torch, checkpoints or real pixels."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
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
import stage2_evaluate_bases as worker


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


@contextmanager
def cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:yield
    finally:os.chdir(previous)


class Shape:
    def __init__(self,shape):self.shape=shape


class FakeVAE:
    def __init__(self):self.clear_cache()
    def encode(self,value):return value
    def clear_cache(self):self._feat_map=[None]*2;self._enc_feat_map=[None]*3


class FakePolicy:
    def __init__(self,vae):
        self.vae,self.ledger,self.dropped_records,self.removed=vae,[],0,False
        def observed(value):
            self.ledger.append({'route':'original_delegate','input_shape':list(value.shape),'encoder_hooks_installed':False})
            return value
        self.wrapper=observed
        vae.encode=observed
        vae._vsr_t1_cache_policy=self
    def remove(self):
        if self.removed:return
        if self.vae.__dict__.get('encode') is self.wrapper:del self.vae.encode
        if getattr(self.vae,'_vsr_t1_cache_policy',None) is self:del self.vae._vsr_t1_cache_policy
        self.removed=True


def fake_install(vae,*,max_records):
    assert max_records==2
    return FakePolicy(vae)


def ledger(rows,enabled,tiles):
    worker.need(enabled is False and tiles==1 and len(rows)==1,'Expected one original delegate')
    worker.need(rows[0]['route']=='original_delegate' and rows[0]['encoder_hooks_installed'] is False,'Numerical encode path changed')


class FakeRestore:
    def __init__(self,*,bad_shape=False,leak_cache=False):
        self.calls=[];self.bad_shape=bad_shape;self.leak_cache=leak_cache
    def __call__(self,model,vae,lr,text,config,**options):
        self.calls.append(options.copy())
        vae.encode(Shape([1,3,33,128,224]))
        if self.leak_cache:vae._feat_map=[object()]
        shape=[1,3,9 if self.bad_shape else 33,128,224]
        return Shape(shape),{'output_shape':shape,'tile_count':1,'tiles':[{}],
            'reference':{'enabled':False,'encode_count':0},
            'sampling':{'steps':20,'seed':options['seed'],'cfg':1.0,'global_noise_shape':[1,16,9,16,28],
                        'noise_device':'cpu','sampler':'local official-config UniPC'},
            'temporal_windowing':False,'padding':{'time_end':0},'inference_inputs':['LRV','native text embedding']}


class Stage2EvaluationMetadataTest(unittest.TestCase):
    def test_import_is_metadata_only(self):
        code=f"import sys;sys.dont_write_bytecode=True;sys.path.insert(0,{str(ROOT/'scripts')!r});import stage2_evaluate_bases;assert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_complete_matrix_preserves_all_pairings(self):
        seal={'cohort':[{'record':{'id':f'r{i}'}} for i in range(32)]}
        rows=worker.ordered_cases(seal)
        self.assertEqual(len(rows),1152)
        self.assertEqual(len(set(rows)),1152)
        for seed in worker.SEEDS:
            self.assertEqual(sum(row[0]==seed for row in rows),384)
            for record in range(32):
                for condition in worker.CONDITIONS:
                    self.assertEqual([r[3] for r in rows if r[:3]==(seed,f'r{record}',condition)],[42,43])

    def test_checkpoint_parser_rejects_selection_or_relative_paths(self):
        with self.assertRaises(ValueError):worker.parse_checkpoints(['20261102=/tmp/a'])
        with self.assertRaises(ValueError):worker.parse_checkpoints(['20261102=relative','20261103=/tmp/b','20261104=/tmp/c'])
        with self.assertRaises(ValueError):worker.parse_checkpoints(['20261102=/tmp/a','20261102=/tmp/b','20261104=/tmp/c'])

    def test_new_output_excludes_old_root_and_input_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'staging';assets=root/'runs/assets'
            self.assertEqual(worker.safe_output(root/'runs/new',root,assets),root/'runs/new')
            for output in (root/'outside',assets/'new'):
                with self.assertRaises(ValueError):worker.safe_output(output,root,assets)
        with self.assertRaises(ValueError):worker.safe_output(worker.OLD/'runs/new',worker.OLD,worker.OLD/'runs/assets')

    def make_checkpoints(self,root):
        # The real queue implementation derives all three effective configs
        # from copies of the actual sealed templates. No torch is imported.
        queue=worker.import_helper(ROOT/'vsr/stage2_base_queue.py','queue_effective_contract_fixture')
        allowed=root/'actual/runs';producer=allowed/'seed20261102'
        manifest=root/'manifest.json'
        write(manifest,{'records':[{'id':f't{i}','split':'train'} for i in range(1641)]})
        cohort=[{'record':{'id':f'r{i}'}} for i in range(32)]
        seal={'base_configs':{},'cohort':cohort,'manifest':str(manifest),'manifest_digest':'manifest-fixture'}
        selection={'manifest_sha256':seal['manifest_digest'],
            'training_ids':sorted((f't{i}' for i in range(1641)),key=lambda x:hashlib.sha256(f'train:20261102:{x}'.encode()).hexdigest()),
            'validation_ids':[f'r{i}' for i in range(32)],'test_accessed':False,
            'caption_policy':worker.read(ROOT/'configs/stage2_20261105/base_seed20261102.json')['caption_policy']}
        geometry={'latent_grid':[9,16,28],'token_grid':[9,8,14]}
        encoders={name:worker.file_sha(ROOT/name) for name in ('vsr/data.py','vsr/preflight.py','vsr/stage2_preparation.py')}
        paths={}
        for seed in worker.SEEDS:
            raw=worker.read(ROOT/f'configs/stage2_20261105/base_seed{seed}.json')
            raw['stage2_contract']['allowed_root']=str(allowed)
            source=root/f'configs/base_seed{seed}.json';write(source,raw)
            seal['base_configs'][str(seed)]={'path':str(source),'sha256':worker.file_sha(source)}
            with patch.object(queue,'ROOT',root/'actual'):
                effective=queue.effective_config(source,cache_from=None if seed==worker.SEEDS[0] else producer)
            self.assertEqual(worker.expected_effective_config(source,cache_from=None if seed==worker.SEEDS[0] else producer),effective)
            run=allowed/f'seed{seed}';path=run/'Base/checkpoint_008192.pt'
            path.parent.mkdir(parents=True);path.write_bytes(b'fixture-not-a-tensor')
            paths[seed]=path
            write(run/'config.json',effective)
            write(run/'status.json',{'status':'completed','seed':seed,'profile':False,'test_accessed':False,'run':str(run)})
            write(path.parent/'summary.json',{'status':'completed','updates':8192})
            write(path.parent/'initialization.json',{'fresh_official_initialization':True,'legacy_state_loaded':False,
                'training_seed':seed,'data_seed':20261102,'official_revision':raw['model']['revision']})
            write(run/'selection.json',selection);write(run/'correspondence.json',geometry)
            write(run/'environment.json',{'code_sha256':encoders})
        required={'config.json','selection.json','correspondence.json','fresh_training_cache.pt','fresh_text_embeddings.pt'}
        required.update(f'clip_{i}/{name}' for i in range(1673) for name in ('metadata.json','paired_rgb.pt','fresh_latents.pt'))
        artifacts={name:'a'*64 for name in required}
        artifacts.update({name:worker.file_sha(producer/name) for name in ('config.json','selection.json','correspondence.json')})
        artifacts['fresh_text_embeddings.pt']=worker.read(producer/'config.json')['data']['text_cache']['sha256']
        write(producer/'cache_asset_seal.json',{'source_run':str(producer),'artifact_sha256':artifacts,'contains_task_initialization':False})
        for seed in worker.SEEDS[1:]:
            config=worker.read(paths[seed].parent.parent/'config.json')
            write(paths[seed].parent.parent/'cache_reuse.json',{'status':'verified','source_run':str(producer),'clip_count':1673,
                'training_clip_count':1641,'validation_clip_count':32,'manifest_sha256':seal['manifest_digest'],
                'official_revision':config['model']['revision'],'encoder_source_hashes':encoders,'legacy_state_read':False,
                'temporal_stride':1,'processing_data_version':'hq-vsr-paired-v2','tensor_loading':'lazy LRU<=2; weights_only=True CPU',
                'manifest_records_checked':True,'source_asset_seal_sha256':worker.file_sha(producer/'cache_asset_seal.json'),
                'source_encoder_hashes':encoders,'task_checkpoint_loaded':False,'T5_loaded':False,'DINO_loaded':False})
        return paths,allowed,seal,producer

    def test_checkpoint_is_exact_new_final_Base_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths,allowed,seal,producer=self.make_checkpoints(Path(temporary))
            provenance=worker.cache_producer_spec(producer,allowed,seal)
            self.assertFalse(provenance['payloads_rehashed_or_loaded_by_evaluation_prepare'])
            for seed,path in paths.items():
                spec=worker.checkpoint_spec(path,seed,allowed,seal,expected_cache_source=producer)
                self.assertEqual(spec['training_seed'],seed)
                self.assertEqual(spec['checkpoint']['sha256'],worker.file_sha(path))
            path=paths[20261102]
            with self.assertRaises(ValueError):worker.checkpoint_spec(path,20261102,worker.OLD/'runs',seal,expected_cache_source=producer)
            cfg=worker.read(path.parent.parent/'config.json');cfg['data']['num_frames']=9
            write(path.parent.parent/'config.json',cfg)
            with self.assertRaises(ValueError):worker.checkpoint_spec(path,20261102,allowed,seal,expected_cache_source=producer)

    def test_incomplete_training_or_different_cohort_fails_before_tensor_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths,allowed,seal,producer=self.make_checkpoints(Path(temporary));path=paths[20261102]
            write(path.parent/'summary.json',{'status':'completed','updates':8191})
            with self.assertRaises(ValueError):worker.checkpoint_spec(path,20261102,allowed,seal,expected_cache_source=producer)
            write(path.parent/'summary.json',{'status':'completed','updates':8192})
            write(path.parent.parent/'selection.json',{'test_accessed':True,'validation_ids':[f'r{i}' for i in range(32)]})
            with self.assertRaises(ValueError):worker.checkpoint_spec(path,20261102,allowed,seal,expected_cache_source=producer)

    def test_runtime_config_cache_source_and_reuse_cannot_be_broadly_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths,allowed,seal,producer=self.make_checkpoints(Path(temporary));path=paths[20261103]
            config_path=path.parent.parent/'config.json';original=worker.read(config_path)
            mutations=[{**original,'_sha256':seal['base_configs']['20261103']['sha256']},
                       {**original,'arbitrary_runtime_override':True},
                       {**original,'data':{**original['data'],'cache_from':str(allowed/'other')}}]
            for value in mutations:
                write(config_path,value)
                with self.assertRaises(ValueError):worker.checkpoint_spec(path,20261103,allowed,seal,expected_cache_source=producer)
            write(config_path,original)
            reuse_path=path.parent.parent/'cache_reuse.json';reuse=worker.read(reuse_path)
            for key,value in [('source_run',str(allowed/'other')),('source_asset_seal_sha256','b'*64),('T5_loaded',True)]:
                write(reuse_path,{**reuse,key:value})
                with self.assertRaises(ValueError):worker.checkpoint_spec(path,20261103,allowed,seal,expected_cache_source=producer)
            write(reuse_path,reuse)
            status=worker.read(producer/'status.json');write(producer/'status.json',{**status,'status':'failed'})
            with self.assertRaises(ValueError):worker.checkpoint_spec(path,20261103,allowed,seal,expected_cache_source=producer)
            write(producer/'status.json',status)
            sealpath=producer/'cache_asset_seal.json';cache_seal=worker.read(sealpath)
            del cache_seal['artifact_sha256']['clip_1672/fresh_latents.pt'];write(sealpath,cache_seal)
            with self.assertRaises(ValueError):worker.cache_producer_spec(producer,allowed,seal)

    def test_protocol_json_roundtrip_and_prepare_are_metadata_only(self):
        # Real CLI-like clean process: other discovered suites may import torch
        # in this parent. Production's strict no-torch prepare guard stays intact.
        code=f"import importlib.util,sys;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location('metadata_fixture',{str(Path(__file__).resolve())!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.Stage2EvaluationMetadataTest()._metadata_prepare_fixture();assert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def _metadata_prepare_fixture(self):
        # Isolate asset/checkpoint/model-weight byte bindings from this metadata
        # fixture; all production collect functions remain strict in real runs.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);assetroot=root/'runs/assets';assetroot.mkdir(parents=True)
            for name in ('vsr/a.py','scripts/stage2_framewise.py','scripts/stage2_metrics.py'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
            cohort=[{'record':{'id':f'r{i}'}} for i in range(32)]
            seal={'project_root':str(root),'seal_digest':'fixture','config':{'caption':'A high quality video.','G2':{}},'cohort':cohort}
            sealpath=root/'seal.json';write(sealpath,seal)
            assets=[]
            for i in range(32):
                specs=[]
                for j in range(7):
                    path=assetroot/f'{i}_{j}.npy';path.write_bytes(b'not-decoded-by-prepare')
                    specs.append({'path':str(path),'sha256':worker.file_sha(path)})
                assets.append({'record_id':f'r{i}','HR':specs[0],'clean':specs[1],
                    'conditions':{c:{'asset':specs[j+2]} for j,c in enumerate(worker.CONDITIONS[1:])}})
            dataset=assetroot/'dataset.json';write(dataset,{'status':'completed_CPU_assets_no_predictions',
                'cohort_seal_digest':'fixture','config':seal['config'],'assets':assets})
            cache=root/'text.pt';cache.write_bytes(b'not-loaded-by-prepare')
            config={'model':{'revision':'fixture'},'data':{'text_cache':{'path':str(cache),'sha256':'cache-sha'}}}
            caption=root/'scripts/frozen_external/outputs/VSR_final_evaluation_text_cache_preflight.json'
            write(caption,{'entries':{'NoRef_Base_seed20261102_step8192':{'cache_path':str(cache),
                'cache_file_sha256':'cache-sha','shape':[1,512,4096],'dtype':'torch.bfloat16','embedding_sha256':'tensor-sha'}}})
            args=SimpleNamespace(project_root=root,seal=sealpath,dataset=dataset,allowed_checkpoint_root=root/'actual/runs',
                checkpoint=[f'{s}={root}/actual/runs/{s}/Base/checkpoint_008192.pt' for s in worker.SEEDS],
                official_manifest=root/'official.json',metric_root=root/'weights',dependency_root=root/'deps',
                desktop_pid=4816,desktop_name='gjs',run_dir=root/'runs/evaluation',expected_protocol_sha=None)
            def bound(path,expected=None):return {'path':str(Path(path).resolve()),'bytes':1,'sha256':expected or 'fixture-sha'}
            with cwd(root),patch.object(worker,'validate_seal'),patch.object(worker,'binding',side_effect=bound),\
                 patch.object(worker,'checkpoint_spec',side_effect=lambda path,seed,*a,**kw:{'training_seed':seed,'config':config}),\
                 patch.object(worker,'cache_producer_spec',return_value={'fixture':'producer'}),\
                 patch.object(worker,'desktop_binding',return_value={'fixture':'desktop'}),\
                 patch.object(worker,'dependency_spec',return_value={'root':str(root/'deps'),'packages':{'lpips':{'version':'fixture'}}}),\
                 patch.object(worker,'native_source_bindings',return_value={name:{'fixture':'source'} for name in worker.NATIVE_SOURCES}),\
                 patch.object(worker,'official_bindings',return_value=({'fixture':'tuple-normalization-regression'},)),\
                 patch.object(worker.importlib.metadata,'version',return_value='fixture'),\
                 patch.object(worker.importlib.metadata,'distribution',return_value=SimpleNamespace(locate_file=lambda x:root/'fake_vae.py')):
                result=worker.collect_protocol(args)
                self.assertEqual(json.loads(json.dumps(result)),result)
                self.assertIsInstance(result['official_files'],list)
                worker.prepare(args)
                self.assertEqual(worker.read(args.run_dir/'status.json')['status'],'planned')
                self.assertFalse(worker.read(args.run_dir/'status.json')['torch_imported'])
                with self.assertRaises(ValueError):worker.prepare(args)


class Stage2DependencyAndGPUGuardTest(unittest.TestCase):
    @staticmethod
    def process_info(pid=4816):
        uid=os.getuid()
        return {'pid':pid,'proc_directory_uid':uid,'exe':'/usr/bin/gjs-console','uid_fields':[uid]*4,
            'cgroups':[{'hierarchy':'0','controllers':'','path':f'/user.slice/user-{uid}.slice/user@{uid}.service/session.slice/org.gnome.Shell@ubuntu.service'}],
            'executable_identity':{'same_system_inode':True,'system_uid':0,'system_mode':0o755,'system_regular_file':True},'read_error':None}

    def test_only_pinned_verified_small_desktop_is_allowed(self):
        classifier=worker.import_helper(ROOT/'scripts/frozen_external/work/gpu_process_classification.py','gpu_classifier_fixture')
        info=self.process_info()
        desktop={'UID':os.getuid(),'PID':4816,'reported_name':'gjs','start_ticks':123,'process_metadata':info,'memory_limit_MiB':256}
        def check(stdout,*,metadata=info,start=123,code=0):
            runner=lambda *a,**kw:subprocess.CompletedProcess(a[0],code,stdout,'complete stderr evidence')
            return worker.gpu_idle_guard(desktop,classifier,runner=runner,proc_reader=lambda pid:metadata,start_reader=lambda pid:start)
        evidence=check('4816, gjs, 64 MiB\n')
        self.assertTrue(evidence['passed']);self.assertEqual(evidence['stdout'],'4816, gjs, 64 MiB\n')
        self.assertEqual(evidence['stderr'],'complete stderr evidence');self.assertEqual(evidence['pinned_desktop_matches'],[True])
        self.assertFalse(check('')['passed'])
        for stdout in ('4816, gjs, 257 MiB\n','4816, gjs, N/A\n',
                       '4816, /usr/bin/gjs, 64 MiB\n','4816, python, 64 MiB\n','9999, gjs, 64 MiB\n',
                       '4816, gjs, 64 MiB\n9999, python, 8000 MiB\n'):
            self.assertFalse(check(stdout)['passed'],stdout)
        self.assertFalse(check('4816, gjs, 64 MiB\n',start=124)['passed'])
        self.assertFalse(check('',code=1)['passed'])
        for metadata in ({**info,'exe':'/usr/bin/python'}, {**info,'uid_fields':[0]*4},
                         {**info,'cgroups':[{'hierarchy':'0','controllers':'','path':'/wrong.service'}]},
                         {**info,'read_error':'process disappeared'}):
            self.assertFalse(check('4816, gjs, 64 MiB\n',metadata=metadata)['passed'])

    def test_three_native_numeric_sources_bind_without_import_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for relative in worker.NATIVE_SOURCES.values():
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# original numeric fixture\n')
            distribution=SimpleNamespace(locate_file=lambda path:root/path)
            imported_before=set(sys.modules)
            with patch.object(worker.importlib.metadata,'distribution',return_value=distribution):
                sources=worker.native_source_bindings(root)
            self.assertEqual(set(sys.modules)-imported_before,set())
            self.assertEqual(set(sources),set(worker.NATIVE_SOURCES))
            classes={name:object() for name in worker.NATIVE_SOURCES}
            def source_for(native):return next(sources[name]['path'] for name,value in classes.items() if value is native)
            with patch.object(worker.inspect,'getfile',side_effect=source_for):
                worker.verify_native_sources(sources,classes)
                for name in worker.NATIVE_SOURCES:
                    path=Path(sources[name]['path']);original=path.read_bytes();path.write_bytes(b'# changed\n')
                    with self.assertRaises(ValueError):worker.verify_native_sources(sources,classes)
                    path.write_bytes(original)
                with self.assertRaises(ValueError):worker.verify_native_sources(sources,{'AutoencoderKLWan':classes['AutoencoderKLWan']})

    def test_shared_gpu_lease_excludes_other_evaluation_and_actual_worker_entry(self):
        # Real OS flock competition in clean CPU subprocesses. The Base entry
        # is the unmodified actual worker main with a harmless fake callback.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);allowed=root/'runs';allowed.mkdir()
            marker=root/'executed'
            evaluation=f"import sys;sys.dont_write_bytecode=True;sys.path.insert(0,{str(ROOT/'scripts')!r});import stage2_evaluate_bases as w;from pathlib import Path;w.execute=lambda a:(w.need(a.desktop_name=='gjs','Default raw driver name differs'),Path({str(marker)!r}).write_text(str(a.run_dir)));sys.argv=['worker','execute','--seal','unused','--dataset','unused','--checkpoint','unused','--allowed-checkpoint-root',{str(allowed)!r},'--run-dir',sys.argv[1]];w.main();assert 'torch' not in sys.modules"
            base=f"import importlib.util,sys;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location('real_queue',{str(ROOT/'vsr/stage2_base_queue.py')!r});q=importlib.util.module_from_spec(s);s.loader.exec_module(q);from pathlib import Path;q.ROOT=Path({str(root)!r});q.worker=lambda a:Path({str(marker)!r}).write_text('base');sys.argv=['queue','worker','--config','unused','--result-file','unused'];q.main();assert 'torch' not in sys.modules"
            with worker.gpu_lease(allowed):
                for code,args in ((evaluation,['different_run_a']),(evaluation,['different_run_b']),(base,[])):
                    result=subprocess.run([sys.executable,'-c',code,*args],capture_output=True,text=True)
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertIn('BlockingIOError',result.stderr)
                    self.assertFalse(marker.exists())
            result=subprocess.run([sys.executable,'-c',evaluation,'new_run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(marker.read_text(),'new_run')
            result=subprocess.run([sys.executable,'-c',base],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(marker.read_text(),'base')
            with self.assertRaisesRegex(RuntimeError,'cleanup'):
                with worker.gpu_lease(allowed):raise RuntimeError('cleanup')
            with worker.gpu_lease(allowed):pass

    def test_explicit_dependency_tree_source_and_distribution_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'deps';root.mkdir()
            names={'lpips':'lpips','scikit-image':'skimage'}
            distributions={}
            for name,module in names.items():
                package=root/module;package.mkdir();(package/'__init__.py').write_text('# fixture\n')
                info=root/(name.replace('-','_')+'-1.0.dist-info');info.mkdir()
                for kind in ('METADATA','RECORD'):(info/kind).write_text(kind+' fixture\n')
                files=[(info/kind).relative_to(root) for kind in ('METADATA','RECORD')]
                distributions[name]=SimpleNamespace(version='1.0',files=files,locate_file=lambda path:root/path)
            with patch.object(worker,'PACKAGE_MODULES',names),patch.object(worker.importlib.metadata,'distribution',side_effect=distributions.get):
                spec=worker.dependency_spec(root)
                self.assertEqual(spec['root'],str(root));self.assertEqual(len(spec['local_tree']),6)
                for item in spec['packages'].values():self.assertTrue(Path(item['source']['path']).is_relative_to(root))
                imports={item['module']:SimpleNamespace(__file__=item['source']['path']) for item in spec['packages'].values()}
                with patch.object(worker.importlib,'import_module',side_effect=imports.get),patch.object(worker.importlib.metadata,'version',return_value='1.0'):
                    worker.verify_imported_dependencies(spec)
                    imports['lpips']=SimpleNamespace(__file__=__file__)
                    with self.assertRaises(ValueError):worker.verify_imported_dependencies(spec)
                (root/'lpips/__init__.py').write_text('# changed\n')
                self.assertNotEqual(worker.dependency_spec(root),spec)
                # No silent site-packages fallback when explicit metric source
                # is absent or resolves outside the declared dependency root.
                outside=SimpleNamespace(origin=str(ROOT/'tests/test_stage2_evaluate_bases.py'))
                with patch.object(worker.importlib.machinery.PathFinder,'find_spec',return_value=outside):
                    with self.assertRaises(ValueError):worker.dependency_spec(root)
            with self.assertRaises(ValueError):worker.dependency_spec(root/'missing')


class Stage2RestoreLifecycleTest(unittest.TestCase):
    def wrapper(self,restore):
        return worker.RotatingCropRestore(restore,fake_install,ledger,lambda x:True,lambda x:True)

    def test_same_original_delegate_rotates_and_finally_removes(self):
        restore=FakeRestore();wrapped=self.wrapper(restore);vae=FakeVAE();config={}
        wrapped.register(vae,config)
        for seed in (42,43):
            video,meta=wrapped(object(),vae,Shape([1,3,33,32,56]),object(),config,seed=seed)
            self.assertEqual(list(video.shape),[1,3,33,128,224])
            self.assertEqual(vae._vsr_t1_cache_policy.ledger,[])
            self.assertTrue(meta['stage2_policy_audit']['original_encode_restored_before_reinstall'])
        role=wrapped.finish(vae,expected=2)
        self.assertEqual(role['T1'],0);self.assertEqual(role['original_delegate'],2)
        self.assertTrue(wrapped.removed(vae));self.assertEqual(len(restore.calls),2)
        self.assertEqual([x['seed'] for x in restore.calls],[42,43])
        self.assertTrue(all(x['reference_image'] is None and x['reference_index'] is None for x in restore.calls))

    def test_wrong_shape_stops_without_retry_and_restores_encode(self):
        restore=FakeRestore(bad_shape=True);wrapped=self.wrapper(restore);vae=FakeVAE();wrapped.register(vae,{})
        with self.assertRaises(ValueError):wrapped(object(),vae,Shape([1,3,33,32,56]),object(),{},seed=42)
        self.assertEqual(len(restore.calls),1)
        self.assertEqual(wrapped.role['restores'],0)
        self.assertTrue(wrapped.removed(vae))
        self.assertTrue(wrapped.last_audit['original_encode_restored_after_failure'])

    def test_native_temporal_cache_leak_is_failure_not_new_encode(self):
        restore=FakeRestore(leak_cache=True);wrapped=self.wrapper(restore);vae=FakeVAE();wrapped.register(vae,{})
        with self.assertRaises(ValueError):wrapped(object(),vae,Shape([1,3,33,32,56]),object(),{},seed=42)
        self.assertEqual(len(restore.calls),1);self.assertTrue(wrapped.removed(vae))
        self.assertTrue(all(x is None for x in vae._feat_map))

    def test_no_partial_role_can_finish(self):
        wrapped=self.wrapper(FakeRestore());vae=FakeVAE();wrapped.register(vae,{})
        with self.assertRaises(ValueError):wrapped.finish(vae)
        vae._vsr_t1_cache_policy.remove()

    def test_original_full_metric_adapter_preserves_pairs_and_gap_direction(self):
        mask=[4<=i<12 for i in range(33)]
        per_frame=[{'psnr_rgb_db':30 if mask[i] else 40,'ssim_rgb':.9} for i in range(33)]
        full={'all_frames':{'per_frame':per_frame,'lpips_per_frame':[.3 if x else .1 for x in mask],
              'per_temporal_pair':[{'indices':[i,i+1],'tLP100_alex_v0_1_TecoGAN_crop':float(i)} for i in range(2,30)]}}
        score=worker.score_masked(full,mask)
        self.assertEqual(score['raw_harm_gap']['psnr_rgb_db'],10)
        self.assertAlmostEqual(score['raw_harm_gap']['lpips_alex_v0_1'],.2)
        self.assertEqual(score['temporal']['mask_touch_original_pairs'],[[i,i+1] for i in range(3,12)])
        self.assertEqual(score['temporal']['boundary_original_pairs'],[[3,4],[11,12]])


class Stage2AdoptedDesktopPolicyTest(unittest.TestCase):
    """Portable captured/fake inventory fixtures; no GPU query or live /proc."""
    GJS,NAUTILUS,PTYXIS=4816,1652088,1652090

    @classmethod
    def setUpClass(cls):
        path=ROOT/'scripts/frozen_external/work/gpu_process_classification.py'
        if hashlib.sha256(path.read_bytes()).hexdigest()!='0f98c8b717c0f0528042d2abcbb38db0b7798b44906707b9b83133f7bd7d42d2':
            raise AssertionError('These fixtures require the unchanged original classifier')
        cls.classifier=worker.import_helper(path,'stage2_adopted_desktop_classifier_fixture')

    @staticmethod
    def process(pid,exe,cgroup):
        uid=os.getuid()
        return {'pid':pid,'proc_directory_uid':uid,'exe':exe,'uid_fields':[uid]*4,
            'cgroups':[{'hierarchy':'0','controllers':'','path':cgroup}],
            'executable_identity':{'same_system_inode':True,'system_uid':0,'system_mode':0o755,'system_regular_file':True},
            'read_error':None}

    def setUp(self):
        prefix=f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service'
        self.info={
            self.GJS:self.process(self.GJS,'/usr/bin/gjs-console',prefix+'/session.slice/org.gnome.Shell@ubuntu.service'),
            self.NAUTILUS:self.process(self.NAUTILUS,'/usr/bin/nautilus',prefix+'/app.slice/app-dbus\\x2d:1.1\\x2dorg.gnome.Nautilus.slice/dbus-:1.1-org.gnome.Nautilus@8.service'),
            self.PTYXIS:self.process(self.PTYXIS,'/usr/bin/ptyxis',prefix+f'/app.slice/app-gnome-xdg\\x2dterminal\\x2dexec-{self.PTYXIS}.scope')}
        self.desktop={'PID':self.GJS,'reported_name':'gjs','UID':os.getuid(),'start_ticks':1771,
            'process_metadata':copy.deepcopy(self.info[self.GJS]),'memory_limit_MiB':256}

    def check(self,stdout,*,info=None,tick=1771,code=0):
        info=self.info if info is None else info
        calls=[]
        def runner(command,**kwargs):
            calls.append((command,kwargs))
            return subprocess.CompletedProcess(command,code,stdout,'captured stderr')
        result=worker.gpu_idle_guard(self.desktop,self.classifier,runner=runner,
            proc_reader=lambda pid:info.get(pid,{'pid':pid,'read_error':'fixture missing metadata'}),
            start_reader=lambda _:tick)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
        self.assertEqual(result['stdout'],stdout)
        self.assertEqual(result['stderr'],'captured stderr')
        return result

    def test_single_bound_gjs_is_still_allowed(self):
        result=self.check(f'{self.GJS}, gjs, 64 MiB\n')
        self.assertTrue(result['passed']);self.assertTrue(result['original_gjs_only_rule_would_pass'])
        self.assertEqual(result['pinned_gjs_match_count'],1)

    def test_verified_nautilus_and_ptyxis_are_only_additional_categories(self):
        for extra in (f'{self.NAUTILUS}, /usr/bin/nautilus, 72 MiB',f'{self.PTYXIS}, /usr/bin/ptyxis, 24 MiB',
                      f'{self.NAUTILUS}, /usr/bin/nautilus, 72 MiB\n{self.PTYXIS}, /usr/bin/ptyxis, 24 MiB'):
            with self.subTest(extra=extra):
                result=self.check(f'{self.GJS}, gjs, 64 MiB\n'+extra+'\n')
                self.assertTrue(result['classification']['all_inventory_processes_verified_allowed'])
                self.assertTrue(result['passed']);self.assertFalse(result['original_gjs_only_rule_would_pass'])

    def test_captured_failed_inventory_changes_outer_policy_only(self):
        # Captured raw row names/memory from the failed run, embedded for clean clones.
        stdout='4816, gjs, 64 MiB\n1652088, /usr/bin/nautilus, 72 MiB\n'
        result=self.check(stdout)
        original=self.classifier.classify_inventory(stdout.splitlines(),own_pid=None,own_worker_verified=False,
            expected_uid=os.getuid(),proc_reader=lambda pid:self.info[pid],query_success=True)
        self.assertEqual(result['classification'],original)
        self.assertTrue(original['all_inventory_processes_verified_allowed'])
        self.assertEqual(result['pinned_desktop_matches'],[True,False])
        self.assertFalse(result['original_gjs_only_rule_would_pass']);self.assertTrue(result['passed'])

    def test_unknown_python_unreadable_or_non_numeric_memory_is_rejected(self):
        python=1652111;info=copy.deepcopy(self.info)
        info[python]=self.process(python,'/home/fixture/env/bin/python3.12','/unverified_model.service')
        for extra in (f'{python}, python, 20 MiB','8888, unknown, 10 MiB',f'{self.NAUTILUS}, /usr/bin/nautilus, N/A'):
            with self.subTest(extra=extra):
                self.assertFalse(self.check(f'{self.GJS}, gjs, 64 MiB\n'+extra+'\n',info=info)['passed'])

    def test_high_memory_is_rejected_for_anchor_and_extra(self):
        for stdout in (f'{self.GJS}, gjs, 257 MiB\n',
                       f'{self.GJS}, gjs, 64 MiB\n{self.NAUTILUS}, /usr/bin/nautilus, 257 MiB\n'):
            self.assertFalse(self.check(stdout)['passed'])

    def test_stale_or_missing_gjs_and_query_failure_are_rejected(self):
        self.assertFalse(self.check(f'{self.GJS}, gjs, 64 MiB\n',tick=1772)['passed'])
        for stdout in ('',f'{self.NAUTILUS}, /usr/bin/nautilus, 72 MiB\n'):
            result=self.check(stdout)
            self.assertFalse(result['passed']);self.assertEqual(result['pinned_gjs_match_count'],0)
        self.assertTrue(self.check('')['original_gjs_only_rule_would_pass'])
        self.assertFalse(self.check(f'{self.GJS}, gjs, 64 MiB\n',code=1)['passed'])
        def unavailable(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],20)
        with self.assertRaises(subprocess.TimeoutExpired):
            worker.gpu_idle_guard(self.desktop,self.classifier,runner=unavailable)

    def test_second_unpinned_gjs_or_any_duplicate_PID_is_rejected(self):
        second=1652222;info=copy.deepcopy(self.info);info[second]=copy.deepcopy(info[self.GJS]);info[second]['pid']=second
        result=self.check(f'{self.GJS}, gjs, 64 MiB\n{second}, gjs, 64 MiB\n',info=info)
        self.assertTrue(result['classification']['all_inventory_processes_verified_allowed']);self.assertFalse(result['passed'])
        for stdout in (f'{self.GJS}, gjs, 64 MiB\n{self.GJS}, gjs, 64 MiB\n',
            f'{self.GJS}, gjs, 64 MiB\n{self.NAUTILUS}, /usr/bin/nautilus, 72 MiB\n{self.NAUTILUS}, /usr/bin/nautilus, 72 MiB\n'):
            result=self.check(stdout);self.assertFalse(result['passed']);self.assertFalse(result['unique_inventory_PIDs'])

    def test_no_basename_UID_inode_cgroup_or_startup_binding_exemption(self):
        changes=[lambda x:x[self.NAUTILUS].update(uid_fields=[-1]*4),
            lambda x:x[self.NAUTILUS].update(exe='/home/fixture/nautilus'),
            lambda x:x[self.NAUTILUS]['executable_identity'].update(same_system_inode=False),
            lambda x:x[self.NAUTILUS]['executable_identity'].update(system_uid=1000),
            lambda x:x[self.NAUTILUS]['executable_identity'].update(system_mode=0o777),
            lambda x:x[self.NAUTILUS]['cgroups'][0].update(path='/unverified'),
            lambda x:x[self.GJS]['executable_identity'].update(system_mode=0o700)]
        for change in changes:
            with self.subTest(change=change):
                info=copy.deepcopy(self.info);change(info)
                self.assertFalse(self.check(f'{self.GJS}, gjs, 64 MiB\n{self.NAUTILUS}, /usr/bin/nautilus, 72 MiB\n',info=info)['passed'])
        self.assertFalse(self.check(f'{self.GJS}, /usr/bin/gjs, 64 MiB\n')['passed'])
        self.desktop['memory_limit_MiB']=512
        self.assertFalse(self.check(f'{self.GJS}, gjs, 64 MiB\n')['passed'])

    def test_real_worker_policy_is_adopted_and_preserves_full_evidence(self):
        result=self.check(f'{self.GJS}, gjs, 64 MiB\n{self.NAUTILUS}, /usr/bin/nautilus, 72 MiB\n')
        self.assertEqual(result['policy_status'],'adopted');self.assertNotIn('proposal_only',result)
        self.assertEqual(result['permitted_pinned_or_verified_extra_desktop'],[True,True])
        self.assertTrue(result['unique_inventory_PIDs'])
        self.assertEqual(set(worker.EXTRA_DESKTOPS),{'verified_desktop_gnome_nautilus','verified_desktop_gnome_ptyxis'})
        self.assertEqual(result['classification']['observations'][1]['process_metadata'],self.info[self.NAUTILUS])


if __name__=='__main__':unittest.main()
