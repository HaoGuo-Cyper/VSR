"""HR latent paired engine/controller contracts; tiny CPU fixtures only."""
import copy
import gc
import io
import tempfile
import unittest
import weakref
from contextlib import ExitStack,contextmanager,redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vsr import train as worker
from vsr import checkpoints
from test_teacher_losses_train import tiny_model,inputs,schedule


def assert_exact_engine(test,left,right):
    a,b=dict(left.model.named_parameters()),dict(right.model.named_parameters())
    test.assertEqual(set(a),set(b))
    for name,p in a.items():
        other=b[name];torch.testing.assert_close(p,other,rtol=0,atol=0,msg=name)
        if p.requires_grad:
            test.assertEqual(set(left.optimizer.state[p]),set(right.optimizer.state[other]))
            for key,value in left.optimizer.state[p].items():
                torch.testing.assert_close(value,right.optimizer.state[other][key],rtol=0,atol=0,msg=f'{name}/{key}')
    for stream in ('noise','timestep','mask','corruption'):
        test.assertTrue(torch.equal(getattr(left.rng,stream).get_state(),getattr(right.rng,stream).get_state()))


class HRLatentEngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.clean,self.low,self.text=inputs()

    def test_zero_weight_is_exact_original_flow_trajectory_with_no_head_or_EMA(self):
        baseline=worker.NoRefContinuationStep(tiny_model(),arm='flow_only',training_seed=29,learning_rate=.002)
        disabled=worker.HRLatentStep(tiny_model(),weight=0.,training_seed=29,learning_rate=.002)
        self.assertIsNone(disabled.alignment);self.assertIsNone(disabled.ema)
        self.assertEqual(len(disabled.parameters),len(baseline.parameters))
        with patch('vsr.teacher.capture_block_output',side_effect=AssertionError('zero-weight may not capture')):
            for sigma in (.8,.2,0.):
                expected=baseline.step(self.clean,self.low,self.text,schedule(sigma))
                actual=disabled.step(self.clean,self.low,self.text,schedule(sigma))
                self.assertEqual(actual['fm'],expected['fm']);self.assertEqual(actual['total'],expected['total'])
                self.assertEqual(actual['align'],0.);self.assertEqual(actual['timestep_index'],expected['timestep_index'])
                self.assertFalse(actual['teacher_forward']);self.assertEqual(actual['EMA_updates'],0)
                assert_exact_engine(self,baseline,disabled)
        self.assertFalse(torch.cuda.is_initialized())

    def test_positive_weight_uses_same_flow_streams_unmodified_LR_and_all_sigmas(self):
        baseline=worker.NoRefContinuationStep(tiny_model(),arm='flow_only',training_seed=31)
        engine=worker.HRLatentStep(tiny_model(),weight=.01,training_seed=31)
        self.assertIsNone(engine.ema);self.assertIsNotNone(engine.alignment)
        original_low=self.low.clone();records=[[],[]]
        def traced(original,bucket):
            def call(*args):
                result=original(*args);bucket.append(tuple(x.detach().clone() if isinstance(x,torch.Tensor) else x for x in result))
                return result
            return call
        forward=engine.model.forward;events=[]
        def student(noisy,timestep,text,low,*args,**kwargs):
            events.append(low.detach().clone());return forward(noisy,timestep,text,low,*args,**kwargs)
        with patch.object(baseline.rng,'flow_inputs',side_effect=traced(baseline.rng.flow_inputs,records[0])),\
             patch.object(engine.rng,'flow_inputs',side_effect=traced(engine.rng.flow_inputs,records[1])),\
             patch.object(engine.model,'forward',side_effect=student) as forwards:
            for sigma in (0.,.2,.8):
                baseline.step(self.clean,self.low,self.text,schedule(sigma))
                row=engine.step(self.clean,self.low,self.text,schedule(sigma))
                self.assertEqual(row['sigma'],float(torch.tensor(sigma)))
                self.assertEqual(row['alignment']['selected_tokens'],36)
                self.assertEqual(row['alignment']['target_elements'],16*9*4*4)
                self.assertTrue(row['alignment']['all_tokens_supervised'])
                self.assertEqual(row['student_layer'],7);self.assertFalse(row['teacher_forward'])
                self.assertEqual(row['EMA_updates'],0);self.assertGreater(row['align'],0.)
                self.assertAlmostEqual(row['total'],row['fm']+.01*row['align'],places=6)
            self.assertEqual(forwards.call_count,3)
        for expected,actual in zip(records[0],records[1]):
            for a,b in zip(expected,actual):
                if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
                else:self.assertEqual(a,b)
        self.assertTrue(all(torch.equal(low,original_low) for low in events))
        self.assertTrue(torch.equal(self.low,original_low));self.assertFalse(torch.cuda.is_initialized())

    def test_head_initialization_is_seeded_and_preserves_default_and_flow_RNG(self):
        model=tiny_model();before=torch.get_rng_state().clone()
        engine=worker.HRLatentStep(model,weight=.01,training_seed=37)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.rand(11)
        another=worker.HRLatentStep(tiny_model(),weight=.1,training_seed=37)
        for name,value in engine.alignment.state_dict().items():
            torch.testing.assert_close(value,another.alignment.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(engine.parameters[-1].dtype,torch.float32)
        baseline=worker.HRLatentStep(tiny_model(),weight=0.,training_seed=37)
        for stream in ('noise','timestep','mask','corruption'):
            self.assertTrue(torch.equal(getattr(engine.rng,stream).get_state(),getattr(baseline.rng,stream).get_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_positive_step_restores_modes_clears_graph_and_preserves_frozen_student(self):
        model=tiny_model();engine=worker.HRLatentStep(model,weight=.1)
        flags=[(m,m.training) for m in model.modules()];head_flags=[(m,m.training) for m in engine.alignment.modules()]
        frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        gradients={};original=engine.optimizer.step
        def step(*args,**kwargs):
            gradients.update({name:p.grad.detach().clone() for name,p in engine._named_optimizer_parameters if p.grad is not None})
            return original(*args,**kwargs)
        with patch.object(engine.optimizer,'step',side_effect=step):row=engine.step(self.clean,self.low,self.text,schedule())
        self.assertTrue(any(name.startswith('student_projector.') for name in gradients))
        self.assertTrue(any(name.startswith('restoration.') for name in gradients))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients.values()))
        self.assertTrue(all(torch.equal(p,frozen[name]) for name,p in model.named_parameters() if not p.requires_grad))
        self.assertTrue(all(p.grad is None for p in engine.parameters));self.assertIsNone(self.clean.grad)
        self.assertTrue(all(m.training==flag for m,flag in flags+head_flags))
        self.assertIsNone(model._condition_tokens);self.assertIsNone(model._captured)
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)
        self.assertEqual(row['step'],1)

    def test_loss_failure_cleans_modes_and_graph_without_optimizer_update(self):
        model=tiny_model();engine=worker.HRLatentStep(model,weight=.01)
        before={n:p.detach().clone() for n,p in model.named_parameters()};flags=[(m,m.training) for m in model.modules()]
        with patch.object(engine.alignment,'forward',side_effect=RuntimeError('fixture target failure')):
            with self.assertRaisesRegex(RuntimeError,'fixture target failure'):engine.step(self.clean,self.low,self.text,schedule())
        self.assertEqual(engine.steps,0);self.assertFalse(engine.optimizer.state)
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in model.named_parameters()))
        self.assertTrue(all(p.grad is None for p in engine.parameters))
        self.assertTrue(all(m.training==flag for m,flag in flags))
        self.assertIsNone(model._condition_tokens);self.assertIsNone(model._captured)
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)

    def test_invalid_weight_layer_or_unfrozen_target_rejects(self):
        for weight in (-.1,float('nan'),float('inf'),True):
            with self.subTest(weight=weight),self.assertRaises(ValueError):worker.HRLatentStep(tiny_model(),weight=weight)
        for layer in (-1,30,7.):
            with self.subTest(layer=layer),self.assertRaises(ValueError):worker.HRLatentStep(tiny_model(),weight=.01,student_layer=layer)
        engine=worker.HRLatentStep(tiny_model(),weight=.01)
        with self.assertRaisesRegex(ValueError,'Frozen finite matched'):engine.step(self.clean.requires_grad_(),self.low,self.text,schedule())
        self.assertFalse(torch.cuda.is_initialized())


class HRLatentMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);official=self.root/'official';(official/'transformer').mkdir(parents=True)
        worker.save(official/'transformer/config.json',{'num_layers':30,'in_channels':16,'patch_size':[1,2,2],
            'num_attention_heads':12,'attention_head_dim':128})
        config={'seed':20261102,'model':{'local_dir':str(official),'repo_id':checkpoints.OFFICIAL_REPO_ID,
            'revision':checkpoints.OFFICIAL_REVISION},'conditioning':{},'reproducibility':{'mode':'strict'},
            'data':dict(num_frames=33,temporal_stride=1,height=128,width=224,scale=4),
            'training':dict(seed=20261102,updates=8192,variants=['Base'],learning_rate=.001,weight_decay=.02,max_grad_norm=2.)}
        self.config=worker._canonical_config(config);config_path=self.root/'BaseSource/config.json';config_path.parent.mkdir()
        worker.save(config_path,self.config);admission=self.root/'admission.json';worker.save(admission,{'expected_gpu_uuid':'GPU-fixture'})
        self.base={'config':self.config,'config_file':worker.binding(config_path),
            'checkpoint':{'path':str(self.root/'BaseSource/Base/checkpoint_008192.pt'),'sha256':worker.SMOKE_BASE_SHA,'bytes':10}}
        self.cache={'root':str(self.root/'cache'),'config':self.base['config_file'],'seal':{'sha256':'a'*64},
            'selection':{'training_ids':[f'train_{i}' for i in range(1641)]}}
        self.inputs=[{'record_id':f'train_{i}','index':i} for i in range(22)]
        self.study={'schema':'hr_vae_study_v1','updates':2048,'eval_updates':[512,1024,2048],
            'checkpoint_updates':[512,1024,1536,2048],
            'arms':[{'label':'baseline','weight':0.},{'label':'hr001','weight':.01},{'label':'hr01','weight':.1}],
            'student_layer':7,'smooth_l1_beta':1.,'training_seed':20261102,'base_config':str(config_path),
            'base_checkpoint':self.base['checkpoint']['path'],'allowed_base_root':str(self.root),
            'cache_root':self.cache['root'],'model_directory':str(official),'gpu_admission':str(admission),
            'gpu_lock':str(self.root/'gpu.lock'),'device':'cuda:0','run_root':str(self.root/'runs')}
        self.path=self.root/'study.json';worker.save(self.path,self.study)

    @contextmanager
    def metadata(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(worker,'_base_spec',return_value=self.base))
            stack.enter_context(patch.object(worker,'_cache_spec',return_value=(self.cache,self.inputs)))
            stack.enter_context(patch.object(worker,'_official_bindings',return_value={'fixture':'CPU'}))
            stack.enter_context(patch.object(worker,'_dependency_bindings',return_value={'fixture':'CPU'}))
            stack.enter_context(patch('torch.load',side_effect=AssertionError('metadata may not deserialize weights')))
            lease=stack.enter_context(patch('vsr.checkpoints.gpu_lease',side_effect=AssertionError('metadata may not query/admit GPU')))
            yield
            lease.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_protocol_is_metadata_only_and_fixes_three_arms_6144_budget_and_no_teacher_inference(self):
        with self.metadata():protocol=worker.collect_hr_protocol(self.path)
        self.assertEqual(protocol['scientific_update_budget'],6144)
        self.assertEqual(protocol['study']['arms'],self.study['arms'])
        self.assertFalse(protocol['online_teacher_forward'])
        self.assertIn('student only',protocol['inference_uses']);self.assertEqual(protocol['optimizer']['learning_rate'],.001)
        self.assertEqual(protocol['base'],self.base);self.assertEqual(protocol['cache'],self.cache)
        self.assertEqual(protocol['sources'],worker._source_bindings())
        self.assertEqual(protocol['strict_environment'],worker.STRICT_ENVIRONMENT)
        self.assertIn('same-protocol',protocol['resume_policy'])

    def test_fixed_design_mutations_reject_before_loading(self):
        for field,value in [('schema','wrong'),('updates',4096),('eval_updates',[512,1024,1536,2048]),
            ('checkpoint_updates',[2048]),('student_layer',9),('smooth_l1_beta',.1),('training_seed',20261103),
            ('arms',[{'label':'baseline','weight':0.},{'label':'hr001','weight':.1}])]:
            changed=copy.deepcopy(self.study);changed[field]=value
            (self.path).write_text(worker.json.dumps(changed))
            with self.subTest(field=field),self.metadata(),self.assertRaises(ValueError):worker.collect_hr_protocol(self.path)
        self.path.write_text(worker.json.dumps(self.study))
        original=self.cache['config'];self.cache['config']={'sha256':'changed'}
        with self.metadata(),self.assertRaisesRegex(ValueError,'original producer cache'):worker.collect_hr_protocol(self.path)
        self.cache['config']=original

    def test_HR_config_uses_plain_Base_inference_architecture_with_training_only_aux_metadata(self):
        with self.metadata():protocol=worker.collect_hr_protocol(self.path)
        for weight in (0.,.01,.1):
            config=worker._hr_config(protocol,weight)
            self.assertNotIn('restoration_architecture',config)
            self.assertEqual(config['training']['variants'],['Base']);self.assertEqual(config['training']['updates'],2048)
            self.assertEqual(config['auxiliary_supervision']['weight'],weight)
            self.assertFalse(config['auxiliary_supervision']['EMA']);self.assertFalse(config['auxiliary_supervision']['condition_corruption'])
            for key in ('model','conditioning','data'):self.assertEqual(config[key],self.config[key])
            self.assertEqual(checkpoints._check_config(config),config['_sha256'])
        self.assertEqual(self.config['training']['updates'],8192)

    def test_prepare_is_fresh_and_wrong_protocol_or_environment_rejects_before_GPU(self):
        run=self.root/'runs/new'
        with self.metadata():planned=worker.prepare_hr(self.path,run)
        self.assertEqual(planned['status'],'planned');self.assertFalse(worker.read(run/'status.json')['GPU_loaded'])
        with self.metadata(),self.assertRaisesRegex(ValueError,'New HR study'):worker.prepare_hr(self.path,run)
        with patch('vsr.checkpoints.gpu_lease') as lease:
            with self.assertRaisesRegex(ValueError,'Exact HR protocol'):worker._hr_protocol(run,'0'*64)
            lease.assert_not_called()
        with self.metadata(),patch.dict(worker.os.environ,worker.STRICT_ENVIRONMENT):
            same,p=worker._hr_protocol(run,planned['protocol_sha256'])
            self.assertEqual(same,run);self.assertEqual(p['study'],self.study)
        with self.metadata(),patch.dict(worker.os.environ,{'HF_HUB_OFFLINE':'0'}):
            with self.assertRaisesRegex(ValueError,'Strict/offline'):worker._hr_protocol(run,planned['protocol_sha256'])
        with patch.object(worker,'collect_hr_protocol',return_value={'changed':True}),patch.dict(worker.os.environ,worker.STRICT_ENVIRONMENT):
            with self.assertRaisesRegex(ValueError,'source/config/assets changed'):worker._hr_protocol(run,planned['protocol_sha256'])

    def test_cli_distinguishes_explicit_resume_from_fresh_prepare(self):
        args=worker.build_parser().parse_args(['hr-prepare','--config',str(self.path),'--run-dir',str(self.root/'runs/x')])
        self.assertEqual(args.command,'hr-prepare')
        args=worker.build_parser().parse_args(['hr-execute','--run-dir','/runs/x','--expected-protocol-sha','a'*64,'--resume'])
        self.assertTrue(args.resume)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            worker.build_parser().parse_args(['hr-probe','--run-dir','/runs/x','--expected-protocol-sha','a'*64,'--resume'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_requires_explicit_resume_and_a_real_same_protocol_probe(self):
        run=self.root/'runs/control';run.mkdir(parents=True)
        with self.metadata():protocol=worker.collect_hr_protocol(self.path)
        worker.save(run/'status.json',{'status':'failed_recoverable'})
        with patch.object(worker,'_hr_protocol',return_value=(run,protocol)),patch('vsr.checkpoints.gpu_lease') as lease, \
             patch('vsr.checkpoints.save_hr_latent_checkpoint',create=True),patch('vsr.checkpoints.restore_hr_latent_training_state',create=True):
            with self.assertRaisesRegex(ValueError,'Explicit checkpoint resume'):worker.execute_hr(run,'a'*64)
            lease.assert_not_called()
        worker.save(run/'status.json',{'status':'planned'},replace=True)
        (run/'probe').mkdir();worker.save(run/'probe/summary.json',{'status':'passed_hr_preflight','protocol_sha256':'wrong'})
        with patch.object(worker,'_hr_protocol',return_value=(run,protocol)),patch('vsr.checkpoints.gpu_lease') as lease, \
             patch('vsr.checkpoints.save_hr_latent_checkpoint',create=True),patch('vsr.checkpoints.restore_hr_latent_training_state',create=True):
            with self.assertRaisesRegex(ValueError,'passed real preflight'):worker.execute_hr(run,'a'*64)
            lease.assert_not_called()
        self.assertFalse((run/'attempt_000.json').exists());self.assertFalse(torch.cuda.is_initialized())

class HRLatentControllerTests(unittest.TestCase):
    setUp=HRLatentMetadataTests.setUp
    metadata=HRLatentMetadataTests.metadata

    def test_GPU_session_orders_failure_receipt_cleanup_then_lease_release(self):
        events=[]
        @contextmanager
        def lease(*unused):
            events.append('lease_enter')
            try:yield {'fixture':'CPU'}
            finally:events.append('lease_release')
        protocol={'gpu_lock':'/CPU-fixture.lock','device':'cuda:0','admission':{}}
        with patch('vsr.checkpoints.gpu_lease',lease):
            with worker._hr_gpu_session(protocol,lambda:events.append('cleanup'),lambda error:events.append('failure_receipt')):
                events.append('work')
        self.assertEqual(events,['lease_enter','work','cleanup','lease_release'])
        events.clear()
        with patch('vsr.checkpoints.gpu_lease',lease),self.assertRaisesRegex(RuntimeError,'CPU fixture failure'):
            with worker._hr_gpu_session(protocol,lambda:events.append('cleanup'),lambda error:events.append('failure_receipt')):
                events.append('work');raise RuntimeError('CPU fixture failure')
        self.assertEqual(events,['lease_enter','work','failure_receipt','cleanup','lease_release'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_step_timer_brackets_H2D_and_full_step_without_actual_CUDA(self):
        events=[]
        class Event:
            def __init__(self,name):self.name=name
            def record(self):events.append(self.name)
            def elapsed_time(self,end):events.append('elapsed');return 125.
        def step(*unused):events.append('step');return {'step':1}
        with patch('torch.cuda.Event',side_effect=[Event('start'),Event('end')]),\
             patch('torch.cuda.synchronize',side_effect=lambda:events.append('synchronize')),\
             patch.object(worker.time,'perf_counter',side_effect=[10.,10.25]):
            result,wall,gpu=worker._hr_step(SimpleNamespace(step=step),{'clean':torch.zeros(1),'low':torch.zeros(1)},
                torch.zeros(1),object(),'cpu')
        self.assertEqual(events,['synchronize','start','step','end','synchronize','elapsed'])
        self.assertEqual(result,{'step':1});self.assertEqual(wall,.25);self.assertEqual(gpu,.125)
        self.assertFalse(torch.cuda.is_initialized())

    @contextmanager
    def mocked_probe(self,*,tamper=False):
        with self.metadata():protocol=worker.collect_hr_protocol(self.path)
        protocol['device']=protocol['study']['device']='cpu'
        root=self.root;run=root/'runs/controller';run.mkdir(parents=True)
        cache=Path(protocol['cache']['root']);cache.mkdir();hashes={};events=[];factories=[];checkpoints_saved=[];saved_states={}
        pair={'clean':torch.ones(1,16,9,4,4),'low':torch.ones(1,16,9,4,4)}
        for index in (0,1641):
            directory=cache/f'clip_{index}';directory.mkdir()
            torch.save({'hr':torch.zeros(1,3,33,32,32)},directory/'paired_rgb.pt')
            torch.save({'clean':pair['clean']},directory/'fresh_latents.pt')
            for name in ('paired_rgb.pt','fresh_latents.pt'):hashes[f'clip_{index}/{name}']=worker.file_sha(directory/name)
        worker.save(cache/'cache_asset_seal.json',{'artifact_sha256':hashes})
        if tamper:(cache/'clip_1641/fresh_latents.pt').write_bytes(b'tampered')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('value',torch.zeros(()));self.projector=None
            def clear_features(self):pass
        class Engine:
            def __init__(self,weight):
                self.steps=0;self.model=Model();self.weight=weight
                self.alignment=torch.nn.Linear(1,1) if weight else None
                self.optimizer=SimpleNamespace(zero_grad=lambda **unused:None)
                self.base_load_audit={'variant':'Base','step':8192,'checkpoint_sha256':worker.SMOKE_BASE_SHA}
            def data_index(self,size):return self.steps%size
            def apply(self):
                self.steps+=1;self.model.value.fill_(self.steps)
                return {'step':self.steps,'fm':1.,'align':.1 if self.weight else 0.,'total':1.+self.weight*.1,
                        'sigma':.7,'grad_norm_before_clip':.25,'native_timestep':700.,'timestep_index':300,
                        'student_layer':7,'teacher_forward':False,'EMA_updates':0,'HR_target_detached':True}
        class Reader:
            ids=[f'train_{i}' for i in range(1641)]
            def __init__(self,*args,**kwargs):events.append('reader_loaded')
            def __len__(self):return len(self.ids)
            def __getitem__(self,index):return pair
            def text_for(self,index):return torch.zeros(1)
            def close(self):events.append('reader_closed')
        def engine(p,weight):factories.append(weight);return Engine(weight)
        def step(engine,*unused):return engine.apply(),.1,.08
        def direct(engine,*unused):return engine.apply()
        def vae(*args,**kwargs):
            model=torch.nn.Linear(1,1).requires_grad_(False)
            weakref.finalize(model,events.append,'VAE_released');return model
        def encode(vae,rgb):events.append('HR_encoded');return pair['clean'].clone()
        def hash_model(model):return {'CPU_value':model.value.item()}
        def save_cp(path,engine,config,**kwargs):
            checkpoints_saved.append((engine.steps,engine.weight));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            saved_states[str(path)]={'step':engine.steps,'seconds':kwargs['cumulative_training_seconds'],
                                     'initialization':copy.deepcopy(kwargs['initialization'])}
            path.write_bytes(b'CPU-only-probe-checkpoint');return {'path':str(path)}
        def load_cp(*args,**kwargs):
            model=Model();model.value.fill_(8);events.append('student_reloaded');return model,{}
        def restore(engine,*args,**kwargs):
            state=saved_states[str(args[0])];engine.steps=state['step'];engine.model.value.fill_(engine.steps)
            events.append('training_restored')
            return {'completed_updates':engine.steps,'cumulative_training_seconds':state['seconds'],
                    'initialization':copy.deepcopy(state['initialization'])}
        @contextmanager
        def lease(*unused):
            events.append('lease_enter')
            try:yield {'CPU_fixture':True}
            finally:
                self.assertIn('reader_closed',events)
                self.assertIn('VAE_released',events)
                if tamper:self.assertTrue((run/'probe/failure.json').exists())
                events.append('lease_release')
        with ExitStack() as stack:
            stack.enter_context(patch.object(worker,'_hr_protocol',return_value=(run,protocol)))
            stack.enter_context(patch('vsr.checkpoints.gpu_lease',lease))
            stack.enter_context(patch('vsr.data.PreparedTrainingData',Reader))
            stack.enter_context(patch('vsr.vae.load_frozen_vae',vae))
            stack.enter_context(patch('vsr.vae.encode_video',encode))
            stack.enter_context(patch.object(worker,'_configure_runtime',return_value={'CPU_fixture':True}))
            stack.enter_context(patch.object(worker,'_training_scheduler',return_value=object()))
            stack.enter_context(patch.object(worker,'_hr_engine',engine))
            stack.enter_context(patch.object(worker,'_hr_step',step))
            stack.enter_context(patch.object(worker.NoRefContinuationStep,'step',direct))
            stack.enter_context(patch('vsr.model.tensor_state_hashes',hash_model))
            stack.enter_context(patch('vsr.checkpoints.save_hr_latent_checkpoint',save_cp,create=True))
            stack.enter_context(patch('vsr.checkpoints.load_training_checkpoint',load_cp))
            stack.enter_context(patch('vsr.checkpoints.restore_hr_latent_training_state',restore,create=True))
            stack.enter_context(patch.object(worker,'_memory',return_value={'CPU_fixture':True}))
            stack.enter_context(patch('torch.cuda.reset_peak_memory_stats'))
            stack.enter_context(patch('torch.cuda.empty_cache'))
            yield run,protocol,events,factories,checkpoints_saved
        self.assertFalse(torch.cuda.is_initialized())

    def test_probe_accounts27_diagnostics_two_fresh_HR_encodings_and_step8_reload(self):
        with self.mocked_probe() as (run,protocol,events,factories,saved):
            result=worker.probe_hr(run,'a'*64)
            self.assertEqual(result['diagnostic_optimizer_updates'],27);self.assertEqual(result['scientific_updates'],0)
            self.assertEqual(len(result['all_rows']),24)
            self.assertEqual([r['role'] for r in result['all_rows']],['baseline']*8+['hr001']*8+['hr01']*8)
            self.assertEqual(result['alignment_off_exact_original_GPU_trajectory_updates'],3)
            self.assertEqual([r['cache_index'] for r in result['fresh_HR_encodings']],[0,1641])
            self.assertEqual(events.count('HR_encoded'),2)
            self.assertEqual(saved,[(8,.01)])
            self.assertEqual(factories,[0.,.01,.01,.1,0.])
            self.assertTrue(result['save_inference_reload_training_resume_passed'])
            self.assertFalse(result['diagnostic_weights_reused_for_science'])
            self.assertLess(events.index('training_restored'),events.index('lease_release'))
            self.assertLess(events.index('VAE_released'),events.index('lease_release'))
            with self.assertRaisesRegex(ValueError,'single attempt'):worker.probe_hr(run,'a'*64)

    def test_tampered_val_HR_latent_is_rejected_by_seal_and_cleanup_precedes_release(self):
        with self.mocked_probe(tamper=True) as (run,protocol,events,factories,saved):
            with self.assertRaisesRegex(ValueError,'Asset checksum changed'):worker.probe_hr(run,'a'*64)
            self.assertTrue((run/'probe/failure.json').exists())
            self.assertEqual(worker.read(run/'probe/failure.json')['diagnostic_updates_returned'],0)
            self.assertLess(events.index('VAE_released'),events.index('lease_release'))
            self.assertEqual(factories,[]);self.assertEqual(saved,[])

    def test_formal_controller_is_three_fresh2048_arms_four_checkpoints_and6144_effective_updates(self):
        with self.mocked_probe() as (run,protocol,events,factories,saved):
            worker.probe_hr(run,'a'*64)
            worker.save(run/'status.json',{'status':'planned','protocol_sha256':'a'*64})
            result=worker.execute_hr(run,'a'*64)
            self.assertEqual(result['scientific_updates'],6144)
            self.assertEqual(result['diagnostic_updates_in_this_training'],0)
            self.assertFalse(result['selection_or_evaluation_done'])
            self.assertEqual(set(result['arms']),{'baseline','hr001','hr01'})
            self.assertEqual(factories[-3:],[0.,.01,.1])
            self.assertEqual(saved[1:],[(step,weight) for weight in (0.,.01,.1) for step in (512,1024,1536,2048)])
            for arm in protocol['study']['arms']:
                record=result['arms'][arm['label']]
                rows=[worker.json.loads(line) for line in Path(record['full_training_log']['path']).read_text().splitlines()]
                self.assertEqual(len(rows),2048);self.assertEqual([r['step'] for r in rows],list(range(1,2049)))
                self.assertEqual([r['data_index'] for r in rows],[i%1641 for i in range(2048)])
                self.assertEqual([r['step'] for r in record['checkpoint_files']],[512,1024,1536,2048])
                self.assertEqual(record['initial_student_hashes'],{'CPU_value':0.})
                self.assertEqual(record['final_student_hashes'],{'CPU_value':2048.})
            self.assertEqual(worker.read(run/'status.json')['status'],'completed')
            self.assertFalse(torch.cuda.is_initialized())

    def test_explicit_controller_resume_preserves_uncheckpointed_log_and_restores_only_valid_CP(self):
        with self.mocked_probe() as (run,protocol,events,factories,saved):
            worker.probe_hr(run,'a'*64)
            worker.save(run/'status.json',{'status':'planned','protocol_sha256':'a'*64})
            original=worker._hr_step
            def interrupted(engine,*args):
                if engine.weight==0. and engine.steps==515:raise RuntimeError('CPU interruption after CP512')
                return original(engine,*args)
            with patch.object(worker,'_hr_step',interrupted):
                with self.assertRaisesRegex(RuntimeError,'CPU interruption after CP512'):worker.execute_hr(run,'a'*64)
            self.assertEqual(worker.read(run/'status.json')['status'],'failed_recoverable')
            log=run/'arm_runs/baseline/training.jsonl';original_bytes=log.read_bytes()
            self.assertEqual(len(original_bytes.splitlines()),515)
            with self.assertRaisesRegex(ValueError,'Explicit checkpoint resume'):worker.execute_hr(run,'a'*64)
            result=worker.execute_hr(run,'a'*64,resume=True)
            preserved=run/'arm_runs/baseline/interrupted_log_attempt_001.jsonl'
            self.assertEqual(preserved.read_bytes(),original_bytes)
            rows=[worker.json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(rows),2048);self.assertEqual([r['step'] for r in rows],list(range(1,2049)))
            self.assertEqual(len(list(run.glob('attempt_*.json'))),2)
            self.assertEqual(worker.read(run/'failure_attempt_000.json')['current_arm_steps_returned'],515)
            self.assertEqual(result['scientific_updates'],6144)
            self.assertEqual([r['step'] for r in result['arms']['baseline']['checkpoint_files']],[512,1024,1536,2048])
            self.assertEqual(events.count('training_restored'),2)  # probe8 and formal512
            self.assertFalse(torch.cuda.is_initialized())

    def test_GPU_session_preserves_original_work_error_when_receipt_and_cleanup_also_fail(self):
        events=[];primary=RuntimeError('original CPU failure')
        @contextmanager
        def lease(*unused):
            events.append('lease_enter')
            try:yield {}
            finally:events.append('lease_release')
        def record(error):
            self.assertIs(error,primary);events.append('failure_receipt');raise OSError('secondary record failure')
        def cleanup():events.append('cleanup');raise OSError('secondary cleanup failure')
        protocol={'gpu_lock':'/CPU-fixture.lock','device':'cuda:0','admission':{}}
        with patch('vsr.checkpoints.gpu_lease',lease),self.assertRaises(RuntimeError) as caught:
            with worker._hr_gpu_session(protocol,cleanup,record):
                events.append('work');raise primary
        self.assertIs(caught.exception,primary)
        self.assertEqual(events,['lease_enter','work','failure_receipt','cleanup','lease_release'])
        self.assertTrue(any('Failure record error:' in note for note in primary.__notes__))
        self.assertTrue(any('Cleanup error:' in note for note in primary.__notes__))
        with self.assertRaisesRegex(OSError,'secondary cleanup failure'):worker._hr_cleanup_preserving(cleanup,None)
        self.assertFalse(torch.cuda.is_initialized())

    def test_copied_sealed_protocol_outside_declared_root_rejects_before_GPU(self):
        run=self.root/'runs/original'
        with self.metadata():planned=worker.prepare_hr(self.path,run)
        copied=self.root/'outside-declared-root';copied.mkdir()
        (copied/'protocol.json').write_bytes((run/'protocol.json').read_bytes())
        with self.metadata(),patch.dict(worker.os.environ,worker.STRICT_ENVIRONMENT):
            with self.assertRaisesRegex(ValueError,'child of its output root'):
                worker._hr_protocol(copied,planned['protocol_sha256'])
        self.assertFalse((copied/'probe').exists());self.assertFalse(torch.cuda.is_initialized())

    def _interrupt_fake_study_at515(self,run):
        worker.probe_hr(run,'a'*64)
        worker.save(run/'status.json',{'status':'planned','protocol_sha256':'a'*64})
        original=worker._hr_step
        def interrupted(engine,*args):
            if engine.weight==0. and engine.steps==515:raise RuntimeError('CPU interruption after CP512')
            return original(engine,*args)
        with patch.object(worker,'_hr_step',interrupted):
            with self.assertRaisesRegex(RuntimeError,'CPU interruption after CP512'):worker.execute_hr(run,'a'*64)
        return run/'arm_runs/baseline/training.jsonl'

    def test_resume_rejects_nonfinite_committed_prefix_without_truncating_log(self):
        with self.mocked_probe() as (run,protocol,events,factories,saved):
            log=self._interrupt_fake_study_at515(run)
            rows=[worker.json.loads(line) for line in log.read_text().splitlines()]
            rows[0]['grad_norm_before_clip']=float('nan')
            log.write_text(''.join(worker.json.dumps(row)+'\n' for row in rows));tampered=log.read_bytes()
            with self.assertRaisesRegex(ValueError,'Nonfinite resume log prefix'):worker.execute_hr(run,'a'*64,resume=True)
            self.assertEqual(log.read_bytes(),tampered)
            self.assertFalse((log.parent/'interrupted_log_attempt_001.jsonl').exists())

    def test_resume_rejects_checkpoint_time_different_from_valid_prefix(self):
        with self.mocked_probe() as (run,protocol,events,factories,saved):
            log=self._interrupt_fake_study_at515(run);original_bytes=log.read_bytes()
            restore=checkpoints.restore_hr_latent_training_state
            def wrong_time(*args,**kwargs):
                audit=restore(*args,**kwargs);audit['cumulative_training_seconds']+=1.;return audit
            with patch('vsr.checkpoints.restore_hr_latent_training_state',wrong_time):
                with self.assertRaisesRegex(ValueError,'checkpoint time differs'):worker.execute_hr(run,'a'*64,resume=True)
            self.assertEqual(log.read_bytes(),original_bytes)

    def test_resume_rejects_changed_original_initialization_sidecar(self):
        with self.mocked_probe() as (run,protocol,events,factories,saved):
            log=self._interrupt_fake_study_at515(run);original_bytes=log.read_bytes()
            (log.parent/'initialization.json').write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError,'Original initialization sidecar differs'):
                worker.execute_hr(run,'a'*64,resume=True)
            self.assertEqual(log.read_bytes(),original_bytes)
            self.assertFalse((log.parent/'interrupted_log_attempt_001.jsonl').exists())


if __name__=='__main__':unittest.main()
