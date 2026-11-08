"""Portable CPU training/controller checks; no official teacher or GPU forward."""
import copy
import inspect
import random
import numpy as np
import tempfile
import unittest
from contextlib import ExitStack,contextmanager
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from vsr import train as worker,checkpoints,losses
from test_teacher_losses_train import tiny_model,inputs
from test_hr_latent_train import assert_exact_engine
from test_trd_primitives import grid


class SmallTeacher(nn.Module):
    """Deterministic synthetic external teacher; not a VideoMAEv2 result."""
    def __init__(self,binding,device='cpu'):
        super().__init__();self.binding=copy.deepcopy(binding);self.forward_count=0;self.parameter_updates=0
        self.model=nn.Linear(1,1).requires_grad_(False).eval()
        # Isolated construction also exercises the engine's teacher-before-graph order.
        with torch.no_grad():self.model.weight.fill_(.125);self.model.bias.zero_()
        self.events=[];self.closed=False
    @torch.no_grad()
    def features(self,hr):
        self.events.append(('teacher',torch.is_grad_enabled()));self.forward_count+=1
        x=torch.arange(64*8,dtype=torch.float32).reshape(1,64,8)
        return torch.sin(x*.37+float(hr.flatten()[0]))
    def close(self):self.closed=True;self.model=None


class SmallAlignment(nn.Module):
    """Same mapping/mixed-dtype objective, smaller spatial/channel fixture."""
    def __init__(self,student_dim,init_seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.projector=nn.Sequential(nn.Linear(student_dim,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,8))
    def forward(self,student,target,token_grid):
        with torch.autocast('cpu',dtype=torch.bfloat16):projected=self.projector(student.float())
        mapped=losses.videomaev2_align_student_time(projected,token_grid)
        return losses.videomaev2_full_gram_loss(mapped,target,(16,*token_grid[1:]))


def engine(arm='videomaev2',weight=.5):
    # Teacher factory in the production code isolates all RNG; preserve the same
    # contract in this deliberately small fixture.
    def teacher(*args,**kwargs):
        with torch.random.fork_rng(devices=[]):return SmallTeacher(*args,**kwargs)
    result=worker.VideoMAEv2Step(tiny_model(),arm=arm,weight=weight,teacher_binding={'synthetic':True},
                student_layer=17,training_seed=31,teacher_factory=teacher,alignment_factory=SmallAlignment)
    result.base_load_audit={'synthetic':True}
    return result


def probe_rows():
    return [dict(data_index=i,nominal_sigma=sigma,fm=1.,unit_aux=.1,fm_grad_norm=2.,aux_grad_norm=1.,head_grad_norm=.5)
            for i in range(5) for sigma in (.2,.5,.8)]


class VideoMAEv2TrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);self.clean,self.low,self.text=inputs()

    def test_zero_weight_both_labels_equal_original_NoRef_three_complete_updates(self):
        base=worker.NoRefContinuationStep(tiny_model(),arm='flow_only',training_seed=31)
        engines=[engine(arm,0.) for arm in ('baseline','videomaev2')]
        with patch('vsr.train.capture_block_output',side_effect=AssertionError('zero weight must not capture')):
            for _ in range(3):
                expected=base.step(self.clean,self.low,self.text,grid())
                for current in engines:
                    actual=current.step(self.clean,self.low,self.text,grid())
                    self.assertEqual(actual['fm'],expected['fm']);self.assertEqual(actual['total'],expected['total'])
                    self.assertEqual(actual['timestep_index'],expected['timestep_index'])
                    self.assertEqual(actual['sigma'],expected['sigma']);self.assertFalse(actual['teacher_forward'])
                    assert_exact_engine(self,base,current)
                    self.assertIsNone(current.teacher);self.assertIsNone(current.alignment)
                    self.assertEqual(current.teacher_forward_count,0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_forced_objectives_do_not_draw_RNG_teacher_precedes_student_and_aux_reaches_backbone(self):
        current=engine();states={k:getattr(current.rng,k).get_state().clone() for k in ('noise','timestep','mask','corruption')}
        noise=torch.randn(self.clean.shape,generator=torch.Generator().manual_seed(66));events=current.teacher.events
        forward=current.model.forward
        def traced(*args,**kwargs):events.append(('student',torch.is_grad_enabled()));return forward(*args,**kwargs)
        with patch.object(current.model,'forward',side_effect=traced):
            fm,aux,diag=current.objectives(self.clean,self.low,self.text,grid(),hr_rgb=torch.ones(1),index=200,noise=noise)
        self.assertEqual(events,[('teacher',False),('student',True)])
        self.assertEqual(diag['alignment']['student_gram_dtype'],'torch.bfloat16')
        self.assertEqual(diag['alignment']['teacher_gram_dtype'],'torch.float32')
        self.assertEqual(diag['alignment']['loss_dtype'],'torch.float32')
        params=tuple(p for p in current.model.parameters() if p.requires_grad)
        head=tuple(current.alignment.parameters())
        gfm=torch.autograd.grad(fm,params+head,retain_graph=True,allow_unused=True)
        gaux=torch.autograd.grad(aux,params+head,allow_unused=True)
        self.assertTrue(all(g is None for g in gfm[len(params):]))
        self.assertGreater(worker._trd_norm(gaux[:len(params)]),0)
        self.assertGreater(worker._trd_norm(gaux[len(params):]),0)
        self.assertTrue(all(p.grad is None for p in current.teacher.model.parameters()))
        self.assertTrue(all(torch.equal(v,getattr(current.rng,k).get_state()) for k,v in states.items()))
        self.assertEqual(current.steps,0);self.assertFalse(current.optimizer.state)
        self.assertEqual(current.teacher_forward_count,0);self.assertEqual(current.teacher.forward_count,1)

    def test_positive_update_changes_head_and_student_with_frozen_teacher_and_cleanup(self):
        current=engine();teacher=copy.deepcopy(current.teacher.model.state_dict());student=copy.deepcopy(current.model.state_dict());head=copy.deepcopy(current.alignment.state_dict())
        modes=[(m,m.training) for m in current.model.modules()]
        row=current.step(self.clean,self.low,self.text,grid(),hr_rgb=torch.ones(1))
        self.assertGreater(row['head_total_grad_norm_before_clip'],0);self.assertGreater(row['student_total_grad_norm_before_clip'],0)
        self.assertTrue(any(not torch.equal(p,student[n]) for n,p in current.model.named_parameters() if p.requires_grad))
        self.assertTrue(all(not torch.equal(p,head[n]) for n,p in current.alignment.named_parameters()))
        self.assertTrue(all(torch.equal(p,teacher[n]) for n,p in current.teacher.model.state_dict().items()))
        self.assertEqual(current.teacher_forward_count,1);self.assertEqual(current.teacher.parameter_updates,0)
        self.assertTrue(all(p.grad is None for p in current.parameters));self.assertTrue(all(m.training==v for m,v in modes))
        self.assertIsNone(current.model._condition_tokens);self.assertIsNone(current.model._captured)
        self.assertEqual(sum(len(b._forward_hooks) for b in current.model.transformer.blocks),0)

    def test_loss_failure_cleans_modes_gradients_features_and_does_not_update(self):
        current=engine();before=copy.deepcopy(current.model.state_dict());flags=[(m,m.training) for m in current.model.modules()]
        with patch.object(current.alignment,'forward',side_effect=RuntimeError('synthetic loss failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic loss failure'):
                current.step(self.clean,self.low,self.text,grid(),hr_rgb=torch.ones(1))
        self.assertEqual(current.steps,0);self.assertEqual(current.teacher_forward_count,0)
        self.assertEqual(current.teacher.forward_count,1);self.assertFalse(current.optimizer.state)
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in current.model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in current.parameters));self.assertTrue(all(m.training==v for m,v in flags))
        self.assertIsNone(current.model._condition_tokens);self.assertEqual(sum(len(b._forward_hooks) for b in current.model.transformer.blocks),0)
        teacher=current.teacher;worker._hr_release(current);self.assertIsNone(current.teacher);self.assertTrue(teacher.closed)

    def test_gate_complete_15_cases_fixed_weight_not_calibration_and_failure_cases(self):
        data=probe_rows();report=worker._vmae_target_gate(data)
        self.assertTrue(report['eligible']);self.assertEqual(report['weight'],.5);self.assertFalse(report['weight_calibrated'])
        self.assertEqual(report['VJP_calls'],30);self.assertEqual(report['optimizer_updates'],0)
        for row in data[8:]:row['aux_grad_norm']=0.
        self.assertTrue(worker._vmae_target_gate(data)['eligible'])
        data[7]['aux_grad_norm']=0.;self.assertFalse(worker._vmae_target_gate(data)['eligible'])
        for change in (probe_rows()[:-1],list(reversed(probe_rows()))):
            with self.assertRaises(ValueError):worker._vmae_target_gate(change)
        for key in ('fm','unit_aux','fm_grad_norm','aux_grad_norm','head_grad_norm'):
            for invalid in (float('nan'),float('inf'),-1.):
                data=probe_rows();data[14][key]=invalid
                with self.subTest(key=key,value=invalid),self.assertRaises(ValueError):worker._vmae_target_gate(data)

    def test_model_construction_restores_all_global_RNG_on_success_and_failure(self):
        p={'study':{'device':'cpu','allowed_base_root':'synthetic','model_directory':'synthetic','training_seed':31},
           'base':{'checkpoint':{'path':'synthetic','sha256':'a'*64},'config':{}},'teacher_binding':None,
           'optimizer':{'learning_rate':1e-4,'weight_decay':.01,'max_grad_norm':1.}}
        def load(*args,**kwargs):
            random.seed(9);random.random();np.random.seed(8);np.random.rand(3);torch.manual_seed(7);torch.rand(5)
            return tiny_model(),{'checkpoint_sha256':'a'*64,'step':8192,'config':{}}
        before=checkpoints._trd_global_rng_state()
        with patch('vsr.checkpoints.load_training_checkpoint',side_effect=load):value=worker._vmae_engine(p,'baseline',0.)
        self.assertTrue(checkpoints._hr_equal(before,checkpoints._trd_global_rng_state()));self.assertEqual(value.steps,0)
        def fail(*args,**kwargs):load();raise RuntimeError('synthetic construction failure')
        with patch('vsr.checkpoints.load_training_checkpoint',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'synthetic construction failure'):worker._vmae_engine(p,'baseline',0.)
        self.assertTrue(checkpoints._hr_equal(before,checkpoints._trd_global_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_post_optimizer_nonfinite_fails_without_returned_update_and_cleans(self):
        current=engine('baseline',0.);counter=worker.UpdateCounter();original=current.optimizer.step
        def corrupt(*args,**kwargs):
            result=original(*args,**kwargs)
            with torch.no_grad():current.parameters[0].flatten()[0]=float('inf')
            return result
        current.optimizer.step=corrupt;counter.attach(current.optimizer,'synthetic')
        with self.assertRaisesRegex(ValueError,'Nonfinite updated parameter'):
            current.step(self.clean,self.low,self.text,grid())
        count=counter.snapshot();self.assertEqual(count['diagnostic_optimizer_updates_applied'],1)
        self.assertEqual(count['completed_step_returns'],0);self.assertEqual(current.steps,0)
        self.assertTrue(all(p.grad is None for p in current.parameters));self.assertIsNone(current.model._condition_tokens)

    def test_full_step_timer_includes_HR_teacher_inside_synchronization(self):
        events=[]
        class Event:
            def __init__(self,name):self.name=name
            def record(self):events.append(self.name)
            def elapsed_time(self,end):events.append('elapsed');return 125.
        class Engine:
            def step(self,*args,hr_rgb):events.append('teacher_and_student');self.rgb=hr_rgb;return {'step':1}
        current=Engine();hr=torch.ones(1)
        with patch('torch.cuda.Event',side_effect=[Event('start'),Event('end')]),patch('torch.cuda.synchronize',side_effect=lambda:events.append('sync')),patch.object(worker.time,'perf_counter',side_effect=[10.,10.25]):
            row,wall,gpu=worker._vmae_step(current,{'clean':self.clean,'low':self.low},self.text,grid(),'cpu',hr_rgb=hr)
        self.assertEqual(events,['sync','start','teacher_and_student','end','sync','elapsed']);self.assertIs(current.rgb,hr)
        self.assertEqual((wall,gpu),(.25,.125));self.assertFalse(torch.cuda.is_initialized())

    @contextmanager
    def mocked_preflight(self,root,fail=False):
        run=root/'preflight';run.mkdir();p={'schema':'vsr_videomaev2_preflight_protocol_v1','study':{'run_root':str(root),'device':'cpu','model_directory':'synthetic','training_seed':31},
            'study_file':{'path':str(root/'study.json')},'cache':{'root':'synthetic','seal':{'sha256':'a'*64}},'sources':{'synthetic':True},'base':{'config':{}},
            'memory_cap_bytes':16*1024**3,'teacher_binding':{'synthetic':True}}
        worker.save(run/'protocol.json',p);worker.save(run/'status.json',{'status':'planned'});expected=worker.file_sha(run/'protocol.json')
        events=[];engines=[];saved={};pair={'clean':self.clean,'low':self.low};text=self.text
        class Reader:
            ids=[f'train_{i}' for i in range(1641)]
            def __init__(self,*a,**kw):events.append('reader_open')
            def __getitem__(self,index):return pair
            def text_for(self,index):return text
            def hr_rgb(self,index):return torch.ones(1)
            def hr_rgb_binding(self,index):return {'synthetic_index':index}
            def close(self):events.append('reader_close')
        def new_engine(p,arm,weight):
            e=engine(arm,weight);engines.append(e)
            if fail and len(engines)==1:
                e.objectives=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('synthetic probe failure'))
            return e
        def timed(e,pair,text,scheduler,device,*,hr_rgb=None):return e.step(pair['clean'],pair['low'],text,scheduler,hr_rgb=hr_rgb),.1,.08
        def save_cp(path,e,config,**kwargs):
            Path(path).parent.mkdir(parents=True);Path(path).write_text('synthetic')
            saved['state']={'student':copy.deepcopy(e.model.state_dict()),'head':copy.deepcopy(e.alignment.state_dict()),'optimizer':copy.deepcopy(e.optimizer.state_dict()),
                'rng':{n:getattr(e.rng,n).get_state().clone() for n in ('noise','timestep','mask','corruption')},'global':checkpoints._trd_global_rng_state(),'steps':e.steps,'teacher_forward_count':e.teacher_forward_count,'seconds':kwargs['cumulative_training_seconds']}
        def restore(e,*a,**kw):
            state=saved['state'];e.model.load_state_dict(state['student']);e.alignment.load_state_dict(state['head']);e.optimizer.load_state_dict(state['optimizer'])
            e.steps=state['steps'];e.teacher_forward_count=state['teacher_forward_count']
            for n,v in state['rng'].items():getattr(e.rng,n).set_state(v)
            checkpoints._trd_restore_global_rng(state['global']);return {'cumulative_training_seconds':state['seconds']}
        def native_load(*a,**kw):
            m=tiny_model();m.load_state_dict(saved['state']['student']);return m,{}
        @contextmanager
        def session(protocol,cleanup,failure):
            events.append('lease_open')
            try:yield {'synthetic_CPU_only':True}
            except BaseException as error:failure(error);raise
            finally:cleanup();events.append('lease_release')
        with ExitStack() as stack:
            for target,value in [('collect_videomaev2_protocol',lambda path:p),('_configure_runtime',lambda c:{}),('_training_scheduler',lambda *a:grid()),('_vmae_engine',new_engine),('_vmae_step',timed),('_vmae_config',lambda *a:{}),('_memory',lambda *a:{'synthetic':True}),('_hr_gpu_session',session)]:stack.enter_context(patch.object(worker,target,value))
            stack.enter_context(patch('vsr.data.PreparedTrainingData',Reader));stack.enter_context(patch('vsr.vae.load_frozen_vae',lambda *a,**kw:nn.Linear(1,1).requires_grad_(False)))
            stack.enter_context(patch('vsr.checkpoints.save_videomaev2_checkpoint',save_cp));stack.enter_context(patch('vsr.checkpoints.restore_videomaev2_training_state',restore));stack.enter_context(patch('vsr.checkpoints.load_training_checkpoint',native_load))
            stack.enter_context(patch('torch.cuda.empty_cache'));stack.enter_context(patch('torch.cuda.reset_peak_memory_stats'))
            # Parent checkpoint reload uses fork_rng(devices=[0]); suppress the
            # CUDA list only in this CPU fixture, retaining CPU RNG isolation.
            fork=torch.random.fork_rng
            stack.enter_context(patch('torch.random.fork_rng',side_effect=lambda devices=None,**kw:fork(devices=[],**kw)))
            stack.enter_context(patch.dict(worker.os.environ,worker.STRICT_ENVIRONMENT))
            yield run,expected,events,engines

    def test_mocked_preflight_real_tiny_steps_exact_26_updates_31_teacher_forwards_30_VJP(self):
        with tempfile.TemporaryDirectory() as temp,self.mocked_preflight(Path(temp)) as (run,expected,events,engines):
            result=worker.execute_videomaev2_preflight(run,expected)
            self.assertEqual(result['status'],'passed_videomaev2_preflight')
            self.assertEqual(result['diagnostic_optimizer_updates'],26);self.assertEqual(result['teacher_forwards'],31)
            self.assertEqual(result['gradient_probe_cases'],15);self.assertEqual(result['gradient_VJP_calls'],30)
            for key in ('diagnostic_optimizer_updates_attempted','diagnostic_optimizer_updates_applied','completed_step_returns'):
                self.assertEqual(result['optimizer_counter'][key],26)
            self.assertTrue(result['zero_weight_exact']);self.assertTrue(result['resume_reports']['videomaev2']['exact_full_engine_state'])
            self.assertEqual(len(result['resources']['baseline']['all8_rows']),8);self.assertEqual(len(result['resources']['videomaev2']['all8_rows']),8)
            self.assertEqual(worker.read(run/'status.json')['status'],'completed');self.assertFalse((run/'failure.json').exists())
            self.assertTrue(all(e.teacher is None for e in engines));self.assertIn('reader_close',events);self.assertEqual(events[-1],'lease_release')
        self.assertFalse(torch.cuda.is_initialized())

    def test_mocked_preflight_failure_stops_before_updates_preserves_failure_and_releases(self):
        with tempfile.TemporaryDirectory() as temp,self.mocked_preflight(Path(temp),fail=True) as (run,expected,events,engines):
            with self.assertRaisesRegex(RuntimeError,'synthetic probe failure'):worker.execute_videomaev2_preflight(run,expected)
            value=worker.read(run/'failure.json');self.assertEqual(value['diagnostic_optimizer_updates_returned'],0)
            self.assertEqual(value['teacher_forwards_returned'],4);self.assertEqual(value['gradient_probe_cases_completed'],0)
            self.assertEqual(worker.read(run/'status.json')['status'],'failed_videomaev2_preflight')
            self.assertTrue(all(e.teacher is None for e in engines));self.assertIn('reader_close',events);self.assertEqual(events[-1],'lease_release')
            self.assertFalse((run/'summary.json').exists())
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main(verbosity=2)
