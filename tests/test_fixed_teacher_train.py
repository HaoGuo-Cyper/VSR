"""Fixed HR VAE/DINO student engines and DINO resume, CPU fixtures only."""
import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from vsr import train as worker
from vsr import checkpoints
from vsr.losses import HRDINOCosineRegression
from test_teacher_losses_train import tiny_model,inputs,schedule
from test_hr_latent_train import assert_exact_engine
from test_hr_latent_checkpoints import (TinyHREngine,TinyRestoration,source_config,canonical,snapshot,PROTOCOL)


class FixedTeacherEngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.clean,self.low,self.text=inputs()
        self.dino=torch.randn((1,36,768),generator=torch.Generator().manual_seed(71))

    def test_zero_weight_both_kinds_exactly_reproduce_original_flow_and_ignore_unused_target(self):
        reference=worker.NoRefContinuationStep(tiny_model(),arm='flow_only',training_seed=41)
        arms=[worker.HRLatentStep(tiny_model(),weight=0.,teacher_kind=kind,training_seed=41) for kind in ('vae','dino')]
        with patch('vsr.teacher.capture_block_output',side_effect=AssertionError('zero weight may not capture')):
            for sigma in (.2,.5,.8):
                expected=reference.step(self.clean,self.low,self.text,schedule(sigma))
                for arm in arms:
                    result=arm.step(self.clean,self.low,self.text,schedule(sigma),teacher_target='unused')
                    self.assertEqual(result['fm'],expected['fm']);self.assertEqual(result['total'],expected['total'])
                    self.assertIsNone(arm.alignment);self.assertIsNone(arm.ema);self.assertEqual(result['align'],0.)
                    self.assertEqual(result['teacher_kind'],arm.teacher_kind)
                    assert_exact_engine(self,reference,arm)
        self.assertFalse(torch.cuda.is_initialized())

    def test_DINO_all_noise_bins_share_FM_streams_and_keep_teacher_and_LR_frozen(self):
        reference=worker.NoRefContinuationStep(tiny_model(),arm='flow_only',training_seed=43)
        engine=worker.HRLatentStep(tiny_model(),weight=.01,teacher_kind='dino',training_seed=43)
        self.assertIsInstance(engine.alignment,HRDINOCosineRegression);self.assertIsNone(engine.ema)
        target_before=self.dino.clone();low_before=self.low.clone();streams=[[],[]];gradients={}
        def trace(original,bucket):
            def call(*args):
                result=original(*args);bucket.append(tuple(v.detach().clone() if isinstance(v,torch.Tensor) else v for v in result));return result
            return call
        step=engine.optimizer.step
        def inspect(*args,**kwargs):
            gradients.update({name:value.grad.detach().clone() for name,value in engine._named_optimizer_parameters if value.grad is not None})
            return step(*args,**kwargs)
        with patch.object(reference.rng,'flow_inputs',side_effect=trace(reference.rng.flow_inputs,streams[0])),\
             patch.object(engine.rng,'flow_inputs',side_effect=trace(engine.rng.flow_inputs,streams[1])),\
             patch.object(engine.optimizer,'step',side_effect=inspect),\
             patch.object(engine.model,'forward',wraps=engine.model.forward) as forward:
            for sigma,bin_name in ((0.,'low'),(.33,'mid'),(.67,'high'),(1.,'high')):
                reference.step(self.clean,self.low,self.text,schedule(sigma))
                result=engine.step(self.clean,self.low,self.text,schedule(sigma),teacher_target=self.dino)
                self.assertEqual(result['noise_bin'],bin_name);self.assertEqual(result['teacher_kind'],'dino')
                self.assertEqual(result['alignment']['selected_tokens'],36)
                self.assertEqual(result['alignment']['target_elements'],36*768)
                self.assertTrue(result['alignment']['all_tokens_supervised']);self.assertFalse(result['teacher_forward'])
                self.assertEqual(result['EMA_updates'],0);self.assertEqual(result['student_layer'],7)
            self.assertEqual(forward.call_count,4)
        for a,b in zip(streams[0],streams[1]):
            for x,y in zip(a,b):
                if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
                else:self.assertEqual(x,y)
        self.assertTrue(any(name.startswith('student_projector.') for name in gradients))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients.values()))
        self.assertTrue(all(p.grad is None for p in engine.parameters));self.assertIsNone(self.dino.grad)
        torch.testing.assert_close(self.dino,target_before,rtol=0,atol=0);torch.testing.assert_close(self.low,low_before,rtol=0,atol=0)
        self.assertEqual(sum(len(block._forward_hooks) for block in engine.model.transformer.blocks),0)

    def test_DINO_target_missing_shape_dtype_gradient_or_finite_errors_stop_before_RNG_and_forward(self):
        engine=worker.HRLatentStep(tiny_model(),weight=.01,teacher_kind='dino')
        invalid=[None,self.dino[:,:35],self.dino[:,:,:767],self.dino.expand(2,-1,-1),self.dino.half(),
            self.dino.clone().requires_grad_(),torch.full_like(self.dino,float('nan')),torch.empty(1,36,768,device='meta')]
        states={n:getattr(engine.rng,n).get_state().clone() for n in ('noise','timestep','mask','corruption')}
        with patch.object(engine.model,'forward',side_effect=AssertionError('invalid target may not forward')):
            for target in invalid:
                with self.subTest(target_type=type(target)),self.assertRaisesRegex(ValueError,'Frozen finite FP32 DINO'):
                    engine.step(self.clean,self.low,self.text,schedule(),teacher_target=target)
                self.assertEqual(engine.steps,0);self.assertFalse(engine.optimizer.state)
                self.assertTrue(all(torch.equal(states[n],getattr(engine.rng,n).get_state()) for n in states))
        self.assertFalse(torch.cuda.is_initialized())

    def test_head_initialization_is_isolated_and_seeded_for_both_representations(self):
        for kind in ('vae','dino'):
            model=tiny_model();state=torch.get_rng_state().clone()
            one=worker.HRLatentStep(model,weight=.01,teacher_kind=kind,training_seed=47)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            two=worker.HRLatentStep(tiny_model(),weight=.1,teacher_kind=kind,training_seed=47)
            for name,value in one.alignment.state_dict().items():torch.testing.assert_close(value,two.alignment.state_dict()[name],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'teacher kind'):worker.HRLatentStep(tiny_model(),weight=.01,teacher_kind='ema')

    def test_config_distinguishes_DINO_cosine_from_VAE_SmoothL1_without_inference_head(self):
        base=source_config();base['training'].update(learning_rate=.001,weight_decay=.01,max_grad_norm=1.)
        protocol={'base':{'config':base},'study':{'updates':2048}}
        vae=worker._hr_config(protocol,.01);dino=worker._hr_config(protocol,.02,kind='dino')
        self.assertEqual(vae['auxiliary_supervision']['loss'],'SmoothL1')
        self.assertEqual(dino['auxiliary_supervision']['loss'],'one_minus_cosine')
        self.assertEqual(dino['auxiliary_supervision']['family'],'dino')
        self.assertEqual(dino['auxiliary_supervision']['head'],'Linear1536to768_FP32')
        self.assertNotIn('beta',dino['auxiliary_supervision']);self.assertFalse(dino['auxiliary_supervision']['EMA'])
        self.assertEqual(dino['training']['variants'],['Base']);self.assertNotIn('restoration_architecture',dino)
        self.assertEqual(checkpoints._check_config(dino),dino['_sha256'])
        for key in ('model','conditioning','data'):self.assertEqual(vae[key],dino[key])

    def test_timer_wrapper_keeps_legacy_call_shape_and_transfers_DINO_target_inside_timing(self):
        class Event:
            def record(self):pass
            def elapsed_time(self,end):return 1.
        events=[]
        class Target:
            def to(self,device):events.append('target_to_device');return self
        target=Target()
        def step(*args,**kwargs):events.append('step');return kwargs
        engine=SimpleNamespace(step=step)
        with patch('torch.cuda.Event',side_effect=lambda **unused:Event()),patch('torch.cuda.synchronize'):
            plain=worker._hr_step(engine,{'clean':torch.zeros(1),'low':torch.zeros(1)},torch.zeros(1),None,'cpu')[0]
            self.assertEqual(plain,{})
            actual=worker._hr_step(engine,{'clean':torch.zeros(1),'low':torch.zeros(1)},torch.zeros(1),None,'cpu',teacher_target=target)[0]
        self.assertIs(actual['teacher_target'],target);self.assertEqual(events,['step','target_to_device','step'])
        self.assertFalse(torch.cuda.is_initialized())


class TinyDINOEngine(TinyHREngine):
    def __init__(self):
        super().__init__(weight=0.)
        self.weight=.01;self.teacher_kind='dino'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.training_seed+5);self.alignment=HRDINOCosineRegression(2,768)
        named=[('restoration.'+n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        named += [('student_projector.'+n,p) for n,p in self.alignment.named_parameters()]
        self._named_optimizer_parameters=tuple(named);self.parameters=[p for _,p in named]
        self.optimizer=torch.optim.AdamW(self.parameters,lr=.002,weight_decay=.01,foreach=False)

    def step(self):
        noise=torch.randn((2,2),generator=self.rng.noise)
        ordinal=torch.randint(0,100,(),generator=self.rng.timestep).float()
        mask=torch.rand((2,2),generator=self.rng.mask)
        corruption=torch.randn((2,2),generator=self.rng.corruption)
        prediction=self.model.lrv_stem(noise+.01*ordinal+.1*mask+.1*corruption)
        aux=self.alignment(prediction[None],torch.ones(1,2,768),(2,1,1))[0]
        loss=prediction.square().mean()+self.weight*aux
        self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True);self.steps+=1
        return loss.detach().clone()


class DINOCheckpointTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def config(self):
        cfg=source_config();cfg['training']['updates']=8
        cfg['auxiliary_supervision']={'family':'dino','teacher_kind':'dino','weight':.01,
            'student_layer':7,'loss':'one_minus_cosine'}
        return canonical(cfg)

    def test_DINO_full_state_roundtrip_and_uninterrupted_trajectory_are_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=self.config();expected=TinyDINOEngine()
            losses=[expected.step() for _ in range(4)]
            interrupted=TinyDINOEngine();actual=[interrupted.step() for _ in range(2)]
            path=root/'dino/Base/checkpoint_000002.pt'
            checkpoints.save_hr_latent_checkpoint(path,interrupted,config,allowed_root=root,protocol_sha256=PROTOCOL,
                cumulative_training_seconds=1.25,initialization=interrupted.base_load_audit)
            payload=torch.load(path,map_location='cpu',weights_only=True);state=payload['training_only']
            self.assertEqual(state['teacher_kind'],'dino');self.assertEqual(state['alignment_spec']['target_dim'],768)
            self.assertIsNone(state['alignment_spec']['latent_channels']);self.assertIsNone(state['alignment_spec']['patch_size'])
            fresh=TinyDINOEngine()
            checkpoints.restore_hr_latent_training_state(fresh,path,allowed_root=root,expected_protocol_sha256=PROTOCOL,expected_config=config)
            self.assertTrue(checkpoints._hr_equal(snapshot(fresh),snapshot(interrupted)))
            actual.extend(fresh.step() for _ in range(2))
            for a,b in zip(actual,losses):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(checkpoints._hr_equal(snapshot(fresh),snapshot(expected)))
            def factory(cfg,variant):return TinyRestoration(),{'official_revision':checkpoints.OFFICIAL_REVISION,'fresh_official_initialization':True,'legacy_state_loaded':False}
            student,audit=checkpoints.load_training_checkpoint(path,device='cpu',allowed_root=root,model_factory=factory)
            self.assertIsNone(student.projector);self.assertFalse(audit['training_only_state_restored'])
            for name,p in student.named_parameters():
                if p.requires_grad:torch.testing.assert_close(p,dict(interrupted.model.named_parameters())[name],rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_teacher_family_rejected_before_any_state_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=self.config();saved=TinyDINOEngine();saved.step()
            path=root/'dino/Base/checkpoint_000001.pt'
            checkpoints.save_hr_latent_checkpoint(path,saved,config,allowed_root=root,protocol_sha256=PROTOCOL,
                cumulative_training_seconds=1.,initialization=saved.base_load_audit)
            payload=torch.load(path,map_location='cpu',weights_only=True);payload['training_only']['teacher_kind']='vae';torch.save(payload,path)
            fresh=TinyDINOEngine();before=snapshot(fresh)
            with self.assertRaisesRegex(ValueError,'Teacher representation differs'):
                checkpoints.restore_hr_latent_training_state(fresh,path,allowed_root=root,expected_protocol_sha256=PROTOCOL,expected_config=config)
            self.assertTrue(checkpoints._hr_equal(before,snapshot(fresh)))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
