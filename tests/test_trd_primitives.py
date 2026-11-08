"""Specific same-Wan TRD regression risks. All fixtures run on CPU."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from vsr.teacher import FP32TrainableTeacher,select_paired_teacher_timestep
from vsr.losses import FullGramTRD
from vsr import train as worker
from test_teacher_losses_train import tiny_model,inputs,TinyRestorer
from test_hr_latent_train import assert_exact_engine

TRDStep=worker.TRDStep


def grid():
    return SimpleNamespace(timesteps=torch.linspace(1000.,1.,1000),
                           sigmas=torch.cat((torch.linspace(1.,.001,1000),torch.zeros(1))))


class RelationTests(unittest.TestCase):
    def test_exact_identity_and_no_default_RNG_consumption(self):
        before=torch.get_rng_state().clone();loss=FullGramTRD(5)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.testing.assert_close(loss.projector.weight,torch.eye(5),rtol=0,atol=0)
        self.assertEqual(int(torch.count_nonzero(loss.projector.bias)),0)

    def test_full_pair_formula_spatial_temporal_weighting_and_teacher_detach(self):
        gen=torch.Generator().manual_seed(3)
        student=torch.randn(2,12,5,generator=gen,requires_grad=True)
        teacher=torch.randn(2,12,5,generator=gen,requires_grad=True)
        alignment=FullGramTRD(5)
        actual,diag=alignment(student,teacher,(3,2,2))
        s=F.normalize(student.float(),dim=-1,eps=1e-6)
        t=F.normalize(teacher.detach().float(),dim=-1,eps=1e-6)
        expected=F.relu((s@s.transpose(1,2)-t@t.transpose(1,2)).abs()-.1).mean()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertAlmostEqual(float(actual.detach()),diag['spatial_loss_mean']/3+2*diag['temporal_loss_mean']/3,places=7)
        self.assertEqual(diag['all_pairs'],288);self.assertEqual(diag['spatial_pairs'],96)
        self.assertEqual(diag['temporal_pairs'],192)
        actual.backward()
        self.assertGreater(float(student.grad.norm()),0.)
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(p.grad is not None and p.grad.norm()>0 for p in alignment.parameters()))

    def test_margin_zero_relation_and_invalid_grid(self):
        x=torch.randn(1,8,4,requires_grad=True);obj=FullGramTRD(4)
        loss,diag=obj(x,x.detach(),(2,2,2));self.assertEqual(float(loss.detach()),0)
        self.assertEqual(diag['active_pairs'],0)
        loss.backward();self.assertEqual(float(x.grad.norm()),0)
        with self.assertRaisesRegex(ValueError,'grid'):obj(x,x.detach(),(2,1,2))
        bad=x.detach().clone();bad[0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):obj(x,bad,(2,2,2))

    def test_FP32_even_inside_bfloat16_autocast(self):
        x=torch.randn(1,8,4,dtype=torch.bfloat16,requires_grad=True)
        t=torch.randn_like(x);obj=FullGramTRD(4)
        with torch.autocast('cpu',dtype=torch.bfloat16):loss,diag=obj(x,t,(2,2,2))
        self.assertEqual(loss.dtype,torch.float32);loss.backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))


class TeacherTests(unittest.TestCase):
    def test_FP32_shadow_frozen_swap_and_no_teacher_gradient(self):
        model=TinyRestorer().to(dtype=torch.bfloat16)
        target=FP32TrainableTeacher(model,mode='frozen')
        shadow=target.state_dict()['shadow']['adapter.weight']
        self.assertEqual(shadow.dtype,torch.float32)
        with torch.no_grad():model.adapter.weight.add_(1)
        before=model.adapter.weight.detach().clone()
        result=target.teacher_forward(model,lambda:model.adapter.weight)
        torch.testing.assert_close(result,shadow.to(torch.bfloat16),rtol=0,atol=0)
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(model.adapter.weight,before,rtol=0,atol=0)
        self.assertEqual(target.num_updates,0)
        with self.assertRaisesRegex(ValueError,'Frozen'):target.update(model)

    def test_EMA_recurrence_state_ownership_and_atomic_invalid_load(self):
        model=TinyRestorer();teacher=FP32TrainableTeacher(model,mode='ema')
        initial=teacher.state_dict()['shadow']['adapter.weight']
        with torch.no_grad():model.adapter.weight.add_(2)
        teacher.update(model)
        expected=initial.mul(.999).add(model.adapter.weight.detach(),alpha=.001)
        torch.testing.assert_close(teacher.state_dict()['shadow']['adapter.weight'],expected,rtol=0,atol=0)
        saved=teacher.state_dict();other=FP32TrainableTeacher(model,mode='ema');other.load_state_dict(saved)
        self.assertEqual(other.num_updates,1)
        bad=copy.deepcopy(saved);bad['num_updates']=9;bad['shadow']['adapter.weight'][0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):other.load_state_dict(bad)
        self.assertEqual(other.num_updates,1)
        torch.testing.assert_close(other.state_dict()['shadow']['adapter.weight'],saved['shadow']['adapter.weight'],rtol=0,atol=0)
        saved['shadow']['adapter.weight'].zero_()
        self.assertGreater(float(other.state_dict()['shadow']['adapter.weight'].norm()),0)

    def test_grid_pairing_half_sigma_ties_and_unpaired_terminal_exclusion(self):
        scheduler=grid();result=select_paired_teacher_timestep(scheduler,200)
        self.assertAlmostEqual(result['teacher_sigma'],.4,places=6)
        result=select_paired_teacher_timestep(scheduler,999)
        self.assertEqual(result['teacher_index'],999);self.assertEqual(result['realized_sigma_ratio'],1)
        scheduler.sigmas[:]=1.;scheduler.sigmas[0]=.5
        scheduler.sigmas[1]=.125;scheduler.sigmas[2]=.375;scheduler.sigmas[3]=.125
        result=select_paired_teacher_timestep(scheduler,0)
        self.assertEqual(result['teacher_index'],1)
        with self.assertRaisesRegex(ValueError,'valid index'):select_paired_teacher_timestep(scheduler,1000)


class StepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.clean,self.low,self.text=inputs()

    def test_zero_weight_three_trajectories_match_native_parameters_optimizer_and_four_RNG(self):
        base=worker.NoRefContinuationStep(tiny_model(),arm='flow_only',training_seed=31)
        engines=[TRDStep(tiny_model(),teacher_kind=kind,weight=0.,training_seed=31) for kind in ('baseline','frozen','ema')]
        heads={engine.teacher_kind:copy.deepcopy(engine.alignment.state_dict()) for engine in engines if engine.alignment is not None}
        with patch.object(TRDStep,'teacher_features',side_effect=AssertionError('zero teacher must not run')):
            for _ in range(3):
                expected=base.step(self.clean,self.low,self.text,grid())
                for engine in engines:
                    actual=engine.step(self.clean,self.low,self.text,grid())
                    self.assertEqual(actual['total'],expected['total']);self.assertFalse(actual['teacher_forward'])
                    assert_exact_engine(self,base,engine)
        self.assertEqual(engines[1].teacher.num_updates,0);self.assertEqual(engines[2].teacher.num_updates,3)
        for engine in engines[1:]:
            for name,value in engine.alignment.state_dict().items():
                torch.testing.assert_close(value,heads[engine.teacher_kind][name],rtol=0,atol=0)
            self.assertTrue(all(p not in engine.optimizer.state for p in engine.alignment.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_forced_objectives_no_RNG_and_exact_FM_shared_epsilon_before_student(self):
        engine=TRDStep(tiny_model(),trd_arm='frozen',weight=.1,normalization_eps=1e-6)
        states={name:getattr(engine.rng,name).get_state().clone() for name in ('noise','timestep','mask','corruption')}
        noise=torch.randn(self.clean.shape,generator=torch.Generator().manual_seed(55))
        calls=[];forward=engine.model.forward
        def traced(noisy,timestep,text,low):
            calls.append((noisy.detach().clone(),timestep.detach().clone(),low.detach().clone(),torch.is_grad_enabled()))
            return forward(noisy,timestep,text,low)
        with patch.object(engine.model,'forward',side_effect=traced):
            fm,aux,diag=engine.objectives(self.clean,self.low,self.text,grid(),index=200,noise=noise)
        self.assertEqual(len(calls),2);self.assertFalse(calls[0][-1]);self.assertTrue(calls[1][-1])
        for call,idx in zip(calls,(diag['teacher_timestep']['teacher_index'],200)):
            sigma=grid().sigmas[idx]
            torch.testing.assert_close(call[0],(1-sigma)*self.clean+sigma*noise,rtol=0,atol=0)
            torch.testing.assert_close(call[2],self.low,rtol=0,atol=0)
        native=tiny_model()
        flow,_=native(calls[1][0],calls[1][1],self.text,self.low.clone())
        torch.testing.assert_close(fm,worker.flow_matching_loss(flow,noise-self.clean),rtol=0,atol=0)
        self.assertTrue(all(torch.equal(state,getattr(engine.rng,name).get_state()) for name,state in states.items()))
        self.assertEqual(engine.steps,0);self.assertFalse(engine.optimizer.state)
        (fm+aux).backward();engine.model.clear_features()
        self.assertTrue(any(p.grad is not None for p in engine.model.parameters() if p.requires_grad))
        self.assertTrue(all(not value.requires_grad for value in engine.teacher.shadow.values()))

    def test_positive_teacher_order_gradient_updates_and_EMA_post_step(self):
        frozen=TRDStep(tiny_model(),teacher_kind='frozen',weight=.1,training_seed=9)
        ema=TRDStep(tiny_model(),teacher_kind='ema',weight=.1,training_seed=9)
        initial=ema.teacher.state_dict()['shadow']
        head={name:p.detach().clone() for name,p in ema.alignment.named_parameters()}
        f=frozen.step(self.clean,self.low,self.text,grid())
        e=ema.step(self.clean,self.low,self.text,grid())
        assert_exact_engine(self,frozen,ema)
        self.assertEqual(f['fm'],e['fm']);self.assertEqual(f['align'],e['align'])
        self.assertEqual(f['teacher_updates'],0);self.assertEqual(e['teacher_updates'],1)
        for name,p in ema.model.named_parameters():
            if p.requires_grad:
                expected=initial[name].mul(.999).add(p.detach(),alpha=.001)
                torch.testing.assert_close(ema.teacher.shadow[name],expected,rtol=0,atol=0)
        self.assertGreater(e['head_total_grad_norm_before_clip'],0)
        self.assertGreater(e['student_total_grad_norm_before_clip'],0)
        self.assertTrue(any(not torch.equal(value,head[name]) for name,value in ema.alignment.named_parameters()))
        self.assertTrue(all(p.grad is None for p in ema.parameters))
        self.assertIsNone(ema.model._condition_tokens);self.assertIsNone(ema.model._captured)
        self.assertEqual(sum(len(block._forward_hooks) for block in ema.model.transformer.blocks),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_loss_failure_restores_modes_teacher_and_no_optimizer_update(self):
        engine=TRDStep(tiny_model(),teacher_kind='ema',weight=.1)
        initial={name:p.detach().clone() for name,p in engine.model.named_parameters()}
        flags=[(module,module.training) for module in engine.model.modules()]
        with patch.object(engine.alignment,'forward',side_effect=RuntimeError('fixture relation failure')):
            with self.assertRaisesRegex(RuntimeError,'fixture relation failure'):engine.step(self.clean,self.low,self.text,grid())
        self.assertEqual(engine.steps,0);self.assertEqual(engine.teacher.num_updates,0)
        self.assertFalse(engine.optimizer.state)
        self.assertTrue(all(torch.equal(value,initial[name]) for name,value in engine.model.named_parameters()))
        self.assertTrue(all(module.training==flag for module,flag in flags))
        self.assertIsNone(engine.model._condition_tokens)


if __name__=='__main__':unittest.main()
