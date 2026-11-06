"""Detached CPU diagnostics and exact tiny-model training-path comparisons."""
import copy
import json
import math
import unittest

import torch

from vsr.method_diagnostics import analyze_feature_snapshot
from vsr.self_distillation import MaskedTemporalAlignment
from vsr.self_distillation_step import NoRefContinuationStep
from tests.test_self_distillation_step import tiny_model,inputs,schedule


def payload(value):
    return {'step':1,'student_layer':9,'teacher_layer':21,'sigma':.8,'selected_tokens':value.shape[0],
        'schedule':{'first_active':True,'every_steps':512},
        'features':{name:value.detach().clone() for name in
            ('student_raw','teacher_raw','student_projected','teacher_normalized')}}


def snapshot_engine(engine):
    parameters=list(engine.model.parameters())+list(engine.alignment.parameters())
    return {'parameters':[(id(p),p._version,p.detach().clone(),None if p.grad is None else p.grad.clone()) for p in parameters],
        'flags':[(id(module),module.training) for parent in (engine.model,engine.alignment) for module in parent.modules()],
        'EMA':engine.ema.state_dict(),'RNG':{key:getattr(engine.rng,key).get_state().clone() for key in ('noise','timestep','mask','corruption')},
        'default_RNG':torch.get_rng_state().clone()}


def assert_tree_equal(test,left,right):
    if isinstance(left,torch.Tensor):test.assertTrue(torch.equal(left,right))
    elif isinstance(left,dict):
        test.assertEqual(set(left),set(right))
        for key in left:assert_tree_equal(test,left[key],right[key])
    elif isinstance(left,(list,tuple)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):assert_tree_equal(test,a,b)
    else:test.assertEqual(left,right)


class FeatureDiagnosticTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_population_std_centered_entropy_rank_and_zero_case(self):
        result=analyze_feature_snapshot(payload(torch.eye(4)))
        for stats in result['statistics'].values():
            self.assertAlmostEqual(stats['rms_population_channel_std'],math.sqrt(3/16))
            self.assertAlmostEqual(stats['normalized_centered_effective_rank'],3.)
            self.assertEqual(stats['maximum_centered_rank'],3)
        result=analyze_feature_snapshot(payload(torch.ones(8,4)))
        self.assertTrue(all(stats['normalized_centered_effective_rank']==0 and stats['rms_population_channel_std']==0
                            for stats in result['statistics'].values()))
        json.dumps(result,allow_nan=False)

    def test_rejects_graph_nonCPU_nonfinite_or_mismatched_snapshot(self):
        base=payload(torch.ones(8,4))
        changes=[lambda x:x['features'].__setitem__('student_raw',torch.ones(8,4,requires_grad=True)),
            lambda x:x['features'].__setitem__('student_raw',torch.ones(8,4,device='meta')),
            lambda x:x['features']['teacher_raw'].__setitem__((0,0),float('nan')),
            lambda x:x.update(selected_tokens=7)]
        for change in changes:
            with self.subTest(change=change):
                item=copy.deepcopy(base);change(item)
                with self.assertRaises(ValueError):analyze_feature_snapshot(item)

    def test_alignment_snapshot_owns_detached_CPU_storage_without_gradient_change(self):
        torch.manual_seed(9)
        first=MaskedTemporalAlignment(4,4,hidden_dim=8);second=copy.deepcopy(first)
        student=torch.randn(1,9,4,requires_grad=True);teacher=torch.randn(1,9,4,requires_grad=True)
        mask=torch.zeros(1,9,dtype=torch.bool);mask[:,3:5]=True
        plain,_=first(student,teacher,mask,(9,1,1),.8)
        captured,diagnostic=second(student,teacher,mask,(9,1,1),.8,capture_diagnostics=True)
        self.assertTrue(torch.equal(plain,captured))
        left=torch.autograd.grad(plain,(student,*first.projector.parameters()),retain_graph=True)
        right=torch.autograd.grad(captured,(student,*second.projector.parameters()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(left,right)))
        raw=diagnostic['feature_snapshot']['student_raw'];saved=raw.clone()
        self.assertTrue(all(value.device.type=='cpu' and value.grad_fn is None and not value.requires_grad
                            for value in diagnostic['feature_snapshot'].values()))
        with torch.no_grad():student.add_(4.)
        self.assertTrue(torch.equal(raw,saved))
        self.assertIsNone(teacher.grad)

    def test_drain_preserves_all_state_and_full_AdamW_trajectory(self):
        clean,low,text=inputs()
        enabled=NoRefContinuationStep(tiny_model(),arm='A2',training_seed=17,hidden_dim=16,
                                      feature_diagnostics_first=True,feature_diagnostics_every=2)
        disabled=NoRefContinuationStep(tiny_model(),arm='A2',training_seed=17,hidden_dim=16,
                                      feature_diagnostics_first=False,feature_diagnostics_every=0)
        for step in range(3):
            result=enabled.step(clean,low,text,schedule())
            control=disabled.step(clean,low,text,schedule())
            for key in ('fm','sd','total','cosine_mean','timestep_index'):self.assertEqual(result[key],control[key])
            self.assertIsInstance(result['cosine_mean'],float)
            self.assertEqual(result['cosine_layers'],{'student':9,'teacher':21})
            self.assertEqual(result['feature_diagnostics_snapshot_collected'],step in (0,1))
            # Drain must preserve even an explicitly supplied pre-existing grad.
            enabled.parameters[-1].grad=torch.ones_like(enabled.parameters[-1])
            before=snapshot_engine(enabled)
            diagnostic=enabled.drain_feature_diagnostics()
            after=snapshot_engine(enabled)
            assert_tree_equal(self,before,after)
            enabled.optimizer.zero_grad(set_to_none=True)
            self.assertEqual(diagnostic is not None,step in (0,1))
            if diagnostic is not None:
                self.assertEqual(diagnostic['selected_tokens'],8)
                json.dumps(diagnostic,allow_nan=False)
            self.assertIsNone(enabled.drain_feature_diagnostics())
            for left,right in zip(enabled.parameters,disabled.parameters):
                self.assertTrue(torch.equal(left,right))
                assert_tree_equal(self,enabled.optimizer.state[left],disabled.optimizer.state[right])
            assert_tree_equal(self,enabled.ema.state_dict(),disabled.ema.state_dict())
        self.assertFalse(torch.cuda.is_initialized())

    def test_first_active_schedule_respects_sigma_window(self):
        clean,low,text=inputs()
        engine=NoRefContinuationStep(tiny_model(),arm='A2',training_seed=17,hidden_dim=16)
        inactive=engine.step(clean,low,text,schedule(.3))
        self.assertIsNone(inactive['cosine_mean']);self.assertFalse(inactive['feature_diagnostics_snapshot_collected'])
        self.assertIsNone(engine.drain_feature_diagnostics())
        active=engine.step(clean,low,text,schedule(.8))
        self.assertTrue(active['feature_diagnostics_snapshot_collected'])
        self.assertEqual(engine.drain_feature_diagnostics()['schedule'],{'first_active':True,'every_steps':512})


if __name__=='__main__':unittest.main()
