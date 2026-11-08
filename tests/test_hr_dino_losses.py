"""HR DINO cosine and prospective gradient-weight calibration, CPU only."""
import copy
import math
import unittest

import torch
import torch.nn.functional as F
from vsr.losses import (HRDINOCosineRegression,HRLatentRegression,
    shared_gradient_norms,calibrated_auxiliary_weights)


class HRDINOCosineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def identity(self):
        loss=HRDINOCosineRegression(2,2)
        with torch.no_grad():loss.projector.weight.copy_(torch.eye(2));loss.projector.bias.zero_()
        return loss

    def test_single_linear_depth_has_reported_distinct_parameter_counts(self):
        dino=HRDINOCosineRegression();vae=HRLatentRegression()
        self.assertIsInstance(dino.projector,torch.nn.Linear)
        self.assertEqual((dino.projector.in_features,dino.projector.out_features),(1536,768))
        self.assertEqual(sum(p.numel() for p in dino.parameters()),1180416)
        self.assertEqual(sum(p.numel() for p in vae.parameters()),98368)
        self.assertTrue(all(p.dtype==torch.float32 for p in dino.parameters()))
        self.assertFalse(hasattr(dino,'ema'));self.assertFalse(hasattr(dino,'teacher_norm'))

    def test_parallel_orthogonal_opposite_vectors_match_hand_mean_over_all_tokens(self):
        module=self.identity()
        student=torch.tensor([[[1.,0.],[1.,0.],[1.,0.]]],requires_grad=True)
        teacher=torch.tensor([[[1.,0.],[0.,3.],[-2.,0.]]],requires_grad=True)
        loss,diagnostic=module(student,teacher,(3,1,1))
        self.assertEqual(loss.item(),(0.+1.+2.)/3)
        self.assertEqual(diagnostic['selected_tokens'],3);self.assertEqual(diagnostic['target_elements'],6)
        self.assertTrue(diagnostic['all_tokens_supervised']);self.assertFalse(diagnostic['target_requires_grad'])
        self.assertEqual(diagnostic['normalization'],'channel_L2');self.assertEqual(diagnostic['reduction'],'token_mean')
        loss.backward();self.assertIsNone(teacher.grad)
        self.assertTrue(bool(torch.isfinite(student.grad).all()))

    def test_one_minus_cosine_matches_official_REPA_negative_cosine_gradients(self):
        module=self.identity();other=copy.deepcopy(module)
        student=torch.tensor([[[.3,.7],[.9,-.2]]],requires_grad=True)
        cloned=student.detach().clone().requires_grad_();teacher=torch.tensor([[[.8,.2],[.4,.6]]])
        ours=module(student,teacher,(2,1,1))[0]
        official=-(F.normalize(other.projector(cloned),dim=-1)*F.normalize(teacher,dim=-1)).sum(-1).mean()
        self.assertAlmostEqual(float((ours-official).detach()),1.,places=6)
        ours.backward();official.backward()
        torch.testing.assert_close(student.grad,cloned.grad,rtol=0,atol=0)
        for a,b in zip(module.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_positive_scale_invariance_and_no_LayerNorm(self):
        module=self.identity();student=torch.tensor([[[2.,1.]]]);target=torch.tensor([[[4.,3.]]])
        loss=module(student,target,(1,1,1))[0]
        torch.testing.assert_close(module(student*5,target*11,(1,1,1))[0],loss,rtol=1e-6,atol=1e-7)
        expected=1.-torch.dot(F.normalize(student[0,0],dim=-1),F.normalize(target[0,0],dim=-1))
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)

    def test_zero_vectors_have_finite_default_normalize_behavior(self):
        module=self.identity();student=torch.zeros(1,1,2,requires_grad=True);target=torch.zeros(1,1,2)
        loss,_=module(student,target,(1,1,1));self.assertEqual(loss.item(),1.)
        loss.backward();self.assertTrue(bool(torch.isfinite(student.grad).all()))

    def test_target_detached_and_all_student_and_head_gradients_finite(self):
        module=HRDINOCosineRegression(4,3)
        student=torch.randn(2,6,4,requires_grad=True);teacher=torch.randn(2,6,3,requires_grad=True)
        before=teacher.detach().clone();loss,_=module(student,teacher,(3,1,2));loss.backward()
        self.assertIsNone(teacher.grad);torch.testing.assert_close(teacher,before,rtol=0,atol=0)
        self.assertTrue(student.grad is not None and bool(torch.isfinite(student.grad).all()))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters()))

    def test_fp32_arithmetic_under_outer_autocast(self):
        module=HRDINOCosineRegression(4,3)
        student=torch.ones(1,2,4,dtype=torch.bfloat16,requires_grad=True);target=torch.randn(1,2,3)
        with torch.autocast('cpu',dtype=torch.bfloat16):loss,_=module(student,target,(2,1,1))
        expected=(1-(F.normalize(module.projector(student.float()),dim=-1)*F.normalize(target,dim=-1)).sum(-1)).mean()
        self.assertEqual(loss.dtype,torch.float32);torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        loss.backward();self.assertTrue(all(p.grad.dtype==torch.float32 for p in module.parameters()))

    def test_geometry_and_type_validation_rejects_bad_inputs(self):
        module=HRDINOCosineRegression(4,3);student=torch.ones(1,2,4);target=torch.ones(1,2,3)
        for s,t,g in [(student,target,(1,1,1)),(student,target,(2,1,True)),(student,target,None),
                     (student.long(),target,(2,1,1)),(student,target.long(),(2,1,1)),
                     (student[:,:,:3],target,(2,1,1)),(student,target[:,:,:2],(2,1,1)),
                     (student,target.expand(2,-1,-1),(2,1,1)),
                     (torch.full_like(student,float('nan')),target,(2,1,1)),
                     (student,torch.full_like(target,float('inf')),(2,1,1))]:
            with self.subTest(grid=g),self.assertRaises(ValueError):module(s,t,g)
        module.double()
        with self.assertRaisesRegex(ValueError,'FP32'):module(student,target,(2,1,1))
        module.to(device='meta',dtype=torch.float32)
        with self.assertRaisesRegex(ValueError,'devices must match'):module(student,target,(2,1,1))
        for dims in [(0,3),(4,0),(True,3),(4,3.)]:
            with self.subTest(dims=dims),self.assertRaises(ValueError):HRDINOCosineRegression(*dims)


class GradientCalibrationTests(unittest.TestCase):
    def test_global_shared_norms_match_manual_None_zero_and_do_not_touch_grad_or_RNG(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));unused=torch.nn.Parameter(torch.ones(3));head=torch.nn.Parameter(torch.tensor(2.))
        p.grad=torch.tensor([.25,.5]);before=p.grad.clone();rng=torch.get_rng_state().clone()
        fm=p.square().sum();aux=(p*3).sum()*head
        result=shared_gradient_norms(fm,aux,[p,unused])
        self.assertAlmostEqual(result['fm_grad_norm'],math.sqrt(20),places=12)
        self.assertAlmostEqual(result['aux_grad_norm'],math.sqrt(72),places=12)
        self.assertEqual(result['shared_parameter_tensors'],2);self.assertEqual(result['shared_parameter_elements'],5)
        torch.testing.assert_close(p.grad,before,rtol=0,atol=0)
        self.assertIsNone(unused.grad);self.assertIsNone(head.grad)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(result['optimizer_updates'],0)
        self.assertFalse(result['parameter_grad_fields_modified'])

    def test_fixed_sigma_order_median_ratio_and_two_significant_digit_rule(self):
        rows=[{'sigma':sigma,'fm_grad_norm':ratio,'aux_grad_norm':1.} for sigma,ratio in zip((.2,.5,.8),(12.345,1000.,2.))]
        before=copy.deepcopy(rows);result=calibrated_auxiliary_weights(rows)
        self.assertEqual(result['median_FM_to_unit_aux_ratio'],12.345)
        self.assertEqual(result['weights'],[1.2,12.]);self.assertEqual(result['relative_strengths'],[.1,1.])
        self.assertEqual(result['rounding'],'two_significant_digits');self.assertFalse(result['validation_used'])
        self.assertEqual(result['optimizer_updates'],0);self.assertEqual(rows,before)
        self.assertEqual(result['candidates'][0]['unrounded_weight'],1.2345000000000002)

    def test_rescaling_unit_aux_gradient_inversely_rescales_weights(self):
        first=[{'sigma':s,'fm_grad_norm':4.,'aux_grad_norm':2.} for s in (.2,.5,.8)]
        second=[dict(r,aux_grad_norm=4.) for r in first]
        a,b=calibrated_auxiliary_weights(first),calibrated_auxiliary_weights(second)
        self.assertEqual(a['weights'],[.2,2.]);self.assertEqual(b['weights'],[.1,1.])
        self.assertEqual([x['realized_median_relative_strength'] for x in a['candidates']],[.1,1.])

    def test_bad_sigma_order_missing_points_zero_nonfinite_and_underflow_are_rejected(self):
        valid=[{'sigma':s,'fm_grad_norm':1.,'aux_grad_norm':1.} for s in (.2,.5,.8)]
        invalid=[valid[:2],list(reversed(valid)),[dict(r,sigma=.3) for r in valid]]
        for key,value in [('fm_grad_norm',0.),('aux_grad_norm',0.),('aux_grad_norm',float('nan')),
                          ('fm_grad_norm',float('inf')),('fm_grad_norm',True)]:
            changed=copy.deepcopy(valid);changed[0][key]=value;invalid.append(changed)
        invalid.append([{'sigma':s,'fm_grad_norm':1e-300,'aux_grad_norm':1e300} for s in (.2,.5,.8)])
        for rows in invalid:
            with self.subTest(rows=rows),self.assertRaises(ValueError):calibrated_auxiliary_weights(rows)

    def test_invalid_shared_parameters_zero_or_nonfinite_gradients_stop_without_updates(self):
        for kind in ('zero_FM','zero_aux','nonfinite_grad','duplicate','frozen'):
            p=torch.nn.Parameter(torch.tensor(0. if kind=='nonfinite_grad' else 1.));p.grad=torch.tensor(.75)
            before=p.detach().clone()
            fm=p.sqrt() if kind=='nonfinite_grad' else p.square()
            aux=p*0 if kind=='zero_aux' else p*3
            if kind=='zero_FM':fm=p*0
            parameters=[p,p] if kind=='duplicate' else [p]
            if kind=='frozen':p.requires_grad_(False)
            with self.subTest(kind=kind),self.assertRaises(ValueError):shared_gradient_norms(fm,aux,parameters)
            torch.testing.assert_close(p,before,rtol=0,atol=0);self.assertEqual(p.grad.item(),.75)

    def test_nonreentrant_tinyWan_supports_two_gradient_measurements_without_parameter_grads(self):
        from test_teacher_losses_train import tiny_model,inputs
        from vsr.teacher import capture_block_output
        torch.set_num_threads(2)
        model=tiny_model(checkpointing=True);clean,low,text=inputs();head=HRDINOCosineRegression(24,8)
        before={name:p.detach().clone() for name,p in model.named_parameters()}
        flow,features=capture_block_output(model,7,clean,torch.tensor([500.]),text,low)
        aux=head(features,torch.randn(1,36,8),model.token_grid)[0]
        fm=F.mse_loss(flow.float(),torch.zeros_like(flow))
        shared=[p for p in model.parameters() if p.requires_grad]
        result=shared_gradient_norms(fm,aux,shared)
        self.assertGreater(result['fm_grad_norm'],0.);self.assertGreater(result['aux_grad_norm'],0.)
        self.assertTrue(all(p.grad is None for p in list(model.parameters())+list(head.parameters())))
        self.assertTrue(all(torch.equal(p,before[name]) for name,p in model.named_parameters()))
        model.clear_features();self.assertEqual(sum(len(b._forward_hooks) for b in model.transformer.blocks),0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
