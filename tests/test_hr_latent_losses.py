"""Frozen HR-target patch ordering and gradient contracts, entirely on CPU."""
import copy
import io
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from vsr.losses import HRLatentRegression,patchify_hr_latent


class HRLatentPatchTests(unittest.TestCase):
    def test_known_indices_match_native_token_and_internal_patch_order(self):
        clean=torch.arange(2*16*3*4*6,dtype=torch.float32).reshape(2,16,3,4,6)
        actual=patchify_hr_latent(clean)
        self.assertEqual(tuple(actual.shape),(2,3*2*3,64))
        expected=[]
        for batch in range(2):
            tokens=[]
            for t in range(3):
                for h in range(2):
                    for w in range(3):
                        tokens.append([clean[batch,c,t,h*2+dh,w*2+dw].item()
                                       for dh in range(2) for dw in range(2) for c in range(16)])
            expected.append(tokens)
        torch.testing.assert_close(actual,torch.tensor(expected),rtol=0,atol=0)
        self.assertEqual(actual[0,0,:16].tolist(),clean[0,:,0,0,0].tolist())
        self.assertEqual(actual[0,0,16:32].tolist(),clean[0,:,0,0,1].tolist())
        self.assertEqual(actual[0,1,:16].tolist(),clean[0,:,0,0,2].tolist())
        self.assertEqual(actual[0,6,:16].tolist(),clean[0,:,1,0,0].tolist())

    def test_patchify_is_exact_inverse_of_native_unpatchify_including_temporal_patch(self):
        # This is the actual native output reshape/permute ordering, not a
        # second invocation of the helper under test.
        for patch_size in ((1,2,2),(2,2,3),(1,1,1)):
            with self.subTest(patch_size=patch_size):
                clean=torch.arange(2*3*4*4*6,dtype=torch.float32).reshape(2,3,4,4,6)
                pt,ph,pw=patch_size;tg,hg,wg=4//pt,4//ph,6//pw
                tokens=patchify_hr_latent(clean,patch_size)
                restored=(tokens.reshape(2,tg,hg,wg,pt,ph,pw,-1)
                          .permute(0,7,1,4,2,5,3,6)
                          .flatten(6,7).flatten(4,5).flatten(2,3))
                torch.testing.assert_close(restored,clean,rtol=0,atol=0)

    def test_protocol33_geometry_has1008_tokens_and64_targets(self):
        clean=torch.zeros(1,16,9,16,28)
        target=patchify_hr_latent(clean)
        self.assertEqual(tuple(target.shape),(1,1008,64))
        self.assertEqual(target.numel(),clean.numel())

    def test_target_is_detached_fp32_without_extra_normalization_or_input_mutation(self):
        clean=torch.full((1,16,3,2,2),7.,dtype=torch.bfloat16,requires_grad=True)
        before=clean.detach().clone();target=patchify_hr_latent(clean)
        self.assertFalse(target.requires_grad);self.assertIsNone(target.grad_fn)
        self.assertEqual(target.dtype,torch.float32)
        self.assertTrue(bool((target==7.).all()))
        torch.testing.assert_close(clean,before,rtol=0,atol=0)
        self.assertIsNone(clean.grad)

    def test_noncontiguous_input_obeys_same_order(self):
        clean=torch.arange(16*3*4*6,dtype=torch.float32).reshape(1,16,3,4,6).transpose(3,4)
        self.assertFalse(clean.is_contiguous())
        torch.testing.assert_close(patchify_hr_latent(clean),patchify_hr_latent(clean.contiguous()),rtol=0,atol=0)

    def test_bad_clean_shapes_types_and_values_reject(self):
        invalid=[None,torch.zeros(1,16,2,2),torch.zeros(1,16,3,2,2,dtype=torch.int64),
                 torch.zeros(0,16,3,2,2),torch.zeros(1,0,3,2,2),torch.zeros(1,16,0,2,2),
                 torch.full((1,16,3,2,2),float('nan')),torch.full((1,16,3,2,2),float('inf'))]
        for value in invalid:
            with self.subTest(value=type(value)),self.assertRaises(ValueError):patchify_hr_latent(value)

    def test_patch_size_and_divisibility_are_strict(self):
        clean=torch.zeros(1,16,3,4,6)
        for size in (None,2,'122',(),(1,2),(1,2,2,2),(0,2,2),(1,-2,2),(True,2,2),(1,2.,2)):
            with self.subTest(size=size),self.assertRaises(ValueError):patchify_hr_latent(clean,size)
        for size in ((2,2,2),(1,3,2),(1,2,4)):
            with self.subTest(size=size),self.assertRaisesRegex(ValueError,'divisible'):patchify_hr_latent(clean,size)


class HRLatentRegressionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_single_linear_production_head_has98368_parameters(self):
        module=HRLatentRegression()
        self.assertIsInstance(module.projector,torch.nn.Linear)
        self.assertEqual((module.projector.in_features,module.projector.out_features),(1536,64))
        self.assertEqual(sum(p.numel() for p in module.parameters()),98368)
        self.assertTrue(all(p.dtype==torch.float32 and p.requires_grad for p in module.parameters()))
        self.assertFalse(hasattr(module,'ema'));self.assertFalse(hasattr(module,'teacher_norm'))

    def test_all_elements_mean_matches_hand_value_including_first_time(self):
        module=HRLatentRegression(2,1,(1,1,1))
        with torch.no_grad():module.projector.weight.zero_();module.projector.bias.zero_()
        clean=torch.tensor([2.,.5,1.,0.]).reshape(1,1,2,1,2)
        student=torch.zeros(1,4,2,requires_grad=True)
        loss,diagnostic=module(student,clean,(2,1,2))
        self.assertEqual(loss.dtype,torch.float32)
        self.assertEqual(loss.item(),(1.5+.125+.5+0.)/4)
        self.assertEqual(diagnostic['selected_tokens'],4)
        self.assertEqual(diagnostic['target_elements'],4)
        self.assertEqual(diagnostic['reduction'],'element_mean')
        self.assertEqual(diagnostic['beta'],1.)
        self.assertTrue(diagnostic['all_tokens_supervised'])
        self.assertFalse(diagnostic['target_requires_grad'])

    def test_gradients_reach_student_and_head_but_never_target(self):
        module=HRLatentRegression(3,2,(1,2,2))
        with torch.no_grad():module.projector.weight.fill_(.1);module.projector.bias.fill_(.2)
        student=torch.ones(2,4,3,requires_grad=True)
        clean=torch.zeros(2,2,2,2,4,requires_grad=True)
        before=clean.detach().clone()
        loss,_=module(student,clean,(2,1,2));loss.backward()
        self.assertTrue(student.grad is not None and bool(torch.isfinite(student.grad).all()) and bool((student.grad!=0).all()))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).all()) for p in module.parameters()))
        self.assertIsNone(clean.grad)
        torch.testing.assert_close(clean,before,rtol=0,atol=0)

    def test_fp32_arithmetic_survives_outer_bfloat16_autocast(self):
        module=HRLatentRegression(4,2,(1,2,2))
        student=torch.ones(1,2,4,dtype=torch.bfloat16,requires_grad=True)
        clean=torch.ones(1,2,2,2,2)
        with torch.autocast('cpu',dtype=torch.bfloat16):loss,_=module(student,clean,(2,1,1))
        expected=F.smooth_l1_loss(module.projector(student.float()),patchify_hr_latent(clean),beta=1.,reduction='mean')
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        self.assertEqual(loss.dtype,torch.float32);loss.backward()
        self.assertTrue(all(p.grad.dtype==torch.float32 for p in module.parameters()))
        self.assertTrue(bool(torch.isfinite(student.grad).all()))

    def test_head_state_is_separate_and_exactly_roundtrips(self):
        module=HRLatentRegression(4,2,(1,2,2));clone=HRLatentRegression(4,2,(1,2,2))
        student=torch.ones(1,2,4);clean=torch.zeros(1,2,2,2,2)
        expected=module(student,clean,(2,1,1))[0]
        file=io.BytesIO();torch.save(module.state_dict(),file);file.seek(0)
        clone.load_state_dict(torch.load(file,weights_only=True),strict=True)
        torch.testing.assert_close(clone(student,clean,(2,1,1))[0],expected,rtol=0,atol=0)
        self.assertEqual(set(module.state_dict()),{'projector.weight','projector.bias'})
        self.assertFalse(torch.cuda.is_initialized())

    def test_feature_target_grid_batch_channel_and_dtype_errors_reject(self):
        module=HRLatentRegression(4,2,(1,2,2));student=torch.zeros(1,2,4);clean=torch.zeros(1,2,2,2,2)
        cases=[(student,clean,(1,1,2)),(student,clean,(2,1,1.)),(student,clean,(True,1,2)),
               (student,clean,(2,1)),(student,clean,(0,1,2)),(student,clean,None),
               (torch.zeros(1,2,3),clean,(2,1,1)),(torch.zeros(2,2,4),clean,(2,1,1)),
               (torch.zeros(1,1,4),clean,(2,1,1)),(torch.zeros(1,2,4,dtype=torch.int64),clean,(2,1,1)),
               (student,torch.zeros(1,3,2,2,2),(2,1,1)),(student,torch.zeros(1,2,2,3,2),(2,1,1)),
               (torch.full_like(student,float('nan')),clean,(2,1,1))]
        for s,t,g in cases:
            with self.subTest(grid=g),self.assertRaises(ValueError):module(s,t,g)
        module.double()
        with self.assertRaisesRegex(ValueError,'FP32'):module(student,clean,(2,1,1))
        module.to(device='meta',dtype=torch.float32)
        with self.assertRaisesRegex(ValueError,'devices must match'):module(student,clean,(2,1,1))

    def test_constructor_and_nonfinite_projector_are_rejected(self):
        for student,channels in ((0,16),(1,0),(True,16),(2.,16),(2,False)):
            with self.subTest(student=student,channels=channels),self.assertRaises(ValueError):HRLatentRegression(student,channels)
        module=HRLatentRegression(2,1,(1,1,1))
        with torch.no_grad():module.projector.weight.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite HR latent'):module(torch.ones(1,1,2),torch.zeros(1,1,1,1,1),(1,1,1))


class HRLatentWanCaptureTests(unittest.TestCase):
    def test_block7_capture_survives_nonreentrant_replay_without_a_second_model_forward(self):
        from test_teacher_losses_train import tiny_model,inputs
        from vsr.teacher import capture_block_output
        torch.set_num_threads(2)
        model=tiny_model(checkpointing=True);clean,low,text=inputs()
        self.assertEqual(len(model.transformer.blocks),30)
        head=HRLatentRegression(24)
        frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        with patch.object(model,'forward',wraps=model.forward) as forward:
            flow,student=capture_block_output(model,7,clean,torch.tensor([500.]),text,low)
            self.assertEqual(tuple(student.shape),(1,36,24));self.assertTrue(student.requires_grad)
            self.assertEqual(model.token_grid,(9,2,2))
            self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)
            auxiliary,diag=head(student,clean,model.token_grid)
            total=F.mse_loss(flow.float(),torch.zeros_like(flow))+auxiliary
            total.backward()
            forward.assert_called_once()
        self.assertEqual(diag['selected_tokens'],36)
        self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))
        self.assertTrue(all(torch.equal(p,frozen[name]) for name,p in model.named_parameters() if not p.requires_grad))
        model.clear_features()
        self.assertIsNone(model._condition_tokens);self.assertIsNone(model._captured)
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_auxiliary_only_reaches_blocks_through7_and_not_later_LR_injection9(self):
        from test_teacher_losses_train import tiny_model,inputs
        from vsr.teacher import capture_block_output
        torch.set_num_threads(2)
        model=tiny_model(checkpointing=True);clean,low,text=inputs();head=HRLatentRegression(24)
        _,student=capture_block_output(model,7,clean,torch.tensor([500.]),text,low)
        head(student,clean,model.token_grid)[0].backward()
        self.assertTrue(any(p.grad is not None for p in model.lrv_residuals['0'].parameters()))
        self.assertTrue(any(p.grad is not None for p in model.lrv_residuals['5'].parameters()))
        self.assertTrue(all(p.grad is None for p in model.lrv_residuals['9'].parameters()))
        model.clear_features()
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
