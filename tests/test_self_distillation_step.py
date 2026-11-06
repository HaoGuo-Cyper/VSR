"""Real ConditionalWan/tiny random Wan CPU fixtures; no experimental training.

Small synthetic AdamW trajectories are allowed integration tests. No official
weights, task checkpoint payload, video, GPU or production optimizer is loaded.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from diffusers import WanTransformer3DModel

from vsr.model import ConditionalWan
from vsr.self_distillation_step import (NoRefContinuationStep, MatchedStepRNG,
    capture_block_output, load_matched_base_continuation)


def tiny_model(seed=123, checkpointing=True):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        native = WanTransformer3DModel(patch_size=(1,2,2), num_attention_heads=2,
            attention_head_dim=12, in_channels=16, out_channels=16, text_dim=16,
            freq_dim=16, ffn_dim=48, num_layers=30, rope_max_seq_len=16)
        model = ConditionalWan(native, rank=2, alpha=2, injection_layers=(0,5,9),
            alignment_layer=9, stem_channels=8, projection=False,
            gradient_checkpointing=checkpointing)
        # A trained Base has an effective LR branch. Give this synthetic fixture
        # nonzero residuals so the corruption checks cannot pass vacuously.
        with torch.no_grad():
            for layer in model.lrv_residuals.values():
                layer.weight.normal_(std=.02)
    return model.train()


def inputs():
    generator = torch.Generator().manual_seed(97)
    return (torch.randn(1,16,9,4,4,generator=generator),
            torch.randn(1,16,9,4,4,generator=generator),
            torch.randn(1,4,16,generator=generator))


def schedule(sigma=.8):
    return SimpleNamespace(timesteps=torch.linspace(999.,0.,1000),
                           sigmas=torch.full((1001,),sigma))


def named_values(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


class WanIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.clean, self.low, self.text = inputs()

    def test_equal_EMA_no_corruption_matches_at_the_same_layer(self):
        model = tiny_model()
        engine = NoRefContinuationStep(model, arm="A2", training_seed=5, hidden_dim=16)
        time = torch.tensor([500.])
        teacher9 = engine.teacher_features(self.clean,time,self.text,self.low,layer=9)
        _, student9 = capture_block_output(model,9,self.clean,time,self.text,self.low)
        torch.testing.assert_close(teacher9,student9,rtol=0,atol=0)
        self.assertFalse(teacher9.requires_grad)
        self.assertTrue(student9.requires_grad)
        # Different blocks are intentionally not asserted equal.
        model.clear_features()
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_teacher_first_same_noisy_HR_sigma_and_only_LR_asymmetry(self):
        model = tiny_model()
        engine = NoRefContinuationStep(model,arm="A2",training_seed=5,hidden_dim=16)
        frozen = {name:value for name,value in named_values(model).items()
                  if not dict(model.named_parameters())[name].requires_grad}
        flags = [(module,module.training) for module in model.modules()]
        before_low = self.low.clone()
        events=[];forward=model.forward
        def spy(noisy,timestep,text,low,*args,**kwargs):
            events.append({"grad":torch.is_grad_enabled(),"training":model.training,
                           "noisy":noisy.detach().clone(),"timestep":timestep.detach().clone(),
                           "low":low.detach().clone()})
            return forward(noisy,timestep,text,low,*args,**kwargs)
        with patch.object(model,"forward",side_effect=spy):
            result=engine.step(self.clean,self.low,self.text,schedule())
        self.assertEqual(len(events),2)
        self.assertEqual([event['grad'] for event in events],[False,True])
        self.assertEqual([event['training'] for event in events],[False,True])
        self.assertTrue(torch.equal(events[0]['noisy'],events[1]['noisy']))
        self.assertTrue(torch.equal(events[0]['timestep'],events[1]['timestep']))
        self.assertTrue(torch.equal(events[0]['low'],before_low))
        self.assertTrue(torch.equal(events[1]['low'][:,:,0],before_low[:,:,0]))
        self.assertFalse(torch.equal(events[1]['low'],before_low))
        self.assertTrue(torch.equal(self.low,before_low))
        self.assertTrue(result['teacher_forward'])
        self.assertEqual(result['selected_tokens'],8)  # 2 masked times × 2 × 2
        self.assertEqual(result['EMA_updates'],1)
        self.assertEqual((result['student_layer'],result['teacher_layer']),(9,21))
        self.assertTrue(all(torch.equal(parameter,frozen[name]) for name,parameter in model.named_parameters() if not parameter.requires_grad))
        self.assertTrue(all(module.training==training for module,training in flags))
        self.assertTrue(all(parameter.grad is None for parameter in engine.parameters))
        self.assertIsNone(model._condition_tokens);self.assertIsNone(model._captured)
        self.assertTrue(all('projector' not in name for name in engine.ema.names))
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)

    def test_rho_zero_matches_core_Base_full_tiny_AdamW_trajectory(self):
        reference=tiny_model()
        arms=[NoRefContinuationStep(tiny_model(),arm=arm,training_seed=19,rho=0.,hidden_dim=16,
                                    learning_rate=.002) for arm in ('flow_only','C1','A2')]
        params=[parameter for parameter in reference.parameters() if parameter.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.002,weight_decay=.01,foreach=False)
        streams=MatchedStepRNG(19,'cpu')
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            noisy,target,time,_,index=streams.flow_inputs(self.clean,schedule())
            flow,_=reference(noisy,time,self.text,self.low)
            loss=torch.nn.functional.mse_loss(flow.float(),target)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();reference.clear_features();optimizer.zero_grad(set_to_none=True)
            for arm in arms:
                result=arm.step(self.clean,self.low,self.text,schedule())
                self.assertEqual(result['fm'],float(loss.detach()))
                self.assertEqual(result['timestep_index'],index)
                self.assertFalse(result['teacher_forward'])
                self.assertEqual(result['sd'],0.)
                for name,parameter in reference.named_parameters():
                    other=dict(arm.model.named_parameters())[name]
                    torch.testing.assert_close(parameter,other,rtol=0,atol=0,msg=f'{arm.arm}/{step}/{name}')
                    if parameter.requires_grad:
                        self.assertEqual(set(optimizer.state[parameter]),set(arm.optimizer.state[other]))
                        for key in optimizer.state[parameter]:
                            torch.testing.assert_close(optimizer.state[parameter][key],arm.optimizer.state[other][key],rtol=0,atol=0)
            self.assertEqual(arms[0].data_index(2),(step+1)%2)
        self.assertTrue(all(parameter.grad is None for arm in arms for parameter in arm.parameters))

    def test_gamma_zero_corrupted_A2_matches_C1_not_clean_flow(self):
        control=NoRefContinuationStep(tiny_model(),arm='C1',training_seed=11,learning_rate=.002,hidden_dim=16)
        disabled=NoRefContinuationStep(tiny_model(),arm='A2',training_seed=11,gamma=0.,learning_rate=.002,hidden_dim=16)
        clean=NoRefContinuationStep(tiny_model(),arm='flow_only',training_seed=11,learning_rate=.002,hidden_dim=16)
        for _ in range(2):
            c1=control.step(self.clean,self.low,self.text,schedule())
            a2=disabled.step(self.clean,self.low,self.text,schedule())
            clean.step(self.clean,self.low,self.text,schedule())
            self.assertEqual(c1['fm'],a2['fm'])
            self.assertFalse(a2['teacher_forward'])
            for name,parameter in control.model.named_parameters():
                torch.testing.assert_close(parameter,dict(disabled.model.named_parameters())[name],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(parameter,dict(clean.model.named_parameters())[name])
                            for name,parameter in control.model.named_parameters() if parameter.requires_grad))

    def test_teacher_exception_cleans_hooks_restores_student_and_never_steps_optimizer(self):
        model=tiny_model().eval()
        engine=NoRefContinuationStep(model,arm='A2',training_seed=5,hidden_dim=16)
        before=named_values(model)
        def fail(_module,_args,_output):
            if not torch.is_grad_enabled():raise RuntimeError('teacher fixture failure')
        handle=model.transformer.blocks[21].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'teacher fixture failure'):
                engine.step(self.clean,self.low,self.text,schedule())
        finally:handle.remove()
        self.assertEqual(engine.steps,0);self.assertEqual(len(engine.optimizer.state),0)
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(parameter,before[name]) for name,parameter in model.named_parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in engine.parameters))
        self.assertIsNone(model._condition_tokens)
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)

    def test_late_nonfinite_gradients_reject_before_clipping_or_any_update(self):
        for location, invalid in (("restoration", float("nan")), ("student_projector", float("inf"))):
            with self.subTest(location=location):
                model = tiny_model().eval()
                engine = NoRefContinuationStep(model, arm="A2", training_seed=5, hidden_dim=16)
                name, target = [(name, p) for name, p in engine._named_optimizer_parameters
                                if name.startswith(location + ".")][-1]
                before = named_values(model)
                projector_before = named_values(engine.alignment.projector)
                ema_before = engine.ema.state_dict()
                frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
                handle = target.register_hook(lambda gradient: torch.full_like(gradient, invalid))
                try:
                    with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
                        with patch.object(engine.optimizer, "step", wraps=engine.optimizer.step) as update:
                            with self.assertRaisesRegex(ValueError, "Nonfinite continuation gradients: " + name):
                                engine.step(self.clean, self.low, self.text, schedule())
                        update.assert_not_called()
                    clip.assert_not_called()
                finally:
                    handle.remove()
                self.assertEqual(engine.steps, 0)
                self.assertEqual(len(engine.optimizer.state), 0)
                self.assertEqual(engine.ema.num_updates, 0)
                self.assertFalse(model.training)
                self.assertTrue(all(torch.equal(p, before[name]) for name, p in model.named_parameters()))
                self.assertTrue(all(torch.equal(p, projector_before[name]) for name, p in engine.alignment.projector.named_parameters()))
                self.assertTrue(all(torch.equal(value, ema_before["shadow"][name])
                                    for name, value in engine.ema.state_dict()["shadow"].items()))
                self.assertEqual(frozen_versions, {name: p._version for name, p in model.named_parameters() if not p.requires_grad})
                self.assertTrue(all(p.grad is None for p in engine.parameters))
                self.assertIsNone(model._condition_tokens)
                self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks), 0)

    def test_disabled_auxiliary_rejects_late_shadow_before_optimizer_mutation(self):
        model = tiny_model().eval()
        engine = NoRefContinuationStep(model, arm="A2", training_seed=5, gamma=0., hidden_dim=16)
        last = engine.ema.names[-1]
        engine.ema._shadow[last].view(-1)[-1] = float("inf")
        before = named_values(model)
        versions = {name: p._version for name, p in model.named_parameters()}
        with patch.object(model, "forward", wraps=model.forward) as forward:
            with self.assertRaisesRegex(ValueError, "Invalid EMA shadow: " + last):
                engine.step(self.clean, self.low, self.text, schedule())
        forward.assert_not_called()
        self.assertEqual(engine.steps, 0)
        self.assertEqual(len(engine.optimizer.state), 0)
        self.assertEqual(engine.ema.num_updates, 0)
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in model.named_parameters()))
        self.assertEqual(versions, {name: p._version for name, p in model.named_parameters()})

    def test_sigma_window_skips_teacher_without_altering_flags(self):
        model=tiny_model()
        engine=NoRefContinuationStep(model,arm='A2',training_seed=5,hidden_dim=16)
        before=(torch.is_grad_enabled(),torch.are_deterministic_algorithms_enabled(),
                torch.backends.cuda.matmul.allow_tf32,torch.cuda.is_initialized())
        result=engine.step(self.clean,self.low,self.text,schedule(sigma=.3))
        self.assertFalse(result['teacher_forward'])
        self.assertEqual(result['selected_tokens'],0)
        self.assertEqual(before,(torch.is_grad_enabled(),torch.are_deterministic_algorithms_enabled(),
                                torch.backends.cuda.matmul.allow_tf32,torch.cuda.is_initialized()))

    def test_future_loader_requires_matched_Base_and_fresh_optimizer(self):
        model=tiny_model()
        config={'seed':20261102,'training':{'seed':20261103,'learning_rate':1e-4,'weight_decay':.01,'max_grad_norm':1.},
                'data':{'num_frames':33,'temporal_stride':1,'height':128,'width':224,'scale':4}}
        audit={'checkpoint_sha256':'a'*64,'step':8192,'variant':'Base','config':config,
               'optimizer_state_restored':False,'scheduler_state_restored':False,'rng_state_restored':False}
        with patch('vsr.checkpoints.load_training_checkpoint',return_value=(model,audit)) as loader:
            arm=load_matched_base_continuation('/new/runs/base/Base/checkpoint_008192.pt',expected_checkpoint_sha256='a'*64,
                expected_training_seed=20261103,allowed_root='/new/runs',arm='A2',hidden_dim=16)
            loader.assert_called_once_with('/new/runs/base/Base/checkpoint_008192.pt',device='cpu',allowed_root=Path('/new/runs'),variant='Base')
        self.assertEqual(len(arm.optimizer.state),0)
        self.assertEqual(arm.ema.num_updates,0)
        self.assertEqual(arm.base_load_audit,audit)
        with patch('vsr.checkpoints.load_training_checkpoint',return_value=(tiny_model(),audit)):
            with self.assertRaisesRegex(ValueError,'seed/protocol'):
                load_matched_base_continuation('unused',expected_checkpoint_sha256='a'*64,expected_training_seed=20261104,
                                              allowed_root='/new/runs',arm='flow_only')


if __name__=='__main__':
    unittest.main()
