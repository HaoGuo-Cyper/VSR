"""CPU refactor parity: tiny Wan, local tensors/checkpoints and fake host inventory."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import WanTransformer3DModel
from vsr.model import ConditionalWan, gradient_norms, make_model, validate_restoration_architecture
from vsr.checkpoints import (load_training_checkpoint, OFFICIAL_REPO_ID, OFFICIAL_REVISION,
                             save_base_checkpoint, gpu_lease)
from vsr.infer import restore_video, sample_tile_latent, tile_positions


def tiny_wan():
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
        in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48,
        num_layers=3, rope_max_seq_len=16,
    )

class ConditionalWanTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        self.noisy = torch.randn(1, 4, 2, 4, 4)
        self.lr = torch.randn_like(self.noisy)
        self.time = torch.tensor([500.0])
        self.text = torch.randn(1, 4, 16)

    def make_model(self, native=None, **kwargs):
        return ConditionalWan(
            native if native is not None else tiny_wan(),
            rank=2, alpha=2, injection_layers=(0, 1), alignment_layer=1,
            stem_channels=8, projector_hidden=32, teacher_dim=16, **kwargs,
        )

    def test_zero_initialization_keeps_native_output_and_base_has_no_projection(self):
        native = tiny_wan().eval()
        original = copy.deepcopy(native)
        model = self.make_model(native, projection=False, gradient_checkpointing=False).eval()
        with torch.no_grad():
            expected = original(self.noisy, self.time, self.text, return_dict=False)[0]
            actual, tokens = model(self.noisy, self.time, self.text, self.lr)
            disabled, _ = model(self.noisy, self.time, self.text, self.lr, enable_lrv=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(disabled, expected, rtol=0, atol=0)
        self.assertIsNone(tokens)
        self.assertEqual(model.token_grid, (2, 2, 2))

    def test_alignment_only_gradients_reach_generator_with_checkpointing(self):
        model = self.make_model(gradient_checkpointing=True).train()
        teacher = F.normalize(torch.randn(1, 8, 16), dim=-1)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        records = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            _, student = model(self.noisy, self.time, self.text, self.lr)
            loss = (1 - (F.normalize(student, dim=-1) * teacher).sum(-1)).mean()
            loss.backward()
            records.append(gradient_norms(model))
            optimizer.step()
            model.clear_features()
        # Stem and LoRA-A can initially have zero gradients due to zero branch
        # maps / LoRA-B. They must receive gradients after the branch updates.
        for group in ("lrv_stem", "lrv_residuals", "lora", "projector"):
            self.assertGreater(records[-1][group]["l2"], 0, group)
            self.assertTrue(records[-1][group]["finite"], group)
        self.assertEqual(records[-1]["frozen"]["elements_with_grad"], 0)
        self.assertIsNone(model._condition_tokens)
        self.assertIsNone(model._captured)

    def test_condition_changes_output_after_updates_and_missing_branch_is_exact(self):
        model = self.make_model(projection=False, gradient_checkpointing=True).train()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            flow, _ = model(self.noisy, self.time, self.text, self.lr)
            flow.square().mean().backward()
            optimizer.step()
            model.clear_features()
        model.eval()
        with torch.no_grad():
            correct, _ = model(self.noisy, self.time, self.text, self.lr)
            shuffled, _ = model(self.noisy, self.time, self.text, self.lr.flip(-1))
            disabled1, _ = model(self.noisy, self.time, self.text, self.lr, enable_lrv=False)
            disabled2, _ = model(self.noisy, self.time, self.text, self.lr * 100, enable_lrv=False)
        self.assertGreater(float((correct - shuffled).abs().max()), 0)
        torch.testing.assert_close(disabled1, disabled2, rtol=0, atol=0)

    def test_checkpoint_replay_preserves_external_lrv_gradient(self):
        direct = self.make_model(gradient_checkpointing=False).train()
        replay = self.make_model(gradient_checkpointing=True).train()
        # Nonzero residuals expose the complete LRV stem gradient path, including
        # the condition tensor captured by block hooks during checkpoint replay.
        with torch.no_grad():
            for layer in direct.lrv_residuals.values():
                layer.weight.normal_(std=0.02)
        replay.load_state_dict(direct.state_dict())
        teacher = F.normalize(torch.randn(1, 8, 16), dim=-1)
        for model in (direct, replay):
            flow, student = model(self.noisy, self.time, self.text, self.lr)
            loss = flow.square().mean() + (1 - (F.normalize(student, dim=-1) * teacher).sum(-1)).mean()
            loss.backward()
        for (name, parameter), (other_name, other) in zip(direct.named_parameters(), replay.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(parameter.grad is None, other.grad is None, name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-7, msg=name)

class CheckpointFakeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.full((2,), 11.0), requires_grad=False)
        self.lora = nn.Parameter(torch.zeros(2, 3))
        self.lrv = nn.Parameter(torch.zeros(3))

def config():
    value = {"seed": 42, "model": {"repo_id": OFFICIAL_REPO_ID, "revision": OFFICIAL_REVISION}, "training": {"variants": ["Base"]}}
    value["_sha256"] = hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
    return value

def make_factory(model=None):
    def factory(cfg, variant):
        return model if model is not None else CheckpointFakeModel(), {"official_revision": OFFICIAL_REVISION, "fresh_official_initialization": True, "legacy_state_loaded": False}
    return factory

def fixture(root):
    path = root / "fresh_run/Base/checkpoint_000012.pt"
    path.parent.mkdir(parents=True)
    payload = {"format": "vsr_training_v1", "step": 12, "config": config(),
               "trainable_state": {"lora": torch.arange(6).reshape(2, 3).float() / 7, "lrv": torch.tensor([-2., 0., 5.])},
               # These values must never be restored into the evaluation model.
               "optimizer": {"do_not_restore": 123}, "noise_rng": torch.zeros(4, dtype=torch.uint8), "index_rng": torch.ones(4, dtype=torch.uint8)}
    torch.save(payload, path)
    (path.parent.parent / "config.json").write_text(json.dumps(payload["config"]))
    return path, payload

class CheckpointTests(unittest.TestCase):

    def test_task_only_float_restore_frozen_unchanged_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, payload = fixture(root)
            model = CheckpointFakeModel()
            original_rng = torch.random.get_rng_state().clone()
            real_load = torch.load
            with patch('vsr.checkpoints.torch.load', wraps=real_load) as loader:
                restored, meta = load_training_checkpoint(path, device='cpu', allowed_root=root, model_factory=make_factory(model))
                loader.assert_called_once_with(path, map_location='cpu', weights_only=True)
            self.assertIs(restored, model)
            self.assertTrue(torch.equal(model.lora, payload['trainable_state']['lora']))
            self.assertTrue(torch.equal(model.lrv, payload['trainable_state']['lrv']))
            self.assertTrue(torch.equal(model.frozen, torch.full((2,), 11.0)))
            self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
            self.assertFalse(model.training)
            self.assertFalse(meta['optimizer_state_restored'])
            self.assertFalse(meta['scheduler_state_restored'])
            self.assertFalse(meta['rng_state_restored'])
            self.assertEqual(meta['task_initialization_origin'], 'new VSR run checkpoint')
            self.assertEqual(meta['checkpoint_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(meta['variant'], 'Base')

    def test_all_keys_shapes_and_dtypes_checked_before_any_copy(self):
        changes = [lambda x: x.pop('lrv'), lambda x: x.update(frozen=torch.zeros(2)), lambda x: x.update(lrv=torch.zeros(4)), lambda x: x.update(lrv=torch.ones(3, dtype=torch.int64)), lambda x: x.update(lrv=torch.full((3,), float('nan'))), lambda x: x.update(lrv=torch.full((3,), 1e+100, dtype=torch.float64))]
        for mutate in changes:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, payload = fixture(root)
                mutate(payload['trainable_state'])
                torch.save(payload, path)
                model = CheckpointFakeModel()
                with self.assertRaises(ValueError):
                    load_training_checkpoint(path, device='cpu', allowed_root=root, model_factory=make_factory(model))
                self.assertTrue(torch.equal(model.lora, torch.zeros(2, 3)))
                self.assertTrue(torch.equal(model.lrv, torch.zeros(3)))

    def test_format_step_pinned_repo_and_config_hash(self):
        mutations = [lambda p: p.update(format='old-format'), lambda p: p.update(step=0), lambda p: p['config']['model'].update(revision='old-finetune'), lambda p: p['config'].update(seed=99)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, payload = fixture(root)
                mutate(payload)
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    load_training_checkpoint(path, device='cpu', allowed_root=root, model_factory=make_factory())

    def test_legacy_and_path_escape_rejected_before_torch_load(self):
        with patch('vsr.checkpoints.torch.load') as loader:
            with self.assertRaisesRegex(ValueError, 'Legacy'):
                load_training_checkpoint('/home/hugo/Projects/ReprVSR-pretrained/outputs/old.pt', device='cpu')
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, _ = fixture(root)
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    load_training_checkpoint(path, device='cpu', model_factory=make_factory())
            loader.assert_not_called()

class PackingVAE(nn.Module):
    """Lossless pixel/time packing with Wan's 8x spatial / 4x temporal grid.

    More latent channels make the fixture invertible. It tests all output
    pixels rather than assuming a real lossy VAE should reproduce bicubic.
    """

    def __init__(self):
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.channels = 3 * 8 * 8 * 4
        self.config = SimpleNamespace(temperal_downsample=[False, True, True], dim_mult=[1, 2, 4, 4], z_dim=self.channels, latents_mean=[0.0] * self.channels, latents_std=[1.0] * self.channels)
        self.encoded = []

    def encode(self, pixels):
        self.encoded.append(pixels.detach().clone())
        batch, colors, frames, height, width = pixels.shape
        assert (frames - 1) % 4 == 0 and height % 8 == 0 and (width % 8 == 0)
        spatial = F.pixel_unshuffle(pixels.permute(0, 2, 1, 3, 4).reshape(batch * frames, colors, height, width), 8)
        spatial = spatial.reshape(batch, frames, 192, height // 8, width // 8)
        first = spatial[:, :1].expand(-1, 4, -1, -1, -1).reshape(batch, 1, self.channels, height // 8, width // 8)
        tail = spatial[:, 1:].reshape(batch, (frames - 1) // 4, self.channels, height // 8, width // 8)
        latent = torch.cat((first, tail), dim=1).permute(0, 2, 1, 3, 4).contiguous()
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, latent):
        batch, channels, compressed_frames, height, width = latent.shape
        chunks = latent.permute(0, 2, 1, 3, 4).reshape(batch, compressed_frames, 4, 192, height, width)
        spatial = torch.cat((chunks[:, 0, :1], chunks[:, 1:].reshape(batch, -1, 192, height, width)), dim=1)
        frames = spatial.shape[1]
        rgb = F.pixel_shuffle(spatial.reshape(batch * frames, 192, height, width), 8)
        return SimpleNamespace(sample=rgb.reshape(batch, frames, 3, height * 8, width * 8).permute(0, 2, 1, 3, 4).contiguous())

class InferenceFakeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(()))
        self.eval_child = nn.Identity().eval()
        self.reference_context = None
        self.feature_clears = 0
        self.reference_clears = 0
        self.calls = []

    def clear_features(self):
        self.feature_clears += 1

    def clear_reference(self):
        self.reference_context = None
        self.reference_clears += 1

    def forward(self, noisy, timestep, text, low, enable_lrv=True, **reference):
        self.reference_context = reference
        self.calls.append({'noise': noisy.clone(), 'low': low.clone(), 'reference': reference, 'grad': torch.is_grad_enabled(), 'training': self.training})
        return (low, None)

class FakeScheduler:

    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps, 0, -1, device=device, dtype=torch.float32)

    def step(self, velocity, timestep, latent, return_dict=False):
        return (velocity,)

class RecordingIdentitySampler:

    def __init__(self):
        self.calls = []

    def __call__(self, model, low, text, config, **kwargs):
        self.calls.append({**kwargs, 'initial_noise': kwargs['initial_noise'].detach().clone(), 'low_shape': tuple(low.shape), 'reference_pointer': kwargs['ref_latent'].data_ptr() if kwargs['ref_valid'] else None})
        return low.detach().cpu()

class InferenceTest(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(2)
        self.model, self.vae = (InferenceFakeModel(), PackingVAE())
        self.text = torch.zeros(1, 2, 4)
        self.config = {'model': {'local_dir': 'local-scheduler-mocked'}}
        self.input = torch.rand((1, 3, 3, 19, 31), generator=torch.Generator().manual_seed(11))

    def test_x4_all_pixels_padding_blending_and_full_temporal_return(self):
        sampler = RecordingIdentitySampler()
        self.model.train()
        self.model.eval_child.eval()
        restored, metadata = restore_video(self.model, self.vae, self.input, self.text, self.config, tile_hw=(32, 48), overlap=16, steps=2, sampler_fn=sampler)
        expected = F.interpolate(self.input[0].permute(1, 0, 2, 3), size=(76, 124), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        expected = expected.permute(1, 0, 2, 3).unsqueeze(0)
        self.assertEqual(tuple(restored.shape), (1, 3, 3, 76, 124))
        self.assertEqual(restored.device.type, 'cpu')
        torch.testing.assert_close(restored, expected, rtol=2e-06, atol=2e-07)
        self.assertEqual(metadata['padding'], {'mode': 'replicate', 'time_end': 2, 'bottom': 4, 'right': 4})
        self.assertEqual(metadata['padded_shape'], [1, 3, 5, 80, 128])
        self.assertFalse(metadata['temporal_windowing'])
        self.assertTrue(metadata['blend']['all_pixels_covered'])
        self.assertGreater(metadata['blend']['minimum_weight_sum'], 0)
        self.assertEqual(metadata['input_frames'], metadata['output_frames'])
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.eval_child.training)
        self.assertTrue(self.vae.training)
        self.assertTrue(all((pixels.shape[2] == 5 for pixels in self.vae.encoded)))
        for pixels in self.vae.encoded:
            torch.testing.assert_close(pixels[:, :, 3], pixels[:, :, 2], rtol=0, atol=0)
            torch.testing.assert_close(pixels[:, :, 4], pixels[:, :, 2], rtol=0, atol=0)
        self.assertTrue(all((call['ref_latent'] is None and call['ref_index'] is None and (call['ref_valid'] is False) for call in sampler.calls)))

    def test_noise_is_global_and_overlaps_are_identical_and_seed_repeatable(self):
        first = RecordingIdentitySampler()
        _, metadata = restore_video(self.model, self.vae, self.input, self.text, self.config, tile_hw=(32, 48), overlap=16, seed=42, sampler_fn=first)
        shape = metadata['sampling']['global_noise_shape']
        global_noise = torch.randn(shape, generator=torch.Generator().manual_seed(42))
        for tile, call in zip(metadata['tiles'], first.calls):
            top, left, height, width = tile['latent_tlhw']
            self.assertTrue(all((value % 16 == 0 for value in tile['hr_tlhw'][:2])))
            expected = global_noise[:, :, :, top:top + height, left:left + width]
            torch.testing.assert_close(call['initial_noise'], expected, rtol=0, atol=0)
        a, b = metadata['tiles'][:2]
        top_a, left_a, height_a, width_a = a['latent_tlhw']
        top_b, left_b, height_b, width_b = b['latent_tlhw']
        overlap_start, overlap_end = (max(left_a, left_b), min(left_a + width_a, left_b + width_b))
        torch.testing.assert_close(first.calls[0]['initial_noise'][..., overlap_start - left_a:overlap_end - left_a], first.calls[1]['initial_noise'][..., overlap_start - left_b:overlap_end - left_b], rtol=0, atol=0)
        repeat = RecordingIdentitySampler()
        restore_video(self.model, self.vae, self.input, self.text, self.config, tile_hw=(32, 48), overlap=16, seed=42, sampler_fn=repeat)
        for a, b in zip(first.calls, repeat.calls):
            torch.testing.assert_close(a['initial_noise'], b['initial_noise'], rtol=0, atol=0)
        changed = RecordingIdentitySampler()
        restore_video(self.model, self.vae, self.input, self.text, self.config, tile_hw=(32, 48), overlap=16, seed=43, sampler_fn=changed)
        self.assertFalse(torch.equal(first.calls[0]['initial_noise'], changed.calls[0]['initial_noise']))

    def test_invalid_shapes_budget_and_reference_do_not_silently_change_input(self):
        for changed in [self.input[:, :2], self.input * 2, self.input + float('nan')]:
            with self.assertRaises(ValueError):
                restore_video(self.model, self.vae, changed, self.text, self.config)
        for kwargs in [{'tile_hw': (31, 48)}, {'overlap': 17}, {'overlap': 256}, {'reference_index': 0}, {'reference_image': torch.zeros(1, 3, 1, 75, 124), 'reference_index': 0}, {'reference_image': torch.zeros(1, 3, 1, 76, 124), 'reference_index': 3}]:
            with self.assertRaises(ValueError):
                restore_video(self.model, self.vae, self.input, self.text, self.config, **kwargs)
        with self.assertRaisesRegex(ValueError, 'exceeding explicit max_frames=2'):
            restore_video(self.model, self.vae, self.input, self.text, {'inference': {'max_frames': 2}})
        self.assertEqual(self.vae.encoded, [])
        self.assertEqual(tile_positions(80, 32, 16), ([0, 16, 32, 48], 32))



class MainlineConstructionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_parameter_names_and_forward_equal_archived_base(self):
        archive = Path(__file__).resolve().parents[1] / 'legacy/pre_refactor_20261106/vsr/model.py'
        spec = importlib.util.spec_from_file_location('_frozen_base_model_for_test', archive)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        native = tiny_wan()
        options = dict(rank=2, alpha=2, injection_layers=(0,1), alignment_layer=1,
                       stem_channels=8, projection=False, gradient_checkpointing=False)
        torch.manual_seed(1234)
        expected = old.ConditionalWan(copy.deepcopy(native), **options).eval()
        torch.manual_seed(1234)
        actual = ConditionalWan(copy.deepcopy(native), **options).eval()
        self.assertEqual(set(expected.state_dict()), set(actual.state_dict()))
        for name,value in expected.state_dict().items():
            torch.testing.assert_close(value,actual.state_dict()[name],rtol=0,atol=0)
        x,low,text = torch.randn(1,4,2,4,4),torch.randn(1,4,2,4,4),torch.randn(1,4,16)
        timestep = torch.tensor([500.])
        with torch.no_grad():
            a,_ = expected(x,timestep,text,low)
            b,_ = actual(x,timestep,text,low)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_factory_has_no_train_dependency_and_exact_fresh_audit(self):
        cfg = {'seed':12,'model':{'local_dir':'mocked','revision':OFFICIAL_REVISION},
               'training':{'variants':['Base'],'gradient_checkpointing':False},
               'conditioning':{'rank':2,'alpha':2,'injection_layers':[0,1],
                   'alignment_layer':1,'stem_channels':8,'projector_hidden':32,'teacher_dim':16}}
        info = {key:[] for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')}
        with patch('diffusers.WanTransformer3DModel.from_pretrained',return_value=(tiny_wan(),info)) as loader:
            model,audit=make_model(cfg,'Base',device='cpu')
        self.assertIsNone(model.projector)
        self.assertFalse(audit['reference_enabled'])
        self.assertTrue(audit['fresh_official_initialization'])
        self.assertEqual(audit['official_revision'],OFFICIAL_REVISION)
        self.assertEqual(loader.call_args.kwargs['dtype'],torch.float32)
        self.assertTrue(loader.call_args.kwargs['local_files_only'])
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaises(ValueError):validate_restoration_architecture(cfg,'Align')
        with self.assertRaises(ValueError):validate_restoration_architecture({**cfg,'reference':{'enabled':True}},'Base')

    def test_new_base_atomic_save_load_and_no_overwrite(self):
        options=dict(rank=2,alpha=2,injection_layers=(0,1),alignment_layer=1,
                     stem_channels=8,projection=False,gradient_checkpointing=False)
        model=ConditionalWan(tiny_wan(),**options)
        cfg={'seed':12,'model':{'repo_id':OFFICIAL_REPO_ID,'revision':OFFICIAL_REVISION},
             'training':{'variants':['Base'],'updates':8192}}
        cfg['_sha256']=hashlib.sha256((json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'new_run/Base/checkpoint_000001.pt'
            audit=save_base_checkpoint(path,model,cfg,step=1,allowed_root=root)
            factory=lambda config,variant:(ConditionalWan(tiny_wan(),**options),{
                'official_revision':OFFICIAL_REVISION,'fresh_official_initialization':True,'legacy_state_loaded':False})
            restored,metadata=load_training_checkpoint(path,allowed_root=root,device='cpu',model_factory=factory)
            self.assertEqual(metadata['checkpoint_sha256'],audit['sha256'])
            for (name,p),(other,q) in zip(model.named_parameters(),restored.named_parameters()):
                self.assertEqual(name,other)
                if p.requires_grad:torch.testing.assert_close(p,q,rtol=0,atol=0)
            with self.assertRaises(ValueError):save_base_checkpoint(path,model,cfg,step=1,allowed_root=root)
            self.assertFalse(list(root.rglob('*.partial')))

    def test_infer_cli_help_and_explicit_execution_guard(self):
        import vsr.infer as infer
        with patch('vsr.infer.load_training_checkpoint') as loader:
            with self.assertRaises(ValueError):infer.run_inference(SimpleNamespace(execute=False))
            loader.assert_not_called()
            with self.assertRaises(SystemExit) as result:infer.build_parser().parse_args(['--help'])
            self.assertEqual(result.exception.code,0)
        args=infer.build_parser().parse_args(['preview','--checkpoint','/tmp/example.pt',
                                             '--allowed-checkpoint-root','/tmp'])
        self.assertEqual(args.command,'preview')

    def test_native_noref_sampler_has_no_teacher_or_reference_inputs(self):
        model=InferenceFakeModel()
        low=torch.zeros(1,4,2,4,4)
        text=torch.zeros(1,2,4)
        noise=torch.randn(low.shape,generator=torch.Generator().manual_seed(2))
        with patch('diffusers.UniPCMultistepScheduler.from_pretrained',return_value=FakeScheduler()):
            sampled=sample_tile_latent(model,low,text,{'model':{'local_dir':'mocked'}},
                steps=2,seed=42,initial_noise=noise,frame_count=5)
        torch.testing.assert_close(sampled,low,rtol=0,atol=0)
        self.assertTrue(all(not row['reference'] for row in model.calls))
        self.assertTrue(all(not row['grad'] for row in model.calls))
        with self.assertRaises(ValueError):
            sample_tile_latent(model,low,text,{'model':{'local_dir':'mocked'}},
                steps=2,seed=42,initial_noise=noise,frame_count=5,ref_valid=True)


class RuntimePathTests(unittest.TestCase):
    def test_override_is_disclosed_without_rewriting_source_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path,payload=fixture(root)
            original=copy.deepcopy(payload['config'])
            received=[]
            def factory(config,variant):
                received.append(copy.deepcopy(config))
                return make_factory()(config,variant)
            _,audit=load_training_checkpoint(path,device='cpu',allowed_root=root,
                model_factory=factory,model_directory=root/'server_models')
            self.assertEqual(audit['config'],original)
            self.assertEqual(audit['config_sha256'],original['_sha256'])
            self.assertEqual(received[0]['model']['local_dir'],str(root/'server_models'))
            self.assertEqual(audit['runtime_model_directory'],str(root/'server_models'))
            self.assertTrue(audit['runtime_model_directory_overridden'])
            self.assertEqual(json.loads((path.parent.parent/'config.json').read_text()),original)

    def test_cuda_inference_requires_lease_before_loading(self):
        import vsr.infer as infer
        with patch('vsr.infer.load_training_checkpoint') as loader:
            with self.assertRaisesRegex(ValueError,'gpu-lock'):
                infer.run_inference(SimpleNamespace(execute=True,device='cuda:0'))
            loader.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())


class GPULeaseTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-fixture'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.admission={'expected_gpu_uuid':'GPU-fixture'}

    def test_empty_inventory_receipt_and_exception_release(self):
        query=mock.Mock(side_effect=[[['0','GPU-fixture']],[],[['0','GPU-fixture']],[]])
        with tempfile.TemporaryDirectory() as directory,patch('vsr.checkpoints._gpu_query',query):
            path=Path(directory)/'gpu.lock'
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with gpu_lease(path,'cuda:0',self.admission) as receipt:
                    self.assertEqual(receipt['inventory'],[])
                    self.assertEqual(receipt['gpu_uuid'],'GPU-fixture')
                    raise RuntimeError('synthetic')
            with gpu_lease(path,'cuda:0',self.admission) as receipt:
                self.assertTrue(receipt['lease_held_through_cleanup'])
            self.assertEqual(query.call_count,4)

    def test_overlap_is_rejected_before_second_gpu_query(self):
        query=mock.Mock(side_effect=[[['0','GPU-fixture']],[]])
        with tempfile.TemporaryDirectory() as directory,patch('vsr.checkpoints._gpu_query',query):
            path=Path(directory)/'gpu.lock'
            with gpu_lease(path,'cuda:0',self.admission):
                with self.assertRaisesRegex(RuntimeError,'already held'):
                    with gpu_lease(path,'cuda:0',self.admission):pass
            self.assertEqual(query.call_count,2)

    def test_uuid_unknown_process_and_duplicate_pid_fail_closed(self):
        for inventory in [[['33','python','100']], [['33','gjs','1'],['33','gjs','1']]]:
            with self.subTest(inventory=inventory),tempfile.TemporaryDirectory() as directory:
                with patch('vsr.checkpoints._gpu_query',side_effect=[[['0','GPU-fixture']],inventory]):
                    with self.assertRaises(ValueError):
                        with gpu_lease(Path(directory)/'gpu.lock','cuda:0',self.admission):pass
        with tempfile.TemporaryDirectory() as directory,patch('vsr.checkpoints._gpu_query',return_value=[['0','GPU-other']]):
            with self.assertRaises(ValueError):
                with gpu_lease(Path(directory)/'gpu.lock','cuda:0',self.admission):pass

    def test_desktop_requires_full_pin_and_memory_bound(self):
        uid=os.getuid()
        actual={'pid':33,'uid':uid,'uids':[uid]*4,'exe':'/usr/bin/gjs-console',
                'exe_sha256':'a'*64,'start_ticks':1234,'exe_inode':5,'exe_device':6,
                'exe_mode':0o755,'exe_owner':0,
                'cgroup':f'0::/user.slice/user-{uid}.slice/user@{uid}.service/session.slice/org.gnome.Shell@ubuntu.service\n'}
        pin={key:actual[key] for key in ('pid','uid','exe','exe_sha256','start_ticks','exe_inode','exe_device','exe_mode')}
        pin.update(process_name='gjs',max_memory_mib=256)
        allowed={'expected_gpu_uuid':'GPU-fixture','desktop_processes':[pin]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gpu.lock'
            with patch('vsr.checkpoints._gpu_query',side_effect=[[['0','GPU-fixture']],[['33','gjs','64']]]),patch('vsr.checkpoints._desktop_identity',return_value=actual):
                with gpu_lease(path,'cuda:0',allowed) as receipt:self.assertEqual(len(receipt['admitted_desktop']),1)
            for changed in [{**pin,'start_ticks':9999},{**pin,'max_memory_mib':257},{key:value for key,value in pin.items() if key!='exe_inode'}]:
                with patch('vsr.checkpoints._gpu_query',side_effect=[[['0','GPU-fixture']],[['33','gjs','64']]]),patch('vsr.checkpoints._desktop_identity',return_value=actual):
                    with self.assertRaises(ValueError):
                        with gpu_lease(path,'cuda:0',{'expected_gpu_uuid':'GPU-fixture','desktop_processes':[changed]}):pass

    def test_verified_file_manager_dbus_endpoints_and_terminal_scope(self):
        uid=os.getuid();prefix=f'/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/'
        for endpoint in ('org.gnome.Nautilus','org.freedesktop.FileManager1'):
            cgroup=prefix+f'app-dbus\\x2d:1.1\\x2d{endpoint}.slice/dbus-:1.1-{endpoint}@1.service'
            self._desktop_lease_case('/usr/bin/nautilus','/usr/bin/nautilus','0::'+cgroup+'\n',True)
        self._desktop_lease_case('/usr/bin/ptyxis','ptyxis',
            '0::'+prefix+r'app-gnome-xdg\x2dterminal\x2dexec-33.scope'+'\n',True)

    def _desktop_lease_case(self,exe,name,cgroup,passed):
        uid=os.getuid()
        actual={'pid':33,'uid':uid,'uids':[uid]*4,'exe':exe,'exe_sha256':'a'*64,
                'start_ticks':1234,'exe_inode':5,'exe_device':6,'exe_mode':0o755,
                'exe_owner':0,'cgroup':cgroup}
        pin={key:actual[key] for key in ('pid','uid','exe','exe_sha256','start_ticks','exe_inode','exe_device','exe_mode')}
        pin.update(process_name=name,max_memory_mib=256)
        with self.subTest(cgroup=cgroup),tempfile.TemporaryDirectory() as directory:
            with patch('vsr.checkpoints._gpu_query',side_effect=[[['0','GPU-fixture']],[['33',name,'40']]]),patch('vsr.checkpoints._desktop_identity',return_value=actual):
                if passed:
                    with gpu_lease(Path(directory)/'gpu.lock','cuda:0',{'expected_gpu_uuid':'GPU-fixture','desktop_processes':[pin]}) as receipt:
                        self.assertEqual(receipt['admitted_desktop'][0]['pid'],33)
                else:
                    with self.assertRaises(ValueError):
                        with gpu_lease(Path(directory)/'gpu.lock','cuda:0',{'expected_gpu_uuid':'GPU-fixture','desktop_processes':[pin]}):pass

    def test_desktop_cgroup_rejects_other_user_bus_endpoint_pid_and_substrings(self):
        uid=os.getuid();prefix=f'/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/'
        valid=prefix+r'app-dbus\x2d:1.1\x2dorg.freedesktop.FileManager1.slice/dbus-:1.1-org.freedesktop.FileManager1@1.service'
        bad=[valid.replace(f'user-{uid}.slice',f'user-{uid+1}.slice'),
             valid.replace('dbus-:1.1-','dbus-:1.2-'),
             valid.replace('dbus-:1.1-org.freedesktop.FileManager1','dbus-:1.1-org.gnome.Nautilus'),
             valid+'/gnome', '/unverified/gnome.scope']
        for path in bad:self._desktop_lease_case('/usr/bin/nautilus','/usr/bin/nautilus','0::'+path+'\n',False)
        self._desktop_lease_case('/usr/bin/nautilus','/usr/bin/nautilus','1:cpu:'+valid+'\n',False)
        self._desktop_lease_case('/usr/bin/nautilus','/usr/bin/nautilus','0::'+valid+'\n0::'+valid+'\n',False)
        self._desktop_lease_case('/usr/bin/ptyxis','ptyxis',
            '0::'+prefix+r'app-gnome-xdg\x2dterminal\x2dexec-34.scope'+'\n',False)

    def test_second_distinct_gnome_shell_is_rejected(self):
        pins=[{'pid':pid,'exe':'/usr/bin/gjs-console'} for pid in (33,34)]
        with tempfile.TemporaryDirectory() as directory:
            with patch('vsr.checkpoints._gpu_query',side_effect=[[['0','GPU-fixture']],[['33','gjs','40'],['34','gjs','40']]]):
                with self.assertRaisesRegex(ValueError,'Only one pinned'):
                    with gpu_lease(Path(directory)/'gpu.lock','cuda:0',{'expected_gpu_uuid':'GPU-fixture','desktop_processes':pins}):pass


if __name__=='__main__':
    unittest.main()
