"""CPU tiling mechanics using an exactly invertible fake VAE, not VSR scores."""
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F

from vsr.inference import restore_video, sample_tile_latent, tile_positions


class PackingVAE(nn.Module):
    """Lossless pixel/time packing with Wan's 8x spatial / 4x temporal grid.

    More latent channels make the fixture invertible. It tests all output
    pixels rather than assuming a real lossy VAE should reproduce bicubic.
    """
    def __init__(self):
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.channels = 3 * 8 * 8 * 4
        self.config = SimpleNamespace(temperal_downsample=[False, True, True], dim_mult=[1, 2, 4, 4],
            z_dim=self.channels, latents_mean=[0.] * self.channels, latents_std=[1.] * self.channels)
        self.encoded = []

    def encode(self, pixels):
        self.encoded.append(pixels.detach().clone())
        batch, colors, frames, height, width = pixels.shape
        assert (frames - 1) % 4 == 0 and height % 8 == 0 and width % 8 == 0
        spatial = F.pixel_unshuffle(pixels.permute(0, 2, 1, 3, 4).reshape(batch*frames, colors, height, width), 8)
        spatial = spatial.reshape(batch, frames, 192, height//8, width//8)
        first = spatial[:, :1].expand(-1, 4, -1, -1, -1).reshape(batch, 1, self.channels, height//8, width//8)
        tail = spatial[:, 1:].reshape(batch, (frames-1)//4, self.channels, height//8, width//8)
        latent = torch.cat((first, tail), dim=1).permute(0, 2, 1, 3, 4).contiguous()
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, latent):
        batch, channels, compressed_frames, height, width = latent.shape
        chunks = latent.permute(0, 2, 1, 3, 4).reshape(batch, compressed_frames, 4, 192, height, width)
        spatial = torch.cat((chunks[:, 0, :1], chunks[:, 1:].reshape(batch, -1, 192, height, width)), dim=1)
        frames = spatial.shape[1]
        rgb = F.pixel_shuffle(spatial.reshape(batch*frames, 192, height, width), 8)
        return SimpleNamespace(sample=rgb.reshape(batch, frames, 3, height*8, width*8).permute(0, 2, 1, 3, 4).contiguous())


class FakeModel(nn.Module):
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
        self.calls.append({"noise": noisy.clone(), "low": low.clone(), "reference": reference,
                           "grad": torch.is_grad_enabled(), "training": self.training})
        return low, None


class FakeScheduler:
    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps, 0, -1, device=device, dtype=torch.float32)

    def step(self, velocity, timestep, latent, return_dict=False):
        return (velocity,)


class RecordingIdentitySampler:
    def __init__(self):
        self.calls = []

    def __call__(self, model, low, text, config, **kwargs):
        self.calls.append({**kwargs, "initial_noise": kwargs["initial_noise"].detach().clone(),
                           "low_shape": tuple(low.shape), "reference_pointer": kwargs["ref_latent"].data_ptr() if kwargs["ref_valid"] else None})
        # Exact latent identity lets the invertible fixture verify interpolation,
        # padding, coverage and normalization independently of learned quality.
        return low.detach().cpu()


class InferenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model, self.vae = FakeModel(), PackingVAE()
        self.text = torch.zeros(1, 2, 4)
        self.config = {"model": {"local_dir": "local-scheduler-mocked"}}
        self.input = torch.rand((1, 3, 3, 19, 31), generator=torch.Generator().manual_seed(11))

    def test_x4_all_pixels_padding_blending_and_full_temporal_return(self):
        sampler = RecordingIdentitySampler()
        self.model.train()
        self.model.eval_child.eval()
        restored, metadata = restore_video(self.model, self.vae, self.input, self.text, self.config,
                            tile_hw=(32, 48), overlap=16, steps=2, sampler_fn=sampler)
        expected = F.interpolate(self.input[0].permute(1, 0, 2, 3), size=(76, 124),
                                 mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        expected = expected.permute(1, 0, 2, 3).unsqueeze(0)
        self.assertEqual(tuple(restored.shape), (1, 3, 3, 76, 124))
        self.assertEqual(restored.device.type, "cpu")
        torch.testing.assert_close(restored, expected, rtol=2e-6, atol=2e-7)
        self.assertEqual(metadata["padding"], {"mode": "replicate", "time_end": 2, "bottom": 4, "right": 4})
        self.assertEqual(metadata["padded_shape"], [1, 3, 5, 80, 128])
        self.assertFalse(metadata["temporal_windowing"])
        self.assertTrue(metadata["blend"]["all_pixels_covered"])
        self.assertGreater(metadata["blend"]["minimum_weight_sum"], 0)
        self.assertEqual(metadata["input_frames"], metadata["output_frames"])
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.eval_child.training)
        self.assertTrue(self.vae.training)
        # Every VAE input uses the whole padded time span; the repeated tail
        # exactly equals the last actual input frame, without temporal windows.
        self.assertTrue(all(pixels.shape[2] == 5 for pixels in self.vae.encoded))
        for pixels in self.vae.encoded:
            torch.testing.assert_close(pixels[:, :, 3], pixels[:, :, 2], rtol=0, atol=0)
            torch.testing.assert_close(pixels[:, :, 4], pixels[:, :, 2], rtol=0, atol=0)
        self.assertTrue(all(call["ref_latent"] is None and call["ref_index"] is None and call["ref_valid"] is False
                            for call in sampler.calls))

    def test_noise_is_global_and_overlaps_are_identical_and_seed_repeatable(self):
        first = RecordingIdentitySampler()
        _, metadata = restore_video(self.model, self.vae, self.input, self.text, self.config,
                                     tile_hw=(32, 48), overlap=16, seed=42, sampler_fn=first)
        shape = metadata["sampling"]["global_noise_shape"]
        global_noise = torch.randn(shape, generator=torch.Generator().manual_seed(42))
        for tile, call in zip(metadata["tiles"], first.calls):
            top, left, height, width = tile["latent_tlhw"]
            self.assertTrue(all(value % 16 == 0 for value in tile["hr_tlhw"][:2]))
            expected = global_noise[:, :, :, top:top+height, left:left+width]
            torch.testing.assert_close(call["initial_noise"], expected, rtol=0, atol=0)
        # Compare the actual common columns of the first two adjacent tiles.
        a, b = metadata["tiles"][:2]
        top_a, left_a, height_a, width_a = a["latent_tlhw"]
        top_b, left_b, height_b, width_b = b["latent_tlhw"]
        overlap_start, overlap_end = max(left_a, left_b), min(left_a+width_a, left_b+width_b)
        torch.testing.assert_close(first.calls[0]["initial_noise"][..., overlap_start-left_a:overlap_end-left_a],
                                   first.calls[1]["initial_noise"][..., overlap_start-left_b:overlap_end-left_b], rtol=0, atol=0)
        repeat = RecordingIdentitySampler()
        restore_video(self.model, self.vae, self.input, self.text, self.config,
                      tile_hw=(32, 48), overlap=16, seed=42, sampler_fn=repeat)
        for a, b in zip(first.calls, repeat.calls):
            torch.testing.assert_close(a["initial_noise"], b["initial_noise"], rtol=0, atol=0)
        changed = RecordingIdentitySampler()
        restore_video(self.model, self.vae, self.input, self.text, self.config,
                      tile_hw=(32, 48), overlap=16, seed=43, sampler_fn=changed)
        self.assertFalse(torch.equal(first.calls[0]["initial_noise"], changed.calls[0]["initial_noise"]))

    def test_whole_reference_encoded_once_and_shared_with_original_index(self):
        sampler = RecordingIdentitySampler()
        reference = torch.rand((1, 3, 1, 76, 124), generator=torch.Generator().manual_seed(3))
        _, metadata = restore_video(self.model, self.vae, self.input, self.text, self.config,
                    tile_hw=(32, 48), overlap=16, reference_image=reference, reference_index=2, sampler_fn=sampler)
        self.assertEqual(metadata["reference"]["encode_count"], 1)
        self.assertEqual(metadata["reference"]["latent_shape"], [1, self.vae.channels, 1, 10, 16])
        self.assertEqual(len(self.vae.encoded), metadata["tile_count"] + 1)
        self.assertEqual(tuple(self.vae.encoded[0].shape), (1, 3, 1, 80, 128))
        torch.testing.assert_close(self.vae.encoded[0][:, :, :, :76, :124], reference*2-1, rtol=0, atol=0)
        self.assertTrue(all(call["ref_index"] == 2 and call["frame_count"] == 3 and call["ref_valid"] for call in sampler.calls))
        self.assertEqual(len({call["reference_pointer"] for call in sampler.calls}), 1)
        self.assertTrue(all(tuple(call["ref_latent"].shape[-2:]) == (10, 16) for call in sampler.calls))
        self.assertTrue(all(tuple(call["low_shape"][-2:]) == (4, 6) for call in sampler.calls))
        self.assertIsNone(self.model.reference_context)

    def test_native_sampler_passes_reference_every_step_and_off_is_explicit(self):
        low = torch.zeros(1, self.vae.channels, 2, 4, 6)
        noise = torch.randn(low.shape, generator=torch.Generator().manual_seed(1))
        reference = torch.ones(1, self.vae.channels, 1, 10, 16)
        with mock.patch("diffusers.UniPCMultistepScheduler.from_pretrained", side_effect=lambda *a, **k: FakeScheduler()):
            sample_tile_latent(self.model, low, self.text, self.config, steps=3, seed=42, initial_noise=noise,
                               frame_count=3, ref_latent=reference, ref_index=2, ref_valid=True)
            sample_tile_latent(self.model, low, self.text, self.config, steps=2, seed=42, initial_noise=noise,
                               frame_count=3, ref_latent=None, ref_index=None, ref_valid=False)
        torch.testing.assert_close(self.model.calls[0]["noise"], noise, rtol=0, atol=0)
        for call in self.model.calls[:3]:
            self.assertIs(call["reference"]["ref_latent"], reference)
            self.assertEqual(call["reference"]["ref_index"], 2)
            self.assertEqual(call["reference"]["frame_count"], 3)
            self.assertTrue(bool(call["reference"]["ref_valid"][0]))
        for call in self.model.calls[3:]:
            self.assertIsNone(call["reference"]["ref_latent"])
            self.assertIsNone(call["reference"]["ref_index"])
            self.assertFalse(bool(call["reference"]["ref_valid"][0]))
        self.assertIsNone(self.model.reference_context)
        self.assertTrue(all(not call["grad"] for call in self.model.calls))

    def test_invalid_shapes_budget_and_reference_do_not_silently_change_input(self):
        for changed in [self.input[:, :2], self.input*2, self.input+float("nan")]:
            with self.assertRaises(ValueError):
                restore_video(self.model, self.vae, changed, self.text, self.config)
        for kwargs in [{"tile_hw": (31, 48)}, {"overlap": 17}, {"overlap": 256},
                       {"reference_index": 0}, {"reference_image": torch.zeros(1, 3, 1, 75, 124), "reference_index": 0},
                       {"reference_image": torch.zeros(1, 3, 1, 76, 124), "reference_index": 3}]:
            with self.assertRaises(ValueError):
                restore_video(self.model, self.vae, self.input, self.text, self.config, **kwargs)
        with self.assertRaisesRegex(ValueError, "exceeding explicit max_frames=2"):
            restore_video(self.model, self.vae, self.input, self.text, {"inference": {"max_frames": 2}})
        self.assertEqual(self.vae.encoded, [])
        self.assertEqual(tile_positions(80, 32, 16), ([0, 16, 32, 48], 32))

    def test_exception_clears_context_and_restores_mixed_training_flags(self):
        self.model.train()
        self.model.eval_child.eval()
        def failing(*args, **kwargs):
            self.model.reference_context = kwargs["ref_latent"]
            raise RuntimeError("synthetic inference failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            restore_video(self.model, self.vae, self.input, self.text, self.config,
                reference_image=torch.zeros(1, 3, 1, 76, 124), reference_index=0, sampler_fn=failing)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.eval_child.training)
        self.assertTrue(self.vae.training)
        self.assertIsNone(self.model.reference_context)


if __name__ == "__main__":
    unittest.main()
