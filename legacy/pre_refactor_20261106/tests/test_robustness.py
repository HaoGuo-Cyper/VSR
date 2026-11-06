"""Fixed LR stress protocols on synthetic CPU clips, never benchmark pixels."""
from fractions import Fraction
import shutil
import unittest

import torch

from vsr.robustness import apply_robustness, protocol_spec, fps_fraction, _validate_stream, NOISE_SIGMA


class RobustnessTests(unittest.TestCase):
    def test_identity_preserves_pixels_geometry_and_does_not_alias(self):
        video = torch.rand((1,3,5,16,24), generator=torch.Generator().manual_seed(14))
        result, info = apply_robustness(video, fps=Fraction(24000,1001))
        self.assertTrue(torch.equal(result, video))
        self.assertNotEqual(result.data_ptr(), video.data_ptr())
        self.assertEqual(info['fps'], '24000/1001')
        self.assertFalse(info['real_degradation_fidelity_claimed'])
        self.assertEqual(info['canonical_input_rgb_f32_sha256'], info['output_rgb_f32_sha256'])

    def test_video_constant_blur_uses_one_kernel_and_reflection_has_no_dark_border(self):
        constant = torch.full((1,3,4,16,24), .5)
        result, info = apply_robustness(constant, 'blur', fps=25)
        self.assertTrue(torch.allclose(result, constant, atol=1e-6))
        self.assertEqual(info['protocol']['blur']['sigma_lr_pixels'],1.0)
        self.assertEqual(info['protocol']['blur']['kernel'],7)
        frame = torch.rand((1,3,1,16,24), generator=torch.Generator().manual_seed(9))
        video = frame.expand(-1,-1,4,-1,-1).contiguous()
        first, _ = apply_robustness(video, 'blur', fps=25)
        second, _ = apply_robustness(video, 'blur', fps=25)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[:,:,0], first[:,:,-1]))
        self.assertFalse(torch.equal(first, video))

    def test_noise_seed_reproducible_frame_independent_and_global_rng_untouched(self):
        video = torch.full((1,3,4,64,96),.5)
        state = torch.random.get_rng_state().clone()
        first, info = apply_robustness(video,'noise',fps=29.97,seed=73)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        second, _ = apply_robustness(video,'noise',fps=29.97,seed=73)
        other, _ = apply_robustness(video,'noise',fps=29.97,seed=74)
        self.assertTrue(torch.equal(first,second))
        self.assertFalse(torch.equal(first,other))
        self.assertFalse(torch.equal(first[:,:,0],first[:,:,1]))
        for i in range(4):
            self.assertAlmostEqual(float((first[:,:,i]-.5).std()),NOISE_SIGMA,delta=NOISE_SIGMA*.05)
        self.assertEqual(info['protocol']['noise']['sigma_rgb_01'],5/255)

    def test_invalid_inputs_rates_and_modes_rejected(self):
        invalid=[torch.zeros(3,5,16,24),torch.zeros(1,3,0,16,24),torch.zeros(1,3,5,16,24,dtype=torch.uint8),torch.full((1,3,1,8,8),float('nan')),torch.full((1,3,1,8,8),1.1)]
        for value in invalid:
            with self.assertRaises(ValueError):apply_robustness(value,fps=25)
        for fps in [0,-1,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):fps_fraction(fps)
        self.assertEqual(fps_fraction(29.97002997002997),Fraction(30000,1001))
        self.assertEqual(fps_fraction('24000/1001'),Fraction(24000,1001))
        with self.assertRaises(ValueError):protocol_spec('unknown')
        with self.assertRaises(ValueError):protocol_spec('noise',seed=True)
        with self.assertRaisesRegex(ValueError,'reflection'):apply_robustness(torch.zeros(1,3,2,2,8),'blur',fps=25)
        with self.assertRaisesRegex(ValueError,'even LR'):apply_robustness(torch.zeros(1,3,2,9,8),'compression',fps=25)

    def test_stream_contract_rejects_truncation_resize_fps_or_codec_changes(self):
        correct={'codec_name':'h264','pix_fmt':'yuv420p','height':32,'width':48,'nb_read_frames':'5','avg_frame_rate':'24000/1001','r_frame_rate':'24000/1001'}
        _validate_stream(correct,frames=5,height=32,width=48,fps=Fraction(24000,1001))
        for key,value in [('nb_read_frames','4'),('width',46),('avg_frame_rate','24/1'),('codec_name','hevc'),('pix_fmt','yuv444p')]:
            with self.assertRaises(ValueError):_validate_stream({**correct,key:value},frames=5,height=32,width=48,fps=Fraction(24000,1001))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'CPU FFmpeg/FFprobe not installed')
    def test_real_cpu_h264_synthetic_roundtrip_and_combined_order(self):
        video=torch.rand((1,3,5,32,48),generator=torch.Generator().manual_seed(31))
        result,info=apply_robustness(video,'compression',fps=Fraction(24000,1001))
        repeated,repeated_info=apply_robustness(video,'compression',fps=Fraction(24000,1001))
        self.assertTrue(torch.equal(result,repeated))
        self.assertEqual(info['compression']['encoded_mp4_sha256'],repeated_info['compression']['encoded_mp4_sha256'])
        self.assertEqual(result.shape,video.shape)
        self.assertEqual(result.device.type,'cpu')
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual(info['compression']['stream']['nb_read_frames'],'5')
        self.assertEqual(info['compression']['stream']['avg_frame_rate'],'24000/1001')
        self.assertTrue(info['compression']['geometry_verified'])
        self.assertEqual(info['compression']['stream']['pix_fmt'],'yuv420p')
        combined,combined_info=apply_robustness(video,'combined',fps=25,seed=19)
        self.assertEqual(combined.shape,video.shape)
        self.assertEqual(combined_info['protocol']['sequence'],['blur','noise','compression'])
        self.assertEqual(combined_info['compression']['stream']['avg_frame_rate'],'25/1')
        self.assertFalse(combined_info['parameters_selected_on_test'])


if __name__=='__main__':unittest.main()
