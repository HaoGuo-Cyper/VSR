"""Explicit local resource audit; already executed once within the72-case check.
Separated from portable repository tests without changing the tested method.
No real encoder forward or GPU operation occurs here.
"""
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from vsr import teacher
from test_videomaev2_primitives import rng,equal_rng
ASSETS=Path('/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_assets_20261108')

class OfficialVideoMAEv2ResourceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_official_binding_strict_load_frozen_RNG_no_forward(self):
        before=rng();binding=teacher.videomaev2_teacher_binding(ASSETS/'VideoREPA_VideoMAEv2_8d581dc.py',ASSETS/'vit_b_k710_dl_from_giant.pth')
        m=teacher.VideoMAEv2Teacher(binding,'cpu');self.assertTrue(equal_rng(before,rng()))
        self.assertEqual(len(m.model.state_dict()),158);self.assertEqual(sum(p.numel() for p in m.parameters()),86225664)
        self.assertTrue(all(not p.requires_grad and p.grad is None and p.dtype==torch.float32 for p in m.parameters()))
        self.assertEqual(tuple(m.model.pos_embed.shape),(1,1792,768));self.assertEqual(teacher._videomaev2_tensor_sha(m.model.pos_embed),binding['position_embedding']['sha256'])
        self.assertFalse(m.training);m.train(True);self.assertFalse(m.training);self.assertFalse(m.model.training)
        self.assertEqual(m.forward_count,0);self.assertEqual(m.parameter_updates,0)
        b=m.binding;b['feature_spec']['grid'][0]=99;self.assertEqual(m.binding['feature_spec']['grid'][0],16)
        self.assertFalse(torch.cuda.is_initialized())
        # Mocked target verifies the wrapper precision/mode/count without a real
        # official encoder forward or consumption of the scientific GPU budget.
        seen=[]
        def target_only(frames):
            seen.append(frames.clone())
            self.assertEqual(frames.dtype,torch.float32)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_autocast_enabled('cpu'))
            return torch.zeros(1,1792,768,dtype=torch.float32)
        hr=torch.full((1,3,33,128,224),.25)
        with patch.object(m.model,'forward_features',side_effect=target_only):
            with torch.autocast('cpu',dtype=torch.bfloat16):out=m.features(hr)
        self.assertEqual(m.forward_count,1);self.assertFalse(out.requires_grad)
        torch.testing.assert_close(seen[0],teacher.videomaev2_preprocess_hr(hr),rtol=0,atol=0)
        with torch.no_grad():next(m.model.parameters()).add_(1)
        with self.assertRaises(ValueError):m.features(hr)
        m.close()
        with self.assertRaises(ValueError):m.features(torch.zeros(1,3,33,128,224))


if __name__=='__main__':unittest.main(verbosity=2)
