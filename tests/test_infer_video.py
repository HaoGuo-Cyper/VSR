"""CPU parser/decoder and mocked user-video pipeline checks; no GPU run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

from vsr.checkpoints import OFFICIAL_REPO_ID, OFFICIAL_REVISION, PRODUCTION_RUNS_ROOT
from vsr.infer_video import (build_parser, decode_lrv, preview_training_config,
                             read_hri, run_inference)


class FakeCapture:
    def __init__(self, frames, fps=23.976, reported=None):
        self.frames = frames
        self.fps = fps
        self.reported = len(frames) if reported is None else reported
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        import cv2
        return self.fps if key == cv2.CAP_PROP_FPS else self.reported

    def read(self):
        if self.index == len(self.frames):
            return False, None
        result = self.frames[self.index]
        self.index += 1
        return True, result.copy()

    def release(self):
        self.released = True


class InferVideoTests(unittest.TestCase):
    def test_parser_defaults_preserve_all_frames_and_requires_explicit_hri_index(self):
        args = build_parser().parse_args(["--checkpoint", "own.pt", "--input", "lr.mp4", "--output", "sr.mp4"])
        self.assertIsNone(args.frames)
        self.assertEqual(args.caption, "A high quality video.")
        self.assertEqual((args.seed, args.steps, args.tile_height, args.tile_width, args.overlap), (42, 20, 256, 448, 64))
        self.assertFalse(args.save_rgb)
        self.assertEqual(args.allowed_checkpoint_root,PRODUCTION_RUNS_ROOT)
        args.hri = "provided.png"
        with self.assertRaises(ValueError):
            run_inference(args, device="cpu")

    def test_decoder_reads_all_or_explicit_prefix_and_keeps_rgb_order_fps(self):
        frames = [np.full((3, 5, 3), [index, 20, 100], dtype=np.uint8) for index in range(5)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "user_lr.mp4"
            path.touch()
            capture = FakeCapture(frames)
            with mock.patch("cv2.VideoCapture", return_value=capture):
                video, metadata = decode_lrv(path)
            self.assertEqual(video.shape, (1, 3, 5, 3, 5))
            self.assertEqual(metadata["fps"], 23.976)
            self.assertTrue(metadata["full_input_decoded"])
            self.assertEqual(metadata["source_frame_indices"], list(range(5)))
            self.assertAlmostEqual(float(video[0, 0, 4, 0, 0]), 100/255)
            self.assertAlmostEqual(float(video[0, 2, 4, 0, 0]), 4/255)
            self.assertTrue(capture.released)
            prefix_capture = FakeCapture(frames)
            with mock.patch("cv2.VideoCapture", return_value=prefix_capture):
                prefix, metadata = decode_lrv(path, frame_limit=3)
            self.assertEqual(prefix.shape[2], 3)
            self.assertEqual(prefix_capture.index, 3)
            self.assertFalse(metadata["full_input_decoded"])
            with mock.patch("cv2.VideoCapture", return_value=FakeCapture(frames, reported=6)), self.assertRaises(ValueError):
                decode_lrv(path)

    def test_hri_requires_exact_target_geometry_and_preview_rejects_wrong_official_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "supplied.png"
            Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(image)
            self.assertEqual(read_hri(image, (12, 20)).shape, (1, 3, 1, 12, 20))
            with self.assertRaises(ValueError):
                read_hri(image, (16, 20))
            checkpoint = root / "Base" / "checkpoint_1.pt"
            checkpoint.parent.mkdir()
            config = {"model": {"repo_id": OFFICIAL_REPO_ID, "revision": OFFICIAL_REVISION}, "training": {"variants": ["Base"]}}
            torch.save({"format": "vsr_training_v1", "step": 1, "config": config}, checkpoint)
            preview = preview_training_config(checkpoint, allowed_root=root)
            self.assertEqual(preview["config"], config)
            config["model"]["revision"] = "wrong"
            torch.save({"format": "vsr_training_v1", "step": 1, "config": config}, checkpoint)
            with self.assertRaises(ValueError):
                preview_training_config(checkpoint, allowed_root=root)
            with self.assertRaises(ValueError):
                preview_training_config(checkpoint, allowed_root=root / "different")

    def test_mocked_pipeline_load_order_and_source_only_inference_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path, output = directory / "user_lr.mp4", directory / "restored.mp4"
            input_path.write_bytes(b"user supplied lr only")
            args = build_parser().parse_args(["--checkpoint", "own/Base/checkpoint.pt", "--input", str(input_path),
                                             "--output", str(output), "--caption", "A moving red object.", "--save-rgb"])
            config = {"model": {"repo_id": OFFICIAL_REPO_ID, "revision": OFFICIAL_REVISION}, "reference": {"enabled": False}}
            preview = {"path": "/own/VSR/runs/new/Base/checkpoint.pt", "config": config, "checkpoint_sha256": "hash"}
            lrv = torch.zeros(1, 3, 5, 3, 5)
            source = {"fps": 23.976, "decoded_frames": 5, "source_frame_indices": list(range(5)), "full_input_decoded": True}
            calls = []
            model = mock.Mock()
            vae = object()
            def caption_encoder(*a, **k):
                calls.append("text encoded and released")
                return torch.zeros(1, 512, 4), {"caption_provided": args.caption}
            def checkpoint_loader(*a, **k):
                calls.append("DiT")
                return model, {"config": copy.deepcopy(config), "checkpoint_sha256": "hash"}
            def vae_loader(*a, **k):
                calls.append("VAE")
                return vae
            def restore(used_model, used_vae, supplied_lr, text, used_config, **kwargs):
                calls.append("restore")
                self.assertIs(used_model, model)
                self.assertIs(used_vae, vae)
                self.assertIs(supplied_lr, lrv)
                self.assertIsNone(kwargs["reference_image"])
                self.assertNotIn("hr", kwargs)
                self.assertNotIn("teacher", kwargs)
                return torch.ones(1, 3, 5, 12, 20) * .5, {"padding": {"time_end": 0}, "tiles": [1]}
            def writer(path, video, fps):
                self.assertEqual(fps, 23.976)
                Path(path).write_bytes(b"mock playback")
            with mock.patch("vsr.infer_video.preview_training_config", return_value=preview) as previewer, \
                 mock.patch("vsr.infer_video.decode_lrv", return_value=(lrv, source)) as decoder, \
                 mock.patch("vsr.infer_video.encode_caption", side_effect=caption_encoder), \
                 mock.patch("vsr.infer_video.load_training_checkpoint", side_effect=checkpoint_loader) as loader, \
                 mock.patch("vsr.infer_video.load_frozen_vae", side_effect=vae_loader), \
                 mock.patch("vsr.infer_video.restore_video", side_effect=restore), \
                 mock.patch("vsr.infer_video.write_video", side_effect=writer), \
                 mock.patch("vsr.infer_video.verify_output_video", return_value={"frame_count": 5, "height": 12, "width": 20, "fps": 23.976}):
                metadata = run_inference(args, device="cpu")
            self.assertEqual(calls, ["text encoded and released", "DiT", "VAE", "restore"])
            previewer.assert_called_once_with(args.checkpoint,allowed_root=PRODUCTION_RUNS_ROOT)
            loader.assert_called_once_with(preview['path'],device=torch.device('cpu'),allowed_root=PRODUCTION_RUNS_ROOT)
            decoder.assert_called_once_with(str(input_path), frame_limit=None)
            self.assertFalse(metadata["hr_target_accessed"])
            self.assertFalse(metadata["output"]["audio_preserved"])
            self.assertEqual(metadata["output"]["shape"], [1, 3, 5, 12, 20])
            self.assertTrue(output.with_suffix(".mp4.rgb.pt").is_file())
            saved = json.loads(output.with_suffix(".mp4.metadata.json").read_text())
            self.assertEqual(saved["caption"]["caption_provided"], args.caption)

    def test_true_method_arm_pipeline_uses_explicit_new_root_for_preview_and_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);allowed=directory/'new_runs'
            checkpoint=allowed/'pilot/arm_runs/A2/A2/checkpoint_002048.pt'
            source=directory/'user_lr.mp4';source.write_bytes(b'only user input')
            output=directory/'student.mp4'
            args=build_parser().parse_args(['--checkpoint',str(checkpoint),'--allowed-checkpoint-root',str(allowed),
                '--input',str(source),'--output',str(output)])
            config={'model':{'repo_id':OFFICIAL_REPO_ID,'revision':OFFICIAL_REVISION},
                'training':{'variants':['A2']},'reference':{'enabled':False},'restoration_architecture':'noref_base_v1'}
            preview={'path':str(checkpoint),'checkpoint_sha256':'bound','variant':'A2','config':config}
            audit={'config':config,'checkpoint_sha256':'bound','variant':'A2','inference_weights':'student',
                   'EMA_weights_selected':False,'training_only_state_restored':False}
            model=mock.Mock();vae=object()
            lrv=torch.zeros(1,3,5,2,3)
            decoded={'fps':24.,'decoded_frames':5,'source_frame_indices':list(range(5)),'full_input_decoded':True}
            def write_output(path,video,fps):Path(path).write_bytes(b'mock student output')
            with mock.patch('vsr.infer_video.preview_training_config',return_value=preview) as previewer,\
                 mock.patch('vsr.infer_video.decode_lrv',return_value=(lrv,decoded)),\
                 mock.patch('vsr.infer_video.encode_caption',return_value=(torch.zeros(1,512,4),{'cached_fixture':True})),\
                 mock.patch('vsr.infer_video.load_training_checkpoint',return_value=(model,audit)) as loader,\
                 mock.patch('vsr.infer_video.load_frozen_vae',return_value=vae),\
                 mock.patch('vsr.infer_video.restore_video',return_value=(torch.zeros(1,3,5,8,12),{})),\
                 mock.patch('vsr.infer_video.write_video',side_effect=write_output),\
                 mock.patch('vsr.infer_video.verify_output_video',return_value={'frame_count':5,'height':8,'width':12,'fps':24.}):
                metadata=run_inference(args,device='cpu')
            self.assertEqual(args.allowed_checkpoint_root,allowed)
            previewer.assert_called_once_with(str(checkpoint),allowed_root=allowed)
            loader.assert_called_once_with(str(checkpoint),device=torch.device('cpu'),allowed_root=allowed)
            self.assertEqual(metadata['checkpoint']['variant'],'A2')
            self.assertEqual(metadata['checkpoint']['inference_weights'],'student')
            self.assertFalse(metadata['checkpoint']['EMA_weights_selected'])
            self.assertFalse(metadata['teacher_accessed'])
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
