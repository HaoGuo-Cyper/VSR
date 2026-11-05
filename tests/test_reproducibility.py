"""CPU-only reproducibility flags and task-state hashing; no CUDA init."""
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vsr.reproducibility import configure_reproducibility, reproducibility_flags, tensor_state_hashes


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
        self.lrv_stem = nn.Linear(2, 2)
        self.lrv_residuals = nn.Linear(2, 2)
        self.lora_A = nn.Linear(2, 2, bias=False)
        self.projector = nn.Linear(2, 2)
        self.reference_adapter = nn.Linear(2, 2)


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.flags = reproducibility_flags()
        self.workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    def tearDown(self):
        f = self.flags
        torch.use_deterministic_algorithms(f["deterministic_algorithms"], warn_only=f["deterministic_warn_only"])
        torch.backends.cudnn.benchmark = f["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = f["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = f["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = f["matmul_allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = f["matmul_allow_fp16_reduced_precision_reduction"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = f["matmul_allow_bf16_reduced_precision_reduction"]
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(f["sdp_math_allow_fp16_bf16_reduction"])
        torch.backends.cuda.enable_flash_sdp(f["sdp_flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(f["sdp_memory_efficient"])
        torch.backends.cuda.enable_cudnn_sdp(f["sdp_cudnn"])
        torch.backends.cuda.enable_math_sdp(f["sdp_math"])
        if self.workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.workspace

    def test_seeded_default_changes_no_flags_or_rng(self):
        rng = torch.random.get_rng_state().clone()
        with patch("torch.cuda.init", side_effect=AssertionError("GPU initialization forbidden")):
            result = configure_reproducibility({"seed": 42})
        self.assertEqual(result["mode"], "seeded")
        self.assertFalse(result["behavior_changed"])
        self.assertEqual(result["before"], result["flags"])
        self.assertEqual(self.flags, reproducibility_flags())
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(result["bitwise_reproducibility_verified"])

    def test_strict_cpu_flags_no_gpu_and_no_rng_changes(self):
        rng = torch.random.get_rng_state().clone()
        with patch("torch.cuda.is_initialized", return_value=False), patch("torch.cuda.init", side_effect=AssertionError("No GPU init")), patch("torch.use_deterministic_algorithms", wraps=torch.use_deterministic_algorithms) as deterministic:
            result = configure_reproducibility({"reproducibility": {"mode": "strict"}})
            deterministic.assert_called_once_with(True, warn_only=False)
        f = result["flags"]
        self.assertTrue(f["deterministic_algorithms"])
        self.assertFalse(f["deterministic_warn_only"])
        self.assertTrue(f["cudnn_deterministic"])
        self.assertFalse(f["cudnn_benchmark"])
        self.assertFalse(f["cudnn_allow_tf32"])
        self.assertFalse(f["matmul_allow_tf32"])
        self.assertFalse(f["matmul_allow_fp16_reduced_precision_reduction"])
        self.assertFalse(f["matmul_allow_bf16_reduced_precision_reduction"])
        self.assertFalse(f["sdp_flash"])
        self.assertFalse(f["sdp_memory_efficient"])
        self.assertFalse(f["sdp_cudnn"])
        self.assertTrue(f["sdp_math"])
        self.assertFalse(f["sdp_math_allow_fp16_bf16_reduction"])
        self.assertEqual(f["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(result["bitwise_reproducibility_verified"])
        self.assertIn("CPU", result["strict_relation_requirement"])

    def test_already_initialized_wrong_workspace_fails_before_mutation(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), patch("torch.cuda.is_initialized", return_value=True), patch("torch.use_deterministic_algorithms") as setter:
            with self.assertRaisesRegex(RuntimeError, "before CUDA initialization"):
                configure_reproducibility({"reproducibility": {"mode": "strict"}})
            setter.assert_not_called()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")
        with self.assertRaises(ValueError):
            configure_reproducibility({"reproducibility": {"mode": "unknown"}})

    def test_hashes_repeat_and_change_only_affected_trainable_group(self):
        torch.manual_seed(11)
        model = FakeModel()
        first = tensor_state_hashes(model)
        self.assertEqual(first, tensor_state_hashes(model))
        self.assertEqual(first["groups"]["lrv"]["tensor_count"], 4)
        self.assertEqual(first["groups"]["lora"]["tensor_count"], 1)
        self.assertEqual(first["groups"]["hri"]["tensor_count"], 2)
        self.assertFalse(first["frozen_tensors_copied"])
        with torch.no_grad():
            model.frozen.add_(100)
        self.assertEqual(first, tensor_state_hashes(model))
        with torch.no_grad():
            model.projector.weight[0,0] += 1
        second = tensor_state_hashes(model)
        self.assertNotEqual(first["groups"]["projector"]["sha256"], second["groups"]["projector"]["sha256"])
        self.assertNotEqual(first["all_trainable"]["sha256"], second["all_trainable"]["sha256"])
        for group in ["lrv", "lora", "hri", "other_trainable"]:
            self.assertEqual(first["groups"][group], second["groups"][group])

    def test_hash_includes_dtype_shape_name_and_bf16_raw_bytes(self):
        model = nn.Module()
        model.register_parameter("projector", nn.Parameter(torch.tensor([1.,2.],dtype=torch.bfloat16)))
        first = tensor_state_hashes(model)["all_trainable"]["sha256"]
        model.projector = nn.Parameter(model.projector.float())
        self.assertNotEqual(first, tensor_state_hashes(model)["all_trainable"]["sha256"])
        flat = tensor_state_hashes(model)["all_trainable"]["sha256"]
        model.projector = nn.Parameter(model.projector.reshape(1,2))
        self.assertNotEqual(flat, tensor_state_hashes(model)["all_trainable"]["sha256"])
        renamed = nn.Module()
        renamed.register_parameter("other", nn.Parameter(model.projector.detach().clone()))
        self.assertNotEqual(tensor_state_hashes(model)["all_trainable"]["sha256"], tensor_state_hashes(renamed)["all_trainable"]["sha256"])


if __name__ == "__main__":
    unittest.main()
