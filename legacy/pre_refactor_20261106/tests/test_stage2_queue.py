"""CPU checks for config/checkpoint compatibility before expensive training."""
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vsr.checkpoints import _check_config
from vsr import stage2_base_queue as queue


class ProtocolConfigTests(unittest.TestCase):
    def test_formal_probe_and_cache_reuse_configs_remain_loadable(self):
        source = Path(__file__).resolve().parents[1]/'configs/stage2_20261105/base_seed20261102.json'
        with patch.object(queue,'ROOT',Path('/home/hugo/Projects/VSR_')):
            formal = queue.effective_config(source)
            diagnostic = queue.effective_config(source,profile=True)
            reused = queue.effective_config(source,cache_from='/home/hugo/Projects/VSR_/runs/new_sealed_run')
        for config in (formal,diagnostic,reused):
            self.assertEqual(_check_config(config),config['_sha256'])
            changed = deepcopy(config)
            changed['training']['seed'] += 1
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                _check_config(changed)
        self.assertEqual(len({x['_sha256'] for x in (formal,diagnostic,reused)}),3)
        self.assertEqual(diagnostic['data']['val_limit'],0)
        self.assertEqual(diagnostic['data']['train_limit'],2)
        self.assertEqual(diagnostic['training']['updates'],8)
        self.assertEqual(formal['training']['updates'],8192)
        self.assertEqual(formal['data']['val_limit'],32)
        self.assertEqual(reused['data']['cache_from'],'/home/hugo/Projects/VSR_/runs/new_sealed_run')

    def test_wrong_geometry_resume_or_auxiliary_variants_rejected(self):
        source = Path(__file__).resolve().parents[1]/'configs/stage2_20261105/base_seed20261102.json'
        with patch.object(queue,'ROOT',Path('/home/hugo/Projects/VSR_')):
            config = queue.effective_config(source)
            for section,field,value in [('data','temporal_stride',4),('training','resume',True),('training','variants',['Base','Align'])]:
                changed = deepcopy(config)
                changed[section][field] = value
                with self.assertRaises(RuntimeError):
                    queue.validate_config(changed)


class WorkerLeaseTests(unittest.TestCase):
    def test_worker_lease_survives_queue_lock_release_and_ends_with_worker(self):
        """Exercise the production worker-entry lock in real CPU subprocesses."""
        code_root = Path(__file__).resolve().parents[1]
        child = r'''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from vsr import stage2_base_queue as queue
root, hold = Path(sys.argv[2]), sys.argv[3] == "hold"
queue.ROOT = root
assert "torch" not in sys.modules
def fake_worker(args):
    print("WORKER_LOCK_HELD", flush=True)
    if hold:
        sys.stdin.readline()
queue.worker = fake_worker
sys.argv = ["lease-fixture", "worker", "--config", "unused.json",
            "--result-file", str(root / "unused-result.json")]
try:
    queue.main()
except BlockingIOError:
    print("GPU_LEASE_BUSY", flush=True)
    sys.exit(42)
assert "torch" not in sys.modules
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-c", child, str(code_root), str(root)]
            # This represents the queue parent; its lease is not the GPU lease.
            parent_lock = (root / "runs/stage2_queue.lock").open("a")
            fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = subprocess.Popen(command + ["hold"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, env=env)
            try:
                ready, _, _ = select.select([owner.stdout], [], [], 5)
                self.assertTrue(ready, "Worker did not reach its protected entry")
                self.assertEqual(owner.stdout.readline().strip(), "WORKER_LOCK_HELD")
                fcntl.flock(parent_lock, fcntl.LOCK_UN)
                parent_lock.close()
                blocked = subprocess.run(command + ["once"], text=True,
                                         capture_output=True, env=env, timeout=5)
                self.assertEqual(blocked.returncode, 42, blocked.stderr)
                self.assertIn("GPU_LEASE_BUSY", blocked.stdout)
                self.assertNotIn("WORKER_LOCK_HELD", blocked.stdout)
                owner.communicate(input="release\n", timeout=5)
                self.assertEqual(owner.returncode, 0)
                available = subprocess.run(command + ["once"], text=True,
                                           capture_output=True, env=env, timeout=5)
                self.assertEqual(available.returncode, 0, available.stderr)
                self.assertIn("WORKER_LOCK_HELD", available.stdout)
                self.assertFalse((root / "unused-result.json").exists())
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.communicate(timeout=5)
                if not parent_lock.closed:
                    parent_lock.close()


if __name__=='__main__':
    unittest.main()
