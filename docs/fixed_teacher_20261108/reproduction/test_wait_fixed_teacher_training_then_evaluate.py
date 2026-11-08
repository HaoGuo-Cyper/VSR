"""One-shot waiter tests; systemctl and every child subprocess are mocked."""
from pathlib import Path
import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name('wait_fixed_teacher_training_then_evaluate.py')


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Fixture:
    def __init__(self, folder):
        self.root = Path(folder); self.run = self.root/'training'; self.run.mkdir()
        self.protocol = self.run/'protocol.json'; save(self.protocol, {'schema': 'synthetic_only'})
        save(self.run/'summary.json', {'scientific_updates': 10240})
        save(self.run/'progress.json', {'scientific_updates': 4})
        self.helper = self.root/'helper.py'; self.helper.write_text('# synthetic helper\n')
        self.source = self.root/'source.py'; self.source.write_text('# synthetic frozen core\n')
        self.pipeline = self.root/'pipeline_training'
        self.receipt_path = self.root/'waiter'/'receipt.json'
        self.receipt = {
            'training_run': str(self.run), 'training_unit': 'synthetic-training.service',
            'training_protocol_sha256': sha(self.protocol),
            'source_freeze': {'source': {'path': str(self.source), 'sha256': sha(self.source)}},
            'bound_helpers': {'helper': {'path': str(self.helper), 'sha256': sha(self.helper)}},
            'pipeline_dir': str(self.pipeline), 'source_root': str(self.root),
            'command': [sys.executable, str(self.helper), '--after-training', '--execute'],
        }
        save(self.receipt_path, self.receipt)
        self.status_path = self.receipt_path.parent/'status.json'

    def state(self, status):
        state = {'status': status, 'scientific_updates': 10240 if status == 'completed' else 4,
                 'protocol_sha256': sha(self.protocol)}
        if status == 'completed': state['summary_sha256'] = sha(self.run/'summary.json')
        return state

    def execute(self, observations, *, after_sleep=None, child_code=0, completion=None, clocks=None):
        remaining = iter(observations)
        def show(command, **kwargs):
            self.test.assertEqual(command, ['systemctl', '--user', 'show', 'synthetic-training.service',
                                            '-p', 'MainPID', '-p', 'ExecMainStatus', '-p', 'Result'])
            pid, code, result, state = next(remaining)
            save(self.run/'status.json', self.state(state) if isinstance(state, str) else state)
            return f'MainPID={pid}\nExecMainStatus={code}\nResult={result}\n'
        def child(command, **kwargs):
            self.test.assertEqual(command, self.receipt['command'])
            self.test.assertTrue('--after-training' in command and '--include-training' not in command)
            self.test.assertEqual(kwargs, {'cwd': str(self.root), 'check': False})
            if child_code == 0:
                save(self.pipeline/'completion.json', completion or {
                    'status': 'awaiting_root_delivery', 'validation_restores': 1024, 'test_restores': 192})
            return subprocess.CompletedProcess(command, child_code)
        with mock.patch.object(sys, 'argv', [str(SCRIPT), str(self.receipt_path)]), \
             mock.patch('subprocess.check_output', side_effect=show) as show_mock, \
             mock.patch('subprocess.run', side_effect=child) as child_mock, \
             mock.patch('time.sleep', side_effect=(lambda seconds: after_sleep() if after_sleep else None)) as sleep_mock, \
             mock.patch('time.monotonic', side_effect=clocks or list(range(100))):
            self.show_mock = show_mock; self.child_mock = child_mock; self.sleep_mock = sleep_mock
            return runpy.run_path(str(SCRIPT), run_name='__main__')


class WaiterTests(unittest.TestCase):
    def fixture(self, folder):
        f = Fixture(folder); f.test = self; return f

    def test_running_then_success_starts_only_one_after_training_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            f = self.fixture(folder)
            f.execute([(98, 0, 'success', 'running'), (0, 0, 'success', 'completed')])
            self.assertEqual(f.child_mock.call_count, 1); self.assertEqual(f.sleep_mock.call_count, 1)
            self.assertEqual(json.loads(f.status_path.read_text())['status'], 'awaiting_root_delivery')

    def test_completed_metadata_while_process_exits_waits_for_unit_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            f = self.fixture(folder)
            f.execute([(98, 0, 'success', 'completed'), (0, 0, 'success', 'completed')])
            self.assertEqual(f.show_mock.call_count, 2); self.assertEqual(f.sleep_mock.call_count, 1)
            self.assertEqual(f.child_mock.call_count, 1)

    def test_training_failed_stops_without_child_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            f = self.fixture(folder)
            with self.assertRaises(AssertionError): f.execute([(0, 1, 'exit-code', 'failed')])
            self.assertEqual(f.child_mock.call_count, 0)
            status = json.loads(f.status_path.read_text())
            self.assertEqual(status['status'], 'failed_stopped'); self.assertFalse(status['automatic_retry'])

    def test_source_change_during_wait_stops_before_child(self):
        with tempfile.TemporaryDirectory() as folder:
            f = self.fixture(folder)
            with self.assertRaisesRegex(AssertionError, 'Frozen core source changed'):
                f.execute([(98, 0, 'success', 'running')], after_sleep=lambda: f.source.write_text('changed\n'))
            self.assertEqual(f.child_mock.call_count, 0)

    def test_helper_change_before_handoff_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            f = self.fixture(folder)
            with self.assertRaisesRegex(AssertionError, 'Handoff helper changed before evaluation'):
                f.execute([(98, 0, 'success', 'running'), (0, 0, 'success', 'completed')],
                          after_sleep=lambda: f.helper.write_text('changed\n'))
            self.assertEqual(f.child_mock.call_count, 0)

    def test_existing_pipeline_refuses_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            f = self.fixture(folder); f.pipeline.mkdir()
            with self.assertRaisesRegex(AssertionError, 'Pipeline path already exists'):
                f.execute([(0, 0, 'success', 'completed')])
            self.assertEqual(f.child_mock.call_count, 0)

    def test_child_failure_stops_after_one_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            f = self.fixture(folder)
            with self.assertRaisesRegex(AssertionError, 'Serial pipeline failed'):
                f.execute([(0, 0, 'success', 'completed')], child_code=9)
            self.assertEqual(f.child_mock.call_count, 1)
            self.assertFalse(json.loads(f.status_path.read_text())['automatic_retry'])

    def test_partial_or_inconsistent_completed_chain_refuses_handoff(self):
        for field, bad in [('scientific_updates', 10239), ('summary_sha256', 'not-the-summary')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                f = self.fixture(folder); state = f.state('completed'); state[field] = bad
                with self.assertRaises(AssertionError): f.execute([(0, 0, 'success', state)])
                self.assertEqual(f.child_mock.call_count, 0)

    def test_declared_timeout_stops_without_child(self):
        with tempfile.TemporaryDirectory() as folder:
            f = self.fixture(folder)
            with self.assertRaisesRegex(AssertionError, '3h service ceiling'):
                f.execute([], clocks=[0, 10801])
            self.assertEqual(f.show_mock.call_count, 0); self.assertEqual(f.child_mock.call_count, 0)


if __name__ == '__main__': unittest.main()
