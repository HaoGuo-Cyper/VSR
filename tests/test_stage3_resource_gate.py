"""CPU-only G3 seals, counting and aggregation fixtures; no task tensors/GPU."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'scripts'))
import stage3_resource_gate as worker


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def timing_rows(*,a2_seconds=1.5,sigma=.8):
    result = []
    for role in worker.ROLES:
        for ordinal in range(worker.WARMUP+worker.TIMED):
            active = role == 'A2' and sigma > worker.SIGMA_BOUNDARY
            result.append({'role':role,'ordinal':ordinal,'phase':'warmup' if ordinal<2 else 'timed',
                'seconds':a2_seconds if role == 'A2' else 1.,'fm':2.,'sd':.2 if active else 0.,
                'total':2.16 if active else 2.,'sigma':sigma,'teacher_forward':active,
                'mask_spans':[] if role == 'flow_only' else [[[2,4]]],
                'matched_corruption_RNG':{'mask':'same','corruption':'same'},
                'matched_FM':{'data_index':ordinal,'record_id':str(ordinal),'noise':'same','sigma':sigma}})
    return result


def competing_worker_lock_probe(root):
    """A real second process, independent file descriptor and nonblocking flock."""
    code = """import fcntl,sys
with open(sys.argv[1],'a') as stream:
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:sys.exit(73)
"""
    return subprocess.run([sys.executable,'-c',code,str(root/'stage2_gpu.lock')],
                          capture_output=True,text=True,timeout=10).returncode


class MetadataAndAggregationTests(unittest.TestCase):
    def test_import_and_help_do_not_import_torch(self):
        code = f"import sys;sys.dont_write_bytecode=True;sys.path.insert(0,{str(ROOT/'scripts')!r});import stage3_resource_gate;assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result = subprocess.run([sys.executable,str(ROOT/'scripts/stage3_resource_gate.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_budget_default_and_whole_timed_gate_boundary(self):
        self.assertEqual(worker.BUDGET['benchmark_updates'],3*(2+20))
        self.assertEqual(worker.BUDGET['total_diagnostic_updates'],66+4)
        self.assertEqual((worker.METHOD['frames'],worker.METHOD['latent_frames'],worker.METHOD['rho'],worker.METHOD['span']),(33,9,.25,2))
        result = worker.summarize_roles(timing_rows())
        self.assertEqual(result['status'],'passed_G3_resource_cost')
        self.assertEqual(result['primary_ratio_A2_over_flow_all20'],1.5)
        self.assertEqual(result['secondary']['teacher_active_count_timed'],20)
        self.assertEqual(result['role_timing']['A2']['n'],20)
        self.assertEqual(worker.summarize_roles(timing_rows(a2_seconds=1.500001))['status'],'failed_G3_overhead_stop')

    def test_all_outliers_retained_and_sample_SD(self):
        rows = timing_rows(a2_seconds=1.)
        rows[-1]['seconds'] = 21.
        result = worker.summarize_roles(rows)
        self.assertEqual(result['status'],'failed_G3_overhead_stop')
        self.assertEqual(result['primary_ratio_A2_over_flow_all20'],2.)
        self.assertAlmostEqual(result['role_timing']['A2']['sample_SD_seconds'],20/(20**.5))
        self.assertEqual(result['outliers_removed'],0)
        self.assertEqual(len(rows),66)

    def test_no_active_sigma_is_null_and_float32_boundary_is_excluded(self):
        result = worker.summarize_roles(timing_rows(sigma=worker.SIGMA_BOUNDARY))
        self.assertEqual(result['secondary']['teacher_active_count_timed'],0)
        self.assertIsNone(result['secondary']['A2_over_flow_same_active_indices'])
        self.assertIsNone(result['secondary']['active_A2_timing']['mean_seconds'])
        self.assertIsNone(result['secondary']['active_A2_timing']['sample_SD_seconds'])

    def test_mixed_default_window_keeps_inactive_steps_in_primary_cost(self):
        rows = timing_rows(a2_seconds=1.)
        for row in rows:
            sigma = .8 if row['ordinal'] in (2,3) else .1
            row['sigma'] = row['matched_FM']['sigma'] = sigma
            active = row['role']=='A2' and sigma>.3
            row.update(teacher_forward=active,sd=.2 if active else 0.,total=2.16 if active else 2.)
            if active:row['seconds'] = 4.
        result = worker.summarize_roles(rows)
        self.assertEqual(result['status'],'passed_G3_resource_cost')
        self.assertEqual(result['primary_ratio_A2_over_flow_all20'],1.3)
        self.assertEqual(result['secondary']['A2_over_flow_same_active_indices'],4.)
        self.assertEqual(result['secondary']['teacher_active_count_timed'],2)

    def test_memory_limit_checks_reserved_allocated_and_physical_capacity(self):
        GiB = 1024**3
        def fake(allocated,reserved,physical=16*GiB):
            return SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:allocated,
                max_memory_reserved=lambda:reserved,get_device_properties=lambda _:SimpleNamespace(total_memory=physical)))
        self.assertEqual(worker._memory(fake(4*GiB,5*GiB))['peak_reserved_bytes'],5*GiB)
        for device in (fake(17*GiB,17*GiB),fake(4*GiB,17*GiB),fake(4*GiB,5*GiB,4*GiB)):
            with self.assertRaisesRegex(ValueError,'memory ceiling'):worker._memory(device)

    def test_source_freeze_includes_new_script_tests_and_shared_numeric_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ['vsr/self_distillation_step.py','scripts/stage3_resource_gate.py',
                'scripts/stage2_evaluate_bases.py','scripts/stage2_framewise.py','scripts/stage2_metrics.py',
                'tests/test_stage3_resource_gate.py','tests/test_self_distillation.py','tests/test_self_distillation_step.py']
            names += ['tests/test_method_checkpoint.py','tests/test_method_diagnostics.py','tests/test_restoration_architecture.py']
            for name in names:
                path = root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen fixture\n')
            before = worker.source_bindings(root)
            self.assertEqual(set(before),set(names))
            (root/'scripts/stage3_resource_gate.py').write_text('# changed independently after seal\n')
            self.assertNotEqual(before,worker.source_bindings(root))

    def test_rejects_missing_or_mismatched_data_stream_window_or_finite(self):
        changes = [lambda rows:rows.pop(),
            lambda rows:rows[-1]['matched_FM'].update(noise='different'),
            lambda rows:rows[-1].update(phase='warmup'),
            lambda rows:rows[-1].update(teacher_forward=False),
            lambda rows:rows[-1].update(seconds=float('nan')),
            lambda rows:rows[-1].update(sd=float('inf')),
            lambda rows:rows[-1]['matched_corruption_RNG'].update(mask='different'),
            lambda rows:rows[-1].update(mask_spans=[[[6,8]]])]
        for change in changes:
            with self.subTest(change=change):
                rows = timing_rows();change(rows)
                with self.assertRaises(ValueError):worker.summarize_roles(rows)

    def test_successful_optimizer_call_count_survives_later_failure(self):
        optimizer = SimpleNamespace(step=lambda:None)
        counter = worker.UpdateCounter();counter.attach(optimizer,'A2')
        with self.assertRaisesRegex(ValueError,'post-update EMA'):
            optimizer.step()
            raise ValueError('post-update EMA fixture failure')
        report = counter.snapshot()
        self.assertEqual(report['diagnostic_optimizer_updates_applied'],1)
        self.assertEqual(report['diagnostic_optimizer_updates_attempted'],1)
        self.assertEqual(report['completed_step_returns'],0)
        self.assertEqual(report['scientific_training_updates'],0)

    def test_optimizer_throw_records_unknown_partial_effect_and_preserves_original_error(self):
        def fail():raise RuntimeError('AdamW fixture OOM')
        optimizer = SimpleNamespace(step=fail)
        counter = worker.UpdateCounter();counter.attach(optimizer,'A2')
        with self.assertRaisesRegex(RuntimeError,'AdamW fixture OOM'):optimizer.step()
        self.assertEqual(counter.snapshot()['optimizer_calls_not_returned_partial_effect_unknown'],1)
        self.assertEqual(counter.snapshot()['diagnostic_optimizer_updates_applied'],0)

    def test_failure_traceback_reference_does_not_mask_original_OOM(self):
        class Engine:
            def fail(self):raise RuntimeError('original fixture OOM')
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))
        engine = Engine();reference = weakref.ref(engine);counter = worker.UpdateCounter()
        with self.assertRaisesRegex(RuntimeError,'original fixture OOM'):
            try:engine.fail()
            finally:
                successful = sys.exc_info()[0] is None
                engine = None
                worker._check_release(torch,reference,successful,counter,'failure_fixture')
        self.assertTrue(counter.cleanup_events[0]['release_assert_skipped_during_original_exception'])
        self.assertTrue(counter.cleanup_events[0]['model_reference_alive_via_possible_traceback'])
        held = Engine()
        with self.assertRaisesRegex(ValueError,'remains resident'):
            worker._check_release(torch,weakref.ref(held),True,counter,'success_fixture')

    def test_prepare_requires_fresh_output_and_does_not_mutate_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(project_root=root,evaluation_run=root/'runs/T2',run_dir=root/'runs/G3')
            protocol = {'fixed':'CPU fixture','budget':copy.deepcopy(worker.BUDGET)}
            with patch.object(worker,'collect_protocol',return_value=protocol):
                result = worker.prepare(args)
                self.assertEqual(result['budget']['total_diagnostic_updates'],70)
                self.assertEqual(worker.read(args.run_dir/'protocol.json'),protocol)
                self.assertFalse(worker.read(args.run_dir/'status.json')['G3_passed'])
                with self.assertRaisesRegex(ValueError,'Fresh-only'):worker.prepare(args)
            args.run_dir = args.evaluation_run/'bad'
            with self.assertRaisesRegex(ValueError,'independent'):worker.safe_run(args)


class CompletedT2Tests(unittest.TestCase):
    def fixture(self,root):
        directory = root/'runs/T2';directory.mkdir(parents=True)
        cohort = [{'record':{'id':f'r{i}','source_id':f's{i}'},'mask':[False]*4+[True]*8+[False]*21} for i in range(32)]
        conditions = worker.t2.CONDITIONS[1:]
        seal = {'seal_digest':'cohort-fixture','cohort':cohort,'config':{'training_seeds':worker.t2.SEEDS,
            'inference':{'sampling_seeds':[42,43]},'conditions':conditions}}
        sealpath = root/'seal.json';write(sealpath,seal)
        checkpoints = {str(seed):{'checkpoint':{'path':str(root/f'actual/runs/{seed}/Base/checkpoint_008192.pt'),
            'sha256':str(index+1)*64}} for index,seed in enumerate(worker.t2.SEEDS)}
        protocol = {'schema':'stage2_complete_Base_val32_evaluation_v1','project_root':str(root),
            'allowed_checkpoint_root':str(root/'actual/runs'),'training_updates':0,'cohort_seal':{'path':str(sealpath)},
            'checkpoints':checkpoints,'dataset_manifest':{'path':str(root/'dataset.json')},
            'official_model_manifest':{'path':str(root/'official.json')},'metric_weights':[{'path':str(root/'weights/alex.pth')}],
            'dependencies':{'root':str(root/'deps')},'desktop_binding':{'PID':4816,'reported_name':'gjs'}}
        write(directory/'protocol.json',protocol)
        psha = worker.file_sha(directory/'protocol.json')
        rows = []
        for seed in worker.t2.SEEDS:
            for item in cohort:
                for sample in (42,43):
                    for condition in worker.t2.CONDITIONS:
                        corrupt = condition != 'clean'
                        rows.append({'training_seed':seed,'record_id':item['record']['id'],'source_id':item['record']['source_id'],
                            'sampling_seed':sample,'condition':condition,'cohort_seal_digest':seal['seal_digest'],
                            'checkpoint_sha256':checkpoints[str(seed)]['checkpoint']['sha256'],
                            'score':{'mask':item['mask'],'raw_harm_gap':{'psnr_rgb_db':1. if corrupt else 0.,
                                                                       'lpips_alex_v0_1':.1 if corrupt else 0.}}})
        observations = {'protocol_sha256':psha,'training_updates':0,'observations':rows}
        gate = worker.t2.summarize_g2(rows,seal)
        write(directory/'observations.json',observations);write(directory/'G2.json',gate)
        roles = [{'training_seed':seed,'trainable_sha256_before':'same','trainable_sha256_after':'same'} for seed in worker.t2.SEEDS]
        summary = {'status':'completed_full_matrix','G2_status':'passed_G2','protocol_sha256':psha,'training_updates':0,
            'observations':1152,'prediction_PNGs':38016,'test_or_external_pixels_used':False,'roles':roles,
            'G2_sha256':worker.file_sha(directory/'G2.json'),'observations_sha256':worker.file_sha(directory/'observations.json')}
        write(directory/'summary.json',summary)
        write(directory/'status.json',{'status':'completed_full_matrix','G2_status':'passed_G2','protocol_sha256':psha,
            'training_updates':0,'summary_sha256':worker.file_sha(directory/'summary.json')})
        return directory,protocol

    def reseal(self,directory):
        summary = worker.read(directory/'summary.json')
        summary['G2_sha256'] = worker.file_sha(directory/'G2.json')
        summary['observations_sha256'] = worker.file_sha(directory/'observations.json')
        write(directory/'summary.json',summary)
        status = worker.read(directory/'status.json');status['summary_sha256'] = worker.file_sha(directory/'summary.json')
        write(directory/'status.json',status)

    def test_valid_json_roundtrip_G2_and_all_three_Base_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);directory,protocol = self.fixture(root)
            with patch.object(worker,'ACTUAL_RUNS',root/'actual/runs'),patch.object(worker.t2,'collect_protocol',return_value=protocol) as validate:
                result,evidence = worker.completed_t2(directory,root)
                self.assertEqual(result,protocol)
                self.assertEqual(evidence['observations'],1152)
                self.assertTrue(evidence['recomputed_G2'])
                self.assertTrue(evidence['all3_completed_Base8192'])
                self.assertEqual(len(validate.call_args.args[0].checkpoint),3)

    def test_missing_duplicate_wrong_source_mask_or_checkpoint_is_rejected_even_resealed(self):
        mutations = [lambda rows:rows.pop(),lambda rows:rows.__setitem__(-1,copy.deepcopy(rows[0])),
            lambda rows:rows[-1].update(source_id='wrong'),lambda rows:rows[-1]['score']['mask'].__setitem__(0,True),
            lambda rows:rows[-1].update(checkpoint_sha256='f'*64)]
        for mutate in mutations:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);directory,protocol=self.fixture(root)
                observations=worker.read(directory/'observations.json');mutate(observations['observations'])
                write(directory/'observations.json',observations);self.reseal(directory)
                with patch.object(worker,'ACTUAL_RUNS',root/'actual/runs'),patch.object(worker.t2,'collect_protocol',return_value=protocol):
                    with self.assertRaises(ValueError):worker.completed_t2(directory,root)

    def test_rejects_failed_G2_tampered_seal_missing_Base_role_or_Base_validation_failure(self):
        for kind in ('failed_G2','tampered','missing_role','Base_not_completed'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);directory,protocol=self.fixture(root)
                if kind == 'failed_G2':
                    gate=worker.read(directory/'G2.json');gate['status']='failed_G2_stop';write(directory/'G2.json',gate);self.reseal(directory)
                elif kind == 'tampered':
                    with (directory/'observations.json').open('a') as stream:stream.write(' ')
                elif kind == 'missing_role':
                    summary=worker.read(directory/'summary.json');summary['roles'].pop();write(directory/'summary.json',summary);self.reseal(directory)
                with patch.object(worker,'ACTUAL_RUNS',root/'actual/runs'),patch.object(worker.t2,'collect_protocol',
                    side_effect=ValueError('formal Base not completed8192') if kind=='Base_not_completed' else None,return_value=protocol):
                    with self.assertRaises(ValueError):worker.completed_t2(directory,root)


class ExecutionRefusalTests(unittest.TestCase):
    def test_real_second_worker_cannot_lease_during_success_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);events=[]
            def cleanup():events.append(('cleanup',competing_worker_lock_probe(root)))
            def unexpected(*args):self.fail('Success path must not write a failure')
            with worker._leased_lifecycle(root,cleanup,unexpected,unexpected):
                events.append(('body',competing_worker_lock_probe(root)))
            self.assertEqual(events,[('body',73),('cleanup',73)])
            self.assertEqual(competing_worker_lock_probe(root),0)

    def test_primary_and_cleanup_errors_retained_and_lock_covers_both_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);actual=root/'actual/runs';actual.mkdir(parents=True)
            args=SimpleNamespace(project_root=root,evaluation_run=root/'runs/T2',run_dir=root/'runs/G3',expected_protocol_sha=None)
            protocol={'CPU_fixture':True}
            with patch.object(worker,'collect_protocol',return_value=protocol):worker.prepare(args)
            args.expected_protocol_sha=worker.file_sha(args.run_dir/'protocol.json')
            observed=[]
            original_save=worker.save
            def audit_save(path,value,**options):
                if Path(path).name in ('failure.json','cleanup_failure.json'):
                    observed.append((Path(path).name,competing_worker_lock_probe(actual)))
                return original_save(path,value,**options)
            def fail_cleanup():
                observed.append(('cleanup',competing_worker_lock_probe(actual)))
                raise RuntimeError('secondary cleanup fixture error')
            with patch.object(worker,'ACTUAL_RUNS',actual),patch.object(worker,'save',side_effect=audit_save),\
                 patch.object(worker,'collect_protocol',side_effect=RuntimeError('primary fixture OOM')),\
                 patch.object(worker.gc,'collect',side_effect=fail_cleanup):
                with self.assertRaisesRegex(RuntimeError,'primary fixture OOM'):
                    worker.execute(args)
            self.assertEqual(observed,[('failure.json',73),('cleanup',73),('cleanup_failure.json',73)])
            failure=worker.read(args.run_dir/'failure.json')
            secondary=worker.read(args.run_dir/'cleanup_failure.json')
            status=worker.read(args.run_dir/'status.json')
            self.assertEqual(failure['error'],'RuntimeError: primary fixture OOM')
            self.assertEqual(secondary['primary_error_preserved'],failure['error'])
            self.assertEqual(secondary['primary_failure_sha256'],worker.file_sha(args.run_dir/'failure.json'))
            self.assertEqual(secondary['secondary_cleanup_error'],'RuntimeError: secondary cleanup fixture error')
            self.assertEqual(status['status'],'failed_stop_no_retry');self.assertFalse(status['G3_passed'])
            self.assertEqual(status['cleanup_failure_sha256'],worker.file_sha(args.run_dir/'cleanup_failure.json'))
            self.assertEqual(competing_worker_lock_probe(actual),0)

    def test_cleanup_only_failure_revokes_pass_and_preserves_computed_cost_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'run';run.mkdir();counter=worker.UpdateCounter();held=[]
            def cleanup():
                held.append(competing_worker_lock_probe(root))
                raise RuntimeError('cleanup-only fixture error')
            def fail(error,details):
                held.append(competing_worker_lock_probe(root))
                worker._save_execution_failure(run,{'protocol_sha256':'sealed-fixture','phase':'final_GPU_cleanup',
                    'error':f'{type(error).__name__}: {error}','traceback':details,'G3_passed':False},counter.snapshot())
            def unexpected(*args):self.fail('No prior primary failure, so this must be the primary cleanup error')
            with self.assertRaisesRegex(RuntimeError,'cleanup-only fixture error'):
                with worker._leased_lifecycle(root,cleanup,fail,unexpected):
                    write(run/'G3.json',{'status':'passed_G3_resource_cost','primary_ratio_A2_over_flow_all20':1.25})
                    write(run/'summary.json',{'status':'passed_G3_resource_cost','G3_passed':True,
                        'G3_sha256':worker.file_sha(run/'G3.json')})
                    write(run/'status.json',{'status':'passed_G3_resource_cost','G3_passed':True})
            self.assertEqual(held,[73,73])
            summary=worker.read(run/'summary.json');status=worker.read(run/'status.json')
            self.assertEqual(summary['status'],status['status'],'failed_stop_no_retry')
            self.assertEqual(summary['status'],'failed_stop_no_retry')
            self.assertFalse(summary['G3_passed']);self.assertFalse(status['G3_passed'])
            self.assertEqual(summary['computed_cost_gate_status_before_execution_failure'],'passed_G3_resource_cost')
            self.assertEqual(worker.read(run/'G3.json')['status'],'passed_G3_resource_cost')
            self.assertEqual(status['summary_sha256'],worker.file_sha(run/'summary.json'))
            self.assertEqual(status['failure_sha256'],worker.file_sha(run/'failure.json'))
            self.assertEqual(competing_worker_lock_probe(root),0)

    def test_busy_lease_or_failed_GPU_guard_stops_before_torch_and_disallows_retry(self):
        for refusal in ('lease','guard'):
            with self.subTest(refusal=refusal),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                args=SimpleNamespace(project_root=root,evaluation_run=root/'runs/T2',run_dir=root/'runs/G3',expected_protocol_sha=None)
                protocol={'evaluation_protocol':{'helpers':{'gpu_classifier':{'path':'classifier-fixture'}},
                    'desktop_binding':{'fixture':True}}}
                with patch.object(worker,'collect_protocol',return_value=protocol):
                    worker.prepare(args)
                args.expected_protocol_sha=worker.file_sha(args.run_dir/'protocol.json')
                calls=[]
                @contextmanager
                def lease(path):
                    calls.append(Path(path))
                    if refusal=='lease':raise BlockingIOError('GPU lease held by Base')
                    yield
                with patch.object(worker,'ACTUAL_RUNS',root/'actual/runs'),patch.object(worker.t2,'gpu_lease',lease),\
                     patch.object(worker,'collect_protocol',return_value=protocol),patch.dict(os.environ,worker.t2.ENVIRONMENT),\
                     patch.object(worker.t2,'import_helper',return_value=object()),\
                     patch.object(worker.t2,'gpu_idle_guard',return_value={'passed':False}),\
                     patch.dict(sys.modules,{'torch':None}):
                    with self.assertRaises((BlockingIOError,ValueError)):worker.execute(args)
                    self.assertEqual(worker.read(args.run_dir/'status.json')['status'],'failed_stop_no_retry')
                    failure=worker.read(args.run_dir/'failure.json')
                    self.assertEqual(failure['diagnostic_optimizer_updates_applied'],0)
                    self.assertFalse(failure['G3_passed']);self.assertFalse(failure['automatic_retry'])
                    with self.assertRaisesRegex(ValueError,'never-started'):worker.execute(args)
                self.assertEqual(calls,[root/'actual/runs'])


if __name__ == '__main__':unittest.main()
