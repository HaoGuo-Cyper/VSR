"""Stdlib-only prospective pilot evaluation fixtures; no pixels/tensors/GPU."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/'scripts'))
import stage4_evaluate_pilot as worker


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


@contextmanager
def cwd(path):
    previous = Path.cwd();os.chdir(path)
    try:yield
    finally:os.chdir(previous)


def criteria(evaluation):
    # Fixture definitions only; this is not a real project G4 decision rule.
    return {'schema':'stage4_G4_prospective_criteria_v1','status':'frozen','frozen_by':'CPU fixture author',
        'decision_reference':'synthetic CPU fixture authorized plan only','frozen_utc':'2026-01-01T00:00:00Z',
        'score_access_before_freeze':False,'scope':{'training_seed':worker.TRAINING_SEED,'arms':worker.ARMS,
            'updates':2048,'cohort_seal_digest':evaluation['cohort_seal_digest'],
            'Base_evaluation_protocol_sha256':evaluation['protocol_sha256']},
        'definitions':{name:'Synthetic fixture definition for '+name for name in worker.CRITERIA_DEFINITIONS}}


def score(psnr,lpips,tlp,mask):
    pools = {pool:{'psnr_rgb_db':psnr+offset,'ssim_rgb':.7+offset*.001,
                   'lpips_alex_v0_1':lpips-offset*.001} for pool,offset in [('masked',0),('neighbors',2),('all_frames',1)]}
    return {'mask':list(mask),'pools':pools,'temporal':{'mask_touch_tLP100':tlp,'all28pairs_tLP100':tlp+.2},
        'raw_harm_gap':{'psnr_rgb_db':pools['neighbors']['psnr_rgb_db']-pools['masked']['psnr_rgb_db'],
            'lpips_alex_v0_1':pools['masked']['lpips_alex_v0_1']-pools['neighbors']['lpips_alex_v0_1']}}


def matrix_fixture():
    cohort = [{'record':{'id':f'r{i}','source_id':f's{i}'},'mask':[False]*4+[True]*8+[False]*21} for i in range(32)]
    seal = {'cohort':cohort,'seal_digest':'cohort-fixture'}
    hashes = {arm:str(i+1)*64 for i,arm in enumerate(worker.ARMS)}
    base_hashes = {str(seed):str(i+4)*64 for i,seed in enumerate(worker.t2.SEEDS)}
    protocol = {'checkpoints':{arm:{'checkpoint':{'sha256':sha}} for arm,sha in hashes.items()},
        'evaluation_protocol':{'checkpoints':{seed:{'checkpoint':{'sha256':sha}} for seed,sha in base_hashes.items()}},
        'G4_criteria':{'file':{'path':'/fixture/approved.json','sha256':'a'*64}}}
    rows,bases = [],[]
    for item in cohort:
        for condition_index,condition in enumerate(worker.CONDITIONS):
            for sample in worker.SAMPLES:
                common = {'record_id':item['record']['id'],'source_id':item['record']['source_id'],
                    'cohort_seal_digest':seal['seal_digest'],'condition':condition,'sampling_seed':sample}
                clip_index = int(item['record']['id'][1:]);sample_effect = 10 if sample==42 else -10
                for arm in worker.ARMS:
                    factor = {'flow_only':0.,'C1':.25,'A2':.75}[arm]
                    improvement = factor*condition_index
                    rows.append({**common,'arm':arm,'training_seed':worker.TRAINING_SEED,'method_updates':2048,
                        'checkpoint_sha256':hashes[arm],'score':score(30+clip_index*.01+sample_effect+improvement,
                            .3+clip_index*.001-improvement*.1,5-improvement*.2,item['mask'])})
                for seed_index,seed in enumerate(worker.t2.SEEDS):
                    bases.append({**common,'training_seed':seed,'checkpoint_sha256':base_hashes[str(seed)],
                        'score':score(30+clip_index*.01+sample_effect+seed_index,
                            .3+clip_index*.001+seed_index*.01,5+seed_index*.1,item['mask'])})
    return rows,bases,seal,protocol


class ProspectiveCriteriaTests(unittest.TestCase):
    def test_import_and_CLI_help_are_stdlib_only(self):
        code=f"import sys;sys.dont_write_bytecode=True;sys.path.insert(0,{str(ROOT/'scripts')!r});import stage4_evaluate_pilot;assert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/stage4_evaluate_pilot.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--expected-g4-criteria-sha256',result.stdout)

    def test_valid_criteria_bound_without_defining_or_computing_a_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation={'cohort_seal_digest':'cohort','protocol_sha256':'b'*64};path=Path(temporary)/'criteria.json'
            value=criteria(evaluation);write(path,value)
            result=worker.criteria_spec(path,worker.file_sha(path),evaluation,prepared_utc='2026-11-06T00:00:00Z')
            self.assertEqual(result['criteria'],value);self.assertFalse(result['automatic_G4_verdict'])
            with self.assertRaises(ValueError):worker.criteria_spec(path,'f'*64,evaluation)
            with self.assertRaises(ValueError):worker.criteria_spec(path,None,evaluation)

    def test_missing_placeholder_unfrozen_postscore_future_and_wrong_scope_fail(self):
        changes=[lambda x:x.update(status='draft'),lambda x:x.update(frozen_by=''),
            lambda x:x.update(score_access_before_freeze=True),lambda x:x.update(decision_reference='TODO'),
            lambda x:x['definitions'].pop('clean_fidelity'),
            lambda x:x['definitions'].update(base_seed_spread='TBD'),
            lambda x:x['definitions'].update(decision_rule={'threshold':'待定'}),
            lambda x:x['scope'].update(arms=['A2']),lambda x:x['scope'].update(training_seed=20261103),
            lambda x:x.update(frozen_utc='2099-01-01T00:00:00Z'),lambda x:x.update(frozen_utc='2026-01-01')]
        with tempfile.TemporaryDirectory() as temporary:
            evaluation={'cohort_seal_digest':'cohort','protocol_sha256':'b'*64};path=Path(temporary)/'criteria.json'
            for change in changes:
                with self.subTest(change=change):
                    value=criteria(evaluation);change(value);write(path,value)
                    with self.assertRaises(ValueError):worker.criteria_spec(path,worker.file_sha(path),evaluation,prepared_utc='2026-11-06T00:00:00Z')


class CompletePairedStatisticsTests(unittest.TestCase):
    def test_single_seed_three_arms_complete_pairing_and_Base_spread(self):
        rows,bases,seal,protocol=matrix_fixture();report=worker.summarize(rows,bases,seal,protocol)
        self.assertEqual(report['method_observations'],1152);self.assertEqual(report['Base_observations_reused'],1152)
        self.assertEqual(report['method_training_seed_count'],1);self.assertEqual(report['Base_training_seed_count'],3)
        pair=report['paired_comparisons']['A2_minus_C1']['equal_five_type_macro']
        self.assertAlmostEqual(pair['masked.psnr_rgb_db']['mean'],1.5)
        self.assertAlmostEqual(pair['masked.lpips_alex_v0_1']['mean'],-.15)
        base=report['descriptive_Base_seed_spread']['clean']['masked.psnr_rgb_db']
        self.assertAlmostEqual(base['sample_SD_across_three_Base_training_seed_means_ddof1'],1.)
        self.assertFalse(base['automatic_G4_threshold']);self.assertFalse(report['p_values_computed'])
        self.assertEqual(report['G4_decision'],'not_computed_requires_review')
        self.assertEqual(len(report['paired_clip_appendix']),6*6*32)
        self.assertEqual(set(report['per_arm']),set(worker.ARMS));self.assertEqual(len(report['per_arm']['A2']),7)
        json.dumps(report,allow_nan=False)
        csv=worker.compact_csv(report);self.assertIn('training_seed_means,3',csv)
        self.assertIn('clips_after_sampler_mean,32',csv)

    def test_sampler_means_precede_between_clip_SD_and_clean_subtraction(self):
        rows,bases,seal,protocol=matrix_fixture();report=worker.summarize(rows,bases,seal,protocol)
        clean=report['per_arm']['A2']['clean']['masked.psnr_rgb_db']
        self.assertAlmostEqual(clean['between_clip_sample_SD'],statistics.stdev(i*.01 for i in range(32)))
        appendix=next(x for x in report['paired_clip_appendix'] if x['comparison']=='A2_minus_C1' and x['condition']=='blur')
        self.assertAlmostEqual(appendix['candidate_minus_reference']['masked.psnr_rgb_db'],.5)
        self.assertAlmostEqual(appendix['clean_subtracted_candidate_minus_reference']['masked.psnr_rgb_db'],.5)
        # Every type remains represented, including an unfavorable one.
        for row in rows:
            if row['arm']=='A2' and row['condition']=='drop_black':row['score']['pools']['masked']['psnr_rgb_db']-=100
        changed=worker.summarize(rows,bases,seal,protocol)
        self.assertAlmostEqual(changed['paired_comparisons']['A2_minus_C1']['equal_five_type_macro']['masked.psnr_rgb_db']['mean'],1.5-20)

    def test_missing_duplicate_type_wrong_seed_mask_source_checkpoint_and_nonfinite_fail(self):
        changes=[lambda x:x.pop(),lambda x:x.__setitem__(-1,copy.deepcopy(x[0])),
            lambda x:x[-1].update(training_seed=20261103),lambda x:x[-1].update(method_updates=4096),
            lambda x:x[-1].update(condition='chosen_favorable_type'),lambda x:x[-1].update(source_id='wrong'),
            lambda x:x[-1].update(checkpoint_sha256='f'*64),lambda x:x[-1]['score']['mask'].__setitem__(0,True),
            lambda x:x[-1]['score']['pools']['masked'].update(lpips_alex_v0_1=float('nan'))]
        for change in changes:
            with self.subTest(change=change):
                rows,bases,seal,protocol=matrix_fixture();change(rows)
                with self.assertRaises(ValueError):worker.summarize(rows,bases,seal,protocol)
        rows,bases,seal,protocol=matrix_fixture();bases.pop()
        with self.assertRaises(ValueError):worker.summarize(rows,bases,seal,protocol)


class MetadataPreparationAndFailureTests(unittest.TestCase):
    def _pilot_fixture(self,root):
        # Real stdlib method_config from the separate runner, using the sealed
        # Base source template. Tensor files remain deliberately undecodable.
        helper=worker.pilot_helper(ROOT)
        raw=ROOT/'configs/stage2_20261105/base_seed20261102.json'
        base={'config':worker.t2.expected_effective_config(raw),'checkpoint':{'path':'/fixture/actual/Base/checkpoint_008192.pt','sha256':'a'*64}}
        directory=root/'runs/pilot';configs={};arms={}
        for arm in worker.ARMS:
            cfg=helper.method_config(base['config'],arm);configs[arm]=cfg
            run=directory/'arm_runs'/arm;path=run/arm/'checkpoint_002048.pt'
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'not-loaded-by-metadata-prepare')
            receipt={'path':str(path),'sha256':worker.file_sha(path),'variant':arm,'step':2048,
                'inference_parameters':'student','restoration_architecture':'noref_base_v1'}
            report={'status':'completed','arm':arm,'variant':arm,'updates':2048,'training_seed':worker.TRAINING_SEED,
                'config_sha256':cfg['_sha256'],'final_checkpoint':receipt,'final_trainable_sha256':'b'*64}
            arms[arm]=report;write(run/'config.json',cfg);write(run/'summary.json',report)
            write(run/'status.json',{'status':'completed','updates':2048,'summary_sha256':worker.file_sha(run/'summary.json')})
        base_directory=root/'runs/Base_evaluation';assets=root/'runs/assets'
        evaluation={'project_root':str(root),'cohort_seal_digest':'cohort-fixture',
            'dataset_manifest':{'path':str(assets/'dataset.json')}}
        write(base_directory/'protocol.json',evaluation);write(base_directory/'observations.json',{'fixture':'never-scored'})
        write(base_directory/'summary.json',{'fixture':'already-completed-T2-contract'})
        pilot_protocol={'schema':'stage4_pilot_training_v1','matched_Base':base,'method_configs':configs,
            'quality_evaluation':{'criteria':None},'completed_G3':{'directory':str(root/'runs/G3')},
            'g3_protocol':{'evaluation_protocol':evaluation,'completed_T2':{'directory':str(base_directory)}}}
        write(directory/'protocol.json',pilot_protocol)
        summary={'status':'completed_pilot_training','completed_arms':worker.ARMS,'scientific_training_updates':6144,
            'G4_evaluated':False,'arms':arms,'protocol_sha256':worker.file_sha(directory/'protocol.json')}
        write(directory/'summary.json',summary)
        write(directory/'status.json',{'status':'completed_pilot_training','summary_sha256':worker.file_sha(directory/'summary.json'),
            'protocol_sha256':worker.file_sha(directory/'protocol.json')})
        path=root/'criteria.json';write(path,criteria({'cohort_seal_digest':evaluation['cohort_seal_digest'],
            'protocol_sha256':worker.file_sha(base_directory/'protocol.json')}))
        for relative in ('scripts/stage4_evaluate_pilot.py','scripts/stage4_pilot.py','tests/test_stage4_evaluate_pilot.py'):
            p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# metadata fixture only\n')
        args=SimpleNamespace(project_root=root,pilot_run=directory,run_dir=root/'runs/evaluation',g4_criteria=path,
            expected_g4_criteria_sha256=worker.file_sha(path),expected_protocol_sha=None)
        return args,pilot_protocol,evaluation,SimpleNamespace(completed_pilot=lambda directory,root:(pilot_protocol,{'directory':str(directory)}))

    def test_collect_uses_completed_pilot_G2_G3_contract_and_staging_student_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);args,pilot,evaluation,helper=self._pilot_fixture(root)
            with (cwd(root),patch.object(worker,'__file__',str(root/'scripts/stage4_evaluate_pilot.py')),
                  patch.object(worker,'pilot_helper',return_value=helper) as prerequisite,
                  patch.object(worker.g3,'completed_t2',return_value=(evaluation,{'directory':str(root/'runs/Base_evaluation')})) as original):
                result=worker.collect_protocol(args)
                self.assertEqual(prerequisite.call_args.args,(root,));self.assertEqual(original.call_args.args[1],root)
                self.assertEqual(result['scope']['total_restores_and_observations'],1152)
                self.assertEqual(result['scope']['method_training_seeds'],[worker.TRAINING_SEED])
                self.assertEqual(result['scope']['Base_restores'],0)
                self.assertEqual(result['allowed_method_checkpoint_root'],str(root/'runs'))
                self.assertEqual(set(result['checkpoints']),set(worker.ARMS))
                self.assertEqual(result,json.loads(json.dumps(result)))
                self.assertFalse(result['statistical_scope']['G4_verdict_computed'])
                self.assertEqual(result['output']['G4_decision'],'not_computed_requires_review')
                # An existing training-bound criteria SHA cannot be replaced.
                pilot['quality_evaluation']['criteria']={'sha256':'f'*64}
                with self.assertRaisesRegex(ValueError,'already bound'):worker.collect_protocol(args)

    def test_wrong_method_config_checkpoint_or_completion_rejected_without_tensor_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);args,pilot,evaluation,helper=self._pilot_fixture(root)
            summary=worker.read(args.pilot_run/'summary.json');arm='A2';run=args.pilot_run/'arm_runs'/arm
            configs=worker.read(run/'config.json')
            for mutate in (lambda x:x['training'].update(updates=4096),lambda x:x['training'].update(variants=['Base']),
                           lambda x:x['conditioning'].update(alpha=16),lambda x:x.update(restoration_architecture='unknown')):
                cfg=copy.deepcopy(configs);mutate(cfg);cfg.pop('_sha256')
                cfg['_sha256']=hashlib.sha256((json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
                write(run/'config.json',cfg)
                with self.assertRaises(ValueError):worker.method_checkpoints(args.pilot_run,pilot,summary,root)
            write(run/'config.json',configs)
            cp=run/arm/'checkpoint_002048.pt';cp.write_bytes(b'changed-checkpoint')
            with self.assertRaises(ValueError):worker.method_checkpoints(args.pilot_run,pilot,summary,root)
            with (cwd(root),patch.object(worker,'__file__',str(root/'scripts/stage4_evaluate_pilot.py')),
                  patch.object(worker,'pilot_helper',return_value=SimpleNamespace(completed_pilot=lambda *a:(_ for _ in ()).throw(ValueError('real G3 prerequisite failed'))))):
                with self.assertRaisesRegex(ValueError,'real G3'):worker.collect_protocol(args)

    def test_prepare_in_clean_real_subprocess_and_refuses_reuse(self):
        code=f"import importlib.util,sys;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location('stage4_metadata_fixture',{str(Path(__file__).resolve())!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.MetadataPreparationAndFailureTests()._prepare_fixture();assert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def _prepare_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);args=SimpleNamespace(project_root=root,pilot_run=root/'runs/pilot',run_dir=root/'runs/eval')
            protocol={'prerequisite_directories':[str(args.pilot_run),str(root/'runs/assets')],
                'G4_criteria':{'file':{'sha256':'a'*64}},'fixture':'metadata-only'}
            with patch.object(worker,'collect_protocol',return_value=protocol):
                result=worker.prepare(args);self.assertEqual(result['status'],'planned')
                self.assertFalse(worker.read(args.run_dir/'status.json')['torch_imported'])
                self.assertEqual(worker.read(args.run_dir/'protocol.json'),protocol)
                with self.assertRaises(ValueError):worker.prepare(args)
                args.run_dir=root/'runs/assets/eval'
                with self.assertRaises(ValueError):worker.prepare(args)
            args.run_dir=args.pilot_run/'nested'
            with self.assertRaises(ValueError):worker.safe_run(args)

    def _planned(self,root,protocol):
        args=SimpleNamespace(project_root=root,pilot_run=root/'runs/pilot',run_dir=root/'runs/eval',expected_protocol_sha=None)
        with patch.object(worker,'collect_protocol',return_value=protocol):worker.prepare(args)
        args.expected_protocol_sha=worker.file_sha(args.run_dir/'protocol.json')
        return args

    def test_busy_shared_lease_and_failed_GPU_guard_stop_before_torch_no_retry(self):
        for refusal in ('lease','guard'):
            with self.subTest(refusal=refusal),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);actual=root/'actual/runs';actual.mkdir(parents=True)
                protocol={'prerequisite_directories':[],'G4_criteria':{'file':{'sha256':'a'*64}},
                    'evaluation_protocol':{'helpers':{'gpu_classifier':{'path':'fixture'}},'desktop_binding':{'fixture':True}}}
                # Pure prepare is deliberately separate from imported-torch test
                # processes; create the equivalent planned metadata directly.
                args=SimpleNamespace(project_root=root,pilot_run=root/'runs/pilot',run_dir=root/'runs/eval',expected_protocol_sha=None)
                write(args.run_dir/'protocol.json',protocol);args.expected_protocol_sha=worker.file_sha(args.run_dir/'protocol.json')
                write(args.run_dir/'status.json',{'status':'planned','protocol_sha256':args.expected_protocol_sha})
                @contextmanager
                def lease(path):
                    if refusal=='lease':raise BlockingIOError('CPU fixture competing owner')
                    yield
                original_import=__import__
                def no_torch(name,*a,**kw):
                    if name=='torch':raise AssertionError('GPU guard must fail before importing torch')
                    return original_import(name,*a,**kw)
                with patch.object(worker.g3,'ACTUAL_RUNS',actual),patch.object(worker.t2,'gpu_lease',lease),\
                     patch.object(worker,'collect_protocol',return_value=protocol),patch.dict(os.environ,worker.t2.ENVIRONMENT),\
                     patch.object(worker.t2,'import_helper',return_value=object()),patch.object(worker.t2,'gpu_idle_guard',return_value={'passed':False}),\
                     patch('builtins.__import__',side_effect=no_torch):
                    with self.assertRaises((BlockingIOError,ValueError)):worker.execute(args)
                    status=worker.read(args.run_dir/'status.json');self.assertEqual(status['status'],'failed_stop_no_retry')
                    self.assertEqual(status['training_updates'],0);self.assertEqual(status['G4_decision'],'not_computed_requires_review')
                    with self.assertRaisesRegex(ValueError,'never-started'):worker.execute(args)

    def test_cleanup_double_error_preserves_primary_and_completed_summary_is_revoked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);actual=root/'actual/runs';actual.mkdir(parents=True)
            protocol={'prerequisite_directories':[],'G4_criteria':{'file':{'sha256':'a'*64}}}
            args=SimpleNamespace(project_root=root,pilot_run=root/'runs/pilot',run_dir=root/'runs/eval',expected_protocol_sha=None)
            write(args.run_dir/'protocol.json',protocol);args.expected_protocol_sha=worker.file_sha(args.run_dir/'protocol.json')
            write(args.run_dir/'status.json',{'status':'planned','protocol_sha256':args.expected_protocol_sha})
            with (patch.object(worker.g3,'ACTUAL_RUNS',actual),
                  patch.object(worker,'collect_protocol',side_effect=RuntimeError('primary CPU fixture')),
                  patch.object(worker.gc,'collect',side_effect=RuntimeError('secondary cleanup fixture'))):
                with self.assertRaisesRegex(RuntimeError,'primary CPU fixture'):worker.execute(args)
            self.assertIn('primary CPU fixture',worker.read(args.run_dir/'failure.json')['error'])
            self.assertIn('secondary cleanup fixture',worker.read(args.run_dir/'cleanup_failure.json')['secondary_cleanup_error'])
            run=root/'completed';write(run/'summary.json',{'status':'completed_full_method_matrix_descriptive_only'})
            worker._failure(run,RuntimeError('cleanup-only'),'',args.expected_protocol_sha,'cleanup',[],[],0.)
            self.assertEqual(worker.read(run/'summary.json')['status'],'failed_stop_no_retry')
            self.assertTrue(worker.read(run/'summary.json')['completion_revoked_by_execution_failure'])


if __name__=='__main__':unittest.main()
