"""Synthetic fixed-teacher val1024/test192 analysis; no real model scores/GPU."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import analyze_fixed_teacher_study as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def bound(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':a.sha(path)}


class Fixture:
    def __init__(self,root):
        self.root=root;self.val=root/'val';self.val.mkdir()
        calibration=root/'calibration.json';write(calibration,{'split':'train','quality_used':False})
        plan=root/'test_plan.json';write(plan,{'split':'test','metadata_only':True})
        integrity=root/'integrity.json';write(integrity,{'scope':'integrity_only','model_scores_used':False})
        self.exception={'scope':'integrity_only','audit_receipt':bound(integrity),'automated':True,
                        'human_quality_review':False,'model_scores_used':False}
        self.study={'schema':'fixed_teacher_study_v1','updates':2048,'arms':[
            dict(label='baseline',family='baseline',setting='baseline',weight=0.),
            dict(label='vl',family='hr_vae',setting='low',weight=.02),dict(label='vh',family='hr_vae',setting='high',weight=.2),
            dict(label='dl',family='hr_dino',setting='low',weight=.005),dict(label='dh',family='hr_dino',setting='high',weight=.05)],
            'calibration':bound(calibration),'evaluation':{'test_cohort_plan':bound(plan),
                'test_integrity_thumbnail_exception':self.exception}}
        write(root/'study.json',self.study);self.study_file=bound(root/'study.json')
        self.sources={}
        for i in range(11):
            path=root/'sources'/f'm{i}.py';path.parent.mkdir(exist_ok=True);path.write_text(f'#{i}\n');self.sources[path.name]=bound(path)
        done=root/'training.json';write(done,{'status':'completed','scientific_updates':10240})
        self.training={'scientific_updates':10240,'study_file':self.study_file,'files':{'complete':bound(done)}}
        self.roles=[dict(label='shared',arm='shared',family='baseline',setting='shared_initial',weight=0.,
                        update=0,step=8192,cumulative_training_seconds=0.,checkpoint={'sha256':'b'*64})]
        for j,arm in enumerate(self.study['arms']):
            for update in (512,1024,2048):
                self.roles.append({**arm,'label':f'{arm["label"]}_{update}','arm':arm['label'],'update':update,'step':update,
                    'cumulative_training_seconds':update*(.01+j*.002),
                    'checkpoint':{'sha256':hashlib.sha256(f'{j}{update}'.encode()).hexdigest()}})
        self.val_protocol,self.val_rows=self.phase(self.val,'val',self.roles)

    def phase(self,run,phase,roles,*,selection=None):
        run.mkdir(exist_ok=True);cohort=[{'record':{'id':f'{phase}c{i:02d}','source_id':f'{phase}s{i:02d}'}} for i in range(32)]
        seal=self.root/(phase+'_seal.json');write(seal,{'cohort':cohort,'split':phase})
        protocol={'schema':'fixed_teacher_clean_evaluation_protocol_v1','phase':phase,'sources':self.sources,
            'roles':roles,'study_file':self.study_file,'training_evidence':self.training,'cohort':cohort,
            'cohort_seal':bound(seal),'cohort_seal_digest':phase+'-fixed',
            'scope':dict(total_restores=len(roles)*64,roles=len(roles),source_split=phase,conditions=['clean'],
                         sampling_seeds=[42,43],primary_pool='all33_frames'),
            'data_independence':{'test_prior_exposure':'historical_models_previously_evaluated',**a._test_access_policy(self.study)}}
        if selection:
            protocol.update(selection_receipt=bound(selection),validation_selection=json.loads(selection.read_text()))
        write(run/'protocol.json',protocol);psha=a.sha(run/'protocol.json')
        lp={'baseline':{512:.4,1024:.3,2048:.25},'vl':{512:.3,1024:.22,2048:.18},
            'vh':{512:.31,1024:.24,2048:.19},'dl':{512:.28,1024:.19,2048:.17},'dh':{512:.29,1024:.2,2048:.16}}
        rows=[]
        for role in roles:
            for i,item in enumerate(cohort):
                for seed in (42,43):
                    value=.5 if role['update']==0 else lp[role['arm']][role['update']]
                    if phase=='test':value={'baseline':.2,'vl':.3,'dh':.35}[role['arm']]
                    metrics={'lpips_alex_v0_1':value+i*.001+(seed-42)*.0002,
                        'psnr_rgb_db':20.+role['update']/512-(.1 if role['arm']!='baseline' and role['update'] else 0.),
                        'ssim_rgb':.7+role['update']/20480-(.01 if role['arm']!='baseline' and role['update'] else 0.),
                        'tLP100_alex_v0_1_TecoGAN_crop':value*2,'tOF_Farneback_TecoGAN_crop':value}
                    row={k:role[k] for k in ('arm','family','setting','weight','update','step','cumulative_training_seconds')}
                    row.update(role=role['label'],record_id=item['record']['id'],source_id=item['record']['source_id'],sampling_seed=seed,
                         split=phase,condition='clean',training_seed=20261102,cohort_seal_digest=phase+'-fixed',
                         checkpoint_sha256=role['checkpoint']['sha256'],metrics=metrics)
                    path=run/'results'/role['label']/item['record']['id']/f'{seed}.json';write(path,{'observation':row,'restore_seconds':.2})
                    rows.append(row|{'row_metrics_path':str(path),'row_metrics_sha256':a.sha(path)})
        write(run/'observations.json',{'protocol_sha256':psha,'observations':rows})
        write(run/'summary.json',dict(protocol_sha256=psha,restores=len(rows)))
        write(run/'status.json',dict(status='completed',protocol_sha256=psha,restores=len(rows),training_updates=0,
                                    summary_sha256=a.sha(run/'summary.json')))
        return protocol,rows

    def selected_test(self):
        choices=a.compute(self.val)['selected_by_family_from_validation']
        final={r['arm']:r for r in self.roles if r['update']==2048}
        selected=[final['baseline'],final[choices['hr_vae']['arm']],final[choices['hr_dino']['arm']]]
        receipt={'study_file':self.study_file,'selected_after_complete_validation':True,'test_pixels_read_by_selection':False,
            **a._test_access_policy(self.study),
            'selected_roles':selected,**{key:bound(self.val/name) for key,name in (
                ('validation_protocol','protocol.json'),('validation_status','status.json'),
                ('validation_summary','summary.json'),('validation_observations','observations.json'))}}
        path=self.root/'selection.json';write(path,receipt);run=self.root/'test'
        protocol,rows=self.phase(run,'test',selected,selection=path)
        return run,protocol,path


class FixedAnalysisTests(unittest.TestCase):
    def test_full1024_five_arm_curves_and_independent_family_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));r=a.compute(f.val)
            self.assertEqual(r['matrix_rows'],1024);self.assertEqual(r['scientific_training_updates'],10240)
            self.assertEqual(len(r['curves']),5);self.assertEqual(len(r['paired_comparisons']),4)
            self.assertEqual(r['selected_by_family_from_validation']['hr_vae']['arm'],'vl')
            self.assertEqual(r['selected_by_family_from_validation']['hr_dino']['arm'],'dh')
            self.assertTrue(r['final_comparisons']['vl']['PSNR_fidelity_tradeoff'])
            self.assertFalse(r['bootstrap']['p_values_computed'])
            self.assertEqual(r['test_access_policy']['test_integrity_thumbnail_exception'],f.exception)
            self.assertFalse(r['test_access_policy']['no_preselection_test_pixel_access_claimed'])

    def test_integrity_exception_tamper_or_omitted_disclosure_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));protocol=copy.deepcopy(f.val_protocol)
            protocol['data_independence'].pop('test_integrity_thumbnail_exception')
            write(f.val/'protocol.json',protocol)
            state=json.loads((f.val/'status.json').read_text());state['protocol_sha256']=a.sha(f.val/'protocol.json')
            summary=json.loads((f.val/'summary.json').read_text());summary['protocol_sha256']=state['protocol_sha256']
            write(f.val/'summary.json',summary);state['summary_sha256']=a.sha(f.val/'summary.json');write(f.val/'status.json',state)
            with self.assertRaisesRegex(ValueError,'integrity-only access disclosure'):a.compute(f.val)
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));write(Path(f.exception['audit_receipt']['path']),{'scope':'changed'})
            with self.assertRaises(ValueError):a.compute(f.val)

    def test_test192_keeps_val_choices_even_when_test_quality_order_is_worse(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));run,_,_=f.selected_test();r=a.compute(run)
            self.assertEqual(r['matrix_rows'],192);self.assertEqual(len(r['curves']),3)
            self.assertEqual(r['selected_by_family_from_validation']['hr_vae']['arm'],'vl')
            self.assertEqual(r['selected_by_family_from_validation']['hr_dino']['arm'],'dh')
            self.assertEqual(r['final_comparisons']['vl']['LPIPS_status'],'no_observed_gain')
            self.assertEqual(r['efficiency']['status'],'not_estimated_from_test')
            self.assertFalse(r['heldout_test_used_to_choose_weights']);self.assertFalse(r['historically_pristine_test'])

    def test_wrong_val_choice_rejected_before_test_observations_are_read(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));run,protocol,receipt=f.selected_test()
            value=json.loads(receipt.read_text());wrong=next(x for x in f.roles if x['arm']=='vh' and x['update']==2048)
            value['selected_roles'][1]=wrong;write(receipt,value)
            protocol.update(selection_receipt=bound(receipt),validation_selection=value);write(run/'protocol.json',protocol)
            state=json.loads((run/'status.json').read_text());state['protocol_sha256']=a.sha(run/'protocol.json');write(run/'status.json',state)
            summary=json.loads((run/'summary.json').read_text());summary['protocol_sha256']=state['protocol_sha256'];write(run/'summary.json',summary)
            state['summary_sha256']=a.sha(run/'summary.json');write(run/'status.json',state)
            original=a.read;seen=[]
            def tracking(path):seen.append(Path(path));return original(path)
            with patch.object(a,'read',side_effect=tracking):
                with self.assertRaisesRegex(ValueError,'not selected solely'):a.compute(run)
            self.assertNotIn(run/'observations.json',seen)

    def test_incomplete_or_duplicate_phase_rows_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));path=f.val/'observations.json';bundle=json.loads(path.read_text())
            bundle['observations'][-1]=copy.deepcopy(bundle['observations'][0]);write(path,bundle)
            with self.assertRaisesRegex(ValueError,'Duplicate'):a.compute(f.val)

    def test_all_per_video_CSV_rows_values_pair_signs_and_candidates_are_lossless(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f=Fixture(root);out=root/'tables';report=a.analyze(f.val,out,make_plots=False)
            def rows(name):
                with (out/name).open(newline='') as stream:return list(csv.DictReader(stream))
            clips=rows('per_clip_metrics.csv');pairs=rows('paired_clip_differences.csv')
            self.assertEqual(len(clips),512);self.assertEqual(len(pairs),544)
            self.assertEqual(len({(x['role'],x['record_id']) for x in clips}),512)
            self.assertEqual(sum(x['role']=='shared' for x in clips),32)
            self.assertAlmostEqual(float(clips[0]['lpips_alex_v0_1']),.5001)
            item=next(x for x in pairs if x['method_arm']=='vl' and x['reference_arm']=='baseline' and x['update']=='512')
            self.assertAlmostEqual(float(item['lpips_alex_v0_1']),.1)
            self.assertAlmostEqual(float(item['psnr_rgb_db']),-.1)
            self.assertEqual(len(rows('summary.csv')),16);self.assertEqual(len(rows('paired_summary.csv')),85)
            candidates=rows('candidate_settings.csv');self.assertEqual(len(candidates),5)
            self.assertEqual({x['arm'] for x in candidates if x['selected_for_test']=='True'},{'baseline','vl','dh'})
            manifest=json.loads((out/'delivery_manifest.json').read_text())
            for artifact in manifest['artifacts']:self.assertEqual(a.sha(artifact['path']),artifact['sha256'])
            self.assertEqual(report['cost_accounting']['evaluation_execution']['canonical_successful_restore_seconds_sum'],204.8)
            cost=next(x for x in rows('costs.csv') if x['measurement']=='execution_wall_seconds')
            self.assertEqual(cost['value'],'');self.assertEqual(cost['available'],'False')

    def test_selected_test_CSV_retains_all_five_settings_and_only_three_measured_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f=Fixture(root);run,_,_=f.selected_test();out=root/'test_tables'
            report=a.analyze(run,out,make_plots=False)
            def rows(name):
                with (out/name).open(newline='') as stream:return list(csv.DictReader(stream))
            self.assertEqual(len(rows('per_clip_metrics.csv')),96)
            self.assertEqual(len(rows('paired_clip_differences.csv')),96)
            self.assertEqual(len(rows('summary.csv')),3);self.assertEqual(len(rows('paired_summary.csv')),15)
            candidates=rows('candidate_settings.csv');self.assertEqual(len(candidates),5)
            self.assertEqual(sum(x['evaluated_in_this_phase']=='True' for x in candidates),3)
            self.assertEqual(sum(x['selected_for_test']=='True' for x in candidates),3)
            self.assertEqual(report['selected_by_family_from_validation']['hr_vae']['arm'],'vl')

    def test_bound_training_cache_calibration_and_eval_cost_diagnostics_do_not_double_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f=Fixture(root)
            ledger={'committed_training_step_seconds':800.,'known_discarded_completed_step_seconds':12.,
                'known_discarded_completed_step_returns':3,'wall_seconds_all_training_attempts':1000.}
            training_summary=root/'training_summary.json'
            write(training_summary,{'compute_cost':ledger,'arms':{'baseline':{'cumulative_training_seconds':100.,'updates':2048}}})
            cache=root/'cache_completion.json'
            write(cache,{'timings':{'total_cache_build_seconds':174.45672749297228,
                'teacher_feature_extraction_seconds':96.91283185733482,'other_io_binding_seconds':77.54389563563745,
                'pure_GPU_encoder_seconds':None,'feature_scope':'Includes CPU resize/H2D/FP32forward/CPUcopy/grouping.',
                'total_scope':'Builder through seal; excludes caller admission/final cleanup.'},
                'training_count':1641,'validation_count':32,'teacher_frames':55209,
                'new_VAE_feature_extraction_seconds_this_round':0.,'VAE_cache_cost':'shared sunk preparation'})
            training_protocol=root/'training_protocol.json';write(training_protocol,{'dino_cache':{'completion_receipt':bound(cache)}})
            calibration=Path(f.study['calibration']['path'])
            write(calibration,{'preflight_wall_seconds_including_metadata_io':30.,
                'gradient_calibration_wall_seconds_including_fresh_model_load':{'vae':4.,'dino':5.},
                'calibrations':{'vae':{'weights':[.02,.2],'rows':[{'sigma':.2,'native_sigma':.2001,'head_unit_aux_grad_norm':2.}]}},
                'resources':{'hr_vae':{'timed6_mean_seconds':.2,'timed6_sample_SD_seconds':.01,'memory':{'peak_allocated_bytes':123}}},
                'HR_VAE_reconstruction_diagnostics':[{'source_cache_index':0,'diagnosis_only_not_SR_upper_bound':True,'metrics':{'lpips_alex_v0_1':.01}}]})
            f.study['calibration']=bound(calibration);write(root/'study.json',f.study);f.study_file=bound(root/'study.json')
            protocol=copy.deepcopy(f.val_protocol);protocol['study_file']=f.study_file
            protocol['training_evidence']['study_file']=f.study_file
            protocol['training_evidence']['files'].update({'summary.json':bound(training_summary),'protocol.json':bound(training_protocol)})
            write(f.val/'protocol.json',protocol)
            psha=a.sha(f.val/'protocol.json');summary=json.loads((f.val/'summary.json').read_text());summary['protocol_sha256']=psha
            write(f.val/'summary.json',summary);state=json.loads((f.val/'status.json').read_text())
            state.update(protocol_sha256=psha,summary_sha256=a.sha(f.val/'summary.json'));write(f.val/'status.json',state)
            bundle=json.loads((f.val/'observations.json').read_text());bundle['protocol_sha256']=psha;write(f.val/'observations.json',bundle)
            wall={'schema':'fixed_teacher_execution_wall_v1','status':'completed','stage':'evaluation_val',
                'run':str(f.val),'protocol_sha256':psha,'wall_seconds':250.,'scope':'Execute subprocess start through cleanup/exit; excludes prepare/verify.'}
            write(f.val/'execution_wall_time.json',wall);out=root/'cost_tables';report=a.analyze(f.val,out,make_plots=False)
            with (out/'costs.csv').open(newline='') as stream:costs=list(csv.DictReader(stream))
            cache_total=next(x for x in costs if x['measurement']=='total_cache_build_seconds')
            self.assertEqual(float(cache_total['value']),174.45672749297228)
            pure=next(x for x in costs if x['measurement']=='pure_GPU_encoder_seconds')
            self.assertEqual(pure['value'],'');self.assertEqual(pure['available'],'False')
            wallrow=next(x for x in costs if x['measurement']=='execution_wall_seconds')
            self.assertEqual(float(wallrow['value']),250.);self.assertEqual(wallrow['pure_GPU_time'],'False')
            self.assertIsNone(report['cost_accounting']['overall_total_seconds'])
            self.assertEqual(report['cost_accounting']['training_attempts']['compute_cost'],ledger)
            with (out/'calibration_diagnostics.csv').open(newline='') as stream:diagnostics=list(csv.DictReader(stream))
            paths={x['field_path'] for x in diagnostics}
            self.assertIn('calibrations.vae.rows.0.head_unit_aux_grad_norm',paths)
            self.assertIn('resources.hr_vae.memory.peak_allocated_bytes',paths)
            self.assertIn('HR_VAE_reconstruction_diagnostics.0.diagnosis_only_not_SR_upper_bound',paths)
            wall['protocol_sha256']='0'*64;write(f.val/'execution_wall_time.json',wall)
            with self.assertRaisesRegex(ValueError,'wall receipt'):a.compute(f.val)

    def test_time_axis_is_committed_wall_with_separate_bound_cost_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f=Fixture(root)
            ledger={'committed_training_step_seconds':800.,'known_discarded_completed_step_seconds':12.,
                'known_discarded_completed_step_returns':3,'wall_seconds_all_training_attempts':1000.}
            training_summary=root/'training_summary.json';write(training_summary,{'compute_cost':ledger})
            cache=root/'cache_completion.json';cache_time={'total_cache_build_seconds':20.,
                'teacher_feature_extraction_seconds':16.,'other_io_binding_seconds':4.,'pure_GPU_encoder_seconds':None}
            write(cache,{'timings':cache_time,'VAE_cache_cost':'existing shared sunk preparation'})
            training_protocol=root/'training_protocol.json';write(training_protocol,{'dino_cache':{'completion_receipt':bound(cache)}})
            calibration=Path(f.study['calibration']['path'])
            write(calibration,{'preflight_wall_seconds_including_metadata_io':30.,
                'gradient_calibration_wall_seconds_including_fresh_model_load':{'vae':4.,'dino':5.}})
            f.study['calibration']=bound(calibration);write(root/'study.json',f.study)
            f.study_file=bound(root/'study.json');protocol=copy.deepcopy(f.val_protocol)
            protocol['study_file']=f.study_file;protocol['training_evidence']['study_file']=f.study_file
            protocol['training_evidence']['files'].update({'summary.json':bound(training_summary),'protocol.json':bound(training_protocol)})
            write(f.val/'protocol.json',protocol)
            state=json.loads((f.val/'status.json').read_text());state['protocol_sha256']=a.sha(f.val/'protocol.json')
            summary=json.loads((f.val/'summary.json').read_text());summary['protocol_sha256']=state['protocol_sha256']
            write(f.val/'summary.json',summary);state['summary_sha256']=a.sha(f.val/'summary.json');write(f.val/'status.json',state)
            bundle=json.loads((f.val/'observations.json').read_text());bundle['protocol_sha256']=state['protocol_sha256'];write(f.val/'observations.json',bundle)
            report=a.compute(f.val);axis=report['time_axis_contract'];cost=report['cost_accounting']
            self.assertFalse(axis['pure_GPU_elapsed']);self.assertFalse(axis['total_experiment_cost'])
            self.assertIn('Committed synchronized training-step wall',axis['meaning'])
            self.assertEqual(cost['training_attempts']['compute_cost'],ledger)
            self.assertEqual(cost['offline_DINO_cache']['timings'],cache_time)
            self.assertEqual(cost['train_only_calibration']['preflight_wall_seconds_including_metadata_io'],30.)
            self.assertIsNone(cost['overall_total_seconds']);self.assertTrue(cost['missing_costs_are_unknown_not_zero'])
            self.assertIn('Do not sum',cost['aggregation_rule'])
            markdown=a.markdown(report)
            self.assertIn('既不是纯GPU elapsed/kernel时间，也不是总实验花费',markdown)
            self.assertIn('checkpoint与评价I/O',markdown)

    def test_five_arm_scientific_plot_delivery(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f=Fixture(root);out=root/'output';a.analyze(f.val,out)
            self.assertTrue((out/'analysis.json').is_file());self.assertIn('HR-DINO',(out/'report.md').read_text())
            for name in ('quality_vs_updates.png','quality_vs_training_seconds.png'):
                self.assertEqual((out/name).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
                self.assertGreater((out/name).stat().st_size,10000)


if __name__=='__main__':unittest.main(verbosity=2)
