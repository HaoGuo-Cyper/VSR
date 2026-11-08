"""Complete448/128 synthetic analysis tests; no model weights, pixels or GPU."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import analyze_videomaev2_study as mod


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n')


class Fixture:
    def __init__(self,root):
        self.root=root
        sources={}
        for i in range(11):
            p=root/'sources'/f'{i}.py';p.parent.mkdir(exist_ok=True);p.write_text('# synthetic');sources[p.name]=mod.binding(p)
        self.sources=sources;self.arms=[dict(label=a,weight=0. if a=='baseline' else .5) for a in mod.ARMS]
        evaluation={'test_prior_exposure':'historical_models_previously_evaluated'}
        for name in ('validation_seal','validation_dataset','test_cohort_plan','test_dataset'):
            p=root/'assets'/(name+'.json');write(p,{'name':name});evaluation[name]=mod.binding(p)
        self.evaluation=evaluation
        self.study={'schema':'vsr_videomaev2_study_v1','arms':self.arms,'updates':2048,'eval_updates':[512,1024,2048],
                    'checkpoint_updates':[512,1024,1536,2048],'evaluation':evaluation}
        write(root/'study.json',self.study);self.study_file=mod.binding(root/'study.json')
        cal=root/'calibration.json';write(cal,dict(status='passed_videomaev2_preflight',scientific_updates=0,diagnostic_optimizer_updates=26,
            wall_seconds=40.,calibration={'weight':.5},resources={a:{'timed6_mean_seconds':.4,'timed6_SD_seconds':.01} for a in mod.ARMS}))
        tp=root/'training_protocol.json';write(tp,{'schema':'vsr_videomaev2_training_protocol_v1','calibration':{'summary':mod.binding(cal)}})
        ts={'schema':'vsr_videomaev2_training_summary_v1','status':'completed','scientific_updates':4096,'arms':{},
            'compute_cost':{'committed_training_step_seconds':3072.,'known_discarded_completed_step_seconds':10.,
                'known_discarded_completed_step_returns':20,'wall_seconds_all_training_attempts':3300.}}
        for n,arm in enumerate(mod.ARMS):
            p=root/'logs'/(arm+'.jsonl');p.parent.mkdir(exist_ok=True)
            p.write_text(''.join(json.dumps({'step':i,'teacher_forward':arm!='baseline','hr_read_seconds':.1 if arm!='baseline' else 0.,'cumulative_hr_read_seconds':i/10 if arm!='baseline' else 0.})+'\n' for i in range(1,2049)))
            ts['arms'][arm]={'updates':2048,'full_training_log':mod.binding(p),'EMA_updates':0,'teacher_parameter_updates':0,'teacher_forward_count':0 if arm=='baseline' else 2048,'cumulative_hr_read_seconds':0. if arm=='baseline' else 204.8,
                'cumulative_training_seconds':(n+1)*512.,'head_parameters':0 if n==0 else 8917760,'memory':{'peak_allocated_bytes':1000}}
        write(root/'training_summary.json',ts)
        self.training={'scientific_updates':4096,'study_file':self.study_file,
            'files':{'protocol.json':mod.binding(tp),'summary.json':mod.binding(root/'training_summary.json')}}
        self.training_times={a:ts['arms'][a]['cumulative_training_seconds'] for a in mod.ARMS}
        self.val=self.make_run('val')

    def make_run(self,phase):
        run=self.root/phase;roles=[]
        if phase=='val':roles=[{'label':'shared_initial','arm':'shared','family':'baseline','setting':'shared_initial','weight':0.,'update':0,'step':8192,'cumulative_training_seconds':0.,'checkpoint':{'sha256':'initial'}}]
        for a in self.arms:
            for u in ((512,1024,2048) if phase=='val' else (2048,)):
                roles.append({'label':f'{a["label"]}_{u:06d}','arm':a['label'],'family':a['label'],'setting':'common',
                    'weight':a['weight'],'update':u,'step':u,'cumulative_training_seconds':self.training_times[a['label']]*u/2048,
                    'checkpoint':{'sha256':f'{a["label"]}-{u}'},'config_binding':{}})
        cohort=[{'record':{'id':f'clip{i:02d}','source_id':f'source{i:02d}','split':phase}} for i in range(32)]
        count=len(roles)*64
        protocol={'schema':'vsr_videomaev2_evaluation_protocol_v1','phase':phase,'sources':self.sources,'study_file':self.study_file,
            'training_evidence':self.training,'roles':roles,'cohort':cohort,'cohort_seal_digest':phase+'-seal',
            'cohort_seal':self.evaluation['validation_seal' if phase=='val' else 'test_cohort_plan'],
            'dataset':self.evaluation['validation_dataset' if phase=='val' else 'test_dataset'],
            'scope':{'source_split':phase,'conditions':['clean'],'sampling_seeds':[42,43],'primary_pool':'all33_frames',
                     'training_updates':0,'roles':len(roles),'total_restores':count},
            'data_independence':{'test_prior_exposure':'historical_models_previously_evaluated','historically_pristine_test':False}}
        if phase=='test':
            vp=mod.read(self.val[0]/'protocol.json');vv=mod.read(self.val[1])
            lock={'schema':'vsr_videomaev2_validation_final_lock_v1','study_file':self.study_file,'metric_values_used_to_select':False,
                'test_cohort_plan':self.evaluation['test_cohort_plan'],'test_dataset':self.evaluation['test_dataset'],
                **{f'validation_{k}':mod.binding(self.val[0]/(k+'.json')) for k in ('protocol','status','summary','observations')},
                'validation_verification':vv,'final_roles':[r for r in vp['roles'] if r['update']==2048]}
            write(self.root/'finals_lock.json',lock);protocol.update(finals_receipt=mod.binding(self.root/'finals_lock.json'),validation_final_lock=lock)
        write(run/'protocol.json',protocol);psha=mod.sha(run/'protocol.json');rows=[]
        for role in roles:
            for i,record in enumerate(cohort):
                for seed in (42,43):
                    arm=role['arm'];update=role['update'];n={'shared':0,'baseline':0,'videomaev2':1}[arm]
                    progress=update/2048
                    lpips=.4-.1*progress-.01*n*progress+ i*.0002 + (seed-42)*.0001
                    lpips-=n*progress*i*.00005
                    metrics={'lpips_alex_v0_1':lpips,'psnr_rgb_db':20+progress+n*.1*progress+i*.01,
                        'ssim_rgb':.7+progress*.01+n*.01*progress+i*.0001,
                        'tLP100_alex_v0_1_TecoGAN_crop':10-progress-n*.1*progress+i*.02,
                        'tOF_Farneback_TecoGAN_crop':3-progress*.1-n*.01*progress+i*.001}
                    row={k:role[k] for k in ('arm','family','setting','weight','step','update','cumulative_training_seconds')}
                    row.update(role=role['label'],record_id=record['record']['id'],source_id=record['record']['source_id'],
                        sampling_seed=seed,condition='clean',split=phase,training_seed=20261102,checkpoint_sha256=role['checkpoint']['sha256'],
                        cohort_seal_digest=protocol['cohort_seal_digest'],metrics=metrics)
                    p=run/'results'/role['label']/record['record']['id']/str(seed)/'metrics.json'
                    write(p,{'observation':row,'restore_seconds':.2});write(p.parent/'complete.json',{'protocol_sha256':psha,'metrics_sha256':mod.sha(p),'PNG_count':33})
                    rows.append(row|{'row_metrics_path':str(p),'row_metrics_sha256':mod.sha(p)})
        write(run/'observations.json',{'protocol_sha256':psha,'observations':rows})
        write(run/'summary.json',{'protocol_sha256':psha,'restores':count,'quality_gate_applied':False})
        write(run/'status.json',{'status':'completed','training_updates':0,'restores':count,'protocol_sha256':psha,'summary_sha256':mod.sha(run/'summary.json')})
        verification={'status':'verified_complete_clean_evaluation','run':str(run),'protocol_sha256':psha,
            'summary_sha256':mod.sha(run/'summary.json'),'observations_sha256':mod.sha(run/'observations.json'),'restores':count,
            'prediction_PNGs_verified':count*33,'prediction_MP4s_verified':count,'GT_PNGs_verified':1056,'GT_MP4s_verified':32,
            'training_updates':0,'training_only_checkpoint_roles_verified':True,'both_arms_retained':True,'weight_or_checkpoint_selection':False}
        receipt=run/'verification.json';write(receipt,verification)
        return run,receipt


class AnalysisTests(unittest.TestCase):
    def test_full_val448_one_paired_contrast_and_exact_bootstrap(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));report=mod.compute(*f.val)
            self.assertEqual(report['matrix_rows'],448);self.assertEqual(len(report['per_clip_metrics']),224)
            self.assertEqual(len(report['paired_clip_differences']),128)
            self.assertEqual(list(report['curves']),['baseline','videomaev2'])
            self.assertEqual(report['quality_verdict']['classification'],'single_seed_descriptive_LPIPS_gain')
            vector=np.asarray([.01+i*.00005 for i in range(32)])
            rng=np.random.Generator(np.random.PCG64(20261102));draws=rng.integers(0,32,size=(10000,32),dtype=np.int64)
            expected=np.quantile(vector[draws].mean(axis=1),[.025,.975],method='linear')
            got=report['paired_comparisons']['videomaev2_vs_baseline'][2048]['lpips_alex_v0_1']
            np.testing.assert_allclose(got['paired_clip_95pct_CI'],expected,rtol=0,atol=1e-15)
            self.assertAlmostEqual(got['mean'],float(vector.mean()),places=14)
            self.assertEqual(report['bootstrap']['p_values_computed'],False)
            self.assertEqual(report['efficiency']['equal_time_budget_observed_points'][0]['arms']['videomaev2']['latest_observed_update_within_budget'],0)

    def test_full_test128_allfinals_kept_and_no_test_efficiency_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));test=f.make_run('test');report=mod.compute(*test)
            self.assertEqual(report['matrix_rows'],128);self.assertEqual(len(report['per_clip_metrics']),64)
            self.assertEqual(len(report['paired_clip_differences']),32)
            self.assertEqual(report['efficiency']['status'],'not_estimated_from_test')
            self.assertFalse(report['weight_or_checkpoint_selection'])
            self.assertFalse(report['data_independence']['historically_pristine_test'])

    def test_partial_status_rejected_before_quality_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write(root/'status.json',{'status':'running','restores':600,'training_updates':0})
            calls=[];original=mod.read
            def tracked(path):calls.append(Path(path).name);return original(path)
            with patch.object(mod,'read',side_effect=tracked):
                with self.assertRaisesRegex(ValueError,'before any quality read'):mod.compute(root,root/'none.json')
            self.assertEqual(calls,['status.json'])

    def test_missing_or_duplicate_rows_rejected_with_rebound_receipt(self):
        for kind in ('missing','duplicate'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                f=Fixture(Path(folder));run,receipt=f.val;bundle=mod.read(run/'observations.json')
                if kind=='missing':bundle['observations'].pop()
                else:bundle['observations'][-1]=bundle['observations'][0]
                write(run/'observations.json',bundle);v=mod.read(receipt);v['observations_sha256']=mod.sha(run/'observations.json');write(receipt,v)
                with self.assertRaises(ValueError):mod.compute(run,receipt)

    def test_wrong_native_verification_or_test_final_CP_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));run,receipt=f.val;v=mod.read(receipt);v['prediction_PNGs_verified']-=1;write(receipt,v)
            with self.assertRaisesRegex(ValueError,'Native verification'):mod.compute(run,receipt)
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));run,receipt=f.make_run('test');p=mod.read(run/'protocol.json')
            p['roles'][-1]['checkpoint']={'sha256':'changed'}
            # Direct lock validation isolates the source/final identity risk.
            with self.assertRaisesRegex(ValueError,'locked final'):mod._verify_final_lock(p,f.study)

    def test_primary_LPIPS_only_with_all_other_costs_reported_and_zero_boundaries(self):
        good={m:{'mean':.01,'paired_clip_95pct_CI':[.005,.015]} for m in mod.METRICS}
        comp={'videomaev2_vs_baseline':{2048:copy.deepcopy(good)}}
        self.assertEqual(mod._quality_verdict(comp)['classification'],'single_seed_descriptive_LPIPS_gain')
        comp['videomaev2_vs_baseline'][2048]['ssim_rgb']['mean']=-1e-14
        verdict=mod._quality_verdict(comp)
        self.assertEqual(verdict['classification'],'single_seed_descriptive_LPIPS_gain')
        self.assertTrue(verdict['point_metric_declines_positive_favors_method']['ssim_rgb'])
        self.assertFalse(verdict['quality_AND_gate_applied'])
        for limits,status in [([0.,.01],'insufficient_evidence'),([-.01,0.],'insufficient_evidence'),
                              ([0.,0.],'no_observed_difference'),([-.01,-.001],'single_seed_descriptive_LPIPS_degradation')]:
            comp['videomaev2_vs_baseline'][2048]['lpips_alex_v0_1']['paired_clip_95pct_CI']=limits
            self.assertEqual(mod._quality_verdict(comp)['classification'],status)

    def test_tensor_checkpoint_bytes_are_not_reopened_by_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));run,receipt=f.val
            cp=Path(folder)/'training_cp.pt';cp.write_bytes(b'checkpoint placeholder')
            protocol=mod.read(run/'protocol.json');protocol['training_evidence']['files']['cp']=mod.binding(cp)
            write(run/'protocol.json',protocol);psha=mod.sha(run/'protocol.json')
            bundle=mod.read(run/'observations.json');bundle['protocol_sha256']=psha
            for row in bundle['observations']:
                marker=Path(row['row_metrics_path']).parent/'complete.json';value=mod.read(marker);value['protocol_sha256']=psha;write(marker,value)
            write(run/'observations.json',bundle)
            summary=mod.read(run/'summary.json');summary['protocol_sha256']=psha;write(run/'summary.json',summary)
            state=mod.read(run/'status.json');state.update(protocol_sha256=psha,summary_sha256=mod.sha(run/'summary.json'));write(run/'status.json',state)
            v=mod.read(receipt);v.update(protocol_sha256=psha,summary_sha256=state['summary_sha256'],observations_sha256=mod.sha(run/'observations.json'));write(receipt,v)
            # The narrow text evidence loop is also checked dynamically by rejecting any CP content read.
            original=mod.verify_file
            def forbidden(value):
                if Path(value['path']).suffix=='.pt':raise AssertionError('checkpoint contents reopened')
                return original(value)
            with patch.object(mod,'verify_file',side_effect=forbidden):
                report=mod.compute(run,receipt)
            self.assertFalse(report['large_checkpoint_content_rehashed'])
            self.assertFalse(report['tensor_payload_loaded'])

    def test_shared_initial_at_target_cannot_claim_acceleration(self):
        curves={a:{u:{'cumulative_training_seconds':float(u),'metrics':{'lpips_alex_v0_1':{'mean':.2 if u==0 else .5}}} for u in mod.UPDATES} for a in mod.ARMS}
        result=mod.common.first_target_crossings(curves,'baseline',.3)
        self.assertEqual(result['status'],'uninterpretable_initial_already_meets_target')
        self.assertTrue(all(v is None for v in result['speedups_vs_baseline'].values()))

    def test_online_teacher_cost_and_attempt_wall_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));run,receipt=f.val
            write(run/'execution_wall_time.json',{'status':'completed','stage':'evaluation_val','run':str(run),
                'protocol_sha256':mod.sha(run/'protocol.json'),'wall_seconds':150.,'scope':'execute loop includingmetrics media IO'})
            r=mod.compute(run,receipt);cost=r['cost_accounting']
            self.assertEqual(cost['per_arm']['videomaev2']['online_teacher_forwards'],2048)
            self.assertEqual(cost['per_arm']['videomaev2']['EMA_updates'],0)
            self.assertEqual(cost['per_arm']['videomaev2']['hr_disk_read_seconds'],204.8)
            self.assertEqual(cost['per_arm']['baseline']['online_teacher_forwards'],0)
            self.assertIsNone(cost['total_experiment_seconds']);self.assertIsNone(cost['teacher_only_seconds'])
            self.assertAlmostEqual(cost['evaluation_execution']['canonical_restore_seconds_sum'],89.6)
            self.assertEqual(cost['evaluation_execution']['receipt']['wall_seconds'],150.)
            self.assertEqual(cost['train_only_preflight']['diagnostic_optimizer_updates'],26)
            self.assertEqual(cost['offline_teacher_cache_this_study']['new_feature_extraction_seconds'],0)

    def test_delivery_csv_curves_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));output=Path(folder)/'delivery';result=mod.deliver(*f.val,output)
            self.assertEqual(result['matrix_rows'],448)
            expected={'per_clip_metrics.csv':224,'paired_clip_differences.csv':128,'summary.csv':40,'paired_summary.csv':20}
            for filename,count in expected.items():
                with (output/filename).open(newline='') as stream:self.assertEqual(len(list(csv.DictReader(stream))),count)
            for filename in ('quality_vs_updates.png','quality_vs_training_seconds.png'):
                self.assertGreater((output/filename).stat().st_size,10000)
            for item in mod.read(output/'delivery_manifest.json')['files']:mod.verify_file(item)
            with self.assertRaisesRegex(ValueError,'immutable'):mod.deliver(*f.val,output)


if __name__=='__main__':unittest.main(verbosity=2)
