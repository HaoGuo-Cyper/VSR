"""Synthetic completed576-row fixtures; no current research quality is read."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import analyze_videomaev2_study as published
from test_analyze_videomaev2_study import Fixture,write
import review_videomaev2_final_statistics as audit


def rebound(root,receipt):
    protocol=audit.read(root/'protocol.json');roles={r['label']:r for r in protocol['roles']};psha=audit.digest(root/'protocol.json')
    b=audit.read(root/'observations.json');b['protocol_sha256']=psha
    for row in b['observations']:
        row['checkpoint_sha256']=roles[row['role']]['checkpoint']['sha256'];mp=Path(row['row_metrics_path']);raw=audit.read(mp)
        raw['observation']={k:v for k,v in row.items() if not k.startswith('row_metrics_')};write(mp,raw);row['row_metrics_sha256']=audit.digest(mp)
        write(mp.parent/'complete.json',{'protocol_sha256':psha,'metrics_sha256':row['row_metrics_sha256'],'PNG_count':33})
    write(root/'observations.json',b);su=audit.read(root/'summary.json');su['protocol_sha256']=psha;write(root/'summary.json',su)
    st=audit.read(root/'status.json');st.update(protocol_sha256=psha,summary_sha256=audit.digest(root/'summary.json'));write(root/'status.json',st)
    v=audit.read(receipt);v.update(protocol_sha256=psha,summary_sha256=st['summary_sha256'],observations_sha256=audit.digest(root/'observations.json'));write(receipt,v)


def complete_fixture(root):
    f=Fixture(root);ts=audit.read(root/'training_summary.json')
    for n,arm in enumerate(audit.ARMS):
        log=Path(ts['arms'][arm]['full_training_log']['path']);seconds=(n+1)/4;hr=.1 if n else 0.
        log.write_text(''.join(json.dumps({'step':i,'teacher_forward':arm!='baseline','seconds':seconds,
            'cumulative_training_seconds':i*seconds,'hr_read_seconds':hr,'cumulative_hr_read_seconds':i/10 if n else 0.})+'\n' for i in range(1,2049)))
        ts['arms'][arm]['full_training_log']=audit.bind(log)
    ts['compute_cost']['committed_training_step_seconds']=1536.;ts['compute_cost']['committed_hr_read_seconds']=204.8
    write(root/'training_summary.json',ts)
    cal=audit.read(root/'calibration.json');cal.update(gradient_probe_cases=15,gradient_VJP_calls=30,teacher_forwards=31);write(root/'calibration.json',cal)
    tp=audit.read(root/'training_protocol.json');tp['calibration']['summary']=audit.bind(root/'calibration.json');write(root/'training_protocol.json',tp)
    f.training['files']['summary.json']=audit.bind(root/'training_summary.json');f.training['files']['protocol.json']=audit.bind(root/'training_protocol.json')
    f.training['run']=str(root/'paired_training');f.training['arms']={}
    for arm in audit.ARMS:
        f.training['arms'][arm]={'checkpoint_files':[]}
        for step in (512,1024,1536,2048):
            cp=root/'paired_training/arm_runs'/arm/'Base'/f'checkpoint_{step:06d}.pt';cp.parent.mkdir(parents=True,exist_ok=True)
            cp.write_bytes(('synthetic-not-a-tensor-'+arm+str(step)).encode());b=audit.bind(cp)
            f.training['files'][f'{arm}/checkpoint_{step:06d}']=b;f.training['arms'][arm]['checkpoint_files'].append(b|{'step':step,'update':step})
    f.val=f.make_run('val');vp=audit.read(f.val[0]/'protocol.json')
    for role in vp['roles']:
        if role['update']>0:role['checkpoint']=f.training['files'][f'{role["arm"]}/checkpoint_{role["update"]:06d}']
    write(f.val[0]/'protocol.json',vp);rebound(*f.val);test=f.make_run('test');p=audit.read(test[0]/'protocol.json')
    for role in p['roles']:role['checkpoint']=f.training['files'][f'{role["arm"]}/checkpoint_{role["update"]:06d}']
    for c in p['cohort']:c['record']['source_id']='test_'+c['record']['source_id']
    write(test[0]/'protocol.json',p);b=audit.read(test[0]/'observations.json')
    for row in b['observations']:row['source_id']='test_'+row['source_id']
    write(test[0]/'observations.json',b);rebound(*test);outputs=[]
    for (run,receipt),phase in [(f.val,'val'),(test,'test')]:
        write(run/'execution_wall_time.json',{'status':'completed','stage':'evaluation_'+phase,'run':str(run),
            'protocol_sha256':audit.digest(run/'protocol.json'),'wall_seconds':500. if phase=='val' else 200.,'scope':'synthetic complete execution'})
        ap=root/'analysis'/phase/'analysis.json';write(ap,published.compute(run,receipt))
        write(ap.parent/'delivery_manifest.json',{'schema':'vsr_videomaev2_analysis_delivery_v1','phase':phase,'files':[audit.bind(ap)]})
        outputs.append(ap)
    return (f.val[0],f.val[1],test[0],test[1],outputs[0],outputs[1])


class AuditTests(unittest.TestCase):
    def test_all576_rows25_intervals_perclip_costs_and_no_CP_read(self):
        with tempfile.TemporaryDirectory() as d:
            args=complete_fixture(Path(d));opening=Path.open;digest=audit.digest
            def guarded_open(p,*a,**k):
                if Path(p).suffix=='.pt':raise AssertionError('CP contents opened')
                return opening(p,*a,**k)
            def guarded_hash(p):
                if Path(p).suffix=='.pt':raise AssertionError('CP contents hashed')
                return digest(p)
            with patch.object(Path,'open',new=guarded_open),patch.object(audit,'digest',side_effect=guarded_hash):r=audit.audit(*args)
            self.assertEqual(r['status'],'passed');self.assertEqual(r['observations_recomputed'],576)
            self.assertEqual(r['paired_metric_intervals_checked'],25);self.assertEqual(r['unique_trained_checkpoints_metadata_checked'],8)
            self.assertEqual(r['phases']['test']['quality_verdict']['classification'],'single_seed_descriptive_LPIPS_gain')
            self.assertAlmostEqual(r['training_cost']['per_arm']['videomaev2']['hr_disk_read_seconds'],204.8)
            self.assertFalse(r['phases']['test']['quality_verdict']['quality_AND_gate_applied'])
            self.assertEqual(r['training_cost']['per_arm']['videomaev2']['recomputed_sum_step_seconds'],1024.)
            self.assertAlmostEqual(r['phases']['test']['evaluation_cost']['canonical_restore_seconds_sum'],25.6)

    def test_incomplete_either_matrix_and_missing_analysis_early_reject(self):
        for stage in ('val','test','missing_analysis','bad_native','missing_test_manifest'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as d:
                root=Path(d);v=root/'val';t=root/'test'
                write(v/'status.json',{'status':'running' if stage=='val' else 'completed','restores':448,'training_updates':0,'protocol_sha256':'a'*64,'summary_sha256':'b'*64})
                write(t/'status.json',{'status':'running' if stage=='test' else 'completed','restores':128,'training_updates':0,'protocol_sha256':'a'*64,'summary_sha256':'b'*64})
                if stage in ('bad_native','missing_test_manifest'):
                    for folder,count in ((v,448),(t,128)):
                        write(folder/'analysis',{'must_not_be_read':'quality'})
                        write(folder/'verify',{'status':'failed' if folder==t and stage=='bad_native' else 'verified_complete_clean_evaluation',
                            'run':str(folder),'protocol_sha256':'a'*64,'summary_sha256':'b'*64,'observations_sha256':'c'*64,
                            'restores':count,'prediction_PNGs_verified':count*33,'prediction_MP4s_verified':count,
                            'GT_PNGs_verified':1056,'GT_MP4s_verified':32,'training_updates':0,
                            'training_only_checkpoint_roles_verified':True,'both_arms_retained':True,'weight_or_checkpoint_selection':False})
                if stage=='missing_test_manifest':
                    write(v/'delivery_manifest.json',{'schema':'vsr_videomaev2_analysis_delivery_v1','phase':'val','files':[audit.bind(v/'analysis')]})
                calls=[];reader=audit.read
                def track(p):calls.append(Path(p));return reader(p)
                with patch.object(audit,'read',side_effect=track):
                    with self.assertRaisesRegex(ValueError,'before any quality read'):audit.audit(v,v/'verify',t,t/'verify',v/'analysis',t/'analysis')
                self.assertEqual(calls,[v/'status.json'] if stage=='val' else [v/'status.json',t/'status.json']+([v/'verify',t/'verify'] if stage in ('bad_native','missing_test_manifest') else [])+([v/'delivery_manifest.json'] if stage=='missing_test_manifest' else []))

    def test_altered_interval_clip_table_and_cost_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            args=complete_fixture(Path(d));original=audit.read(args[-1])
            changes=[lambda x:x['paired_comparisons']['videomaev2_vs_baseline']['2048']['lpips_alex_v0_1']['paired_clip_95pct_CI'].__setitem__(0,.9),
                     lambda x:x['per_clip_metrics'][0].__setitem__('ssim_rgb',.01),
                     lambda x:x['cost_accounting']['per_arm']['videomaev2'].__setitem__('hr_disk_read_seconds',203.)]
            for change in changes:
                altered=copy.deepcopy(original);change(altered);write(args[-1],altered)
                write(args[-1].parent/'delivery_manifest.json',{'schema':'vsr_videomaev2_analysis_delivery_v1','phase':'test','files':[audit.bind(args[-1])]})
                with self.assertRaisesRegex(ValueError,'Numerical parity mismatch'):audit.audit(*args)

    def test_duplicate_identity_and_changed_CP_size_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            args=complete_fixture(Path(d));run=args[2];b=audit.read(run/'observations.json');b['observations'][-1]=b['observations'][0];write(run/'observations.json',b)
            v=audit.read(args[3]);v['observations_sha256']=audit.digest(run/'observations.json');write(args[3],v)
            with self.assertRaisesRegex(ValueError,'Duplicate/foreign'):audit.audit(*args)
        with tempfile.TemporaryDirectory() as d:
            args=complete_fixture(Path(d));cp=Path(d)/'paired_training/arm_runs/videomaev2/Base/checkpoint_002048.pt'
            with cp.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'path/size metadata'):audit.audit(*args)

    def test_percentile_zero_boundaries_and_no_fidelity_AND_gate(self):
        ordered=np.arange(10000,dtype=np.float64)
        self.assertAlmostEqual(audit.percentile_linear(ordered,.025),249.975,places=11)
        self.assertAlmostEqual(audit.percentile_linear(ordered,.975),9749.025,places=11)
        good={m:{'mean':.1,'paired_clip_95pct_CI':[.01,.2]} for m in audit.METRICS}
        good['ssim_rgb']['mean']=-1e-14;comp={'videomaev2_vs_baseline':{2048:good}}
        r=audit.verdict(comp);self.assertEqual(r['classification'],'single_seed_descriptive_LPIPS_gain')
        self.assertTrue(r['point_metric_declines_positive_favors_method']['ssim_rgb']);self.assertFalse(r['quality_AND_gate_applied'])
        for limits,expected in [([-.1,0.],'insufficient_evidence'),([0.,.1],'insufficient_evidence'),([0.,0.],'no_observed_difference'),([-.2,-.1],'single_seed_descriptive_LPIPS_degradation')]:
            self.assertEqual(audit.lpips_class(limits),expected)

    def test_shared0_target_and_sparse_time_no_interpolation(self):
        curves={arm:{u:{'cumulative_training_seconds':float(u)*(1 if arm=='baseline' else 1.1),
                      'metrics':{audit.METRICS[0]:{'mean':.2 if u==0 else .3}}} for u in audit.POINTS} for arm in audit.ARMS}
        r=audit.efficiency(curves);self.assertEqual(r['status'],'uninterpretable_initial_already_meets_target')
        self.assertTrue(all(x is None for x in r['speedups_vs_baseline'].values()))
        self.assertEqual(r['equal_time_updates']['512']['videomaev2'],0)

if __name__=='__main__':unittest.main(verbosity=2)
