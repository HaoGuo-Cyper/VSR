"""Synthetic text exporter tests: never initialize CUDA or read binary artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from export_fixed_teacher_progress_v2 import (CAL,EVAL,TRAIN,METRICS,encoded,sha,export_run)


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))
def declared(path,raw=b'not-opened'):
    return {'path':str(path.resolve()),'bytes':len(raw),'sha256':sha(raw)}


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.top=Path(self.temp.name);self.root=self.top/'source';self.root.mkdir()
        source=self.root/'vsr/core.py';source.parent.mkdir();source.write_text('# frozen synthetic source\n')
        self.sources={'core.py':declared(source,source.read_bytes())}
        self.run_root=self.top/'private_runs';self.run_root.mkdir();self.run=self.run_root/'fixture';self.run.mkdir()
        self.dest=self.top/'public_docs';self.out=self.dest/self.run.name
        self.arms=[{'label':'baseline','family':'baseline','setting':'baseline','weight':0.},
            {'label':'vae_low','family':'hr_vae','setting':'low','weight':.01},
            {'label':'vae_high','family':'hr_vae','setting':'high','weight':.1},
            {'label':'dino_low','family':'hr_dino','setting':'low','weight':.02},
            {'label':'dino_high','family':'hr_dino','setting':'high','weight':.2}]
        self.ids=[f'train_{i}' for i in range(1641)]

    def protocol(self,p,status='running',**extra):
        self.p=p;write(self.run/'protocol.json',p);self.psha=sha((self.run/'protocol.json').read_bytes())
        write(self.run/'status.json',{'status':status,'protocol_sha256':self.psha,**extra})

    def train_protocol(self):
        self.protocol({'schema':TRAIN,'sources':self.sources,'scientific_update_budget':10240,
            'study':{'updates':2048,'checkpoint_updates':[512,1024,1536,2048],'arms':self.arms},
            'cache':{'selection':{'training_ids':self.ids}}})
        write(self.run/'attempt_000.json',{'protocol_sha256':self.psha,'resume_requested':False})

    def train_log(self,arm,n,changed_at=None):
        rows=[];cumulative=0.
        for i in range(n):
            seconds=.2 if i+1==changed_at else .1;cumulative+=seconds
            rows.append({'step':i+1,'arm':arm['label'],'weight':arm['weight'],'family':arm['family'],
                'setting':arm['setting'],'record_id':self.ids[i%1641],'data_index':i%1641,
                'fm':1.,'align':.1,'total':1.+arm['weight']*.1,'grad_norm_before_clip':.2,
                'sigma':.5,'seconds':seconds,'cumulative_training_seconds':cumulative})
        path=self.run/'arm_runs'/arm['label']/'training.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b''.join(encoded(x) for x in rows));return path,rows

    def export(self):return export_run(self.run,root=self.root,run_root=self.run_root,dest=self.dest)

    def forbid_binary(self):
        original=Path.read_bytes
        def read(path):
            if path.suffix.lower() in ('.pt','.npy','.png','.mp4','.safetensors'):
                raise AssertionError('A binary artifact was opened')
            return original(path)
        return patch.object(Path,'read_bytes',read)

    def test_training_complete_lines_append_and_partial_JSON_or_UTF8_wait(self):
        self.train_protocol();path,rows=self.train_log(self.arms[0],1)
        path.write_bytes(path.read_bytes()+b'{"step":2,"unfinished":"\xe4')
        with self.forbid_binary():first=self.export()
        self.assertEqual(first['exported_training_rows_current_attempt'],1)
        trace=self.out/'attempt_traces/attempt_000/baseline/training.jsonl';old=trace.read_bytes()
        self.train_log(self.arms[0],3);self.export()
        self.assertTrue(trace.read_bytes().startswith(old));self.assertEqual(len(trace.read_bytes().splitlines()),3)
        (self.run/'summary.json').write_bytes(b'{"unfinished":')
        self.assertIn(str(self.run/'summary.json'),self.export()['pending_incomplete_text'])
        self.assertFalse((self.out/'summary.json').exists())

    def test_same_attempt_log_rewrite_or_immutable_source_rewrite_is_refused(self):
        self.train_protocol();path,rows=self.train_log(self.arms[0],2)
        init=path.parent/'initialization.json';write(init,{'origin':'Base8192'})
        self.export();original=(self.out/'attempt_traces/attempt_000/baseline/training.jsonl').read_bytes()
        rows[0]['fm']=2.;path.write_bytes(b''.join(encoded(x) for x in rows))
        with self.assertRaisesRegex(ValueError,'Append-only'):self.export()
        self.assertEqual((self.out/'attempt_traces/attempt_000/baseline/training.jsonl').read_bytes(),original)
        self.train_log(self.arms[0],2);write(init,{'origin':'changed'})
        with self.assertRaisesRegex(ValueError,'immutable source changed'):self.export()

    def test_resume_keeps_attempt0_and_requires_preserved_rollback_provenance(self):
        self.train_protocol();path,_=self.train_log(self.arms[0],515);self.export();old=path.read_bytes()
        write(self.run/'attempt_001.json',{'protocol_sha256':self.psha,'resume_requested':True})
        self.train_log(self.arms[0],513,changed_at=513)
        with self.assertRaises(ValueError):self.export()
        archive=path.parent/'interrupted_log_attempt_001.jsonl';archive.write_bytes(old)
        snapshot=self.export()
        self.assertEqual(snapshot['rollback_provenance'][0]['observed_common_prefix_steps'],512)
        self.assertEqual((self.out/'attempt_traces/attempt_000/baseline/training.jsonl').read_bytes(),old)
        self.assertEqual(len((self.out/'attempt_traces/attempt_001/baseline/training.jsonl').read_bytes().splitlines()),513)
        self.assertEqual((self.out/'arm_runs/baseline/interrupted_log_attempt_001.jsonl').read_bytes(),old)

    def test_resume_start_does_not_publish_old_tail_as_new_attempt_before_rollback(self):
        self.train_protocol();path,_=self.train_log(self.arms[0],515);self.export();old=path.read_bytes()
        write(self.run/'attempt_001.json',{'protocol_sha256':self.psha,'resume_requested':True})
        self.export()
        new=self.out/'attempt_traces/attempt_001/baseline/training.jsonl'
        self.assertFalse(new.exists())
        (path.parent/'interrupted_log_attempt_001.jsonl').write_bytes(old);self.export()
        self.assertFalse(new.exists())
        self.train_log(self.arms[0],512);self.export()
        self.assertEqual(len(new.read_bytes().splitlines()),512)
        self.train_log(self.arms[0],513,changed_at=513);self.export()
        self.assertEqual(len(new.read_bytes().splitlines()),513)

    def test_resume_malformed_preserved_tail_is_hashed_but_only_valid_rows_are_published(self):
        self.train_protocol();path,_=self.train_log(self.arms[0],515);self.export();old=path.read_bytes()
        write(self.run/'attempt_001.json',{'protocol_sha256':self.psha,'resume_requested':True})
        archive=path.parent/'interrupted_log_attempt_001.jsonl'
        malformed=b'{"partial":"\xe4\n';archive.write_bytes(old+malformed)
        write(path.parent/'rollback_attempt_001.json',{'checkpoint_step':512,'discarded_logged_steps':3})
        write(self.run/'completion_attempt_000.json',{'wall_seconds':12.,'status':'failed_recoverable'})
        self.train_log(self.arms[0],513,changed_at=513);result=self.export()
        published=self.out/'arm_runs/baseline/interrupted_log_attempt_001.jsonl'
        self.assertEqual(published.read_bytes(),old)
        item=json.loads((self.out/'export_state.json').read_bytes())['immutable']['arm_runs/baseline/interrupted_log_attempt_001.jsonl']
        self.assertEqual(item['source_sha256'],sha(old+malformed));self.assertTrue(item['complete_prefix_only'])
        self.assertEqual(result['rollback_provenance'][0]['unexported_partial_source_bytes'],len(malformed))
        self.assertTrue((self.out/'completion_attempt_000.json').exists())
        self.assertTrue((self.out/'arm_runs/baseline/rollback_attempt_001.json').exists())
        self.export()  # whole-source hash and unchanged prefix also revalidate
        archive.write_bytes(old+malformed+b'x')
        with self.assertRaisesRegex(ValueError,'immutable source changed'):self.export()

    def test_explicit_bound_metadata_is_copied_but_tensor_references_are_not_followed(self):
        self.train_protocol();cache=self.top/'owned_cache';cache.mkdir()
        manifest=cache/'manifest.json';seal=cache/'seal.json'
        write(manifest,{'teacher':'DINO','tensor_path':str(cache/'target.pt')})
        write(seal,{'artifact_sha256':{'target.pt':'b'*64}})
        (cache/'target.pt').write_bytes(b'\x00not-readable')
        p=copy.deepcopy(self.p);p['dino_cache']={'manifest':declared(manifest,manifest.read_bytes()),'seal':declared(seal,seal.read_bytes())}
        self.protocol(p)
        with self.forbid_binary():self.export()
        self.assertEqual((self.out/'bound/dino_cache_manifest.json').read_bytes(),manifest.read_bytes())
        self.assertFalse(list(self.out.rglob('*.pt')))
        write(manifest,{'changed':True})
        with self.assertRaisesRegex(ValueError,'Bound JSON SHA/size changed'):self.export()

    def test_full10240_training_exports_new_canonical_snapshot_and_never_opens_checkpoints(self):
        self.train_protocol();records={}
        for arm in self.arms:
            path,rows=self.train_log(arm,2048)
            cps=[]
            for step in (512,1024,1536,2048):
                binary=path.parent/'Base'/f'checkpoint_{step:06d}.pt';binary.parent.mkdir(exist_ok=True)
                binary.write_bytes(b'\x00fake-weight');cps.append(declared(binary,b'\x00fake-weight')|{'step':step})
            records[arm['label']]={'updates':2048,'weight':arm['weight'],'family':arm['family'],'setting':arm['setting'],
                'full_training_log':declared(path,path.read_bytes()),'checkpoint_files':cps}
            write(path.parent/'completion.json',records[arm['label']])
        summary={'schema':'fixed_teacher_training_summary_v1','status':'completed','protocol_sha256':self.psha,
            'scientific_updates':10240,'arms':records}
        write(self.run/'summary.json',summary)
        write(self.run/'status.json',{'status':'completed','protocol_sha256':self.psha,'scientific_updates':10240,
            'summary_sha256':sha((self.run/'summary.json').read_bytes())})
        with self.forbid_binary():snapshot=self.export()
        self.assertEqual(snapshot['exported_training_rows_current_attempt'],10240)
        self.assertEqual(len(list((self.out/'final_snapshot').rglob('training.jsonl'))),5)
        self.assertFalse(list(self.out.rglob('*.pt')));self.assertFalse(snapshot['binary_artifacts_opened_or_exported'])

    def eval_protocol(self):
        roles=[{'label':label,'arm':label,'family':family,'setting':'low','weight':weight,
            'step':2048,'update':2048,'cumulative_training_seconds':200.,
            'checkpoint':declared(self.top/f'{label}.pt',label.encode())}
            for label,family,weight in (('baseline','baseline',0.),('vae_low','hr_vae',.01),('dino_low','hr_dino',.02))]
        cohort=[{'record':{'id':f'clip{i:02d}','source_id':f'source{i:02d}'},'source_indices':list(range(33))} for i in range(32)]
        self.protocol({'schema':EVAL,'sources':self.sources,'phase':'test','roles':roles,'cohort':cohort,
            'cohort_seal_digest':'sealed-test','scope':{'roles':3,'clips':32,'source_split':'test','conditions':['clean'],
                'sampling_seeds':[42,43],'total_restores':192,'training_updates':0}})
        self.cases=[(role,item,sample) for role in roles for item in cohort for sample in (42,43)]

    def eval_row(self,index):
        role,item,sample=self.cases[index];directory=self.run/'results'/role['label']/item['record']['id']/f'seed_{sample}'
        full={'frame_count':33,'per_frame':[{}]*33,'temporal_pair_count':28,
            'per_temporal_pair':[{'indices':[i,i+1]} for i in range(2,30)],**{k:1. for k in METRICS}}
        obs={k:role[k] for k in ('arm','family','setting','weight','step','update','cumulative_training_seconds')}
        obs.update(role=role['label'],training_seed=20261102,record_id=item['record']['id'],source_id=item['record']['source_id'],
            sampling_seed=sample,condition='clean',checkpoint_sha256=role['checkpoint']['sha256'],
            cohort_seal_digest=self.p['cohort_seal_digest'],split=self.p['phase'],metrics={k:1. for k in METRICS})
        video=declared(directory/'prediction.mp4')
        row={'status':'completed','training_updates':0,'teacher_or_HR_input_to_inferrer':False,
            'inference_inputs':['LRV','native text embedding','student checkpoint'],
            'restoration':{'inference_inputs':['LRV','native text embedding']},'observation':obs,
            'source_metadata':{'split':self.p['phase'],'source_frame_indices':list(range(33))},
            'full_frame_pair_metrics':{'all_frames':full},
            'PNG_artifacts':[declared(directory/'png'/f'{i:06d}.png') for i in range(33)],'video_artifact':video}
        write(directory/'metrics.json',row);msha=sha((directory/'metrics.json').read_bytes())
        write(directory/'complete.json',{'protocol_sha256':self.psha,'metrics_sha256':msha,'PNG_count':33,'video_artifact':video})
        return obs|{'row_metrics_path':str(directory/'metrics.json'),'row_metrics_sha256':msha}

    def test_eval_partial_marker_waits_then_canonical_prefix_appends(self):
        self.eval_protocol();obs=self.eval_row(0);marker=Path(obs['row_metrics_path']).parent/'complete.json';raw=marker.read_bytes()
        marker.write_bytes(b'{"unfinished":')
        self.assertEqual(self.export()['exported_completed_rows'],0)
        marker.write_bytes(raw);self.assertEqual(self.export()['exported_completed_rows'],1)
        self.eval_row(1);self.assertEqual(self.export()['exported_completed_rows'],2)
        self.assertEqual(len((self.out/'completed_observations.jsonl').read_bytes().splitlines()),2)

    def test_full192_test_rows_match_declared_scope_and_binary_media_stay_unopened(self):
        self.eval_protocol();rows=[self.eval_row(i) for i in range(192)]
        write(self.run/'observations.json',{'protocol_sha256':self.psha,'observations':rows})
        write(self.run/'role_audits.json',[{'role':r['label'],'restores':64,'training_updates':0,'inference_weights':'student'} for r in self.p['roles']])
        write(self.run/'summary.json',{'status':'completed_clean_evaluation','protocol_sha256':self.psha,
            'restores':192,'roles':3,'training_updates':0})
        write(self.run/'status.json',{'status':'completed','protocol_sha256':self.psha,'restores':192,'training_updates':0,
            'summary_sha256':sha((self.run/'summary.json').read_bytes())})
        with self.forbid_binary():result=self.export()
        self.assertEqual(result['exported_completed_rows'],192);self.assertEqual(result['expected_restores'],192)
        self.assertEqual(len(list(self.out.rglob('metrics.json'))),192);self.assertFalse(list(self.out.rglob('*.png')))

    def test_full1024_validation_rows_keep_all16_roles_and_scope(self):
        self.eval_protocol();p=copy.deepcopy(self.p)
        roles=[{'label':'shared_0','arm':'shared','family':'baseline','setting':'shared_initial','weight':0.,
            'step':8192,'update':0,'cumulative_training_seconds':0.,'checkpoint':declared(self.top/'initial.pt')}]
        for arm in self.arms:
            for step in (512,1024,2048):
                label=arm['label']+'_'+str(step)
                roles.append({'label':label,'arm':arm['label'],'family':arm['family'],'setting':arm['setting'],
                    'weight':arm['weight'],'step':step,'update':step,'cumulative_training_seconds':float(step),
                    'checkpoint':declared(self.top/(label+'.pt'),label.encode())})
        p.update(phase='val',roles=roles,cohort_seal_digest='sealed-val')
        p['scope'].update(roles=16,total_restores=1024,source_split='val')
        self.protocol(p);self.cases=[(role,item,sample) for role in roles for item in p['cohort'] for sample in (42,43)]
        rows=[self.eval_row(i) for i in range(1024)]
        write(self.run/'observations.json',{'protocol_sha256':self.psha,'observations':rows})
        write(self.run/'role_audits.json',[{'role':r['label'],'restores':64,'training_updates':0,'inference_weights':'student'} for r in roles])
        write(self.run/'summary.json',{'status':'completed_clean_evaluation','protocol_sha256':self.psha,
            'restores':1024,'roles':16,'training_updates':0})
        write(self.run/'status.json',{'status':'completed','protocol_sha256':self.psha,'restores':1024,'training_updates':0,
            'summary_sha256':sha((self.run/'summary.json').read_bytes())})
        with self.forbid_binary():result=self.export()
        self.assertEqual(result['exported_completed_rows'],1024);self.assertEqual(result['phase'],'val')
        exported=[json.loads(line) for line in (self.out/'completed_observations.jsonl').read_bytes().splitlines()]
        self.assertEqual(exported,rows);self.assertEqual(len({r['role'] for r in exported}),16)

    def test_eval_gap_bad_marker_hash_checkpoint_or_source_window_rejected(self):
        self.eval_protocol();self.eval_row(1)
        with self.assertRaisesRegex(ValueError,'canonical prefix'):self.export()
        obs=self.eval_row(0);path=Path(obs['row_metrics_path']);original=path.read_bytes()
        path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'SHA changed'):self.export()
        row=json.loads(original);row['observation']['checkpoint_sha256']='0'*64;write(path,row)
        marker=path.parent/'complete.json';value=json.loads(marker.read_bytes());value['metrics_sha256']=sha(path.read_bytes());write(marker,value)
        with self.assertRaisesRegex(ValueError,'identity/checkpoint/source'):self.export()
        row=json.loads(original);row['source_metadata']['source_frame_indices'][0]=99;write(path,row)
        value['metrics_sha256']=sha(path.read_bytes());write(marker,value)
        with self.assertRaisesRegex(ValueError,'source/window'):self.export()

    def test_numeric_source_or_credential_shaped_allowlist_text_rejected(self):
        self.train_protocol();(self.root/'vsr/core.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'numeric source SHA changed'):self.export()
        (self.root/'vsr/core.py').write_text('# frozen synthetic source\n')
        write(self.run/'config.json',{'api_key':'not-a-real-secret'})
        with self.assertRaisesRegex(ValueError,'Credential-shaped'):self.export()

    def test_completed_calibration_exports_zero_science_and27_diagnostic_accounting(self):
        self.protocol({'schema':CAL,'sources':self.sources,'scientific_update_budget':0,'diagnostic_optimizer_budget':27,
            'calibration_sigmas':[.2,.5,.8],'calibration_training_index':0})
        rows=[{'role':role,'step':i+1,'fm':1.,'align':.1,'total':1.,'sigma':.5,'grad_norm_before_clip':.2,'seconds':.1}
            for role in ('baseline','hr_vae','hr_dino') for i in range(8)]
        write(self.run/'summary.json',{'status':'passed_fixed_teacher_preflight','protocol_sha256':self.psha,'sources':self.sources,
            'scientific_updates':0,'diagnostic_optimizer_updates':27,'all_step_rows':rows})
        write(self.run/'status.json',{'status':'completed','protocol_sha256':self.psha,'scientific_updates':0,
            'summary_sha256':sha((self.run/'summary.json').read_bytes())})
        with self.forbid_binary():result=self.export()
        self.assertEqual(result['scientific_updates'],0);self.assertEqual(result['expected_diagnostic_optimizer_updates'],27)


    def test_safe_leading_hyphen_components_keep_original_size_and_path_boundaries(self):
        from export_fixed_teacher_progress_v2 import component
        safe=('-mSvhSt_Q8g_85_92to239','-a','-','--','-_','-a.b_9','-'+'a'*127)
        for value in safe:
            with self.subTest(value=value):self.assertEqual(component(value),value)
        bad=('', '.', '..', '../a', '-../a', '/tmp/a', 'a/b', '-a/b', 'a\\b', '-a\\b',
             '-a\x00b', '-a\nb', '-a\rb', '-a b', '-中', '.hidden', '-'+'a'*128, None, True, 7, b'-bytes')
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):component(value)

    def test_only_new_component_values_start_with_hyphen(self):
        import itertools
        from export_fixed_teacher_progress import component as original
        from export_fixed_teacher_progress_v2 import component
        for length in range(5):
            for chars in itertools.product('a_.-/\\',repeat=length):
                value=''.join(chars)
                try:original(value);old_ok=True
                except ValueError:old_ok=False
                try:component(value);new_ok=True
                except ValueError:new_ok=False
                if old_ok:self.assertTrue(new_ok)
                if new_ok and not old_ok:
                    self.assertTrue(value.startswith('-'));self.assertNotIn('/',value);self.assertNotIn('\\',value)
                    self.assertNotIn(value,('.', '..'));self.assertLessEqual(len(value),128)

    def test_full192_test_with_real_leading_hyphen_record_name_exports_text_only(self):
        self.eval_protocol();p=copy.deepcopy(self.p);name='-mSvhSt_Q8g_85_92to239'
        p['cohort'][0]['record']['id']=name;self.protocol(p)
        self.cases=[(role,item,sample) for role in p['roles'] for item in p['cohort'] for sample in (42,43)]
        rows=[self.eval_row(i) for i in range(192)]
        write(self.run/'observations.json',{'protocol_sha256':self.psha,'observations':rows})
        write(self.run/'role_audits.json',[{'role':r['label'],'restores':64,'training_updates':0,'inference_weights':'student'} for r in p['roles']])
        write(self.run/'summary.json',{'status':'completed_clean_evaluation','protocol_sha256':self.psha,'restores':192,'roles':3,'training_updates':0})
        write(self.run/'status.json',{'status':'completed','protocol_sha256':self.psha,'restores':192,'training_updates':0,
            'summary_sha256':sha((self.run/'summary.json').read_bytes())})
        with self.forbid_binary():result=self.export()
        self.assertEqual(result['exported_completed_rows'],192)
        exported=[json.loads(line) for line in (self.out/'completed_observations.jsonl').read_bytes().splitlines()]
        self.assertEqual(exported,rows);self.assertEqual(sum(row['record_id']==name for row in exported),6)
        self.assertTrue((self.out/'results'/'baseline'/name/'seed_42/metrics.json').is_file())
        self.assertFalse(list(self.out.rglob('*.pt')));self.assertFalse(list(self.out.rglob('*.png')))

    def test_v2_source_diff_is_exactly_one_regex_literal_change(self):
        import ast
        old=(Path(__file__).with_name('export_fixed_teacher_progress.py')).read_bytes()
        new=(Path(__file__).with_name('export_fixed_teacher_progress_v2.py')).read_bytes()
        before=b"r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}'";after=b"r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}'"
        self.assertEqual(old.count(before),1);self.assertEqual(new,old.replace(before,after,1))
        def stripped(raw):
            tree=ast.parse(raw);tree.body=[node for node in tree.body if not (isinstance(node,ast.FunctionDef) and node.name=='component')]
            return ast.dump(tree,include_attributes=False)
        self.assertEqual(stripped(old),stripped(new))


if __name__=='__main__':unittest.main()
