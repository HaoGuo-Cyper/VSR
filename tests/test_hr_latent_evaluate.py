"""Independent HR clean-study metadata, artifact and recovery CPU fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from vsr import evaluate as ev


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=True)+'\n')


class TrainingFixture:
    def __init__(self,root):
        self.root=root;self.run=root/'training';self.run.mkdir()
        source=root/'source.txt';source.write_text('fixture source')
        self.sources={'fixture.py':ev.binding(source)}
        self.study={'arms':[{'label':'baseline','weight':0.},{'label':'hr001','weight':.01},{'label':'hr01','weight':.1}],
                    'updates':2048,'eval_updates':[512,1024,2048],'checkpoint_updates':[512,1024,1536,2048]}
        study=root/'study.json';write(study,self.study);self.study_file=ev.binding(study)
        self.basecfg={'model':{'local_dir':'fixture'},'conditioning':{},'data':{'frames':33},
                      'reproducibility':{'mode':'strict'},'seed':20261102}
        cp=root/'starting'/'Base'/'checkpoint_008192.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'base')
        write(cp.parent.parent/'config.json',self.basecfg)
        self.base={'checkpoint':ev.binding(cp),'config':self.basecfg}
        ids=[f'train{i}' for i in range(1641)]
        self.protocol={'schema':'hr_vae_training_protocol_v1','study_file':self.study_file,'study':self.study,
                       'sources':self.sources,'base':self.base,'cache':{'selection':{'training_ids':ids}},
                       'inputs':[],'official_model_files':{},'dependencies':{}}
        write(self.run/'protocol.json',self.protocol);self.psha=ev.sha(self.run/'protocol.json')
        self.summary={'schema':'hr_vae_training_summary_v1','status':'completed','protocol_sha256':self.psha,
                      'scientific_updates':6144,'arms':{}}
        for arm in self.study['arms']:
            label,weight=arm['label'],arm['weight'];directory=self.run/'arm_runs'/label
            directory.mkdir(parents=True)
            rows=[dict(step=i,arm=label,weight=weight,data_index=(i-1)%1641,record_id=ids[(i-1)%1641],
                       fm=.2,align=.1,total=.2+weight*.1,grad_norm_before_clip=1.,sigma=.5,
                       seconds=1.,cumulative_training_seconds=float(i),gpu_event_seconds=.5) for i in range(1,2049)]
            log=directory/'training.jsonl';log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            initialization={'checkpoint_sha256':self.base['checkpoint']['sha256'],'step':8192,'variant':'Base',
                            'optimizer_state_restored':False,'scheduler_state_restored':False,'rng_state_restored':False}
            cps=[]
            for i in (512,1024,1536,2048):
                path=directory/'Base'/f'checkpoint_{i:06d}.pt';path.parent.mkdir(exist_ok=True);path.write_bytes(f'{label}:{i}'.encode())
                cps.append(ev.binding(path)|dict(step=i,update=i,variant='Base',cumulative_training_seconds=float(i)))
            record={'weight':weight,'updates':2048,'teacher_is_frozen_cached_HR':True,'initialization':initialization,
                    'initial_student_hashes':{'all_trainable':{'sha256':'same core'}},
                    'checkpoint_files':cps,'full_training_log':ev.binding(log),'cumulative_training_seconds':2048.}
            self.summary['arms'][label]=record
            write(directory/'config.json',self.basecfg);write(directory/'initialization.json',initialization)
            write(directory/'completion.json',record)
        self.publish_summary()

    def publish_summary(self):
        write(self.run/'summary.json',self.summary)
        write(self.run/'status.json',dict(status='completed',protocol_sha256=self.psha,
              summary_sha256=ev.sha(self.run/'summary.json'),scientific_updates=6144))

    def checked(self):
        with patch.object(ev,'_hr_sources',return_value=self.sources):
            return ev._hr_training_evidence(self.run,self.study_file)


class HRTrainingEvidenceTests(unittest.TestCase):
    def test_all6144_logs_and12_CP_and10_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            f=TrainingFixture(Path(folder));evidence=f.checked()
            self.assertTrue(evidence['all6144_rows_finite_contiguous'])
            roles=[dict(label='shared',checkpoint=f.base['checkpoint'],step=8192,arm='shared',weight=0.,update=0,
                        cumulative_training_seconds=0.)]
            for arm,record in evidence['arms'].items():
                for cp in record['checkpoint_files']:
                    if cp['step']==1536:continue
                    roles.append(dict(label=f'{arm}_{cp["step"]}',checkpoint=cp,step=cp['step'],arm=arm,
                                      weight=record['weight'],update=cp['step'],
                                      cumulative_training_seconds=cp['cumulative_training_seconds']))
            cfg={'roles':roles,'allowed_checkpoint_roots':[str(f.root)]}
            with patch('vsr.checkpoints._check_config'):
                self.assertEqual(len(ev._hr_role_specs(cfg,evidence)),10)
                bad=copy.deepcopy(cfg);bad['roles'][-1]['checkpoint']=bad['roles'][-2]['checkpoint']
                with self.assertRaises(ValueError):ev._hr_role_specs(bad,evidence)
            self.assertFalse(torch.cuda.is_initialized())

    def test_different_initial_student_and_nonfinite_log_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=TrainingFixture(Path(folder))
            f.summary['arms']['hr001']['initial_student_hashes']={'all_trainable':{'sha256':'wrong'}}
            f.publish_summary()
            with self.assertRaisesRegex(ValueError,'exact initial student'):f.checked()
            f.summary['arms']['hr001']['initial_student_hashes']={'all_trainable':{'sha256':'same core'}}
            log=f.run/'arm_runs/hr001/training.jsonl';rows=[json.loads(x) for x in log.read_text().splitlines()]
            rows[-1]['grad_norm_before_clip']=float('nan');log.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            f.summary['arms']['hr001']['full_training_log']=ev.binding(log);f.publish_summary()
            with self.assertRaisesRegex(ValueError,'Nonfinite'):f.checked()


class HRCompleteMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();run=cls.run_dir=Path(cls.temp.name)
        cls.roles=[dict(label=f'r{i}',arm=f'arm{i//3}',weight=0.,update=0 if i==0 else 512,
                        step=8192 if i==0 else 512,cumulative_training_seconds=0.,checkpoint={'sha256':str(i)*64})
                   for i in range(10)]
        cls.seal={'cohort':[{'record':{'id':f'clip{i:02d}','source_id':f'source{i:02d}'},
                             'source_indices':list(range(33)),'cached_header':{'fps':24.}} for i in range(32)]}
        write(run/'seal.json',cls.seal)
        write(run/'dataset.json',{'assets':[{'record_id':f'clip{i:02d}','HR':{'sha256':f'HR{i}'}} for i in range(32)]})
        cls.protocol={'schema':ev.HR_EVALUATION_SCHEMA,'config':{},'config_binding':{'path':'synthetic'},
                      'roles':cls.roles,'cohort_seal':ev.binding(run/'seal.json'),'cohort_seal_digest':'fixed',
                      'dataset':ev.binding(run/'dataset.json'),'fixed_video_clip_ids':['clip00','clip01','clip02']}
        write(run/'protocol.json',cls.protocol);cls.psha=ev.sha(run/'protocol.json')
        raw=run/'synthetic_payload';raw.write_bytes(b'synthetic artifact bytes, not a real image')
        full={'all_frames':{'frame_count':33,'per_frame':[{}]*33,'temporal_pair_count':28,
                           'per_temporal_pair':[{'indices':[i,i+1]} for i in range(2,30)],
                           **{k:.1 for k in ev.HR_SCALAR_METRICS}}}
        observations=[]
        for role,item,sample in ev._hr_cases(cls.protocol,cls.seal):
            directory=ev._hr_directory(run,role,item,sample);(directory/'png').mkdir(parents=True)
            pngs=[]
            for i in range(33):
                path=directory/'png'/f'{i:06d}.png';os.link(raw,path);pngs.append(ev.binding(path))
            video=None
            if item['record']['id'] in cls.protocol['fixed_video_clip_ids']:
                path=directory/'prediction.mp4';os.link(raw,path);video=ev.binding(path)
            observation=ev._hr_identity(cls.protocol,role,item,sample)|{'metrics':ev._hr_scalars(full)}
            write(directory/'metrics.json',{'status':'completed','training_updates':0,'observation':observation,
                  'full_frame_pair_metrics':full,'source_metadata':{'split':'val','source_frame_indices':list(range(33)),
                  'output_fps':24.},'PNG_artifacts':pngs,'video_artifact':video,
                  'teacher_or_HR_input_to_inferrer':False,'inference_inputs':['LRV','native text embedding','student checkpoint'],
                  'restoration':{'inference_inputs':['LRV','native text embedding']}})
            write(directory/'complete.json',dict(protocol_sha256=cls.psha,metrics_sha256=ev.sha(directory/'metrics.json'),
                                                 PNG_count=33,video_artifact=video))
            observations.append(observation|dict(row_metrics_path=str(directory/'metrics.json'),row_metrics_sha256=ev.sha(directory/'metrics.json')))
        cls.observations=observations
        write(run/'observations.json',{'observations':observations,'protocol_sha256':cls.psha})
        audits=[]
        for role in cls.roles:
            write(run/'roles'/role['label']/'state_audit_001.json',{'trainable_before':{'same':True},
                        'trainable_after':{'same':True},'training_updates':0})
            audits.append(dict(role=role['label'],restores=64,training_updates=0,inference_weights='student'))
        write(run/'role_audits.json',audits)
        for i in range(3):
            directory=run/'ground_truth'/f'clip{i:02d}';(directory/'png').mkdir(parents=True)
            pngs=[]
            for j in range(33):
                path=directory/'png'/f'{j:06d}.png';os.link(raw,path);pngs.append(ev.binding(path))
            video=directory/'GT.mp4';os.link(raw,video)
            write(directory/'complete.json',dict(protocol_sha256=cls.psha,record_id=f'clip{i:02d}',source_id=f'source{i:02d}',
                      source_HR_sha256=f'HR{i}',PNG_artifacts=pngs,video_artifact=ev.binding(video),frames=33,FPS=24.))
        write(run/'summary.json',dict(protocol_sha256=cls.psha,restores=640,quality_gate_applied=False))
        write(run/'status.json',dict(status='completed',protocol_sha256=cls.psha,restores=640,training_updates=0,
                                    summary_sha256=ev.sha(run/'summary.json')))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def verified(self):
        with patch.object(ev,'hr_collect_protocol',return_value=self.protocol),\
             patch('vsr.infer.verify_output_video',return_value={'frames':33}):
            return ev.hr_verify(self.run_dir)

    def test_all640_rows21120_prediction_PNG_and_fixed_videos(self):
        result=self.verified()
        self.assertEqual(result['restores'],640);self.assertEqual(result['prediction_PNGs_verified'],21120)
        self.assertEqual(result['prediction_MP4s_verified'],60);self.assertEqual(result['GT_MP4s_verified'],3)
        self.assertTrue(result['validation_not_heldout_test']);self.assertFalse(torch.cuda.is_initialized())

    def test_missing_observation_and_changed_marker_rejected(self):
        path=self.run_dir/'observations.json';original=path.read_bytes()
        try:
            data=json.loads(original);data['observations'].pop();write(path,data)
            with self.assertRaisesRegex(ValueError,'observations'):self.verified()
        finally:path.write_bytes(original)
        path=Path(self.observations[0]['row_metrics_path']).parent/'complete.json';original=path.read_bytes()
        try:
            data=json.loads(original);data['PNG_count']=32;write(path,data)
            with self.assertRaisesRegex(ValueError,'marker'):self.verified()
        finally:path.write_bytes(original)

    def test_resume_prefix_rejects_a_gap_or_corrupted_completed_prediction(self):
        marker=Path(self.observations[2]['row_metrics_path']).parent/'complete.json'
        original=marker.read_bytes();marker.unlink()
        try:
            with patch('vsr.infer.verify_output_video'):
                with self.assertRaisesRegex(ValueError,'canonical prefix'):
                    ev._hr_completed_prefix(self.protocol,self.run_dir,self.psha,self.seal)
        finally:marker.write_bytes(original)
        image=Path(self.observations[0]['row_metrics_path']).parent/'png/000000.png';original=image.read_bytes()
        try:
            image.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Bound input changed'):self.verified()
        finally:image.write_bytes(original)

    def test_resume_final_publish_and_incomplete_attempt_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'summary.json'
            ev._hr_save_final(path,{'same':1});ev._hr_save_final(path,{'same':1})
            with self.assertRaisesRegex(ValueError,'aggregate changed'):ev._hr_save_final(path,{'same':2})
            partial=root/'results'/'row';partial.mkdir(parents=True);(partial/'partial.bin').write_bytes(b'preserve')
            ev._hr_quarantine(root,partial)
            self.assertFalse(partial.exists())
            self.assertEqual((root/'incomplete_attempts/results/row.attempt_001/partial.bin').read_bytes(),b'preserve')


if __name__=='__main__':unittest.main(verbosity=2)
