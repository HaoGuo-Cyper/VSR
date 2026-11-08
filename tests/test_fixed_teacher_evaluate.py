"""Fixed-teacher family/budget, val-selection and split-specific CPU checks."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from vsr import evaluate as ev
from test_hr_latent_evaluate import TrainingFixture,write


class FixedTrainingFixture(TrainingFixture):
    def __init__(self, root):
        super().__init__(root)
        calibration=root/'calibration.json';write(calibration,{'split':'train','quality_used':False})
        self.study={'schema':'fixed_teacher_study_v1','arms':[
            dict(label='baseline',family='baseline',setting='baseline',weight=0.),
            dict(label='vae_low',family='hr_vae',setting='low',weight=.02),
            dict(label='vae_high',family='hr_vae',setting='high',weight=.2),
            dict(label='dino_low',family='hr_dino',setting='low',weight=.005),
            dict(label='dino_high',family='hr_dino',setting='high',weight=.05)],
            'updates':2048,'eval_updates':[512,1024,2048],'checkpoint_updates':[512,1024,1536,2048],
            'calibration':ev.binding(calibration)}
        write(root/'study.json',self.study);self.study_file=ev.binding(root/'study.json')
        self.protocol.update(schema='fixed_teacher_training_protocol_v1',study=self.study,study_file=self.study_file)
        self.summary.update(schema='fixed_teacher_training_summary_v1',scientific_updates=10240)
        template=copy.deepcopy(self.summary['arms']['baseline']);self.summary['arms']={}
        ids=self.protocol['cache']['selection']['training_ids']
        for arm in self.study['arms']:
            label=arm['label'];directory=self.run/'arm_runs'/label;directory.mkdir(exist_ok=True)
            record=copy.deepcopy(template);record.update(weight=arm['weight'],family=arm['family'],setting=arm['setting'])
            log=directory/'training.jsonl'
            rows=[dict(step=i,arm=label,weight=arm['weight'],data_index=(i-1)%1641,record_id=ids[(i-1)%1641],
                       fm=.2,align=.1,total=.2+arm['weight']*.1,grad_norm_before_clip=1.,sigma=.5,
                       seconds=1.,cumulative_training_seconds=float(i),gpu_event_seconds=.5) for i in range(1,2049)]
            log.write_text(''.join(json.dumps(x)+'\n' for x in rows));record['full_training_log']=ev.binding(log)
            cps=[]
            for step in (512,1024,1536,2048):
                cp=directory/'Base'/f'checkpoint_{step:06d}.pt';cp.parent.mkdir(exist_ok=True);cp.write_bytes(f'{label}:{step}'.encode())
                cps.append(ev.binding(cp)|dict(step=step,update=step,variant='Base',cumulative_training_seconds=float(step)))
            record['checkpoint_files']=cps
            for name,value in [('config.json',self.basecfg),('initialization.json',record['initialization']),('completion.json',record)]:write(directory/name,value)
            self.summary['arms'][label]=record
        write(self.run/'protocol.json',self.protocol);self.psha=ev.sha(self.run/'protocol.json');self.summary['protocol_sha256']=self.psha
        self.publish_summary()

    def publish_summary(self):
        write(self.run/'summary.json',self.summary)
        write(self.run/'status.json',dict(status='completed',protocol_sha256=self.psha,
              summary_sha256=ev.sha(self.run/'summary.json'),scientific_updates=self.summary['scientific_updates']))

    def checked(self):
        with patch.object(ev,'_hr_sources',return_value=self.sources):
            return ev._hr_training_evidence(self.run,self.study_file,fixed_teacher=True)


class FixedStudyMetadataTests(unittest.TestCase):
    def test_all10240_steps20_CP_and16_validation_or3_selected_test_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture=FixedTrainingFixture(Path(folder));evidence=fixture.checked()
            self.assertEqual(evidence['scientific_updates'],10240);self.assertEqual(len(evidence['arms']),5)
            self.assertEqual(sum(len(a['checkpoint_files']) for a in evidence['arms'].values()),20)
            roles=[dict(label='shared',checkpoint=fixture.base['checkpoint'],step=8192,arm='shared',weight=0.,update=0,
                        cumulative_training_seconds=0.)]
            for arm,record in evidence['arms'].items():
                for cp in record['checkpoint_files']:
                    if cp['step']==1536:continue
                    roles.append(dict(label=f'{arm}_{cp["step"]}',checkpoint=cp,step=cp['step'],arm=arm,
                        weight=record['weight'],update=cp['step'],cumulative_training_seconds=cp['cumulative_training_seconds']))
            cfg={'roles':roles,'allowed_checkpoint_roots':[str(fixture.root)]}
            with patch('vsr.checkpoints._check_config'):
                checked=ev._hr_role_specs(cfg,evidence,fixed_teacher=True)
                self.assertEqual(len(checked),16);self.assertEqual({r['family'] for r in checked},{'baseline','hr_vae','hr_dino'})
                selected=[r for r in checked if r['update']==2048 and r['arm'] in ('baseline','vae_high','dino_low')]
                test=dict(cfg,roles=selected)
                self.assertEqual(len(ev._hr_role_specs(test,evidence,fixed_teacher=True,phase='test',selection={'selected_roles':selected})),3)
                bad=copy.deepcopy(test);bad['roles'][1]['checkpoint']=bad['roles'][2]['checkpoint']
                with self.assertRaises(ValueError):ev._hr_role_specs(bad,evidence,fixed_teacher=True,phase='test',selection={'selected_roles':selected})
            self.assertFalse(torch.cuda.is_initialized())

    def test_family_or_setting_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=FixedTrainingFixture(Path(folder));f.summary['arms']['dino_low']['family']='hr_vae';f.publish_summary()
            with self.assertRaisesRegex(ValueError,'family/setting'):f.checked()

    def test_selection_recomputes_per_family_and_tie_uses_weaker_weight_without_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'validation';run.mkdir()
            arms=[dict(label='baseline',family='baseline',weight=0.),dict(label='vl',family='hr_vae',weight=.02),
                  dict(label='vh',family='hr_vae',weight=.2),dict(label='dl',family='hr_dino',weight=.005),
                  dict(label='dh',family='hr_dino',weight=.05)]
            plan=root/'test_metadata_only.json';write(plan,{'status':'frozen_metadata_only_no_pixels'})
            audit=root/'integrity.json';write(audit,{'scope':'integrity_only','quality_scores_used':False})
            exception={'scope':'integrity_only','audit_receipt':ev.binding(audit),'automated':True,
                       'human_quality_review':False,'model_scores_used':False}
            study={'arms':arms,'evaluation':{'test_cohort_plan':ev.binding(plan),'test_prior_exposure':'historical_models_previously_evaluated',
                    'test_pixels_after_selection':True,'test_integrity_thumbnail_exception':exception}}
            write(root/'study.json',study)
            roles=[dict(label=a['label']+'_2048',arm=a['label'],family=a['family'],setting='baseline' if a['family']=='baseline' else 'low' if a['label'] in ('vl','dl') else 'high',
                weight=a['weight'],update=2048,step=2048,cumulative_training_seconds=10.,checkpoint={'path':str(root/a['label']), 'sha256':a['label']},config_binding={}) for a in arms]
            cohort=[{'record':{'id':f'c{i}','source_id':f's{i}'}} for i in range(32)]
            shared={'label':'shared','arm':'shared','update':0}
            intermediate=[{'label':f'x{i}','arm':f'aux{i}','update':512} for i in range(10)]
            protocol={'phase':'val','scope':{'total_restores':1024},'study_file':ev.binding(root/'study.json'),
                      'cohort':cohort,'roles':[shared,*intermediate,*roles],'sources':{}}
            write(run/'protocol.json',protocol)
            means={'baseline':.5,'vl':.3,'vh':.2,'dl':.25,'dh':.25};rows=[]
            for role in protocol['roles']:
                for item in cohort:
                    for sample in (42,43):rows.append({'role':role['label'],'record_id':item['record']['id'],'sampling_seed':sample,
                        'metrics':{'lpips_alex_v0_1':means.get(role['arm'],.9)}})
            write(run/'observations.json',{'observations':rows})
            write(run/'status.json',{'status':'completed'});write(run/'summary.json',{'restores':1024})
            with patch.object(ev,'ft_verify',return_value={'restores':1024}):
                payload=ev._ft_selection_payload(run)
            selected={r['family']:r for r in payload['selected_roles']}
            self.assertEqual(selected['hr_vae']['arm'],'vh');self.assertEqual(selected['hr_dino']['arm'],'dl')
            self.assertFalse(payload['test_pixels_read_by_selection']);self.assertFalse(payload['pristine_test_claimed'])
            self.assertEqual(payload['test_cohort_plan'],ev.binding(plan))
            self.assertEqual(payload['test_integrity_thumbnail_exception'],exception)
            self.assertEqual(payload['test_pixels_after_selection_scope'],'formal_RGB_evaluation_assets_and_model_scores')
            self.assertFalse(payload['no_preselection_test_pixel_access_claimed'])

    def test_integrity_thumbnail_exception_is_bound_and_cannot_hide_quality_access(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt=Path(folder)/'integrity.json';write(receipt,{'scope':'integrity_only'})
            policy={'test_pixels_after_selection':True,'test_prior_exposure':'historical_models_previously_evaluated',
                'test_integrity_thumbnail_exception':{'scope':'integrity_only','audit_receipt':ev.binding(receipt),
                    'automated':True,'human_quality_review':False,'model_scores_used':False}}
            self.assertEqual(ev._ft_test_access_policy(policy)['test_integrity_thumbnail_exception'],policy['test_integrity_thumbnail_exception'])
            bad=copy.deepcopy(policy);bad['test_integrity_thumbnail_exception']['model_scores_used']=True
            with self.assertRaisesRegex(ValueError,'integrity-only'):ev._ft_test_access_policy(bad)
            write(receipt,{'scope':'replaced'})
            with self.assertRaises(ValueError):ev._ft_test_access_policy(policy)

    def test_wrong_selection_or_early_test_pixels_fail_before_reader_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);study=root/'study.json';plan=root/'test_plan.json';dataset=root/'test_dataset.json'
            write(plan,{'status':'metadata_only'});plan_binding=ev.binding(plan)
            value={'evaluation':{'validation_seal':{},'validation_dataset':{},'test_cohort_plan':plan_binding,
                    'test_pixels_after_selection':True,'test_prior_exposure':'historical_models_previously_evaluated'}}
            write(study,value);study_binding=ev.binding(study)
            expected={'study_file':study_binding,'test_cohort_plan':plan_binding,'selected_roles':[{'arm':'vae_high'}]}
            receipt=root/'selection.json';write(receipt,dict(expected,selected_roles=[{'arm':'vae_low'}]))
            write(dataset,{})
            cfg={'phase':'test','run_root':str(root/'eval'),'device':'cuda:0','gpu_lock':str(root/'shared.lock'),
                 'gpu_admission':{'expected_gpu_uuid':'GPU-fixture'},'study_config':study_binding,
                 'training_run':str(root/'training'),'seal':str(plan),'dataset':str(dataset),
                 'selection_receipt':ev.binding(receipt),'validation_run':str(root/'validation')}
            config=root/'config.json';write(config,cfg)
            evidence={'study':value}
            with patch.object(ev,'_hr_training_evidence',return_value=evidence),\
                 patch.object(ev,'_ft_selection_payload',return_value=expected),\
                 patch('vsr.data.FixedCleanReader',create=True) as reader:
                with self.assertRaisesRegex(ValueError,'selection must reproduce'):ev.ft_collect_protocol(config,root/'eval'/'new')
                reader.assert_not_called()
                write(receipt,expected);cfg['selection_receipt']=ev.binding(receipt);write(config,cfg)
                write(dataset,{'selection_receipt':ev.binding(receipt),'cohort_seal':plan_binding,
                               'materialization':{'selection_frozen_before_decode':False}})
                with self.assertRaisesRegex(ValueError,'pixels must be materialized'):ev.ft_collect_protocol(config,root/'eval'/'new')
                reader.assert_not_called()


def full_matrix(root, phase, count):
    run=root/'run';run.mkdir();raw=root/'payload';raw.write_bytes(b'synthetic artifact, no real pixels')
    roles=[dict(label=f'r{i}',arm=f'a{i}',family=('baseline' if i==0 else 'hr_vae' if i%2 else 'hr_dino'),setting='low',
        weight=.01,update=2048,step=2048,cumulative_training_seconds=1.,checkpoint={'sha256':str(i)*64}) for i in range(count)]
    cohort=[{'record':{'id':f'c{i:02d}','source_id':f's{i:02d}','split':phase},
             'source_indices':list(range(33)),'cached_header':{'fps':24.}} for i in range(32)]
    protocol={'schema':ev.FT_EVALUATION_SCHEMA,'phase':phase,'roles':roles,'cohort':cohort,
        'cohort_seal_digest':'fixed','config_binding':{'path':'synthetic'},'scope':{'total_restores':count*64,'source_split':phase},
        'fixed_video_clip_ids':['c00','c01','c02'],'clean_assets':[{'record_id':f'c{i:02d}','HR':{'sha256':f'HR{i}'}} for i in range(32)]}
    write(run/'protocol.json',protocol);psha=ev.sha(run/'protocol.json')
    full={'all_frames':{'frame_count':33,'per_frame':[{}]*33,'temporal_pair_count':28,
          'per_temporal_pair':[{'indices':[i,i+1]} for i in range(2,30)],**{m:.1 for m in ev.HR_SCALAR_METRICS}}}
    rows=[]
    for role,item,sample in ev._hr_cases(protocol,{'cohort':cohort}):
        directory=ev._hr_directory(run,role,item,sample);(directory/'png').mkdir(parents=True);pngs=[]
        for i in range(33):
            path=directory/'png'/f'{i:06d}.png';os.link(raw,path);pngs.append(ev.binding(path))
        path=directory/'prediction.mp4';os.link(raw,path);video=ev.binding(path)
        obs=ev._hr_identity(protocol,role,item,sample)|{'metrics':ev._hr_scalars(full)}
        write(directory/'metrics.json',{'status':'completed','training_updates':0,'observation':obs,'full_frame_pair_metrics':full,
            'source_metadata':{'split':phase,'source_frame_indices':list(range(33)),'output_fps':24.},'PNG_artifacts':pngs,
            'video_artifact':video,'teacher_or_HR_input_to_inferrer':False,
            'inference_inputs':['LRV','native text embedding','student checkpoint'],'restoration':{'inference_inputs':['LRV','native text embedding']}})
        write(directory/'complete.json',dict(protocol_sha256=psha,metrics_sha256=ev.sha(directory/'metrics.json'),PNG_count=33,video_artifact=video))
        rows.append(obs|{'row_metrics_path':str(directory/'metrics.json'),'row_metrics_sha256':ev.sha(directory/'metrics.json')})
    write(run/'observations.json',{'observations':rows,'protocol_sha256':psha})
    for role in roles:write(run/'roles'/role['label']/'state_audit_001.json',dict(trainable_before={'same':1},trainable_after={'same':1},training_updates=0))
    write(run/'role_audits.json',[dict(role=r['label'],restores=64,training_updates=0,inference_weights='student') for r in roles])
    for i in range(32):
        directory=run/'ground_truth'/f'c{i:02d}';(directory/'png').mkdir(parents=True);pngs=[]
        for j in range(33):
            path=directory/'png'/f'{j:06d}.png';os.link(raw,path);pngs.append(ev.binding(path))
        path=directory/'GT.mp4';os.link(raw,path)
        write(directory/'complete.json',dict(protocol_sha256=psha,record_id=f'c{i:02d}',source_id=f's{i:02d}',
              source_HR_sha256=f'HR{i}',PNG_artifacts=pngs,video_artifact=ev.binding(path),frames=33,FPS=24.))
    write(run/'summary.json',dict(restores=count*64,protocol_sha256=psha,quality_gate_applied=False))
    write(run/'status.json',dict(status='completed',restores=count*64,protocol_sha256=psha,training_updates=0,summary_sha256=ev.sha(run/'summary.json')))
    return run,protocol,rows


class FixedStudyArtifactTests(unittest.TestCase):
    def test_val1024_and_test192_all_rows_PNG_MP4_and_split_are_checked(self):
        for phase,count in [('val',16),('test',3)]:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as folder:
                run,protocol,rows=full_matrix(Path(folder),phase,count)
                with patch.object(ev,'ft_collect_protocol',return_value=protocol),patch('vsr.infer.verify_output_video'):
                    result=ev.ft_verify(run)
                    self.assertEqual(result['restores'],count*64)
                    self.assertEqual(result['prediction_PNGs_verified'],count*64*33)
                    self.assertEqual(result['prediction_MP4s_verified'],count*64)
                    self.assertEqual(result['GT_MP4s_verified'],32)
                    self.assertEqual(result['validation_not_heldout_test'],phase=='val')
                    path=Path(rows[0]['row_metrics_path']);data=json.loads(path.read_text());data['source_metadata']['split']='train';write(path,data)
                    marker=path.parent/'complete.json';data=json.loads(marker.read_text());data['metrics_sha256']=ev.sha(path);write(marker,data)
                    with self.assertRaisesRegex(ValueError,'validation window'):ev.ft_verify(run)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main(verbosity=2)
