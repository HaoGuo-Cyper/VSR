"""TRD fair-training evidence, fixed-final lock and complete native artifact checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from vsr import evaluate as ev
from test_hr_latent_evaluate import TrainingFixture,write
import test_fixed_teacher_evaluate as ft


class TRDTrainingFixture(TrainingFixture):
    def __init__(self,root):
        super().__init__(root)
        write(root/'calibration_protocol.json',{'schema':'wan_trd_calibration_protocol_v1'})
        write(root/'calibration_summary.json',{'status':'passed_trd_preflight','sources':self.sources,
              'protocol_sha256':ev.sha(root/'calibration_protocol.json'),'scientific_updates':0,
              'diagnostic_optimizer_updates':41,'calibration':{'weight':.1},'teacher_initial_hashes':{'trainable.weight':'a'*64}})
        write(root/'calibration_status.json',{'status':'completed','protocol_sha256':ev.sha(root/'calibration_protocol.json'),
              'summary_sha256':ev.sha(root/'calibration_summary.json')})
        self.study={'schema':'wan_trd_study_v1','arms':[
            dict(label=label,family=label,setting='common',weight=0. if label=='baseline' else .1)
            for label in ('baseline','frozen','ema')],
            'updates':2048,'eval_updates':[512,1024,2048],'checkpoint_updates':[512,1024,1536,2048],
            'calibration_report':str(root/'calibration_summary.json')}
        write(root/'study.json',self.study);self.study_file=ev.binding(root/'study.json')
        self.protocol.update(schema='wan_trd_training_protocol_v1',study=self.study,study_file=self.study_file,
            calibration={key:ev.binding(root/f'calibration_{name}.json') for key,name in
                         [('protocol','protocol'),('summary','summary'),('status_file','status')]})
        self.summary.update(schema='wan_trd_training_summary_v1')
        template=copy.deepcopy(self.summary['arms']['baseline']);self.summary['arms']={}
        ids=self.protocol['cache']['selection']['training_ids']
        for arm in self.study['arms']:
            label=arm['label'];directory=self.run/'arm_runs'/label;directory.mkdir(exist_ok=True)
            record=copy.deepcopy(template);record.update(weight=arm['weight'],teacher_kind=label,
                family=label,setting='common',EMA_updates=2048 if label=='ema' else 0,
                teacher_initial_hashes=None if label=='baseline' else {'trainable.weight':'a'*64})
            log=directory/'training.jsonl'
            rows=[dict(step=i,arm=label,weight=arm['weight'],data_index=(i-1)%1641,record_id=ids[(i-1)%1641],
                fm=.2,align=.1,total=.2+arm['weight']*.1,grad_norm_before_clip=1.,sigma=.5,
                seconds=1.,cumulative_training_seconds=float(i),gpu_event_seconds=.5,
                teacher_forward=label!='baseline',EMA_updates=i if label=='ema' else 0,
                timestep_index=50,native_timestep=500.,teacher_sigma=None if label=='baseline' else .25,
                teacher_timestep_index=None if label=='baseline' else 75,
                teacher_native_timestep=None if label=='baseline' else 250.) for i in range(1,2049)]
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

    def checked(self):
        with patch.object(ev,'_hr_sources',return_value=self.sources):
            return ev._hr_training_evidence(self.run,self.study_file,trd=True)

    def edit_row(self,arm,index,**changes):
        record=self.summary['arms'][arm];path=Path(record['full_training_log']['path'])
        rows=[json.loads(x) for x in path.read_text().splitlines()];rows[index].update(changes)
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows));record['full_training_log']=ev.binding(path)
        write(self.run/'arm_runs'/arm/'completion.json',record);self.publish_summary()


def role_config(f,evidence):
    roles=[dict(label='shared',checkpoint=f.base['checkpoint'],step=8192,arm='shared',weight=0.,update=0,cumulative_training_seconds=0.)]
    for arm,record in evidence['arms'].items():
        for cp in record['checkpoint_files']:
            if cp['step']==1536:continue
            roles.append(dict(label=f'{arm}_{cp["step"]}',checkpoint=cp,step=cp['step'],arm=arm,
                weight=record['weight'],update=cp['step'],cumulative_training_seconds=cp['cumulative_training_seconds']))
    return {'roles':roles,'allowed_checkpoint_roots':[str(f.root)]}


class TRDMetadataTests(unittest.TestCase):
    def test_complete6144_12checkpoints_val10roles_testall3final(self):
        with tempfile.TemporaryDirectory() as folder:
            f=TRDTrainingFixture(Path(folder));evidence=f.checked();cfg=role_config(f,evidence)
            self.assertEqual(evidence['scientific_updates'],6144)
            self.assertEqual(sum(len(a['checkpoint_files']) for a in evidence['arms'].values()),12)
            with patch('vsr.checkpoints._check_config'):
                roles=ev._hr_role_specs(cfg,evidence,trd=True)
                self.assertEqual(len(roles),10)
                finals=[r for r in roles if r['update']==2048]
                self.assertEqual([r['arm'] for r in finals],['baseline','frozen','ema'])
                self.assertEqual(len(ev._hr_role_specs(dict(cfg,roles=finals),evidence,trd=True,phase='test',selection={'final_roles':finals})),3)
                bad=copy.deepcopy(finals);bad[-1]['update']=1024
                with self.assertRaises(ValueError):ev._hr_role_specs(dict(cfg,roles=bad),evidence,trd=True,phase='test',selection={'final_roles':finals})

    def test_teacher_kind_update_initialization_and_completion_rejected(self):
        for change in ('kind','update','initial','completion'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                f=TRDTrainingFixture(Path(folder))
                if change=='kind':f.summary['arms']['ema']['teacher_kind']='frozen'
                if change=='update':f.summary['arms']['ema']['EMA_updates']=2047
                if change=='initial':f.summary['arms']['ema']['teacher_initial_hashes']={'all_trainable':'different'}
                if change=='completion':write(f.run/'arm_runs'/'ema'/'completion.json',{})
                f.publish_summary()
                with self.assertRaises(ValueError):f.checked()

    def test_data_timestep_teacher_schedule_and_nonfinite_rejected(self):
        for update in ({'native_timestep':400.},{'teacher_sigma':.3},{'teacher_forward':False},
                       {'EMA_updates':0},{'fm':float('nan')}):
            with self.subTest(update=update),tempfile.TemporaryDirectory() as folder:
                f=TRDTrainingFixture(Path(folder));f.edit_row('ema',0,**update)
                with self.assertRaises(ValueError):f.checked()

    def test_one_common_positive_weight_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            f=TRDTrainingFixture(Path(folder));f.study['arms'][-1]['weight']=.2
            write(f.root/'study.json',f.study);f.study_file=ev.binding(f.root/'study.json')
            f.protocol.update(study=f.study,study_file=f.study_file);write(f.run/'protocol.json',f.protocol)
            f.psha=ev.sha(f.run/'protocol.json');f.summary['protocol_sha256']=f.psha;f.publish_summary()
            with self.assertRaisesRegex(ValueError,'common positive'):f.checked()

    def test_native_student_load_must_match_training_identity_not_teacher_weights(self):
        role={'arm':'ema','update':1024,'weight':.1};protocol={'training_evidence':{'protocol_sha256':'training'}}
        metadata=dict(schema='vsr_trd_training_state_v1',protocol_sha256='training',arm='ema',completed_updates=1024,
                      training_seed=20261102,weight=.1,student_layer=9,EMA_updates=1024)
        audit=dict(training_only_state_restored=False,EMA_weights_selected=False,trd_training_metadata=metadata)
        ev._trd_inference_checkpoint_audit(protocol,role,audit)
        for key,value in [('arm','frozen'),('completed_updates',2048),('EMA_updates',1023),('student_layer',7)]:
            bad=copy.deepcopy(audit);bad['trd_training_metadata'][key]=value
            with self.assertRaises(ValueError):ev._trd_inference_checkpoint_audit(protocol,role,bad)
        bad=dict(audit,EMA_weights_selected=True)
        with self.assertRaises(ValueError):ev._trd_inference_checkpoint_audit(protocol,role,bad)

    def test_final_lock_keeps_every_final_without_using_metric_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'validation';run.mkdir()
            for name in ('test_plan','test_dataset'):write(root/(name+'.json'),{'prior_materialized':True})
            study={'evaluation':{'test_cohort_plan':ev.binding(root/'test_plan.json'),
                    'test_dataset':ev.binding(root/'test_dataset.json'),'test_prior_exposure':'historical_models_previously_evaluated'}}
            write(root/'study.json',study)
            roles=[dict(label=a+'_2048',arm=a,family=a,setting='common',weight=0. if a=='baseline' else .1,
                update=2048,step=2048,cumulative_training_seconds=10.,checkpoint={'sha256':a},config_binding={})
                for a in ('baseline','frozen','ema')]
            protocol={'phase':'val','roles':roles,'study_file':ev.binding(root/'study.json'),'sources':{}}
            write(run/'protocol.json',protocol)
            for name in ('status','summary','observations'):write(run/(name+'.json'),{'metric_values':'deliberately not numerical'})
            with patch.object(ev,'trd_verify',return_value={'restores':640}):
                payload=ev._trd_finals_payload(run)
                output=root/'lock.json';ev.trd_lock_finals(run,output)
                with self.assertRaisesRegex(ValueError,'immutable'):ev.trd_lock_finals(run,output)
            self.assertEqual([r['arm'] for r in payload['final_roles']],['baseline','frozen','ema'])
            self.assertFalse(payload['metric_values_used_to_select']);self.assertFalse(payload['pristine_test_claimed'])

    def test_clean_asset_collection_normalizes_legacy_text_and_keeps_old_test_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);text=root/'text.pt';text.write_bytes(b'only hashed, never loaded')
            weights={}
            for name in ('alexnet','linear'):
                path=root/(name+'.pt');path.write_bytes(name.encode());weights[name]=ev.binding(path)
            evaluation={'test_prior_exposure':'historical_models_previously_evaluated'}
            for key in ('validation_seal','validation_dataset','test_cohort_plan','test_dataset'):
                path=root/'assets'/(key+'.json');write(path,{'historical_assets':True});evaluation[key]=ev.binding(path)
            study=root/'study.json';write(study,{'evaluation':evaluation})
            study_file=ev.binding(study);raw=root/'raw.npy';raw.write_bytes(b'hashed clean pixels')
            assets=[{'record_id':f'c{i}','HR':ev.binding(raw),'clean':ev.binding(raw)} for i in range(32)]
            for phase in ('val','test'):
                cohort=[{'record':{'id':f'c{i}','source_id':f's{i:02d}','split':phase}} for i in range(32)]
                dataset=evaluation['validation_dataset' if phase=='val' else 'test_dataset']
                description={'source_split':phase,'dataset_binding':dataset,'cohort':cohort,'assets':assets,
                             'seal_digest':'fixed','data_independence':{'prior_test_access':True}}
                legacy={'path':str(text),'sha256':ev.sha(text),'source_run':'historical','source_sha256':'legacy'}
                cfg={'phase':phase,'run_root':str(root/'evaluations'),'device':'cuda:0','gpu_lock':str(root/'gpu.lock'),
                     'gpu_admission':{'expected_gpu_uuid':'GPU-test'},'study_config':study_file,'training_run':'unused',
                     'seal':evaluation['validation_seal' if phase=='val' else 'test_cohort_plan']['path'],
                     'dataset':dataset['path'],'text_cache':legacy,'metric_weights':weights,'model_directory':'fixture'}
                receipt={'study_file':study_file,'test_cohort_plan':evaluation['test_cohort_plan'],'test_dataset':evaluation['test_dataset']}
                write(root/'lock.json',receipt)
                if phase=='test':cfg.update(finals_receipt=ev.binding(root/'lock.json'),validation_run='unused')
                config=root/(phase+'.json');write(config,cfg)
                roles=[{'config':{'data':{'text_cache':legacy}}} for _ in range(10 if phase=='val' else 3)]
                with patch.object(ev,'_hr_training_evidence',return_value={'study':{'evaluation':evaluation}}),\
                     patch.object(ev,'_hr_role_specs',return_value=roles),\
                     patch.object(ev,'_trd_finals_payload',return_value=receipt),\
                     patch.object(ev,'model_bindings',return_value={}),patch.object(ev,'runtime_bindings',return_value={}),\
                     patch('vsr.data.FixedCleanReader') as reader:
                    reader.return_value.describe.return_value=description
                    protocol=ev.trd_collect_protocol(config,root/'evaluations'/phase)
                    self.assertEqual(protocol['scope']['total_restores'],640 if phase=='val' else 192)
                    self.assertEqual(protocol['text_cache'],ev.binding(text))
                    self.assertFalse(protocol['data_independence']['historically_pristine_test'])
                    self.assertFalse(protocol['data_independence']['this_study_weight_selection'])
                    self.assertTrue(protocol['data_independence']['test_assets_previously_materialized'])
                    reader.return_value.close.assert_called_once()

    def test_bad_lock_rejected_before_reader_or_model_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('seal','dataset','study','receipt'):write(root/(name+'.json'),{})
            evaluation={'test_prior_exposure':'historical_models_previously_evaluated'}
            cfg={'phase':'test','run_root':str(root/'eval'),'device':'cuda:0','gpu_lock':str(root/'gpu.lock'),
                 'gpu_admission':{'expected_gpu_uuid':'GPU-test'},'study_config':ev.binding(root/'study.json'),
                 'training_run':'unused','seal':str(root/'seal.json'),'dataset':str(root/'dataset.json'),
                 'finals_receipt':ev.binding(root/'receipt.json'),'validation_run':'unused'}
            write(root/'cfg.json',cfg)
            with patch.object(ev,'_hr_training_evidence',return_value={'study':{'evaluation':evaluation}}),\
                 patch.object(ev,'_trd_finals_payload',return_value={'different':True}),\
                 patch('vsr.data.FixedCleanReader') as reader:
                with self.assertRaisesRegex(ValueError,'final lock must reproduce'):ev.trd_collect_protocol(root/'cfg.json',root/'eval'/'new')
                reader.assert_not_called()


def trd_full_matrix(root,phase):
    def prepare_protocol(path,value):
        if path.name=='protocol.json':
            value['training_evidence']={'protocol_sha256':'training'}
            for i,role in enumerate(value['roles']):
                if phase=='val' and i==0:arm,update='shared',0
                elif phase=='val':arm,update=('baseline','frozen','ema')[(i-1)//3],(512,1024,2048)[(i-1)%3]
                else:arm,update=('baseline','frozen','ema')[i],2048
                role.update(arm=arm,update=update,step=8192 if update==0 else update,config={},
                            family='baseline' if arm=='shared' else arm,setting='shared_initial' if arm=='shared' else 'common',
                            weight=0. if arm in ('shared','baseline') else .1)
        return write(path,value)
    with patch.object(ev,'FT_EVALUATION_SCHEMA',ev.TRD_EVALUATION_SCHEMA),patch.object(ft,'write',side_effect=prepare_protocol):
        run,protocol,rows=ft.full_matrix(root,phase,10 if phase=='val' else 3)
    for role in protocol['roles']:
        audit={'step':role['step'],'checkpoint_sha256':role['checkpoint']['sha256'],'config':{},
               'training_only_state_restored':False,'EMA_weights_selected':False}
        if role['update']:
            audit['trd_training_metadata']={'schema':'vsr_trd_training_state_v1','protocol_sha256':'training',
                'arm':role['arm'],'completed_updates':role['update'],'training_seed':20261102,
                'weight':role['weight'],'student_layer':9,'EMA_updates':role['update'] if role['arm']=='ema' else 0}
        write(run/'roles'/role['label']/'load_audit_001.json',audit)
    return run,protocol,rows


class TRDArtifactTests(unittest.TestCase):
    def test_full_val640_test192_all_PNGs_MP4s_and_training_identity(self):
        for phase,total in [('val',640),('test',192)]:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as folder:
                run,protocol,rows=trd_full_matrix(Path(folder),phase)
                with patch.object(ev,'trd_collect_protocol',return_value=protocol),patch('vsr.infer.verify_output_video'):
                    result=ev.trd_verify(run)
                    self.assertEqual(result['restores'],total);self.assertEqual(result['prediction_PNGs_verified'],total*33)
                    self.assertEqual(result['prediction_MP4s_verified'],total);self.assertTrue(result['all_three_arms_retained'])
                    path=run/'roles'/protocol['roles'][-1]['label']/'load_audit_001.json'
                    value=ev.read(path);value['trd_training_metadata']['EMA_updates']=1;write(path,value)
                    with self.assertRaisesRegex(ValueError,'training-only identity'):ev.trd_verify(run)
                self.assertFalse(torch.cuda.is_initialized())

    def test_parser_exposes_no_implicit_execution(self):
        parser=ev.build_parser()
        args=parser.parse_args(['trd-build-config','--training-run','t','--phase','val','--output','o','--run-root','r'])
        self.assertEqual(args.command,'trd-build-config')
        args=parser.parse_args(['trd-prepare','--config','c','--run-dir','r']);self.assertEqual(args.command,'trd-prepare')
        args=parser.parse_args(['trd-execute','--run-dir','r','--expected-protocol-sha','x','--resume']);self.assertTrue(args.resume)
        args=parser.parse_args(['trd-lock-finals','--validation-run','v','--output','o']);self.assertEqual(args.output,Path('o'))


class TRDBuilderTests(unittest.TestCase):
    def test_valid_metadata_val_and_locked_test_config_without_GPU(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f=TRDTrainingFixture(root);evidence=f.checked();study=evidence['study']
            evaluation={'test_prior_exposure':'historical_models_previously_evaluated'}
            for key in ('validation_seal','validation_dataset','test_cohort_plan','test_dataset'):
                path=root/'assets'/(key+'.json');write(path,{'synthetic':True});evaluation[key]=ev.binding(path)
            text=root/'text.pt';text.write_bytes(b'only digest');legacy={'path':str(text),'sha256':ev.sha(text),'source_run':'historical'}
            evidence['base']['config']['data']['text_cache']=legacy
            write(root/'starting'/'config.json',evidence['base']['config'])
            for arm in ('baseline','frozen','ema'):write(f.run/'arm_runs'/arm/'config.json',evidence['base']['config'])
            study.update(evaluation=evaluation,allowed_base_root=str(root/'starting'),model_directory='model',
                         device='cuda:0',gpu_lock=str(root/'gpu.lock'),metric_weights={'alexnet':ev.binding(text),'linear':ev.binding(text)})
            f.protocol['admission']={'expected_gpu_uuid':'GPU-fixture'};write(f.run/'protocol.json',f.protocol)
            with patch.object(ev,'_hr_training_evidence',return_value=evidence),patch('vsr.checkpoints._check_config'):
                val=root/'valconfig.json'
                out=ev.trd_build_config(training_run=f.run,phase='val',output=val,run_root=root/'evaluations')
                self.assertEqual(out['restores'],640);self.assertEqual(ev.read(val)['text_cache'],ev.binding(text))
                with self.assertRaisesRegex(ValueError,'immutable'):ev.trd_build_config(training_run=f.run,phase='val',output=val,run_root=root/'evaluations')
                roles=ev._hr_role_specs(ev.read(val),evidence,trd=True)
                receipt={'study_file':f.study_file,'final_roles':[r for r in roles if r['update']==2048]}
                lock=root/'lock.json';write(lock,receipt)
                with patch.object(ev,'_trd_finals_payload',return_value=receipt):
                    out=ev.trd_build_config(training_run=f.run,phase='test',output=root/'testconfig.json',run_root=root/'evaluations',
                                      validation_run=root/'validation',finals_receipt=lock)
                self.assertEqual(out['restores'],192);self.assertFalse(out['metrics_used_for_selection'])
                self.assertEqual([r['arm'] for r in ev.read(root/'testconfig.json')['roles']],['baseline','frozen','ema'])

    def test_incomplete_training_rejected_before_config_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);study=root/'study.json';write(study,{})
            run=root/'training';write(run/'protocol.json',{'study_file':ev.binding(study)})
            with patch.object(ev,'_hr_training_evidence',side_effect=ValueError('incomplete training')):
                with self.assertRaisesRegex(ValueError,'incomplete training'):
                    ev.trd_build_config(training_run=run,phase='val',output=root/'config.json',run_root=root/'evaluations')
            self.assertFalse((root/'config.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
