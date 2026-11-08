"""Synthetic controller orchestration only; no model/GPU/real experiment assets."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import fixed_teacher_pipeline_v2 as pipeline


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def file(path,value='synthetic fixture, never real model evidence'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value);return pipeline.binding(path)


class Fixture:
    def __init__(self,root,*,after_training=True):
        self.root=Path(root);self.source=self.root/'source';self.helpers=self.root/'work';self.runs=self.root/'runs';self.train=self.runs/'training'
        self.helpers.mkdir();self.train.mkdir(parents=True)
        for name in pipeline.CORE_NAMES:file(self.source/'vsr'/name)
        for name in pipeline.HELPERS:file(self.helpers/name)
        self.sources={name:pipeline.binding(self.source/'vsr'/name) for name in sorted(pipeline.CORE_NAMES)}
        text=file(self.root/'text.pt');self.basecfg={'model':{'family':'syntheticWan'},'conditioning':{'kind':'LR_residual'},'data':{'text_cache':text}}
        self.basecp=file(self.root/'base/Base/checkpoint_008192.pt');self.official={'transformer/config.json':file(self.root/'official/transformer/config.json')}
        self.natives={name:file(self.root/'native'/name) for name in ('models/transformers/transformer_wan.py','models/autoencoders/autoencoder_kl_wan.py','schedulers/scheduling_unipc_multistep.py')}
        self.arms=[{'label':'baseline','family':'baseline','setting':'baseline','weight':0.}]
        for family,short,weights in (('hr_vae','vae',(.11,1.1)),('hr_dino','dino',(.094,.94))):
            self.arms += [{'label':short+'_'+setting,'family':family,'setting':setting,'weight':weight} for setting,weight in zip(('low','high'),weights)]
        self.val_cohort=self.cohort('val');self.test_cohort=self.cohort('test')
        val=file(self.root/'val_plan.json',json.dumps({'cohort':self.val_cohort}));dataset=file(self.root/'val_dataset.json')
        test=file(self.root/'test_plan.json',json.dumps({'cohort':self.test_cohort}))
        metrics={key:file(self.root/f'{key}.pth') for key in ('alexnet','linear')}
        cal=self.root/'calibration';save(cal/'protocol.json',{'sources':self.sources})
        save(cal/'summary.json',{'status':'passed_fixed_teacher_preflight','scientific_updates':0,'diagnostic_optimizer_updates':27,
            'protocol_sha256':pipeline.sha(cal/'protocol.json'),'sources':self.sources,'calibrations':{'vae':{'weights':[.11,1.1]},'dino':{'weights':[.094,.94]}}})
        save(cal/'status.json',{'status':'completed','protocol_sha256':pipeline.sha(cal/'protocol.json'),'summary_sha256':pipeline.sha(cal/'summary.json')})
        self.study={'schema':'fixed_teacher_study_v1','arms':self.arms,'updates':2048,'training_seed':20261102,'student_layer':7,
            'eval_updates':[512,1024,2048],'checkpoint_updates':[512,1024,1536,2048],'head_depth':1,'smooth_l1_beta':1.,
            'EMA':False,'condition_corruption':False,'run_root':str(self.runs),'device':'cuda:0','gpu_lock':str(self.root/'shared.lock'),'gpu_uuid':'GPU-fixture',
            'model_directory':str(self.root/'official'),'metric_weights':metrics,'calibration_report':str(cal/'summary.json'),
            'evaluation':{'test_pixels_after_selection':True,'test_prior_exposure':'historical_models_previously_evaluated',
                          'validation_seal':val,'validation_dataset':dataset,'test_cohort_plan':test}}
        self.study_path=self.root/'study.json';save(self.study_path,self.study)
        self.protocol={'schema':'fixed_teacher_training_protocol_v1','study':self.study,'study_file':pipeline.binding(self.study_path),
            'sources':self.sources,'scientific_update_budget':10240,'ordinary_inference_weight_EMA':False,'memory_cap_bytes':16*1024**3,
            'device':'cuda:0','gpu_lock':self.study['gpu_lock'],'admission':{'expected_gpu_uuid':'GPU-fixture'},
            'cache':{'selection':{'training_ids':[f'train{i}' for i in range(1641)]}},'base':{'config':self.basecfg},'official_model_files':self.official,'dependencies':{'diffusers_native_sources':self.natives},
            'calibration':{'protocol':pipeline.binding(cal/'protocol.json'),'summary':pipeline.binding(cal/'summary.json'),'status_file':pipeline.binding(cal/'status.json')}}
        self.reseal()
        self.options=pipeline.Options(self.source,self.study_path,self.train,self.runs,Path(sys.executable),self.root/'outputs/result',
                                      after_training=after_training,helper_root=self.helpers)
        if after_training:self.complete_training()

    def reseal(self):
        save(self.train/'protocol.json',self.protocol)
        save(self.train/'status.json',{'status':'planned','scientific_updates':0,'protocol_sha256':pipeline.sha(self.train/'protocol.json')})

    @staticmethod
    def cohort(split):
        return [{'record':{'id':f'{split}{i}','source_id':f'{split}source{i}','split':split},'source_indices':list(range(33))} for i in range(32)]

    def complete_training(self):
        arms={}
        for arm in self.arms:
            directory=self.train/'arm_runs'/arm['label'];cp=[]
            cumulative=0.;times={}
            for step in range(1,2049):cumulative+=.1;times[step]=cumulative
            for step in (512,1024,1536,2048):cp.append(file(directory/f'Base/checkpoint_{step:06d}.pt')|{'step':step,'cumulative_training_seconds':times[step]})
            rows=[{'step':step,'arm':arm['label'],'family':arm['family'],'weight':arm['weight'],
                   'fm':1.,'align':0.,'total':1.,'grad_norm_before_clip':1.,'sigma':.5,'seconds':.1,'cumulative_training_seconds':times[step],
                   'data_index':(step-1)%1641,'record_id':f'train{(step-1)%1641}'} for step in range(1,2049)]
            log=directory/'training.jsonl';log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            arms[arm['label']]={'updates':2048,'family':arm['family'],'setting':arm['setting'],'cumulative_training_seconds':times[2048],'weight':arm['weight'],'checkpoint_files':cp,
                               'initial_student_hashes':{'same':'synthetic_initial'},'full_training_log':pipeline.binding(log)}
            save(directory/'completion.json',arms[arm['label']])
        save(self.train/'summary.json',{'schema':'fixed_teacher_training_summary_v1','protocol_sha256':pipeline.sha(self.train/'protocol.json'),'status':'completed','scientific_updates':10240,'diagnostic_updates_in_this_training':0,'arms':arms})
        save(self.train/'status.json',{'status':'completed','scientific_updates':10240,'protocol_sha256':pipeline.sha(self.train/'protocol.json'),
                                      'summary_sha256':pipeline.sha(self.train/'summary.json')})

    def roles(self,phase,selection=None):
        if phase=='test':return copy.deepcopy(selection['selected_roles'])
        roles=[{'label':'shared_initial','arm':'shared','family':'baseline','setting':'shared_initial','weight':0.,'step':8192,'update':0,'checkpoint':self.basecp}]
        summary=pipeline.read(self.train/'summary.json')
        for arm in self.arms:
            for update in (512,1024,2048):
                cp=next(c for c in summary['arms'][arm['label']]['checkpoint_files'] if c['step']==update)
                roles.append(arm|{'label':arm['label']+f'_u{update}','arm':arm['label'],'step':update,'update':update,
                                 'checkpoint':{k:cp[k] for k in ('path','bytes','sha256')}})
        for r in roles:r.update(variant='Base',config=copy.deepcopy(self.basecfg))
        return roles


class FakeRunner:
    def __init__(self,fixture,fail=None,mutate=None,mutate_protocol=False):self.f=fixture;self.calls=[];self.fail=fail;self.mutate=mutate;self.mutate_protocol=mutate_protocol;self.active=0
    def run(self,command,*,cwd,env,log):
        self.active+=1;assert self.active==1;phase=Path(log).stem;self.calls.append(phase)
        assert cwd==str(self.f.source.resolve()) and all(env[k]==v for k,v in pipeline.OFFLINE.items())
        assert env['CUDA_VISIBLE_DEVICES']=='GPU-fixture';Path(log).write_text('synthetic subprocess, no execution\n')
        try:
            if phase==self.fail:return 19
            args=list(command);get=lambda key:args[args.index(key)+1];root=self.f.runs/('pipeline_'+self.f.train.name)
            if phase=='01_train':self.f.complete_training()
            elif phase in ('02_val_config','08_test_config'):
                which='val' if phase=='02_val_config' else 'test';selected=pipeline.read(root/'selection.json') if which=='test' else None
                cfg={'schema':'fixed_teacher_evaluation_config_v1','phase':which,'study_config':pipeline.binding(self.f.study_path),
                     'training_run':str(self.f.train),'run_root':str(root),'roles':self.f.roles(which,selected),
                     'device':'cuda:0','gpu_lock':self.f.study['gpu_lock'],'gpu_admission':self.f.protocol['admission'],
                     'seal':self.f.study['evaluation']['validation_seal' if which=='val' else 'test_cohort_plan']['path'],
                     'dataset':self.f.study['evaluation']['validation_dataset']['path'] if which=='val' else str(root/'test_assets/dataset.json'),
                     'model_directory':self.f.study['model_directory'],'metric_weights':self.f.study['metric_weights']}
                if which=='test':cfg['selection_receipt']=pipeline.binding(root/'selection.json')
                save(get('--output'),cfg)
            elif phase in ('03_val_prepare','09_test_prepare'):
                cfg=pipeline.read(get('--config'));which=cfg['phase'];run=Path(get('--run-dir'));count=16 if which=='val' else 3
                scope={'roles':count,'clips':32,'sampling_seeds':[42,43],'conditions':['clean'],'total_restores':count*64,
                       'source_split':which,'frames':33,'HR_hw':[128,224],'LR_hw':[32,56],'steps':20,'CFG':1.,'tile_hw':[256,448],
                       'overlap':64,'primary_pool':'all33_frames','primary_metric':'lpips_alex_v0_1','video_policy':'all_clips','training_updates':0}
                natives={'transformer':'models/transformers/transformer_wan.py','vae':'models/autoencoders/autoencoder_kl_wan.py','scheduler':'schedulers/scheduling_unipc_multistep.py'}
                p={'schema':'fixed_teacher_clean_evaluation_protocol_v1','phase':which,'sources':self.f.sources,'study_file':pipeline.binding(self.f.study_path),
                   'scope':scope,'roles':cfg['roles'],'cohort':self.f.val_cohort if which=='val' else self.f.test_cohort,
                   'cohort_seal':pipeline.binding(cfg['seal']),'dataset':pipeline.binding(cfg['dataset']),'official_model':self.f.official,
                   'runtime':{'native':{k:self.f.natives[v] for k,v in natives.items()}},'metric_weights':self.f.study['metric_weights'],
                   'text_cache':pipeline.bound(self.f.basecfg['data']['text_cache']),'config':cfg}
                if which=='test':p.update(selection_receipt=pipeline.binding(root/'selection.json'),validation_selection=pipeline.read(root/'selection.json'))
                save(run/'protocol.json',p);save(run/'status.json',{'status':'planned','restores':0,'training_updates':0,'protocol_sha256':pipeline.sha(run/'protocol.json')})
            elif phase in ('04_val_execute','10_test_execute'):
                run=Path(get('--run-dir'));p=pipeline.read(run/'protocol.json');count=p['scope']['total_restores']
                rows=[{'role':r['label'],'record_id':i['record']['id'],'sampling_seed':s} for r in p['roles'] for i in p['cohort'] for s in (42,43)]
                save(run/'observations.json',{'observations':rows});save(run/'summary.json',{'protocol_sha256':pipeline.sha(run/'protocol.json'),'restores':count})
                save(run/'status.json',{'status':'completed','restores':count,'training_updates':0,'protocol_sha256':pipeline.sha(run/'protocol.json'),'summary_sha256':pipeline.sha(run/'summary.json')})
            elif phase=='06_val_select':
                assert '05_val_verify' in self.calls and not (root/'test_assets').exists()
                final={r['arm']:r for r in pipeline.read(root/'validation/protocol.json')['roles'] if r['update']==2048}
                selected=[final[n] for n in ('baseline','vae_low','dino_low')]
                s={'schema':'fixed_teacher_validation_selection_v1','study_file':pipeline.binding(self.f.study_path),
                   'selected_after_complete_validation':True,'test_pixels_read_by_selection':False,'selected_roles':selected,
                   'test_cohort_plan':self.f.study['evaluation']['test_cohort_plan']}
                for key,name in (('validation_protocol','protocol.json'),('validation_status','status.json'),('validation_summary','summary.json'),('validation_observations','observations.json')):s[key]=pipeline.binding(root/'validation'/name)
                save(get('--output'),s)
            elif phase=='07_test_materialize':
                assert self.calls.index('06_val_select')<self.calls.index(phase) and (root/'selection.json').is_file()
                save(Path(get('--output'))/'dataset.json',{'schema':'vsr_fixed_clean_dataset_v1','status':'completed_CPU_clean_assets','split':'test',
                    'selection_receipt':pipeline.binding(root/'selection.json'),'cohort_seal':self.f.study['evaluation']['test_cohort_plan'],
                    'materialization':{'selection_frozen_before_decode':True},'assets':[{'record_id':str(i)} for i in range(32)]})
            elif phase in ('12_val_analyze','13_test_analyze'):
                run=Path(get('--run'));which='val' if phase=='12_val_analyze' else 'test';dest=Path(get('--output-dir'))
                save(dest/'analysis.json',{'schema':'fixed_teacher_descriptive_analysis_v1','phase':which,'study_file':pipeline.binding(self.f.study_path),
                    'runtime_sources':self.f.sources,'protocol_sha256':pipeline.sha(run/'protocol.json'),'matrix_rows':1024 if which=='val' else 192})
                for name in ('report.md','quality_vs_updates.png','quality_vs_training_seconds.png'):file(dest/name)
            if phase=='04_val_execute' and self.mutate_protocol:
                run=root/'validation';p=pipeline.read(run/'protocol.json');p['scope']['steps']=21;save(run/'protocol.json',p)
                summary=pipeline.read(run/'summary.json');summary['protocol_sha256']=pipeline.sha(run/'protocol.json');save(run/'summary.json',summary)
                state=pipeline.read(run/'status.json');state['protocol_sha256']=pipeline.sha(run/'protocol.json');state['summary_sha256']=pipeline.sha(run/'summary.json');save(run/'status.json',state)
            if phase==self.mutate:file(self.f.source/'vsr/data.py','source changed during child')
            return 0
        finally:self.active-=1


class PipelineTests(unittest.TestCase):
    def test_after_training_serial_full_chain_and_wall_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);runner=FakeRunner(f);result=pipeline.Pipeline(f.options,runner).execute()
            self.assertEqual(result['status'],'awaiting_root_delivery');self.assertEqual(len(runner.calls),12)
            self.assertNotIn('01_train',runner.calls);self.assertEqual(runner.calls[0],'02_val_config')
            root=f.runs/'pipeline_training'
            for name,stage in (('validation','evaluation_val'),('test','evaluation_test')):
                receipt=pipeline.read(root/name/'execution_wall_time.json');self.assertEqual(receipt['stage'],stage)
                self.assertGreaterEqual(receipt['wall_seconds'],0.);self.assertEqual(len(receipt['attempts']),1)
            with self.assertRaises(ValueError):pipeline.Pipeline(f.options,FakeRunner(f)).execute()

    def test_legacy_text_cache_without_bytes_and_extra_provenance_runs_complete_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);original=f.basecfg['data']['text_cache']
            legacy={'source_run':'/historical/reference/text_embedding_run','path':original['path'],'sha256':original['sha256']}
            f.basecfg['data']['text_cache']=legacy;f.reseal();f.complete_training()
            self.assertEqual(pipeline.bound(legacy),original)
            runner=FakeRunner(f);result=pipeline.Pipeline(f.options,runner).execute()
            self.assertEqual(result['status'],'awaiting_root_delivery')
            self.assertEqual(pipeline.read(f.runs/'pipeline_training/validation/protocol.json')['text_cache'],original)
            self.assertNotIn('bytes',legacy);self.assertIn('source_run',legacy)

    def test_canonical_binding_rejects_wrong_SHA_size_or_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'text.pt';actual=file(path)
            legacy={'source_run':'/history','path':str(path),'sha256':actual['sha256']}
            self.assertEqual(set(pipeline.bound(legacy)),{'path','bytes','sha256'})
            for item in (legacy|{'sha256':'f'*64},legacy|{'bytes':actual['bytes']+1},{'path':str(path)}):
                with self.subTest(item=item),self.assertRaises(ValueError):pipeline.bound(item)
            self.assertEqual(pipeline.bound(legacy|{'bytes':actual['bytes']}),actual)

    def test_explicit_include_training_uses_one_training_child(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder,after_training=False);runner=FakeRunner(f);pipeline.Pipeline(f.options,runner).execute()
            self.assertEqual(runner.calls[0],'01_train');self.assertEqual(runner.calls.count('01_train'),1)

    def test_failed_execute_stops_before_selection_test_pixels_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);runner=FakeRunner(f,fail='04_val_execute')
            with self.assertRaises(ValueError):pipeline.Pipeline(f.options,runner).execute()
            root=f.runs/'pipeline_training';failure=pipeline.read(root/'failure.json')
            self.assertEqual(failure['phase'],'04_val_execute');self.assertEqual(failure['exit_code'],19)
            self.assertFalse(failure['automatic_retry']);self.assertFalse((root/'test_assets').exists())
            self.assertEqual(runner.calls[-1],'04_val_execute')

    def test_changed_frozen_source_failstops_even_after_zero_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);runner=FakeRunner(f,mutate='02_val_config')
            with self.assertRaisesRegex(ValueError,'Bound input changed'):pipeline.Pipeline(f.options,runner).execute()
            self.assertEqual(runner.calls,['02_val_config'])

    def test_reviewed_protocol_change_cannot_be_hidden_by_self_consistent_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);runner=FakeRunner(f,mutate_protocol=True)
            with self.assertRaisesRegex(ValueError,'Bound input changed'):pipeline.Pipeline(f.options,runner).execute()
            self.assertEqual(runner.calls[-1],'04_val_execute')
            self.assertFalse((f.runs/'pipeline_training'/'selection.json').exists())

    def test_metadata_inspection_never_creates_a_pipeline_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);result=pipeline.inspect_inputs(f.options)
            self.assertEqual(result['mode'],'after_training')
            self.assertFalse(Path(result['pipeline_dir']).exists());self.assertFalse(f.options.output_dir.exists())

    def test_resigned_wrong_arm_data_time_or_method_metadata_are_rejected(self):
        for bad in ('log_arm','checkpoint_arm','setting','data_order','time_prefix','summary_protocol','EMA','corruption'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder)
                if bad in ('EMA','corruption'):
                    f.study['EMA' if bad=='EMA' else 'condition_corruption']=True;save(f.study_path,f.study)
                    f.protocol['study_file']=pipeline.binding(f.study_path);f.reseal();f.complete_training()
                else:
                    summary=pipeline.read(f.train/'summary.json');row=summary['arms']['baseline']
                    if bad=='log_arm':row['full_training_log']=summary['arms']['vae_low']['full_training_log']
                    elif bad=='checkpoint_arm':row['checkpoint_files'][0]=summary['arms']['vae_low']['checkpoint_files'][0]
                    elif bad=='setting':row['setting']='high'
                    elif bad=='summary_protocol':summary['protocol_sha256']='f'*64
                    else:
                        path=Path(row['full_training_log']['path']);rows=[json.loads(x) for x in path.read_text().splitlines()]
                        if bad=='data_order':rows[16]['data_index']=99
                        else:rows[16]['cumulative_training_seconds']+=1.
                        path.write_text(''.join(json.dumps(r)+'\n' for r in rows));row['full_training_log']=pipeline.binding(path)
                    save(f.train/'arm_runs/baseline/completion.json',row);save(f.train/'summary.json',summary)
                    state=pipeline.read(f.train/'status.json');state['summary_sha256']=pipeline.sha(f.train/'summary.json');save(f.train/'status.json',state)
                runner=FakeRunner(f)
                with self.assertRaises(ValueError):pipeline.Pipeline(f.options,runner).execute()
                self.assertFalse(runner.calls)

    def test_default_after_training_refuses_running_or_planned(self):
        for state in ('planned','running'):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder);p=pipeline.read(f.train/'status.json');p['status']=state;save(f.train/'status.json',p)
                runner=FakeRunner(f)
                with self.assertRaises(ValueError):pipeline.Pipeline(f.options,runner).execute()
                self.assertFalse(runner.calls)

    def test_source_budget_and_output_path_mismatches_refuse_before_any_child(self):
        for bad in ('source','budget','output','partial_log','bad_CP','cal_cleanup'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder)
                if bad=='source':file(f.source/'vsr/model.py','changed')
                elif bad=='budget':f.protocol['scientific_update_budget']=12288;f.reseal();f.complete_training()
                elif bad=='output':f.options=pipeline.Options(f.source,f.study_path,f.train,f.runs,Path(sys.executable),f.root/'outside',helper_root=f.helpers)
                elif bad=='partial_log':
                    row=pipeline.read(f.train/'summary.json')['arms']['baseline'];Path(row['full_training_log']['path']).write_text('')
                elif bad=='bad_CP':
                    row=pipeline.read(f.train/'summary.json')['arms']['baseline'];Path(row['checkpoint_files'][0]['path']).write_text('corrupt')
                else:
                    p=Path(f.protocol['calibration']['status_file']['path']);save(p,{'status':'failed'})
                runner=FakeRunner(f)
                with self.assertRaises(ValueError):pipeline.Pipeline(f.options,runner).execute()
                self.assertFalse(runner.calls)


if __name__=='__main__':unittest.main()
