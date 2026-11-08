"""Synthetic byte-exact regression tests; no tensors, real results or GPU."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import fixed_teacher_shared_initial_regression as regression


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def file(path, value=b'synthetic bytes'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value);return regression.binding(path)


class Fixture:
    def __init__(self, root, *, pngs=True):
        self.root=Path(root);self.old=self.root/'old';self.new=self.root/'new';self.old.mkdir();self.new.mkdir()
        self.assetroot=self.root/'assets';self.assetroot.mkdir()
        self.png=file(self.root/'one_fixture.png',b'\x89PNG\r\n\x1a\nsynthetic test bytes, no decode performed')
        self.old_items=[];self.new_items=[];oldassets=[];newassets=[]
        for i in range(32):
            rid=f'clip{i:02d}';record={'id':rid,'source_id':f'source{i:02d}','split':'val'}
            item={'record':record,'source_indices':list(range(33))};self.old_items.append(item)
            newrid=rid if i<31 else 'replacement'
            newitem=copy.deepcopy(item);newitem['record']['id']=newrid
            newitem['retained_original_window']=i<31;self.new_items.append(newitem)
            om={'record_id':rid,'source_id':record['source_id'],'source_sha256':'raw-video-not-read',
                'source_indices':list(range(33)),'temporal_stride':1,'original_fps':24.,'output_fps':24.,
                'source_hw':[720,1280],'resize_hw':[128,228],'crop_tlhw':[0,2,128,224],
                'degradation':{'scale':4},'valid_region':'entire_crop_no_padding'}
            assets={k:file(self.assetroot/rid/(k+'.npy'),('synthetic-'+rid+k).encode())|{'shape':[1,3,33,128,224] if k=='HR' else [1,3,33,32,56],'dtype':'torch.float32','rgb_f32_sha256':'same-rgb-'+rid+k} for k in ('HR','clean')}
            oldassets.append({'record_id':rid,'source_id':record['source_id'],'source_metadata':om,**assets})
            nm=copy.deepcopy(om);nm['record_id']=newrid
            newassets.append({'record_id':newrid,'source_id':record['source_id'],'metadata':nm,
                              'reused_original_RGB_bytes':i<31,**copy.deepcopy(assets)})
        oldseal=self.root/'oldseal.json';save(oldseal,{'project_root':str(self.root),'cohort':self.old_items,'seal_digest':'old-seal'})
        olddata=self.root/'olddata.json';save(olddata,{'assets':oldassets})
        self.newseal=self.root/'newseal.json';save(self.newseal,{'project_root':str(self.root),'cohort':self.new_items,'original_cohort':regression.binding(oldseal),'seal_digest':'new-seal'})
        self.newdata=self.root/'newdata.json';save(self.newdata,{'split':'val','cohort_seal':regression.binding(self.newseal),'cohort_seal_digest':'new-seal','assets':newassets})
        sources={};new_sources={}
        symbols={'checkpoints':('_sha256_file','_safe_checkpoint','_check_config','load_training_checkpoint'),
                 'evaluate':('_save_pngs','_crop_policy','_validate_restore')}
        for mod in ('__init__','data','teacher','losses','train','infer','model','vae','metrics','checkpoints','evaluate'):
            code='\n'.join('def '+name+'():\n    return 1\n' for name in symbols.get(mod,('dummy',)))
            sources[mod]=file(self.root/'oldsource'/(mod+'.py'),code.encode())
            new_sources[mod+'.py']=file(self.root/'newsource'/(mod+'.py'),code.encode())
        self.cp=file(self.root/'Base/checkpoint_008192.pt');self.text=file(self.root/'text.pt')
        official={'transformer/config.json':file(self.root/'official/config.json',b'{}'), 'weights.safetensors':file(self.root/'official/weights.safetensors')}
        runtime={'python':'synthetic','packages':{'torch':{'version':'synthetic-no-import'}},'native':{'transformer':file(self.root/'native.py')}}
        config={'model':{'repo_id':'official'},'data':{'num_frames':33},'training':{'variants':['Base']}}
        self.oldrole={'role':'20261102','training_seed':20261102,'variant':'Base','expected_step':8192,'checkpoint':self.cp,'config':config}
        self.newrole={'label':'shared_initial','arm':'shared','family':'baseline','setting':'shared_initial','update':0,'weight':0.,'step':8192,'variant':'Base','checkpoint':self.cp,'config':config}
        self.oldp={'schema':'vsr_evaluation_v2','kind':'bases','roles':[self.oldrole], 'source_bindings':sources,
                   'cohort_seal':regression.binding(oldseal),'dataset':regression.binding(olddata),'text_cache':self.text,
                   'official_model':official,'runtime':runtime,'strict_environment':{'CUBLAS':'fixed'},'path_map':{},
                   'scope':{'conditions':['clean','blur','noise','codec_allintra','drop_repeat','drop_black'],'sampling_seeds':[42,43],'frames':33,'steps':20,'CFG':1.,'tile_hw':[256,448],'overlap':64,'training_updates':0}}
        roles=[self.newrole]+[dict(self.newrole,label=f'other{j}',arm=f'arm{j}',update=512) for j in range(15)]
        self.newp={'schema':'fixed_teacher_clean_evaluation_protocol_v1','phase':'val','roles':roles,'sources':new_sources,
                   'cohort_seal':regression.binding(self.newseal),'dataset':regression.binding(self.newdata),'cohort':self.new_items,
                   'text_cache':self.text,'official_model':official,'runtime':runtime,'strict_environment':{'CUBLAS':'fixed'},
                   'scope':self.oldp['scope']|{'conditions':['clean'],'roles':16,'clips':32,'total_restores':1024}}
        save(self.old/'protocol.json',self.oldp);save(self.new/'protocol.json',self.newp)
        audit={'checkpoint_sha256':self.cp['sha256'],'format':'vsr_training_v1','step':8192,'variant':'Base','config':config,
               'official_repo_id':'official','official_revision':'frozen','restored_parameter_count':1,'restored_parameter_names':['LoRA'],
               'checkpoint_tensor_dtypes':['float32'],'destination_tensor_dtypes':['float32'],'optimizer_state_restored':False,
               'scheduler_state_restored':False,'rng_state_restored':False,'frozen_official_parameters_restored_from_task_state':False,
               'inference_weights':'student','training_only_state_restored':False,'EMA_weights_selected':False,'training_only_fields_present':[]}
        save(self.old/'roles/20261102/load_audit.json',audit);save(self.new/'roles/shared_initial/load_audit_001.json',audit)
        state={'role':'20261102','training_updates':0,'inference_weights':'student','trainable_before':{'same':'hash'},'trainable_after':{'same':'hash'}}
        save(self.old/'role_audits.json',[state]);save(self.new/'roles/shared_initial/state_audit_001.json',state|{'role':'shared_initial'})
        self.oldrows=[];self.newrows=[]
        if pngs:
            for item in self.old_items:
                for seed in (42,43):
                    self.oldrows.append(self.make_row(self.old,self.oldp,item,seed,old=True))
            for role in roles:
                for item in self.new_items:
                    for seed in (42,43):
                        if role['label']=='shared_initial':self.newrows.append(self.make_row(self.new,self.newp,item,seed,old=False))
                        else:self.newrows.append({'role':role['label'],'record_id':item['record']['id'],'sampling_seed':seed})
        # Remaining historical rows are metadata-only; only Base01 clean has PNG payloads.
        self.oldp['roles'] += [dict(self.oldrole, role=str(seed), training_seed=seed) for seed in (20261103,20261104)]
        save(self.old/'protocol.json',self.oldp)
        for entry in self.oldrows:
            marker=Path(entry['row_metrics_path']).parent/'complete.json';value=regression.read(marker);value['protocol_sha256']=regression.sha(self.old/'protocol.json');save(marker,value)
        if pngs:
            for role in self.oldp['roles']:
                for item in self.old_items:
                    for condition in self.oldp['scope']['conditions']:
                        for seed in (42,43):
                            if role['training_seed']==20261102 and condition=='clean':continue
                            self.oldrows.append({'training_seed':role['training_seed'],'record_id':item['record']['id'],'condition':condition,'sampling_seed':seed})
        self.seal_states()

    def make_row(self, run, protocol, item, seed, *, old):
        rid=item['record']['id'];directory=run/'results'/('20261102' if old else 'shared_initial')/rid
        if old:directory=directory/'clean'
        directory=directory/f'seed_{seed}';png=[]
        for frame in range(33):
            path=directory/'png'/f'{frame:06d}.png';path.parent.mkdir(parents=True,exist_ok=True);os.link(self.png['path'],path)
            png.append({'path':str(path.resolve()),'bytes':self.png['bytes'],'sha256':self.png['sha256']})
        identity={'training_seed':20261102,'record_id':rid,'source_id':item['record']['source_id'],'condition':'clean','sampling_seed':seed,'checkpoint_sha256':self.cp['sha256']}
        if not old:identity['role']='shared_initial'
        payload={'status':'completed','training_updates':0,'observation':identity,'PNG_artifacts':png,
                 'restoration':{'inference_inputs':['LRV','native text embedding']},'VAE_ledger':[{'route':'original_delegate'}],
                 'inference_inputs':['LRV','native text embedding','student checkpoint'],'teacher_or_HR_input_to_inferrer':False,
                 'source_metadata':{'source_frame_indices':item['source_indices'],'source_id':item['record']['source_id']}}
        save(directory/'metrics.json',payload)
        save(directory/'complete.json',{'protocol_sha256':regression.sha(run/'protocol.json'),'metrics_sha256':regression.sha(directory/'metrics.json'),'PNG_count':33})
        return identity|{'row_metrics_path':str(directory/'metrics.json'),'row_metrics_sha256':regression.sha(directory/'metrics.json')}

    def seal_states(self):
        for run,rows,old in ((self.old,self.oldrows,True),(self.new,self.newrows,False)):
            save(run/'observations.json',{'observations':rows,'protocol_sha256':regression.sha(run/'protocol.json')})
            save(run/'summary.json',{'restores':1152 if old else 1024,'protocol_sha256':regression.sha(run/'protocol.json')})
            save(run/'status.json',{'status':'completed','restores':1152 if old else 1024,'training_updates':0,
                 'gate_status':'passed_G2' if old else None,'protocol_sha256':regression.sha(run/'protocol.json'),'summary_sha256':regression.sha(run/'summary.json')})

    def reseal_new_protocol(self):
        save(self.new/'protocol.json',self.newp)
        for entry in self.newrows:
            if 'row_metrics_path' in entry:
                path=Path(entry['row_metrics_path']).parent/'complete.json';value=regression.read(path)
                value['protocol_sha256']=regression.sha(self.new/'protocol.json');save(path,value)
        self.seal_states()

    def resign_new_row(self, row_index=0):
        entry=self.newrows[row_index];path=Path(entry['row_metrics_path'])
        marker=regression.read(path.parent/'complete.json');marker['metrics_sha256']=regression.sha(path)
        save(path.parent/'complete.json',marker);entry['row_metrics_sha256']=regression.sha(path);self.seal_states()


class SharedInitialTests(unittest.TestCase):
    def test_exact_full_2046_PNG_pairs_pass_no_quality_or_GPU(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);report=regression.compare(f.old,f.new)
            self.assertEqual(report['PNG_pairs_compared'],2046);self.assertEqual(report['restore_pairs_compared'],62)
            self.assertEqual(report['reference']['excluded_replaced_new_records'],['replacement'])
            self.assertFalse(report['GPU_work_performed']);self.assertIsNone(report['quality_conclusion'])

    def test_incomplete_new_run_reads_only_status(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder,pngs=False);save(f.new/'status.json',{'status':'running','restores':63,'training_updates':0})
            original=regression.read;seen=[]
            def guarded(path):seen.append(Path(path));return original(path)
            with mock.patch.object(regression,'read',side_effect=guarded),self.assertRaisesRegex(regression.Mismatch,'new_validation_not_complete'):
                regression.compare(f.old,f.new)
            self.assertEqual(seen,[f.new/'status.json'])

    def test_reference_readiness_uses_no_prediction_or_tensor_files(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder,pngs=False);original=Path.open
            def guarded(path,*args,**kwargs):
                self.assertNotIn(path.suffix,('.pt','.npy','.png','.safetensors'));return original(path,*args,**kwargs)
            with mock.patch.object(Path,'open',guarded):report=regression.reference_metadata(f.old,f.newseal,f.newdata)
            self.assertEqual(report['retained_clips'],31);self.assertFalse(report['new_validation_results_read'])

    def test_changed_checkpoint_native_source_and_sampler_rejected_before_PNG(self):
        for bad in ('checkpoint','native','source','sampler'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder,pngs=False)
                if bad=='checkpoint':f.newp['roles'][0]['checkpoint']=file(f.root/'other.pt',b'changed')
                elif bad=='native':
                    f.newp['runtime']=copy.deepcopy(f.newp['runtime']);f.newp['runtime']['native']['transformer']=file(f.root/'other_native.py',b'changed')
                elif bad=='source':f.newp['sources']['infer.py']=file(f.root/'changed_infer.py',b'changed')
                else:f.newp['scope']['steps']=21
                f.reseal_new_protocol()
                with self.assertRaises(regression.Mismatch):regression.compare(f.old,f.new)

    def test_checkpoint_numeric_AST_change_rejected_but_unrelated_training_append_allowed(self):
        for numeric in (True, False):
            with self.subTest(numeric=numeric),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder,pngs=not numeric);path=Path(f.newp['sources']['checkpoints.py']['path'])
                code=path.read_text()
                path.write_text(code.replace('return 1','return 2',1) if numeric else code+'\ndef unrelated_training_only_head():\n    return 99\n')
                f.newp['sources']['checkpoints.py']=regression.binding(path);f.reseal_new_protocol()
                if numeric:
                    with self.assertRaisesRegex(regression.Mismatch,'numerical_AST_changed'):regression.compare(f.old,f.new)
                else:self.assertEqual(regression.compare(f.old,f.new)['PNG_pairs_compared'],2046)

    def test_resigned_executed_window_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);path=Path(f.newrows[0]['row_metrics_path']);value=regression.read(path)
            value['source_metadata']['source_frame_indices']=[99]+list(range(1,33));save(path,value);f.resign_new_row()
            with self.assertRaisesRegex(regression.Mismatch,'executed_source_window_changed'):regression.compare(f.old,f.new)

    def test_retained_NPY_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);(f.assetroot/'clip00/HR.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(regression.Mismatch,'retained_NPY_bytes'):regression.compare(f.old,f.new)

    def test_resigned_PNG_change_is_exact_failure_with_frame_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);path=f.new/'results/shared_initial/clip00/seed_42/png/000007.png';path.unlink();file(path,b'changed prediction')
            metric=path.parent.parent/'metrics.json';row=regression.read(metric);row['PNG_artifacts'][7]=regression.binding(path);save(metric,row);f.resign_new_row()
            with self.assertRaises(regression.Mismatch) as caught:regression.compare(f.old,f.new)
            self.assertEqual(caught.exception.stage,'prediction_PNG_bytes_changed')
            self.assertEqual(caught.exception.details['original_frame_index'],7)
            self.assertEqual(caught.exception.details['record_id'],'clip00')

    def test_teacher_leak_and_student_state_change_are_rejected(self):
        for bad in ('teacher','state'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder)
                if bad=='teacher':
                    path=Path(f.newrows[0]['row_metrics_path']);value=regression.read(path);value['teacher_or_HR_input_to_inferrer']=True;save(path,value);f.resign_new_row()
                else:
                    path=f.new/'roles/shared_initial/state_audit_001.json';value=regression.read(path);value['trainable_after']['same']='changed';save(path,value)
                with self.assertRaises(regression.Mismatch):regression.compare(f.old,f.new)

    def test_missing_or_reordered_PNG_and_marker_corruption_are_rejected(self):
        for bad in ('missing','order','marker'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder);metric=Path(f.newrows[0]['row_metrics_path'])
                if bad=='missing':(metric.parent/'png/000032.png').unlink()
                elif bad=='order':
                    value=regression.read(metric);value['PNG_artifacts'].reverse();save(metric,value);f.resign_new_row()
                else:
                    path=metric.parent/'complete.json';value=regression.read(path);value['metrics_sha256']='wrong';save(path,value)
                with self.assertRaises((regression.Mismatch,FileNotFoundError)):regression.compare(f.old,f.new)

    def test_duplicate_full_matrix_and_retained_window_relabel_refused(self):
        for bad in ('matrix','window'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder)
                if bad=='matrix':f.newrows[-1]=dict(f.newrows[-2]);f.seal_states()
                else:
                    value=regression.read(f.newseal);value['cohort'][0]['source_indices'][0]=99;save(f.newseal,value)
                    data=regression.read(f.newdata);data['cohort_seal']=regression.binding(f.newseal);save(f.newdata,data)
                    f.newp['cohort_seal']=regression.binding(f.newseal);f.newp['dataset']=regression.binding(f.newdata);f.reseal_new_protocol()
                with self.assertRaises(regression.Mismatch):regression.compare(f.old,f.new)


if __name__=='__main__':unittest.main()
