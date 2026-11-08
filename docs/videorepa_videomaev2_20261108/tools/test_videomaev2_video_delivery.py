"""Small synthetic CPU video/render tests; no real test RGB or model scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from vsr import data,evaluate as ev,infer
import videomaev2_video_delivery as v


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def source_for(record):
    index=int(record[-2:]);frames=torch.arange(33,dtype=torch.float32).view(1,1,33,1,1)
    hr=(torch.ones(1,3,33,128,224)*.5+frames*.001).contiguous()
    lr=(torch.ones(1,3,33,32,56)*.25+frames*.001).contiguous()
    return {'hr':hr,'lr':lr,'fps':12.5,'metadata':{'split':'test',
        'source_frame_indices':list(range(100+index,133+index))}}


class VideoDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.test_run=cls.root/'test_run';cls.test_run.mkdir()
        root=cls.root;run=cls.test_run;cls.events=[]
        roles=[]
        for family,arm,weight in [('baseline','baseline',0.),('videomaev2','videomaev2',.5)]:
            cp=root/(arm+'.pt');cp.write_bytes(arm.encode())
            roles.append(dict(label=arm+'_2048',arm=arm,family=family,setting='baseline' if not weight else 'low',
                              weight=weight,step=2048,update=2048,checkpoint=v.binding(cp)))
        cls.roles={r['family']:r for r in roles}
        cohort=[{'record':{'id':f'clip{i:02d}','source_id':f'source{i:02d}','split':'test'},
                 'source_indices':list(range(100+i,133+i)),'cached_header':{'fps':12.5}} for i in range(32)]
        seal=root/'test_seal.json';write(seal,{'cohort':cohort})
        dataset=root/'dataset.json';write(dataset,{'synthetic_fixture':True})
        study=root/'study.json';write(study,{'synthetic_complete_study':True})
        selection=root/'selection.json';write(selection,{'final_roles':roles})
        assets=[]
        for item in cohort:
            row={'record_id':item['record']['id']}
            for key in ('HR','clean'):
                p=root/'dummy_bound_assets'/(row['record_id']+'_'+key+'.npy');p.parent.mkdir(exist_ok=True)
                p.write_bytes(b'Synthetic metadata binding; arrays supplied by fake CPU reader.')
                row[key]=v.binding(p)
            assets.append(row)
        protocol={'schema':ev.VMAE_EVALUATION_SCHEMA,'phase':'test','scope':{'total_restores':128},'roles':roles,
            'cohort':cohort,'fixed_video_clip_ids':['clip00','clip01','clip02'],'sources':ev._hr_sources(),'runtime':{},
            'study_file':v.binding(study),'finals_receipt':v.binding(selection),'validation_final_lock':{'final_roles':roles},
            'config':{'dataset':str(dataset),'seal':str(seal)},'dataset':v.binding(dataset),'cohort_seal':v.binding(seal),
            'clean_assets':assets,'data_independence':{'historically_pristine_test':False,'synthetic_fixture':True}}
        write(run/'protocol.json',protocol);psha=v.binding(run/'protocol.json')['sha256'];observations=[]
        for role_index,role in enumerate(roles):
            for item in cohort:
                for sample in (42,43):
                    directory=ev._hr_directory(run,role,item,sample);path=directory/'metrics.json'
                    if item['record']['id'] in protocol['fixed_video_clip_ids']:
                        pngs=[]
                        for frame in range(33):
                            p=directory/'png'/f'{frame:06d}.png';p.parent.mkdir(parents=True,exist_ok=True)
                            color=(50+role_index*10+frame,100+sample-42,150)
                            Image.new('RGB',(224,128),color).save(p);pngs.append(v.binding(p))
                        write(path,{'status':'completed','training_updates':0,'PNG_artifacts':pngs,
                            'source_metadata':{'split':'test','source_frame_indices':item['source_indices']}})
                        write(directory/'complete.json',{'protocol_sha256':psha,'metrics_sha256':v.binding(path)['sha256'],'PNG_count':33})
                        row_hash=v.binding(path)['sha256']
                    else:row_hash='not_read_by_display_fixture'
                    observations.append(dict(role=role['label'],record_id=item['record']['id'],sampling_seed=sample,
                        row_metrics_path=str(path),row_metrics_sha256=row_hash))
        write(run/'observations.json',{'observations':observations});write(run/'status.json',
            {'status':'completed','restores':128,'training_updates':0,'protocol_sha256':psha})
        cls.protocol=protocol
        class Reader:
            def __init__(self,*args,**kwargs):cls.events.append('reader')
            def load(self,record,**kwargs):return source_for(record)
            def close(self):pass
        cls.Reader=Reader
        def verify(run):
            cls.events.append('verify')
            return {'status':'verified_complete_clean_evaluation','restores':128,'training_updates':0,
                'protocol_sha256':v.binding(Path(run)/'protocol.json')['sha256'],
                'prediction_PNGs_verified':4224,'prediction_MP4s_verified':128,
                'both_arms_retained':True,'training_only_checkpoint_roles_verified':True,'synthetic_fixture':True}
        cls.output=root/'outputs/video_compare_fixture'
        with patch.object(v,'WORKSPACE',root),patch.object(ev,'vmae_verify',side_effect=verify),\
             patch.object(ev,'verify_imported_runtime',side_effect=lambda _:cls.events.append('runtime')),\
             patch.object(data,'FixedCleanReader',Reader),patch.object(infer,'restore_video',side_effect=AssertionError('No inference allowed')),\
             patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No GPU allowed')):
            cls.result=v.deliver(ev,test_run=run,output_dir=cls.output)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_six_full33_frame_MP4_and_fixed_contact_positions_preserve_pixel_panels(self):
        self.assertEqual(self.result['MP4_count'],6);self.assertEqual(self.events[:3],['verify','runtime','reader'])
        manifest=v.read(self.output/'manifest.json');self.assertEqual(len(manifest['cases']),6)
        self.assertEqual({x['sampling_seed'] for x in manifest['cases']},{42,43})
        for case in manifest['cases']:
            self.assertEqual(case['MP4_audit']['fully_decoded_frames'],33)
            self.assertEqual((case['MP4_audit']['height'],case['MP4_audit']['width'],case['original_FPS']),(172,896,12.5))
            self.assertEqual(case['contact_frame_indices'],[0,8,16,24,32])
            self.assertEqual(case['contact_relative_seconds'],[0.,.64,1.28,1.92,2.56])
            with Image.open(case['contact_sheet_PNG']['path']) as image:
                self.assertEqual(image.size,(896,860))
                self.assertEqual(image.getpixel((224*2+10,24+10)),(50,100+case['sampling_seed']-42,150))
        self.assertFalse(torch.cuda.is_initialized());self.assertEqual(manifest['new_inference_calls'],0)

    def test_output_only_original_outputs_no_overwrite_or_symlink_escape(self):
        with patch.object(v,'WORKSPACE',self.root):
            with self.assertRaisesRegex(ValueError,'Preserve previous'):v._output_directory(self.output)
            with self.assertRaisesRegex(ValueError,'original workspace outputs'):v._output_directory(self.root/'other/video_compare_bad')
            outside=self.root/'outside';outside.mkdir();link=self.root/'outputs/link';link.symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'original workspace outputs'):v._output_directory(link/'video_compare_bad')

    def test_running_or_changed_predeclared_clips_rejected_before_source_RGB(self):
        for running in (True,False):
            with self.subTest(running=running),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);run=root/'bad';run.mkdir();protocol=copy.deepcopy(self.protocol)
                if not running:protocol['fixed_video_clip_ids']=['clip00','clip01','clip03']
                write(run/'protocol.json',protocol);write(run/'status.json',dict(status='running' if running else 'completed',
                    restores=128,training_updates=0,protocol_sha256=v.binding(run/'protocol.json')['sha256']))
                with patch.object(v,'WORKSPACE',root),patch.object(ev,'vmae_verify',return_value={'status':'verified_complete_clean_evaluation','restores':128,'training_updates':0,'protocol_sha256':v.binding(run/'protocol.json')['sha256'],'prediction_PNGs_verified':4224,'prediction_MP4s_verified':128,'both_arms_retained':True,'training_only_checkpoint_roles_verified':True}) as verifier,\
                     patch.object(data,'FixedCleanReader') as reader:
                    with self.assertRaises(ValueError):v.deliver(ev,test_run=run,output_dir=root/'outputs/video_compare_bad')
                    reader.assert_not_called()
                    if running:verifier.assert_not_called()
                    self.assertFalse((root/'outputs/video_compare_bad').exists())

    def test_changed_output_hash_is_rejected(self):
        manifest=v.read(self.output/'manifest.json');path=Path(manifest['cases'][0]['contact_sheet_PNG']['path']);original=path.read_bytes()
        try:
            path.write_bytes(original+b'tampered')
            with self.assertRaisesRegex(ValueError,'Bound display input changed'):v.verify_delivery(self.output)
        finally:path.write_bytes(original)

    def test_prediction_PNG_reordering_or_missing_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);role=self.roles['baseline'];item=self.protocol['cohort'][0]
            directory=ev._hr_directory(run,role,item,42);pngs=[]
            for frame in range(33):
                path=directory/'png'/f'{frame:06d}.png';path.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(224,128),(frame,1,2)).save(path);pngs.append(v.binding(path))
            pngs[0],pngs[1]=pngs[1],pngs[0]
            path=directory/'metrics.json';write(path,{'status':'completed','training_updates':0,'PNG_artifacts':pngs,
                'source_metadata':{'split':'test','source_frame_indices':item['source_indices']}})
            psha='f'*64;write(directory/'complete.json',{'protocol_sha256':psha,'metrics_sha256':v.binding(path)['sha256'],'PNG_count':33})
            index={(role['label'],item['record']['id'],42):{'row_metrics_path':str(path),'row_metrics_sha256':v.binding(path)['sha256']}}
            with self.assertRaisesRegex(ValueError,'frame chronology'):v._prediction_frames(ev,run,psha,role,item,42,index)
            pngs[0],pngs[1]=pngs[1],pngs[0]
            write(path,{'status':'completed','training_updates':0,'PNG_artifacts':pngs,
                'source_metadata':{'split':'test','source_frame_indices':item['source_indices']}})
            write(directory/'complete.json',{'protocol_sha256':psha,'metrics_sha256':v.binding(path)['sha256'],'PNG_count':33})
            index[role['label'],item['record']['id'],42]['row_metrics_sha256']=v.binding(path)['sha256']
            Path(pngs[2]['path']).unlink()
            with self.assertRaises(FileNotFoundError):v._prediction_frames(ev,run,psha,role,item,42,index)

    def test_integer_preview_is_nearest_pixel_enlargement_only(self):
        source=source_for('clip00');predictions=[np.full((33,128,224,3),20+i,np.uint8) for i in range(2)]
        frames=v._panels(source,predictions,clip_id='clip00',source_indices=list(range(100,133)),sample=42,roles=self.roles,scale=2)
        self.assertEqual(frames.shape,(33,344,1792,3))
        pixel=frames[0,48+20,224*2*2+20]
        self.assertTrue(np.array_equal(pixel,np.array([20,20,20],np.uint8)))
        self.assertTrue(np.array_equal(frames[0,48+20:48+22,224*2*2+20:224*2*2+22],np.full((2,2,3),20,np.uint8)))


if __name__=='__main__':unittest.main(verbosity=2)
