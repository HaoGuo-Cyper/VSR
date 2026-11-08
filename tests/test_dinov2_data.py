"""Synthetic CPU feature/cache/clean-reader contracts; no DINO/Wan weights."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from vsr import data,teacher


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def binding(path):
    path=Path(path);return {'path':str(path),'bytes':path.stat().st_size,'sha256':data.file_hash(path)}


class SyntheticTeacher:
    """Injected index features for cache fixtures, never real DINO evidence."""
    def __init__(self):
        self.binding={'architecture':'dinov2_vitb14','source_revision':teacher.DINOV2_REVISION,
            'weights':{'path':'/synthetic_only','bytes':0,'sha256':teacher.DINOV2_WEIGHT_SHA256},
            'source_files':{'synthetic_only':'a'*64},'feature_spec':teacher.dinov2_feature_spec(),
            'precision':'FP32_no_autocast','runtime':{'synthetic_only':True},'frame_batch_size':8}
        self.prepared=False;self.closed=False
    def prepare(self,device):self.prepared=True;return self
    def encode_hr(self,hr):
        assert self.prepared
        return teacher.group_dinov2_time(torch.arange(33).float().view(33,1,1).expand(33,112,768))
    def close(self):self.closed=True


def paired_source(root):
    root=Path(root);root.mkdir()
    ids=['train_source_clip','val_source_clip']
    write(root/'config.json',{'data':{'num_frames':33,'temporal_stride':1,'height':128,'width':224,'scale':4}})
    write(root/'selection.json',{'training_ids':ids[:1],'validation_ids':ids[1:],'test_accessed':False})
    write(root/'correspondence.json',{'latent_grid':[9,16,28],'token_grid':[9,8,14]})
    for i,record in enumerate(ids):
        meta={'record_id':record,'source_id':f'source_{i}','source_sha256':str(i+1)*64,
              'split':'train' if i==0 else 'val','source_indices':list(range(i*40,i*40+33)),
              'crop_tlhw':[0,0,128,224],'source_hw':[128,224],'resize_hw':[128,224],
              'temporal_stride':1,'original_fps':30.,'output_fps':30.}
        meta['binding']={'data_cache_key':data._digest(meta)}
        write(root/f'clip_{i}/metadata.json',meta)
        torch.save({'hr':torch.full((1,3,33,128,224),.25+i*.25)},root/f'clip_{i}/paired_rgb.pt')
    hashes={str(p.relative_to(root)):data.file_hash(p) for p in root.rglob('*') if p.is_file()}
    write(root/'cache_asset_seal.json',{'artifact_sha256':hashes})
    return root


def resign_cache(root):
    manifest=data.read_json(root/'manifest.json')
    for r in manifest['records']:
        for key in ('metadata','features'):r[key]['sha256']=data.file_hash(root/r[key]['path'])
    write(root/'manifest.json',manifest)
    write(root/'seal.json',{'schema':'vsr_dinov2_cache_seal_v1',
        'manifest_sha256':data.file_hash(root/'manifest.json'),'records':manifest['records']})
    if (root/'completion_receipt.json').exists():
        receipt=data.read_json(root/'completion_receipt.json');receipt['seal_sha256']=data.file_hash(root/'seal.json')
        write(root/'completion_receipt.json',receipt)


class DINOFeatureTests(unittest.TestCase):
    def test_all33_time_groups_and_token_order_no_early_channel_normalization(self):
        values=torch.arange(33).float().view(33,1,1).expand(33,112,768).clone()
        values+=torch.arange(112).view(1,112,1)*100
        grouped=teacher.group_dinov2_time(values).reshape(9,112,768)
        expected=[0.,*[i+1.5 for i in range(1,33,4)]]
        for index,mean in enumerate(expected):
            torch.testing.assert_close(grouped[index,:,0],mean+torch.arange(112)*100,rtol=0,atol=0)
        self.assertGreater(float(grouped[-1].norm(dim=-1).mean()),1.)
        self.assertFalse(grouped.requires_grad)
        with self.assertRaises(ValueError):teacher.group_dinov2_time(values[:9])

    def test_full_crop_resize_and_ImageNet_normalization(self):
        hr=torch.tensor([.2,.4,.6]).view(1,3,1,1,1).expand(1,3,33,128,224).clone()
        out=teacher.dinov2_preprocess_hr(hr)
        self.assertEqual(tuple(out.shape),(33,3,112,196))
        expected=(torch.tensor([.2,.4,.6])-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
        torch.testing.assert_close(out,expected.view(1,3,1,1).expand_as(out),rtol=1e-6,atol=1e-6)
        self.assertFalse(teacher.dinov2_feature_spec()['crop']);self.assertFalse(teacher.dinov2_feature_spec()['padding'])
        with self.assertRaises(ValueError):teacher.dinov2_preprocess_hr(hr[:,:,:9])
        with self.assertRaises(ValueError):teacher.dinov2_preprocess_hr(hr.requires_grad_())

    def test_official_namespace_paths_are_bound_and_shadow_paths_fail(self):
        import types,sys
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'source';directory=root/'dinov2/hub/xray_dino';directory.mkdir(parents=True)
            path=directory/'backbones.py';path.write_text('bound = True\n')
            b={'source_dir':str(root),'source_files':{'dinov2/hub/xray_dino/backbones.py':data.file_hash(path)}}
            namespace=types.ModuleType('dinov2.hub.xray_dino');namespace.__path__=[str(directory)]
            child=types.ModuleType('dinov2.hub.xray_dino.backbones');child.__file__=str(path)
            with patch.dict(sys.modules,{'dinov2.hub.xray_dino':namespace,'dinov2.hub.xray_dino.backbones':child}):
                teacher._verify_dinov2_imported_sources(b)
                namespace.__path__.append('/tmp/untrusted_dinov2')
                with self.assertRaises(ValueError):teacher._verify_dinov2_imported_sources(b)
                namespace.__path__=[str(directory)];path.write_text('bound = False\n')
                with self.assertRaises(ValueError):teacher._verify_dinov2_imported_sources(b)

    def test_cache_constructor_is_metadata_only_and_train_val_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            source=paired_source(Path(folder)/'paired');target=Path(folder)/'dino';t=SyntheticTeacher()
            result=data.prepare_dinov2_cache(source,target,teacher=t)
            self.assertTrue(t.closed);self.assertEqual(result['teacher_frames'],66)
            receipt=data.read_json(target/'completion_receipt.json');timings=receipt['timings']
            self.assertEqual(result['completion_receipt']['sha256'],data.file_hash(target/'completion_receipt.json'))
            self.assertEqual(receipt['feature_tensor_bytes'],2*1008*768*4)
            self.assertIsNone(receipt['GPU_memory']);self.assertIsNone(timings['pure_GPU_encoder_seconds'])
            self.assertGreaterEqual(timings['teacher_feature_extraction_seconds'],0.)
            self.assertGreaterEqual(timings['other_io_binding_seconds'],0.)
            self.assertAlmostEqual(timings['total_cache_build_seconds'],timings['teacher_feature_extraction_seconds']+timings['other_io_binding_seconds'])
            self.assertEqual(receipt['new_VAE_feature_extraction_seconds_this_round'],0.)
            with patch('torch.load',side_effect=AssertionError('constructor must not load features')):
                reader=data.PreparedDINOData(target,source,expected_teacher_binding=t.binding,expected_seal_sha256=result['seal_sha256'])
            self.assertEqual(len(reader),1);self.assertEqual(reader.validation_ids,['val_source_clip'])
            features=reader[0];self.assertEqual(tuple(features.shape),(1,1008,768))
            features.fill_(-99);self.assertNotEqual(float(reader[0][0,0,0]),-99.)
            self.assertEqual(reader.record(1)['binding']['split'],'val')
            self.assertEqual(tuple(reader.load(1).shape),(1,1008,768))
            with self.assertRaises(IndexError):reader[1]
            reader.close()
        self.assertFalse(torch.cuda.is_initialized())

    def test_cache_rejects_changed_clip_window_crop_source_or_teacher_binding(self):
        for field in ('source_indices','crop_tlhw','source_id','paired_RGB_sha256','HR_f32_sha256','header_digest'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                source=paired_source(Path(folder)/'paired');target=Path(folder)/'dino';t=SyntheticTeacher()
                result=data.prepare_dinov2_cache(source,target,teacher=t)
                path=target/'clip_0/metadata.json';meta=data.read_json(path)
                meta['binding'][field]='wrong';write(path,meta)
                # The original seal must catch any edited artifact immediately.
                with self.assertRaises(ValueError):data.PreparedDINOData(target,source,expected_seal_sha256=result['seal_sha256'])
                if field!='HR_f32_sha256':
                    resign_cache(target)
                    with self.assertRaises(ValueError):data.PreparedDINOData(target,source)

    def test_features_missing_or_wrong_binding_are_rejected_lazily(self):
        with tempfile.TemporaryDirectory() as folder:
            source=paired_source(Path(folder)/'paired');target=Path(folder)/'dino'
            data.prepare_dinov2_cache(source,target,teacher=SyntheticTeacher())
            path=target/'clip_0/features.pt';payload=torch.load(path,weights_only=True)
            payload['binding']['source_indices']=[0]*33;torch.save(payload,path);resign_cache(target)
            reader=data.PreparedDINOData(target,source)
            with self.assertRaises(ValueError):reader[0]
            path.unlink()
            with self.assertRaises(FileNotFoundError):reader[0]

    def test_oldstride4_partial_and_wrong_teacher_caches_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            source=paired_source(Path(folder)/'paired');target=Path(folder)/'dino';t=SyntheticTeacher()
            data.prepare_dinov2_cache(source,target,teacher=t)
            wrong=copy.deepcopy(t.binding);wrong['weights']['sha256']='f'*64
            with self.assertRaises(ValueError):data.PreparedDINOData(target,source,expected_teacher_binding=wrong)
            m=data.read_json(target/'manifest.json');m['records'].pop();write(target/'manifest.json',m);resign_cache(target)
            with self.assertRaises(ValueError):data.PreparedDINOData(target,source)
        with tempfile.TemporaryDirectory() as folder:
            source=paired_source(Path(folder)/'paired');config=data.read_json(source/'config.json');config['data']['temporal_stride']=4;write(source/'config.json',config)
            seal=data.read_json(source/'cache_asset_seal.json');seal['artifact_sha256']['config.json']=data.file_hash(source/'config.json');write(source/'cache_asset_seal.json',seal)
            t=SyntheticTeacher()
            with self.assertRaises(ValueError):data.prepare_dinov2_cache(source,Path(folder)/'dino',teacher=t)
            self.assertFalse(t.prepared)

    def test_CUDA_builder_and_CLI_cannot_run_without_explicit_admission_and_execution(self):
        with patch('vsr.checkpoints.gpu_lease',side_effect=AssertionError('no GPU lease expected')):
            with self.assertRaises(ValueError):data.prepare_dinov2_cache('/missing','/missing',teacher=SyntheticTeacher(),device='cuda:0')
            with patch.object(teacher,'FrozenDINOv2',side_effect=AssertionError('no model metadata needed')):
                with self.assertRaises(ValueError):data.main(['prepare-dino-cache','--training-cache-root','/missing',
                    '--output','/missing','--teacher-source','/missing','--teacher-weights','/missing'])
        with self.assertRaises(SystemExit) as e:data.main(['prepare-dino-cache','--help'])
        self.assertEqual(e.exception.code,0);self.assertFalse(torch.cuda.is_initialized())

    def test_CUDA_flags_precede_initialization_and_peak_reset_inside_lease(self):
        from contextlib import contextmanager
        calls=[]
        @contextmanager
        def lease(*args):
            calls.append('lease');yield {'fresh':'GPU_admission'};calls.append('release')
        enabled=('deterministic_algorithms','cudnn_deterministic','sdp_math')
        disabled=('deterministic_warn_only','cudnn_benchmark','cudnn_allow_tf32','matmul_allow_tf32',
                  'matmul_allow_fp16_reduced_precision_reduction','matmul_allow_bf16_reduced_precision_reduction',
                  'sdp_flash','sdp_memory_efficient','sdp_cudnn','sdp_math_allow_fp16_bf16_reduction')
        runtime={'mode':'strict','flags':dict.fromkeys(enabled,True)|dict.fromkeys(disabled,False)|{'cuda_initialized':False}}
        with tempfile.TemporaryDirectory() as folder:
            source=paired_source(Path(folder)/'paired')
            with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}),patch('vsr.checkpoints.gpu_lease',lease), \
                 patch('torch.cuda.is_initialized',return_value=False), \
                 patch('vsr.model.configure_reproducibility',side_effect=lambda cfg:(calls.append('flags') or runtime)), \
                 patch('torch.cuda.init',side_effect=lambda:calls.append('init')), \
                 patch('torch.cuda.reset_peak_memory_stats',side_effect=lambda device:calls.append('reset')), \
                 patch.object(data,'_prepare_dinov2_cache',side_effect=lambda *a:(calls.append('build') or {'status':'mocked'})):
                result=data.prepare_dinov2_cache(source,Path(folder)/'dino',teacher=SyntheticTeacher(),
                                                device='cuda:0',gpu_lock='/unused',gpu_admission={})
        self.assertEqual(result,{'status':'mocked'})
        self.assertEqual(calls,['lease','flags','init','reset','build','release'])

    def test_failed_extraction_preserves_primary_error_and_never_publishes_complete_cache(self):
        class BrokenTeacher(SyntheticTeacher):
            def encode_hr(self,hr):raise RuntimeError('primary extraction')
            def close(self):raise OSError('secondary cleanup')
        with tempfile.TemporaryDirectory() as folder:
            source=paired_source(Path(folder)/'paired');target=Path(folder)/'dino'
            with self.assertRaisesRegex(RuntimeError,'primary extraction') as error:
                data.prepare_dinov2_cache(source,target,teacher=BrokenTeacher())
            self.assertIn('secondary cleanup',' '.join(error.exception.__notes__))
            self.assertFalse((target/'seal.json').exists());self.assertFalse((target/'manifest.json').exists())
            self.assertEqual(data.read_json(target/'failure.json')['completed_records'],0)


def clean_fixture(root,split):
    root=Path(root);root.mkdir()
    cohort=[];assets=[]
    for i in range(32):
        cohort.append({'record':{'id':f'clip{i}','source_id':f'source{i}','split':split,'sha256':f'{i:064x}'},
                       'source_indices':list(range(33)),'cached_header':{'fps':30.}})
    seal={'schema':'vsr_fixed_clean_cohort_v1','split':split,'project_root':str(root),
          'config':{'frames':33},'cohort':cohort}
    seal['seal_digest']=data._digest(seal);write(root/'cohort.json',seal)
    write(root/'selection.json',{'status':'settings_selected'})
    np.save(root/'HR.npy',np.full((1,3,33,128,224),.5,dtype=np.float32))
    np.save(root/'LR.npy',np.full((1,3,33,32,56),.5,dtype=np.float32))
    for i in range(32):
        directory=root/f'clip{i}';directory.mkdir()
        specs={}
        for key,name in (('HR','HR.npy'),('clean','LR.npy')):
            path=directory/name;os.link(root/name,path);specs[key]={'path':str(path),'sha256':data.file_hash(path)}
        metadata={'record_id':f'clip{i}','source_id':f'source{i}','split':split,'source_indices':list(range(33)),
                  'source_sha256':f'{i:064x}','crop_tlhw':[0,0,128,224],'output_fps':30.}
        assets.append({'record_id':f'clip{i}','source_id':f'source{i}','split':split,'metadata':metadata,**specs})
    dataset={'schema':'vsr_fixed_clean_dataset_v1','status':'completed_CPU_clean_assets' if split=='test' else 'completed_CPU_assets_no_predictions',
             'split':split,'config':seal['config'],'cohort_seal':binding(root/'cohort.json'),
             'cohort_seal_digest':seal['seal_digest'],'selection_receipt':binding(root/'selection.json'),
             'materialization':{'selection_frozen_before_decode':True},'assets':assets}
    write(root/'dataset.json',dataset)
    return root/'dataset.json',root/'cohort.json'


class FixedCleanTests(unittest.TestCase):
    def test_test_assets_keep_actual_split_full33_and_metadata_only_describe(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset,seal=clean_fixture(Path(folder)/'clean','test')
            with patch('numpy.load',side_effect=AssertionError('describe must be metadata-only')):
                reader=data.FixedCleanReader(dataset,expected_split='test',cohort_seal=seal)
                info=reader.describe()
            self.assertEqual(info['split'],'test');self.assertEqual(len(info['cohort']),32)
            self.assertEqual(info['seal_binding'],binding(seal))
            result=reader.load('clip0')
            self.assertEqual(result['metadata']['split'],'test');self.assertEqual(result['metadata']['source_frame_indices'],list(range(33)))
            self.assertEqual(tuple(result['hr'].shape),(1,3,33,128,224));self.assertEqual(tuple(result['lr'].shape),(1,3,33,32,56))
            with self.assertRaises(ValueError):reader.load('clip0','noise')
            with self.assertRaises(ValueError):reader.load('clip0',start=1)
            with self.assertRaises(ValueError):data.FixedCleanReader(dataset,expected_split='val',cohort_seal=seal)

    def test_val_uses_original_seal_validation_without_relabeling(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset,seal=clean_fixture(Path(folder)/'clean','val')
            payload=data.read_json(dataset);payload.pop('schema');write(dataset,payload)
            with patch.object(data,'validate_seal') as check:
                reader=data.FixedCleanReader(dataset,expected_split='val',cohort_seal=data.read_json(seal))
            check.assert_called_once();self.assertIsNone(reader.describe()['seal_binding'])
            self.assertEqual(reader.load('clip0')['metadata']['split'],'val')

    def test_generic_prospective_val_assets_use_actual_val_without_test_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset,seal=clean_fixture(Path(folder)/'clean','val')
            payload=data.read_json(dataset);payload['status']='completed_CPU_clean_assets'
            payload['materialization']={'validation_assets_before_training':True};payload.pop('selection_receipt')
            write(dataset,payload)
            with patch.object(data,'validate_seal',side_effect=AssertionError('new val is not the historical corruption cohort')):
                reader=data.FixedCleanReader(dataset,expected_split='val',cohort_seal=seal)
            self.assertEqual(reader.load('clip0')['metadata']['split'],'val')
            payload['materialization']['validation_assets_before_training']=False;write(dataset,payload)
            with self.assertRaises(ValueError):data.FixedCleanReader(dataset,expected_split='val',cohort_seal=seal)

    def test_original_val_source_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset,seal=clean_fixture(Path(folder)/'clean','val')
            payload=data.read_json(dataset);payload.pop('schema')
            for row in payload['assets']:
                row['source_metadata']=row.pop('metadata')
            write(dataset,payload)
            with patch.object(data,'validate_seal'):
                reader=data.FixedCleanReader(dataset,expected_split='val',cohort_seal=seal)
            metadata=reader.load('clip0')['metadata']
            self.assertEqual(metadata['crop_tlhw'],[0,0,128,224])
            self.assertEqual(metadata['source_sha256'],f'{0:064x}')

    def test_test_requires_unchanged_selection_receipt_and_no_source_or_split_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset,seal=clean_fixture(Path(folder)/'clean','test');original=data.read_json(dataset)
            for field in ('receipt','before_decode','split','source'):
                d=copy.deepcopy(original)
                if field=='receipt':d['selection_receipt']['sha256']='f'*64
                elif field=='before_decode':d['materialization']['selection_frozen_before_decode']=False
                elif field=='split':d['assets'][0]['metadata']['split']='val'
                else:d['assets'][0]['source_id']='different'
                write(dataset,d)
                with self.subTest(field=field),self.assertRaises(ValueError):
                    data.FixedCleanReader(dataset,expected_split='test',cohort_seal=seal)


if __name__=='__main__':unittest.main()
