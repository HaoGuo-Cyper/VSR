"""CPU data/VAE equivalence and source-bound cache regression fixtures."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from contextlib import contextmanager
import unittest
from unittest.mock import patch

import torch

from vsr import data, vae


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        self.config=SimpleNamespace(latents_mean=[.1,.2,.3],latents_std=[.5,1.,2.])
        self.events=[]
    def encode(self,x):
        self.events.append('encode')
        def mode(): self.events.append('mode'); return x.clone()
        def sample(): raise AssertionError('posterior sampling is forbidden')
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=mode,sample=sample))
    def decode(self,z): self.events.append('decode'); return SimpleNamespace(sample=z.clone())
    def clear_cache(self):self.events.append('clear')


class ProtocolFakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        self.config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        self.events=[]
    def encode(self,x):
        self.events.append('encode')
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:torch.zeros(1,16,9,16,28)))
    def decode(self,x):
        self.events.append('decode')
        return SimpleNamespace(sample=torch.zeros(1,3,33,128,224))
    def clear_cache(self):self.events.append('clear')


class PolicyEncoder(torch.nn.Module):
    def forward(self, x, *, feat_cache, feat_idx):
        if feat_cache is not None: feat_cache[0]=x
        feat_idx[0]+=1
        return x


class PolicyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        self.encoder=PolicyEncoder()
        self.config=SimpleNamespace(patch_size=None,is_residual=False)
        self.use_tiling=self.use_slicing=False
        self._enc_feat_map=None;self._feat_map=None
        self.encode_inputs=[]
    def encode(self, x, **kwargs):
        self.encode_inputs.append((x,kwargs))
        self._enc_feat_map=[None];self._enc_conv_idx=[0]
        value=self.encoder(x,feat_cache=self._enc_feat_map,feat_idx=self._enc_conv_idx)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value))
    def clear_cache(self):
        self._enc_feat_map=None;self._feat_map=None


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,allow_nan=False))


def cache_fixture(root, *, train=2,val=1):
    root.mkdir()
    config={'seed':20261102,'data':{'num_frames':33,'height':128,'width':224,'scale':4,'temporal_stride':1},
            'model':{'repo_id':'Wan-AI/Wan2.1-T2V-1.3B-Diffusers','revision':'0fad780a534b6463e45facd96134c9f345acfa5b'}}
    write(root/'config.json',config)
    ids=[f'record{i}' for i in range(train+val)]
    write(root/'selection.json',{'training_ids':ids[:train],'validation_ids':ids[train:],'manifest_sha256':'manifest','test_accessed':False})
    write(root/'correspondence.json',{'latent_grid':[9,16,28],'token_grid':[9,8,14]})
    torch.save({data.CAPTION_PLACEHOLDER:torch.zeros(1,512,4096,dtype=torch.bfloat16)},root/'fresh_text_embeddings.pt')
    for i,record in enumerate(ids):
        metadata={'record_id':record,'source_id':f'source{i}','split':'train' if i<train else 'val','temporal_stride':1,
                  'source_indices':list(range(33)),'crop_tlhw':[0,0,128,224],'degradation':{'scale':4},
                  'original_fps':24.,'output_fps':24.,'caption':data.CAPTION_PLACEHOLDER}
        binding={'data_cache_key':data._digest(metadata),'manifest_sha256':'manifest'}
        write(root/f'clip_{i}/metadata.json',{**metadata,'binding':binding})
        pair={k:torch.full((1,16,9,16,28),i*.01+(0.1 if k=='clean' else 0.2)) for k in ['clean','low']}
        torch.save({**pair,'binding':binding},root/f'clip_{i}/fresh_latents.pt')
    names=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]
    write(root/'cache_asset_seal.json',{'contains_task_initialization':False,'artifact_sha256':{n:data.file_hash(root/n) for n in names}})
    return ids


def reader_fixture(root):
    import numpy as np
    root.mkdir()
    source=root/'implementation.py';source.write_text('source fixture\n')
    config=root/'config.json';write(config,{'fixture':True})
    manifest=root/'manifest.json';write(manifest,{'fixture':True})
    def store(name,shape):
        array=np.full(shape,.25,dtype=np.float32);p=root/name
        with p.open('wb') as f:np.save(f,array,allow_pickle=False)
        return {'path':str(p),'sha256':data.file_hash(p),'rgb_f32_sha256':hashlib.sha256(array.tobytes()).hexdigest()}
    lr=store('lr.npy',(1,3,33,32,56));hr=store('hr.npy',(1,3,33,128,224))
    conditions=['blur','noise','codec_allintra','drop_repeat','drop_black']
    cfg={'conditions':conditions}
    mask=[8<=i<16 for i in range(33)]
    cohort=[];assets=[]
    for i in range(32):
        record=f'record{i}'
        cohort.append({'record':{'id':record,'source_id':f'source{i}','split':'val'},'source_indices':list(range(33)),
                       'mask':mask,'noise_seeds_by_masked_frame':{str(j):j for j in range(8,16)}})
        metadata={'record_id':record,'source_id':f'source{i}','split':'val','source_indices':list(range(33)),'output_fps':24.}
        assets.append({'record_id':record,'source_id':f'source{i}','mask':mask,'base_pair_cache_key':'cache',
                       'source_metadata':metadata,'HR':hr,'clean':lr,'conditions':{c:{'asset':lr} for c in conditions}})
    seal={'project_root':str(root),'config':cfg,'config_path':str(config),'config_sha256':data.file_hash(config),
          'manifest':str(manifest),'manifest_file_sha256':data.file_hash(manifest),'manifest_digest':'manifest',
          'implementation_bindings':{str(source):data.file_hash(source)},'cohort':cohort,
          'codec':{},'CPU_runtime_packages':{}}
    seal['seal_digest']=data.digest(seal)
    dataset=root/'dataset.json'
    write(dataset,{'status':'completed_CPU_assets_no_predictions','cohort_seal_digest':seal['seal_digest'],'config':cfg,'assets':assets})
    return dataset,seal


class DataVaeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_atomic_json_new_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';data.write_json_new(p,{'value':1})
            with self.assertRaises(FileExistsError):data.write_json_new(p,{'value':2})
            self.assertEqual(data.read_json(p),{'value':1})
            with self.assertRaises(ValueError):data.write_json_new(Path(t)/'nan.json',{'x':float('nan')})
            self.assertFalse((Path(t)/'nan.json').exists())

    def test_bicubic_matches_archived_numerics(self):
        old=Path(__file__).resolve().parents[1]/'legacy/pre_refactor_20261106/vsr/data.py'
        spec=importlib.util.spec_from_file_location('old_data_fixture',old);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        x=torch.rand(1,3,5,32,48,generator=torch.Generator().manual_seed(4))
        actual=data.degrade_x4(x);expected=module.degrade_x4(x)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))

    def test_masks_and_corruption_are_local_and_repeatable(self):
        mask=data.sample_span_mask(2,9,rho=.25,span=2,generator=torch.Generator().manual_seed(8))
        self.assertFalse(mask.mask[:,0].any());self.assertEqual(mask.mask.sum().item(),4)
        clean=torch.arange(2*3*9*2*2,dtype=torch.float32).reshape(2,3,9,2,2)
        result=data.corrupt_latent_condition(clean,mask.mask,generator=torch.Generator().manual_seed(9))
        mask5=mask.mask[:,None,:,None,None].expand_as(clean)
        self.assertTrue(torch.equal(result[~mask5],clean[~mask5]));self.assertFalse(torch.equal(result[mask5],clean[mask5]))
        g=torch.Generator().manual_seed(9);before=g.get_state().clone()
        self.assertTrue(torch.equal(clean,data.corrupt_latent_condition(clean,torch.zeros(2,9,dtype=torch.bool),generator=g)))
        self.assertTrue(torch.equal(before,g.get_state()))

    def test_prepared_cache_reads_original_schema_without_mutating_assets(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'cache';ids=cache_fixture(root)
            dataset=data.PreparedTrainingData(root,training_count=2)
            self.assertEqual(len(dataset),2);self.assertEqual(dataset.ids,ids[:2]);self.assertEqual(dataset.validation_ids,ids[2:])
            a=dataset[0];self.assertEqual(a['clean'].shape,(1,16,9,16,28));self.assertEqual(dataset.text_for(0).shape,(1,512,4096))
            a['clean'].zero_();self.assertNotEqual(float(dataset[0]['clean'].mean()),0.)
            self.assertEqual(dataset.record(0)['record_id'],ids[0]);self.assertEqual(len(dataset[-1]['low'].shape),5)
            self.assertEqual(dataset[0]['clean'].device.type,'cpu')
            with self.assertRaises(IndexError):dataset[2]

    def test_sealed_corruption_and_source_leak_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'cache';cache_fixture(root)
            ds=data.PreparedTrainingData(root)
            path=root/'clip_0/fresh_latents.pt';path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):ds[0]
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'cache';cache_fixture(root)
            p=root/'clip_2/metadata.json';m=data.read_json(p);m['source_id']='source0';m['binding']['data_cache_key']=data._digest({k:v for k,v in m.items() if k!='binding'});write(p,m)
            seal=data.read_json(root/'cache_asset_seal.json');seal['artifact_sha256']['clip_2/metadata.json']=data.file_hash(p);write(root/'cache_asset_seal.json',seal)
            with self.assertRaisesRegex(ValueError,'Source leaks'):data.PreparedTrainingData(root)

    def test_cache_path_escape_and_seal_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'cache';cache_fixture(root)
            with self.assertRaisesRegex(ValueError,'seal SHA'):data.PreparedTrainingData(root,expected_seal_sha256='0'*64)
            seal=data.read_json(root/'cache_asset_seal.json');seal['artifact_sha256']['../outside']='0'*64;write(root/'cache_asset_seal.json',seal)
            with self.assertRaisesRegex(ValueError,'escapes'):data.PreparedTrainingData(root)

    def test_vae_normalization_roundtrip_and_posterior_mode(self):
        v=FakeVAE().eval();x=torch.rand((1,3,5,4,6),generator=torch.Generator().manual_seed(12))
        pair,restored,report=vae.encode_pair(v,{'hr':x,'lr_up':x*.7},return_roundtrip=True)
        self.assertTrue(torch.allclose(restored,x,atol=1e-7));self.assertLess(report['mse_rgb_01'],1e-12)
        self.assertEqual(v.events,['encode','mode','encode','mode','decode','clear'])
        decoded=vae.decode_latent(v,pair['clean']);self.assertTrue(torch.allclose(decoded,x,atol=1e-7))
        self.assertTrue(torch.equal(vae.inverse_normalize(pair['clean'],v),vae.normalize_latent(pair['clean'],v,inverse=True)))
        self.assertFalse(pair['clean'].requires_grad)

    def test_vae_rejects_trainable_or_bad_normalization(self):
        v=FakeVAE();v.anchor.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'frozen'):vae.encode_video(v,torch.zeros(1,3,1,4,4))
        v.anchor.requires_grad_(False);v.config.latents_std=[0.,1.,1.]
        with self.assertRaisesRegex(ValueError,'normalization'):vae.normalize_latent(torch.zeros(1,3,1,2,2),v)
        with self.assertRaises(FileNotFoundError):vae.load_frozen_vae('/not/an/asset','cpu')

    def test_geometry_is_original_causal_33_to9(self):
        result=vae.latent_geometry({'temperal_downsample':[False,True,True],'dim_mult':[1,2,4,4]}, {'patch_size':[1,2,2]},33,128,224)
        self.assertEqual(result['latent_grid'],[9,16,28]);self.assertEqual(result['token_grid'],[9,8,14])
        with self.assertRaisesRegex(ValueError,'frames = 1 mod'):vae.latent_geometry({'temperal_downsample':[False,True,True],'dim_mult':[1,2,4,4]}, {'patch_size':[1,2,2]},32,128,224)

    def test_framewise_reader_keeps_frozen_val32_and_hashes(self):
        with tempfile.TemporaryDirectory() as t:
            path,seal=reader_fixture(Path(t)/'assets')
            reader=data.FramewiseReader(path,seal,verify_runtime=False)
            self.assertEqual(len(reader.records['Stage2-val-clean']),32)
            row=reader.load('Stage2-val-noise','record0')
            self.assertEqual(row['lr'].shape,(1,3,33,32,56));self.assertEqual(row['hr'].shape,(1,3,33,128,224))
            self.assertEqual(row['metadata']['framewise_mask'],seal['cohort'][0]['mask'])
            with self.assertRaisesRegex(ValueError,'Only sealed'):reader.load('Stage2-val-clean','record0',start=1)
            spec=reader.assets['record0']['clean'];Path(spec['path']).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'digest'):reader.load('Stage2-val-clean','record0')

    def test_reader_relative_assets_resolve_sealed_root_before_explicit_migration(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);original=root/'old';path,seal=reader_fixture(original)
            document=data.read_json(path)
            for row in document['assets']:
                row['clean']['path']='lr.npy';row['HR']['path']='hr.npy'
                for entry in row['conditions'].values():entry['asset']['path']='lr.npy'
            write(path,document)
            migrated=root/'migrated';original.rename(migrated)
            unrelated=root/'unrelated';unrelated.mkdir()
            before=Path.cwd()
            try:
                os.chdir(unrelated)
                reader=data.FramewiseReader(migrated/'dataset.json',seal,path_map={str(original):str(migrated)},verify_runtime=False)
                item=reader.load('Stage2-val-clean','record0')
            finally:os.chdir(before)
            self.assertEqual(item['lr'].shape,(1,3,33,32,56))
            self.assertEqual(float(item['hr'].mean()),.25)
            self.assertEqual(Path.cwd(),before)

    def test_seal_explicit_path_mapping_checks_original_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            path,seal=reader_fixture(Path(t)/'assets')
            old=Path(next(iter(seal['implementation_bindings'])))
            moved=old.with_name('moved.py');old.rename(moved)
            result=data.validate_seal(seal,path_map={str(old):str(moved)},verify_runtime=False)
            self.assertTrue(result['source_hashes_verified']);self.assertFalse(result['runtime_verified'])
            moved.write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'implementation'):data.validate_seal(seal,path_map={str(old):str(moved)},verify_runtime=False)

    def _preparation_inputs(self,root):
        raw=root/'raw';raw.mkdir()
        for i in range(6):(raw/f'source{i:05d}_0_0to99.mp4').write_bytes(f'raw fixture{i}'.encode())
        manifest=data.build_manifest(raw);write(root/'manifest.json',manifest)
        text=root/'text.pt';torch.save({data.CAPTION_PLACEHOLDER:torch.zeros(1,512,4096,dtype=torch.bfloat16)},text)
        config={'seed':20261102,'model':{'repo_id':'Wan-AI/Wan2.1-T2V-1.3B-Diffusers','revision':'0fad780a534b6463e45facd96134c9f345acfa5b','local_dir':'unused-fixture'},
                'data':{'manifest':str(root/'manifest.json'),'selection':'train','train_limit':2,'val_limit':1,'num_frames':33,'height':128,'width':224,'scale':4,'temporal_stride':1,
                        'text_cache':{'path':str(text),'sha256':data.file_hash(text)}}}
        def decode(record,**options):
            metadata={'record_id':record['id'],'source_id':record['source_id'],'source_path':record['path'],'source_sha256':record['sha256'],'split':record['split'],'temporal_stride':1,
                      'source_indices':list(range(33)),'crop_tlhw':[0,0,128,224],'degradation':{'scale':4},'original_fps':24.,'output_fps':24.,'caption':data.CAPTION_PLACEHOLDER}
            return {'hr':torch.full((1,3,33,128,224),.5),'lr':torch.full((1,3,33,32,56),.5),'lr_up':torch.full((1,3,33,128,224),.5),
                    'metadata':metadata,'cache_key':data._digest(metadata),'fps':24.}
        return config,decode

    def test_streaming_preparation_produces_sealed_reader_compatible_cache(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config,decode=self._preparation_inputs(root);v=ProtocolFakeVAE()
            with patch.object(data,'decode_clip',side_effect=decode),patch.object(vae,'load_frozen_vae',side_effect=AssertionError('injected fixture should skip model load')):
                result=data.prepare_training_cache(config,root/'prepared',vae=v)
            self.assertEqual(result['optimizer_updates'],0);self.assertEqual(result['training_count'],2);self.assertEqual(result['validation_count'],1)
            ds=data.PreparedTrainingData(root/'prepared',verify_all=True)
            self.assertEqual(ds[0]['clean'].shape,(1,16,9,16,28))
            self.assertEqual(data.file_hash(root/'prepared/fresh_text_embeddings.pt'),config['data']['text_cache']['sha256'])
            self.assertEqual(v.events.count('encode'),6);self.assertEqual(v.events.count('decode'),3)
            self.assertFalse(data.read_json(root/'prepared/preparation_complete.json')['task_model_or_optimizer_loaded'])
            with self.assertRaisesRegex(ValueError,'fresh absent'):data.prepare_training_cache(config,root/'prepared',vae=v)

    def test_streaming_preparation_failure_has_no_usable_seal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config,decode=self._preparation_inputs(root);v=ProtocolFakeVAE()
            with patch.object(data,'decode_clip',side_effect=decode),patch.object(vae,'encode_pair',side_effect=RuntimeError('synthetic VAE failure')):
                with self.assertRaisesRegex(RuntimeError,'synthetic VAE failure'):data.prepare_training_cache(config,root/'failed',vae=v)
            self.assertFalse((root/'failed/cache_asset_seal.json').exists())
            report=data.read_json(root/'failed/failure.json');self.assertEqual(report['status'],'failed_no_retry');self.assertEqual(report['optimizer_updates'],0)

    def test_cuda_preparation_requires_admission_before_any_model_or_query(self):
        with patch.object(data,'_prepare_training_cache',side_effect=AssertionError('inner must not run')),patch('vsr.checkpoints.gpu_lease',side_effect=AssertionError('lease must not run')):
            with self.assertRaisesRegex(ValueError,'gpu_lock and gpu_admission'):
                data.prepare_training_cache({},'/not-created',device='cuda:0')
            with self.assertRaisesRegex(ValueError,'gpu_lock and gpu_admission'):
                data.prepare_training_cache({},'/not-created',device='cuda:0',gpu_lock='/tmp/fixture.lock')
        self.assertFalse(torch.cuda.is_initialized())

    def test_cuda_preparation_shared_lease_covers_inner_cleanup_without_GPU(self):
        events=[];active=[False]
        @contextmanager
        def lease(lock,device,admission):
            self.assertEqual(str(device),'cuda:1');self.assertEqual(lock,'fixture-lock');self.assertEqual(admission,{'fixture':True})
            active[0]=True;events.append('enter')
            try:yield {'fixture':True}
            finally:events.append('exit');active[0]=False
        def inner(*args,**kwargs):
            self.assertTrue(active[0]);events.append('prepare')
            try:return {'status':'fixture-only'}
            finally:self.assertTrue(active[0]);events.append('cleanup')
        with patch('vsr.checkpoints.gpu_lease',side_effect=lease),patch.object(data,'_prepare_training_cache',side_effect=inner):
            result=data.prepare_training_cache({},'/not-created',device='cuda:1',gpu_lock='fixture-lock',gpu_admission={'fixture':True})
        self.assertEqual(result,{'status':'fixture-only'});self.assertEqual(events,['enter','prepare','cleanup','exit'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_cpu_preparation_never_opens_GPU_lease(self):
        with patch('vsr.checkpoints.gpu_lease',side_effect=AssertionError('CPU must not query/admit GPU')),patch.object(data,'_prepare_training_cache',return_value={'status':'cpu-fixture'}) as inner:
            self.assertEqual(data.prepare_training_cache({},'/not-created'),{'status':'cpu-fixture'})
            self.assertEqual(inner.call_args.kwargs['device'],'cpu')

    def test_cache_policy_multiframe_is_exact_delegate_and_bounded(self):
        v=PolicyVAE();x=torch.randn(1,3,33,4,4)
        policy=vae.install_cache_policy(v,max_records=2)
        with torch.no_grad():
            out=v.encode(x,return_dict=True).latent_dist.mode()
        self.assertIs(out,x);self.assertIs(v.encode_inputs[0][0],x)
        self.assertEqual(v.encode_inputs[0][1],{'return_dict':True})
        self.assertEqual(policy.ledger,[{'route':'original_delegate','input_shape':[1,3,33,4,4],'encoder_hooks_installed':False}])
        vae.validate_cache_ledger(policy.ledger,False,1)
        policy.remove();self.assertNotIn('encode',v.__dict__)
        self.assertIsNone(getattr(v,'_vsr_t1_cache_policy',None))

    def test_cache_policy_t1_and_original_delegate_ledger(self):
        v=PolicyVAE();policy=vae.install_cache_policy(v,max_records=2)
        with torch.no_grad():
            x=torch.ones(1,3,1,4,4)
            self.assertIs(v.encode(x).latent_dist.mode(),x)
            self.assertTrue(all(y is None for y in v._enc_feat_map))
            v.encode(torch.zeros(1,3,33,4,4))
        vae.validate_cache_ledger(policy.ledger,True,1)
        self.assertEqual(policy.dropped_records,0)
        json.dumps(policy.ledger)
        self.assertEqual(len(v.encoder._forward_hooks),0)
        self.assertEqual(len(v.encoder._forward_pre_hooks),0)
        policy.remove();policy.remove()

    def test_cache_policy_ledger_mismatch_and_duplicate_policy_rejected(self):
        v=PolicyVAE();policy=vae.install_cache_policy(v,max_records=1)
        with self.assertRaisesRegex(RuntimeError,'already attached'):vae.install_cache_policy(v)
        with torch.no_grad():
            v.encode(torch.zeros(1,3,33,4,4));v.encode(torch.zeros(1,3,33,4,4))
        self.assertEqual(policy.dropped_records,1)
        with self.assertRaisesRegex(RuntimeError,'Missing or extra'):vae.validate_cache_ledger(policy.ledger,False,2)
        policy.remove()

    def test_no_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
