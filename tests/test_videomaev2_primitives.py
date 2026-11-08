"""Focused CPU checks; official weights load but no real teacher forward/GPU."""
import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vsr import data,losses,teacher


def rng():return (copy.deepcopy(random.getstate()),copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
def equal_rng(a,b):return a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2])
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')

def prepared(root):
    """A sealed producer-shaped cache with exactly one training and one val clip."""
    root.mkdir();ids=['train_clip','validation_clip']
    cfg={'data':dict(num_frames=33,height=128,width=224,scale=4,temporal_stride=1),
         'model':{'repo_id':'Wan-AI/Wan2.1-T2V-1.3B-Diffusers','revision':'0fad780a534b6463e45facd96134c9f345acfa5b'}}
    write(root/'config.json',cfg);write(root/'selection.json',{'test_accessed':False,'training_ids':ids[:1],'validation_ids':ids[1:],'manifest_sha256':'a'*64})
    write(root/'correspondence.json',{'latent_grid':[9,16,28],'token_grid':[9,8,14]})
    torch.save({data.CAPTION_PLACEHOLDER:torch.zeros(1,512,4096,dtype=torch.bfloat16)},root/'fresh_text_embeddings.pt')
    for i,name in enumerate(ids):
        raw={'record_id':name,'source_id':'source'+str(i),'source_sha256':str(i+1)*64,'split':'train' if i==0 else 'val',
             'temporal_stride':1,'source_indices':list(range(40*i,40*i+33)),'crop_tlhw':[0,0,128,224],
             'source_hw':[128,224],'resize_hw':[128,224],'degradation':{'scale':4},'original_fps':25.,'output_fps':25.,'caption':data.CAPTION_PLACEHOLDER}
        raw['binding']={'data_cache_key':data._digest(raw),'manifest_sha256':'a'*64}
        write(root/f'clip_{i}/metadata.json',raw)
        torch.save({'hr':torch.full((1,3,33,128,224),.25),'lr':torch.zeros(1,3,33,32,56),'lr_up':torch.zeros(1,3,33,128,224)},root/f'clip_{i}/paired_rgb.pt')
        torch.save({'clean':torch.zeros(1,16,9,16,28),'low':torch.zeros(1,16,9,16,28),'binding':raw['binding']},root/f'clip_{i}/fresh_latents.pt')
    write(root/'cache_asset_seal.json',{'contains_task_initialization':False,'artifact_sha256':{str(p.relative_to(root)):data.file_hash(p) for p in root.rglob('*') if p.is_file()}})
    return data.PreparedTrainingData(root)


class VideoMAEv2PrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_preprocess_keeps_whole_crop_and_frame1_to32(self):
        hr=torch.arange(33).float().view(1,1,33,1,1).expand(1,3,33,128,224).clone()/32
        result=teacher.videomaev2_preprocess_hr(hr)
        mean=torch.tensor([.485,.456,.406]).view(1,3,1,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1,1)
        torch.testing.assert_close(result,(hr[:,:,1:]-mean)/std,rtol=0,atol=0)
        self.assertEqual(tuple(result.shape),(1,3,32,128,224))
        self.assertEqual(teacher.videomaev2_feature_spec()['nominal_tubelets'],[[i,i+1] for i in range(1,33,2)])
        for bad in [hr[:,:,:32],hr.double(),hr+1,hr.clone().requires_grad_()]:
            with self.assertRaises(ValueError):teacher.videomaev2_preprocess_hr(bad)
        with torch.autocast('cpu',dtype=torch.bfloat16):self.assertEqual(teacher.videomaev2_preprocess_hr(hr).dtype,torch.float32)

    def test_positional_interpolation_matches_official_order(self):
        # Structured channels/time/space test checks actual indices, not onlyshape.
        original=(torch.arange(1568).view(1,1568,1).float()+torch.arange(768).view(1,1,768).float()/1000).contiguous()
        actual=teacher.videomaev2_position_embedding(original)
        pos=original.reshape(1,8,14,14,768).reshape(8,14,14,768).permute(0,3,1,2)
        pos=F.interpolate(pos,size=(8,14),mode='bicubic',align_corners=False)
        pos=pos.permute(0,2,3,1).reshape(1,8,8,14,768).permute(0,4,1,2,3)
        expected=F.interpolate(pos,size=(16,8,14),mode='trilinear',align_corners=False).permute(0,2,3,4,1).flatten(1,3)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(actual.dtype,torch.float32);self.assertFalse(actual.requires_grad)
        with self.assertRaises(ValueError):teacher.videomaev2_position_embedding(original[:,:1008])

    def test_student_time_drop0_clamp_and_spatial_order(self):
        x=(torch.arange(9).float().view(1,9,1,1,1)*100+torch.arange(2).view(1,1,2,1,1)*10+torch.arange(3).view(1,1,1,3,1)).reshape(1,54,1).requires_grad_()
        y=losses.videomaev2_align_student_time(x,(9,2,3)).reshape(1,16,2,3,1)
        for j in range(16):
            expected=min(8.,max(1.,1+(j+.5)/2-.5))*100+torch.arange(2).view(2,1)*10+torch.arange(3).view(1,3)
            torch.testing.assert_close(y[0,j,:,:,0],expected,rtol=0,atol=0)
        y.sum().backward();self.assertTrue(torch.equal(x.grad.reshape(1,9,2,3,1)[:,0],torch.zeros(1,2,3,1)))
        self.assertTrue(bool((x.grad.reshape(1,9,2,3,1)[:,1:]>0).all()))

    def test_relation_original_dtypes_value_and_gradient_exact_reference(self):
        gen=torch.Generator().manual_seed(12)
        student=torch.randn(1,64,8,generator=gen,requires_grad=True);target=torch.randn(1,64,8,generator=gen,requires_grad=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):value,diag=losses.videomaev2_full_gram_loss(student,target,(16,2,2))
        value.backward();gradient=student.grad.clone();self.assertIsNone(target.grad)
        independent=student.detach().clone().requires_grad_()
        a=F.normalize(independent.to(torch.bfloat16),dim=-1,eps=1e-12);b=F.normalize(target.detach(),dim=-1,eps=1e-12)
        expected=F.relu((a@a.transpose(1,2)-b@b.transpose(1,2)).abs()-.1).mean();expected.backward()
        torch.testing.assert_close(value,expected,rtol=0,atol=0);torch.testing.assert_close(gradient,independent.grad,rtol=0,atol=0)
        self.assertEqual(diag['student_gram_dtype'],'torch.bfloat16');self.assertEqual(diag['teacher_gram_dtype'],'torch.float32');self.assertEqual(diag['loss_dtype'],'torch.float32')
        self.assertEqual(diag['all_pairs'],4096);self.assertEqual(diag['spatial_pairs'],256);self.assertEqual(diag['temporal_pairs'],3840)
        self.assertAlmostEqual(float(value.detach()),(diag['spatial_loss_mean']+15*diag['temporal_loss_mean'])/16,places=7)
        self.assertGreater(float(gradient.norm()),0)
        with self.assertRaises(ValueError):losses.videomaev2_full_gram_loss(student,target.bfloat16(),(16,2,2))

    def test_zero_relation_isfinite_and_nonfinite_inputs_reject(self):
        z=torch.zeros(1,16,4,requires_grad=True);value,diag=losses.videomaev2_full_gram_loss(z,torch.zeros_like(z),(16,1,1));value.backward()
        self.assertEqual(float(value),0);self.assertTrue(bool(torch.isfinite(z.grad).all()))
        bad=z.detach().clone();bad[0,0,0]=float('nan')
        with self.assertRaises(ValueError):losses.videomaev2_full_gram_loss(bad,z.detach(),(16,1,1))

    def test_real_MLP_architecture_rng_and_all_six_gradient_paths(self):
        before=rng();m=losses.VideoMAEv2TRDAlignment();after=rng();self.assertTrue(equal_rng(before,after))
        self.assertEqual(sum(p.numel() for p in m.parameters()),8917760)
        self.assertEqual(list(m.projector.state_dict()),['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias'])
        self.assertEqual(m.spec['project_before_drop0'],True)
        g=torch.Generator().manual_seed(18);x=torch.randn(1,16,1536,generator=g,requires_grad=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):z=m.projector(x)
        target=torch.randn(1,16,768,generator=g)
        value,diag=losses.videomaev2_full_gram_loss(z,target,(16,1,1));value.backward()
        self.assertGreater(float(x.grad.norm()),0);self.assertTrue(all(p.grad is not None and float(p.grad.norm())>0 for p in m.parameters()))
        self.assertTrue(all(p.dtype==torch.float32 for p in m.parameters()))
        with self.assertRaises(ValueError):m(x,target,(9,8,14))

    def test_complete_alignment_shape_dtypes_and_drop0_backprop(self):
        m=losses.VideoMAEv2TRDAlignment(student_dim=4)
        gen=torch.Generator().manual_seed(77)
        x=torch.randn(1,1008,4,generator=gen,requires_grad=True)
        target=torch.randn(1,1792,768,generator=gen,requires_grad=True)
        value,diag=m(x,target,(9,8,14));value.backward()
        self.assertEqual(diag['projected_dtype'],'torch.bfloat16')
        self.assertEqual(diag['interpolated_dtype'],'torch.bfloat16')
        self.assertEqual(diag['all_pairs'],1792**2)
        self.assertEqual(diag['spatial_pairs'],16*112**2)
        self.assertIsNone(target.grad)
        self.assertTrue(torch.equal(x.grad[:,:112],torch.zeros_like(x.grad[:,:112])))
        self.assertGreater(float(x.grad[:,112:].norm()),0)
        self.assertTrue(all(p.grad is not None and float(p.grad.norm())>0 for p in m.parameters()))

    def test_online_HR_uses_same_FM_binding_and_forbids_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=prepared(Path(tmp)/'cache');self.assertEqual(p.hr_rgb_binding(0)['FM_binding'],p.record(0)['binding'])
            x=p.hr_rgb(0);self.assertEqual(tuple(x.shape),(1,3,33,128,224));self.assertEqual(float(x.mean()),.25)
            a=p.verify_hr_rgb_assets();self.assertEqual(a['training_count'],1);self.assertFalse(a['validation_or_test_RGB_loaded']);self.assertFalse(a['RGB_payload_deserialized'])
            for invalid in [-1,1,2]:
                with self.assertRaises(IndexError):p.hr_rgb(invalid)
            (p.root/'clip_0/paired_rgb.pt').write_bytes(b'changed')
            with patch('torch.load',side_effect=AssertionError('must reject bytes before deserialization')):
                with self.assertRaises(ValueError):p.hr_rgb(0)

    def test_online_HR_wrong_region_or_range_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=prepared(Path(tmp)/'cache');path=p.root/'clip_0/metadata.json';raw=json.loads(path.read_text());raw['source_indices']=list(range(10,43));write(path,raw)
            with self.assertRaises(ValueError):p.hr_rgb(0)
        with tempfile.TemporaryDirectory() as tmp:
            p=prepared(Path(tmp)/'cache');rgb=p.root/'clip_0/paired_rgb.pt';torch.save({'hr':torch.ones(1,3,33,128,224)*2,'lr':None,'lr_up':None},rgb)
            p._hashes['clip_0/paired_rgb.pt']=data.file_hash(rgb)
            with self.assertRaises(ValueError):p.hr_rgb(0)


if __name__=='__main__':unittest.main(verbosity=2)
