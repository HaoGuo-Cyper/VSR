"""CPU fakes exercise real tiler/sampler delegation and observed failures."""
from copy import deepcopy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vsr import inference
from vsr import native_qa_restore as auditing
from vsr.native_qa_restore import AuditedNativeRestore, NativeQAAuditError, _rng_snapshot, _rng_equal


class VAE(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.zeros(()),requires_grad=False)
        self.child=nn.Identity().eval(); self.encodes=[]
        self.config=SimpleNamespace(temperal_downsample=[False,True,True],dim_mult=[1,2,4,4],
            z_dim=16,latents_mean=[0.]*16,latents_std=[1.]*16)
    def encode(self,pixels):
        self.encodes.append(tuple(pixels.shape)); _,_,t,h,w=pixels.shape
        v=F.interpolate(pixels,size=((t-1)//4+1,h//8,w//8),mode='nearest').repeat(1,6,1,1,1)[:,:16]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:v))
    def decode(self,z):
        _,_,t,h,w=z.shape
        return SimpleNamespace(sample=F.interpolate(z[:,:3],size=((t-1)*4+1,h*8,w*8),mode='nearest'))


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.zeros(())); self.child=nn.Identity().eval()
        self._reference_context=None; self._captured=None; self.calls=[]
    def clear_features(self): self._captured=None
    def clear_reference(self): self._reference_context=None
    def forward(self,noisy,timestep,text,low,enable_lrv=True,**ref):
        self._reference_context=ref; self._captured=noisy
        self.calls.append({'noise':noisy.clone(),'reference':ref,'grad':torch.is_grad_enabled()})
        return low,None


class Scheduler:
    def set_timesteps(self,steps,device): self.timesteps=torch.arange(steps,0,-1,device=device,dtype=torch.float32)
    def step(self,velocity,timestep,noisy,return_dict=False): return (velocity,)


def plan():
    boxes=[[0,x,32,32] for x in [0,16,32,48]]
    return {'scope':'syntheticCPUfixture only','input_preparation':{
        'LR_shape':[1,3,3,8,20],'HR_shape':[1,3,3,32,80]},'restoration':{
        'steps':2,'sampling_seed':42,'HR_tile_hw':[32,32],'overlap':16,
        'padded_HR_T_HW':[5,32,80],'global_CPU_Gaussian_noise_shape':[1,16,2,4,10],
        'tile_count':4,'tile_hr_tlhw':boxes,'tile_latent_tlhw':[[a//8,b//8,c//8,d//8] for a,b,c,d in boxes]}}


class NativeQARestoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); self.model,self.vae=Model(),VAE()
        self.lr=torch.rand(1,3,3,8,20,generator=torch.Generator().manual_seed(8)); self.text=torch.zeros(1,2,4)
        self.config={'model':{'local_dir':'synthetic-no-officialweights'}}; self.factory=AuditedNativeRestore([plan()])
        self.scheduler=patch('diffusers.UniPCMultistepScheduler.from_pretrained',side_effect=lambda *a,**k:Scheduler())
        self.scheduler.start(); self.addCleanup(self.scheduler.stop)
    def call(self,**kwargs):
        return self.factory(self.model,self.vae,self.lr,self.text,self.config,steps=2,seed=42,tile_hw=(32,32),overlap=16,**kwargs)

    def test_real_tiler_native_sampler_observed_noise_and_state(self):
        self.model.weight.grad=torch.tensor(3.); before=_rng_snapshot(); result,meta=self.call()
        a=meta['native_operator_audit']; self.assertEqual(result.shape,(1,3,3,32,80))
        self.assertEqual(a['status'],'passed'); self.assertTrue(all(a['checks'].values()))
        self.assertEqual(a['full_noise_sha256'],a['expected_global_noise_sha256'])
        self.assertEqual(a['actual_sampler_calls'],4); self.assertEqual(a['actual_VAE_encode_calls'],4)
        self.assertEqual(len(self.model.calls),8); self.assertTrue(all(not x['grad'] for x in self.model.calls))
        self.assertTrue(_rng_equal(before,_rng_snapshot())); self.assertEqual(float(self.model.weight.grad),3.)
        self.assertTrue(self.model.training); self.assertFalse(self.model.child.training)
        self.assertTrue(self.vae.training); self.assertFalse(self.vae.child.training)
        self.assertIsNone(self.model._captured); self.assertIsNone(self.model._reference_context)
        self.assertNotIn('encode',self.vae.__dict__)

    def test_HRI_T1_encode_shared_latent_and_on_off_identical_noise(self):
        _,on=self.call(reference_image=torch.ones(1,3,1,32,80),reference_index=0); on=on['native_operator_audit']
        self.assertEqual(on['actual_VAE_encode_calls'],5); self.assertEqual(on['VAE_encode_calls'][0]['shape'],[1,3,1,32,80])
        self.assertEqual(on['VAE_encode_calls'][0]['role'],'whole_HRI_T1')
        self.assertEqual(len({x['reference_latent_sha256'] for x in on['sample_tile_latent_calls']}),1)
        self.assertTrue(all(x['reference_valid'] and x['reference_index']==0 for x in on['sample_tile_latent_calls']))
        _,off=self.call(); off=off['native_operator_audit']; self.assertEqual(off['actual_VAE_encode_calls'],4)
        self.assertEqual(on['full_noise_sha256'],off['full_noise_sha256'])
        self.assertTrue(all(x['reference_latent_sha256'] is None for x in off['sample_tile_latent_calls']))
        for row in self.model.calls[-8:]:
            self.assertIsNone(row['reference']['ref_latent']); self.assertFalse(bool(row['reference']['ref_valid'][0]))

    def test_corrupt_actual_tile_noise_raises_and_cleans(self):
        original=inference.restore_video
        def corrupt(*args,**kwargs):
            sampler=kwargs['sampler_fn']
            def change(*a,**k):
                k['initial_noise']=k['initial_noise']+1
                return sampler(*a,**k)
            kwargs['sampler_fn']=change
            return original(*args,**kwargs)
        with patch.object(inference,'restore_video',side_effect=corrupt),self.assertRaisesRegex(NativeQAAuditError,'tile noise differs'):
            self.call()
        self.assertEqual(self.factory.last_audit['status'],'failed'); self.assertIsNone(self.model._reference_context)
        self.assertTrue(self.model.training); self.assertFalse(self.model.child.training); self.assertNotIn('encode',self.vae.__dict__)

    def test_regenerated_full_field_per_tile_detected(self):
        original=inference.restore_video; retained=[]
        def regenerate(*args,**kwargs):
            sampler=kwargs['sampler_fn']
            def change(*a,**k):
                base=k['initial_noise']._base.clone(); retained.append(base)
                top,left,height,width=plan()['restoration']['tile_latent_tlhw'][len(retained)-1]
                k['initial_noise']=base[:,:,:,top:top+height,left:left+width]
                return sampler(*a,**k)
            kwargs['sampler_fn']=change
            return original(*args,**kwargs)
        with patch.object(inference,'restore_video',side_effect=regenerate),self.assertRaisesRegex(NativeQAAuditError,'Different full Gaussian fields'):
            self.call()
        self.assertEqual(self.factory.last_audit['status'],'failed')

    def test_RNG_consumption_fails_before_defensive_restoration(self):
        native=inference.sample_tile_latent; before=_rng_snapshot()
        def consumes(*a,**k):
            random.random(); np.random.rand(); torch.rand(())
            return native(*a,**k)
        with patch.object(inference,'sample_tile_latent',side_effect=consumes),self.assertRaisesRegex(NativeQAAuditError,'structural/state audit failed'):
            self.call()
        self.assertFalse(self.factory.last_audit['checks']['global_RNG_not_consumed'])
        self.assertTrue(_rng_equal(before,_rng_snapshot()))

    def test_native_exception_preserves_cause_cleanup(self):
        def fail(*args,**kwargs):
            self.model._reference_context=kwargs['ref_latent']; self.model._captured=torch.ones(1)
            raise RuntimeError('synthetic native failure')
        with patch.object(inference,'sample_tile_latent',side_effect=fail),self.assertRaisesRegex(NativeQAAuditError,'synthetic native failure') as cm:
            self.call(reference_image=torch.zeros(1,3,1,32,80),reference_index=0)
        self.assertIsInstance(cm.exception.__cause__,RuntimeError); self.assertEqual(cm.exception.audit['status'],'failed')
        self.assertIsNone(self.model._reference_context); self.assertIsNone(self.model._captured)
        self.assertTrue(self.model.training); self.assertFalse(self.model.child.training); self.assertNotIn('encode',self.vae.__dict__)

    def test_parameter_mutation_fails(self):
        native=inference.sample_tile_latent
        def mutates(*a,**k):
            self.model.weight.add_(1)
            return native(*a,**k)
        with patch.object(inference,'sample_tile_latent',side_effect=mutates),self.assertRaises(NativeQAAuditError): self.call()
        self.assertFalse(self.factory.last_audit['checks']['all_parameters_buffers_unchanged'])

    def test_extra_encoder_call_observed_and_rejected(self):
        native=inference.sample_tile_latent
        def extra(*a,**k):
            self.vae.encode(torch.zeros(1,3,1,32,80))
            return native(*a,**k)
        with patch.object(inference,'sample_tile_latent',side_effect=extra),self.assertRaisesRegex(NativeQAAuditError,'encode order/shape'): self.call()
        self.assertEqual(self.factory.last_audit['status'],'failed')

    def test_wall_timer_includes_pre_restore_parameter_hashing(self):
        clock=[0.]; original=auditing._network_state
        def hash_with_cost(network):
            value=original(network); clock[0]+=2.
            return value
        with patch.object(auditing.time,'perf_counter',side_effect=lambda:clock[0]),patch.object(auditing,'_network_state',side_effect=hash_with_cost):
            _,metadata=self.call()
        self.assertEqual(metadata['native_operator_audit']['wall_seconds'],8.)

    def test_predeclaration_and_outstanding_context_checked_before_encoding(self):
        changed=deepcopy(plan()); changed['restoration']['sampling_seed']=43; factory=AuditedNativeRestore(changed)
        with self.assertRaisesRegex(ValueError,'seed differs'):
            factory(self.model,self.vae,self.lr,self.text,self.config,steps=2,seed=42,tile_hw=(32,32),overlap=16)
        self.assertEqual(factory.last_audit['status'],'failed')
        self.model._captured=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'outstanding'): self.call()
        self.assertEqual(self.vae.encodes,[])


if __name__=='__main__': unittest.main()
