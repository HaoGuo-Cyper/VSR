"""CPU full-state persistence for the actual six-parameter projector geometry."""
import copy
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from vsr import checkpoints as cp
from vsr.losses import VideoMAEv2TRDAlignment
from vsr.teacher import (videomaev2_feature_spec,VIDEOMAEV2_SOURCE_REVISION,VIDEOMAEV2_SOURCE_SHA256,
    VIDEOMAEV2_WEIGHT_REVISION,VIDEOMAEV2_WEIGHT_SHA256,VIDEOMAEV2_WEIGHT_BYTES)
from vsr.train import MatchedStepRNG
from test_hr_latent_checkpoints import TinyRestoration,source_config,canonical,PROTOCOL


def teacher_binding():
    return {'schema':'vsr_videomaev2_teacher_v1','architecture':'vit_base_patch16_224',
        'source_revision':VIDEOMAEV2_SOURCE_REVISION,'weight_revision':VIDEOMAEV2_WEIGHT_REVISION,
        'source':{'path':'/official/VideoMAEv2.py','bytes':42,'sha256':VIDEOMAEV2_SOURCE_SHA256},
        'weights':{'path':'/official/teacher.pth','bytes':VIDEOMAEV2_WEIGHT_BYTES,'sha256':VIDEOMAEV2_WEIGHT_SHA256},
        'encoder_load_contract':{'checkpoint_container':'module','optional_prefix':'module.',
            'excluded_keys':['fc_norm.bias','fc_norm.weight','head.bias','head.weight'],'strict':True,
            'encoder_state_keys':158,'encoder_parameters':86225664,'depth':12,'width':768,'heads':12,'FP32_parameters':True},
        'position_embedding':{'origin':'official_flattened_1D_sincos1568_not_checkpoint_parameter',
            'original_shape':[1,1568,768],'original_sha256':'b'*64,'shape':[1,1792,768],
            'dtype':'torch.float32','sha256':'c'*64,'spatial':'bicubic_14x14_to8x14_align_cornersFalse',
            'temporal':'trilinear8_to16_align_cornersFalse','learned':False},
        'feature_spec':videomaev2_feature_spec(),'runtime':{'torch':'synthetic','numpy':'synthetic','timm':'synthetic'}}


class TinyExternalTeacher(nn.Module):
    """State API spy; actual official encoder loading is tested in teacher tests."""
    def __init__(self,binding):
        super().__init__();self.model=nn.Linear(2,2).requires_grad_(False).eval()
        self._binding=copy.deepcopy(binding);self.parameter_updates=0;self.eval()
    @property
    def binding(self):return copy.deepcopy(self._binding)
    def state_dict(self,*a,**k):raise AssertionError('External teacher tensors must never enter checkpoints')


def seed_global():random.seed(101);np.random.seed(102);torch.manual_seed(103)


class TinyEngine:
    def __init__(self,arm='videomaev2'):
        self.arm=arm;self.weight=0. if arm=='baseline' else .5
        self.training_seed,self.student_layer,self.steps,self.training_clip_count=20261102,17,0,1641
        self.teacher_forward_count=0;self.teacher_binding=None if arm=='baseline' else teacher_binding()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73);self.model=TinyRestoration()
            self.alignment=None if arm=='baseline' else VideoMAEv2TRDAlignment(1536,init_seed=20261102)
            self.teacher=None if arm=='baseline' else TinyExternalTeacher(self.teacher_binding)
        self.rng=MatchedStepRNG(self.training_seed,'cpu')
        named=[('restoration.'+n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        if self.alignment is not None:named.extend(('student_projector.'+n,p) for n,p in self.alignment.projector.named_parameters())
        self._named_optimizer_parameters=tuple(named);self.parameters=[p for n,p in named]
        self.optimizer=torch.optim.AdamW(self.parameters,lr=.002,weight_decay=.01,foreach=False)
        self.base_load_audit={'checkpoint_path':'/origin/Base/checkpoint_008192.pt','checkpoint_sha256':'a'*64,
            'step':8192,'variant':'Base','config':source_config(),'optimizer_state_restored':False,
            'scheduler_state_restored':False,'rng_state_restored':False,'fresh_note':'original'}
    def step(self):
        x=torch.randn((2,2),generator=self.rng.noise)+.01*torch.randint(0,100,(),generator=self.rng.timestep).float()
        x+=.1*torch.rand((2,2),generator=self.rng.mask)+.1*torch.randn((2,2),generator=self.rng.corruption)
        x+=.03*random.random()+.02*float(np.random.rand())+.01*torch.rand(())
        y=self.model.lrv_stem(x);loss=y.square().mean()
        if self.alignment is not None:
            # Actual MLP dimensions/parameters; intentionally tiny token batch.
            z=self.alignment.projector(y.repeat(1,768))
            loss+=self.weight*(z-1).square().mean()
        self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        self.steps+=1;self.teacher_forward_count+=self.arm!='baseline'
        return loss.detach().clone()


def config_for(engine):
    config=source_config();config['stage']='videomaev2_test';config['training']['updates']=8
    config['auxiliary_supervision']=cp.videomaev2_auxiliary_spec(engine.arm,engine.weight,engine.teacher_binding)
    return canonical(config)


def snapshot(engine):
    return cp._cpu_owned({'student':engine.model.state_dict(),'head':None if engine.alignment is None else engine.alignment.state_dict(),
        'teacher_binding':engine.teacher_binding,'teacher':None if engine.teacher is None else dict(engine.teacher.model.named_parameters()),
        'optimizer':engine.optimizer.state_dict(),'explicit_RNG':cp._trd_explicit_rng(engine)[0],
        'global_RNG':cp._trd_global_rng_state(),'steps':engine.steps,'teacher_forward_count':engine.teacher_forward_count})


class VideoMAEv2CheckpointTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);seed_global()
    def save(self,folder,engine,name='round'):
        config=config_for(engine);path=Path(folder)/name/'Base'/f'checkpoint_{engine.steps:06d}.pt'
        result=cp.save_videomaev2_checkpoint(path,engine,config,allowed_root=Path(folder),protocol_sha256=PROTOCOL,
            cumulative_training_seconds=12.5,initialization=engine.base_load_audit)
        return path,config,result
    def equal(self,left,right):self.assertTrue(cp._hr_equal(left,right))

    def test_exact_resume_both_arms_with_actual_six_tensor_head_and_all_rng(self):
        for arm in ('baseline','videomaev2'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as folder:
                seed_global();reference=TinyEngine(arm);expected=[reference.step() for _ in range(4)];end=snapshot(reference)
                seed_global();broken=TinyEngine(arm);actual=[broken.step() for _ in range(2)];path,config,binding=self.save(folder,broken)
                fresh=TinyEngine(arm);fresh.base_load_audit['fresh_note']='another process'
                random.random();np.random.rand(7);torch.rand(13)
                receipt=cp.restore_videomaev2_training_state(fresh,path,allowed_root=Path(folder),expected_protocol_sha256=PROTOCOL,expected_config=config)
                self.assertEqual(receipt['cumulative_training_seconds'],12.5)
                self.assertEqual(receipt['data_cursor'],{'completed_updates':2,'training_clip_count':1641,'next_training_index':2})
                self.assertEqual(receipt['teacher_forward_count'],0 if arm=='baseline' else 2)
                self.assertFalse(receipt['teacher_tensor_state_restored']);self.assertTrue(receipt['global_RNG_restored'])
                actual.extend(fresh.step() for _ in range(2))
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.equal(snapshot(fresh),end)
                self.assertFalse(binding['teacher_tensor_state_saved'])
                payload=torch.load(path,map_location='cpu',weights_only=True)
                self.assertNotIn('teacher_state',payload['training_only'])
                self.assertEqual(len(payload['training_only']['alignment_state'] or {}),0 if arm=='baseline' else 6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_student_only_native_inference_ignores_aux_tensors_and_emits_scalar_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=TinyEngine();engine.step();path,config,_=self.save(folder,engine)
            payload=torch.load(path,map_location='cpu',weights_only=True)
            for value in payload['training_only']['alignment_state'].values():value.fill_(float('nan'))
            torch.save(payload,path)
            def factory(config,variant):return TinyRestoration(),{'official_revision':cp.OFFICIAL_REVISION,'fresh_official_initialization':True,'legacy_state_loaded':False}
            model,audit=cp.load_training_checkpoint(path,device='cpu',allowed_root=Path(folder),model_factory=factory)
            self.assertIsNone(model.projector);self.assertFalse(audit['training_only_state_restored']);self.assertFalse(audit['EMA_weights_selected'])
            for n,p in model.named_parameters():
                if p.requires_grad:torch.testing.assert_close(p,dict(engine.model.named_parameters())[n],rtol=0,atol=0)
            self.assertEqual(audit['videomaev2_training_metadata'],{'schema':cp.VIDEOMAEV2_TRAINING_STATE_SCHEMA,
                'protocol_sha256':PROTOCOL,'arm':'videomaev2','completed_updates':1,'training_seed':20261102,'weight':.5,
                'student_layer':17,'teacher_forward_count':1,'teacher_parameter_updates':0,'EMA_updates':0})

    def test_outer_head_or_teacher_injection_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=TinyEngine('baseline');engine.step();path,config,_=self.save(folder,engine);original=torch.load(path,map_location='cpu',weights_only=True)
            for target in ('head_key','teacher_key','outer_head','outer_teacher'):
                payload=copy.deepcopy(original)
                if target=='head_key':payload['trainable_state']['student_projector.0.weight']=torch.zeros(1)
                elif target=='teacher_key':payload['trainable_state']['teacher.model.weight']=torch.zeros(1)
                else:payload[target]=torch.zeros(1)
                torch.save(payload,path);factory=unittest.mock.Mock(side_effect=AssertionError('No construction before rejection'))
                with self.subTest(target=target),self.assertRaises(ValueError):
                    cp.load_training_checkpoint(path,device='cpu',allowed_root=Path(folder),model_factory=factory)
                factory.assert_not_called()

    def test_corrupt_state_rejected_before_all_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=TinyEngine('baseline');engine.step();path,config,_=self.save(folder,engine);original=torch.load(path,map_location='cpu',weights_only=True)
            for case in ('schema','protocol','arm','layer','Base','student','moment','variance','count','RNG','global_python','global_numpy','global_cpu','global_cuda','cursor','seconds','scheduler','teacher_count','teacher_updates','extra_teacher_state'):
                payload=copy.deepcopy(original);state=payload['training_only'];pid=state['optimizer']['param_groups'][0]['params'][0]
                if case=='schema':state['schema']='old'
                elif case=='protocol':state['protocol_sha256']='d'*64
                elif case=='arm':state['arm']='ema'
                elif case=='layer':state['student_layer']=9
                elif case=='Base':state['matched_Base']['checkpoint_sha256']='d'*64
                elif case=='student':payload['trainable_state']['lrv_stem.weight']=torch.zeros(1)
                elif case=='moment':state['optimizer']['state'][pid]['exp_avg']=torch.zeros(1)
                elif case=='variance':state['optimizer']['state'][pid]['exp_avg_sq'].fill_(-1)
                elif case=='count':state['optimizer']['state'][pid]['step']=torch.tensor(2.)
                elif case=='RNG':state['explicit_RNG']['noise']=torch.zeros(1,dtype=torch.uint8)
                elif case=='global_python':state['global_RNG']['python']=('bad',)
                elif case=='global_numpy':state['global_RNG']['numpy']['keys'][0]=-1
                elif case=='global_cpu':state['global_RNG']['torch_cpu']=torch.zeros(1,dtype=torch.uint8)
                elif case=='global_cuda':state['global_RNG']['torch_cuda']=[torch.zeros(1,dtype=torch.uint8)]
                elif case=='cursor':state['data_cursor']['next_training_index']=9
                elif case=='seconds':state['cumulative_training_seconds']=float('nan')
                elif case=='scheduler':state['learning_rate_schedule']['type']='cosine'
                elif case=='teacher_count':state['teacher_forward_count']=1
                elif case=='teacher_updates':state['teacher_parameter_updates']=1
                else:state['teacher_state']={'shadow':torch.zeros(1)}
                torch.save(payload,path);fresh=TinyEngine('baseline');before=snapshot(fresh)
                with self.subTest(case=case),self.assertRaises((ValueError,TypeError,RuntimeError)):
                    cp.restore_videomaev2_training_state(fresh,path,allowed_root=Path(folder),expected_protocol_sha256=PROTOCOL,expected_config=config)
                self.equal(before,snapshot(fresh))

    def test_method_six_head_spec_and_immutable_teacher_binding_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=TinyEngine();engine.step();path,config,_=self.save(folder,engine);original=torch.load(path,map_location='cpu',weights_only=True)
            for case in ('head_shape','head_spec','position','preprocess','forward_count'):
                payload=copy.deepcopy(original);state=payload['training_only']
                if case=='head_shape':state['alignment_state']['projector.4.bias']=torch.zeros(1)
                elif case=='head_spec':state['alignment_spec']['target_grid']=[8,8,14]
                elif case=='position':state['teacher_binding']['position_embedding']['sha256']='d'*64
                elif case=='preprocess':state['teacher_binding']['feature_spec']['input_range']='wrong'
                else:state['teacher_forward_count']=0
                torch.save(payload,path);fresh=TinyEngine();before=snapshot(fresh)
                with self.subTest(case=case),self.assertRaises(ValueError):cp.restore_videomaev2_training_state(fresh,path,allowed_root=Path(folder),expected_protocol_sha256=PROTOCOL,expected_config=config)
                self.equal(before,snapshot(fresh))

    def test_transactional_rollback_on_optimizer_apply_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=TinyEngine();engine.step();path,config,_=self.save(folder,engine);fresh=TinyEngine();before=snapshot(fresh);original=fresh.optimizer.load_state_dict;calls=[]
            def fail_once(value):
                calls.append(1)
                if len(calls)==1:raise RuntimeError('injected optimizer apply failure')
                return original(value)
            with patch.object(fresh.optimizer,'load_state_dict',side_effect=fail_once),self.assertRaisesRegex(RuntimeError,'injected optimizer'):
                cp.restore_videomaev2_training_state(fresh,path,allowed_root=Path(folder),expected_protocol_sha256=PROTOCOL,expected_config=config)
            self.assertEqual(len(calls),2);self.equal(before,snapshot(fresh))

    def test_publication_no_overwrite_and_invalid_frozen_contract_or_tensor_binding_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base=TinyEngine('baseline');base.step();path,_,_=self.save(folder,base);old=path.read_bytes()
            with self.assertRaises(ValueError):self.save(folder,base)
            self.assertEqual(old,path.read_bytes())
            method=TinyEngine();method.step();method.teacher.model.train()
            with self.assertRaisesRegex(ValueError,'frozen eval'):self.save(folder,method,'bad_teacher')
            method.teacher.eval();method.teacher_binding['extra_tensor']=torch.zeros(1)
            with self.assertRaisesRegex(ValueError,'must not contain tensor'):self.save(folder,method,'tensor_binding')
            self.assertFalse((Path(folder)/'bad_teacher').exists());self.assertFalse((Path(folder)/'tensor_binding').exists())


if __name__=='__main__':unittest.main(verbosity=2)
