"""Exact TRD save/resume, corruption rejection and student-only inference on CPU."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from vsr import checkpoints as cp
from vsr.losses import FullGramTRD
from vsr.teacher import FP32TrainableTeacher
from vsr.train import MatchedStepRNG
from test_hr_latent_checkpoints import canonical, source_config, TinyRestoration, PROTOCOL


def run_config(arm, weight):
    config = source_config()
    config['stage'] = 'wan_trd_teacher_targets'
    config['training']['updates'] = 8
    config['trd_supervision'] = {'arm':arm,'weight':weight,'student_layer':9,'margin':.1,
                                 'normalize_eps':1e-6,'teacher_sigma_ratio':.5}
    return canonical(config)


def seed_global():
    random.seed(101);np.random.seed(102);torch.manual_seed(103)


class TinyTRDEngine:
    def __init__(self, arm='ema', weight=.1):
        self.trd_arm = self.teacher_kind = arm
        self.weight = 0. if arm == 'baseline' else weight
        self.training_seed,self.student_layer,self.teacher_layer,self.steps = 20261102,9,9,0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            self.model = TinyRestoration()
            self.alignment = FullGramTRD(2) if arm != 'baseline' else None
        self.teacher = None if arm == 'baseline' else FP32TrainableTeacher(self.model,mode=arm)
        self.rng = MatchedStepRNG(self.training_seed,'cpu')
        named = [('restoration.'+name,p) for name,p in self.model.named_parameters() if p.requires_grad]
        if self.alignment is not None:
            named += [('student_projector.'+name,p) for name,p in self.alignment.projector.named_parameters()]
        self._named_optimizer_parameters = tuple(named)
        self.parameters = [p for name,p in named]
        self.optimizer = torch.optim.AdamW(self.parameters,lr=.002,weight_decay=.01,foreach=False)
        self.base_load_audit = {'checkpoint_path':'/origin/Base/checkpoint_008192.pt','checkpoint_sha256':'a'*64,
            'variant':'Base','step':8192,'config':source_config(),'optimizer_state_restored':False,
            'scheduler_state_restored':False,'rng_state_restored':False,'fresh_process_note':'initial'}

    def step(self):
        value = torch.randn((2,2),generator=self.rng.noise)
        value += .01*torch.randint(0,100,(),generator=self.rng.timestep).float()
        value += .1*torch.rand((2,2),generator=self.rng.mask)
        value += .1*torch.randn((2,2),generator=self.rng.corruption)
        # These global draws make the resume test sensitive to all global RNGs.
        value += .03*random.random()+.02*float(np.random.rand())+.01*torch.rand(())
        prediction = self.model.lrv_stem(value)
        loss = prediction.square().mean()
        if self.alignment is not None:
            with torch.no_grad():
                shadow = self.teacher.shadow
                target = value@shadow['lrv_stem.weight'].T+shadow['lrv_stem.bias']
            loss = loss+self.weight*(self.alignment.projector(prediction)-target-1).square().mean()
        self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step()
        if self.trd_arm == 'ema':self.teacher.update(self.model)
        self.optimizer.zero_grad(set_to_none=True);self.steps += 1
        return loss.detach().clone()


def snapshot(engine):
    return cp._cpu_owned({'student':engine.model.state_dict(),
        'head':None if engine.alignment is None else engine.alignment.state_dict(),
        'teacher':None if engine.teacher is None else engine.teacher.state_dict(),
        'optimizer':engine.optimizer.state_dict(),
        'explicit_RNG':{name:getattr(engine.rng,name).get_state() for name in cp._TRD_STREAMS},
        'global_RNG':cp._trd_global_rng_state(),'steps':engine.steps})


class TRDCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);seed_global()

    def assert_tree_equal(self, a, b):
        self.assertTrue(cp._hr_equal(a,b))

    def save(self, folder, engine, name='round'):
        config = run_config(engine.trd_arm,engine.weight)
        path = Path(folder)/name/'Base'/f'checkpoint_{engine.steps:06d}.pt'
        binding = cp.save_trd_checkpoint(path,engine,config,allowed_root=Path(folder),
            protocol_sha256=PROTOCOL,cumulative_training_seconds=12.5,initialization=engine.base_load_audit)
        return path,config,binding

    def test_exact_full_trajectory_resume_for_all_arms_and_all_RNGs(self):
        for arm in ('baseline','frozen','ema'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as folder:
                seed_global();reference = TinyTRDEngine(arm)
                expected = [reference.step() for _ in range(4)];reference_state = snapshot(reference)
                seed_global();interrupted = TinyTRDEngine(arm)
                actual = [interrupted.step() for _ in range(2)]
                path,config,binding = self.save(folder,interrupted)
                fresh = TinyTRDEngine(arm);fresh.base_load_audit['fresh_process_note'] = 'different'
                random.random();np.random.rand(4);torch.rand(13)
                audit = cp.restore_trd_training_state(fresh,path,allowed_root=Path(folder),
                    expected_protocol_sha256=PROTOCOL,expected_config=config)
                self.assertEqual(audit['steps'],2);self.assertEqual(audit['arm'],arm)
                self.assertEqual(audit['cumulative_training_seconds'],12.5)
                self.assertTrue(audit['global_RNG_restored'])
                actual.extend(fresh.step() for _ in range(2))
                for left,right in zip(actual,expected):torch.testing.assert_close(left,right,rtol=0,atol=0)
                self.assert_tree_equal(snapshot(fresh),reference_state)
                self.assertEqual(binding['training_only_schema'],cp.TRD_TRAINING_STATE_SCHEMA)
                self.assertEqual(float(fresh.model.frozen),11.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_inference_loads_student_only_and_reports_scalar_provenance(self):
        for arm in ('baseline','frozen','ema'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as folder:
                engine = TinyTRDEngine(arm);engine.step();path,config,_ = self.save(folder,engine)
                payload = torch.load(path,map_location='cpu',weights_only=True)
                if engine.teacher is not None:
                    for value in payload['training_only']['teacher_state']['shadow'].values():value.fill_(float('nan'))
                    payload['training_only']['alignment_state']['projector.weight'].fill_(float('nan'))
                torch.save(payload,path)
                def factory(config,variant):
                    return TinyRestoration(),{'official_revision':cp.OFFICIAL_REVISION,
                        'fresh_official_initialization':True,'legacy_state_loaded':False}
                loaded,audit = cp.load_training_checkpoint(path,device='cpu',allowed_root=Path(folder),model_factory=factory)
                for name,p in loaded.named_parameters():
                    if p.requires_grad:torch.testing.assert_close(p,dict(engine.model.named_parameters())[name],rtol=0,atol=0)
                self.assertIsNone(loaded.projector)
                self.assertFalse(audit['EMA_weights_selected']);self.assertFalse(audit['training_only_state_restored'])
                self.assertEqual(audit['trd_training_metadata'],{
                    'schema':cp.TRD_TRAINING_STATE_SCHEMA,'protocol_sha256':PROTOCOL,'arm':arm,
                    'completed_updates':1,'training_seed':20261102,'weight':engine.weight,
                    'student_layer':9,'EMA_updates':1 if arm == 'ema' else 0})

    def test_corrupt_state_is_rejected_before_any_engine_or_global_rng_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = TinyTRDEngine();saved.step();path,config,_ = self.save(folder,saved)
            original = torch.load(path,map_location='cpu',weights_only=True)
            cases = ('schema','protocol','arm','weight','layer','Base','student','head','head_spec','teacher_count',
                     'teacher_mode','teacher_shape','teacher_nan','teacher_dtype','optimizer_step','optimizer_variance',
                     'RNG_device','RNG_state','global_python','global_numpy','global_torch','global_cuda','cursor','time')
            for case in cases:
                payload = copy.deepcopy(original);state = payload['training_only']
                pid = state['optimizer']['param_groups'][0]['params'][0]
                if case == 'schema':state['schema'] = 'old'
                elif case == 'protocol':state['protocol_sha256'] = 'f'*64
                elif case == 'arm':state['arm'] = 'frozen'
                elif case == 'weight':state['weight'] = .2
                elif case == 'layer':state['student_layer'] = 8
                elif case == 'Base':state['matched_Base']['checkpoint_sha256'] = 'b'*64
                elif case == 'student':payload['trainable_state']['lrv_stem.weight'] = torch.zeros(1)
                elif case == 'head':state['alignment_state']['projector.weight'] = torch.zeros(1)
                elif case == 'head_spec':state['alignment_spec']['margin'] = 0.
                elif case == 'teacher_count':state['teacher_state']['num_updates'] = 3
                elif case == 'teacher_mode':state['teacher_state']['mode'] = 'frozen'
                elif case == 'teacher_shape':state['teacher_state']['shadow']['lrv_stem.weight'] = torch.zeros(1)
                elif case == 'teacher_nan':state['teacher_state']['shadow']['lrv_stem.weight'].fill_(float('nan'))
                elif case == 'teacher_dtype':state['teacher_state']['shadow']['lrv_stem.weight'] = state['teacher_state']['shadow']['lrv_stem.weight'].double()
                elif case == 'optimizer_step':state['optimizer']['state'][pid]['step'] = torch.tensor(2.)
                elif case == 'optimizer_variance':state['optimizer']['state'][pid]['exp_avg_sq'].fill_(-1)
                elif case == 'RNG_device':state['RNG_devices']['noise'] = 'cuda:0'
                elif case == 'RNG_state':state['explicit_RNG']['noise'] = torch.zeros(1,dtype=torch.uint8)
                elif case == 'global_python':state['global_RNG']['python'] = ('bad',)
                elif case == 'global_numpy':state['global_RNG']['numpy']['keys'][0] = -1
                elif case == 'global_torch':state['global_RNG']['torch_cpu'] = torch.zeros(1,dtype=torch.uint8)
                elif case == 'global_cuda':state['global_RNG']['torch_cuda'] = [torch.zeros(1,dtype=torch.uint8)]
                elif case == 'cursor':state['data_cursor']['next_training_index'] = 42
                else:state['cumulative_training_seconds'] = float('nan')
                torch.save(payload,path);fresh = TinyTRDEngine();before = snapshot(fresh)
                with self.subTest(case=case),self.assertRaises((ValueError,RuntimeError,TypeError)):
                    cp.restore_trd_training_state(fresh,path,allowed_root=Path(folder),
                        expected_protocol_sha256=PROTOCOL,expected_config=config)
                self.assert_tree_equal(before,snapshot(fresh))

    def test_restore_rolls_back_after_a_failure_during_optimizer_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = TinyTRDEngine();saved.step();path,config,_ = self.save(folder,saved)
            fresh = TinyTRDEngine();before = snapshot(fresh);original = fresh.optimizer.load_state_dict
            calls = []
            def fail_once(value):
                calls.append(1)
                if len(calls) == 1:raise RuntimeError('injected apply failure')
                return original(value)
            with patch.object(fresh.optimizer,'load_state_dict',side_effect=fail_once):
                with self.assertRaisesRegex(RuntimeError,'injected apply failure'):
                    cp.restore_trd_training_state(fresh,path,allowed_root=Path(folder),
                        expected_protocol_sha256=PROTOCOL,expected_config=config)
            self.assertEqual(len(calls),2);self.assert_tree_equal(snapshot(fresh),before)

    def test_frozen_teacher_update_count_and_FP32_are_required_on_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = TinyTRDEngine('frozen');engine.step();engine.teacher.num_updates = 1
            with self.assertRaises(ValueError):self.save(folder,engine)
            self.assertFalse((Path(folder)/'round').exists())
            engine.teacher.num_updates = 0
            name = next(iter(engine.teacher.shadow));engine.teacher.shadow[name] = engine.teacher.shadow[name].double()
            with self.assertRaises(ValueError):self.save(folder,engine)

    def test_publication_never_overwrites_and_incomplete_optimizer_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = TinyTRDEngine();engine.step();path,config,binding = self.save(folder,engine)
            original = path.read_bytes()
            with self.assertRaises(ValueError):self.save(folder,engine)
            self.assertEqual(original,path.read_bytes())
            engine.optimizer.state.pop(next(iter(engine.optimizer.state)))
            with self.assertRaises(ValueError):self.save(folder,engine,'incomplete')
            self.assertFalse((Path(folder)/'incomplete').exists())

    def test_protocol_config_wrong_origin_missing_sidecar_and_started_engine_reject(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as other:
            saved = TinyTRDEngine();saved.step();path,config,_ = self.save(folder,saved)
            for case in ('protocol','config','origin','started','path'):
                fresh = TinyTRDEngine()
                if case == 'origin':fresh.base_load_audit['checkpoint_sha256'] = 'd'*64
                if case == 'started':fresh.step()
                before = snapshot(fresh)
                with self.subTest(case=case),self.assertRaises(ValueError):
                    cp.restore_trd_training_state(fresh,path,allowed_root=Path(other) if case == 'path' else Path(folder),
                        expected_protocol_sha256='d'*64 if case == 'protocol' else PROTOCOL,
                        expected_config=canonical(config|{'changed':True}) if case == 'config' else config)
                self.assert_tree_equal(before,snapshot(fresh))
            (path.parent.parent/'config.json').unlink();fresh = TinyTRDEngine()
            with self.assertRaises(ValueError):cp.restore_trd_training_state(fresh,path,allowed_root=Path(folder),expected_protocol_sha256=PROTOCOL)


    def test_finite_frozen_teacher_drift_rejects_before_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = TinyTRDEngine('frozen');saved.step();path,config,_ = self.save(folder,saved)
            payload = torch.load(path,map_location='cpu',weights_only=True)
            payload['training_only']['teacher_state']['shadow']['lrv_stem.weight'].add_(.1)
            torch.save(payload,path);fresh = TinyTRDEngine('frozen');before = snapshot(fresh)
            with self.assertRaisesRegex(ValueError,'fresh Base initialization'):
                cp.restore_trd_training_state(fresh,path,allowed_root=Path(folder),
                    expected_protocol_sha256=PROTOCOL,expected_config=config)
            self.assert_tree_equal(before,snapshot(fresh))


if __name__ == '__main__':unittest.main()
