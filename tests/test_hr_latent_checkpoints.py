"""CPU exact same-round resume; no Wan weights, model loading or GPU work."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from vsr.checkpoints import (OFFICIAL_REPO_ID, OFFICIAL_REVISION,
    save_hr_latent_checkpoint, restore_hr_latent_training_state, load_training_checkpoint)
from vsr.losses import HRLatentRegression
from vsr.train import MatchedStepRNG


PROTOCOL = 'e'*64


def canonical(config):
    config=copy.deepcopy(config);config.pop('_sha256',None)
    config['_sha256']=hashlib.sha256((json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    return config


def source_config():
    return canonical({'seed':20261102,
        'model':{'repo_id':OFFICIAL_REPO_ID,'revision':OFFICIAL_REVISION,'local_dir':'/explicit/official'},
        'conditioning':{'hri_enabled':False},'reference':{'enabled':False},
        'data':{'num_frames':33,'temporal_stride':1,'height':128,'width':224,'scale':4},
        'training':{'variants':['Base'],'updates':8192,'seed':20261102}})


def run_config(weight):
    config=source_config();config['stage']='hr_latent_round';config['training']['updates']=8
    config['hr_latent_aux']={'weight':weight,'student_layer':7,'loss':'SmoothL1','beta':1.,'reduction':'mean'}
    return canonical(config)


class TinyRestoration(nn.Module):
    def __init__(self):
        super().__init__();self.lrv_stem=nn.Linear(2,2);self.frozen=nn.Parameter(torch.tensor([11.]),requires_grad=False)
        self.projector=None
    def clear_features(self):pass


class TinyHREngine:
    def __init__(self,weight=.1,head_input=2):
        self.training_seed=20261102;self.weight=weight;self.student_layer=7;self.steps=0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73);self.model=TinyRestoration()
            self.alignment=HRLatentRegression(head_input,2,(1,1,1)) if weight else None
        self.rng=MatchedStepRNG(self.training_seed,'cpu')
        named=[('restoration.'+n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        if self.alignment is not None:named += [('hr_head.'+n,p) for n,p in self.alignment.named_parameters()]
        self._named_optimizer_parameters=tuple(named);self.parameters=[p for n,p in named]
        self.optimizer=torch.optim.AdamW(self.parameters,lr=.002,weight_decay=.01,foreach=False)
        self.base_load_audit={'checkpoint_path':'/origin/Base/checkpoint_008192.pt','checkpoint_sha256':'a'*64,
            'variant':'Base','step':8192,'config':source_config(),'optimizer_state_restored':False,
            'scheduler_state_restored':False,'rng_state_restored':False,
            'runtime_before_observation':'may differ across fresh processes'}
    def step(self):
        # Exercise every explicit stream and every parameter, independently of
        # any task data. This tests state persistence, not the scientific loss.
        noise=torch.randn((2,2),generator=self.rng.noise)
        ordinal=torch.randint(0,100,(),generator=self.rng.timestep).float()
        mask=torch.rand((2,2),generator=self.rng.mask)
        corruption=torch.randn((2,2),generator=self.rng.corruption)
        value=noise+.01*ordinal+.1*mask+.1*corruption
        prediction=self.model.lrv_stem(value)
        loss=prediction.square().mean()
        if self.alignment is not None:loss=loss+self.weight*F.smooth_l1_loss(self.alignment.projector(prediction),torch.ones_like(prediction))
        self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True);self.steps+=1
        return loss.detach().clone()


def snapshot(engine):
    return {'model':copy.deepcopy(engine.model.state_dict()),
        'head':None if engine.alignment is None else copy.deepcopy(engine.alignment.state_dict()),
        'optimizer':copy.deepcopy(engine.optimizer.state_dict()),
        'RNG':{n:getattr(engine.rng,n).get_state().clone() for n in ('noise','timestep','mask','corruption')},'steps':engine.steps}


class HRCheckpointTests(unittest.TestCase):
    def assert_tree_equal(self,left,right):
        self.assertEqual(type(left),type(right))
        if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,rtol=0,atol=0)
        elif isinstance(left,dict):
            self.assertEqual(set(left),set(right))
            for k in left:self.assert_tree_equal(left[k],right[k])
        elif isinstance(left,(tuple,list)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.assert_tree_equal(a,b)
        else:self.assertEqual(left,right)

    def save(self,root,engine,name='round'):
        config=run_config(engine.weight);path=Path(root)/name/'Base'/f'checkpoint_{engine.steps:06d}.pt'
        binding=save_hr_latent_checkpoint(path,engine,config,allowed_root=Path(root),
            protocol_sha256=PROTOCOL,cumulative_training_seconds=12.5,initialization=engine.base_load_audit)
        return path,config,binding

    def test_exact_uninterrupted_and_resume_trajectory_for_baseline_and_auxiliary(self):
        for weight in (0.,.01,.1):
            with self.subTest(weight=weight),tempfile.TemporaryDirectory() as folder:
                uninterrupted=TinyHREngine(weight);expected=[uninterrupted.step() for _ in range(4)]
                interrupted=TinyHREngine(weight);actual=[interrupted.step() for _ in range(2)]
                path,config,binding=self.save(folder,interrupted)
                self.assertEqual(binding['variant'],'Base');self.assertEqual(binding['updates'],2)
                fresh=TinyHREngine(weight);fresh.base_load_audit['runtime_before_observation']='different fresh process'
                audit=restore_hr_latent_training_state(fresh,path,allowed_root=Path(folder),
                    expected_protocol_sha256=PROTOCOL,expected_config=config)
                self.assertEqual(audit['steps'],2);self.assertEqual(audit['cumulative_training_seconds'],12.5)
                self.assertEqual(audit['initialization'],interrupted.base_load_audit)
                actual.extend(fresh.step() for _ in range(2))
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
                self.assert_tree_equal(snapshot(fresh),snapshot(uninterrupted))
                self.assertEqual(float(fresh.model.frozen),11.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_generic_inference_restores_only_student_and_ignores_private_training_state(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=TinyHREngine();engine.step();path,config,binding=self.save(folder,engine)
            payload=torch.load(path,map_location='cpu',weights_only=True)
            self.assertTrue(all(not n.startswith('hr_head.') for n in payload['trainable_state']))
            # A generic inference loader must not even validate these fields.
            payload['training_only']={'schema':'poison for inference','optimizer':{'bad':torch.tensor(float('nan'))}}
            torch.save(payload,path)
            def factory(cfg,variant):
                self.assertEqual(variant,'Base')
                return TinyRestoration(),{'official_revision':OFFICIAL_REVISION,'fresh_official_initialization':True,'legacy_state_loaded':False}
            model,audit=load_training_checkpoint(path,device='cpu',allowed_root=Path(folder),model_factory=factory)
            for n,p in model.named_parameters():
                if p.requires_grad:torch.testing.assert_close(p,dict(engine.model.named_parameters())[n],rtol=0,atol=0)
            self.assertIsNone(model.projector);self.assertFalse(audit['optimizer_state_restored'])
            self.assertFalse(audit['rng_state_restored']);self.assertFalse(audit['training_only_state_restored'])
            self.assertEqual(float(model.frozen),11.)

    def test_protocol_weight_layer_config_head_and_fresh_engine_mismatch_reject_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            saved=TinyHREngine();saved.step();path,config,_=self.save(folder,saved)
            for case in ('protocol','weight','layer','config','head_shape','already_started'):
                engine=TinyHREngine(.01 if case=='weight' else .1,3 if case=='head_shape' else 2)
                if case=='layer':engine.student_layer=8
                if case=='already_started':engine.step()
                before=snapshot(engine);expected=canonical(config|{'other':'different'}) if case=='config' else config
                with self.subTest(case=case),self.assertRaises(ValueError):
                    restore_hr_latent_training_state(engine,path,allowed_root=Path(folder),
                        expected_protocol_sha256='f'*64 if case=='protocol' else PROTOCOL,expected_config=expected)
                self.assert_tree_equal(before,snapshot(engine))

    def test_corrupt_resume_state_is_rejected_before_student_optimizer_or_RNG_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            saved=TinyHREngine();saved.step();path,config,_=self.save(folder,saved)
            original=torch.load(path,map_location='cpu',weights_only=True)
            for case in ('old_Base','group_lr','group_order','missing_moment','moment_shape','negative_variance','moment_step','head_shape','head_spec','student_shape','RNG_device','RNG_state','time','Base_origin'):
                payload=copy.deepcopy(original);state=payload['training_only'];pid=state['optimizer']['param_groups'][0]['params'][0]
                if case=='old_Base':payload['training_only']={}
                elif case=='group_lr':state['optimizer']['param_groups'][0]['lr']=.3
                elif case=='group_order':state['optimizer_parameter_names'][0].reverse()
                elif case=='missing_moment':state['optimizer']['state'].pop(pid)
                elif case=='moment_shape':state['optimizer']['state'][pid]['exp_avg']=torch.zeros(1)
                elif case=='negative_variance':state['optimizer']['state'][pid]['exp_avg_sq'].fill_(-1)
                elif case=='moment_step':state['optimizer']['state'][pid]['step']=torch.tensor(99.)
                elif case=='head_shape':state['alignment_state']['projector.weight']=torch.zeros(1)
                elif case=='head_spec':state['alignment_spec']['patch_size']=[1,1,4]
                elif case=='student_shape':payload['trainable_state']['lrv_stem.weight']=torch.zeros(1)
                elif case=='RNG_device':state['RNG_devices']['noise']='cuda:0'
                elif case=='RNG_state':state['explicit_RNG']['noise']=torch.zeros(1,dtype=torch.uint8)
                elif case=='time':state['cumulative_training_seconds']=float('nan')
                else:state['matched_Base']['checkpoint_sha256']='b'*64
                torch.save(payload,path);fresh=TinyHREngine();before=snapshot(fresh)
                with self.subTest(case=case),self.assertRaises((ValueError,RuntimeError)):
                    restore_hr_latent_training_state(fresh,path,allowed_root=Path(folder),expected_protocol_sha256=PROTOCOL,expected_config=config)
                self.assert_tree_equal(before,snapshot(fresh))

    def test_publication_never_overwrites_and_does_not_accept_incomplete_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=TinyHREngine();engine.step();path,config,binding=self.save(folder,engine)
            before=path.read_bytes()
            with self.assertRaises(ValueError):self.save(folder,engine)
            self.assertEqual(path.read_bytes(),before)
            engine.optimizer.state.pop(next(iter(engine.optimizer.state)))
            with self.assertRaises(ValueError):self.save(folder,engine,'incomplete')
            self.assertFalse((Path(folder)/'incomplete').exists())

    def test_resume_rejects_missing_or_changed_sidecar_and_path_outside_allowed_root(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as other:
            engine=TinyHREngine();engine.step();path,config,_=self.save(folder,engine)
            sidecar=path.parent.parent/'config.json';saved=sidecar.read_bytes();sidecar.unlink()
            with self.assertRaises(ValueError):restore_hr_latent_training_state(TinyHREngine(),path,allowed_root=Path(folder),expected_protocol_sha256=PROTOCOL)
            sidecar.write_bytes(saved)
            with self.assertRaises(ValueError):restore_hr_latent_training_state(TinyHREngine(),path,allowed_root=Path(other),expected_protocol_sha256=PROTOCOL)


if __name__=='__main__':unittest.main()
