"""Synthetic CPU checkpoint fixtures, with no official weights or GPU."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vsr.checkpoints import load_training_checkpoint, OFFICIAL_REPO_ID, OFFICIAL_REVISION
from vsr.method_checkpoint import build_method_payload, save_method_checkpoint
from vsr.self_distillation import TrainableEMA


class TinyRestoration(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.weight = nn.Parameter(torch.eye(3),requires_grad=False)
        self.backbone.lora_A = nn.Parameter(torch.full((3,3),.1))
        self.lrv_stem = nn.Linear(3,3)
        self.lrv_residuals = nn.ModuleDict({'0':nn.Linear(3,3,bias=False)})
        self.projector = None
    def forward(self,value):
        return value @ (self.backbone.weight+self.backbone.lora_A).T + self.lrv_residuals['0'](self.lrv_stem(value))


def config(arm):
    value = {'seed':20261102,'restoration_architecture':'noref_base_v1',
        'model':{'repo_id':OFFICIAL_REPO_ID,'revision':OFFICIAL_REVISION,'local_dir':'/unused/official/model'},
        'data':{'num_frames':33,'temporal_stride':1,'height':128,'width':224,'scale':4},
        'training':{'seed':20261102,'variants':[arm],'updates':2048},
        'reference':{'enabled':False},'conditioning':{'hri_enabled':False,'rank':8,'alpha':8,
            'injection_layers':[0,5,9],'alignment_layer':9,'stem_channels':128,
            'projector_hidden':1024,'teacher_dim':768}}
    value['_sha256'] = hashlib.sha256((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    return value


def engine(arm='A2'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = TinyRestoration()
        alignment = SimpleNamespace(projector=nn.Sequential(nn.Linear(3,5),nn.SiLU(),nn.Linear(5,3))) if arm=='A2' else None
    ema = TrainableEMA(model) if arm=='A2' else None
    parameters = [p for p in model.parameters() if p.requires_grad]
    if alignment is not None:parameters += list(alignment.projector.parameters())
    optimizer = torch.optim.AdamW(parameters,lr=.001,foreach=False)
    # One synthetic update provides real AdamW moments, never task training.
    for parameter in parameters:parameter.grad = torch.full_like(parameter,.25)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    if ema is not None:ema.update(model)
    streams = SimpleNamespace(**{name:torch.Generator().manual_seed(100+index)
        for index,name in enumerate(('noise','timestep','mask','corruption'))})
    audit = {'variant':'Base','step':8192,'checkpoint_sha256':'a'*64,
        'checkpoint_path':'/fixture/new/runs/Base/checkpoint_008192.pt','optimizer_state_restored':False,
        'scheduler_state_restored':False,'rng_state_restored':False}
    base_config = config(arm)
    base_config.pop('_sha256');base_config.pop('restoration_architecture')
    base_config['training'].update(variants=['Base'],updates=8192)
    base_config['_sha256'] = hashlib.sha256((json.dumps(base_config,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    audit['config'] = base_config
    return SimpleNamespace(model=model,arm=arm,steps=1,training_seed=20261102,
        alignment=alignment,ema=ema,optimizer=optimizer,rng=streams,base_load_audit=audit)


def factory(cfg,variant):
    return TinyRestoration(),{'official_revision':OFFICIAL_REVISION,
        'fresh_official_initialization':True,'legacy_state_loaded':False}


class MethodCheckpointTests(unittest.TestCase):
    def test_three_true_arm_names_roundtrip_student_only_and_frozen_unchanged(self):
        for arm in ('flow_only','C1','A2'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);current=engine(arm);cfg=config(arm)
                path=root/'new_run'/arm/'checkpoint_000001.pt'
                receipt=save_method_checkpoint(path,current,cfg,allowed_root=root)
                payload=torch.load(path,map_location='cpu',weights_only=True)
                self.assertEqual(payload['variant'],arm)
                self.assertEqual(payload['format'],'vsr_training_v1')
                self.assertEqual(payload['config']['model'],cfg['model'])
                expected={name for name,p in current.model.named_parameters() if p.requires_grad}
                self.assertEqual(set(payload['trainable_state']),expected)
                self.assertFalse(any('projector' in key or 'EMA' in key for key in expected))
                self.assertEqual(receipt['student_parameter_names'],sorted(expected))
                self.assertEqual(receipt['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(payload['training_only']['matched_Base'],current.base_load_audit)
                self.assertEqual(set(payload['training_only']['explicit_RNG']),{'noise','timestep','mask','corruption'})
                self.assertEqual(payload['training_only']['EMA'] is not None,arm=='A2')
                self.assertEqual(payload['training_only']['student_projector'] is not None,arm=='A2')
                restored,audit=load_training_checkpoint(path,device='cpu',allowed_root=root,model_factory=factory)
                value=torch.arange(6).reshape(2,3).float()
                torch.testing.assert_close(restored(value),current.model(value),atol=0,rtol=0)
                self.assertTrue(torch.equal(restored.backbone.weight,torch.eye(3)))
                self.assertEqual(audit['variant'],arm)
                self.assertFalse(audit['optimizer_state_restored'])
                self.assertFalse(audit['training_only_state_restored'])
                self.assertFalse(audit['EMA_weights_selected'])
                self.assertEqual(audit['inference_weights'],'student')
                self.assertFalse(torch.cuda.is_initialized())

    def test_training_only_mutations_never_change_inference_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);current=engine();path=root/'run/A2/checkpoint_000001.pt'
            save_method_checkpoint(path,current,config('A2'),allowed_root=root)
            payload=torch.load(path,map_location='cpu',weights_only=True)
            for value in payload['training_only']['EMA']['shadow'].values():value.fill_(1234.)
            for value in payload['training_only']['student_projector'].values():value.fill_(-4321.)
            payload['training_only']['optimizer']={'intentionally_not_resumed':True}
            # A deliberate fixture rewrite verifies the reader ignores all
            # training-only alternatives; production writer never overwrites.
            torch.save(payload,path)
            actual,audit=load_training_checkpoint(path,device='cpu',allowed_root=root,model_factory=factory)
            self.assertTrue(all(torch.equal(p,dict(current.model.named_parameters())[name])
                                for name,p in actual.named_parameters()))
            self.assertFalse(audit['EMA_weights_selected'])

    def test_payload_owns_every_tensor_and_does_not_advance_RNG(self):
        current=engine();states={name:getattr(current.rng,name).get_state() for name in ('noise','timestep','mask','corruption')}
        default=torch.random.get_rng_state().clone()
        payload=build_method_payload(current,config('A2'))
        for value in payload['trainable_state'].values():value.zero_()
        self.assertTrue(any(bool(p.detach().abs().sum()) for p in current.model.parameters() if p.requires_grad))
        for value in payload['training_only']['EMA']['shadow'].values():value.zero_()
        self.assertTrue(any(bool(x.abs().sum()) for x in current.ema.state_dict()['shadow'].values()))
        for name,state in payload['training_only']['explicit_RNG'].items():
            state.zero_();self.assertTrue(torch.equal(getattr(current.rng,name).get_state(),states[name]))
        self.assertTrue(torch.equal(default,torch.random.get_rng_state()))

    def test_canonical_marker_step_seed_and_named_parameter_contracts_reject_before_output(self):
        changes = [lambda e,c:c.pop('_sha256'),lambda e,c:c.update(seed=99),
            lambda e,c:c.pop('restoration_architecture'),lambda e,c:setattr(e,'steps',0),
            lambda e,c:setattr(e,'training_seed',99),lambda e,c:e.model.register_parameter('unrelated',nn.Parameter(torch.ones(1))),
            lambda e,c:e.ema._shadow.update({e.ema.names[-1]:torch.zeros(99)}),
            lambda e,c:e.ema._shadow[e.ema.names[-1]].fill_(float('nan')),
            lambda e,c:e.alignment.projector[0].weight.requires_grad_(False),
            lambda e,c:e.alignment.projector.double()]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);current=engine();cfg=config('A2');change(current,cfg)
                with self.assertRaises(ValueError):save_method_checkpoint(root/'run/A2/checkpoint_000001.pt',current,cfg,allowed_root=root)
                self.assertEqual(list(root.iterdir()),[])

    def test_no_overwrite_or_config_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);current=engine();cfg=config('A2');path=root/'run/A2/checkpoint_000001.pt'
            save_method_checkpoint(path,current,cfg,allowed_root=root)
            before=path.read_bytes();config_before=(root/'run/config.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'overwrite'):save_method_checkpoint(path,current,cfg,allowed_root=root)
            self.assertEqual(path.read_bytes(),before)
            next_path=root/'other/A2/checkpoint_000001.pt';next_path.parent.mkdir(parents=True)
            (root/'other/config.json').write_text('{"unrelated":"existing"}\n')
            with self.assertRaisesRegex(ValueError,'Existing run config'):save_method_checkpoint(next_path,current,cfg,allowed_root=root)
            self.assertEqual((root/'other/config.json').read_text(),'{"unrelated":"existing"}\n')
            self.assertFalse(next_path.exists());self.assertEqual((root/'run/config.json').read_bytes(),config_before)

    def test_wrong_matched_Base_seed_geometry_family_or_method_geometry_are_rejected(self):
        changes = [lambda e,c:e.base_load_audit.pop('config'),
            lambda e,c:e.base_load_audit['config']['training'].update(seed=20261103),
            lambda e,c:e.base_load_audit['config'].update(seed=20261103),
            lambda e,c:e.base_load_audit['config']['training'].update(variants=['Align']),
            lambda e,c:e.base_load_audit['config']['training'].update(updates=8),
            lambda e,c:e.base_load_audit['config']['model'].update(revision='wrong'),
            lambda e,c:e.base_load_audit['config']['data'].update(temporal_stride=4),
            lambda e,c:e.base_load_audit['config']['reference'].update(enabled=True),
            lambda e,c:e.base_load_audit['config']['conditioning'].update(hri_enabled=True),
            lambda e,c:c['data'].update(num_frames=17)]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);current=engine();cfg=config('A2');change(current,cfg)
                # Keep altered metadata internally canonical so these fixtures
                # exercise seed/geometry/family checks, rather than SHA alone.
                base_cfg=current.base_load_audit.get('config')
                if isinstance(base_cfg,dict):
                    base_cfg.pop('_sha256')
                    base_cfg['_sha256']=hashlib.sha256((json.dumps(base_cfg,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
                if cfg['data']['num_frames'] != 33:
                    cfg.pop('_sha256')
                    cfg['_sha256']=hashlib.sha256((json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
                with self.assertRaises(ValueError):save_method_checkpoint(root/'run/A2/checkpoint_000001.pt',current,cfg,allowed_root=root)
                self.assertEqual(list(root.iterdir()),[])

    def test_serialization_failure_and_competing_writer_leave_no_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);current=engine();cfg=config('A2');path=root/'run/A2/checkpoint_000001.pt'
            with patch('vsr.method_checkpoint.torch.save',side_effect=RuntimeError('fixture serialize failure')):
                with self.assertRaisesRegex(RuntimeError,'fixture serialize failure'):save_method_checkpoint(path,current,cfg,allowed_root=root)
            self.assertFalse(path.exists());self.assertEqual(list(root.rglob('*.partial')),[])
            import os
            original=os.link
            def compete(source,destination):
                if Path(destination)==path:path.write_bytes(b'competing-complete-file')
                return original(source,destination)
            with patch('vsr.method_checkpoint.os.link',side_effect=compete):
                with self.assertRaises(FileExistsError):save_method_checkpoint(path,current,cfg,allowed_root=root)
            self.assertEqual(path.read_bytes(),b'competing-complete-file')
            self.assertEqual(list(root.rglob('*.partial')),[])

    def test_wrong_arm_path_escape_and_existing_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);current=engine();cfg=config('A2')
            for path in (root/'run/Base/checkpoint_000001.pt',root/'run/A2/checkpoint_000002.pt',root.parent/'escaped/A2/checkpoint_000001.pt'):
                with self.assertRaises(ValueError):save_method_checkpoint(path,current,cfg,allowed_root=root)
            target=root/'run/A2/checkpoint_000001.pt';target.parent.mkdir(parents=True)
            target.symlink_to(root/'missing')
            with self.assertRaisesRegex(ValueError,'symlink'):save_method_checkpoint(target,current,cfg,allowed_root=root)
            self.assertTrue(target.is_symlink());self.assertFalse((root/'missing').exists())

    def test_ancestor_allowed_root_cannot_write_resolved_historical_or_legacy_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);old=root/'historical';legacy=root/'legacy'
            for protected in (old,legacy):
                protected.mkdir();(protected/'sentinel.txt').write_bytes(b'immutable fixture asset')
            (root/'old_alias').symlink_to(old,target_is_directory=True)
            current=engine();cfg=config('A2')
            with patch('vsr.method_checkpoint.HISTORICAL',old),patch('vsr.method_checkpoint.LEGACY',legacy):
                for source in (old,legacy,root/'old_alias'):
                    path=source/'new_run/A2/checkpoint_000001.pt'
                    with self.subTest(source=source),self.assertRaisesRegex(ValueError,'historical or legacy'):
                        save_method_checkpoint(path,current,cfg,allowed_root=root)
                for protected in (old,legacy):
                    self.assertEqual([p.name for p in protected.iterdir()],['sentinel.txt'])
                    self.assertEqual((protected/'sentinel.txt').read_bytes(),b'immutable fixture asset')
                # A broad root still permits its explicitly new, unprotected
                # run; the rejection above is specific to protected targets.
                receipt=save_method_checkpoint(root/'new_run/A2/checkpoint_000001.pt',current,cfg,allowed_root=root)
                self.assertTrue(Path(receipt['path']).is_file())

    def test_restoration_model_and_conditioning_cannot_drift_from_matched_Base(self):
        changes = [lambda c:c['model'].update(local_dir='/different/official/location'),
            lambda c:c['model'].update(extra_architecture_knob=True),
            lambda c:c['conditioning'].update(alpha=16),lambda c:c['conditioning'].update(rank=16),
            lambda c:c['conditioning'].update(injection_layers=[0,5,10]),
            lambda c:c['conditioning'].update(stem_channels=64),
            lambda c:c['conditioning'].update(alignment_layer=8),
            lambda c:c['conditioning'].update(projector_hidden=2048),
            lambda c:c['conditioning'].update(teacher_dim=1024)]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);current=engine();cfg=config('A2');change(cfg)
                cfg.pop('_sha256')
                cfg['_sha256']=hashlib.sha256((json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
                with self.assertRaisesRegex(ValueError,'complete matched Base'):
                    save_method_checkpoint(root/'run/A2/checkpoint_000001.pt',current,cfg,allowed_root=root)
                self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':unittest.main()
