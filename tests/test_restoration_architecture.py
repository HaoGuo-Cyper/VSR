"""CPU random-Wan factory regression; no official weights or GPU are loaded."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from diffusers import WanTransformer3DModel

from vsr.checkpoints import _check_config, load_training_checkpoint, OFFICIAL_REPO_ID, OFFICIAL_REVISION
from vsr.infer_video import preview_training_config
from vsr.restoration_architecture import validate_restoration_architecture, METHOD_ARMS
from vsr.train import make_model


def config(variants,*,method=False,reference=False):
    result = {'seed':7,'model':{'repo_id':OFFICIAL_REPO_ID,'revision':OFFICIAL_REVISION,'local_dir':'not-opened'},
        'conditioning':{'rank':2,'alpha':2,'injection_layers':[0,1],'alignment_layer':1,
            'stem_channels':8,'projector_hidden':32,'teacher_dim':16,'hri_enabled':reference},
        'reference':{'enabled':reference,'layer':1,'stem_dim':8},
        'training':{'variants':list(variants),'gradient_checkpointing':False}}
    if method:result['restoration_architecture']='noref_base_v1'
    return result


def native(*args,**kwargs):
    return WanTransformer3DModel(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=12,
        in_channels=4,out_channels=4,text_dim=16,freq_dim=16,ffn_dim=48,num_layers=3,rope_max_seq_len=16),{}


def seal(value):
    value=copy.deepcopy(value)
    value['_sha256']=hashlib.sha256((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    return value


class RestorationArchitectureTests(unittest.TestCase):
    def test_legacy_four_variants_and_HRI_keep_projection_behavior(self):
        for reference in (False,True):
            cfg=config(('Base','Align','Relation','Both'),reference=reference)
            with patch('vsr.train.release'),patch('vsr.train.seed_all',side_effect=torch.manual_seed),\
                 patch.object(WanTransformer3DModel,'from_pretrained',side_effect=native):
                for variant in cfg['training']['variants']:
                    with self.subTest(reference=reference,variant=variant):
                        model,audit=make_model(cfg,variant,device='cpu')
                        self.assertEqual(model.projector is not None,variant!='Base')
                        self.assertEqual(hasattr(model,'reference_adapter'),reference)
                        self.assertEqual(audit['reference_enabled'],reference)
                        self.assertTrue(audit['fresh_official_initialization'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_method_true_names_use_Base_restoration_and_unchanged_model_identity(self):
        cfg=config(METHOD_ARMS,method=True)
        identity=copy.deepcopy(cfg['model'])
        with patch('vsr.train.release'),patch('vsr.train.seed_all',side_effect=torch.manual_seed),\
             patch.object(WanTransformer3DModel,'from_pretrained',side_effect=native):
            for variant in METHOD_ARMS:
                model,_=make_model(cfg,variant,device='cpu')
                self.assertIsNone(model.projector)
                self.assertFalse(hasattr(model,'reference_adapter'))
                self.assertTrue(all(name.startswith(('lrv_stem.','lrv_residuals.')) or '.lora_' in name
                                    for name,p in model.named_parameters() if p.requires_grad))
        self.assertEqual(cfg['model'],identity)
        _check_config(seal(cfg))

    def test_generic_loader_default_factory_and_CLI_preview_use_true_method_names(self):
        for arm in METHOD_ARMS:
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);cfg=seal(config((arm,),method=True))
                with patch('vsr.train.release'),patch('vsr.train.seed_all',side_effect=torch.manual_seed),\
                     patch.object(WanTransformer3DModel,'from_pretrained',side_effect=native):
                    original,_=make_model(cfg,arm,device='cpu')
                    with torch.no_grad():
                        for p in original.parameters():
                            if p.requires_grad:p.add_(.125)
                    state={name:p.detach().clone() for name,p in original.named_parameters() if p.requires_grad}
                    path=root/'new_run'/arm/'checkpoint_000001.pt';path.parent.mkdir(parents=True)
                    torch.save({'format':'vsr_training_v1','variant':arm,'step':1,'config':cfg,'trainable_state':state,
                        'training_only':{'EMA':{'deliberately_not_student':torch.full((2,),99.)},'student_projector':{'unused':True}}},path)
                    (path.parent.parent/'config.json').write_text(json.dumps(cfg))
                    preview=preview_training_config(path,allowed_root=root)
                    self.assertEqual(preview['variant'],arm)
                    restored,audit=load_training_checkpoint(path,device='cpu',allowed_root=root)
                self.assertIsNone(restored.projector)
                self.assertEqual(audit['variant'],arm)
                self.assertEqual(audit['inference_weights'],'student')
                self.assertFalse(audit['training_only_state_restored'])
                self.assertFalse(audit['EMA_weights_selected'])
                self.assertTrue(all(torch.equal(p,state[name]) for name,p in restored.named_parameters() if p.requires_grad))
                self.assertEqual(set(audit['restored_parameter_names']),set(state))
        self.assertFalse(torch.cuda.is_initialized())

    def test_missing_unknown_or_cross_family_marker_fails_before_model_load(self):
        cases=[config(('A2',)),config(('Base',),method=True)]
        unknown=config(('A2',),method=True);unknown['restoration_architecture']='unknown';cases.append(unknown)
        for key in ('reference','conditioning'):
            bad=config(('A2',),method=True)
            bad[key]['enabled' if key=='reference' else 'hri_enabled']=True
            cases.append(bad)
        for cfg in cases:
            variant=cfg['training']['variants'][0]
            with self.subTest(config=cfg),patch.object(WanTransformer3DModel,'from_pretrained') as load:
                with self.assertRaises(ValueError):make_model(cfg,variant,device='cpu')
                load.assert_not_called()
                with self.assertRaises(ValueError):_check_config(cfg)
        with self.assertRaisesRegex(ValueError,'canonical'):_check_config(config(('A2',),method=True))


if __name__=='__main__':unittest.main()
