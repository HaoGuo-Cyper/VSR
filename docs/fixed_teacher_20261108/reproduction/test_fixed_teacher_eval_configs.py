"""CPU synthetic completed-training/config tests; no model tensors or scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fixed_teacher_eval_configs as builder
from vsr import evaluate as ev
from test_fixed_teacher_evaluate import FixedTrainingFixture,write


class Fixture(FixedTrainingFixture):
    def __init__(self, root):
        super().__init__(root)
        for name in ('official','deps'):(root/name).mkdir()
        for name,value in [('val_seal.json',{'split':'val'}),('val_dataset.json',{'split':'val'}),
                           ('test_plan.json',{'split':'test'}),('admission.json',{'expected_gpu_uuid':'GPU-synthetic'})]:
            write(root/name,value)
        for name in ('text.pt','alexnet.pth','linear.pth'):(root/name).write_bytes(name.encode())
        self.basecfg['data']['text_cache']=ev.binding(root/'text.pt')
        write(root/'starting/config.json',self.basecfg)
        for label in self.summary['arms']:write(self.run/'arm_runs'/label/'config.json',self.basecfg)
        self.study.update(model_directory=str(root/'official'),gpu_admission=str(root/'admission.json'),
            gpu_uuid='GPU-synthetic',device='cuda:0',gpu_lock=str(root/'shared.lock'),allowed_base_root=str(root/'starting'),
            metric_weights={'alexnet':ev.binding(root/'alexnet.pth'),'linear':ev.binding(root/'linear.pth')},
            evaluation={'validation_seal':ev.binding(root/'val_seal.json'),'validation_dataset':ev.binding(root/'val_dataset.json'),
                'test_cohort_plan':ev.binding(root/'test_plan.json'),'test_pixels_after_selection':True,
                'test_prior_exposure':'historical_models_previously_evaluated'})
        write(root/'study.json',self.study);self.study_file=ev.binding(root/'study.json')
        self.protocol.update(study=self.study,study_file=self.study_file,device=self.study['device'],
            gpu_lock=self.study['gpu_lock'],admission=ev.read(root/'admission.json'),admission_file=ev.binding(root/'admission.json'))
        write(self.run/'protocol.json',self.protocol);self.psha=ev.sha(self.run/'protocol.json')
        self.summary['protocol_sha256']=self.psha;self.publish_summary()

    def build(self, phase='val', **kwargs):
        with patch.object(ev,'_hr_sources',return_value=self.sources),patch('vsr.checkpoints._check_config'):
            return builder.build_config(ev,phase=phase,study_file=self.root/'study.json',training_run=self.run,
                run_root=self.root/'evaluations',dependency_root=self.root/'deps',**kwargs)

    def selection(self):
        val=self.build();vrun=self.root/'validation';vrun.mkdir()
        for name in ('protocol.json','status.json','summary.json','observations.json'):
            write(vrun/name,{'synthetic_metadata':True})
        selected=[r for r in val['roles'] if r['update']==2048 and r['arm'] in ('baseline','vae_low','dino_high')]
        receipt={'schema':'fixed_teacher_validation_selection_v1','study_file':self.study_file,
            'validation_run':str(vrun),'selected_after_complete_validation':True,'test_pixels_read_by_selection':False,
            'test_cohort_plan':self.study['evaluation']['test_cohort_plan'],'selected_roles':selected,
            **ev._ft_test_access_policy(self.study['evaluation']),
            **{k:ev.binding(vrun/name) for k,name in [('validation_protocol','protocol.json'),
                ('validation_status','status.json'),('validation_summary','summary.json'),('validation_observations','observations.json')]}}
        path=self.root/'selection.json';write(path,receipt)
        dataset=self.root/'test_dataset.json';write(dataset,{'selection_receipt':ev.binding(path),
            'cohort_seal':self.study['evaluation']['test_cohort_plan'],'materialization':{'selection_frozen_before_decode':True}})
        return {'validation_run':vrun,'selection_receipt':path,'test_dataset':dataset},receipt


class EvaluationConfigTests(unittest.TestCase):
    def test_validation_all16_roles_use_current_bound_cohort_and_real_CP_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));cfg=f.build()
            self.assertEqual(len(cfg['roles']),16);self.assertEqual(cfg['configuration_generation']['expected_restores'],1024)
            self.assertEqual(cfg['seal'],f.study['evaluation']['validation_seal']['path'])
            self.assertEqual(cfg['dataset'],f.study['evaluation']['validation_dataset']['path'])
            self.assertEqual(cfg['roles'][0]['update'],0)
            self.assertTrue(all(r['update'] in (0,512,1024,2048) for r in cfg['roles']))
            self.assertEqual([r['cumulative_training_seconds'] for r in cfg['roles'][1:4]],[512.,1024.,2048.])
            self.assertFalse(cfg['configuration_generation']['GPU_work_performed'])

    def test_test_three_roles_are_exact_frozen_selection_without_reading_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));kwargs,receipt=f.selection();cfg=f.build('test',**kwargs)
            self.assertEqual(cfg['roles'],receipt['selected_roles']);self.assertEqual(len(cfg['roles']),3)
            self.assertEqual(cfg['configuration_generation']['expected_restores'],192)
            self.assertEqual(cfg['selection_receipt'],ev.binding(kwargs['selection_receipt']))
            self.assertTrue(all(r['step']==r['update']==2048 for r in cfg['roles']))

    def test_incomplete_training_cannot_generate_configs(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));state=ev.read(f.run/'status.json');state['status']='running';write(f.run/'status.json',state)
            with self.assertRaisesRegex(ValueError,'completion hash chain'):f.build()

    def test_wrong_selection_CP_or_early_materialization_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));kwargs,receipt=f.selection()
            dataset=ev.read(kwargs['test_dataset']);dataset['materialization']['selection_frozen_before_decode']=False
            write(kwargs['test_dataset'],dataset)
            with self.assertRaisesRegex(ValueError,'not materialized'):f.build('test',**kwargs)
            receipt['selected_roles'][1]['checkpoint']=receipt['selected_roles'][2]['checkpoint']
            write(kwargs['selection_receipt'],receipt);dataset['selection_receipt']=ev.binding(kwargs['selection_receipt'])
            dataset['materialization']['selection_frozen_before_decode']=True;write(kwargs['test_dataset'],dataset)
            with self.assertRaises(ValueError):f.build('test',**kwargs)

    def test_frozen_input_or_admission_changes_are_rejected(self):
        for filename in ('linear.pth','admission.json'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as folder:
                f=Fixture(Path(folder));(f.root/filename).write_bytes(b'changed')
                with self.assertRaises(ValueError):f.build()

    def test_output_is_new_immutable_JSON_and_val_rejects_test_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(Path(folder));cfg=f.build();path=builder.write_config(f.root/'configs/val.json',cfg)
            self.assertEqual(ev.read(path),cfg)
            with self.assertRaisesRegex(ValueError,'immutable'):builder.write_config(path,cfg)
            with self.assertRaisesRegex(ValueError,'does not consume'):f.build(validation_run=f.root/'validation')


if __name__=='__main__':unittest.main(verbosity=2)
