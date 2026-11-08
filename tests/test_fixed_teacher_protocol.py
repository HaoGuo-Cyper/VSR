"""CPU metadata/provenance gates for fixed teacher calibration and five arms."""
import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from vsr import train as worker
import test_hr_latent_train as hr_test


class FixedTeacherProtocolTests(unittest.TestCase):
    def setUp(self):
        hr_test.HRLatentMetadataTests.setUp(self)
        self.study.update(schema='fixed_teacher_design_v1',head_depth=1,dino_cache_root=str(self.root/'DINO'))
        dino=self.root/'DINO';dino.mkdir()
        worker.save(dino/'manifest.json',{'teacher':{'fixture':'frozen HR DINOv2-B/14'}})
        worker.save(dino/'seal.json',{'fixture':'CPU metadata'})
        worker.save(dino/'completion_receipt.json',{'status':'completed','fixture':'synthetic metadata only'})
        assets={}
        for name in ('validation_seal','validation_dataset','test_cohort_plan','alexnet','linear'):
            path=self.root/f'{name}.json';worker.save(path,{'fixture':name});assets[name]=worker.binding(path)
        self.study['evaluation']={k:assets[k] for k in ('validation_seal','validation_dataset','test_cohort_plan')}
        self.study['metric_weights']={k:assets[k] for k in ('alexnet','linear')}
        self.study['dinov2']={'source_dir':str(self.root/'DINOsource'),'weights':str(self.root/'DINOweights'),
            'frame_batch_size':8}
        self.path.write_text(worker.json.dumps(self.study));self.closed=[]

    @contextmanager
    def metadata(self):
        ids=self.cache['selection']['training_ids'];closed=self.closed
        class DINOReader:
            def __init__(self,*args,**kwargs):self.ids=ids
            def close(self):closed.append(True)
        with hr_test.HRLatentMetadataTests.metadata(self),\
             patch('vsr.data.PreparedDINOData',DINOReader,create=True),\
             patch.object(worker,'_source_bindings',return_value={'train.py':{'sha256':'1'*64,'path':'/CPU_fixture','bytes':0}}):
            yield

    def finalize(self):
        with self.metadata():calibration=worker.collect_ft_calibration_protocol(self.path)
        run=self.root/'runs/calibration';run.mkdir(parents=True)
        worker.save(run/'protocol.json',calibration)
        psha=worker.file_sha(run/'protocol.json')
        summary={'status':'passed_fixed_teacher_preflight','scientific_updates':0,'diagnostic_optimizer_updates':27,
            'sources':calibration['sources'],'protocol_sha256':psha,
            'calibrations':{'vae':{'weights':[.012,.12]},'dino':{'weights':[.034,.34]}}}
        worker.save(run/'summary.json',summary)
        worker.save(run/'status.json',{'status':'completed','protocol_sha256':psha,
            'summary_sha256':worker.file_sha(run/'summary.json'),'scientific_updates':0})
        study=copy.deepcopy(self.study);study.update(schema='fixed_teacher_study_v1',calibration_report=str(run/'summary.json'))
        study['arms']=[{'label':'baseline','family':'baseline','setting':'baseline','weight':0.},
            {'label':'vae_low','family':'hr_vae','setting':'low','weight':.012},
            {'label':'vae_high','family':'hr_vae','setting':'high','weight':.12},
            {'label':'dino_low','family':'hr_dino','setting':'low','weight':.034},
            {'label':'dino_high','family':'hr_dino','setting':'high','weight':.34}]
        finalized=self.root/'finalized.json';worker.save(finalized,study)
        return run,finalized,study

    def test_calibration_prepare_is_metadata_only_zero_science27_diagnostic_and_fixed_train_sigmas(self):
        run=self.root/'runs/prepared'
        with self.metadata():
            p=worker.collect_ft_calibration_protocol(self.path)
            planned=worker.prepare_ft_calibration(self.path,run)
            with self.assertRaisesRegex(ValueError,'new calibration'):worker.prepare_ft_calibration(self.path,run)
        self.assertEqual(p['schema'],'fixed_teacher_calibration_protocol_v1')
        self.assertEqual(p['scientific_update_budget'],0);self.assertEqual(p['diagnostic_optimizer_budget'],27)
        self.assertEqual(p['calibration_training_index'],0);self.assertEqual(p['calibration_sigmas'],[.2,.5,.8])
        self.assertEqual(p['relative_auxiliary_gradient_strengths'],[.1,1.])
        self.assertFalse(p['selection_uses_validation_or_test']);self.assertFalse(planned['GPU_loaded'])
        self.assertGreater(len(self.closed),0);self.assertFalse(torch.cuda.is_initialized())

    def test_final_protocol_accepts_only_calibrated_five_arms_and_bound_completed_preflight(self):
        run,path,study=self.finalize()
        with self.metadata():p=worker.collect_fixed_teacher_protocol(path)
        self.assertEqual(p['schema'],'fixed_teacher_training_protocol_v1');self.assertEqual(p['scientific_update_budget'],10240)
        self.assertEqual(p['study']['arms'],study['arms'])
        self.assertEqual(p['calibration']['protocol'],worker.binding(run/'protocol.json'))
        self.assertEqual(p['calibration']['summary'],worker.binding(run/'summary.json'))
        changed=copy.deepcopy(study);changed['arms'][-1]['weight']=.5;path.write_text(worker.json.dumps(changed))
        with self.metadata(),self.assertRaisesRegex(ValueError,'Candidate settings'):worker.collect_fixed_teacher_protocol(path)
        self.assertFalse(torch.cuda.is_initialized())

    def test_passed_looking_summary_from_failed_calibration_is_rejected(self):
        run,path,study=self.finalize()
        worker.save(run/'status.json',{'status':'failed_calibration','protocol_sha256':worker.file_sha(run/'protocol.json'),
            'summary_sha256':worker.file_sha(run/'summary.json')},replace=True)
        worker.save(run/'failure.json',{'status':'failed_calibration','error':'cleanup failed after summary publication'})
        with self.metadata(),self.assertRaises(ValueError):worker.collect_fixed_teacher_protocol(path)
        self.assertFalse(torch.cuda.is_initialized())

    def test_calibration_completion_summary_hash_and_source_changes_are_rejected(self):
        run,path,study=self.finalize();state=worker.read(run/'status.json');state['summary_sha256']='0'*64
        worker.save(run/'status.json',state,replace=True)
        with self.metadata(),self.assertRaises(ValueError):worker.collect_fixed_teacher_protocol(path)
        state['summary_sha256']=worker.file_sha(run/'summary.json');worker.save(run/'status.json',state,replace=True)
        summary=worker.read(run/'summary.json');summary['sources']={'wrong':True};(run/'summary.json').write_text(worker.json.dumps(summary))
        state['summary_sha256']=worker.file_sha(run/'summary.json');worker.save(run/'status.json',state,replace=True)
        with self.metadata(),self.assertRaises(ValueError):worker.collect_fixed_teacher_protocol(path)

    def test_changed_calibration_source_inputs_stop_before_GPU_admission(self):
        run=self.root/'runs/prepared'
        with self.metadata():planned=worker.prepare_ft_calibration(self.path,run)
        with patch('vsr.data.PreparedDINOData',create=True),patch('vsr.teacher.FrozenDINOv2',create=True),\
             patch.object(worker,'collect_ft_calibration_protocol',return_value={'changed':True}),\
             patch('vsr.checkpoints.gpu_lease') as lease:
            with self.assertRaisesRegex(ValueError,'source/input protocol changed'):
                worker.execute_ft_calibration(run,planned['protocol_sha256'])
            lease.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
