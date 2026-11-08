"""CPU simulated fixed-teacher controller; no scientific or GPU updates."""
import copy
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from vsr import train as worker
import test_hr_latent_train as hr_test


class FixedTeacherControllerTests(unittest.TestCase):
    setUp=hr_test.HRLatentMetadataTests.setUp
    metadata=hr_test.HRLatentMetadataTests.metadata

    @contextmanager
    def fixture(self):
        with hr_test.HRLatentControllerTests.mocked_probe(self) as (run,p,events,factories,saved):
            p['schema']='fixed_teacher_training_protocol_v1';p['scientific_update_budget']=10240
            p['study']['arms']=[{'label':'baseline','family':'baseline','setting':'baseline','weight':0.},
                {'label':'vae_low','family':'hr_vae','setting':'low','weight':.01},
                {'label':'vae_high','family':'hr_vae','setting':'high','weight':.1},
                {'label':'dino_low','family':'hr_dino','setting':'low','weight':.02},
                {'label':'dino_high','family':'hr_dino','setting':'high','weight':.2}]
            cal=run/'calibration';cal.mkdir()
            worker.save(cal/'protocol.json',{'sources':p['sources'],'schema':'CPU_preflight_fixture'})
            worker.save(cal/'summary.json',{'status':'passed_fixed_teacher_preflight','sources':p['sources'],
                'protocol_sha256':worker.file_sha(cal/'protocol.json'),'scientific_updates':0,'diagnostic_optimizer_updates':27})
            p['calibration']={'protocol':worker.binding(cal/'protocol.json'),'summary':worker.binding(cal/'summary.json')}
            p['dino_cache']={'root':str(self.root/'dino_cache'),'seal':{'sha256':'b'*64}}
            worker.save(run/'status.json',{'status':'planned','protocol_sha256':'a'*64})
            reads=[];constructor=[];paired=[];kinds=[]
            class DINOReader:
                ids=[f'train_{i}' for i in range(1641)]
                def __init__(self,root,vae_root,*,expected_seal_sha256):
                    constructor.append((root,vae_root,expected_seal_sha256));events.append('DINO_loaded')
                def __len__(self):return len(self.ids)
                def __getitem__(self,index):
                    reads.append(index);return torch.tensor([[[float(index)]]])
                def close(self):events.append('DINO_closed')
            original_engine=worker._hr_engine;original_step=worker._hr_step
            def engine(protocol,weight,kind='vae'):
                result=original_engine(protocol,weight);result.teacher_kind=kind;kinds.append(kind);return result
            def step(engine,pair,text,scheduler,device,*,teacher_target=None):
                if engine.teacher_kind=='dino':
                    self.assertIsNotNone(teacher_target)
                    self.assertEqual(teacher_target.item(),engine.steps%1641)
                    paired.append((engine.weight,engine.steps,teacher_target.item()))
                else:self.assertIsNone(teacher_target)
                return original_step(engine,pair,text,scheduler,device)
            with patch('vsr.data.PreparedDINOData',DINOReader,create=True),\
                 patch.object(worker,'_hr_engine',engine),patch.object(worker,'_hr_step',step):
                yield run,p,events,factories,saved,reads,constructor,paired,kinds
            self.assertFalse(torch.cuda.is_initialized())

    def test_five_fresh_arms_have10240_rows20_checkpoints_and_only_DINO_reads_targets(self):
        with self.fixture() as (run,p,events,factories,saved,reads,constructor,paired,kinds):
            summary=worker.execute_hr(run,'a'*64)
            self.assertEqual(summary['schema'],'fixed_teacher_training_summary_v1')
            self.assertEqual(summary['scientific_updates'],10240);self.assertEqual(len(summary['arms']),5)
            self.assertEqual(summary['diagnostic_updates_in_this_training'],0)
            self.assertEqual(factories,[0.,.01,.1,.02,.2]);self.assertEqual(kinds,['vae']*3+['dino']*2)
            self.assertEqual(saved,[(step,weight) for weight in (0.,.01,.1,.02,.2) for step in (512,1024,1536,2048)])
            self.assertEqual(reads,[i%1641 for i in range(2048)]*2);self.assertEqual(len(paired),4096)
            self.assertEqual(constructor,[(p['dino_cache']['root'],p['cache']['root'],'b'*64)])
            self.assertEqual(events.count('DINO_closed'),1)
            self.assertLess(events.index('DINO_closed'),events.index('lease_release'))
            for arm in p['study']['arms']:
                record=summary['arms'][arm['label']]
                self.assertEqual((record['family'],record['setting']),(arm['family'],arm['setting']))
                rows=[worker.json.loads(line) for line in Path(record['full_training_log']['path']).read_text().splitlines()]
                self.assertEqual(len(rows),2048);self.assertEqual([r['step'] for r in rows],list(range(1,2049)))
                self.assertTrue(all(r['family']==arm['family'] and r['setting']==arm['setting'] for r in rows))
                self.assertEqual(record['initial_student_hashes'],{'CPU_value':0.})
                self.assertEqual(record['final_student_hashes'],{'CPU_value':2048.})
            self.assertFalse((run/'probe').exists())

    def test_DINO_resume_keeps_completed_three_arms_and_paired_target_indices(self):
        with self.fixture() as (run,p,events,factories,saved,reads,constructor,paired,kinds):
            original=worker._hr_step
            def fail(engine,*args,**kwargs):
                if engine.teacher_kind=='dino' and engine.weight==.02 and engine.steps==515:
                    raise RuntimeError('CPU DINO interruption after512')
                return original(engine,*args,**kwargs)
            with patch.object(worker,'_hr_step',fail):
                with self.assertRaisesRegex(RuntimeError,'CPU DINO interruption'):worker.execute_hr(run,'a'*64)
            failure=worker.read(run/'failure_attempt_000.json')
            self.assertEqual(failure['completed_arms'],['baseline','vae_low','vae_high'])
            self.assertEqual(failure['effective_updates_completed_arms'],6144)
            self.assertEqual(failure['current_arm_steps_returned'],515)
            log=run/'arm_runs/dino_low/training.jsonl';prefix=log.read_bytes()
            self.assertEqual(len(prefix.splitlines()),515)
            summary=worker.execute_hr(run,'a'*64,resume=True)
            self.assertEqual(summary['scientific_updates'],10240)
            self.assertEqual((log.parent/'interrupted_log_attempt_001.jsonl').read_bytes(),prefix)
            rows=[worker.json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([row['step'] for row in rows],list(range(1,2049)))
            self.assertTrue(all(row['family']=='hr_dino' and row['setting']=='low' for row in rows))
            self.assertEqual([cp['step'] for cp in summary['arms']['dino_low']['checkpoint_files']],[512,1024,1536,2048])
            self.assertEqual(reads,[i%1641 for i in range(516)]+[i%1641 for i in range(512,2048)]+[i%1641 for i in range(2048)])
            self.assertEqual(events.count('DINO_closed'),2)
            self.assertEqual(events.count('training_restored'),1)

    def test_changed_preflight_bytes_rejects_before_GPU_lease_or_any_model(self):
        with self.fixture() as (run,p,events,factories,saved,reads,constructor,paired,kinds):
            Path(p['calibration']['summary']['path']).write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError,'Asset checksum changed'):worker.execute_hr(run,'a'*64)
            self.assertEqual(factories,[]);self.assertEqual(constructor,[]);self.assertNotIn('lease_enter',events)
            self.assertFalse((run/'attempt_000.json').exists())

    def test_wrong_preflight_sources_reject_even_with_self_consistent_new_binding(self):
        with self.fixture() as (run,p,events,factories,saved,reads,constructor,paired,kinds):
            path=Path(p['calibration']['summary']['path']);summary=worker.read(path)
            summary['sources']={'wrong':'implementation'};path.write_text(worker.json.dumps(summary))
            p['calibration']['summary']=worker.binding(path)
            with self.assertRaisesRegex(ValueError,'same-source fixed-teacher preflight'):worker.execute_hr(run,'a'*64)
            self.assertEqual(factories,[]);self.assertNotIn('lease_enter',events)
            self.assertFalse((run/'attempt_000.json').exists())


if __name__=='__main__':unittest.main()
