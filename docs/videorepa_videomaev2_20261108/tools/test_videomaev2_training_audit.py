"""Only synthetic completed training fixtures; no Wan/teacher forward or GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from test_videomaev2_checkpoints import TinyEngine,config_for,teacher_binding
from test_hr_latent_checkpoints import canonical
from vsr.checkpoints import save_videomaev2_checkpoint

HELPER=Path('/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/videomaev2_training_audit.py')
spec=importlib.util.spec_from_file_location('videomaev2_training_audit',HELPER)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.run=self.root/'run';self.source=self.root/'source';self.output=self.root/'deliveries/audit'
        self.run.mkdir();self.source.mkdir();sources={}
        for name in audit.CORE:
            p=self.source/'vsr'/name;p.parent.mkdir(exist_ok=True);p.write_text('#synthetic '+name+'\n');sources[name]=audit.binding(p)
        engine=TinyEngine('baseline');initial={n:p.detach().clone() for n,p in engine.model.named_parameters() if p.requires_grad}
        self.base=self.root/'origin/Base/checkpoint_008192.pt';self.base.parent.mkdir(parents=True)
        baseconfig=engine.base_load_audit['config'];torch.save({'format':'vsr_training_v1','variant':'Base','step':8192,'config':baseconfig,'trainable_state':initial},self.base)
        write(self.base.parent.parent/'config.json',baseconfig);base_digest=audit.trainable_digest(torch,initial)
        teacher=teacher_binding();rgb={'records':1641,'sealed':'synthetic'}
        self.study={'schema':'vsr_videomaev2_study_v1','mode':'training','updates':2048,'training_seed':20261102,
            'student_layer':17,'margin':.1,'normalization_eps':1e-12,'online_teacher':True,
            'checkpoint_updates':list(audit.STEPS),'arms':[{'label':'baseline','weight':0.},{'label':'videomaev2','weight':.5}]}
        write(self.root/'study.json',self.study)
        cal=self.root/'preflight';write(cal/'protocol.json',{'schema':'vsr_videomaev2_preflight_protocol_v1','sources':sources,'teacher_binding':teacher,'rgb_assets':rgb})
        calp=audit.binding(cal/'protocol.json');write(cal/'summary.json',{'status':'passed_videomaev2_preflight','sources':sources,
            'protocol_sha256':calp['sha256'],'scientific_updates':0,'diagnostic_optimizer_updates':26,'teacher_forwards':31,
            'gradient_probe_cases':15,'gradient_VJP_calls':30,'target_viability':{'eligible':True},'zero_weight_exact':True,
            'resume_reports':{'videomaev2':{'exact_full_engine_state':True}}})
        cals=audit.binding(cal/'summary.json');write(cal/'status.json',{'status':'completed','protocol_sha256':calp['sha256'],'summary_sha256':cals['sha256']})
        self.protocol={'schema':'vsr_videomaev2_training_protocol_v1','scientific_update_budget':4096,'study':self.study,
            'study_file':audit.binding(self.root/'study.json'),'sources':sources,'device':'cpu',
            'dependencies':{'torch':{'version':str(torch.__version__)}},'teacher_binding':teacher,'rgb_assets':rgb,
            'base':{'checkpoint':audit.binding(self.base),'config':baseconfig,'config_file':audit.binding(self.base.parent.parent/'config.json')},
            'cache':{'selection':{'training_ids':[f'train_{i}' for i in range(1641)]}},
            'calibration':{'protocol':calp,'summary':cals,'status_file':audit.binding(cal/'status.json')}}
        write(self.run/'protocol.json',self.protocol);self.psha=audit.sha(self.run/'protocol.json')
        self.summary={'schema':'vsr_videomaev2_training_summary_v1','status':'completed','protocol_sha256':self.psha,
            'scientific_updates':4096,'diagnostic_updates_in_this_training':0,'arms':{},'compute_cost':{
                'committed_training_step_seconds':4096.,'committed_hr_read_seconds':256.,
                'hr_read_time_scope':'Outside synchronized Step; contained in whole attempt wall, never added again to attempt wall'}}
        for arm in self.study['arms']:
            label,weight=arm['label'],arm['weight'];engine=TinyEngine(label)
            engine.base_load_audit.update(checkpoint_path=str(self.base),checkpoint_sha256=audit.sha(self.base))
            engine.step();directory=self.run/'arm_runs'/label;checkpoints=[]
            for step in audit.STEPS:
                engine.steps=step;engine.teacher_forward_count=step if weight else 0
                for state in engine.optimizer.state.values():state['step'].fill_(step)
                cfg=config_for(engine);cfg['training']['updates']=2048;cfg=canonical(cfg)
                path=directory/'Base'/f'checkpoint_{step:06d}.pt'
                item=save_videomaev2_checkpoint(path,engine,cfg,allowed_root=self.run,protocol_sha256=self.psha,
                    cumulative_training_seconds=float(step),initialization=engine.base_load_audit)
                item.update(update=step);checkpoints.append(item)
            rows=[];hr=0.
            for i in range(1,2049):
                read_seconds=.125 if weight else 0.;hr+=read_seconds
                row={'step':i,'arm':label,'family':label,'teacher_kind':label,'setting':'common','weight':weight,
                    'data_index':(i-1)%1641,'record_id':self.protocol['cache']['selection']['training_ids'][(i-1)%1641],
                    'sigma':.5,'timestep_index':50,'native_timestep':500.,'student_layer':17,
                    'fm':.2,'align':.1 if weight else 0.,'total':.2+weight*.1,'grad_norm_before_clip':1.,
                    'student_total_grad_norm_before_clip':.5,'head_total_grad_norm_before_clip':.5 if weight else 0.,
                    'seconds':1.,'gpu_event_seconds':.5,'cumulative_training_seconds':float(i),
                    'hr_read_seconds':read_seconds,'cumulative_hr_read_seconds':hr,'noise_bin':'mid',
                    'teacher_forward':bool(weight),'teacher_forward_count':i if weight else 0,'teacher_parameter_updates':0,'EMA_updates':0}
                if weight:row['alignment']={'margin':.1,'normalize_eps':1e-12,'tokens_per_example':1792,'all_pairs':3211264,
                    'spatial_pairs':200704,'temporal_pairs':3010560,'all_tokens_supervised':True,'diagonal_included':True,
                    'ordered_pairs':True,'projected_all_latent_positions':9,'dropped_latent_positions':[0],'target_grid':[16,8,14],
                    'student_normalized_dtype':'torch.bfloat16','student_gram_dtype':'torch.bfloat16',
                    'teacher_normalized_dtype':'torch.float32','teacher_gram_dtype':'torch.float32',
                    'difference_dtype':'torch.float32','loss_dtype':'torch.float32','active_pair_fraction':.2}
                rows.append(row)
            log=directory/'training.jsonl';log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            current={n:p for n,p in engine.model.named_parameters() if p.requires_grad}
            record={'updates':2048,'weight':weight,'family':label,'teacher_kind':label,'setting':'common','EMA_updates':0,
                'teacher_parameter_updates':0,'teacher_forward_count':2048 if weight else 0,'teacher_binding':teacher if weight else None,
                'teacher_is_frozen_cached_HR':False,'checkpoint_files':checkpoints,'initialization':engine.base_load_audit,
                'full_training_log':audit.binding(log),'initial_student_hashes':{'all_trainable':base_digest},
                'final_student_hashes':{'all_trainable':audit.trainable_digest(torch,current)},'cumulative_training_seconds':2048.,
                'cumulative_hr_read_seconds':hr,'head_parameters':8917760 if weight else 0}
            write(directory/'initialization.json',engine.base_load_audit);write(directory/'completion.json',record);self.summary['arms'][label]=record
            del engine
        self.publish()
    def publish(self):
        write(self.run/'summary.json',self.summary)
        write(self.run/'status.json',{'status':'completed','scientific_updates':4096,'protocol_sha256':self.psha,'summary_sha256':audit.sha(self.run/'summary.json')})
    def update_record(self,arm):write(self.run/'arm_runs'/arm/'completion.json',self.summary['arms'][arm]);self.publish()
    def mutate_cp(self,case):
        item=self.summary['arms']['videomaev2']['checkpoint_files'][0];payload=torch.load(item['path'],map_location='cpu',weights_only=True)
        state=payload['training_only']
        if case=='rng':state['explicit_RNG']['noise'][10]^=1
        elif case=='teacher':state['teacher_binding']['position_embedding']['sha256']='d'*64
        elif case=='global':state['global_RNG']['torch_cpu']=torch.zeros(1,dtype=torch.uint8)
        elif case=='moment':next(iter(state['optimizer']['state'].values()))['step'].add_(1)
        elif case=='headmoment':
            for group,names in zip(state['optimizer']['param_groups'],state['optimizer_parameter_names']):
                for identifier,name in zip(group['params'],names):
                    if name.startswith('student_projector.'):state['optimizer']['state'][identifier]['exp_avg_sq'].zero_()
        torch.save(payload,item['path']);item.update(audit.binding(item['path']));self.update_record('videomaev2')
    def perform(self):return audit.audit(self.run,self.output,source_root=self.source,expected_protocol_sha=self.psha,allowed_output_root=self.root/'deliveries')


class AuditTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_running_rejected_before_other_read_hash_or_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';write(run/'status.json',{'status':'running','scientific_updates':4096})
            with patch.object(audit,'sha',side_effect=AssertionError('no early hashes')),patch.object(torch,'load',side_effect=AssertionError('no early torchload')):
                with self.assertRaisesRegex(ValueError,'incomplete'):audit.metadata(run,Path(tmp)/'source')
    def test_complete4096_all8CP_six_head_four_rng_externalbinding_and_cost_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);result=f.perform();self.assertEqual(result['status'],'passed_videomaev2_training_audit');self.assertEqual(result['diagnostic_flags'],[])
            r=audit.read(f.output/'training_audit.json');self.assertEqual(len(r['checkpoint_checks']['rng_rows']),8)
            self.assertEqual(len(list(f.output.glob('*.csv'))),6);self.assertFalse(r['quality_scores_read']);self.assertFalse(torch.cuda.is_initialized())
            self.assertTrue(all(row['head_tensors']==6 for row in r['checkpoint_checks']['update_rows'] if row['arm']=='videomaev2'))
            self.assertEqual(r['cost_rows'][1]['separate_CPU_HR_read_seconds'],256.)
            self.assertNotIn('exp_avg_sq',(f.output/'training_audit.json').read_text())
            with self.assertRaisesRegex(ValueError,'new child'):f.perform()
    def test_source_change_and_missing_CP_rejected_by_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);path=f.source/'vsr/train.py';original=path.read_text();path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Changed input'):audit.metadata(f.run,f.source)
            path.write_text(original);f.summary['arms']['videomaev2']['checkpoint_files'].pop();f.update_record('videomaev2')
            with self.assertRaisesRegex(ValueError,'identity/count'):audit.metadata(f.run,f.source)
    def test_nonfinite_data_sequence_or_HR_cost_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);record=f.summary['arms']['videomaev2'];path=Path(record['full_training_log']['path']);raw=path.read_text()
            for key,value in [('native_timestep',400.),('hr_read_seconds',.5),('fm',float('nan'))]:
                rows=[json.loads(line) for line in raw.splitlines()];rows[0][key]=value;path.write_text(''.join(json.dumps(r)+'\n' for r in rows));record['full_training_log']=audit.binding(path);f.update_record('videomaev2')
                with self.subTest(key=key),self.assertRaises(ValueError):audit.metadata(f.run,f.source)
    def test_RNG_teacher_global_rng_and_AdamW_corruptions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);item=f.summary['arms']['videomaev2']['checkpoint_files'][0];original=Path(item['path']).read_bytes()
            for case in ('rng','teacher','global','moment'):
                Path(item['path']).write_bytes(original);item.update(audit.binding(item['path']));f.update_record('videomaev2');f.mutate_cp(case)
                with self.subTest(case=case),self.assertRaises((ValueError,RuntimeError)):f.perform()
    def test_missing_all_head_second_moment_evidence_is_reported_as_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);f.mutate_cp('headmoment');r=f.perform();self.assertEqual(r['status'],'completed_with_diagnostic_flags');self.assertTrue(r['diagnostic_flags'])


if __name__=='__main__':unittest.main(verbosity=2)
