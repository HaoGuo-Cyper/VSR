"""Tiny synthetic complete-study tensors; never real training or GPU work."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import fixed_teacher_training_diagnostics as diag


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def text(path,value='synthetic source only'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value);return diag.binding(path)


class Fixture:
    def __init__(self,root,*,same_shape=False):
        self.root=Path(root);self.run=self.root/'run';self.source=self.root/'source';self.output=self.root/'outputs/new'
        self.sources={name:text(self.source/'vsr'/name) for name in sorted(diag.CORE)}
        config={'model':{'synthetic':True},'conditioning':{'synthetic':'LR'},'data':{'frames':33},'training':{'variants':['Base'],'updates':8192}}
        self.base_state={'lrv_stem.weight':torch.zeros(2,3),'transformer.lora_A.weight':torch.zeros(2,3) if same_shape else torch.zeros(1,2)}
        basepath=self.root/'base/Base/checkpoint_008192.pt';basepath.parent.mkdir(parents=True)
        torch.save({'format':'vsr_training_v1','step':8192,'config':config,'trainable_state':self.base_state},basepath)
        self.base=diag.binding(basepath);self.origin={'checkpoint_path':str(basepath),'checkpoint_sha256':self.base['sha256'],'step':8192,'variant':'Base','config':config}
        self.initial={'all_trainable':diag.model_trainable_digest(torch,self.base_state)}
        self.arms=[{'label':'baseline','family':'baseline','setting':'baseline','weight':0.}]
        for family,short,weights in (('hr_vae','vae',(.11,1.1)),('hr_dino','dino',(.094,.94))):
            self.arms += [{'label':short+'_'+setting,'family':family,'setting':setting,'weight':weight} for setting,weight in zip(('low','high'),weights)]
        self.study={'schema':'fixed_teacher_study_v1','updates':2048,'training_seed':20261102,'checkpoint_updates':list(diag.STEPS),
                    'EMA':False,'condition_corruption':False,'arms':self.arms}
        studypath=self.root/'study.json';save(studypath,self.study)
        cal=self.root/'cal';save(cal/'protocol.json',{'sources':self.sources})
        save(cal/'summary.json',{'status':'passed_fixed_teacher_preflight','scientific_updates':0,'diagnostic_optimizer_updates':27,
            'protocol_sha256':diag.sha(cal/'protocol.json'),'sources':self.sources})
        save(cal/'status.json',{'status':'completed','protocol_sha256':diag.sha(cal/'protocol.json'),'summary_sha256':diag.sha(cal/'summary.json')})
        self.protocol={'schema':'fixed_teacher_training_protocol_v1','scientific_update_budget':10240,'study':self.study,'study_file':diag.binding(studypath),
            'sources':self.sources,'dependencies':{'torch':{'version':torch.__version__,'files':{}}},'device':'cuda:0','cache':{'selection':{'training_ids':[f'record{i}' for i in range(1641)]}},'base':{'checkpoint':self.base},
            'calibration':{'protocol':diag.binding(cal/'protocol.json'),'summary':diag.binding(cal/'summary.json'),'status_file':diag.binding(cal/'status.json')}}
        save(self.run/'protocol.json',self.protocol);psha=diag.sha(self.run/'protocol.json');completed={}
        for arm in self.arms:
            directory=self.run/'arm_runs'/arm['label'];directory.mkdir(parents=True)
            rows=[]
            for step in range(1,2049):
                sigma=(.2,.5,.8)[step%3];rows.append({'step':step,'arm':arm['label'],'family':arm['family'],'setting':arm['setting'],'weight':arm['weight'],
                    'record_id':f'record{(step-1)%1641}','data_index':(step-1)%1641,'sigma':sigma,'noise_bin':'low' if sigma<.33 else 'mid' if sigma<.67 else 'high',
                    'timestep_index':step%1000,'native_timestep':float(999-step%1000),'fm':1.+step/2048,'align':0. if arm['weight']==0 else .5,
                    'total':1.+step/2048+arm['weight']*.5,'grad_norm_before_clip':2.,'seconds':.125,'cumulative_training_seconds':step*.125,
                    'teacher_forward':False,'EMA_updates':0})
            log=directory/'training.jsonl';log.write_text(''.join(json.dumps(x)+'\n' for x in rows));cps=[]
            for step in diag.STEPS:
                student={n:t+step/2048 for n,t in self.base_state.items()};head=None;spec=None;kind='dino' if arm['family']=='hr_dino' else 'vae'
                if arm['weight']:
                    spec={'class':'vsr.losses.HRDINOCosineRegression' if kind=='dino' else 'vsr.losses.HRLatentRegression',
                          'student_dim':3,'target_dim':4 if kind=='dino' else 2,'latent_channels':None,'patch_size':None}
                    head={n:t+step/2048 for n,t in diag.initial_head(torch,spec,20261102).items()}
                parameters={'restoration.'+n:t for n,t in student.items()}
                if head:parameters.update({'student_projector.'+n.removeprefix('projector.'):t for n,t in head.items()})
                names=list(parameters);optim={'state':{},'param_groups':[{'params':list(range(len(names))),'foreach':False}]}
                for i,name in enumerate(names):optim['state'][i]={'step':torch.tensor(float(step)),'exp_avg':torch.ones_like(parameters[name])*.01,'exp_avg_sq':torch.ones_like(parameters[name])*.001}
                rng={name:torch.tensor([step//256,i,4,9],dtype=torch.uint8) for i,name in enumerate(diag.STREAMS)}
                training={'schema':'vsr_hr_latent_training_state_v1','protocol_sha256':psha,'completed_updates':step,'training_seed':20261102,'weight':arm['weight'],
                    'student_layer':7,'cumulative_training_seconds':step*.125,'teacher_kind':kind,'restored_by_generic_inference_loader':False,
                    'matched_Base':self.origin,'explicit_RNG':rng,'RNG_devices':{'noise':'cuda:0','timestep':'cpu','mask':'cpu','corruption':'cuda:0'},
                    'alignment_state':head,'alignment_spec':spec,'optimizer':optim,'optimizer_parameter_names':[names]}
                value={'format':'vsr_training_v1','variant':'Base','step':step,'config':config,'inference_parameters':'student',
                       'initialization':self.origin,'trainable_state':student,'training_only':training}
                path=directory/f'Base/checkpoint_{step:06d}.pt';path.parent.mkdir(exist_ok=True);torch.save(value,path)
                cps.append(diag.binding(path)|{'step':step,'cumulative_training_seconds':step*.125})
            completed[arm['label']]={'updates':2048,'family':arm['family'],'setting':arm['setting'],'weight':arm['weight'],'checkpoint_files':cps,
                'initial_student_hashes':self.initial,'final_student_hashes':{'all_trainable':diag.model_trainable_digest(torch,student)},
                'full_training_log':diag.binding(log),'cumulative_training_seconds':256.}
            save(directory/'completion.json',completed[arm['label']])
        self.summary={'schema':'fixed_teacher_training_summary_v1','status':'completed','protocol_sha256':psha,'scientific_updates':10240,
                      'diagnostic_updates_in_this_training':0,'arms':completed};self.publish_summary()

    def publish_summary(self):
        save(self.run/'summary.json',self.summary)
        save(self.run/'status.json',{'status':'completed','scientific_updates':10240,'protocol_sha256':diag.sha(self.run/'protocol.json'),'summary_sha256':diag.sha(self.run/'summary.json')})

    def resign_arm(self,label):
        save(self.run/'arm_runs'/label/'completion.json',self.summary['arms'][label]);self.publish_summary()

    def execute(self):return diag.diagnose(self.run,self.output,source_root=self.source,allowed_output_root=self.root/'outputs')


class DiagnosticTests(unittest.TestCase):
    def test_complete_tiny_five_arm_audit_publishes_only_digests_and_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);result=f.execute();self.assertEqual(result['status'],'passed_completed_training_diagnostics')
            report=diag.read(f.output/'training_diagnostics.json');checks=report['checkpoint_checks']
            self.assertEqual(len(checks['rng_rows']),20);self.assertEqual(len(checks['update_rows']),20)
            self.assertTrue(checks['four_streams_bytewise_equal_across_all_arms_at_each_checkpoint'])
            self.assertTrue(all(x['student_changed_from_Base_tensors']==2 for x in checks['update_rows']))
            self.assertTrue(all(x['head_changed_from_seeded_initial_tensors']==2 for x in checks['update_rows'] if x['family']!='baseline'))
            self.assertFalse(report['public_raw_RNG_or_weight_bytes']);self.assertFalse(torch.cuda.is_initialized())
            self.assertEqual(len((f.output/'arm_windows.csv').read_text().splitlines()),11)
            with self.assertRaises(ValueError):f.execute()

    def test_legacy_source_Base_can_lack_variant_but_new_CP_cannot(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder)
            self.assertNotIn('variant',torch.load(f.base['path'],weights_only=True))
            result=f.execute();self.assertEqual(result['status'],'passed_completed_training_diagnostics')
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);label='baseline';cp=f.summary['arms'][label]['checkpoint_files'][0]
            value=torch.load(cp['path'],weights_only=True);value.pop('variant');torch.save(value,cp['path'])
            cp.update(diag.binding(cp['path']));f.resign_arm(label)
            with self.assertRaises((ValueError,KeyError)):f.execute()

    def test_incomplete_run_reads_status_only_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';save(run/'status.json',{'status':'running','scientific_updates':3072});seen=[]
            original=diag.read
            def guarded(path):
                seen.append(Path(path).name);self.assertEqual(Path(path).name,'status.json');return original(path)
            with patch.object(diag,'read',guarded),patch('torch.load',side_effect=AssertionError('must not read any tensors')):
                with self.assertRaisesRegex(ValueError,'Reject incomplete'):diag.diagnose(run,root/'outputs/new',allowed_output_root=root/'outputs')
            self.assertEqual(seen,['status.json']);self.assertFalse((root/'outputs/new').exists())

    def test_resigned_sigma_or_record_sequence_mismatch_is_rejected_before_checkpoints(self):
        for field in ('sigma','record_id'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder);label='dino_high';record=f.summary['arms'][label];path=Path(record['full_training_log']['path'])
                rows=[json.loads(x) for x in path.read_text().splitlines()]
                if field=='sigma':rows[20]['sigma']=.25  # Stilllow, but no longer the exact matched value.
                else:rows[20]['record_id']='wrong_record'
                path.write_text(''.join(json.dumps(x)+'\n' for x in rows));record['full_training_log']=diag.binding(path);f.resign_arm(label)
                with patch('torch.load',side_effect=AssertionError('metadata must reject before tensors')):
                    with self.assertRaises(ValueError):f.execute()
                self.assertFalse(f.output.exists())

    def test_resigned_checkpoint_RNG_bit_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);label='vae_high';cp=f.summary['arms'][label]['checkpoint_files'][1]
            value=torch.load(cp['path'],weights_only=True);value['training_only']['explicit_RNG']['noise'][0]^=1;torch.save(value,cp['path'])
            cp.update(diag.binding(cp['path']));f.resign_arm(label)
            with self.assertRaisesRegex(ValueError,'RNG stream differs'):f.execute()
            self.assertFalse(f.output.exists())

    def test_wrong_initial_student_or_nonfinite_gradient_fail_closed(self):
        for bad in ('initial','gradient'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder);label='vae_low';record=f.summary['arms'][label]
                if bad=='initial':record['initial_student_hashes']=copy.deepcopy(record['initial_student_hashes']);record['initial_student_hashes']['all_trainable']['sha256']='e'*64
                else:
                    path=Path(record['full_training_log']['path']);rows=[json.loads(x) for x in path.read_text().splitlines()]
                    rows[100]['grad_norm_before_clip']=float('nan');path.write_text(''.join(json.dumps(x)+'\n' for x in rows));record['full_training_log']=diag.binding(path)
                f.resign_arm(label)
                with self.assertRaises(ValueError):f.execute()

    def test_repeated_optimizer_ID_with_orphan_state_is_rejected_even_for_same_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder,same_shape=True);label='baseline';cp=f.summary['arms'][label]['checkpoint_files'][0]
            value=torch.load(cp['path'],weights_only=True);state=value['training_only']
            state['optimizer']['param_groups'][0]['params'][1]=state['optimizer']['param_groups'][0]['params'][0]
            torch.save(value,cp['path']);cp.update(diag.binding(cp['path']));f.resign_arm(label)
            with self.assertRaisesRegex(ValueError,'bijection'):f.execute()

    def test_head_family_or_shape_mismatch_is_rejected(self):
        for bad in ('class','shape'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder);label='vae_low';cp=f.summary['arms'][label]['checkpoint_files'][0]
                value=torch.load(cp['path'],weights_only=True);state=value['training_only']
                if bad=='class':state['alignment_spec']['class']='vsr.losses.HRDINOCosineRegression'
                else:state['alignment_state']['projector.weight']=state['alignment_state']['projector.weight'].t().contiguous()
                torch.save(value,cp['path']);cp.update(diag.binding(cp['path']));f.resign_arm(label)
                with self.assertRaises(ValueError):f.execute()

    def test_zero_head_second_moments_are_flagged_without_claiming_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);label='vae_low';cp=f.summary['arms'][label]['checkpoint_files'][0]
            value=torch.load(cp['path'],weights_only=True);state=value['training_only']
            for group,names in zip(state['optimizer']['param_groups'],state['optimizer_parameter_names']):
                for identifier,name in zip(group['params'],names):
                    if name.startswith('student_projector.'):
                        state['optimizer']['state'][identifier]['exp_avg'].zero_();state['optimizer']['state'][identifier]['exp_avg_sq'].zero_()
            torch.save(value,cp['path']);cp.update(diag.binding(cp['path']));f.resign_arm(label)
            result=f.execute();self.assertEqual(result['status'],'completed_with_diagnostic_flags')
            self.assertIn('no_nonzero_head_second_moment',result['diagnostic_flags'][0]['flag'])

    def test_single_CPU_head_initialization_preserves_global_RNG(self):
        state=torch.get_rng_state().clone();spec={'class':'vsr.losses.HRLatentRegression','student_dim':3,'target_dim':2}
        first=diag.initial_head(torch,spec,20261102);second=diag.initial_head(torch,spec,20261102)
        self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertTrue(torch.equal(first['projector.weight'],second['projector.weight']))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
