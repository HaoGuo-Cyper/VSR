"""Serial orchestration CPU fixtures: no systemd, model loads or GPU calls."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import videomaev2_pipeline as mod


def metadata(training,phase):
    points=([('shared',0)]+[(a,u) for a in ('baseline','videomaev2') for u in (512,1024,2048)]
            if phase=='val' else [(a,2048) for a in ('baseline','videomaev2')])
    return {'schema':'vsr_videomaev2_evaluation_protocol_v1','phase':phase,
        'sources':training['sources'],'study_file':training['study_file'],'training_evidence':{'scientific_updates':4096},
        'roles':[{'arm':a,'update':u,'weight':0. if a in ('shared','baseline') else .5,'checkpoint':{'sha256':f'{a}-{u}'}} for a,u in points],
        'scope':{'total_restores':len(points)*64,'roles':len(points),'clips':32,'sampling_seeds':[42,43],
                 'conditions':['clean'],'frames':33,'HR_hw':[128,224],'LR_hw':[32,56],'steps':20,'CFG':1.,
                 'training_updates':0,'weight_or_checkpoint_selection':False},
        **{key:{} for key in ('text_cache','metric_weights','official_model','runtime','strict_environment')}}


class PipelineTests(unittest.TestCase):
    def test_fixed448_128_protocol_pair_and_reject_sampler_weight_or_final_CP_changes(self):
        training={'sources':{'fixed':'same'},'study_file':{'fixed':'study'}}
        val,test=metadata(training,'val'),metadata(training,'test')
        mod.review_evaluation_protocol(val,training,'val')
        mod.review_evaluation_protocol(test,training,'test',validation=val)
        for kind in ('sampler','weight','checkpoint','runtime'):
            bad=copy.deepcopy(test)
            if kind=='sampler':bad['scope']['steps']=10
            if kind=='weight':bad['roles'][-1]['weight']=.1
            if kind=='checkpoint':bad['roles'][-1]['checkpoint']={'sha256':'wrong'}
            if kind=='runtime':bad['runtime']={'different':1}
            with self.subTest(kind=kind),self.assertRaises(ValueError):mod.review_evaluation_protocol(bad,training,'test',validation=val)

    def fake_main(self,root,*,unit_failed=False,audit_flags=False,child_failure=None):
        source=root/'source';train=root/'training';pipeline=root/'pipeline';output=root/'outputs'
        file=source/'vsr/train.py';file.parent.mkdir(parents=True);file.write_text('# unchanged')
        protocol={'schema':'vsr_videomaev2_training_protocol_v1','scientific_update_budget':4096,
            'sources':{'train.py':{'path':str(file),'sha256':mod.sha(file)}},'study_file':{'sha256':'study'}}
        mod.write(train/'protocol.json',protocol);mod.write(train/'status.json',{'status':'completed'})
        mod.write(train/'summary.json',{'scientific_updates':4096})
        for name in ('videomaev2_training_audit.py','analyze_videomaev2_study.py','analyze_hr_vae_study.py'):(root/name).write_text('# immutable helper fixture')
        mod.write(root/'active_videomaev2.json',{'environment':{'PYTHONPATH':str(source)}})
        commands=[]
        def child(args,**kwargs):
            commands.append(args);result={'status':'fixture'}
            command=args[3] if len(args)>3 and args[1:3]==['-m','vsr.evaluate'] else 'audit' if 'videomaev2_training_audit.py' in args[1] else 'analysis'
            if command=='audit':result={'status':'passed_videomaev2_training_audit','diagnostic_flags':['bad'] if audit_flags else []}
            elif command=='vmae-prepare':
                run=Path(args[args.index('--run-dir')+1]);phase='val' if run.name=='validation' else 'test'
                mod.write(run/'protocol.json',metadata(protocol,phase));result={'protocol_sha256':mod.sha(run/'protocol.json')}
            elif command=='vmae-lock-finals':mod.write(args[args.index('--output')+1],{'fixed':'both final2048'})
            kwargs['stdout'].write(json.dumps(result));kwargs['stdout'].flush()
            return types.SimpleNamespace(returncode=1 if child_failure==command else 0)
        argv=['pipeline','--training-run',str(train),'--training-unit','already-running.service','--source-root',str(source),
            '--pipeline-dir',str(pipeline),'--output-dir',str(output)]
        with patch.object(mod,'WORK',root),patch('sys.argv',argv),patch.object(mod.subprocess,'check_output',return_value='MainPID=0\nExecMainStatus='+('1' if unit_failed else '0')+'\n'),patch.object(mod.subprocess,'run',side_effect=child),contextlib.redirect_stdout(io.StringIO()):
            if unit_failed or audit_flags or child_failure:
                with self.assertRaises(ValueError):mod.main()
            else:mod.main()
        return commands,mod.read(pipeline/'status.json'),pipeline

    def test_waits_existing_training_and_never_launches_training_or_extra_matrix(self):
        with tempfile.TemporaryDirectory() as folder:
            commands,state,pipeline=self.fake_main(Path(folder))
            self.assertEqual(state['status'],'awaiting_root_delivery')
            native=[x[3] for x in commands if x[1:3]==['-m','vsr.evaluate']]
            self.assertEqual(native,['vmae-build-config','vmae-prepare','vmae-execute','vmae-verify','vmae-lock-finals',
                'vmae-build-config','vmae-prepare','vmae-execute','vmae-verify'])
            self.assertEqual(sum('vsr.train' in x for x in commands),0)
            for phase in ('validation','test'):
                wall=mod.read(pipeline/phase/'execution_wall_time.json')
                self.assertGreaterEqual(wall['wall_seconds'],0)
                self.assertEqual(wall['stage'],'evaluation_'+('val' if phase=='validation' else 'test'))

    def test_failed_training_unit_stops_before_any_child(self):
        with tempfile.TemporaryDirectory() as folder:
            commands,state,_=self.fake_main(Path(folder),unit_failed=True)
            self.assertEqual(commands,[]);self.assertEqual(state['status'],'failed_stop')

    def test_training_audit_flags_stop_before_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            commands,state,_=self.fake_main(Path(folder),audit_flags=True)
            self.assertEqual(len(commands),1);self.assertEqual(state['status'],'failed_stop')

    def test_execution_failure_is_preserved_without_retry_or_test_start(self):
        with tempfile.TemporaryDirectory() as folder:
            commands,state,pipeline=self.fake_main(Path(folder),child_failure='vmae-execute')
            self.assertEqual(sum('vmae-execute' in x for x in commands),1)
            self.assertEqual(state['phase'],'04_val_execute');self.assertFalse((pipeline/'test').exists())
            self.assertFalse(mod.read(pipeline/'failure.json')['automatic_retry'])


if __name__=='__main__':unittest.main(verbosity=2)
