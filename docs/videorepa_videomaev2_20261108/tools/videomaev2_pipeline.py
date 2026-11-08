"""Serial continuation of the already-started Wan VideoMAEv2 study; never starts training.

All GPU work is performed by the frozen native evaluator with its shared lease.
Any nonzero child or changed binding stops the chain without retry/retuning.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,time,traceback

WORK=Path(__file__).resolve().parent
PYTHON='/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,p)
def need(value,message):
    if not value:raise ValueError(message)


def review_evaluation_protocol(evaluation,training,phase,*,validation=None):
    """Pure metadata check before the unique frozen native executor is called."""
    expected=([('shared',0)]+[(a,u) for a in ('baseline','videomaev2') for u in (512,1024,2048)]
              if phase=='val' else [(a,2048) for a in ('baseline','videomaev2')])
    scope=evaluation['scope']
    need(evaluation['schema']=='vsr_videomaev2_evaluation_protocol_v1' and evaluation['phase']==phase
         and evaluation['sources']==training['sources']
         and evaluation['study_file']==training['study_file']
         and evaluation['training_evidence']['scientific_updates']==4096,
         'Evaluation must bind this complete unchanged two-arm training')
    need([(r['arm'],r['update']) for r in evaluation['roles']]==expected
         and scope['total_restores']==(448 if phase=='val' else 128)
         and scope['roles']==len(expected) and scope['clips']==32 and scope['sampling_seeds']==[42,43]
         and scope['conditions']==['clean'] and scope['frames']==33 and scope['HR_hw']==[128,224]
         and scope['LR_hw']==[32,56] and scope['steps']==20 and scope['CFG']==1.
         and scope['training_updates']==0 and scope['weight_or_checkpoint_selection'] is False,
         'Prospective complete native inference matrix changed')
    need([r['weight'] for r in evaluation['roles']]==[0. if a in ('shared','baseline') else .5 for a,_ in expected],
         'Evaluation roles changed the fixed alignment weights')
    if phase=='test':
        need(validation is not None,'Test review requires already verified validation protocol')
        for key in ('sources','study_file','text_cache','metric_weights','official_model','runtime','strict_environment'):
            need(evaluation[key]==validation[key],'Test changed matched numeric inputs: '+key)
        finals={r['arm']:r['checkpoint'] for r in validation['roles'] if r['update']==2048}
        need(all(r['checkpoint']==finals[r['arm']] for r in evaluation['roles']),
             'Test changed the prospectively locked final checkpoints')
    return {'status':'reviewed_fixed448_128_protocol','phase':phase,'training_updates':0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--training-unit',required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--pipeline-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    options=parser.parse_args()
    source=options.source_root.resolve();train=options.training_run.resolve()
    directory=options.pipeline_dir.resolve();output=options.output_dir.resolve()
    protocol=read(train/'protocol.json');psha=sha(train/'protocol.json')
    need(protocol['schema']=='vsr_videomaev2_training_protocol_v1' and protocol['scientific_update_budget']==4096,'Wrong existing training')
    need(protocol['sources']['train.py']['path']==str(source/'vsr/train.py'),'Wrong source root')
    need(not directory.exists() and not output.exists(),'Use a new continuation and delivery directory')
    helpers=[Path(__file__).resolve(),WORK/'videomaev2_training_audit.py',WORK/'analyze_videomaev2_study.py',
             WORK/'analyze_hr_vae_study.py']
    bindings={str(path):sha(path) for path in helpers}
    directory.mkdir(parents=True);output.mkdir(parents=True)
    write(directory/'bindings.json',{'training_protocol_sha256':psha,'sources':protocol['sources'],'helpers':bindings,
        'command':vars(options)|{'training_run':str(train),'source_root':str(source),'pipeline_dir':str(directory),'output_dir':str(output)}})
    env=dict(read(WORK/'active_videomaev2.json')['environment'])
    need(env['PYTHONPATH'].endswith(str(source)),'Declared Python source differs')
    phase='00_wait_existing_training'
    def state(status,**extra):
        value={'status':status,'phase':phase,'updated_UTC':datetime.now(timezone.utc).isoformat(),
               'training_run':str(train),'training_protocol_sha256':psha,'scientific_update_budget':4096,**extra}
        write(directory/'status.json',value)
        pointer=read(WORK/'active_videomaev2.json')
        pointer['continuation']=pointer.get('continuation',{})|{'pipeline_dir':str(directory),'pipeline_status':str(directory/'status.json'),
            'output_dir':str(output),'phase':phase,'status':status}
        if 'current_execution' in extra:pointer['current_execution']=extra['current_execution']
        write(WORK/'active_videomaev2.json',pointer)
    def verify_bindings():
        need(sha(train/'protocol.json')==psha,'Training protocol changed')
        for item in protocol['sources'].values():need(sha(item['path'])==item['sha256'],'Frozen numeric source changed: '+item['path'])
        for path,digest in bindings.items():need(sha(path)==digest,'Frozen continuation helper changed: '+path)
    def command(name,args,*,cpu=False,json_result=True):
        nonlocal phase
        phase=name;verify_bindings()
        out=directory/'logs'/(name+'.stdout');err=directory/'logs'/(name+'.stderr')
        out.parent.mkdir(parents=True,exist_ok=True)
        write(directory/'phases'/(name+'.json'),{'command':args,'CPU_only':cpu,'status':'running'})
        current=None
        if '-m' in args and 'vsr.evaluate' in args and 'vmae-execute' in args:
            r=Path(args[args.index('--run-dir')+1]);current={'stage':name,'run_dir':str(r),'status':'running',
                'protocol_sha256':sha(r/'protocol.json'),'training_updates':0}
        state('running',**({'current_execution':current} if current else {}))
        runenv=dict(env)
        if cpu:runenv['CUDA_VISIBLE_DEVICES']=''
        started=time.perf_counter()
        with out.open('w') as stdout,err.open('w') as stderr:
            result=subprocess.run(args,cwd=source,env=runenv,stdout=stdout,stderr=stderr)
        record={'command':args,'returncode':result.returncode,'wall_seconds':time.perf_counter()-started,
                'stdout':str(out),'stderr':str(err),'CPU_only':cpu,'status':'completed' if result.returncode==0 else 'failed'}
        write(directory/'phases'/(name+'.json'),record)
        need(result.returncode==0,'Phase failed: '+name+'; see '+str(err))
        if current is not None:
            execution_protocol=read(Path(current['run_dir'])/'protocol.json')
            write(Path(current['run_dir'])/'execution_wall_time.json',{
                'status':'completed','stage':'evaluation_'+execution_protocol['phase'],
                'run':current['run_dir'],'protocol_sha256':current['protocol_sha256'],
                'wall_seconds':record['wall_seconds'],
                'scope':'One native evaluator execute process, including model load, restores, metrics, media I/O and cleanup; excludes prepare/verify/analysis; not pure GPU kernel time.'})
        value=read(out) if json_result else None
        if json_result:write(directory/'phases'/(name+'.result.json'),value)
        return value
    def core(*args):return [PYTHON,'-m','vsr.evaluate',*map(str,args)]
    try:
        state('waiting_existing_training',current_execution={'stage':phase,'run_dir':str(train),'status':'running','protocol_sha256':psha})
        while True:
            verify_bindings()
            raw=subprocess.check_output(['systemctl','--user','show',options.training_unit,'-p','MainPID','-p','ExecMainStatus'],text=True)
            status=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
            if status.get('MainPID')=='0':
                need(status.get('ExecMainStatus')=='0','Existing training service failed; no automatic retry')
                break
            need(read(train/'status.json')['status'] in ('planned','running','completed'),'Training failed while service still exiting')
            time.sleep(10)
        need(read(train/'status.json')['status']=='completed' and read(train/'summary.json')['scientific_updates']==4096,
             'Training did not complete all4096 updates')
        audit=command('01_training_audit',[PYTHON,str(WORK/'videomaev2_training_audit.py'),'--run',str(train),
                      '--source-root',str(source),'--output-dir',str(output/'training_diagnostics'),'--expected-protocol-sha',psha],cpu=True)
        need(audit.get('status')=='passed_videomaev2_training_audit' and audit.get('diagnostic_flags')==[],
             'Complete independent training audit must pass before any evaluation')
        valcfg=directory/'val_config.json';val=directory/'validation'
        command('02_val_config',core('vmae-build-config','--training-run',train,'--phase','val','--output',valcfg,'--run-root',directory))
        prepared=command('03_val_prepare',core('vmae-prepare','--config',valcfg,'--run-dir',val))
        vp=read(val/'protocol.json')
        need(vp['scope']['total_restores']==448 and vp['scope']['frames']==33 and len(vp['roles'])==7,'Wrong full validation protocol')
        review_evaluation_protocol(vp,protocol,'val')
        command('04_val_execute',core('vmae-execute','--run-dir',val,'--expected-protocol-sha',prepared['protocol_sha256']))
        command('05_val_verify',core('vmae-verify','--run-dir',val))
        finals=directory/'finals_receipt.json'
        command('06_lock_all_finals',core('vmae-lock-finals','--validation-run',val,'--output',finals))
        testcfg=directory/'test_config.json';test=directory/'test'
        command('07_test_config',core('vmae-build-config','--training-run',train,'--phase','test','--output',testcfg,
                      '--run-root',directory,'--validation-run',val,'--finals-receipt',finals))
        prepared=command('08_test_prepare',core('vmae-prepare','--config',testcfg,'--run-dir',test))
        tp=read(test/'protocol.json')
        need(tp['scope']['total_restores']==128 and len(tp['roles'])==2,'Wrong both-final test protocol')
        review_evaluation_protocol(tp,protocol,'test',validation=vp)
        command('09_test_execute',core('vmae-execute','--run-dir',test,'--expected-protocol-sha',prepared['protocol_sha256']))
        command('10_test_verify',core('vmae-verify','--run-dir',test))
        for phase_name,run_path,verify_name in (('11_val_analysis',val,'05_val_verify'),('12_test_analysis',test,'10_test_verify')):
            command(phase_name,[PYTHON,str(WORK/'analyze_videomaev2_study.py'),'--run',str(run_path),
                    '--output-dir',str(output/('val' if run_path==val else 'test')),
                    '--verification',str(directory/'phases'/(verify_name+'.result.json'))],cpu=True)
        phase='awaiting_root_delivery'
        state('awaiting_root_delivery',current_execution={'stage':'training_val_test_analysis_completed','run_dir':str(test),
              'status':'completed','training_updates':4096,'validation_restores':448,'test_restores':128},
              pending='root independent statistics review, fixed videos/visual observations, final report and text sync')
        print(json.dumps({'status':'awaiting_root_delivery','pipeline':str(directory),'outputs':str(output)}))
    except BaseException as error:
        failure={'status':'failed_stop','phase':phase,'error':str(error),'traceback':traceback.format_exc(),
                 'automatic_retry':False,'scientific_settings_changed':False}
        write(directory/'failure.json',failure);state('failed_stop',error=str(error))
        raise


if __name__=='__main__':main()
