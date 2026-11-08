"""Reviewed one-shot CPU parent launcher; never starts or restarts training.

Prepared only. A human-authorized root execution must provide its final read-only
postprocessing review receipt. The bound pipeline waits for the exact already
running training service, then serially evaluates 448 validation and128 test rows.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess

WORK = Path('/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work')
SOURCE = Path('/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync')
TRAIN = WORK/'videomaev2_runs/paired_training_20261108T150511Z'
PIPELINE = WORK/'videomaev2_runs/pipeline_paired_training_20261108T150511Z'
OUTPUT = WORK.parent/'outputs/VideoREPA_VideoMAEv2_20261108'
PROTOCOL_SHA = '5292bbd39d21303ac4f57541d80c2e897b02782291b4e73817eb01bd97676bdf'
PYTHON = '/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python'
HELPERS = {
    'videomaev2_pipeline.py': '4421df92463f7a5ac587c8473bfefe2df0570d0930035f4f5d791673008d07ae',
    'videomaev2_training_audit.py': '83b044718708394a7291075d7f535c5b9b7a08d7c2972bb11d45d17af113be0c',
    'analyze_videomaev2_study.py': '6a335f88c4cb9f59e9d1b08c191ca117c12534f1917696d25d92ddb956d0430f',
    'analyze_hr_vae_study.py': '260a4e9673771a31033528293c09610b225ac551bd9778f2527caec8d4d3b83f',
    'export_videomaev2_text.py': '6a6dd04fe75a54db55c192c45eeccde2918c58d2093c2b99dcca9d0d564cc8a0',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def write(path, value, *, new=False):
    path = Path(path)
    if new:
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2) + '\n')
    else:
        temp = path.with_name(path.name + '.launcher-tmp')
        need(not temp.exists(), 'Launcher temporary output occupied')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', required=True,
                        help='Root final postprocessing read-only review: passed/ready status and empty blocking_findings.')
    args = parser.parse_args()
    pointer_path = WORK/'active_videomaev2.json'
    pointer = read(pointer_path)
    protocol = read(TRAIN/'protocol.json')
    need(sha(TRAIN/'protocol.json') == PROTOCOL_SHA, 'Exact prepared training protocol required')
    need(protocol['schema'] == 'vsr_videomaev2_training_protocol_v1'
         and protocol['scientific_update_budget'] ==4096 and protocol['diagnostic_optimizer_budget']==0,
         'Wrong training or budgets')
    need(pointer['phase']=='videomaev2_training' and Path(pointer['run_dir']).resolve()==TRAIN
         and pointer['protocol_sha256']==PROTOCOL_SHA and pointer['scientific_update_budget']==4096,
         'Pointer must declare this already-started training')
    need(not pointer.get('continuation'), 'A continuation was already declared; no duplicate parent')
    need(not PIPELINE.exists() and not OUTPUT.exists(), 'Fresh pipeline and final delivery directories required')
    training_unit = pointer['unit']
    need(training_unit.startswith('vsr-videomaev2-training-') and training_unit.endswith('.service'),
         'Expected exact VideoMAEv2 training unit')
    launch = read(TRAIN/'launch.json')
    need(launch['unit']==training_unit and launch['protocol_sha256']==PROTOCOL_SHA,
         'Training launch receipt differs from the active pointer')
    trainstate=read(TRAIN/'status.json')
    need(trainstate['protocol_sha256']==PROTOCOL_SHA and trainstate['status'] in ('planned','running','completed'),
         'Failed training cannot receive a continuation')
    service_raw=subprocess.check_output(['systemctl','--user','show',training_unit,
        '-p','LoadState','-p','MainPID','-p','ExecMainStatus'], text=True)
    service=dict(line.split('=',1) for line in service_raw.splitlines() if '=' in line)
    need(service.get('LoadState')=='loaded', 'Already-existing training service is required')
    if service.get('MainPID')=='0':
        need(service.get('ExecMainStatus')=='0' and trainstate['status']=='completed',
             'Stopped or failed training cannot be restarted by this parent')
    else:
        need(int(service.get('MainPID','0'))>0, 'Training PID invalid')
    for item in protocol['sources'].values():
        need(sha(item['path'])==item['sha256'], 'Frozen numerical source changed')
    for name,digest in HELPERS.items():
        need(sha(WORK/name)==digest, 'Unreviewed helper changed: '+name)
    review_path=Path(args.review).resolve();review=read(review_path)
    need(str(review.get('status','')).lower().startswith(('passed','ready'))
         and review.get('blocking_findings')==[], 'Final independent postprocessing review must pass')
    env=dict(pointer['environment'])
    need(env['PYTHONPATH'].endswith(str(SOURCE)), 'Wrong declared source')
    parent_env=dict(env);parent_env['CUDA_VISIBLE_DEVICES']=''
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    unit='vsr-videomaev2-continuation-'+stamp.lower()
    log=WORK/'videomaev2_runs'/(unit+'.log')
    receipt_path=WORK/'videomaev2_runs'/(unit+'_launch.json')
    command=[PYTHON,str(WORK/'videomaev2_pipeline.py'),'--training-run',str(TRAIN),
             '--training-unit',training_unit,'--source-root',str(SOURCE),
             '--pipeline-dir',str(PIPELINE),'--output-dir',str(OUTPUT)]
    receipt={'status':'prepared_for_unique_launch','unit':unit+'.service','training_run':str(TRAIN),
             'training_unit':training_unit,'training_protocol_sha256':PROTOCOL_SHA,
             'training_launch_receipt':str(TRAIN/'launch.json'),
             'pipeline_dir':str(PIPELINE),'pipeline_status':str(PIPELINE/'status.json'),
             'status_file':str(PIPELINE/'status.json'),'output_dir':str(OUTPUT),
             'command':command,'parent_environment':parent_env,'child_environment_source':str(pointer_path),
             'log':str(log),'launch_receipt':str(receipt_path),'Restart':'no','RuntimeMaxSec':21600,
             'CPU_parent':True,'starts_or_restarts_training':False,'new_scientific_update_budget':0,
             'serial_validation_restores':448,'serial_test_restores':128,'teacher_cache_created':False,
             'helpers':{str(WORK/k):v for k,v in HELPERS.items()},
             'root_review':{'path':str(review_path),'sha256':sha(review_path)},
             'training_service_before_launch':service,'automatic_retry':False}
    write(receipt_path,receipt,new=True)
    pointer['continuation']=receipt
    write(pointer_path,pointer)
    call=['systemd-run','--user','--unit='+unit,'--property=Type=exec','--property=Restart=no',
          '--property=RemainAfterExit=yes','--property=RuntimeMaxSec=21600',
          '--property=WorkingDirectory='+str(SOURCE),
          '--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log)]
    call += ['--setenv='+key+'='+value for key,value in parent_env.items()] + ['--',*command]
    result=subprocess.run(call,text=True,capture_output=True)
    receipt.update(status='launched' if result.returncode==0 else 'launch_failed_no_retry',
        launched_UTC=datetime.now(timezone.utc).isoformat(),systemd_returncode=result.returncode,
        systemd_response=result.stdout+result.stderr)
    write(receipt_path,receipt)
    # The pipeline may already have written its phase; preserve that more recent state.
    latest=read(pointer_path);current=latest.get('continuation',{})
    if current.get('status') == 'prepared_for_unique_launch':
        current['status'] = receipt['status']
    latest['continuation']=receipt|current|{'launch_receipt':str(receipt_path),
        'launch_status':receipt['status'],'systemd_response':receipt['systemd_response']}
    write(pointer_path,latest)
    need(result.returncode==0, 'Parent launch failed; retained receipt, no automatic retry')
    print(json.dumps({'unit':receipt['unit'],'pipeline':str(PIPELINE),'output':str(OUTPUT),
                      'training_not_restarted':True,'CPU_parent':True}))


if __name__=='__main__':
    main()
