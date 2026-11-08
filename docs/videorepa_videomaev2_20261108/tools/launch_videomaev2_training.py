"""Launch the sole preflight-gated paired study; no preparation or retry here."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,argparse
work=Path('/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work');source=Path('/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--review',required=True);args=a.parse_args()
run=Path(args.run).resolve();p=read(run/'protocol.json');psha=sha(run/'protocol.json');pointer=read(work/'active_videomaev2.json')
assert p['schema']=='vsr_videomaev2_training_protocol_v1' and p['scientific_update_budget']==4096 and p['diagnostic_optimizer_budget']==0
assert read(run/'status.json')['status']=='planned' and not (run/'launch.json').exists()
for b in p['sources'].values():assert sha(b['path'])==b['sha256']
preflight=Path(pointer['run_dir']);assert preflight!=run and preflight==Path(p['calibration']['summary']['path']).parent
assert read(preflight/'status.json')['status']=='completed' and read(preflight/'summary.json')['status']=='passed_videomaev2_preflight'
review=read(args.review);assert review['status']=='passed_independent_videomaev2_preflight_review' and review['diagnostic_flags']==[]
assert review['protocol_sha256']==sha(preflight/'protocol.json')
service=subprocess.check_output(['systemctl','--user','show',pointer['unit'],'-p','MainPID','-p','ExecMainStatus'],text=True)
assert 'MainPID=0' in service and 'ExecMainStatus=0' in service
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');unit='vsr-videomaev2-training-'+stamp.lower();log=run/'console.log';env=dict(pointer['environment'])
command=['/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python','-m','vsr.train','vmae-execute','--run-dir',str(run),'--expected-protocol-sha',psha]
receipt={'phase':'videomaev2_training','unit':unit+'.service','run_dir':str(run),'protocol_sha256':psha,'source_root':str(source),'source_freeze':p['sources'],
 'command':command,'environment':env,'log':str(log),'preflight_run':str(preflight),'preflight_unit':pointer['unit'],
 'preflight_review':{'path':str(Path(args.review).resolve()),'sha256':sha(args.review)},'scientific_update_budget':4096,
 'diagnostic_updates_in_this_training':0,'Restart':'no','RuntimeMaxSec':10800,'started_UTC':datetime.now(timezone.utc).isoformat()}
launch=run/'launch.json';launch.write_text(json.dumps(receipt,indent=2)+'\n')
call=['systemd-run','--user','--unit='+unit,'--property=Type=exec','--property=Restart=no','--property=RemainAfterExit=yes','--property=RuntimeMaxSec=10800','--property=WorkingDirectory='+str(source),'--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log)]
call+=['--setenv='+k+'='+v for k,v in env.items()]+['--',*command]
r=subprocess.run(call,text=True,capture_output=True,check=True);receipt['systemd_response']=r.stdout+r.stderr;launch.write_text(json.dumps(receipt,indent=2)+'\n')
pointer.update(receipt,status='running',GPU_started=True,scientific_updates=0,diagnostic_updates=26,launch_receipt=str(launch))
(work/'active_videomaev2.json').write_text(json.dumps(pointer,indent=2)+'\n')
print(json.dumps({'unit':receipt['unit'],'run':str(run),'protocol_sha256':psha,'science_budget':4096,'new_diagnostics':0}))
