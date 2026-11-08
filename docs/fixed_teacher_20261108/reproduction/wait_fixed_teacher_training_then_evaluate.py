"""One-shot serial handoff: wait for the existing training unit, never launch it."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,sys,time,os,tempfile,traceback
WORK=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt_path=Path(sys.argv[1]).resolve();receipt=read(receipt_path)
run=Path(receipt['training_run']);status=receipt_path.parent/'status.json';started=time.monotonic()
def save(value):
 value={'at_UTC':datetime.now(timezone.utc).isoformat(),**value}
 fd,tmp=tempfile.mkstemp(prefix='.status.',dir=status.parent)
 with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,status)
try:
 for item in receipt['bound_helpers'].values():assert sha(item['path'])==item['sha256'],'Handoff helper changed before waiting'
 assert sha(run/'protocol.json')==receipt['training_protocol_sha256']
 while True:
  assert time.monotonic()-started<10800,'Existing training did not finish within the declared3h service ceiling'
  assert sha(run/'protocol.json')==receipt['training_protocol_sha256']
  for item in receipt['source_freeze'].values():assert sha(item['path'])==item['sha256'],'Frozen core source changed'
  text=subprocess.check_output(['systemctl','--user','show',receipt['training_unit'],'-p','MainPID','-p','ExecMainStatus','-p','Result'],text=True)
  values=dict(line.split('=',1) for line in text.splitlines() if '=' in line)
  state=read(run/'status.json')
  if values.get('MainPID')=='0':
   assert values.get('ExecMainStatus')=='0' and values.get('Result')=='success',text
   assert state.get('status')=='completed' and state.get('scientific_updates')==10240,'Training did not fully complete'
   assert state.get('protocol_sha256')==receipt['training_protocol_sha256'] and state.get('summary_sha256')==sha(run/'summary.json')
   break
  assert state.get('status') in ('planned','running','completed'),state
  if state.get('status')=='completed':
   assert state.get('scientific_updates')==10240 and state.get('protocol_sha256')==receipt['training_protocol_sha256']
   assert state.get('summary_sha256')==sha(run/'summary.json'),'Incomplete transient completion chain'
  save({'status':'waiting_for_existing_training','training_unit':receipt['training_unit'],'training_state':state,
        'training_progress':read(run/'progress.json') if (run/'progress.json').exists() else None,'GPU_work_started_by_waiter':False})
  time.sleep(45)
 for item in receipt['bound_helpers'].values():assert sha(item['path'])==item['sha256'],'Handoff helper changed before evaluation'
 assert not Path(receipt['pipeline_dir']).exists(),'Pipeline path already exists; duplicate handoff refused'
 save({'status':'running_serial_evaluation_pipeline','command':receipt['command'],'training_unit_exited0':True,
       'training_run':str(run),'pipeline_dir':receipt['pipeline_dir']})
 code=subprocess.run(receipt['command'],cwd=receipt['source_root'],check=False).returncode
 assert code==0,'Serial pipeline failed; inspect its phase/status, no automatic retry'
 completion=read(Path(receipt['pipeline_dir'])/'completion.json')
 assert completion['status']=='awaiting_root_delivery' and completion['validation_restores']==1024 and completion['test_restores']==192
 save({'status':'awaiting_root_delivery','pipeline_completion':completion,'training_or_evaluation_retried':False})
except BaseException as error:
 save({'status':'failed_stopped','error':str(error),'traceback':traceback.format_exc(),'automatic_retry':False})
 raise
