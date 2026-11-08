"""Export explicitly named fixed-teacher runs as verified text only.

--root is the frozen source checkout; --run-root is the private output parent;
--dest is the publication base. The actual destination is DEST/RUN.name.
No task module is imported, no binary artifact is opened, and no run is started.
Training traces are append-only per attempt; rollback requires its preserved
interrupted-log provenance. A completed canonical trace gets a new final_snapshot.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

TRAIN='fixed_teacher_training_protocol_v1'
EVAL='fixed_teacher_clean_evaluation_protocol_v1'
CAL='fixed_teacher_calibration_protocol_v1'
METRICS=('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1','tLP_alex_v0_1_TecoGAN_crop',
         'tLP100_alex_v0_1_TecoGAN_crop','tOF_Farneback_TecoGAN_crop')
ROOT_JSON=('protocol.json','config.json','study.json','initialization.json','summary.json','selection.json',
    'selection_receipt.json','observations.json','role_audits.json','failure.json','runtime.json',
    'GPU_admission.json','GPU_idle_before_load.json','launch_receipt.json','host_service_request.json',
    'host_service_launch.json','pre_execution_review.json','service_completion.json')
LOGS=('console.log','stdout.log','stderr.log','service.log','console.txt')
TEXT_SUFFIX={'.json','.jsonl','.log','.txt'}
SECRET_KEYS={'password','passwd','authorization','api_key','access_token','refresh_token','client_secret',
             'private_key','credentials','credential'}
SECRET_PATTERN=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,})')
PROTECTED=(Path('/home/hugo/Projects/VSR'),Path('/home/hugo/Projects/VSR_'),
    Path('/home/hugo/Projects/ReprVSR-pretrained'),
    Path('/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/VSR-sync'))


def need(value,message):
    if not value:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def encoded(value):return (json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()


def component(value):
    need(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}',value)
         and value not in ('.','..'),'Unsafe protocol path component')
    return value


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def inspect_json(value):
    if isinstance(value,dict):
        need(not any(str(k).lower() in SECRET_KEYS for k in value),'Credential-shaped JSON key in publication text')
        for item in value.values():inspect_json(item)
    elif isinstance(value,list):
        for item in value:inspect_json(item)
    elif isinstance(value,float):need(math.isfinite(value),'Nonfinite JSON value')


def raw_text(path,*,source=False):
    path=Path(path)
    need(not path.is_symlink() and path.is_file(),'Missing or symlink text source: '+str(path))
    need(path.suffix in TEXT_SUFFIX or (source and path.suffix=='.py'),'Non-allowlisted artifact access forbidden')
    raw=path.read_bytes();raw.decode('utf-8')
    need(b'\0' not in raw and not SECRET_PATTERN.search(raw),'Binary or credential-shaped publication text')
    return raw


def parsed(raw):
    def reject(value):raise ValueError('Nonfinite JSON constant: '+value)
    value=json.loads(raw,parse_constant=reject);inspect_json(value);return value


def binding_shape(item):
    need(isinstance(item,dict) and isinstance(item.get('path'),str)
         and isinstance(item.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',item['sha256'])
         and type(item.get('bytes')) is int and item['bytes']>=0,'Invalid declared artifact binding')


def atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.is_symlink(),'Destination symlink forbidden')
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


def immutable(path,raw):
    if path.exists():need(not path.is_symlink() and path.read_bytes()==raw,'Immutable text rewrite refused: '+str(path))
    else:atomic(path,raw)


def append(path,raw):
    previous=path.read_bytes() if path.exists() else b''
    need(not path.is_symlink() and raw.startswith(previous),'Append-only text rewrite/shortening refused: '+str(path))
    if len(raw)>len(previous):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('ab') as stream:stream.write(raw[len(previous):]);stream.flush();os.fsync(stream.fileno())
    elif not path.exists():atomic(path,b'')


class Plan:
    def __init__(self,root,run,destination):
        self.root,self.run,self.destination=root,run,destination
        self.copies={};self.traces={};self.dynamic={};self.pending=[];self.provenance=[];self.source_hashes={}
    def json(self,path,relative,*,required=False):
        path=Path(path)
        if not path.exists():
            need(not required,'Required text missing: '+str(path));return None
        try:raw=raw_text(path);value=parsed(raw)
        except (json.JSONDecodeError,UnicodeDecodeError):
            need(not required,'Required JSON is incomplete: '+str(path));self.pending.append(str(path));return None
        self.copies[str(relative)]=(path,raw);return value
    def snapshot(self,path,name):
        if not path.exists():return None
        try:raw=raw_text(path);value=parsed(raw)
        except (json.JSONDecodeError,UnicodeDecodeError):self.pending.append(str(path));return None
        self.dynamic[name]=raw;return value
    def bound(self,item,name):
        binding_shape(item);path=Path(item['path']).resolve()
        # Only explicitly bound owned JSON is followed. Other references remain
        # visible in protocol.json; they never cause arbitrary artifact reads.
        if path.suffix!='.json' or not any(path.is_relative_to(base) for base in (self.root,self.run.parent.parent)):
            return None
        value=self.json(path,Path('bound')/name,required=True);raw=self.copies[str(Path('bound')/name)][1]
        need(len(raw)==item['bytes'] and sha(raw)==item['sha256'],'Bound JSON SHA/size changed')
        return value
    def jsonl(self,path):
        path=Path(path)
        need(path.suffix=='.jsonl' and path.is_file() and not path.is_symlink(),'Invalid JSONL source')
        raw=path.read_bytes();complete=raw[:raw.rfind(b'\n')+1];pieces=complete.split(b'\n')[:-1]
        rows=[];offset=0
        for i,piece in enumerate(pieces):
            line=piece+b'\n'
            try:value=parsed(line)
            except (json.JSONDecodeError,UnicodeDecodeError):
                need(i==len(pieces)-1,'Malformed nonterminal JSONL record')
                complete=complete[:offset];break
            rows.append(value);offset+=len(line)
        complete.decode('utf-8')
        need(b'\0' not in complete and not SECRET_PATTERN.search(complete),'Binary/credential-shaped JSONL')
        if len(complete)!=len(raw):self.pending.append(str(path))
        return complete,rows
    def prior(self):
        path=self.destination/'export_state.json'
        if not path.exists():return
        state=parsed(raw_text(path))
        for relative,item in state['immutable'].items():
            source=Path(item['source'])
            if source.suffix=='.jsonl':
                need(source.is_file() and not source.is_symlink(),'Preserved JSONL source missing')
                raw=source.read_bytes()
            else:raw=raw_text(source)
            need(sha(raw)==item.get('source_sha256',item['sha256']),'Previously exported immutable source changed/deleted: '+relative)
            published=(self.destination/relative).read_bytes()
            need(sha(published)==item['sha256'] and len(published)==item['bytes'],
                 'Previously exported immutable destination changed')
            need(raw.startswith(published) if item.get('complete_prefix_only') else raw==published,
                 'Published immutable text no longer matches its source prefix')
    def publish(self,snapshot):
        self.destination.mkdir(parents=True,exist_ok=True)
        need(not any(p.is_symlink() for p in self.destination.rglob('*')),'Destination symlinks forbidden')
        with (self.destination/'.text_export.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);self.prior()
            # Validate every existing destination before publishing any addition.
            for relative,(_,raw) in self.copies.items():
                path=self.destination/relative
                if path.exists():need(path.read_bytes()==raw,'Immutable text rewrite refused: '+relative)
            for relative,raw in self.traces.items():
                path=self.destination/relative
                if path.exists():need(raw.startswith(path.read_bytes()),'Append-only text rewrite/shortening refused: '+relative)
            for relative,(_,raw) in self.copies.items():immutable(self.destination/relative,raw)
            for relative,raw in self.traces.items():append(self.destination/relative,raw)
            for relative,raw in self.dynamic.items():atomic(self.destination/relative,raw)
            old=parsed(raw_text(self.destination/'export_state.json')) if (self.destination/'export_state.json').exists() else {'immutable':{}}
            records=old['immutable']|{r:{'source':str(p),'sha256':sha(raw),'bytes':len(raw),
                'source_sha256':self.source_hashes.get(r,sha(raw)),
                'complete_prefix_only':self.source_hashes.get(r,sha(raw))!=sha(raw)} for r,(p,raw) in self.copies.items()}
            atomic(self.destination/'export_state.json',encoded({'immutable':records}))
            atomic(self.destination/'text_export_snapshot.json',encoded(snapshot))
            fcntl.flock(lock,fcntl.LOCK_UN)


def source_bindings(protocol,root):
    sources=protocol.get('sources');need(isinstance(sources,dict) and sources,'Frozen source bindings required')
    for name,item in sources.items():
        component(name);binding_shape(item)
        path=Path(item['path']).resolve();need(path==root/'vsr'/name and path.suffix=='.py','Source binding escaped frozen source root')
        raw=raw_text(path,source=True)
        need(sha(raw)==item['sha256'] and len(raw)==item['bytes'],'Frozen numeric source SHA changed')


def role_training_rows(rows,arm,ids,updates):
    need(len(rows)<=updates,'Training trace exceeds declared2048 budget');previous=0.
    for i,row in enumerate(rows):
        need(row['step']==i+1 and type(row['step']) is int and row['arm']==arm['label']
             and row['weight']==arm['weight'] and row['family']==arm['family'] and row['setting']==arm['setting'],
             'Training step/role/weight/family changed')
        need(row['data_index']==i%len(ids) and row['record_id']==ids[i%len(ids)],'Training data order changed')
        need(all(finite(row[k]) for k in ('fm','align','total','grad_norm_before_clip','sigma','seconds','cumulative_training_seconds')),
             'Nonfinite training log row')
        need(0<=row['sigma']<=1 and row['seconds']>=0 and row['cumulative_training_seconds']==previous+row['seconds'],
             'Training sigma/time recursion changed')
        previous=row['cumulative_training_seconds']


def training(plan,p,status,psha):
    study=p['study'];arms=study['arms'];updates=study['updates'];ids=p['cache']['selection']['training_ids']
    need(updates==2048 and len(arms)==5 and p['scientific_update_budget']==10240 and len(ids)==1641,
         'Unexpected fixed-teacher training budget/data count')
    attempts=sorted(plan.run.glob('attempt_[0-9][0-9][0-9].json'))
    attempt=int(attempts[-1].stem.split('_')[-1]) if attempts else 0
    count=0;completed={}
    for arm in arms:
        label=component(arm['label']);directory=plan.run/'arm_runs'/label
        for name in ('config.json','initialization.json','completion.json'):
            value=plan.json(directory/name,Path('arm_runs')/label/name)
            if name=='completion.json' and value is not None:completed[label]=value
        for pattern in ('resume_loader_audit_attempt_[0-9][0-9][0-9].json','rollback_attempt_[0-9][0-9][0-9].json',
                        'interrupted_log_attempt_[0-9][0-9][0-9].jsonl'):
            for path in sorted(directory.glob(pattern)):
                if path.suffix=='.json':plan.json(path,path.relative_to(plan.run))
                else:
                    raw,rows=plan.jsonl(path);role_training_rows(rows,arm,ids,updates)
                    relative=str(path.relative_to(plan.run))
                    plan.copies[relative]=(path,raw)
                    plan.source_hashes[relative]=sha(path.read_bytes())
        path=directory/'training.jsonl'
        if not path.exists():continue
        need(attempts,'Training log without an attempt declaration')
        raw,rows=plan.jsonl(path);role_training_rows(rows,arm,ids,updates)
        relative=Path('attempt_traces')/f'attempt_{attempt:03d}'/label/'training.jsonl'
        if attempt:
            previous=sorted((plan.destination/'attempt_traces').glob(f'attempt_*/{label}/training.jsonl'))
            previous=[x for x in previous if x.parts[-3]!=f'attempt_{attempt:03d}']
            if previous:
                old=previous[-1].read_bytes()
                archive=directory/f'interrupted_log_attempt_{attempt:03d}.jsonl'
                if label not in completed and not (status and status.get('status')=='completed'):
                    # An attempt declaration precedes checkpoint rollback. Do
                    # not mislabel the still-old canonical tail as the new
                    # attempt's trace while the producer is preserving it.
                    waiting=(not archive.exists() and len(old.splitlines()) not in study['checkpoint_updates'] and raw.startswith(old))
                    if archive.exists():
                        key=str(archive.relative_to(plan.run))
                        waiting=key not in plan.copies or raw.startswith(plan.copies[key][1])
                    if waiting:
                        plan.pending.append(str(path)+' awaiting completed checkpoint rollback publication');continue
                if not raw.startswith(old):
                    archive=directory/f'interrupted_log_attempt_{attempt:03d}.jsonl'
                    need(archive.is_file() and not archive.is_symlink(),'Rollback lacks its preserved interrupted log')
                    archived=archive.read_bytes()
                    need(archived.startswith(old),'Rollback lacks its preserved previously exported attempt trace')
                    common=0
                    for a,b in zip(old.splitlines(),raw.splitlines()):
                        if a!=b:break
                        common+=1
                    need(common in study['checkpoint_updates'],'Rollback text common prefix is not a declared checkpoint boundary')
                    plan.provenance.append({'arm':label,'attempt':attempt,'preserved_log':str(archive),
                        'observed_common_prefix_steps':common,'checkpoint_binary_opened':False,
                        'preserved_source_sha256':sha(archived),
                        'unexported_partial_source_bytes':len(archived)-len(plan.copies[str(archive.relative_to(plan.run))][1])})
        plan.traces[str(relative)]=raw;count+=len(rows)
        if status and status.get('status')=='completed':
            need(len(rows)==updates and raw==raw_text(path),'Completed training needs2048 complete rows per arm')
            record=completed.get(label);need(record and record['updates']==updates and record['weight']==arm['weight']
                and record['family']==arm['family'] and record['setting']==arm['setting'],'Incomplete/wrong arm completion')
            item=record['full_training_log'];binding_shape(item)
            need(Path(item['path']).resolve()==path.resolve() and item['sha256']==sha(raw) and item['bytes']==len(raw),
                 'Completion training log SHA changed')
            need([x['step'] for x in record['checkpoint_files']]==study['checkpoint_updates'],'Missing checkpoint declarations')
            for cp in record['checkpoint_files']:binding_shape(cp)
            plan.copies[str(Path('final_snapshot')/'arm_runs'/label/'training.jsonl')]=(path,raw)
    if status and status.get('status')=='completed':
        summary=parsed(plan.copies['summary.json'][1])
        need(count==10240 and status['scientific_updates']==10240 and summary['scientific_updates']==10240
             and summary['schema']=='fixed_teacher_training_summary_v1' and summary['protocol_sha256']==psha
             and set(summary['arms'])=={a['label'] for a in arms},'Completed five-arm scientific accounting differs')
        need(all(summary['arms'][k]==v for k,v in completed.items()) and len(completed)==5,'Final arm records differ')
    return {'exported_training_rows_current_attempt':count,'expected_effective_training_updates':10240,
            'attempt':attempt,'rollback_provenance':plan.provenance,'attempt_traces_are_extra_scientific_updates':False}


def evaluation(plan,p,status,psha):
    scope=p['scope'];roles=p['roles'];cohort=p['cohort'];phase=p['phase'];samples=scope['sampling_seeds']
    need(phase in ('val','test') and scope['source_split']==phase and scope['conditions']==['clean']
         and samples==[42,43] and len(cohort)==scope['clips']==32 and len(roles)==scope['roles']
         and scope['total_restores']==len(roles)*64 and scope['training_updates']==0,'Unexpected clean evaluation scope')
    need(len({x['record']['id'] for x in cohort})==32 and len({x['record']['source_id'] for x in cohort})==32,
         'Cohort sources/IDs must be unique')
    observations=[];gap=False
    for role in roles:
        label=component(role['label']);binding_shape(role['checkpoint'])
        for path in sorted((plan.run/'roles'/label).glob('*.json')):
            if path.name=='load_audit.json' or re.fullmatch(r'state_audit_\d{3}.json',path.name):
                audit=plan.json(path,path.relative_to(plan.run))
                if audit and path.name.startswith('state_audit_'):
                    need(audit['training_updates']==0 and audit['trainable_before']==audit['trainable_after'],'Student state audit changed')
        for item in cohort:
            record=component(item['record']['id'])
            for sample in samples:
                relative=Path('results')/label/record/f'seed_{sample}';directory=plan.run/relative
                marker=plan.json(directory/'complete.json',relative/'complete.json')
                if marker is None:gap=True;continue
                need(not gap,'Completed evaluation rows must form a canonical prefix')
                row=plan.json(directory/'metrics.json',relative/'metrics.json',required=True)
                raw=plan.copies[str(relative/'metrics.json')][1]
                need(marker['protocol_sha256']==psha and marker['metrics_sha256']==sha(raw) and marker['PNG_count']==33,
                     'Completed metrics/protocol SHA changed')
                expected={k:role[k] for k in ('arm','weight','update','step','cumulative_training_seconds','family','setting')}
                expected.update(role=label,training_seed=20261102,record_id=record,source_id=item['record']['source_id'],
                    sampling_seed=sample,condition='clean',checkpoint_sha256=role['checkpoint']['sha256'],
                    cohort_seal_digest=p['cohort_seal_digest'],split=phase)
                obs=row['observation'];need(set(obs)==set(expected)|{'metrics'} and all(obs[k]==v for k,v in expected.items()),
                    'Evaluation row identity/checkpoint/source changed')
                need(row['status']=='completed' and row['training_updates']==0 and row['teacher_or_HR_input_to_inferrer'] is False
                    and row['inference_inputs']==['LRV','native text embedding','student checkpoint']
                    and row['restoration']['inference_inputs']==['LRV','native text embedding'],'Invalid completed inference row')
                meta=row['source_metadata'];need(meta['split']==phase and meta['source_frame_indices']==item['source_indices'],
                    'Frozen source/window changed')
                full=row['full_frame_pair_metrics']['all_frames']
                need(full['frame_count']==33 and len(full['per_frame'])==33 and full['temporal_pair_count']==28
                    and {tuple(x['indices']) for x in full['per_temporal_pair']}=={(i,i+1) for i in range(2,30)},
                    'Evaluation frame/pair pool changed')
                need(set(obs['metrics'])==set(METRICS) and all(finite(full[k]) and obs['metrics'][k]==full[k] for k in METRICS),
                    'Evaluation scalar pool changed/nonfinite')
                pngs=row['PNG_artifacts'];need(len(pngs)==33 and len({x['path'] for x in pngs})==33,'Incomplete PNG declarations')
                for artifact in pngs:
                    binding_shape(artifact);need(Path(artifact['path']).resolve().is_relative_to(directory/'png'),'PNG declaration escaped row')
                video=row['video_artifact'];binding_shape(video)
                need(Path(video['path']).resolve()==directory/'prediction.mp4' and marker['video_artifact']==video,'Video declaration differs')
                observations.append(obs|{'row_metrics_path':str(directory/'metrics.json'),'row_metrics_sha256':sha(raw)})
    if 'observations.json' in plan.copies:
        need(parsed(plan.copies['observations.json'][1])=={'observations':observations,'protocol_sha256':psha},'Final observations differ from canonical rows')
    if status and status.get('status')=='completed':
        total=scope['total_restores'];summary=parsed(plan.copies['summary.json'][1])
        need(len(observations)==total==status['restores']==summary['restores'] and summary['roles']==len(roles)
            and status['training_updates']==summary['training_updates']==0 and summary['protocol_sha256']==psha,
            'Completed evaluation scope/count changed')
        need(parsed(plan.copies['role_audits.json'][1])==[{'role':r['label'],'restores':64,'training_updates':0,'inference_weights':'student'} for r in roles],
            'Completed role audit declarations differ')
    plan.traces['completed_observations.jsonl']=b''.join(encoded(row) for row in observations)
    return {'phase':phase,'source_split':phase,'exported_completed_rows':len(observations),
            'expected_restores':scope['total_restores'],'canonical_prefix_verified':True}


def calibration(plan,p,status,psha):
    for label in ('hr_vae','hr_dino'):
        relative=Path('resume_checks')/label/'config.json'
        plan.json(plan.run/relative,relative)
    need(p['scientific_update_budget']==0 and p['diagnostic_optimizer_budget']==27
         and p['calibration_sigmas']==[.2,.5,.8] and p['calibration_training_index']==0,'Calibration protocol changed')
    if status and status.get('status')=='completed':
        summary=parsed(plan.copies['summary.json'][1])
        need(summary['status']=='passed_fixed_teacher_preflight' and summary['protocol_sha256']==psha
             and summary['sources']==p['sources'] and summary['scientific_updates']==0
             and summary['diagnostic_optimizer_updates']==27 and len(summary['all_step_rows'])==24,
             'Incomplete calibration diagnostics')
        need([r['role'] for r in summary['all_step_rows']]==['baseline']*8+['hr_vae']*8+['hr_dino']*8,'Diagnostic row order differs')
        need(all(r['step']==i%8+1 and all(finite(r[k]) for k in ('fm','align','total','sigma','grad_norm_before_clip','seconds'))
                 for i,r in enumerate(summary['all_step_rows'])),'Nonfinite/incomplete diagnostic rows')
    return {'scientific_updates':0,'expected_diagnostic_optimizer_updates':27,'calibration_uses_training_index':0}


def export_run(run,*,root,run_root,dest):
    root,run_root,dest=(Path(x).resolve() for x in (root,run_root,dest));run=Path(run).resolve()
    need(root.is_dir() and run_root.is_dir() and run.is_dir() and run!=run_root and run.is_relative_to(run_root),
         'Source run must be an existing child of explicit run_root')
    destination=dest/component(run.name)
    need(not destination.is_relative_to(run) and not run.is_relative_to(destination),'Export/source overlap forbidden')
    need(not any(destination.is_relative_to(x.resolve()) for x in PROTECTED),'Historical/actual source destinations are forbidden')
    plan=Plan(root,run,destination);p=plan.json(run/'protocol.json','protocol.json',required=True)
    schema=p.get('schema');need(schema in (TRAIN,EVAL,CAL),'Unsupported fixed-teacher protocol schema')
    psha=sha(plan.copies['protocol.json'][1]);source_bindings(p,root)
    for name in ROOT_JSON[1:]:plan.json(run/name,name)
    for key,name in (('study_file','study.json'),('config_binding','evaluation_config.json'),('selection_receipt','selection_receipt.json')):
        if key in p:plan.bound(p[key],name)
    for key in ('dataset','cohort_seal'):
        if key in p:plan.bound(p[key],key+'.json')
    for group,keys in (('dino_cache',('manifest','seal')),('calibration',('protocol','summary','status_file'))):
        for key in keys:
            if key in p.get(group,{}):plan.bound(p[group][key],group+'_'+key+'.json')
    for folder in (run,run/'attempts'):
        if not folder.is_dir():continue
        for path in sorted(folder.glob('*.json')):
            if re.fullmatch(r'(?:attempt|completion_attempt|failure_attempt|GPU_admission_attempt|GPU_admission|failure)_\d{3}.json',path.name):
                plan.json(path,path.relative_to(run))
    status=plan.snapshot(run/'status.json','status_snapshot.json');progress=plan.snapshot(run/'progress.json','progress_snapshot.json')
    if status is not None:need(status.get('protocol_sha256')==psha,'Live status protocol SHA differs')
    if progress is not None and 'protocol_sha256' in progress:need(progress['protocol_sha256']==psha,'Live progress protocol SHA differs')
    if status and status.get('status')=='completed':
        need('summary.json' in plan.copies and sha(plan.copies['summary.json'][1])==status['summary_sha256'],'Completion summary SHA changed')
    details=training(plan,p,status,psha) if schema==TRAIN else evaluation(plan,p,status,psha) if schema==EVAL else calibration(plan,p,status,psha)
    for name in LOGS:
        path=run/name
        if path.exists():
            try:plan.traces[name]=raw_text(path)
            except UnicodeDecodeError:plan.pending.append(str(path))
    snapshot={'observed_at_UTC':datetime.now(timezone.utc).isoformat(),'run':str(run),'source_root':str(root),
        'protocol_sha256':psha,'schema':schema,'status':status.get('status') if status else None,
        'pending_incomplete_text':plan.pending,'binary_artifacts_opened_or_exported':False,
        'checkpoint_or_media_bytes_verified_by_exporter':False,'source_SHA_verified':True,
        'partial_export_is_a_scientific_verdict':False,**details}
    plan.publish(snapshot);return snapshot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','run-root','run','dest'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args(argv);result=export_run(args.run,root=args.root,run_root=args.run_root,dest=args.dest)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));return result


if __name__=='__main__':main()
