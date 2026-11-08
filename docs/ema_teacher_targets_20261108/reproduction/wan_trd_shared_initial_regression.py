"""Strict32-source shared-Base PNG regression after verified complete TRD val640.

CPU/std-library only: no tensor loading, metrics, model, new restores or GPU.
Both cohorts must be identical to the prior completed fixed-teacher val1024.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import traceback


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def binding(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def fingerprint(value):
    if isinstance(value,dict):
        if {'path','bytes','sha256'}<=value.keys():return {k:value[k] for k in ('bytes','sha256')}
        return {k:fingerprint(v) for k,v in value.items()}
    if isinstance(value,list):return [fingerprint(v) for v in value]
    return value


class Verifier:
    def __init__(self):self.files={}
    def verify(self,item):
        path=Path(item['path']).resolve();before=path.stat();signature=(before.st_ino,before.st_size,before.st_mtime_ns)
        cached=self.files.get(str(path))
        if cached and cached[0]==signature:actual=cached[1]
        else:
            actual=binding(path);after=path.stat()
            need(signature==(after.st_ino,after.st_size,after.st_mtime_ns),'Input changed while hashing')
            self.files[str(path)]=(signature,actual)
        need(item['sha256']==actual['sha256'] and ('bytes' not in item or item['bytes']==actual['bytes']), 'Bound input bytes changed: '+str(path))
        return actual
    def unchanged(self):
        for path,(signature,_) in self.files.items():
            s=Path(path).stat();need(signature==(s.st_ino,s.st_size,s.st_mtime_ns),'Input changed during regression: '+path)


def finished(run,*,historical=False):
    run=Path(run).resolve();state=read(run/'status.json');expected=1024 if historical else 640
    need(state.get('status')=='completed' and state.get('restores')==expected and state.get('training_updates')==0,
         'Prior val1024 or new val640 is incomplete; no output comparison allowed')
    protocol=read(run/'protocol.json');summary=read(run/'summary.json')
    need(state['protocol_sha256']==sha(run/'protocol.json') and state['summary_sha256']==sha(run/'summary.json')
         and summary['protocol_sha256']==state['protocol_sha256'] and summary['restores']==expected,'Completion SHA chain differs')
    need(protocol['schema']==('fixed_teacher_clean_evaluation_protocol_v1' if historical else 'wan_trd_evaluation_protocol_v1')
         and protocol['phase']=='val' and protocol['scope']['total_restores']==expected
         and protocol['scope']['roles']==(16 if historical else 10) and protocol['scope']['clips']==32,'Wrong full validation matrix')
    return run,protocol,state


def source_compatibility(old,new,verify):
    for sources in (old['sources'],new['sources']):
        need(len(sources)==11,'Complete11 numerical sources required')
        for item in sources.values():verify.verify(item)
    evidence={}
    for name in ('data.py','infer.py','model.py','vae.py','metrics.py'):
        a,b=old['sources'][name],new['sources'][name]
        need(fingerprint(a)==fingerprint(b),'Native inference numerical source changed: '+name)
        evidence[name]={'exact_file_bytes':True,'historical':a,'current':b}
    def trees(path):
        return {node.name:node for node in ast.parse(Path(path).read_text()).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    for name,symbols in [('evaluate.py',('_save_pngs','_crop_policy','_validate_restore')),
                         ('checkpoints.py',('_sha256_file','_safe_checkpoint','_check_config'))]:
        left,right=(trees(p['sources'][name]['path']) for p in (old,new))
        need(all(ast.dump(left[s],include_attributes=False)==ast.dump(right[s],include_attributes=False) for s in symbols),
             'Native inference AST changed: '+name)
        evidence[name]={'exact_symbol_AST':list(symbols)}
    a,b=(trees(p['sources']['checkpoints.py']['path'])['load_training_checkpoint'] for p in (old,new))
    additions=[];body=[]
    for node in b.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='trd_metadata':
            need(ast.dump(node.value,include_attributes=False)==ast.dump(ast.parse('_trd_inference_metadata(payload)').body[0].value,include_attributes=False),
                 'Unexpected loader metadata call');additions.append('scalar_metadata_call')
        elif isinstance(node,ast.If) and ast.dump(node.test,include_attributes=False)==ast.dump(ast.parse('trd_metadata is not None').body[0].value,include_attributes=False):
            expected=ast.parse("if trd_metadata is not None:metadata['trd_training_metadata']=trd_metadata").body[0]
            need(ast.dump(node,include_attributes=False)==ast.dump(expected,include_attributes=False),'Unexpected loader metadata branch')
            additions.append('conditional_scalar_audit')
        else:body.append(node)
    need(additions==['scalar_metadata_call','conditional_scalar_audit'],'Expected exactly the reviewed loader metadata addition')
    b.body=body
    need(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'Student inference loader changed beyond scalar provenance')
    evidence['checkpoints.py']['load_training_checkpoint_AST']='identical after exactly2 reviewed scalar-provenance statements; shared Base audit has no training-only fields'
    for key in ('official_model','text_cache','metric_weights'):
        need(fingerprint(old[key])==fingerprint(new[key]),'Official model/text/metric declarations changed: '+key)
    verify.verify(old['text_cache']);verify.verify(new['text_cache'])
    for value in new['metric_weights'].values():verify.verify(value)
    for key,item in new['official_model'].items():
        need(Path(item['path']).is_file() and Path(item['path']).stat().st_size==item['bytes'],'Official model file size changed')
        if key.endswith('.json'):verify.verify(item)
    for key in ('python','packages','native'):need(fingerprint(old['runtime'][key])==fingerprint(new['runtime'][key]),'Runtime declarations changed: '+key)
    for item in new['runtime']['native'].values():verify.verify(item)
    need(old['strict_environment']==new['strict_environment'],'Strict environment changed')
    return evidence


def row(run,protocol_sha,rid,sample,observation,verify):
    directory=run/'results/shared_initial'/rid/f'seed_{sample}';path=directory/'metrics.json'
    need(Path(observation['row_metrics_path']).resolve()==path.resolve(),'Unexpected shared-initial row path')
    item=verify.verify(binding(path));need(observation['row_metrics_sha256']==item['sha256'],'Observation row SHA differs')
    payload=read(path);marker=read(directory/'complete.json');verify.verify(binding(directory/'complete.json'))
    need(marker['protocol_sha256']==protocol_sha and marker['metrics_sha256']==item['sha256'] and marker['PNG_count']==33,
         'Completed-row marker differs')
    need(payload['status']=='completed' and payload['training_updates']==0
         and payload['observation']=={k:v for k,v in observation.items() if k not in ('row_metrics_path','row_metrics_sha256')},
         'Row observation/identity differs')
    need(payload['teacher_or_HR_input_to_inferrer'] is False and payload['inference_inputs']==['LRV','native text embedding','student checkpoint'],
         'Inference used training-only information')
    need(len(payload['PNG_artifacts'])==33 and [Path(x['path']).resolve() for x in payload['PNG_artifacts']]
         ==[(directory/'png'/f'{i:06d}.png').resolve() for i in range(33)],'Missing/out-of-order33 PNGs')
    return payload


def compare(historical_run,validation_run,verification_json):
    # Gate the new run first; an incomplete run permits only its status.json read.
    new_run,new,ns=finished(validation_run)
    old_run,old,os=finished(historical_run,historical=True)
    verification=read(verification_json)
    need(verification.get('status')=='verified_complete_clean_evaluation' and verification.get('restores')==640
         and verification.get('protocol_sha256')==ns['protocol_sha256'] and verification.get('run')==str(new_run)
         and verification.get('prediction_PNGs_verified')==640*33 and verification.get('prediction_MP4s_verified')==640
         and verification.get('observations_sha256')==sha(new_run/'observations.json')
         and verification.get('summary_sha256')==ns['summary_sha256']
         and verification.get('training_only_checkpoint_roles_verified') is True,'Require the real completed core trd-verify receipt')
    verify=Verifier();verify.verify(binding(verification_json))
    for run in (old_run,new_run):
        for name in ('protocol.json','status.json','summary.json','observations.json'):verify.verify(binding(run/name))
    for key in ('cohort_seal','dataset'):
        need(fingerprint(old[key])==fingerprint(new[key]),'Validation input declaration differs: '+key)
        verify.verify(old[key]);verify.verify(new[key])
    identity=lambda cohort:[(x['record']['id'],x['record']['source_id'],x['source_indices']) for x in cohort]
    need(identity(old['cohort'])==identity(new['cohort']) and len(new['cohort'])==32
         and len({x['record']['source_id'] for x in new['cohort']})==32,'Require identical32-source windows')
    assets={}
    for protocol in (old,new):
        current={x['record_id']:x for x in protocol['clean_assets']}
        need(set(current)=={x['record']['id'] for x in protocol['cohort']},'Missing clean input assets')
        if not assets:assets=current
        else:need(fingerprint(assets)==fingerprint(current),'Clean HR/LR metadata differs')
        for item in current.values():
            for kind in ('HR','clean'):verify.verify(item[kind])
    for key,value in {'sampling_seeds':[42,43],'frames':33,'steps':20,'CFG':1.,'tile_hw':[256,448],'overlap':64,'conditions':['clean'],'training_updates':0}.items():
        need(old['scope'].get(key)==new['scope'].get(key)==value,'Native inference protocol changed: '+key)
    old_roles=[r for r in old['roles'] if r['label']=='shared_initial'];new_roles=[r for r in new['roles'] if r['label']=='shared_initial']
    need(len(old_roles)==len(new_roles)==1,'Missing shared-initial role')
    a,b=old_roles[0],new_roles[0]
    need(all(a[k]==b[k] for k in ('arm','weight','step','update','config','variant'))
         and b['arm']=='shared' and b['step']==8192 and b['update']==0 and b['weight']==0
         and fingerprint(a['checkpoint'])==fingerprint(b['checkpoint']),'Shared Base role differs')
    verify.verify(a['checkpoint']);verify.verify(b['checkpoint'])
    source=source_compatibility(old,new,verify)
    keys=('checkpoint_sha256','format','step','variant','config','official_repo_id','official_revision',
        'restored_parameter_count','restored_parameter_names','checkpoint_tensor_dtypes','destination_tensor_dtypes',
        'optimizer_state_restored','scheduler_state_restored','rng_state_restored','inference_weights',
        'training_only_state_restored','EMA_weights_selected','training_only_fields_present')
    load_reference=None;state_reference=None;load_files=[]
    for run in (old_run,new_run):
        loads=sorted((run/'roles/shared_initial').glob('load_audit_*.json'));need(loads,'Missing shared Base load audit')
        for path in loads:
            current=read(path);verify.verify(binding(path));projection={k:current[k] for k in keys}
            if load_reference is None:load_reference=projection
            need(projection==load_reference and current['inference_weights']=='student'
                 and current['training_only_fields_present']==[] and current['EMA_weights_selected'] is False
                 and current['training_only_state_restored'] is False,'Shared Base load semantics changed')
            load_files.append(binding(path))
        states=sorted((run/'roles/shared_initial').glob('state_audit_*.json'));need(states,'Missing state audit')
        for path in states:
            current=read(path);verify.verify(binding(path))
            need(current['training_updates']==0 and current['inference_weights']=='student'
                 and current['trainable_before']==current['trainable_after'],'Evaluation changed shared Base')
            if state_reference is None:state_reference=current['trainable_before']
            need(current['trainable_before']==state_reference,'Loaded shared Base state changed')
    indices=[]
    for run,protocol,state,count in [(old_run,old,os,1024),(new_run,new,ns,640)]:
        observations=read(run/'observations.json');need(observations['protocol_sha256']==state['protocol_sha256'],'Observation protocol differs')
        rows=observations['observations'];wanted={(r['label'],c['record']['id'],s) for r in protocol['roles'] for c in protocol['cohort'] for s in (42,43)}
        need(len(rows)==len(wanted)==count and {(r['role'],r['record_id'],r['sampling_seed']) for r in rows}==wanted,'Incomplete/duplicate entire validation matrix')
        index={(r['record_id'],r['sampling_seed']):r for r in rows if r['role']=='shared_initial'}
        need(len(index)==64,'Missing shared-initial restores');indices.append(index)
    comparisons=[]
    for item in new['cohort']:
        rid=item['record']['id']
        for sample in (42,43):
            left=row(old_run,os['protocol_sha256'],rid,sample,indices[0][rid,sample],verify)
            right=row(new_run,ns['protocol_sha256'],rid,sample,indices[1][rid,sample],verify)
            for payload in (left,right):
                need(payload['source_metadata']['source_id']==item['record']['source_id']
                     and payload['source_metadata']['source_frame_indices']==item['source_indices'],'Executed source window differs')
            need(left['restoration']==right['restoration'] and left['VAE_ledger']==right['VAE_ledger'],'Restoration/VAE metadata changed')
            hashes=[]
            for original,current in zip(left['PNG_artifacts'],right['PNG_artifacts']):
                x,y=verify.verify(original),verify.verify(current)
                need(x['sha256']==y['sha256'],'Shared Base prediction PNG bytes differ: '+rid+' seed'+str(sample))
                hashes.append(x['sha256'])
            comparisons.append({'record_id':rid,'sampling_seed':sample,'frames':33,'PNG_sha256':hashes})
    verify.unchanged();need(len(comparisons)==64 and sum(x['frames'] for x in comparisons)==2112,'Comparison coverage differs')
    return {'schema':'wan_trd_shared_initial_regression_v1','status':'passed_exact_shared_initial_regression',
        'producer':binding(__file__),'historical_protocol':binding(old_run/'protocol.json'),'validation_protocol':binding(new_run/'protocol.json'),
        'core_verification':binding(verification_json),'source_compatibility':source,'load_audits':load_files,
        'restore_pairs_compared':64,'PNG_pairs_compared':2112,'input_NPY_bindings_checked':128,
        'all32_sources_included':True,'replaced_source_from_previous_round_also_included':True,
        'compared':comparisons,'official_model_tensor_bytes_rehashed_here':False,
        'official_model_policy':'Bindings/sizes/JSON compared; complete tensor hash verification belongs to the bound core verify receipt.',
        'metric_values_used_for_selection_or_quality_conclusion':False,'quality_conclusion':None,
        'GPU_work':False,'additional_restores':0,'numerical_tolerance':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-run',type=Path,required=True);parser.add_argument('--validation-run',type=Path,required=True)
    parser.add_argument('--verification-json',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);need(not args.output.exists(),'Output already exists')
    try:value=compare(args.historical_run,args.validation_run,args.verification_json);code=0
    except Exception as error:
        value={'schema':'wan_trd_shared_initial_regression_failure_v1','status':'failed_strict_regression',
               'error':str(error),'traceback':traceback.format_exc(),'GPU_work':False,'additional_restores':0,
               'automatic_retry':False,'tolerance_adjusted':False};code=1
    value['at_UTC']=datetime.now(timezone.utc).isoformat();args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':value['status'],'output':binding(args.output)},indent=2));return code


if __name__=='__main__':raise SystemExit(main())
