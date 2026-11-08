"""Unbound CPU data integrity/held-out clean assets helper; never loads a model.

The metadata cohort is frozen before test model evaluation. Automated thumbnails
are only an integrity screen and are not model outputs or quality selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def binding(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def verify_binding(spec):
    need(binding(spec['path'])==spec,'Input file binding changed')
    return Path(spec['path'])


def write(path,value):
    path=Path(path);need(not path.exists(),'Use a fresh immutable output path')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def header(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        need(cap.isOpened(),'Cannot open source header')
        out={'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'fps':float(cap.get(cv2.CAP_PROP_FPS))}
        need(math.isfinite(out['fps']) and out['fps']>0,'Invalid source FPS')
        return out
    finally:cap.release()


def test_plan(manifest_path,output,*,seed=20261108):
    """Source-hash order, one record/source; source headers only, no decoded pixels."""
    from vsr import data
    manifest=read(manifest_path);data.validate_manifest(manifest)
    groups={}
    for record in manifest['records']:
        if record['split']=='test':groups.setdefault(record['source_id'],[]).append(record)
    selected=[];rejected=[]
    for source in sorted(groups,key=lambda s:(data._digest(['fixed_clean_test_source',seed,s]),s)):
        for record in sorted(groups[source],key=lambda r:(data._digest(['fixed_clean_test_record',seed,r['id']]),r['id'])):
            h=header(record['path'])
            if h['frames']<33 or h['height']<128 or h['width']<224:
                rejected.append({'record_id':record['id'],'source_id':source,'header':h,'reason':'minimum33frames_HR128x224'})
                continue
            need(sha(record['path'])==record['sha256'],'Selected raw source changed')
            start=int(data._digest(['window',seed,record['id']])[:16],16)%(h['frames']-32)
            ratio=max(128/h['height'],224/h['width']);rh=max(128,round(h['height']*ratio));rw=max(224,round(h['width']*ratio))
            selected.append({'record':record,'cached_header':h,'source_indices':list(range(start,start+33)),
                             'source_hw':[h['height'],h['width']],'resize_hw':[rh,rw],
                             'crop_tlhw':[(rh-128)//2,(rw-224)//2,128,224]})
            break
        if len(selected)==32:break
    need(len(selected)==32,'Fewer than32 eligible held-out sources')
    all_sources={s:{r['source_id'] for r in manifest['records'] if r['split']==s} for s in ('train','val','test')}
    all_hashes={s:{r['sha256'] for r in manifest['records'] if r['split']==s} for s in ('train','val','test')}
    need(all(not all_sources[a]&all_sources[b] and not all_hashes[a]&all_hashes[b]
             for a,b in (('train','val'),('train','test'),('val','test'))),'Original split source/byte leakage')
    result={'schema':'vsr_fixed_clean_cohort_v1','split':'test','project_root':manifest['raw_root'],
        'manifest':binding(manifest_path),'helper':binding(__file__),'seed':seed,
        'selection_rule':'SHA256(source namespace,seed,sourceID), first header-eligible SHA256(record namespace,seed,recordID); one/source; no pixel or model score',
        'config':{'frames':33,'height':128,'width':224,'scale':4,'temporal_stride':1,'seed':seed},
        'cohort':selected,'header_rejected':rejected,
        'test_prior_exposure':'historical_models_previously_evaluated','pristine_test_claimed':False,
        'pixel_access_at_selection':False,'model_scores_accessed_at_selection':False,
        'data_independence':{'all_segments_of_original_source_same_split':True,'source_split_overlap':0,
                            'cross_split_identical_raw_SHA256':0,'near_duplicate_status':'separate_automated_thumbnail_screen_required',
                            'limitation':'Source ID and byte hashes do not exclude transcoded/cropped/reuploaded content.'}}
    result['seal_digest']=data._digest(result);write(output,result)
    return {'status':'metadata_only_test32_frozen','cohort':binding(output),'selected':len(selected),'test_pixels_decoded':False}


def materialize(plan_path,study_path,selection_path,output):
    """Only complete validation-derived settings unlock formal test decoding."""
    import numpy as np
    import os,shutil,copy
    from vsr import data
    plan=read(plan_path);split=plan['split'];need(split in ('val','test'),'Invalid clean split')
    study=selection=None
    if split=='test':
        from vsr import evaluate
        need(study_path is not None and selection_path is not None,'Test requires fixed study and selection receipt')
        study,selection=read(study_path),read(selection_path)
        need(study['evaluation']['test_cohort_plan']==binding(plan_path),'Study froze a different test cohort')
        need(selection==evaluate._ft_selection_payload(selection['validation_run']),
             'Selection receipt must reproduce complete validation before test pixels')
        need(selection['schema']=='fixed_teacher_validation_selection_v1'
             and selection['study_file']==binding(study_path)
             and selection['test_cohort_plan']==binding(plan_path)
             and selection['selected_after_complete_validation'] is True
             and selection['test_pixels_read_by_selection'] is False,'Invalid before-test selection')
    else:need(study_path is None and selection_path is None,'Validation producer does not consume method selection')
    need(plan['schema']=='vsr_fixed_clean_cohort_v1'
         and plan['seal_digest']==data._digest({k:v for k,v in plan.items() if k!='seal_digest'}),'Frozen clean metadata changed')
    reuse={};reused=0
    if split=='val' and plan.get('reuse_original_validation_dataset'):
        reuse_spec=plan['reuse_original_validation_dataset'];verify_binding(reuse_spec)
        reuse={r['record_id']:r for r in read(reuse_spec['path'])['assets']}
    out=Path(output).resolve();need(not out.exists(),'Clean assets need a fresh output directory');out.mkdir(parents=True)
    assets=[];started=time.perf_counter()
    try:
        for item in plan['cohort']:
            record=item['record'];directory=out/record['id'];directory.mkdir();old=reuse.get(record['id'])
            if old is not None:
                metadata=copy.deepcopy(old['source_metadata'])
                need(metadata['split']=='val' and metadata['source_id']==record['source_id']
                     and metadata['source_sha256']==record['sha256'] and metadata['source_indices']==item['source_indices']
                     and metadata['crop_tlhw']==item['crop_tlhw'],'Historical validation RGB differs from retained frozen window')
                row={'record_id':record['id'],'source_id':record['source_id'],'split':split,'metadata':metadata,'reused_original_RGB_bytes':True}
                for key in ('HR','clean'):
                    source=Path(old[key]['path'])
                    if not source.is_absolute():source=Path(plan['original_assets_root'])/source
                    need(sha(source)==old[key]['sha256'],'Historical validation RGB changed')
                    destination=directory/f'{key}.npy'
                    try:os.link(source,destination)
                    except OSError:shutil.copyfile(source,destination)
                    row[key]=copy.deepcopy(old[key])|{'path':str(destination)}
                reused+=1
            else:
                clip=data.decode_clip(record,num_frames=33,height=128,width=224,scale=4,
                                      seed=item.get('decode_seed',plan['config']['seed']),temporal_stride=1)
                metadata=clip['metadata']
                need(metadata['split']==split and metadata['source_indices']==item['source_indices']
                     and metadata['crop_tlhw']==item['crop_tlhw'] and metadata['resize_hw']==item['resize_hw']
                     and metadata['source_hw']==item['source_hw'] and clip['fps']==item['cached_header']['fps'],
                     'Decoded source/window/crop/FPS differs from frozen metadata')
                row={'record_id':record['id'],'source_id':record['source_id'],'split':split,'metadata':metadata,'reused_original_RGB_bytes':False}
                for key,tensor in (('HR',clip['hr']),('clean',clip['lr'])):
                    array=tensor.detach().cpu().contiguous().numpy();path=directory/f'{key}.npy'
                    np.save(path,array,allow_pickle=False)
                    row[key]={'path':str(path),'sha256':sha(path),'rgb_f32_sha256':hashlib.sha256(array.tobytes()).hexdigest()}
            assets.append(row)
        result={'schema':'vsr_fixed_clean_dataset_v1','status':'completed_CPU_clean_assets','split':split,
            'config':plan['config'],'cohort_seal':binding(plan_path),'cohort_seal_digest':plan['seal_digest'],
            'materialization':{'selection_frozen_before_decode':split=='test','validation_assets_before_training':split=='val',
                               'CPU_only':True,'seconds':time.perf_counter()-started,'reused_original_RGB_records':reused,'newly_decoded_records':len(assets)-reused},
            'assets':assets,'test_prior_exposure':'historical_models_previously_evaluated','pristine_test_claimed':False}
        if split=='test':result.update(selection_receipt=binding(selection_path),study_file=binding(study_path))
        write(out/'dataset.json',result)
        reader=data.FixedCleanReader(out/'dataset.json',expected_split=split,cohort_seal=plan_path);reader.close()
        return {'status':'completed_CPU_clean_assets','dataset':binding(out/'dataset.json'),'records':len(assets),'reused_original_RGB_records':reused,'GPU_used':False}
    except BaseException as error:
        write(out/'failure.json',{'status':'incomplete','completed_records':len(assets),'error':str(error)})
        raise


FINGERPRINT_RULE={'schema':'vsr_automatic_thumbnail_integrity_v1','positions':'source clip0/mid/end plus bound33 training/evaluation window0/16/32; unique source-frame targets',
    'thumbnail_hw':[64,112],'grayscale':'OpenCV RGB2GRAY float01','dHash_bits':64,'pHash_bits':63,
    'candidate_hamming_max':8,'candidate_rule':'dHash<=8 OR pHash<=8 for any cross-split frame pair',
    'low_texture_gray_std_below':.025,'low_texture_canny_fraction_below':.005,
    'confirm_SSIM_min':.98,'confirm_MSE_max':.0025,
    'confirmed_requires':'at least2 non-low-texture, distinct time-position matches from each clip; never decide from all-black/low-texture alone',
    'scope':'automatic data integrity only; no model scores, no manual test viewing',
    'limitations':['Three thumbnails can miss offset/short-overlap/cropped/reframed/reordered or semantically duplicated clips.',
                   'No candidate is not proof of exhaustive visual independence.','No automatic reassignment of historical source splits.']}


def _fingerprints(record,window_indices=None):
    import cv2
    import numpy as np
    from scipy.fft import dctn
    need(sha(record['path'])==record['sha256'],'Raw integrity source changed since frozen manifest')
    h=header(record['path']);cap=cv2.VideoCapture(record['path']);outputs=[]
    try:
        targets=[(0,'source_clip'),((h['frames']-1)//2,'source_clip'),(h['frames']-1,'source_clip')]
        if window_indices is not None:
            need(len(window_indices)==33 and window_indices==list(range(window_indices[0],window_indices[0]+33)),
                 'Fingerprint window must be the exact frozen33stride1 window')
            targets += [(window_indices[i],'bound33_window') for i in (0,16,32)]
        targets=list({index:(index,kind) for index,kind in targets}.values())
        for position,(index,kind) in enumerate(targets):
            need(0<=index<h['frames'],'Fingerprint target outside source')
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();need(ok,'Cannot decode automatic integrity thumbnail')
            rgb=cv2.cvtColor(cv2.resize(bgr,(112,64),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            dsmall=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA);dbits=(dsmall[:,1:]>dsmall[:,:-1]).reshape(-1)
            psmall=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA);coef=dctn(psmall,type=2,norm='ortho')[:8,:8].reshape(-1)[1:]
            pbits=coef>np.median(coef)
            pack=lambda bits:sum(int(value)<<i for i,value in enumerate(bits))
            edges=float((cv2.Canny(rgb,50,150)>0).mean());std=float(gray.std())
            outputs.append({'position':position,'sample_kind':kind,'target_source_frame':index,'gray':gray,
                'dhash':pack(dbits),'phash':pack(pbits),'gray_std':std,'edge_fraction':edges,
                'low_texture':std<.025 or edges<.005})
        return outputs
    finally:cap.release()


def near_duplicates(manifest_path,training_cache_root,test_plan_path,output,*,val_plan_path=None):
    """Train1641 + frozenval32 + prospective test32, CPU only, no model outputs."""
    import numpy as np
    from skimage.metrics import structural_similarity
    from vsr import data
    manifest,selection,plan=read(manifest_path),read(Path(training_cache_root)/'selection.json'),read(test_plan_path)
    data.validate_manifest(manifest)
    need(plan['manifest']==binding(manifest_path) and plan['split']=='test'
         and plan['seal_digest']==data._digest({k:v for k,v in plan.items() if k!='seal_digest'}),
         'Integrity screen requires the original immutable manifest and planned test cohort')
    need(selection.get('test_accessed') is False,'Training cache declared test access')
    val_ids=selection['validation_ids']
    if val_plan_path is not None:
        val=read(val_plan_path)
        need(val['split']=='val' and val['manifest']==binding(manifest_path)
             and val['seal_digest']==data._digest({k:v for k,v in val.items() if k!='seal_digest'}),'Validation metadata changed')
        val_ids=[i['record']['id'] for i in val['cohort']]
    records={r['id']:r for r in manifest['records']};ids=selection['training_ids']+val_ids+[i['record']['id'] for i in plan['cohort']]
    need(len(ids)==len(set(ids)),'Selected integrity source records overlap')
    out=Path(output).resolve();need(not out.exists(),'Use a fresh immutable integrity audit directory');out.mkdir(parents=True)
    write(out/'protocol.json',{'rule':FINGERPRINT_RULE,'helper':binding(__file__),'manifest':binding(manifest_path),
                             'source_selection':binding(Path(training_cache_root)/'selection.json'),'test_plan':binding(test_plan_path),
                             'val_plan':binding(val_plan_path) if val_plan_path else None,
                             'record_ids':ids,'frozen_before_thumbnail_decode':True})
    windows={i['record']['id']:i['source_indices'] for i in plan['cohort']}
    if val_plan_path is not None:windows.update({i['record']['id']:i['source_indices'] for i in val['cohort']})
    source_seal=read(Path(training_cache_root)/'cache_asset_seal.json')['artifact_sha256']
    for i,record_id in enumerate(selection['training_ids']+selection['validation_ids']):
        if record_id not in ids or record_id in windows:continue
        relative=f'clip_{i}/metadata.json';path=Path(training_cache_root)/relative
        need(sha(path)==source_seal[relative],'Original33 training window metadata changed')
        meta=read(path);need(meta['record_id']==record_id,'Original cache ID/window mismatch')
        windows[record_id]=meta['source_indices']
    fingerprints=[];started=time.perf_counter()
    for index,record_id in enumerate(ids):
        record=records[record_id];frames=_fingerprints(record,windows[record_id]);fingerprints.append((record,frames))
        if index%25==0:print(json.dumps({'status':'CPU_integrity_thumbnail_decode','completed':index+1,'total':len(ids)}),flush=True)
    candidates=[];confirmed=[];unresolved=[]
    for a,(ra,fa) in enumerate(fingerprints):
        for rb,fb in fingerprints[a+1:]:
            if ra['split']==rb['split']:continue
            matches=[]
            for aa in fa:
                for bb in fb:
                    dh=(aa['dhash']^bb['dhash']).bit_count();ph=(aa['phash']^bb['phash']).bit_count()
                    if min(dh,ph)>8:continue
                    ssim=float(structural_similarity(aa['gray'],bb['gray'],data_range=1.));mse=float(np.mean((aa['gray']-bb['gray'])**2))
                    matches.append({'a_position':aa['position'],'b_position':bb['position'],'dHash_distance':dh,'pHash_distance':ph,
                        'SSIM':ssim,'MSE':mse,'low_texture':aa['low_texture'] or bb['low_texture'],
                        'strong':not aa['low_texture'] and not bb['low_texture'] and ssim>=.98 and mse<=.0025})
            if not matches:continue
            item={'a':{k:ra[k] for k in ('id','source_id','split','sha256')},'b':{k:rb[k] for k in ('id','source_id','split','sha256')},'matches':matches}
            candidates.append(item);strong=[m for m in matches if m['strong']]
            if len({m['a_position'] for m in strong})>=2 and len({m['b_position'] for m in strong})>=2:confirmed.append(item)
            elif strong:unresolved.append(item)
    compact=[{'record_id':r['id'],'source_id':r['source_id'],'split':r['split'],'frames':[{k:v for k,v in f.items() if k!='gray'} for f in fs]}
             for r,fs in fingerprints]
    write(out/'fingerprints.json',compact)
    result={'schema':'vsr_thumbnail_independence_audit_v1','status':'flagged_requires_data_review' if confirmed or unresolved else 'no_multiframe_duplicate_flag_in_screen',
        'protocol':binding(out/'protocol.json'),'fingerprints':binding(out/'fingerprints.json'),'records':len(ids),
        'split_counts':{s:sum(r['split']==s for r,_ in fingerprints) for s in ('train','val','test')},
        'candidate_clip_pairs':len(candidates),'confirmed_multiframe_flags':confirmed,'unresolved_oneframe_flags':unresolved,
        'low_texture_matches_alone_never_declared_duplicate':True,'no_historical_split_modified':True,
        'automatic_test_pixels_accessed_for_integrity_only':True,'model_or_human_quality_selection':False,
        'elapsed_CPU_seconds':time.perf_counter()-started,'rule':FINGERPRINT_RULE}
    write(out/'summary.json',result);write(out/'all_candidates.json',candidates)
    return {'status':result['status'],'summary':binding(out/'summary.json'),'flags':len(confirmed),'unresolved':len(unresolved)}


def repair_cohorts(manifest_path,training_cache_root,old_val_seal,old_val_dataset,initial_test_plan,initial_audit,output,*,seed=20261108):
    """Keep unchanged original splits, remove flagged eval sources, fixed hash refill."""
    from vsr import data
    manifest=read(manifest_path);data.validate_manifest(manifest);records={r['id']:r for r in manifest['records']}
    original={'val':read(old_val_seal)['cohort'],'test':read(initial_test_plan)['cohort']}
    excluded={'val':set(),'test':set()};evidence=[binding(initial_audit)]
    out=Path(output).resolve();need(not out.exists(),'Cohort repair is prospective, use a fresh directory');out.mkdir(parents=True)
    rule={'priority':'train unchanged; flagged train-val removes val; train-test removes test; val-test removes test',
          'retention':'keep original nonexcluded31val/29test sources and exact windows; only refill missing sources',
          'refill':'fixed SHA256 source namespace/seed/sourceID order then record namespace/seed/recordID, first eligible header',
          'no_method_scores_or_manual_quality':True,'stop':'complete1705record screen has no strong multiframe OR unresolved strong oneframe flag',
          'near_duplicate_limitations':FINGERPRINT_RULE['limitations']}
    def exclusions(audit):
        before={s:set(v) for s,v in excluded.items()}
        for pair in audit['confirmed_multiframe_flags']+audit['unresolved_oneframe_flags']:
            a,b=pair['a'],pair['b']
            remove=(a if a['split']=='test' else b) if 'test' in (a['split'],b['split']) else (a if a['split']=='val' else b)
            need(remove['split'] in ('val','test'),'Never remove/reassign an original training source')
            excluded[remove['split']].add(remove['source_id'])
        return excluded!=before
    exclusions(read(initial_audit))
    for iteration in range(len({r['source_id'] for r in manifest['records'] if r['split']!='train'})):
        directory=out/f'iteration_{iteration:03d}';directory.mkdir();paths={}
        for split in ('val','test'):
            kept=[x for x in original[split] if x['record']['source_id'] not in excluded[split]];cohort=[]
            for old in kept:
                record=records[old['record']['id']];h=header(record['path']);decode_seed=20261102 if split=='val' else read(initial_test_plan)['config']['seed']
                start=int(data._digest(['window',decode_seed,record['id']])[:16],16)%(h['frames']-32)
                need(old['source_indices']==list(range(start,start+33)),'Retained original clean window changed')
                ratio=max(128/h['height'],224/h['width']);rh=max(128,round(h['height']*ratio));rw=max(224,round(h['width']*ratio))
                cohort.append({'record':record,'cached_header':h,'source_indices':old['source_indices'],'decode_seed':decode_seed,
                    'source_hw':[h['height'],h['width']],'resize_hw':[rh,rw],'crop_tlhw':[(rh-128)//2,(rw-224)//2,128,224],
                    'retained_original_window':True})
            used={x['record']['source_id'] for x in cohort};groups={}
            for record in manifest['records']:
                if record['split']==split and record['source_id'] not in excluded[split]|used:groups.setdefault(record['source_id'],[]).append(record)
            for source in sorted(groups,key=lambda source:(data._digest([f'fixed_clean_{split}_source',seed,source]),source)):
                if len(cohort)==32:break
                for record in sorted(groups[source],key=lambda r:(data._digest([f'fixed_clean_{split}_record',seed,r['id']]),r['id'])):
                    h=header(record['path'])
                    if h['frames']<33 or h['height']<128 or h['width']<224:continue
                    need(sha(record['path'])==record['sha256'],'New evaluation raw source changed')
                    start=int(data._digest(['window',seed,record['id']])[:16],16)%(h['frames']-32)
                    ratio=max(128/h['height'],224/h['width']);rh=max(128,round(h['height']*ratio));rw=max(224,round(h['width']*ratio))
                    cohort.append({'record':record,'cached_header':h,'source_indices':list(range(start,start+33)),'decode_seed':seed,
                        'source_hw':[h['height'],h['width']],'resize_hw':[rh,rw],'crop_tlhw':[(rh-128)//2,(rw-224)//2,128,224],
                        'retained_original_window':False});break
            need(len(cohort)==len({x['record']['source_id'] for x in cohort})==32,'Insufficient eligible nonflagged eval sources')
            plan={'schema':'vsr_fixed_clean_cohort_v1','split':split,'project_root':manifest['raw_root'],
                'manifest':binding(manifest_path),'helper':binding(__file__),'config':{'frames':33,'height':128,'width':224,'scale':4,'temporal_stride':1,'seed':seed},
                'cohort':cohort,'excluded_source_ids':sorted(excluded[split]),'exclusion_evidence':evidence,'repair_rule':rule,
                'original_cohort':binding(old_val_seal if split=='val' else initial_test_plan),
                'historical_split_assignment_unchanged':True,'cohort_selection_uses_pixels_only_for_automated_integrity':True,
                'model_scores_accessed_at_selection':False,'test_prior_exposure':'historical_models_previously_evaluated','pristine_test_claimed':False,
                'data_independence':{'all_original_source_segments_same_split':True,'source_ID_split_overlap':0,'raw_SHA_split_overlap':0,
                                     'known_visual_duplicate_sources_excluded':True,'screen_rule':FINGERPRINT_RULE,
                                     'status':'prospective_cohort_awaits_separate_complete_integrity_receipt'}}
            if split=='val':plan.update(reuse_original_validation_dataset=binding(old_val_dataset),original_assets_root=str(Path(old_val_dataset).resolve().parents[2]))
            plan['seal_digest']=data._digest(plan);paths[split]=directory/f'{split}_plan.json';write(paths[split],plan)
        result=near_duplicates(manifest_path,training_cache_root,paths['test'],directory/'integrity',val_plan_path=paths['val'])
        audit=read(result['summary']['path'])
        if not audit['confirmed_multiframe_flags'] and not audit['unresolved_oneframe_flags']:
            receipt={'status':'repaired_evaluation_cohorts_frozen_no_flag_in_declared_screen','val_plan':binding(paths['val']),'test_plan':binding(paths['test']),
                'integrity_audit':result['summary'],'excluded_sources':{s:sorted(v) for s,v in excluded.items()},'rule':rule,
                'iterations':iteration+1,'original_training_records_unchanged':True,'Base_initialization_unchanged':True,
                'source_splits_reassigned':False,'old_val_cohort_use':'historical_only_not_this_round_heldout_metrics'}
            write(out/'completion.json',receipt);return receipt
        evidence=evidence+[result['summary']];need(exclusions(audit),'Integrity flags did not produce a new deterministic exclusion')
    raise ValueError('Exhausted finite original evaluation source pool')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('test-plan');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20261108)
    p=sub.add_parser('val-materialize');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('repair-cohorts');p.add_argument('--manifest',required=True);p.add_argument('--training-cache-root',required=True);p.add_argument('--old-val-seal',required=True);p.add_argument('--old-val-dataset',required=True);p.add_argument('--initial-test-plan',required=True);p.add_argument('--initial-audit',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('test-materialize');p.add_argument('--plan',required=True);p.add_argument('--study',required=True);p.add_argument('--selection-receipt',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('near-duplicates');p.add_argument('--manifest',required=True);p.add_argument('--training-cache-root',required=True);p.add_argument('--test-plan',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='test-plan':result=test_plan(args.manifest,args.output,seed=args.seed)
    elif args.command=='val-materialize':result=materialize(args.plan,None,None,args.output)
    elif args.command=='repair-cohorts':result=repair_cohorts(args.manifest,args.training_cache_root,args.old_val_seal,args.old_val_dataset,args.initial_test_plan,args.initial_audit,args.output)
    elif args.command=='test-materialize':result=materialize(args.plan,args.study,args.selection_receipt,args.output)
    else:result=near_duplicates(args.manifest,args.training_cache_root,args.test_plan,args.output)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
