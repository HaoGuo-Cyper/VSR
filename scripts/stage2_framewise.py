#!/usr/bin/env python3
"""Freeze/materialize the predeclared crop-scale Stage2 pixel-stress val32.

seal reads metadata only, before any source pixels or new scores. materialize
uses CPU only and creates fresh inputs, never predictions. selftest uses only
synthetic pixels/scores. Internal test/external data are not accepted.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
from stage2_metrics import need, score_pools, summarize_g2, TLP


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def fresh_json(path, value):
    path = Path(path)
    need(not path.exists(), "Refusing to overwrite: "+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def local_seed(namespace, seed, record, frame=None):
    value = [namespace, seed, record] + ([] if frame is None else [frame])
    return int.from_bytes(bytes.fromhex(digest(value))[:8], "big") % 2**63


def mask_for(record, cfg):
    starts = cfg["mask"]["start_candidates"]
    # Use the full unsigned first8bytes for start selection, before RNG's
    # signed-range reduction (which would change modulo5 for some records).
    key = [cfg["mask"]["namespace"], cfg["mask"]["seed"], record]
    start = starts[int(digest(key)[:16], 16) % len(starts)]
    return [start <= i < start+8 for i in range(33)]


def codec_binding():
    path = shutil.which("ffmpeg")
    need(path is not None, "FFmpeg unavailable; codec condition cannot be silently replaced")
    encoders = subprocess.run([path, "-hide_banner", "-encoders"], check=True, capture_output=True, text=True).stdout
    need(" libx264 " in encoders, "libx264 unavailable; no codec fallback allowed")
    version = subprocess.run([path, "-version"], check=True, capture_output=True, text=True).stdout
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path), "version": version}


def seal_protocol(root, config_path):
    cfg = read(config_path)
    need(cfg["split"] == "val" and cfg["frames"] == 33 and cfg["temporal_stride"] == 1
         and (cfg["height"], cfg["width"], cfg["scale"]) == (128, 224, 4), "Only the approved val crop protocol is supported")
    need(cfg['conditions']==['blur','noise','codec_allintra','drop_repeat','drop_black']
         and cfg['mask']['start_candidates']==[4,8,12,16,20] and cfg['mask']['span_pixel_frames']==8
         and cfg['codec_allintra']['encoder']=='libx264' and cfg['codec_allintra']['gop']==cfg['codec_allintra']['keyint_min']==1
         and cfg['codec_allintra']['b_frames']==cfg['codec_allintra']['scene_cut']==0,
         'Frozen mask/conditions/intra-only codec settings are required')
    manifest_path = root / "manifests/hq_vsr.json"
    manifest = read(manifest_path)
    need(digest({k:v for k,v in manifest.items() if k != "manifest_sha256"}) == manifest["manifest_sha256"], "Source split manifest digest changed")
    by_source, by_hash = {}, {}
    for record in manifest["records"]:
        by_source.setdefault(record["source_id"], set()).add(record["split"])
        by_hash.setdefault(record["sha256"], set()).add(record["split"])
    need(all(len(x)==1 for x in [*by_source.values(), *by_hash.values()]), "Training/heldout source or exact-byte leakage")
    records = sorted([r for r in manifest["records"] if r["split"] == "val"],
                     key=lambda r: hashlib.sha256(f"val:{cfg['data_seed']}:{r['id']}".encode()).hexdigest())[:32]
    need(len(records) == len({r['source_id'] for r in records}) == 32, "Require the original32 source-held-out validation clips")
    inventory_path = root / "manifests/hq_temporal_stride4_inventory.json"
    headers = {r["id"]:r for r in read(inventory_path)["records"]}
    cohort = []
    for record in records:
        header = headers[record["id"]]
        need(header["split"] == "val" and header["raw_sha256"] == record["sha256"]
             and header["header"]["frame_count"] >= 33, "Header eligibility/identity mismatch")
        total = header["header"]["frame_count"]
        start = int(digest(["window", cfg["data_seed"], record["id"]])[:16], 16) % (total-33+1)
        mask = mask_for(record["id"], cfg)
        selected = [i for i,x in enumerate(mask) if x]
        cohort.append({"record": record, "cached_header": header["header"], "source_indices": list(range(start,start+33)),
                       "mask": mask, "masked_indices": selected, "neighbor_indices": [selected[0]-1, selected[-1]+1],
                       "noise_seeds_by_masked_frame": {str(i): local_seed(cfg['noise']['namespace'], cfg['noise']['seed'], record['id'], i) for i in selected}})
    base_configs = {}
    for seed in cfg["training_seeds"]:
        path = root / f"configs/stage2_20261105/base_seed{seed}.json"
        base = read(path)
        need(base["seed"] == cfg["data_seed"] and base["training"].get("seed",base["seed"]) == seed
             and base["data"]["num_frames"] == 33 and base["data"]["temporal_stride"] == 1
             and base["data"]["height"] == 128 and base["data"]["width"] == 224,
             "New Base and validation protocols differ")
        base_configs[str(seed)] = {"path":str(path),"sha256":file_sha(path)}
    result = {"schema":"stage2_framewise_cohort_seal_v1", "status":"frozen_before_source_pixels_and_new_scores",
              "project_root":str(root),"config":cfg,"config_path":str(config_path),"config_sha256":file_sha(config_path),
              "manifest":str(manifest_path),"manifest_file_sha256":file_sha(manifest_path),
              "manifest_digest":manifest["manifest_sha256"],"cached_header_inventory_sha256":file_sha(inventory_path),
              "base_configs":base_configs,"codec":codec_binding(),"cohort":cohort,
              "CPU_runtime_packages":{name:importlib.metadata.version(name) for name in ['torch','numpy','opencv-python-headless']},
              "implementation_bindings":{str(p):file_sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('stage2_metrics.py').resolve(),root/'vsr/data.py']},
              "source_pixels_decoded":False,"new_scores_read":False,"test_or_external_pixels_read":False,
              "raw_source_hashes": "copied from sealed manifest now; decode_clip rehashes before later CPU materialization"}
    result["seal_digest"] = digest(result)
    return result


def validate_seal(seal):
    need(digest({k:v for k,v in seal.items() if k!='seal_digest'}) == seal['seal_digest'], "Cohort seal digest mismatch")
    need(file_sha(seal['config_path']) == seal['config_sha256'] and file_sha(seal['manifest']) == seal['manifest_file_sha256'],
         "Sealed protocol/catalogue changed")
    need(all(file_sha(path)==value for path,value in seal['implementation_bindings'].items()), "Sealed implementation changed")
    need(codec_binding() == seal['codec'], "Sealed FFmpeg executable/version changed; no fallback")
    need(all(importlib.metadata.version(name)==version for name,version in seal['CPU_runtime_packages'].items()),
         'CPU decode/corruption runtime versions changed')


def codec_frame(frame, fps, cfg, binding):
    import numpy as np
    import torch
    from fractions import Fraction
    h,w = frame.shape[-2:]
    need(h%2==0 and w%2==0, "Codec requires even native LR dimensions; never resize")
    raw = frame.permute(1,2,0).mul(255).round().byte().contiguous().numpy().tobytes()
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    command = [binding['path'],'-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h}','-framerate',f'{rate.numerator}/{rate.denominator}','-i','pipe:0','-frames:v','1',
        '-an','-c:v','libx264','-preset',cfg['codec_allintra']['preset'],'-crf',str(cfg['codec_allintra']['crf']),'-pix_fmt','yuv420p','-threads','1','-g','1',
        '-keyint_min','1','-sc_threshold','0','-bf','0','-map_metadata','-1','-fflags','+bitexact','-flags:v','+bitexact','-f','h264','pipe:1']
    encoded = subprocess.run(command,input=raw,capture_output=True,check=True,timeout=60).stdout
    decode = [binding['path'],'-hide_banner','-loglevel','error','-nostdin','-f','h264','-i','pipe:0','-frames:v','1',
              '-an','-threads','1','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    payload = subprocess.run(decode,input=encoded,capture_output=True,check=True,timeout=60).stdout
    need(len(payload)==h*w*3 and len(encoded)>0, "Codec changed single-frame geometry/count")
    result = torch.from_numpy(np.frombuffer(payload,dtype=np.uint8).copy()).reshape(h,w,3).permute(2,0,1).float().div(255)
    return result, {"encode_args":command,"decode_args":decode,"encoded_sha256":hashlib.sha256(encoded).hexdigest(),
                    "encoded_bytes":len(encoded),"decoded_RGB24_sha256":hashlib.sha256(payload).hexdigest()}


def corrupt_cpu(clean, condition, item, cfg, codec):
    import torch
    import torch.nn.functional as F
    need(clean.device.type=='cpu' and tuple(clean.shape)==(1,3,33,32,56) and clean.dtype==torch.float32,
         "Only authoritative CPU float32 LR crop33x32x56 accepted")
    need(condition in cfg['conditions'] and bool(torch.isfinite(clean).all()) and float(clean.min())>=0 and float(clean.max())<=1,
         "Invalid condition/LR RGB")
    out = clean.clone()
    indices = item['masked_indices']
    details = []
    if condition=='blur':
        radius=cfg['blur']['kernel']//2
        x=torch.arange(-radius,radius+1,dtype=torch.float64)
        k=torch.exp(-x.square()/(2*cfg['blur']['sigma_LR_pixels']**2));k/=k.sum()
        kernel=torch.outer(k,k).float().view(1,1,2*radius+1,2*radius+1).expand(3,1,-1,-1).contiguous()
        selected=clean[0,:,indices].permute(1,0,2,3)
        out[0,:,indices]=F.conv2d(F.pad(selected,(radius,)*4,mode='reflect'),kernel,groups=3).permute(1,0,2,3).clamp(0,1)
    else:
        for frame in indices:
            if condition=='noise':
                g=torch.Generator(device='cpu').manual_seed(item['noise_seeds_by_masked_frame'][str(frame)])
                out[0,:,frame]=(clean[0,:,frame]+torch.randn(clean[0,:,frame].shape,generator=g)*cfg['noise']['sigma_RGB01']).clamp(0,1)
            elif condition=='drop_repeat':out[0,:,frame]=clean[0,:,indices[0]-1]
            elif condition=='drop_black':out[0,:,frame]=0
            else:
                out[0,:,frame],detail=codec_frame(clean[0,:,frame],item['cached_header']['fps'],cfg,codec)
                details.append({"frame":frame,**detail})
    untouched=[i for i in range(33) if not item['mask'][i]]
    need(torch.equal(out[:,:,untouched],clean[:,:,untouched]) and out.shape==clean.shape and bool(torch.isfinite(out).all()),
         "Corruption leaked into clean frames or changed geometry")
    actual=[i for i in range(33) if not torch.equal(out[:,:,i],clean[:,:,i])]
    need(set(actual).issubset(indices), "Actual pixel support escaped mask")
    return out,{"condition":condition,"intended_masked_indices":indices,"actual_changed_indices":actual,
                "unmasked_float32_byte_identity":True,"codec_frames":details,"fallback":False}


def materialize(seal, output):
    validate_seal(seal)
    need(not output.exists(), "CPU asset output must be a new absent directory")
    import numpy as np
    import torch
    need(not torch.cuda.is_initialized(), "CPU materialization must not initialize CUDA")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0,seal['project_root'])
    from vsr.data import decode_clip
    output.mkdir(parents=True)
    assets=[]
    cfg=seal['config']
    try:
        for item in seal['cohort']:
            record=item['record']
            clip=decode_clip(record,num_frames=33,height=128,width=224,scale=4,seed=cfg['data_seed'],temporal_stride=1)
            need(clip['metadata']['source_indices']==item['source_indices'] and clip['metadata']['split']=='val'
                 and clip['metadata']['original_fps']==item['cached_header']['fps'], "Current source window/header differs from frozen metadata")
            folder=output/record['id'];folder.mkdir()
            def store(name,tensor):
                path=folder/(name+'.npy')
                with path.open('xb') as stream:np.save(stream,tensor.detach().cpu().numpy(),allow_pickle=False)
                return {"path":str(path),"sha256":file_sha(path),"shape":list(tensor.shape),"dtype":str(tensor.dtype),
                        "rgb_f32_sha256":hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            row={"record_id":record['id'],"source_id":record['source_id'],"mask":item['mask'],
                 "base_pair_cache_key":clip['cache_key'],
                 "source_metadata":clip['metadata'],"HR":store('HR',clip['hr']),"clean":store('LR_clean',clip['lr']),"conditions":{}}
            for condition in cfg['conditions']:
                value,provenance=corrupt_cpu(clip['lr'],condition,item,cfg,seal['codec'])
                row['conditions'][condition]={"asset":store('LR_'+condition,value),"provenance":provenance}
            fresh_json(folder/'asset_record.json',row);assets.append(row)
        need(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
        fresh_json(output/'dataset.json',{"status":"completed_CPU_assets_no_predictions","cohort_seal_digest":seal['seal_digest'],
                   "config":cfg,"assets":assets,"GPU_used":False,"restoration_or_metrics_run":False})
    except BaseException as error:
        fresh_json(output/'failure.json',{"status":"failed_no_retry","error":str(error),"completed_records":len(assets),"cohort_seal_digest":seal['seal_digest']})
        raise


class FramewiseReader:
    """CPU asset reader for a later, separately frozen inference worker.

    Construction reads only dataset/metadata JSON. load() verifies and reads
    NPY pixels and exposes the original frozen_evaluation stable-HQ contract.
    This class does not construct models, choose checkpoints or run inference.
    """
    def __init__(self, dataset_manifest, seal):
        validate_seal(seal)
        self.seal=seal
        self.root=Path(dataset_manifest).resolve().parent
        self.document=read(dataset_manifest)
        need(self.document['status']=='completed_CPU_assets_no_predictions'
             and self.document['cohort_seal_digest']==seal['seal_digest'] and self.document['config']==seal['config'],
             'CPU asset manifest is not bound to this frozen cohort')
        self.assets={r['record_id']:r for r in self.document['assets']}
        self.cohort={r['record']['id']:r for r in seal['cohort']}
        need(set(self.assets)==set(self.cohort) and len(self.assets)==32,'Reader must expose the complete val32')
        conditions=['clean',*seal['config']['conditions']]
        self._modes={f'Stage2-val-{c}':c for c in conditions}
        self.records={dataset:{record:copy.deepcopy(item['record']) for record,item in self.cohort.items()} for dataset in self._modes}
        self.metadata={'reader':'Stage2FramewiseReader','source_dataset':'HQ-VSR','split':'val',
            'selected_record_count':32,'selected_record_ids':list(self.cohort),'cohort_seal_digest':seal['seal_digest'],
            'data_seed':20261102,'frames':33,'temporal_stride':1,'target_hw':[128,224],'scale':4,
            'conditions':conditions,'source_group_count':32,'training_allowed':False,'test_pixels_allowed':False,
            'pixel_decode_in_constructor':False,'metrics_computed':False}

    def _tensor(self, spec, expected):
        import numpy as np
        import torch
        path=Path(spec['path']).resolve()
        need(path.is_relative_to(self.root) and file_sha(path)==spec['sha256'],'Asset path or NPY file digest changed')
        array=np.load(path,allow_pickle=False)
        need(array.dtype==np.float32 and list(array.shape)==expected and np.isfinite(array).all()
             and float(array.min())>=0 and float(array.max())<=1,'Invalid authoritative RGB asset')
        need(hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()==spec['rgb_f32_sha256'],
             'Authoritative float32 pixel digest changed')
        return torch.from_numpy(array.copy())

    def load(self,dataset,record_id,num_frames=33,start=0):
        need(dataset in self._modes and record_id in self.cohort and num_frames==33 and start==0,
             'Only sealed val33 windows and declared conditions can be read')
        row,item=self.assets[record_id],self.cohort[record_id]
        condition=self._modes[dataset]
        spec=row['clean'] if condition=='clean' else row['conditions'][condition]['asset']
        lr,hr=self._tensor(spec,[1,3,33,32,56]),self._tensor(row['HR'],[1,3,33,128,224])
        metadata=copy.deepcopy(row['source_metadata'])
        need(metadata['split']=='val' and metadata['source_indices']==item['source_indices'],'Asset window or heldout split changed')
        corruption_digest=digest({'condition':condition,'config':self.seal['config'],'mask':item['mask'],
                                  'noise_seeds':item['noise_seeds_by_masked_frame']})
        metadata.update(dataset=dataset,source_dataset='HQ-VSR',sequence_id=record_id,
            manifest_sha256=self.seal['manifest_digest'],source_frame_indices=item['source_indices'],
            prepared_lr_hw=[32,56],prepared_hr_hw=[128,224],fps_source='unchanged source FPS; ordinal stride1',
            framewise_mask=item['mask'],stage2_cohort_seal_digest=self.seal['seal_digest'],
            input_binding={'manifest_sha256':self.seal['manifest_digest'],'base_pair_cache_key':row['base_pair_cache_key'],
                'record_id':record_id,'robustness_mode':'framewise_'+condition,
                'noise_seed':local_seed('stage2-record-noise-master-v1',20261105,record_id),
                'robustness_protocol_sha256':corruption_digest,'base_lr_rgb_sha256':row['clean']['rgb_f32_sha256'],
                'robust_lr_rgb_sha256':spec['rgb_f32_sha256']})
        return {'lr':lr,'hr':hr,'fps':metadata['output_fps'],'metadata':metadata}

    read=load
    def close(self):return None


def selftest(config, output):
    import torch
    cfg=read(config);codec=codec_binding()
    need(not torch.cuda.is_initialized(), "Synthetic tests must be CPU only")
    torch.set_num_threads(2)
    mask=mask_for('synthetic-record',cfg);indices=[i for i,x in enumerate(mask) if x]
    item={'mask':mask,'masked_indices':indices,'cached_header':{'fps':30.0},
          'noise_seeds_by_masked_frame':{str(i):local_seed(cfg['noise']['namespace'],cfg['noise']['seed'],'synthetic-record',i) for i in indices}}
    g=torch.Generator(device='cpu').manual_seed(17);clean=torch.rand((1,3,33,32,56),generator=g)
    transformed={}
    for condition in cfg['conditions']:
        value,provenance=corrupt_cpu(clean,condition,item,cfg,codec)
        transformed[condition]={'shape_preserved':list(value.shape)==list(clean.shape),
                                'unmasked_identity':provenance['unmasked_float32_byte_identity'],
                                'actual_changed_frames':len(provenance['actual_changed_indices'])}
    rgb={'per_frame':[{'psnr_rgb_db':40.0,'ssim_rgb':.9} for _ in range(33)]}
    p={'lpips_per_frame':[.1]*33,'tecogan':{'per_pair':[{'indices':[i,i+1],TLP:1.0} for i in range(2,30)]}}
    base=score_pools(rgb,p,mask)
    for i in indices:rgb['per_frame'][i]['psnr_rgb_db']=30;p['lpips_per_frame'][i]=.3
    bad=score_pools(rgb,p,mask)
    need(abs(bad['raw_harm_gap']['psnr_rgb_db']-10)<1e-10 and abs(bad['raw_harm_gap']['lpips_alex_v0_1']-.2)<1e-10
         and len(bad['temporal']['mask_touch_original_pairs'])==9,'Synthetic metric orientation/pair pools failed')
    cohort=[{'record':{'id':f'r{i}','source_id':f's{i}'},'mask':mask} for i in range(32)]
    fixture={'config':cfg,'cohort':cohort,'seal_digest':'synthetic-only'}
    def rows(corrupt):
        return [{'training_seed':seed,'record_id':f'r{i}','source_id':f's{i}','sampling_seed':sample,
                 'condition':condition,'cohort_seal_digest':'synthetic-only','checkpoint_sha256':str(seed%10)*64,
                 'score':copy.deepcopy(base if condition=='clean' else corrupt)}
                for seed in cfg['training_seeds'] for i in range(32) for sample in [42,43] for condition in ['clean',*cfg['conditions']]]
    passed=summarize_g2(rows(bad),fixture);failed=summarize_g2(rows(base),fixture)
    need(passed['status']=='passed_G2' and failed['status']=='failed_G2_stop','Synthetic gate must distinguish excess damage from none')
    # Intrinsic hard masked frames have raw gaps but zero paired excess.
    intrinsic=rows(base)
    for row in intrinsic:
        row['score']['raw_harm_gap']={'psnr_rgb_db':10.0,'lpips_alex_v0_1':.2}
    need(summarize_g2(intrinsic,fixture)['status']=='failed_G2_stop','Intrinsic difficulty must not pass G2')
    spread_rows=rows(base)
    for row in spread_rows:
        j=cfg['training_seeds'].index(row['training_seed'])+1
        row['score']['raw_harm_gap']={'psnr_rgb_db':float(j),'lpips_alex_v0_1':j*.01}
        if row['condition']!='clean':
            row['score']['raw_harm_gap']['psnr_rgb_db']+=.5
            row['score']['raw_harm_gap']['lpips_alex_v0_1']+=.005
    spread_report=summarize_g2(spread_rows,fixture)
    need(spread_report['status']=='failed_G2_stop'
         and abs(spread_report['gate_metrics']['psnr_rgb_db']['clean_gap_sample_std_ddof1_threshold']-1)<1e-10
         and abs(spread_report['gate_metrics']['lpips_alex_v0_1']['clean_gap_sample_std_ddof1_threshold']-.01)<1e-10,
         'G2 must compare each excess gap against the corresponding clean-gap3-seed sample SD')
    one_metric=rows(bad)
    for row in one_metric:row['score']['raw_harm_gap']['lpips_alex_v0_1']=base['raw_harm_gap']['lpips_alex_v0_1']
    need(summarize_g2(one_metric,fixture)['status']=='failed_G2_stop','Both PSNR and LPIPS must pass')
    try:summarize_g2(rows(bad)[:-1],fixture)
    except ValueError:missing_rejected=True
    else:missing_rejected=False
    need(missing_rejected and not torch.cuda.is_initialized(),'Missing row or CPU-only guard failed')
    fresh_json(output,{'status':'passed_CPU_synthetic_only','corruptions':transformed,'mask':mask,
        'temporal_pairs':{'all':28,'mask_touch':9,'boundary':2},'positive_damage_gate_passed':True,'zero_excess_gate_failed':True,
        'intrinsic_difficulty_not_mistaken_for_damage':True,'missing_observation_rejected':True,
        'nonzero_clean_seed_spread_threshold_checked':True,'both_metrics_required_to_pass':True,
        'real_source_pixels_or_scores_read':False,'GPU_used':False,'codec':codec})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('seal','materialize','selftest','score-gate'))
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/stage2_framewise_20261105.json')
    parser.add_argument('--seal',type=Path)
    parser.add_argument('--scores',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();need(not args.output.exists(),'Output already exists; choose a fresh path')
    root=args.project_root.resolve();old=Path('/home/hugo/Projects/VSR').resolve();output=args.output.resolve()
    need(root!=old and not root.is_relative_to(old) and output.is_relative_to(root) and not output.is_relative_to(old),
         'Outputs must stay within the selected new project; historical root is read-only')
    if args.mode=='materialize':need(output!=root/'runs' and output.is_relative_to(root/'runs'),'CPU assets require a fresh new-project runs/ child')
    if args.mode=='seal':fresh_json(args.output,seal_protocol(args.project_root.resolve(),args.config.resolve()))
    elif args.mode=='selftest':selftest(args.config,args.output)
    else:
        need(args.seal is not None,'Frozen cohort seal required');seal=read(args.seal);validate_seal(seal)
        if args.mode=='materialize':materialize(seal,args.output)
        else:
            need(args.scores is not None,'Existing bound per-frame score JSON required')
            fresh_json(args.output,summarize_g2(read(args.scores)['observations'],seal))
    print(json.dumps({'status':'completed','mode':args.mode,'output':str(args.output),'GPU_launched':False}))


if __name__=='__main__':main()
