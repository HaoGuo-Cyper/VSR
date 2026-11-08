"""CPU display-only comparison of a complete, verified Wan TRD test192.

Uses only predeclared clips, saved prediction PNGs and bound HR/LR test assets.
No inference, model load, score change or clip selection from quality occurs.
Media stays in the original workspace outputs; only text manifests may be public.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

import fixed_teacher_eval_configs as config_tools

WORKSPACE=Path(__file__).resolve().parent.parent
CONTACT_FRAMES=(0,8,16,24,32)
FAMILIES=('baseline','frozen','ema')
HEADER=24
FOOTER=20


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def binding(path):
    import hashlib
    path=Path(path).resolve()
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha}


def checked(item):
    expected={k:item[k] for k in ('path','bytes','sha256')}
    need(binding(item['path'])==expected,'Bound display input changed: '+item['path'])
    return Path(item['path']).resolve()


def _output_directory(path):
    root=(WORKSPACE/'outputs').resolve();path=Path(path).resolve()
    need(root.is_relative_to(WORKSPACE.resolve()) and path!=root and path.is_relative_to(root) and path.name.startswith('video_compare'),
         'Use a new video_compare directory inside the original workspace outputs')
    need(not path.exists(),'Preserve previous video comparisons; choose a new directory')
    return path


def _rgb_frames(value,shape):
    need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and tuple(value.shape)==shape
         and bool(torch.isfinite(value).all()) and float(value.min())>=0 and float(value.max())<=1,
         'Display source must be the finite CPU RGB clean33 tensor')
    return torch.round(value[0].detach().float().clamp(0,1)*255).byte().permute(1,2,3,0).contiguous().numpy()


def _panels(source,predictions,*,clip_id,source_indices,sample,roles,scale):
    hr=_rgb_frames(source['hr'],(1,3,33,128,224))
    _rgb_frames(source['lr'],(1,3,33,32,56))
    up=F.interpolate(source['lr'][0].permute(1,0,2,3),size=(128,224),mode='bicubic',
                     align_corners=False,antialias=True).clamp(0,1).permute(1,0,2,3).unsqueeze(0)
    lr_up=_rgb_frames(up,(1,3,33,128,224))
    arrays=[hr,lr_up,*predictions]
    need(all(a.dtype==np.uint8 and a.shape==(33,128,224,3) for a in arrays),'Comparison frame shape changed')
    labels=['HR ground truth','LR bicubic x4','Baseline',
        'Frozen Base lambda='+format(roles['frozen']['weight'],'.6g'),
        'EMA lambda='+format(roles['ema']['weight'],'.6g')]
    font=ImageFont.load_default();frames=[];fps=float(source['fps'])
    need(math.isfinite(fps) and fps>0,'Original FPS must be positive')
    for i in range(33):
        canvas=Image.new('RGB',(224*5,128+HEADER+FOOTER),(15,15,15));draw=ImageDraw.Draw(canvas)
        for j,(array,label) in enumerate(zip(arrays,labels)):
            canvas.paste(Image.fromarray(array[i]),(j*224,HEADER))
            draw.text((j*224+5,6),label,font=font,fill=(255,255,255))
        draw.text((5,128+HEADER+5),
            f'{clip_id} | frame {i}/32 | relative {i/fps:.4f}s | source frame {source_indices[i]} | sampler {sample}',
            font=font,fill=(255,255,255))
        if scale!=1:canvas=canvas.resize((canvas.width*scale,canvas.height*scale),resample=Image.Resampling.NEAREST)
        frames.append(np.asarray(canvas).copy())
    return np.stack(frames)


def _video_audit(path,*,frames,height,width,fps):
    from vsr.infer import verify_output_video
    import cv2
    result=verify_output_video(path,frames=frames,height=height,width=width,fps=fps)
    capture=cv2.VideoCapture(str(path));count=0
    try:
        need(capture.isOpened(),'Comparison MP4 cannot be decoded')
        while True:
            ok,frame=capture.read()
            if not ok:break
            need(frame.dtype==np.uint8 and frame.shape==(height,width,3),'Decoded MP4 frame geometry changed')
            count+=1
    finally:capture.release()
    need(count==frames,'Encoded MP4 does not decode all33 frames')
    return result|{'fully_decoded_frames':count,'codec_display_only_lossy':True}


def _prediction_frames(ev,run,protocol_sha,role,item,sample,index):
    record=item['record']['id'];obs=index[role['label'],record,sample]
    path=Path(obs['row_metrics_path']).resolve()
    directory=ev._hr_directory(run,role,item,sample)
    need(path==directory/'metrics.json' and binding(path)['sha256']==obs['row_metrics_sha256'],
         'Saved prediction row binding changed')
    row=read(path);marker=read(directory/'complete.json')
    need(marker['protocol_sha256']==protocol_sha and marker['metrics_sha256']==binding(path)['sha256']
         and marker['PNG_count']==33 and row['status']=='completed' and row['training_updates']==0,
         'Saved prediction is not a completed same-protocol row')
    need(row['source_metadata']['split']=='test' and row['source_metadata']['source_frame_indices']==item['source_indices'],
         'Prediction source window changed')
    files=row['PNG_artifacts'];need(len(files)==33,'Need every saved prediction PNG')
    frames=[]
    for ordinal,spec in enumerate(files):
        png=checked(spec)
        need(png==directory/'png'/f'{ordinal:06d}.png','PNG order/path differs from original frame chronology')
        with Image.open(png) as image:
            need(image.mode=='RGB' and image.size==(224,128),'Prediction PNG must be original RGB128x224')
            frames.append(np.asarray(image).copy())
    return np.stack(frames),{'row_metrics':binding(path),'row_marker':binding(directory/'complete.json'),
                             'prediction_PNGs':files}


def deliver(ev,*,test_run,output_dir,integer_scale=1):
    """Require completed trd-verify before reading formal test RGB or making output."""
    need(type(integer_scale) is int and integer_scale in (1,2),'Preview integer scale must be1 or2')
    need(not torch.cuda.is_initialized(),'CPU video delivery needs a fresh process without a CUDA context')
    run=Path(test_run).resolve();output=_output_directory(output_dir)
    protocol=read(run/'protocol.json');state=read(run/'status.json');psha=binding(run/'protocol.json')['sha256']
    need(protocol.get('schema')==ev.TRD_EVALUATION_SCHEMA and protocol.get('phase')=='test'
         and protocol['scope']['total_restores']==192 and state.get('status')=='completed'
         and state.get('restores')==192 and state.get('training_updates')==0 and state['protocol_sha256']==psha,
         'Only complete Wan TRD final test192 may produce this comparison')
    verification=ev.trd_verify(run)
    need(verification['restores']==192 and verification['training_updates']==0,'Complete trd-verify is required')
    cohort=protocol['cohort'];clips=protocol['fixed_video_clip_ids']
    expected=[x['record']['id'] for x in sorted(cohort,key=lambda x:x['record']['source_id'])[:3]]
    need(clips==expected and len(set(clips))==3,'Use exactly the three prospectively declared source-sorted clips')
    need(all(isinstance(record,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,256}',record) for record in clips),
         'Unsafe source clip ID cannot be used as an output component')
    roles={r['family']:r for r in protocol['roles']}
    need(len(protocol['roles'])==3 and set(roles)==set(FAMILIES),'Need baseline plus all three baseline/frozen/EMA students only')
    selected={r['family']:r for r in protocol['validation_final_lock']['final_roles']}
    for family in FAMILIES:
        need(roles[family]['step']==roles[family]['update']==2048
             and roles[family]['checkpoint']==selected[family]['checkpoint']
             and roles[family]['arm']==selected[family]['arm'],'Display changed the prospectively locked final student')
    observations=read(run/'observations.json')['observations']
    index={(r['role'],r['record_id'],r['sampling_seed']):r for r in observations}
    need(len(observations)==len(index)==192 and set(index)=={
        (role['label'],item['record']['id'],sample) for role in roles.values() for item in cohort for sample in (42,43)},
        'Full completed test matrix identities changed')
    ev.verify_imported_runtime(protocol['runtime'])
    need(not torch.cuda.is_initialized(),'Imported display dependencies initialized CUDA')
    from vsr.data import FixedCleanReader,write_video
    cfg=protocol['config'];reader=FixedCleanReader(cfg['dataset'],expected_split='test',cohort_seal=cfg['seal'],
        path_map=cfg.get('path_map'),expected_dataset_sha256=protocol['dataset']['sha256'])
    source_assets={r['record_id']:r for r in protocol['clean_assets']}
    output.mkdir(parents=True);cases=[]
    try:
        for record in clips:
            item=next(x for x in cohort if x['record']['id']==record)
            source=reader.load(record,condition='clean',num_frames=33,start=0)
            need(source['metadata']['split']=='test' and source['metadata']['source_frame_indices']==item['source_indices']
                 and source['fps']==item['cached_header']['fps'],'Bound LR/HR source window or FPS changed')
            asset=source_assets[record]
            inputs={key:binding(asset[key]['path']) for key in ('HR','clean')}
            need(all(inputs[k]['sha256']==asset[k]['sha256'] for k in inputs),'Bound HR/LR NPY changed')
            for sample in (42,43):
                predictions=[];row_inputs={}
                for family in FAMILIES:
                    frames,proof=_prediction_frames(ev,run,psha,roles[family],item,sample,index)
                    predictions.append(frames);row_inputs[family]=proof
                combined=_panels(source,predictions,clip_id=record,source_indices=item['source_indices'],
                    sample=sample,roles=roles,scale=integer_scale)
                directory=output/record/f'seed_{sample}';directory.mkdir(parents=True)
                mp4=directory/'comparison.mp4'
                tensor=torch.from_numpy(combined.copy()).permute(3,0,1,2).unsqueeze(0).float().div_(255)
                write_video(mp4,tensor,float(source['fps']))
                audit=_video_audit(mp4,frames=33,height=combined.shape[1],width=combined.shape[2],fps=float(source['fps']))
                contact=Image.fromarray(np.concatenate([combined[i] for i in CONTACT_FRAMES],axis=0))
                png=directory/'contact_sheet.png';contact.save(png,format='PNG')
                with Image.open(png) as saved:
                    need(saved.mode=='RGB' and saved.size==(combined.shape[2],combined.shape[1]*5),
                         'Contact sheet geometry changed')
                cases.append({'record_id':record,'source_id':item['record']['source_id'],'sampling_seed':sample,
                    'source_frame_indices':item['source_indices'],'original_FPS':float(source['fps']),
                    'panel_order':['HR','LR_bicubic_x4','baseline','frozen','ema'],
                    'native_panel_hw':[128,224],'integer_preview_scale':integer_scale,
                    'preview_resampling':'nearest integer enlargement, no new detail' if integer_scale==2 else 'native panel pixels',
                    'HR_LR_assets':inputs,'saved_prediction_inputs':row_inputs,
                    'comparison_MP4':binding(mp4),'MP4_audit':audit,'contact_sheet_PNG':binding(png),
                    'contact_sheet_hw':[combined.shape[1]*5,combined.shape[2]],
                    'contact_frame_indices':list(CONTACT_FRAMES),
                    'contact_relative_seconds':[i/float(source['fps']) for i in CONTACT_FRAMES],
                    'contact_source_seconds':[item['source_indices'][i]/float(source['fps']) for i in CONTACT_FRAMES]})
                del tensor,combined,predictions
            del source
    finally:reader.close()
    need(len(cases)==6 and not torch.cuda.is_initialized(),'Unexpected case count or GPU initialization')
    readme=output/'README.md';readme.write_text(
        '# Wan TRD 教师目标视频对比\n\n'
        '预声明的三个test clip，每clip采样42/43各一份，全33帧，原FPS。顺序为HR / LR bicubic x4 / baseline / frozen Base / EMA。\n\n'
        '单panel原生128×224；可选2倍最近邻预览只放大像素，不增加细节。contact sheet固定帧0/8/16/24/32，标记相对时间和源帧位置。\n\n'
        'MP4用于显示，有损编码不能替代原保存PNG/NPY或改变科学指标。展示固定crop，不是完整原始视频时长或原分辨率。\n\n'
        '本工具不重新推理、不训练、不修改评分。三个final2048在完整val后全部锁定，不按分数选择；test资产与结果已有历史使用，不宣称独立盲测。\n\n'
        '**公开仓库仅可同步文本manifest/说明；原始PNG、contact sheet、MP4、权重及tensor不得上传。**\n',encoding='utf-8')
    tool=Path(__file__).resolve()
    manifest={'schema':'wan_trd_video_delivery_v1','status':'completed_CPU_display_only',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'test_run':str(run),'test_protocol':binding(run/'protocol.json'),
        'test_status':binding(run/'status.json'),'test_observations':binding(run/'observations.json'),
        'complete_test_verification':verification,'study_file':protocol['study_file'],
        'dataset':protocol['dataset'],'cohort_seal':protocol['cohort_seal'],'finals_receipt':protocol['finals_receipt'],
        'data_independence':protocol['data_independence'],'final_roles':protocol['validation_final_lock']['final_roles'],
        'fixed_video_clip_ids':clips,'sampling_seeds':[42,43],'cases':cases,
        'tool_sources':{'video_delivery':binding(tool),'checkout_loader':binding(Path(config_tools.__file__))},
        'runtime_sources':protocol['sources'],'python':sys.executable,
        'dependency_versions':{name:importlib.metadata.version(name) for name in ('torch','numpy','Pillow','opencv-python-headless')},
        'README':binding(readme),'CPU_only':True,'new_inference_calls':0,'new_training_updates':0,'scores_changed':False,
        'clip_selection':'Protocol fixed first3 sourceIDs before scores, never selected from appearances or quality.',
        'display_only':True,'publication_policy':{'text_metadata_may_be_public':True,'PNG_MP4_tensor_weights_must_not_be_uploaded':True}}
    manifest_path=output/'manifest.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    completion={'status':'completed_CPU_video_delivery','manifest':binding(manifest_path),'MP4_count':6,'contact_sheet_count':6,
                'frames_per_MP4':33,'GPU_work_performed':False,'new_inference_calls':0}
    (output/'completion.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2)+'\n')
    verify_delivery(output)
    return completion


def verify_delivery(directory):
    directory=Path(directory).resolve();completion=read(directory/'completion.json');checked(completion['manifest'])
    manifest=read(directory/'manifest.json')
    need(completion['status']=='completed_CPU_video_delivery' and completion['MP4_count']==completion['contact_sheet_count']==6
         and completion['frames_per_MP4']==33 and manifest.get('schema')=='wan_trd_video_delivery_v1'
         and manifest['status']=='completed_CPU_display_only' and len(manifest['cases'])==6,
         'Incomplete display manifest')
    need({(x['record_id'],x['sampling_seed']) for x in manifest['cases']}=={
        (record,sample) for record in manifest['fixed_video_clip_ids'] for sample in (42,43)},'Missing/duplicate display case')
    for item in ('test_protocol','test_status','test_observations','study_file','dataset','cohort_seal','finals_receipt','README'):
        checked(manifest[item])
    for item in manifest['tool_sources'].values():checked(item)
    for item in manifest['runtime_sources'].values():checked(item)
    for case in manifest['cases']:
        for item in case['HR_LR_assets'].values():checked(item)
        for row in case['saved_prediction_inputs'].values():
            checked(row['row_metrics']);checked(row['row_marker'])
            for png in row['prediction_PNGs']:checked(png)
        path=checked(case['comparison_MP4']);audit=case['MP4_audit']
        _video_audit(path,frames=33,height=audit['height'],width=audit['width'],fps=case['original_FPS'])
        png=checked(case['contact_sheet_PNG'])
        with Image.open(png) as image:
            need(image.mode=='RGB' and list(reversed(image.size))==case['contact_sheet_hw'],
                 'Saved contact sheet changed geometry')
    need(not torch.cuda.is_initialized(),'Display verification initialized CUDA')
    return completion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--dependency-root',type=Path)
    parser.add_argument('--test-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--integer-scale',type=int,choices=(1,2),default=1)
    args=parser.parse_args(argv);ev=config_tools.load_evaluator(args.source_root,args.dependency_root)
    output=args.output_dir or WORKSPACE/'outputs'/('video_compare_'+args.test_run.name)
    result=deliver(ev,test_run=args.test_run,output_dir=output,integer_scale=args.integer_scale)
    print(json.dumps({'output':str(Path(output).resolve()),**result},indent=2))


if __name__=='__main__':main()
