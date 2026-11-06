"""Sequential protocol33 Base runs with a bounded full-optimizer gate.

Only the new streaming stage2 cache is reused. Every task model/optimizer is
fresh from official weights. One failed process stops the queue, with no retry.
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
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def json_record(value):
    """Use one lossless JSON representation in memory and on disk.

    CUDA's free/total byte counts arrive as a tuple; JSON stores the same two
    integers in a list. Canonicalize before persistence and strict comparison.
    """
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))


def validate_config(config):
    d, t, c = config['data'], config['training'], config['stage2_contract']
    require((d['num_frames'], d['temporal_stride'], d['height'], d['width'], d['scale']) == (33,1,128,224,4), 'Protocol geometry changed')
    require(config['seed'] == 20261102 and t['seed'] in (20261102,20261103,20261104), 'Unexpected data/training seeds')
    require(t['variants'] == t['preparation_variants'] == ['Base'], 'Only NoRef Base is authorized in this queue')
    require(t['updates'] == 8192 and t['resume'] is False and t['batch_size'] == 1, 'Unexpected budget/resume/batch')
    require(t['mixed_precision'] == 'bf16' and t['gradient_checkpointing'] is True and config['reproducibility']['mode'] == 'strict', 'Precision contract changed')
    require(not config.get('reference', {}).get('enabled',False) and config['conditioning']['injection_layers'] == [0,5,9], 'Conditioning contract changed')
    require(c['T5_load_allowed'] is False and c['DINO_load_allowed'] is False, 'Forbidden teacher/text loading')
    require(Path(c['allowed_root']).resolve() == (ROOT/'runs').resolve(), 'Wrong run root')


def optimizer_probe(config, run, data):
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
    from .train import make_model
    from .preflight import autocast, release
    from .model import gradient_norms
    from .reproducibility import tensor_state_hashes
    clips, latents, embeddings, features, flows, geometry, ntrain = data
    require(ntrain == 2 and geometry['latent_grid'] == [9,16,28] and features is None and flows is None, 'Probe must have two train-only clips and no auxiliary teachers')
    model, initialization = make_model(config, 'Base')
    before = initialization['trainable_initialization_hashes']['all_trainable']['sha256']
    write(run/'initialization.json', initialization)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = config['training']
    optimizer = torch.optim.AdamW(params, lr=opt['learning_rate'], weight_decay=opt['weight_decay'])
    scheduler = UniPCMultistepScheduler.from_pretrained(config['model']['local_dir'], subfolder='scheduler', local_files_only=True)
    scheduler.set_timesteps(1000,device='cuda')
    # Match the actual runner's resident evaluation VAE and optimizer state.
    vae = AutoencoderKLWan.from_pretrained(config['model']['local_dir'], subfolder='vae', dtype=torch.float32, local_files_only=True,use_safetensors=True).requires_grad_(False).eval().to('cuda')
    texts = {k:v.to('cuda') for k,v in embeddings.items()}
    noise_rng = torch.Generator(device='cuda').manual_seed(opt['seed']+1)
    index_rng = torch.Generator(device='cpu').manual_seed(opt['seed']+2)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(8):
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.train()
        i = step % ntrain
        clean, low = (latents[i][key].to('cuda') for key in ('clean','low'))
        index = int(torch.randint(0,1000,(),generator=index_rng))
        sigma, timestep = scheduler.sigmas[index].to('cuda'), scheduler.timesteps[index].expand(1)
        noise = torch.randn(clean.shape,generator=noise_rng,device='cuda')
        noisy = (1-sigma)*clean + sigma*noise
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            velocity, _ = model(noisy,timestep,texts[clips[i]['metadata']['caption']],low)
        loss = F.mse_loss(velocity.float(),noise-clean)
        require(bool(torch.isfinite(loss)), f'Nonfinite probe loss at{step+1}')
        loss.backward()
        groups = gradient_norms(model)
        require(all(x['finite'] for x in groups.values()) and groups['frozen']['elements_with_grad'] == 0, 'Probe gradient failure')
        norm = torch.nn.utils.clip_grad_norm_(params,opt['max_grad_norm'],error_if_nonfinite=True)
        optimizer.step()
        model.clear_features()
        torch.cuda.synchronize()
        row = {'step':step+1,'clip_id':clips[i]['metadata']['record_id'],'loss':float(loss.detach()),'grad_norm_before_clip':float(norm),
               'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gradients':groups}
        rows.append(row)
        with (run/'training.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        print(f"Probe {step+1}/8 loss={row['loss']:.6f} peak={row['peak_allocated_bytes']/2**30:.3f}GiB", flush=True)
    after = tensor_state_hashes(model)['all_trainable']['sha256']
    peak = torch.cuda.max_memory_allocated()
    require(before != after and peak < 16*2**30, 'Probe failed trainable update/memory gate')
    elapsed = [row['seconds'] for row in rows[1:]]
    result = {'status':'passed','kind':'train_only_full_optimizer_resource_diagnostic','updates':8,'formal_training_updates':0,
              'ntrain':2,'test_accessed':False,'latent_grid':geometry['latent_grid'],'vae_resident':True,'optimizer_state_elements':sum(v.numel() for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor)),
              'peak_allocated_bytes':peak,'seconds_mean_excluding_first':statistics.mean(elapsed),'seconds_sample_sd_excluding_first':statistics.stdev(elapsed),
              '8192_updates_hours_estimate_excluding_preparation_evaluation':statistics.mean(elapsed)*8192/3600,
              'trainable_sha256_before':before,'trainable_sha256_after':after,'trained_parameters_reused':False,'actual_EMA_A2_profile':False,'rows':rows}
    write(run/'resource_profile.json',result)
    del model,optimizer,vae,params,texts
    release()
    return result


def effective_config(source, *, cache_from=None, profile=False):
    source = Path(source)
    config = json.loads(source.read_text())
    validate_config(config)
    config['_source_config_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    if cache_from:
        config['data']['cache_from'] = str(cache_from)
    if profile:
        config['stage'] = 'stage2_full_optimizer_resource_diagnostic'
        config['experiment_name'] = 'stage2_protocol33_optimizer_probe'
        config['data'].update(train_limit=2,val_limit=0)
        config['training'].update(updates=8,evaluate_at=[],checkpoint_at=[])
        config['stage2_contract'].update(expected_training_clips=2,expected_validation_clips=0)
    # Match checkpoints._check_config exactly, including inserted runtime fields.
    config['_sha256'] = hashlib.sha256((json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    return config


def worker(args):
    import torch
    from .audit import environment, file_hash, verify_official
    from .reproducibility import configure_reproducibility
    from .stage2_preparation import prepare_stage2
    from .train import train_variant
    source = ROOT / args.config
    config = effective_config(source,cache_from=args.cache_from,profile=args.profile)
    run = ROOT/'runs'/f"{config['experiment_name']}_{stamp()}"
    run.mkdir(parents=True,exist_ok=False)
    write(run/'config.json',config)
    write(Path(args.result_file),{'status':'running','run':str(run)})
    write(ROOT/'runs/active_training.json',{'status':'running','run':str(run),'pid':os.getpid()})
    print(f'New independent stage2 experiment: {run}',flush=True)
    begun = time.perf_counter()
    try:
        write(run/'reproducibility.json',configure_reproducibility(config))
        env = json_record(environment(ROOT))
        write(run/'environment.json',env)
        require(env['cuda_available'] and not env['git_status'], 'CUDA unavailable or dirty runtime source')
        write(run/'official_artifacts.json',verify_official(ROOT,config))
        preparation_start = time.perf_counter()
        data = prepare_stage2(config,run,env)
        prep_seconds = time.perf_counter()-preparation_start
        if args.profile:
            summary = optimizer_probe(config,run,data)
        else:
            summary = train_variant('Base',config,run,*data)
            require(summary['status']=='completed' and summary['updates']==8192, 'Incomplete formal Base')
        status = {'status':'completed','run':str(run),'seed':config['training']['seed'],'stage':config['stage'],'profile':args.profile,
                  'preparation_seconds':prep_seconds,'entire_wall_seconds':time.perf_counter()-begun,'test_accessed':False,'summary':summary}
        write(run/'status.json',status)
        if not args.profile and not args.cache_from:
            subprocess.run([sys.executable,str(ROOT/'scripts/seal_cache.py'),str(run)],check=True)
        write(Path(args.result_file),status)
        write(ROOT/'runs/active_training.json',status)
    except BaseException as error:
        failure = {'status':'failed','run':str(run),'error':str(error),'traceback':traceback.format_exc(),'automatic_retry':False}
        write(run/'status.json',failure)
        write(Path(args.result_file),failure)
        write(ROOT/'runs/active_training.json',failure)
        raise


def queue(args):
    run = ROOT/'runs'/f'stage2_protocol33_queue_{stamp()}'
    run.mkdir(parents=True,exist_ok=False)
    lock = (ROOT/'runs/stage2_queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    configs = [f'configs/stage2_20261105/base_seed{s}.json' for s in (20261102,20261103,20261104)]
    env = dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTORCH_ALLOC_CONF='expandable_segments:True',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='20261102')
    state = {'status':'running','queue':str(run),'pid':os.getpid(),'configs':configs,'completed':[],'automatic_retry':False}
    write(ROOT/'runs/active_stage2_queue.json',state)
    write(run/'state.json',state)
    cache = None
    try:
        for name, cfg, profile in [('resource_probe',configs[0],True)]+[(f'seed{s}',c,False) for s,c in zip((20261102,20261103,20261104),configs)]:
            result = run/f'{name}.json'
            command = [sys.executable,'-u','-m','vsr.stage2_base_queue','worker','--config',cfg,'--result-file',str(result)]
            if profile:
                command.append('--profile')
            elif cache:
                command.extend(['--cache-from',cache])
            state.update(current=name,current_result=str(result))
            write(run/'state.json',state)
            write(ROOT/'runs/active_stage2_queue.json',state)
            with (run/f'{name}.console.log').open('x') as console:
                subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,check=True)
            record = json.loads(result.read_text())
            require(record['status']=='completed',f'{name} did not complete')
            if not profile:
                require((Path(record['run'])/'Base/checkpoint_008192.pt').is_file(), 'Final checkpoint absent')
                if cache is None:
                    cache = record['run']
                    require((Path(cache)/'cache_asset_seal.json').is_file(), 'Data cache not sealed')
            state['completed'].append({'name':name,'run':record['run']})
        state.update(status='completed',next_task='T2.2/T2.3 protocol and T2.4 frame-varying evaluation; G2 not yet passed')
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        write(run/'state.json',state)
        write(ROOT/'runs/active_stage2_queue.json',state)
        fcntl.flock(lock,fcntl.LOCK_UN)
    print(json.dumps(state),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['queue','worker'])
    parser.add_argument('--config')
    parser.add_argument('--result-file')
    parser.add_argument('--cache-from')
    parser.add_argument('--profile',action='store_true')
    args = parser.parse_args()
    if args.mode=='worker':
        require(args.config and args.result_file, 'Worker requires config and result-file')
        # The actual GPU owner holds this lease independently of its parent.
        # An orphan worker therefore still blocks another queue/direct worker.
        with (ROOT/'runs/stage2_gpu.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            worker(args)
    else:
        require(not any((args.config,args.result_file,args.cache_from,args.profile)), 'Queue takes no worker options')
        queue(args)


if __name__=='__main__':
    main()
