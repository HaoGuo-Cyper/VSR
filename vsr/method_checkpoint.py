"""New-only student checkpoints for the future NoRef continuation arms.

The standard inference payload restores only student LoRA/LR parameters. EMA,
the external student projector, AdamW and explicit RNG streams are kept under
training_only and are never silently selected for inference. This module does
not load models, resume training, construct an optimizer or initialize CUDA.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from .checkpoints import _check_config


ARMS = ('flow_only', 'C1', 'A2')
HISTORICAL = Path('/home/hugo/Projects/VSR')
LEGACY = Path('/home/hugo/Projects/ReprVSR-pretrained')


def _need(ok,message):
    if not ok:raise ValueError(message)


def _cpu_owned(value):
    """A weights_only-compatible tree with no aliases to the live engine."""
    if isinstance(value,torch.Tensor):
        _need(not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Nonfinite training checkpoint tensor')
        _need(not value.is_complex(), 'Complex training checkpoint tensors are unsupported')
        return value.detach().to('cpu').clone().contiguous()
    if isinstance(value,dict):
        _need(all(type(key) in (str,int) for key in value), 'Checkpoint state keys must be strings or integers')
        return {key:_cpu_owned(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        owned = [_cpu_owned(item) for item in value]
        return tuple(owned) if isinstance(value,tuple) else owned
    _need(value is None or type(value) in (str,bool,int,float), 'Unsupported training checkpoint metadata type')
    _need(type(value) is not float or math.isfinite(value), 'Nonfinite checkpoint metadata')
    return value


def _atomic_publish(path,writer):
    """Publish a complete same-directory file with hardlink O_EXCL semantics."""
    path = Path(path)
    _need(not path.exists(), 'Refusing to overwrite existing checkpoint/config: '+str(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b',prefix='.'+path.name+'.',suffix='.partial',
                                         dir=path.parent,delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # link() atomically fails when another writer won the destination race.
        os.link(temporary,path)
        descriptor = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def _destination(path,allowed_root,arm,step):
    path,root = Path(path),Path(allowed_root)
    _need(path.is_absolute() and root.is_absolute(), 'Explicit absolute checkpoint path and allowed_root required')
    _need(not path.is_symlink(), 'Checkpoint destination must not be an existing symlink')
    root,path = root.resolve(),path.resolve()
    _need(root.is_dir() and root != HISTORICAL and not root.is_relative_to(HISTORICAL)
          and root != LEGACY and not root.is_relative_to(LEGACY), 'Historical/legacy assets are read-only')
    run = path.parent.parent
    _need(all(not path.is_relative_to(protected.resolve()) and not run.is_relative_to(protected.resolve())
              for protected in (HISTORICAL,LEGACY)),
          'Resolved checkpoint/run cannot enter historical or legacy assets, even with an ancestor allowed_root')
    _need(path.is_relative_to(root) and path.parent.parent != root and path.parent.name == arm
          and path.name == f'checkpoint_{step:06d}.pt', 'Checkpoint must be a new run/<true arm>/checkpoint_<step>.pt inside allowed_root')
    _need(not path.exists(), 'Refusing to overwrite existing checkpoint: '+str(path))
    return path,run


def build_method_payload(engine,config,*,initialization=None):
    """Validate a completed step and snapshot student plus training-only state."""
    from .restoration_architecture import validate_restoration_architecture
    arm,step = engine.arm,engine.steps
    _need(arm in ARMS and type(step) is int and step > 0, 'A completed positive method step is required')
    _need(config.get('restoration_architecture') == 'noref_base_v1', 'Explicit noref_base_v1 architecture marker required')
    _need(validate_restoration_architecture(config,arm) is False, 'Method restoration must exclude legacy DINO projector')
    config_sha = _check_config(config)
    _need(isinstance(config_sha,str) and len(config_sha) == 64, 'Canonical config _sha256 is mandatory')
    _need(type(config['training'].get('updates')) is int and 0 < step <= config['training']['updates'], 'Checkpoint step exceeds declared update budget')
    _need(config['training'].get('seed',config.get('seed')) == engine.training_seed, 'Checkpoint training seed differs from the engine')
    model = engine.model
    _need(getattr(model,'projector','not Base') is None and not hasattr(model,'reference_adapter'), 'Only the NoRef restoration model may be checkpointed')
    parameters = {name:p for name,p in model.named_parameters() if p.requires_grad}
    _need(parameters and all(name.startswith(('lrv_stem.','lrv_residuals.')) or '.lora_' in name for name in parameters),
          'trainable_state may contain only student LoRA/LR restoration parameters')
    _need(all(p.is_floating_point() and bool(torch.isfinite(p).all()) and p.grad is None for p in parameters.values()),
          'Student parameters must be finite floating values with cleared gradients')
    base = engine.base_load_audit
    _need(isinstance(base,dict) and base.get('variant') == 'Base' and base.get('step') == 8192
          and isinstance(base.get('checkpoint_sha256'),str) and len(base['checkpoint_sha256']) == 64
          and all(c in '0123456789abcdef' for c in base['checkpoint_sha256']),
          'Matched final Base8192 provenance is required')
    _need(all(base.get(key) is False for key in ('optimizer_state_restored','scheduler_state_restored','rng_state_restored')),
          'Continuation origin must not have restored Base optimizer/scheduler/RNG')
    base_config = base.get('config')
    _need(isinstance(base_config,dict), 'Matched Base provenance requires its originating config')
    _check_config(base_config)
    _need(base_config.get('training',{}).get('variants') == ['Base']
          and base_config['training'].get('updates') == 8192
          and base_config['training'].get('seed',base_config.get('seed')) == engine.training_seed
          and base_config.get('seed') == config.get('seed'), 'Matched Base training/data seed or8192-update identity differs')
    _need(base_config.get('model') == config.get('model'),
          'Method model must equal the complete matched Base model config')
    _need(isinstance(config.get('conditioning'),dict)
          and base_config.get('conditioning') == config['conditioning'],
          'Method conditioning must equal the complete matched Base restoration architecture')
    geometry = {'num_frames':33,'temporal_stride':1,'height':128,'width':224,'scale':4}
    for origin,candidate in (('Base',base_config),('method',config)):
        data = candidate.get('data',{})
        _need(all(type(data.get(key)) is int and data[key] == value for key,value in geometry.items()),
              'Wrong '+origin+' protocol33 stride1 crop geometry')
        _need(candidate.get('reference',{}).get('enabled',False) is False
              and candidate.get('conditioning',{}).get('hri_enabled',False) is False,
              'Matched Base and method checkpoints must both be NoRef')
    alignment,ema = engine.alignment,engine.ema
    if arm == 'A2':
        _need(alignment is not None and ema is not None and not ema._swapping and ema.num_updates == step,
              'A2 EMA/projector must match completed steps and be outside a teacher swap')
        ema._validate(model)
        ema_state = ema.state_dict()
        projector = alignment.projector.state_dict()
        _need(projector and all(p.requires_grad and p.dtype == torch.float32 and p.grad is None
                               for p in alignment.projector.parameters()),
              'Student projector must remain trainable FP32 with cleared gradients')
    else:
        _need(alignment is None and ema is None, 'Flow/C1 may not carry an A2 teacher/projector')
        ema_state = projector = None
    expected_optimizer = list(parameters.values()) + (list(alignment.projector.parameters()) if alignment is not None else [])
    actual_optimizer = [p for group in engine.optimizer.param_groups for p in group['params']]
    _need(isinstance(engine.optimizer,torch.optim.AdamW)
          and len(actual_optimizer) == len({id(p) for p in actual_optimizer})
          and {id(p) for p in actual_optimizer} == {id(p) for p in expected_optimizer},
          'Optimizer must contain exactly student restoration and external-projector parameters')
    rng = {}
    for name in ('noise','timestep','mask','corruption'):
        generator = getattr(engine.rng,name,None)
        _need(isinstance(generator,torch.Generator), 'Missing explicit method RNG stream: '+name)
        state = generator.get_state()
        _need(state.dtype == torch.uint8 and state.ndim == 1 and state.device.type == 'cpu', 'RNG state must be a CPU byte vector')
        rng[name] = state
    payload = {'format':'vsr_training_v1','variant':arm,'step':step,'config':config,
        'trainable_state':parameters,'initialization':initialization,
        'inference_parameters':'student','restoration_architecture':'noref_base_v1',
        'training_only':{'schema':'vsr_method_training_state_v1','arm':arm,'completed_updates':step,
            'EMA':ema_state,'student_projector':projector,'optimizer':engine.optimizer.state_dict(),
            'explicit_RNG':rng,'matched_Base':base,
            'restored_by_generic_inference_loader':False,'resume_implementation_provided':False}}
    return _cpu_owned(payload)


def save_method_checkpoint(path,engine,config,*,allowed_root,initialization=None):
    """Atomically write a new method checkpoint; never overwrite any artifact.

    Existing run/config.json must equal the canonical config exactly. A missing
    one is created atomically inside this new run; no legacy file is modified.
    A failed checkpoint publication may leave that valid new config, never a
    partial checkpoint or a replacement of a previously published checkpoint.
    """
    payload = build_method_payload(engine,config,initialization=initialization)
    path,run = _destination(path,allowed_root,engine.arm,engine.steps)
    path.parent.mkdir(parents=True,exist_ok=True)
    config_path = run/'config.json'
    _need(not config_path.is_symlink() and config_path.resolve().parent == run, 'Run config must not be a symlink')
    canonical = (json.dumps(payload['config'],ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if config_path.exists():
        _need(json.loads(config_path.read_text()) == payload['config'], 'Existing run config differs; refusing to overwrite')
    else:
        try:_atomic_publish(config_path,lambda stream:stream.write(canonical))
        except FileExistsError:
            _need(json.loads(config_path.read_text()) == payload['config'], 'Concurrent run config differs')
    _atomic_publish(path,lambda stream:torch.save(payload,stream))
    with path.open('rb') as stream:digest = hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest,'step':engine.steps,'variant':engine.arm,
        'restoration_architecture':'noref_base_v1','inference_parameters':'student',
        'config_sha256':payload['config']['_sha256'],'student_parameter_names':sorted(payload['trainable_state']),
        'training_only_present':True,'existing_artifacts_overwritten':False}
