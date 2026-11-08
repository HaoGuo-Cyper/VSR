"""Restore task weights from new VSR runs onto a fresh official Wan base.

This loader is for evaluation/inference and never resumes optimizer, scheduler
or RNG state. Training comparisons must still initialize independently using
make_model; loading a task checkpoint must be disclosed as its own origin.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path

import torch
from .model import configure_reproducibility
from .model import validate_restoration_architecture, METHOD_ARMS

PRODUCTION_RUNS_ROOT = Path(os.environ.get("VSR_RUNS_ROOT", str(Path.cwd() / "runs"))).resolve()
LEGACY_PROJECT_ROOT = Path("/home/hugo/Projects/ReprVSR-pretrained")
OFFICIAL_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OFFICIAL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_checkpoint(path, allowed_root):
    path = Path(path).resolve()
    if path.is_relative_to(LEGACY_PROJECT_ROOT):
        raise ValueError("Legacy project checkpoints are forbidden in this experiment")
    if not path.is_relative_to(Path(allowed_root).resolve()):
        raise ValueError("Checkpoint path escapes this experiment's allowed run root")
    if not path.is_file():
        raise FileNotFoundError(f"New VSR checkpoint missing: {path}")
    return path


def _check_config(config):
    if not isinstance(config, dict):
        raise ValueError("Training checkpoint config must be a dictionary")
    model = config.get("model", {})
    if model.get("repo_id") != OFFICIAL_REPO_ID or model.get("revision") != OFFICIAL_REVISION:
        raise ValueError("Checkpoint official Wan repo/revision differs from pinned 1.3B")
    source_hash = config.get("_sha256")
    if config.get("restoration_architecture") is not None and source_hash is None:
        raise ValueError("Method architecture requires canonical config _sha256")
    if source_hash is not None:
        value = {key: item for key, item in config.items() if key != "_sha256"}
        canonical = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        actual = hashlib.sha256(canonical).hexdigest()
        if source_hash != actual:
            raise ValueError("Checkpoint source config SHA256 mismatch")
    variants = config.get("training", {}).get("variants")
    if not isinstance(variants, list) or not variants or any(not isinstance(v, str) or not v for v in variants) or len(variants) != len(set(variants)):
        raise ValueError("Checkpoint config must identify distinct training variants")
    for variant in variants:
        validate_restoration_architecture(config,variant)
    return source_hash


def load_training_checkpoint(path, *, device="cuda", allowed_root=PRODUCTION_RUNS_ROOT,
                             model_factory=None, variant=None, model_directory=None):
    """Return (model,meta), restoring only the exact trainable parameter set.

    A custom model_factory(config,variant) must return (model,initialization)
    and construct a fresh official model. The default imports train.make_model
    lazily. No task state is copied until every key/shape/dtype is checked.
    Floating checkpoint tensors are copied into each destination float dtype;
    source and destination dtype sets are included in the returned audit.
    """
    path = _safe_checkpoint(path, allowed_root)
    checkpoint_sha = _sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "vsr_training_v1":
        raise ValueError("Unsupported VSR training checkpoint format")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("Training checkpoint step must be an integer >=1")
    config = payload.get("config")
    source_config_sha = _check_config(config)
    selected_variant = path.parent.name if variant is None else variant
    if selected_variant not in config["training"]["variants"]:
        raise ValueError("Requested checkpoint variant is not in its training config")
    if payload.get("variant") is not None and payload["variant"] != selected_variant:
        raise ValueError("Checkpoint payload variant differs from requested variant")
    # Check the neighboring originating run config when it is present. An
    # explicit variant override still permits reviewed copies within own runs.
    run_config_path = path.parent.parent / "config.json"
    if run_config_path.is_file():
        if not run_config_path.resolve().is_relative_to(Path(allowed_root).resolve()):
            raise ValueError("Originating config path escapes allowed run root")
        source_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if source_config != config:
            raise ValueError("Checkpoint config differs from originating run config")
    state = payload.get("trainable_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint trainable_state must be a nonempty dictionary")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError("Task checkpoint state requires named floating tensors")
        if selected_variant in METHOD_ARMS and not (key.startswith(("lrv_stem.","lrv_residuals.")) or ".lora_" in key):
            raise ValueError("Method trainable_state must contain only restoration student LoRA/LR: "+key)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Task checkpoint contains nonfinite tensor: {key}")
    runtime_config = copy.deepcopy(config)
    runtime_model_directory = config.get('model', {}).get('local_dir')
    if model_directory is not None:
        runtime_model_directory = str(Path(model_directory).expanduser().resolve())
        runtime_config['model']['local_dir'] = runtime_model_directory
    # Source config and canonical hash remain the original checkpoint record.
    # The explicit runtime path is separately disclosed, never rewritten there.
    if model_factory is None:
        runtime=configure_reproducibility(config)
        from .model import make_model
        result=make_model(runtime_config,selected_variant,device=device)
    else:
        runtime=None
        result = model_factory(runtime_config, selected_variant)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("model_factory must return (fresh_model, initialization)")
    model, fresh_initialization = result
    if not isinstance(fresh_initialization, dict):
        raise ValueError("Fresh model factory must return initialization audit metadata")
    if fresh_initialization.get("official_revision") != OFFICIAL_REVISION or fresh_initialization.get("fresh_official_initialization") is not True or fresh_initialization.get("legacy_state_loaded") is not False:
        raise ValueError("Model factory did not attest fresh pinned official initialization")
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != set(parameters):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise ValueError(f"Task checkpoint trainable key mismatch; missing={missing}; unexpected={unexpected}")
    converted_state = {}
    for key, parameter in parameters.items():
        if tuple(state[key].shape) != tuple(parameter.shape):
            raise ValueError(f"Task checkpoint tensor shape mismatch: {key}")
        if not parameter.is_floating_point():
            raise ValueError(f"Destination task parameter is not floating point: {key}")
        converted = state[key].to(dtype=parameter.dtype)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(f"Task tensor overflows destination float dtype: {key}")
        converted_state[key] = converted
    # All validation precedes mutation. Frozen official parameters are excluded.
    with torch.no_grad():
        for key, parameter in parameters.items():
            parameter.copy_(converted_state[key].to(device=parameter.device))
    model = model.to(device)
    model.eval()
    metadata = {
        "checkpoint_path": str(path), "checkpoint_sha256": checkpoint_sha,
        "format": payload["format"], "step": step, "variant": selected_variant,
        "config": config, "config_sha256": source_config_sha,
        "runtime_model_directory": runtime_model_directory,
        "runtime_model_directory_overridden": model_directory is not None,
        "official_repo_id": OFFICIAL_REPO_ID, "official_revision": OFFICIAL_REVISION,
        "fresh_base_initialization": fresh_initialization,
        "task_initialization_origin": "new VSR run checkpoint",
        "restored_parameter_count": sum(p.numel() for p in parameters.values()),
        "restored_parameter_names": sorted(parameters),
        "checkpoint_tensor_dtypes": sorted({str(value.dtype) for value in state.values()}),
        "destination_tensor_dtypes": sorted({str(value.dtype) for value in parameters.values()}),
        "optimizer_state_restored": False, "scheduler_state_restored": False,
        "rng_state_restored": False, "frozen_official_parameters_restored_from_task_state": False,
        "device": str(device), "training_comparison_initialization": False,
        "reproducibility":runtime,
        "restoration_architecture":config.get("restoration_architecture"),
        "inference_weights":"student",
        "training_only_state_restored":False,
        "EMA_weights_selected":False,
        "training_only_fields_present":sorted(map(str,payload.get("training_only",{}))) if isinstance(payload.get("training_only"),dict) else [],
    }
    return model, metadata


# Student and explicitly separated training-state publication.
import hashlib

import json

import math

import os

from pathlib import Path

import tempfile

import torch

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
    from .model import validate_restoration_architecture
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


def save_base_checkpoint(path, model, config, *, step, allowed_root, initialization=None):
    """Publish a new Base student checkpoint using the legacy loader format."""
    _need(type(step) is int and 0 < step <= config.get('training', {}).get('updates', 0), 'Invalid completed Base step')
    _need(config.get('training', {}).get('variants') == ['Base'], 'Base configuration must identify only Base')
    _need(validate_restoration_architecture(config, 'Base') is False, 'NoRef Base architecture required')
    _need(isinstance(_check_config(config), str), 'Canonical config _sha256 is required for new Base checkpoints')
    state = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    _need(state and all(name.startswith(('lrv_stem.', 'lrv_residuals.')) or '.lora_' in name for name in state),
          'Base state must contain only restoration LoRA/LR parameters')
    _need(all(parameter.is_floating_point() and bool(torch.isfinite(parameter).all()) and parameter.grad is None
              for parameter in state.values()), 'Base parameters must be finite with cleared gradients')
    path, run = _destination(path, allowed_root, 'Base', step)
    payload = _cpu_owned({'format': 'vsr_training_v1', 'variant': 'Base', 'step': step,
        'config': config, 'trainable_state': state, 'initialization': initialization,
        'inference_parameters': 'student', 'training_only': {}})
    path.parent.mkdir(parents=True, exist_ok=True)
    config_path = run / 'config.json'
    _need(not config_path.is_symlink(), 'Run config must not be a symlink')
    encoded = (json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if config_path.exists():
        _need(json.loads(config_path.read_text()) == config, 'Existing run config differs')
    else:
        try:
            _atomic_publish(config_path, lambda stream: stream.write(encoded))
        except FileExistsError:
            _need(json.loads(config_path.read_text()) == config, 'Concurrent run config differs')
    _atomic_publish(path, lambda stream: torch.save(payload, stream))
    return {'path': str(path), 'sha256': _sha256_file(path), 'step': step, 'variant': 'Base',
            'inference_parameters': 'student', 'existing_artifacts_overwritten': False,
            'config_sha256': config.get('_sha256')}


def _hr_equal(left, right):
    """Exact checkpoint metadata comparison, including owned CPU tensors."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_hr_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_hr_equal(a, b) for a, b in zip(left, right))
    return left == right


def _hr_origin(base):
    """Stable Base identity; fresh-process runtime-before observations may vary."""
    _need(isinstance(base, dict) and base.get('variant') == 'Base' and base.get('step') == 8192,
          'HR training must originate from the matched final Base8192')
    sha = base.get('checkpoint_sha256')
    _need(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
          'HR training requires the original Base checksum')
    source_path = base.get('checkpoint_path')
    _need(isinstance(source_path,str) and Path(source_path).is_absolute()
          and Path(source_path).name == 'checkpoint_008192.pt' and Path(source_path).parent.name == 'Base',
          'HR training requires the explicit original Base8192 path')
    _need(all(base.get(k) is False for k in ('optimizer_state_restored', 'scheduler_state_restored', 'rng_state_restored')),
          'Base optimizer/scheduler/RNG state must not initialize HR training')
    cfg = base.get('config'); _check_config(cfg)
    _need(cfg['training'].get('variants') == ['Base'] and cfg['training'].get('updates') == 8192,
          'Invalid HR matched Base source config')
    return {key:base.get(key) for key in ('checkpoint_path', 'checkpoint_sha256', 'step', 'variant', 'config')}


def _hr_engine_contract(engine, config):
    _need(isinstance(_check_config(config), str) and config.get('restoration_architecture') is None
          and config['training'].get('variants') == ['Base'], 'Canonical HR checkpoint config must retain Base architecture')
    budget = config['training'].get('updates')
    _need(type(budget) is int and budget > 0 and type(engine.steps) is int and 0 <= engine.steps <= budget,
          'HR completed steps exceed this round budget')
    _need(type(engine.weight) in (int, float) and math.isfinite(engine.weight) and engine.weight >= 0
          and type(engine.student_layer) is int and engine.student_layer == 7,
          'HR checkpoint requires its finite weight and fixed student layer7')
    origin = _hr_origin(engine.base_load_audit); base_config = origin['config']
    _need(all(config.get(k) == base_config.get(k) for k in ('model', 'conditioning', 'data', 'seed')),
          'HR model/conditioning/data must match its original Base')
    _need(config['training'].get('seed', config['seed']) == engine.training_seed
          == base_config['training'].get('seed', base_config['seed']), 'HR training seed differs')
    _need(getattr(engine.model, 'projector', 'not Base') is None and not hasattr(engine.model, 'reference_adapter'),
          'HR restoration model must remain NoRef Base without an inference head')
    student = {n:p for n,p in engine.model.named_parameters() if p.requires_grad}
    _need(student and all(n.startswith(('lrv_stem.', 'lrv_residuals.')) or '.lora_' in n for n in student),
          'HR student state may contain only restoration LoRA/LR parameters')
    alignment = engine.alignment
    _need((alignment is None) == (engine.weight == 0), 'HR baseline/head weight contract differs')
    head_params = list(alignment.parameters()) if alignment is not None else []
    expected = list(student.values()) + head_params
    _need(all(p.requires_grad and p.is_floating_point() and p.dtype == torch.float32
              and p.grad is None and bool(torch.isfinite(p).all()) for p in expected),
          'HR student/head must be finite FP32 with cleared gradients')
    named = list(engine._named_optimizer_parameters)
    _need(all(isinstance(n, str) and isinstance(p, torch.nn.Parameter) for n,p in named)
          and len({n for n,p in named}) == len(named) == len({id(p) for n,p in named})
          and {id(p) for n,p in named} == {id(p) for p in expected}, 'HR optimizer parameter names/aliases differ')
    _need(isinstance(engine.optimizer, torch.optim.AdamW)
          and all(group.get('foreach') is False for group in engine.optimizer.param_groups),
          'HR training requires AdamW foreach=False')
    by_id = {id(p):n for n,p in named}
    actual = [p for group in engine.optimizer.param_groups for p in group['params']]
    _need(len(actual) == len(named) and len({id(p) for p in actual}) == len(actual)
          and {id(p) for p in actual} == set(by_id), 'HR optimizer must contain exactly student and head')
    groups = [[by_id[id(p)] for p in group['params']] for group in engine.optimizer.param_groups]
    head = None if alignment is None else alignment.state_dict()
    return student, head, dict(named), groups, origin


def _hr_alignment_spec(alignment):
    if alignment is None:
        return None
    patch=getattr(alignment,'patch_size',None)
    return {'class':type(alignment).__module__+'.'+type(alignment).__qualname__,
            'student_dim':alignment.student_dim, 'latent_channels':getattr(alignment,'latent_channels',None),
            'patch_size':None if patch is None else list(patch), 'target_dim':alignment.target_dim}


def _hr_validate_optimizer(saved, current, names, parameters, steps):
    _need(isinstance(saved, dict) and set(saved) == {'state', 'param_groups'}
          and len(saved['param_groups']) == len(current['param_groups']), 'Incomplete HR optimizer state')
    parameters_by_id = {}
    amsgrad = {}
    for old,new,group_names in zip(saved['param_groups'], current['param_groups'], names):
        _need(isinstance(old, dict) and set(old) == set(new)
              and _hr_equal({k:v for k,v in old.items() if k != 'params'},
                            {k:v for k,v in new.items() if k != 'params'})
              and old['params'] == new['params'] and all(type(i) is int for i in old['params'])
              and len(old['params']) == len(group_names),
              'HR optimizer parameter groups or hyperparameters differ')
        for identifier,name in zip(old['params'],group_names):
            parameters_by_id[identifier] = parameters[name]
            amsgrad[identifier] = old.get('amsgrad', False)
    _need(isinstance(saved['state'], dict) and set(saved['state']) == set(parameters_by_id),
          'HR checkpoint requires complete AdamW state for every parameter')
    for identifier,param in parameters_by_id.items():
        state = saved['state'][identifier]
        keys = {'step', 'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if amsgrad[identifier] else set())
        _need(isinstance(state, dict) and set(state) == keys, 'Incomplete HR AdamW moments')
        count = state['step']
        _need(isinstance(count, torch.Tensor) and count.ndim == 0 and not count.requires_grad
              and not count.is_complex() and count.dtype != torch.bool
              and bool(torch.isfinite(count)) and float(count) == steps, 'HR AdamW step counter differs')
        for key in keys - {'step'}:
            value = state[key]
            _need(isinstance(value, torch.Tensor) and value.shape == param.shape and value.dtype == param.dtype
                  and not value.requires_grad and bool(torch.isfinite(value).all()), 'HR AdamW moment shape/dtype differs')
            if key != 'exp_avg':
                _need(bool((value >= 0).all()), 'HR AdamW variance moment is negative')


def save_hr_latent_checkpoint(path, engine, config, *, allowed_root, protocol_sha256,
                              cumulative_training_seconds, initialization):
    """Publish student-only inference weights plus exact same-round resume state."""
    student,head,named,groups,origin = _hr_engine_contract(engine, config)
    _need(engine.steps > 0, 'HR checkpoint requires a completed positive update')
    _need(isinstance(protocol_sha256, str) and len(protocol_sha256) == 64
          and all(c in '0123456789abcdef' for c in protocol_sha256), 'HR source protocol SHA256 required')
    _need(type(cumulative_training_seconds) in (int,float) and math.isfinite(cumulative_training_seconds)
          and cumulative_training_seconds >= 0, 'HR cumulative training seconds must be finite and nonnegative')
    _need(_hr_equal(_hr_origin(initialization), origin), 'HR initialization differs from its Base origin')
    optimizer = _cpu_owned(engine.optimizer.state_dict())
    _hr_validate_optimizer(optimizer, optimizer, groups, named, engine.steps)
    rng,devices = {},{}
    for name in ('noise','timestep','mask','corruption'):
        generator = getattr(engine.rng,name,None)
        _need(isinstance(generator,torch.Generator), 'Missing HR explicit RNG stream: '+name)
        rng[name] = generator.get_state(); devices[name] = str(generator.device)
    state = {'schema':'vsr_hr_latent_training_state_v1', 'protocol_sha256':protocol_sha256,
             'weight':float(engine.weight), 'student_layer':engine.student_layer,
             'completed_updates':engine.steps, 'training_seed':engine.training_seed,
             'teacher_kind':getattr(engine,'teacher_kind','vae'),
             'learning_rate_schedule':{'type':'constant','separate_scheduler':None,'lr_saved_in_optimizer_groups':True},
             'alignment_state':head, 'alignment_spec':_hr_alignment_spec(engine.alignment),
             'optimizer':optimizer, 'optimizer_parameter_names':groups,
             'explicit_RNG':rng, 'RNG_devices':devices, 'matched_Base':engine.base_load_audit,
             'cumulative_training_seconds':float(cumulative_training_seconds),
             'restored_by_generic_inference_loader':False}
    payload = _cpu_owned({'format':'vsr_training_v1', 'variant':'Base', 'step':engine.steps,
                         'config':config, 'trainable_state':student, 'initialization':initialization,
                         'inference_parameters':'student', 'training_only':state})
    path,run = _destination(path,allowed_root,'Base',engine.steps)
    path.parent.mkdir(parents=True,exist_ok=True)
    sidecar = run/'config.json'
    _need(not sidecar.is_symlink(), 'HR run config must not be a symlink')
    encoded = (json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if sidecar.exists():
        _need(json.loads(sidecar.read_text()) == config, 'HR run config differs')
    else:
        try:_atomic_publish(sidecar,lambda stream:stream.write(encoded))
        except FileExistsError:_need(json.loads(sidecar.read_text()) == config, 'Concurrent HR run config differs')
    _atomic_publish(path,lambda stream:torch.save(payload,stream))
    return {'path':str(path), 'bytes':path.stat().st_size, 'sha256':_sha256_file(path),
            'step':engine.steps, 'updates':engine.steps, 'variant':'Base',
            'protocol_sha256':protocol_sha256, 'cumulative_training_seconds':float(cumulative_training_seconds),
            'config_sha256':config['_sha256'], 'inference_parameters':'student',
            'training_only_schema':state['schema'], 'existing_artifacts_overwritten':False}


def restore_hr_latent_training_state(engine, path, *, allowed_root, expected_protocol_sha256,
                                     expected_config=None):
    """Restore a fully committed same-round HR checkpoint into a fresh engine.

    Validate all source, parameter, head, optimizer and explicit-RNG contracts
    before mutation. Original Base optimizer states and generic task checkpoints
    are never accepted as HR resume state. Frozen official weights stay fresh.
    """
    path = _safe_checkpoint(path, allowed_root); checksum = _sha256_file(path)
    payload = torch.load(path,map_location='cpu',weights_only=True)
    _need(isinstance(payload,dict) and payload.get('format') == 'vsr_training_v1'
          and payload.get('variant') == 'Base', 'HR resume requires a Base student payload')
    config = payload.get('config'); _check_config(config)
    _need(expected_config is None or config == expected_config, 'HR resume config differs from expected config')
    sidecar = path.parent.parent/'config.json'
    _need(sidecar.is_file() and not sidecar.is_symlink() and json.loads(sidecar.read_text()) == config,
          'HR resume originating sidecar config differs or is absent')
    student,head,named,groups,origin = _hr_engine_contract(engine,config)
    _need(engine.steps == 0 and not engine.optimizer.state, 'HR resume requires a fresh engine and optimizer')
    step = payload.get('step'); state = payload.get('training_only')
    _need(type(step) is int and 0 < step <= config['training']['updates']
          and path.parent.name == 'Base' and path.name == f'checkpoint_{step:06d}.pt', 'HR resume step/path differs')
    _need(isinstance(state,dict) and state.get('schema') == 'vsr_hr_latent_training_state_v1',
          'Not this round HR resume state; original Base checkpoints cannot resume it')
    _need(isinstance(expected_protocol_sha256,str) and len(expected_protocol_sha256) == 64
          and all(c in '0123456789abcdef' for c in expected_protocol_sha256)
          and state.get('protocol_sha256') == expected_protocol_sha256, 'HR resume source protocol differs')
    _need(type(state.get('completed_updates')) is int and state['completed_updates'] == step
          and type(state.get('training_seed')) is int and state['training_seed'] == engine.training_seed
          and type(state.get('weight')) in (int,float) and state['weight'] == engine.weight
          and type(state.get('student_layer')) is int and state['student_layer'] == engine.student_layer,
          'HR resume step/weight/layer/training seed differs')
    _need(state.get('teacher_kind','vae')==getattr(engine,'teacher_kind','vae'),'Teacher representation differs on resume')
    _need(_hr_equal(_hr_origin(state.get('matched_Base')),origin)
          and _hr_equal(_hr_origin(payload.get('initialization')),origin), 'HR resume original Base identity differs')
    seconds = state.get('cumulative_training_seconds')
    _need(type(seconds) in (int,float) and math.isfinite(seconds) and seconds >= 0, 'Invalid HR cumulative training seconds')
    saved_student = payload.get('trainable_state')
    _need(isinstance(saved_student,dict) and set(saved_student) == set(student), 'HR student parameter names differ')
    for name,param in student.items():
        value = saved_student[name]
        _need(isinstance(value,torch.Tensor) and value.shape == param.shape and value.dtype == param.dtype
              and not value.requires_grad and bool(torch.isfinite(value).all()), 'HR student shape/dtype/values differ: '+name)
    saved_head = state.get('alignment_state')
    _need((saved_head is None) == (head is None), 'HR baseline/head presence differs')
    _need(state.get('alignment_spec') == _hr_alignment_spec(engine.alignment), 'HR head architecture/patch target differs')
    if head is not None:
        _need(isinstance(saved_head,dict) and set(saved_head) == set(head), 'HR head state names differ')
        for name,current in head.items():
            value = saved_head[name]
            _need(isinstance(value,torch.Tensor) and value.shape == current.shape and value.dtype == current.dtype
                  and not value.requires_grad and bool(torch.isfinite(value).all()), 'HR head shape/dtype/values differ: '+name)
    _need(state.get('optimizer_parameter_names') == groups, 'HR optimizer parameter names/order differ')
    optimizer = state.get('optimizer')
    _hr_validate_optimizer(optimizer,engine.optimizer.state_dict(),groups,named,step)
    streams = ('noise','timestep','mask','corruption')
    _need(isinstance(state.get('explicit_RNG'),dict) and set(state['explicit_RNG']) == set(streams)
          and isinstance(state.get('RNG_devices'),dict) and set(state['RNG_devices']) == set(streams), 'Incomplete HR explicit RNG streams')
    for name in streams:
        generator = getattr(engine.rng,name,None); value = state['explicit_RNG'][name]
        _need(isinstance(generator,torch.Generator) and state['RNG_devices'][name] == str(generator.device)
              and isinstance(value,torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1
              and value.device.type == 'cpu', 'HR RNG device/state differs: '+name)
        torch.Generator(device=generator.device).set_state(value)
    _need(_sha256_file(path) == checksum, 'HR checkpoint changed during validation')
    with torch.no_grad():
        for name,param in student.items():param.copy_(saved_student[name].to(param.device))
    if head is not None:engine.alignment.load_state_dict(saved_head,strict=True)
    engine.optimizer.load_state_dict(optimizer)
    for name in streams:getattr(engine.rng,name).set_state(state['explicit_RNG'][name])
    engine.steps = step
    engine.optimizer.zero_grad(set_to_none=True)
    if hasattr(engine.model,'clear_features'):engine.model.clear_features()
    return {'path':str(path), 'sha256':checksum, 'steps':step, 'step':step,
            'cumulative_training_seconds':float(seconds), 'initialization':payload['initialization'],
            'matched_Base':state['matched_Base'], 'config_sha256':config['_sha256'],
            'protocol_sha256':expected_protocol_sha256, 'weight':engine.weight, 'student_layer':engine.student_layer,
            'student_restored':True, 'alignment_restored':head is not None, 'optimizer_state_restored':True,
            'explicit_RNG_restored':list(streams), 'frozen_official_parameters_restored':False,
            'generic_inference_loader_used':False}


# A single admission/lease implementation is shared by GPU entry points.
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import re
import stat
import subprocess


def _gpu_query(arguments):
    result = subprocess.run(['nvidia-smi', *arguments], check=True, capture_output=True,
                            text=True, timeout=15)
    return list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))


def _gpu_selection(device, admission):
    selected = torch.device(device)
    _need(selected.type == 'cuda', 'GPU lease requires an explicit CUDA device')
    expected = admission.get('expected_gpu_uuid')
    _need(isinstance(expected, str) and expected.startswith('GPU-') and ',' not in expected,
          'GPU admission requires expected_gpu_uuid for a full GPU device')
    index = 0 if selected.index is None else selected.index
    rows = _gpu_query(['--query-gpu=index,uuid', '--format=csv,noheader'])
    devices = {str(int(row[0].strip())): row[1].strip() for row in rows if len(row) == 2}
    _need(len(devices) == len(rows) and len(set(devices.values())) == len(rows), 'Ambiguous nvidia-smi device inventory')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        identifiers = [item.strip() for item in visible.split(',')]
        _need(0 <= index < len(identifiers) and len(identifiers) == len(set(identifiers)), 'CUDA_VISIBLE_DEVICES does not identify the requested device uniquely')
        token = identifiers[index]
        uuid = token if token.startswith('GPU-') else devices.get(token)
        _need(uuid in devices.values(), 'CUDA_VISIBLE_DEVICES is not an exact full-GPU UUID/index mapping')
    else:
        _need(len(devices) == 1 or os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID',
              'Multiple GPUs require an explicit UUID visibility mapping or CUDA_DEVICE_ORDER=PCI_BUS_ID')
        uuid = devices.get(str(index))
    _need(uuid == expected, 'Requested CUDA device does not match expected_gpu_uuid')
    return {'torch_device': str(selected), 'gpu_uuid': uuid, 'CUDA_VISIBLE_DEVICES': visible,
            'CUDA_DEVICE_ORDER': os.environ.get('CUDA_DEVICE_ORDER'), 'device_inventory': devices}


def _desktop_identity(pid):
    base = Path('/proc') / str(pid)
    status_text = (base / 'status').read_text()
    uid_line = next(line for line in status_text.splitlines() if line.startswith('Uid:'))
    uids = [int(value) for value in uid_line.split()[1:]]
    executable = (base / 'exe').resolve(strict=True)
    info = executable.stat()
    process_stat = (base / 'stat').read_text()
    start_ticks = int(process_stat.rsplit(')', 1)[1].split()[19])
    return {'pid': pid, 'uid': uids[0], 'uids': uids, 'exe': str(executable),
            'exe_sha256': _sha256_file(executable), 'start_ticks': start_ticks,
            'exe_inode': info.st_ino, 'exe_device': info.st_dev,
            'exe_mode': stat.S_IMODE(info.st_mode), 'exe_owner': info.st_uid,
            'cgroup': (base / 'cgroup').read_text()}


def _desktop_cgroup_matches(actual):
    """Recognize complete same-user desktop units, never a name substring."""
    rows = actual.get('cgroup', '').splitlines()
    if len(rows) != 1 or not rows[0].startswith('0::'):
        return False
    path = rows[0][3:]
    uid = actual['uid']
    if type(actual.get('pid')) is not int or actual['pid'] <= 0:
        return False
    prefix = re.escape(f'/user.slice/user-{uid}.slice/user@{uid}.service')
    if actual['exe'] == '/usr/bin/gjs-console':
        pattern = prefix + r'/session\.slice/org\.gnome\.Shell@ubuntu\.service'
    elif actual['exe'] == '/usr/bin/nautilus':
        pattern = (prefix + r'/app\.slice/app-dbus\\x2d:(?P<bus>\d+\.\d+)\\x2d'
                   r'(?P<endpoint>org\.gnome\.Nautilus|org\.freedesktop\.FileManager1)\.slice/'
                   r'dbus-:(?P=bus)-(?P=endpoint)@\d+\.service')
    elif actual['exe'] == '/usr/bin/ptyxis':
        pattern = (prefix + r'/app\.slice/app-gnome-xdg\\x2dterminal\\x2dexec-'
                   + str(actual['pid']) + r'\.scope')
    else:
        return False
    return re.fullmatch(pattern, path) is not None


def _gpu_admission(device, admission):
    _need(isinstance(admission, dict), 'Explicit GPU admission dictionary required')
    selection = _gpu_selection(device, admission)
    raw = _gpu_query(['--id=' + selection['gpu_uuid'],
                      '--query-compute-apps=pid,process_name,used_gpu_memory',
                      '--format=csv,noheader,nounits'])
    processes = []
    for row in raw:
        _need(len(row) == 3, 'Unrecognized GPU process inventory row')
        pid, name, memory = int(row[0].strip()), row[1].strip(), float(row[2].strip())
        _need(pid > 0 and math.isfinite(memory) and memory >= 0, 'Invalid GPU PID/memory inventory')
        processes.append({'pid': pid, 'process_name': name, 'used_memory_mib': memory})
    _need(len({row['pid'] for row in processes}) == len(processes), 'Duplicate GPU process PID')
    pins = admission.get('desktop_processes', [])
    _need(isinstance(pins, list) and all(isinstance(pin, dict) for pin in pins), 'desktop_processes must be a list of complete pinned identities')
    pinned = {pin.get('pid'): pin for pin in pins}
    _need(len(pinned) == len(pins) and all(type(pid) is int and pid > 0 for pid in pinned), 'Duplicate/invalid desktop admission PIDs')
    _need(sum(pin.get('exe') == '/usr/bin/gjs-console' for pin in pins) <= 1,
          'Only one pinned GNOME Shell process is admitted')
    _need({row['pid'] for row in processes} == set(pinned), 'Unexpected or missing GPU process; no unknown/ML process is admitted')
    admitted = []
    for row in processes:
        pin = pinned[row['pid']]
        required = ('pid', 'uid', 'exe', 'exe_sha256', 'start_ticks', 'exe_inode', 'exe_device', 'exe_mode', 'process_name', 'max_memory_mib')
        _need(all(key in pin for key in required), 'Desktop admission requires a complete pinned process identity')
        maximum = pin['max_memory_mib']
        _need(type(maximum) in (int, float) and 0 < maximum <= 256 and row['used_memory_mib'] <= maximum,
              'Desktop process GPU memory exceeds its explicit <=256MiB cap')
        actual = _desktop_identity(row['pid'])
        _need(pin['uid'] == os.getuid() and actual['uids'] == [os.getuid()] * 4, 'Desktop process UID mismatch')
        _need(actual['exe'] in ('/usr/bin/gjs-console', '/usr/bin/nautilus', '/usr/bin/ptyxis'), 'Only verified system desktop executables are admitted')
        _need(actual['exe_owner'] == 0 and actual['exe_mode'] & 0o022 == 0 and _desktop_cgroup_matches(actual),
              'Desktop executable ownership/mode/session-cgroup mismatch')
        _need(all(actual[key] == pin[key] for key in required if key not in ('max_memory_mib', 'process_name'))
              and row['process_name'] == pin['process_name'], 'Desktop process identity changed')
        admitted.append({**row, **actual})
    return {'checked_at_UTC': datetime.now(timezone.utc).isoformat(), **selection,
            'inventory': processes, 'admitted_desktop': admitted, 'unknown_processes_admitted': False}


@contextmanager
def gpu_lease(lock_path, device, admission):
    """Nonblocking host lock and fresh UUID/process admission through cleanup.

    Importing this module never invokes nvidia-smi. An empty process inventory
    is the default. Optional desktop rows require complete host-pinned metadata;
    no process is killed, filtered, retried or silently allowed.
    """
    path = Path(lock_path)
    _need(path.is_absolute() and path.parent.is_dir(), 'GPU lock requires an explicit absolute file in an existing directory')
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    locked = False
    try:
        info = os.fstat(descriptor)
        _need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
              and stat.S_IMODE(info.st_mode) & 0o022 == 0,
              'GPU lock must be an owned regular file without aliases')
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise RuntimeError('GPU lease is already held; refusing overlapping execution') from error
        receipt = _gpu_admission(device, admission)
        receipt.update(lock_path=str(path.resolve()), lease_held_through_cleanup=True)
        yield receipt
    finally:
        if locked: fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
