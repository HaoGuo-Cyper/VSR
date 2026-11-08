"""Unified NoRef training primitives and explicit prepare/execute entry points.

No command, training queue, model load or CUDA initialization runs on import.
The teacher sees the SAME noised HR and native timestep as the student; only
its LR condition is clean. EMA swapping finishes before the student graph is
built. Device-batched finite checks are CPU-tested, but do not establish the
G3 <=1.5 GPU cost requirement; actual GPU profiling and gates remain pending.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .teacher import TrainableEMA, capture_block_output, _require_finite
from .losses import MaskedTemporalAlignment, flow_matching_loss
from .data import sample_span_mask, corrupt_latent_condition


ARMS = ("flow_only", "C1", "A2")


def _need(value, message):
    if not value:
        raise ValueError(message)


def _modes(module):
    return [(child, child.training) for child in module.modules()]


def _restore_modes(flags):
    for module, training in flags:
        module.training = training


def _frozen(model):
    return {name: (id(p), p.data_ptr(), p._version, tuple(p.shape), p.dtype, p.device)
            for name, p in model.named_parameters() if not p.requires_grad}


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()




class MatchedStepRNG:
    """Independent explicit streams; auxiliary draws cannot perturb FM draws."""

    def __init__(self, seed, device):
        _need(type(seed) is int and seed >= 0, "Explicit nonnegative continuation seed required")
        self.seed, self.device = seed, torch.device(device)
        self.noise = torch.Generator(device=self.device).manual_seed(seed + 1)
        self.timestep = torch.Generator(device="cpu").manual_seed(seed + 2)
        self.mask = torch.Generator(device="cpu").manual_seed(seed + 3)
        self.corruption = torch.Generator(device=self.device).manual_seed(seed + 4)

    def flow_inputs(self, clean, scheduler):
        _need(clean.device == self.device, "FM generator device differs from the latent device")
        _need(len(scheduler.timesteps) == 1000 and len(scheduler.sigmas) >= 1000,
              "Use the original native UniPC 1000-index training grid")
        index = int(torch.randint(0, 1000, (), generator=self.timestep))
        sigma = scheduler.sigmas[index].to(device=clean.device, dtype=torch.float32)
        timestep = scheduler.timesteps[index].to(clean.device).expand(clean.shape[0])
        _need(sigma.ndim == 0 and bool(torch.isfinite(sigma)) and 0 <= float(sigma) <= 1,
              "Expected a finite scalar native scheduler sigma in [0,1]")
        noise = torch.randn(clean.shape, generator=self.noise, device=clean.device, dtype=torch.float32)
        return (1 - sigma) * clean + sigma * noise, noise - clean, timestep, sigma, index


def _restoration_contract(model, student_layer, teacher_layer):
    _need(getattr(model, "projector", "not Base") is None and not hasattr(model, "reference_adapter"),
          "Pass a NoRef Base model; keep the student-only projector outside the restoration model")
    blocks = model.transformer.blocks
    _need(type(student_layer) is int and type(teacher_layer) is int
          and max(model.injection_layers) <= student_layer < teacher_layer < len(blocks),
          "Student must follow all LR injections and teacher must be deeper")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            _need(name.startswith(("lrv_stem.", "lrv_residuals.")) or ".lora_" in name,
                  "Unexpected restoration trainable: " + name)
    _need(all(not isinstance(module, nn.Dropout) or module.p == 0 for module in model.modules()),
          "Matched flow/C1/A2 numerics require the Base model's zero-dropout contract")


class NoRefContinuationStep:
    """One fresh-AdamW continuation arm, without a dataset/queue/checkpoint writer.

    A2/rho=0 is the clean flow-only path. A2/gamma=0 with rho>0 is C1's
    augmentation-only path. Every arm uses matched data order and FM streams.
    Teacher output is detached, from block21 under EMA, before student block9.
    """

    def __init__(self, model, *, arm, training_seed, rho=.25, span=2, gamma=.8,
                 student_layer=9, teacher_layer=21, hidden_dim=1024, ema_decay=.999,
                 learning_rate=1e-4, weight_decay=.01, max_grad_norm=1.,
                 feature_diagnostics_first=True, feature_diagnostics_every=512):
        _need(arm in ARMS, "Unknown continuation arm")
        _need(type(gamma) in (int, float) and math.isfinite(gamma) and gamma >= 0, "gamma must be finite and nonnegative")
        _need(type(max_grad_norm) in (int, float) and math.isfinite(max_grad_norm) and max_grad_norm > 0, "Positive gradient norm limit required")
        _need(type(feature_diagnostics_first) is bool and type(feature_diagnostics_every) is int
              and feature_diagnostics_every >= 0,"Use boolean first-active and nonnegative integer feature diagnostic frequency")
        _restoration_contract(model, student_layer, teacher_layer)
        self.model, self.arm, self.training_seed = model, arm, training_seed
        self.rho, self.span, self.gamma = rho, span, float(gamma)
        self.student_layer, self.teacher_layer = student_layer, teacher_layer
        self.max_grad_norm, self.steps = float(max_grad_norm), 0
        self.feature_diagnostics_first = feature_diagnostics_first
        self.feature_diagnostics_every = feature_diagnostics_every
        self._first_feature_snapshot_collected = False
        self._pending_feature_diagnostic = None
        device = next(model.parameters()).device
        self.rng = MatchedStepRNG(training_seed, device)
        self.ema = TrainableEMA(model, decay=ema_decay) if arm == "A2" else None
        self.alignment = None
        if arm == "A2":
            dimension = model.transformer.config.num_attention_heads * model.transformer.config.attention_head_dim
            # Isolate CPU projector initialization; do not seed or advance any
            # default CUDA generator or the independently matched FM streams.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator(device="cpu").manual_seed(training_seed + 5).get_state())
                self.alignment = MaskedTemporalAlignment(dimension, dimension, hidden_dim=hidden_dim).to(device)
        named = [("restoration." + name, p) for name, p in model.named_parameters() if p.requires_grad]
        if self.alignment is not None:
            named.extend(("student_projector." + name, p) for name, p in self.alignment.projector.named_parameters())
        parameters = [p for _, p in named]
        _need(len({id(p) for p in parameters}) == len(parameters), "Optimizer parameter aliases are forbidden")
        # Explicit foreach setting is common to every future comparison arm.
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay, foreach=False)
        _need(not self.optimizer.state, "Continuation optimizer must start fresh")
        self.parameters = parameters
        self._named_optimizer_parameters = tuple(named)
        self.base_load_audit = None

    def data_index(self, training_clip_count):
        _need(type(training_clip_count) is int and training_clip_count > 0, "Positive fixed training clip count required")
        return self.steps % training_clip_count

    def teacher_features(self, noisy, timestep, text, clean_low, *, layer=None):
        _need(self.ema is not None, "Only A2 owns an EMA teacher")
        target_layer = self.teacher_layer if layer is None else layer

        def call():
            try:
                return capture_block_output(self.model, target_layer, noisy, timestep, text, clean_low)[1]
            finally:
                self.model.clear_features()

        return self.ema.teacher_forward(self.model, call)

    def drain_feature_diagnostics(self):
        """CPU-only low-frequency stats; call outside any Step timing block."""
        if self._pending_feature_diagnostic is None:return None
        from .metrics import analyze_feature_snapshot
        result = analyze_feature_snapshot(self._pending_feature_diagnostic)
        self._pending_feature_diagnostic = None
        return result

    def step(self, clean, low, text, scheduler):
        _need(torch.is_grad_enabled(), "Continuation step requires enabled student autograd")
        _need(clean.ndim == 5 and clean.shape == low.shape and clean.shape[2] == 9
              and clean.dtype == low.dtype == torch.float32 and clean.device == low.device
              and not clean.requires_grad and not low.requires_grad
              and bool(torch.isfinite(clean).all()) and bool(torch.isfinite(low).all()),
              "Use frozen FP32 matched HR/LR latents with nine time positions")
        model_modes = _modes(self.model)
        projector_modes = _modes(self.alignment) if self.alignment is not None else []
        frozen = _frozen(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.clear_features()
        try:
            self.model.train()
            if self.alignment is not None:
                self.alignment.train()
            noisy, target, timestep, sigma, index = self.rng.flow_inputs(clean, scheduler)
            plan = sample_span_mask(clean.shape[0], 9, rho=0. if self.arm == "flow_only" else self.rho,
                                    span=self.span, generator=self.rng.mask)
            student_low = (low.clone() if self.arm == "flow_only" else
                           corrupt_latent_condition(low, plan.mask, generator=self.rng.corruption))
            # Compare in sigma's tensor precision, matching the loss window.
            # Converting FP32 .3 to Python float would spuriously exceed .3.
            active = self.arm == "A2" and self.gamma > 0 and bool(plan.mask.any()) and bool(sigma > .3)
            collect_features = active and ((self.feature_diagnostics_first and not self._first_feature_snapshot_collected)
                or (self.feature_diagnostics_every > 0 and (self.steps+1) % self.feature_diagnostics_every == 0))
            _need(not collect_features or self._pending_feature_diagnostic is None,
                  "Drain the previous detached CPU feature snapshot before another scheduled snapshot")
            # An active teacher validates source AND shadows before its swap.
            # The disabled-auxiliary path must also reject a corrupt EMA before
            # a student optimizer mutation; the post-step update checks again.
            if self.ema is not None and not active:
                self.ema._validate(self.model)
            # SAME noisy HR/timestep/text; teacher's condition alone stays clean.
            teacher = self.teacher_features(noisy, timestep, text, low) if active else None
            if active:
                flow, student = capture_block_output(self.model, self.student_layer, noisy, timestep, text, student_low)
            else:
                with _autocast(clean.device):
                    flow, projection = self.model(noisy, timestep, text, student_low)
                _need(projection is None, "Base must not include the historical DINO projector")
                student = None
            fm = flow_matching_loss(flow, target)
            if active:
                _need(teacher is not None and not teacher.requires_grad, "Teacher feature graph must be detached")
                sd, diagnostic = self.alignment(student, teacher, plan.mask, tuple(self.model.token_grid), sigma,
                                                capture_diagnostics=collect_features)
            else:
                sd = flow.reshape(-1)[:0].sum()
                diagnostic = {"selected_tokens": 0, "cosine_mean": None}
            total = fm + self.gamma * sd
            _need(bool(torch.isfinite(total)), "Nonfinite continuation objective")
            total.backward()
            _need(any(p.grad is not None for p in self.model.parameters() if p.requires_grad), "No restoration gradient")
            _need(all(p.grad is None for p in self.model.parameters() if not p.requires_grad), "Frozen backbone received a gradient")
            _require_finite(("Nonfinite continuation gradients: " + name, p.grad)
                            for name, p in self._named_optimizer_parameters if p.grad is not None)
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)
            _need(frozen == _frozen(self.model), "Frozen backbone parameters changed")
            self.steps += 1
            cosine_mean = float(diagnostic['cosine_mean']) if diagnostic['cosine_mean'] is not None else None
            if collect_features:
                self._pending_feature_diagnostic = {'step':self.steps,'student_layer':self.student_layer,
                    'teacher_layer':self.teacher_layer,'sigma':float(sigma),'selected_tokens':diagnostic['selected_tokens'],
                    'cosine_mean':cosine_mean,'schedule':{'first_active':self.feature_diagnostics_first,
                        'every_steps':self.feature_diagnostics_every},'features':diagnostic['feature_snapshot']}
                self._first_feature_snapshot_collected = True
            return {"step": self.steps, "arm": self.arm, "timestep_index": index, "native_timestep": float(timestep[0]),
                    "sigma": float(sigma), "fm": float(fm.detach()), "sd": float(sd.detach()),
                    "total": float(total.detach()), "grad_norm_before_clip": float(norm),
                    "teacher_forward": active, "teacher_student_same_noisy_HR_and_timestep": True,
                    "rho_requested": plan.rho_requested, "rho_realized_all_latents": plan.realized_ratio_all_latents,
                    "rho_realized_eligible_latents": plan.realized_ratio_eligible_latents,
                    "mask_spans": plan.spans, "mask_rounding": plan.rounding,
                    "selected_tokens": diagnostic["selected_tokens"],
                    "cosine_mean":cosine_mean,"cosine_layers":{"student":self.student_layer,"teacher":self.teacher_layer},
                    "feature_diagnostics_snapshot_collected":collect_features,
                    "student_layer": self.student_layer, "teacher_layer": self.teacher_layer,
                    "optimizer_state_fresh_at_initialization": True, "EMA_updates": self.ema.num_updates if self.ema else 0}
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.clear_features()
            _restore_modes(model_modes)
            _restore_modes(projector_modes)


def load_matched_base_continuation(checkpoint, *, expected_checkpoint_sha256, expected_training_seed,
                                   allowed_root, arm, device="cpu", model_directory=None, **options):
    """Explicit future loader: task parameters restored, optimizer state ignored.

    The actual caller must supply the new-run allowed_root and declared matched
    Base SHA/seed. This function is not called by import, tests use a tiny mock.
    """
    from .checkpoints import load_training_checkpoint
    loader_options = {} if model_directory is None else {"model_directory":model_directory}
    model, audit = load_training_checkpoint(checkpoint, device=device, allowed_root=Path(allowed_root), variant="Base", **loader_options)
    config = audit["config"]
    _need(audit["checkpoint_sha256"] == expected_checkpoint_sha256 and audit["step"] == 8192 and audit["variant"] == "Base",
          "Continuation must use the declared final matched Base checkpoint")
    _need(config["training"].get("seed", config["seed"]) == expected_training_seed
          and [config["data"][k] for k in ("num_frames", "temporal_stride", "height", "width", "scale")] == [33, 1, 128, 224, 4]
          and not config.get("reference", {}).get("enabled", False), "Base training seed/protocol/family mismatch")
    _need(all(audit[key] is False for key in ("optimizer_state_restored", "scheduler_state_restored", "rng_state_restored")),
          "Do not resume Base optimizer/scheduler/RNG state")
    options.setdefault("learning_rate", config["training"]["learning_rate"])
    options.setdefault("weight_decay", config["training"]["weight_decay"])
    options.setdefault("max_grad_norm", config["training"]["max_grad_norm"])
    result = NoRefContinuationStep(model, arm=arm, training_seed=expected_training_seed, **options)
    result.base_load_audit = audit
    return result


# Shared runner: metadata-only preparation, explicit fresh execution, and exact
# resource/pilot budgets. These modes never run as an import side effect.
import argparse
import copy
from contextlib import contextmanager
import gc
import hashlib
import json
import os
import platform
import statistics
import struct
import sys
import time
import traceback
import weakref

from .data import read_json as read, file_hash as file_sha, write_json_new

need = _need
ROLES = ARMS
WARMUP, TIMED, SEMANTIC_STEPS = 2, 20, 2
SIGMA_BOUNDARY = struct.unpack('f',struct.pack('f',.3))[0]
STRICT_ENVIRONMENT = {'CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HUB_OFFLINE':'1',
                      'TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false'}
METHOD = {'training_seed':20261102,'frames':33,'latent_frames':9,'latent_shape':[1,16,9,16,28],
          'rho':.25,'span':2,'gamma':.8,
          'student_layer':9,'teacher_layer':21,'hidden_dim':1024,'ema_decay':.999,
          'feature_diagnostics_first':True,'feature_diagnostics_every':512}
SMOKE_BUDGET = {'scientific_updates':0,'diagnostic_updates':3,'restores':1,'inference_steps':20}
SMOKE_BASE_SHA = '28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f'

G3_BUDGET = {'semantic_updates':4,'benchmark_updates':66,'benchmark_warmup_updates':6,
             'benchmark_timed_updates':60,'total_diagnostic_updates':70,
             'scientific_training_updates':0}


def save(path, value, *, replace=False):
    """Atomic JSON publication; replacement is only for live status/progress."""
    path = Path(path)
    if not replace:
        write_json_new(path,value)
        return
    need(path.name in {'status.json','progress.json'},'Only live status/progress may be replaced')
    import tempfile
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def binding(path, expected=None):
    path=Path(path).resolve()
    need(path.is_file(),'Required asset missing: '+str(path))
    actual=file_sha(path)
    need(expected is None or actual==expected,'Asset checksum changed: '+str(path))
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':actual}


def _memory(torch,memory_cap_bytes):
    device=torch.cuda.current_device()
    result={'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'physical_device_bytes':torch.cuda.get_device_properties(device).total_memory}
    need(max(result['peak_allocated_bytes'],result['peak_reserved_bytes']) <=
         min(memory_cap_bytes,result['physical_device_bytes']),'Declared/physical GPU memory ceiling exceeded')
    return result


def summarize_roles(rows):
    """Pure-stdlib predeclared ratio over every timed row, never selected scores."""
    groups = {}
    for role in ROLES:
        values = [x for x in rows if x['role'] == role]
        need(len(values) == WARMUP+TIMED and [x['ordinal'] for x in values] == list(range(WARMUP+TIMED))
             and [x['phase'] for x in values] == ['warmup']*WARMUP+['timed']*TIMED, 'Incomplete/reordered role timings')
        need(all(type(x['seconds']) in (int,float) and math.isfinite(x['seconds']) and x['seconds'] > 0
                 and all(type(x[k]) in (int,float) and math.isfinite(x[k]) for k in ('fm','sd','total','sigma'))
                 for x in values), 'Nonfinite timing/objective observation')
        groups[role] = values
    need(len(rows) == 66, 'Exactly66 benchmark updates required')
    need([x['role'] for x in rows] == [role for role in ROLES for _ in range(WARMUP+TIMED)], 'Roles must run sequentially flow_only/C1/A2')
    for ordinal in range(WARMUP+TIMED):
        first = groups['flow_only'][ordinal]['matched_FM']
        need(all(groups[role][ordinal]['matched_FM'] == first for role in ROLES), 'Data/noise/sigma streams differ across roles')
        need(groups['C1'][ordinal]['mask_spans'] == groups['A2'][ordinal]['mask_spans'], 'C1/A2 corruption masks differ')
        need(groups['C1'][ordinal]['matched_corruption_RNG'] == groups['A2'][ordinal]['matched_corruption_RNG'],
             'C1/A2 corruption RNG streams differ')
        for role in ROLES:
            row = groups[role][ordinal]
            need(row['sigma'] == row['matched_FM']['sigma'] and 0 <= row['sigma'] <= 1
                 and row['teacher_forward'] == (role == 'A2' and row['sigma'] > SIGMA_BOUNDARY)
                 and (row['sd'] == 0 if not row['teacher_forward'] else True), 'Teacher window or disabled SD mismatch')
    def stats(values):
        return {'n':len(values),'mean_seconds':statistics.mean(values) if values else None,
                'sample_SD_seconds':statistics.stdev(values) if len(values)>1 else None}
    timed = {role:groups[role][WARMUP:] for role in ROLES}
    summaries = {role:stats([x['seconds'] for x in values]) for role,values in timed.items()}
    objectives = {role:{key:{'mean':statistics.mean(x[key] for x in values),
                            'sample_SD':statistics.stdev(x[key] for x in values)}
                        for key in ('fm','sd','total')} for role,values in timed.items()}
    active_indices = [i for i,x in enumerate(timed['A2']) if x['teacher_forward']]
    def ratio(numerator,denominator,indices=None):
        a,b = timed[numerator],timed[denominator]
        indices = list(range(TIMED)) if indices is None else indices
        return statistics.mean(a[i]['seconds'] for i in indices)/statistics.mean(b[i]['seconds'] for i in indices) if indices else None
    primary = ratio('A2','flow_only')
    return {'status':'passed_G3_resource_cost' if primary <= 1.5 else 'failed_G3_overhead_stop',
        'primary_ratio_A2_over_flow_all20':primary,'maximum_ratio':1.5,'role_timing':summaries,'timed_objectives':objectives,
        'secondary':{'A2_over_C1_all20':ratio('A2','C1'),'C1_over_flow_all20':ratio('C1','flow_only'),
            'A2_over_flow_same_active_indices':ratio('A2','flow_only',active_indices),
            'A2_over_C1_same_active_indices':ratio('A2','C1',active_indices),
            'teacher_active_count_timed':len(active_indices),'active_timed_indices':active_indices,
            'teacher_active_count_including_warmup':sum(x['teacher_forward'] for x in groups['A2']),
            'active_A2_timing':stats([timed['A2'][i]['seconds'] for i in active_indices])},
        'observations':66,'timed_observations':60,'warmup_observations':6,'outliers_removed':0,
        'full_rows_retained':True,'scope':'single-seed resource diagnostic on real cached training latents; no quality or RE conclusion'}


class UpdateCounter:
    """Count successful AdamW calls even if later EMA/contract checks fail."""
    def __init__(self):
        self.role = None
        self.by_role = {}
        self.cleanup_events = []
    def attach(self,optimizer,role):
        self.role = role
        bucket = self.by_role.setdefault(role,{'attempted':0,'applied':0,'returned_steps':0})
        original = optimizer.step
        def counted(*args,**kwargs):
            bucket['attempted'] += 1
            result = original(*args,**kwargs)
            bucket['applied'] += 1
            return result
        optimizer.step = counted
    def snapshot(self):
        return {'by_role':copy.deepcopy(self.by_role),
            'diagnostic_optimizer_updates_attempted':sum(x['attempted'] for x in self.by_role.values()),
            'diagnostic_optimizer_updates_applied':sum(x['applied'] for x in self.by_role.values()),
            'completed_step_returns':sum(x['returned_steps'] for x in self.by_role.values()),
            'optimizer_calls_not_returned_partial_effect_unknown':sum(x['attempted']-x['applied'] for x in self.by_role.values()),
            'applied_count_scope':'AdamW calls that successfully returned; a throwing AdamW call may have partial effects and is never resumed',
            'cleanup_events':copy.deepcopy(self.cleanup_events),'scientific_training_updates':0}


def _state_signature(model):
    return [(name,id(value),value.data_ptr(),value._version,tuple(value.shape),str(value.dtype),str(value.device),value.requires_grad)
            for name,value in list(model.named_parameters())+list(model.named_buffers())]


def _frozen_signature(model):
    return [(name,id(p),p.data_ptr(),p._version,tuple(p.shape),str(p.dtype),str(p.device))
            for name,p in model.named_parameters() if not p.requires_grad]


def _tensor_digest(torch,value):
    cpu = value.detach().cpu().contiguous()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(json.dumps([str(cpu.dtype),list(cpu.shape)]).encode()+b'\n'+raw).hexdigest()


def _streams(torch,engine,keys=('noise','timestep')):
    return {key:_tensor_digest(torch,getattr(engine.rng,key).get_state()) for key in keys}


def _cleanup_engine(torch,engine,*,successful=True,counter=None):
    try:
        if engine is not None:
            engine.optimizer.zero_grad(set_to_none=True)
            engine.model.clear_features()
    except BaseException as error:
        if successful:raise
        counter.cleanup_events.append({'cleanup_error':f'{type(error).__name__}: {error}',
            'original_exception_preserved':True})


def _check_release(torch,reference,successful,counter,role):
    """Propagating traceback frames legitimately keep a failed engine alive."""
    try:
        gc.collect();torch.cuda.empty_cache()
        released = reference is None or reference() is None
        if successful:need(released,'Diagnostic model/optimizer remains resident across roles: '+role)
        else:counter.cleanup_events.append({'role':role,'release_assert_skipped_during_original_exception':True,
            'model_reference_alive_via_possible_traceback':not released,'process_must_stop':True})
    except BaseException as error:
        if successful:raise
        counter.cleanup_events.append({'role':role,'cleanup_error':f'{type(error).__name__}: {error}',
            'original_exception_preserved':True})


def _snapshot_core(engine):
    result = {}
    for name,p in engine.model.named_parameters():
        if p.requires_grad:
            result[name] = {'parameter':p.detach().cpu().clone(),
                'optimizer':{key:value.detach().cpu().clone() if hasattr(value,'detach') else value
                             for key,value in engine.optimizer.state[p].items()}}
    return result


def _equal_snapshots(torch,expected,actual):
    need(set(expected) == set(actual), 'rho0 trainable names differ')
    for name,left in expected.items():
        right = actual[name]
        need(torch.equal(left['parameter'],right['parameter']) and set(left['optimizer']) == set(right['optimizer']),
             'rho0 exact parameter/optimizer mismatch: '+name)
        for key,value in left['optimizer'].items():
            other = right['optimizer'][key]
            need(torch.equal(value,other) if isinstance(value,torch.Tensor) else value == other,
                 'rho0 exact AdamW moment/count mismatch: '+name+'/'+key)


def run_semantics(torch,protocol,pairs,text,scheduler,counter):
    """Independent fresh models; no diagnostic state enters the timed roles."""
    from .teacher import capture_block_output
    from .teacher import _require_finite
    from .model import tensor_state_hashes
    reports, trajectories, scalar_rows = {}, [], []
    engine = clean = low = teacher = student = flow = target = noisy = loss = None
    reference = None
    try:
        engine = _new_engine(protocol,'A2',rho=0.)
        reference = weakref.ref(engine.model)
        before = tensor_state_hashes(engine.model)['all_trainable']['sha256']
        frozen = _frozen_signature(engine.model)
        flags = [(module,module.training) for module in engine.model.modules()]
        clean,low = (pairs[0][key].to('cuda') for key in ('clean','low'))
        noisy,target,timestep,sigma,_ = engine.rng.flow_inputs(clean,scheduler)
        teacher = engine.teacher_features(noisy,timestep,text,low,layer=9)
        engine.model.train()
        flow,student = capture_block_output(engine.model,9,noisy,timestep,text,low)
        need(not teacher.requires_grad and student.requires_grad and torch.equal(teacher,student),
             'GPU same-layer9 EMA=student exact equality failed; stop without relaxing tolerance')
        engine.model.clear_features()
        for module,flag in flags:module.training = flag
        need(before == tensor_state_hashes(engine.model)['all_trainable']['sha256']
             and frozen == _frozen_signature(engine.model)
             and all(p.grad is None for p in engine.model.parameters()), 'Teacher swap did not restore the student/frozen state')
        reports['same_layer'] = {'layer':9,'exact_equal':True,'teacher_detached':True,'student_restored':True,'optimizer_updates':0}
    finally:
        successful = sys.exc_info()[0] is None
        _cleanup_engine(torch,engine,successful=successful,counter=counter)
        flags = [];module = None
        engine = clean = low = teacher = student = flow = target = noisy = loss = None
        _check_release(torch,reference,successful,counter,'semantic_same_layer')
    for role in ('semantic_direct_core','semantic_A2_rho0'):
        reference = None
        try:
            engine = _new_engine(protocol,'flow_only' if role == 'semantic_direct_core' else 'A2',rho=0.)
            reference = weakref.ref(engine.model)
            counter.attach(engine.optimizer,role)
            frozen = _frozen_signature(engine.model)
            for ordinal in range(SEMANTIC_STEPS):
                clean,low = (pairs[ordinal][key].to('cuda') for key in ('clean','low'))
                if role == 'semantic_direct_core':
                    engine.model.train()
                    noisy,target,timestep,sigma,index = engine.rng.flow_inputs(clean,scheduler)
                    engine.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):flow,projection = engine.model(noisy,timestep,text,low)
                    need(projection is None,'Direct core must be Base-only')
                    loss = torch.nn.functional.mse_loss(flow.float(),target)
                    need(bool(torch.isfinite(loss)), 'Nonfinite semantic direct-core objective')
                    loss.backward()
                    _require_finite(('Nonfinite semantic core gradient: '+name,p.grad) for name,p in engine.model.named_parameters() if p.grad is not None)
                    need(all(p.grad is None for p in engine.model.parameters() if not p.requires_grad),'Frozen semantic reference received gradients')
                    torch.nn.utils.clip_grad_norm_(engine.parameters,engine.max_grad_norm,error_if_nonfinite=True)
                    engine.optimizer.step();engine.model.clear_features();engine.optimizer.zero_grad(set_to_none=True)
                    row = {'fm':float(loss.detach()),'sigma':float(sigma),'timestep_index':index}
                    trajectories.append(_snapshot_core(engine));scalar_rows.append(row)
                else:
                    result = engine.step(clean,low,text,scheduler)
                    row = {key:result[key] for key in ('fm','sigma','timestep_index')}
                    need(row == scalar_rows[ordinal] and result['sd'] == 0 and not result['teacher_forward'],
                         'GPU rho0 direct-core flow trajectory differs')
                    _equal_snapshots(torch,trajectories[ordinal],_snapshot_core(engine))
                counter.by_role[role]['returned_steps'] += 1
                need(frozen == _frozen_signature(engine.model), 'Frozen semantic parameters changed')
            reports[role] = {'updates':2,'exact_trajectory_equal':True if role.endswith('rho0') else None,
                'optimizer_foreach':False,'memory':_memory(torch,protocol['memory_cap_bytes'])}
        finally:
            successful = sys.exc_info()[0] is None
            _cleanup_engine(torch,engine,successful=successful,counter=counter)
            engine = clean = low = teacher = student = flow = target = noisy = loss = None
            _check_release(torch,reference,successful,counter,role)
    trajectories.clear()
    return {'status':'passed_exact_GPU_semantics','checks':reports,'diagnostic_optimizer_updates':4,
        'historical_GPU_foreach_None_equivalence_claimed':False,'tolerance_fallback':False}


def run_benchmark_role(torch,protocol,role,pairs,text,scheduler,counter,rows,run):
    from .teacher import _require_finite
    from .model import tensor_state_hashes
    engine = clean = low = None
    reference = None
    torch.cuda.reset_peak_memory_stats()
    try:
        engine = _new_engine(protocol,role)
        reference = weakref.ref(engine.model)
        counter.attach(engine.optimizer,role)
        frozen = _frozen_signature(engine.model)
        flags = [(module,module.training) for module in engine.model.modules()]
        before = tensor_state_hashes(engine.model)['all_trainable']['sha256']
        _require_finite(('Nonfinite loaded parameter: '+name,p) for name,p in engine.model.named_parameters())
        for ordinal in range(WARMUP+TIMED):
            item = protocol['inputs'][ordinal]
            before_rng = _streams(torch,engine)
            before_aux = _streams(torch,engine,('mask','corruption'))
            torch.cuda.synchronize()
            started = time.perf_counter()
            clean,low = (pairs[ordinal][key].to('cuda') for key in ('clean','low'))
            result = engine.step(clean,low,text,scheduler)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            counter.by_role[role]['returned_steps'] += 1
            feature_diagnostics = engine.drain_feature_diagnostics()
            after_rng = _streams(torch,engine)
            after_aux = _streams(torch,engine,('mask','corruption'))
            need(all(module.training == flag for module,flag in flags)
                 and frozen == _frozen_signature(engine.model)
                 and all(p.grad is None for p in engine.parameters)
                 and engine.model._condition_tokens is None and engine.model._captured is None,
                 'Role flags/frozen/gradient/feature cleanup contract changed')
            memory = _memory(torch,protocol['memory_cap_bytes'])
            row = {'role':role,'ordinal':ordinal,'phase':'warmup' if ordinal<WARMUP else 'timed','seconds':elapsed,
                **result,'memory':memory,'matched_FM':{'data_index':ordinal,'record_id':item['record_id'],
                    'latent_file_sha256':item['latents']['sha256'],'before_rng':before_rng,'after_rng':after_rng,
                    **{k:result[k] for k in ('timestep_index','native_timestep','sigma')}},
                'matched_corruption_RNG':{'before':before_aux,'after':after_aux},
                'feature_diagnostics_CPU_outside_step_timing':feature_diagnostics}
            rows.append(row)
            save(run/'progress.json',{'role':role,'completed_rows':len(rows),**counter.snapshot()},replace=True)
        _require_finite(('Nonfinite final parameter: '+name,p) for name,p in engine.model.named_parameters())
        if engine.ema is not None:engine.ema._validate(engine.model)
        return {'role':role,'fresh_Base_SHA256':protocol['matched_Base']['checkpoint']['sha256'],
            'trainable_sha256_before':before,'trainable_sha256_after':tensor_state_hashes(engine.model)['all_trainable']['sha256'],
            'returned_updates':engine.steps,'optimizer_updates_applied':counter.by_role[role]['applied'],
            'EMA_updates':engine.ema.num_updates if engine.ema else 0,'memory':_memory(torch,protocol['memory_cap_bytes']),
            'frozen_identity_versions_unchanged':True,'mode_and_gradient_cleanup_passed':True}
    finally:
        successful = sys.exc_info()[0] is None
        _cleanup_engine(torch,engine,successful=successful,counter=counter)
        flags = []
        engine = clean = low = None
        _check_release(torch,reference,successful,counter,role)



def _canonical_config(config):
    config=copy.deepcopy(config)
    config.pop('_sha256',None)
    config['_sha256']=hashlib.sha256((json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    return config


def _source_bindings():
    return {p.name:binding(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def _cache_spec(cache_root):
    root=Path(cache_root).resolve()
    selection=read(root/'selection.json')
    need(len(selection.get('training_ids',[]))==1641 and len(selection.get('validation_ids',[]))==32
         and selection.get('test_accessed') is False,'Use the complete fixed train1641/val32 cache')
    seal_path=root/'cache_asset_seal.json'
    seal=read(seal_path)
    hashes=seal.get('artifact_sha256')
    need(isinstance(hashes,dict) and hashes,'Complete cache artifact seal required')
    text_path=root/'fresh_text_embeddings.pt'
    result={'root':str(root),'selection':selection,'selection_file':binding(root/'selection.json'),
            'config':binding(root/'config.json'),'seal':binding(seal_path),
            'text':binding(text_path,hashes.get('fresh_text_embeddings.pt'))}
    need('fresh_text_embeddings.pt' in hashes,'Text cache must be sealed')
    inputs=[]
    for i,record_id in enumerate(selection['training_ids'][:WARMUP+TIMED]):
        relative=f'clip_{i}'
        mpath,lpath=root/relative/'metadata.json',root/relative/'fresh_latents.pt'
        meta=read(mpath)
        need(meta['record_id']==record_id and meta['split']=='train','G3 inputs must be the first22 fixed training clips')
        need(meta['source_indices']==list(range(meta['source_indices'][0],meta['source_indices'][0]+33)),
             'Cached training source indices must be33 contiguous frames')
        need(relative+'/metadata.json' in hashes and relative+'/fresh_latents.pt' in hashes,'Unsealed diagnostic input')
        inputs.append({'index':i,'record_id':record_id,'binding':meta['binding'],
                       'metadata':binding(mpath,hashes[relative+'/metadata.json']),
                       'latents':binding(lpath,hashes[relative+'/fresh_latents.pt'])})
    return result,inputs


def _base_spec(path, expected_sha, allowed_root, training_seed):
    path=Path(path).resolve()
    allowed_root=Path(allowed_root).resolve()
    need(path.is_relative_to(allowed_root) and path.name=='checkpoint_008192.pt' and path.parent.name=='Base',
         'Use an explicitly allowed final Base8192 checkpoint')
    config=read(path.parent.parent/'config.json')
    need(config.get('training',{}).get('updates')==8192 and config['training'].get('variants')==['Base']
         and config['training'].get('seed',config['seed'])==training_seed,'Matched Base8192 seed/config required')
    need([config['data'][k] for k in ('num_frames','temporal_stride','height','width','scale')]==[33,1,128,224,4]
         and not config.get('reference',{}).get('enabled',False),'Matched NoRef protocol33 Base required')
    return {'checkpoint':binding(path,expected_sha),'config':config,'config_file':binding(path.parent.parent/'config.json')}


def _passed_g2(directory):
    from .evaluate import validate_completed_base_evaluation
    return validate_completed_base_evaluation(directory)


def _passed_g3(directory,g2_evidence):
    directory=Path(directory).resolve()
    status,summary,protocol,rows=(read(directory/name) for name in ('status.json','summary.json','protocol.json','benchmark_rows.json'))
    need(status.get('status')=='completed' and summary.get('G3_passed') is True
         and summary.get('diagnostic_optimizer_updates_applied')==70
         and summary.get('scientific_training_updates')==0,'Require a completed real70-update passed G3')
    psha=file_sha(directory/'protocol.json')
    need(status['protocol_sha256']==summary['protocol_sha256']==psha
         and status['summary_sha256']==file_sha(directory/'summary.json')
         and summary['benchmark_rows_sha256']==file_sha(directory/'benchmark_rows.json'),'G3 completion hash chain differs')
    need(protocol.get('schema')=='vsr_training_protocol_v2' and protocol.get('mode')=='resource'
         and protocol.get('sources')==_source_bindings(),'G3 source/protocol differs from the current implementation')
    need(protocol.get('G2_evidence')==g2_evidence,'G3 did not use this completed Base/G2 evidence')
    need(collect_protocol(argparse.Namespace(**protocol['prepared_arguments']))==protocol,
         'G3 cache/model/dependency/admission inputs no longer match its sealed protocol')
    recomputed=summarize_roles(rows)
    need(recomputed==summary['resource_gate'] and recomputed['status']=='passed_G3_resource_cost'
         and summary['semantics']['status']=='passed_exact_GPU_semantics','G3 resource/semantic verdict does not reproduce')
    semantics=summary['semantics']
    checks=semantics['checks']
    need(semantics['diagnostic_optimizer_updates']==4 and checks['same_layer']['optimizer_updates']==0
         and all(checks['same_layer'].get(key) is True for key in ('exact_equal','teacher_detached','student_restored'))
         and checks['semantic_direct_core']['updates']==checks['semantic_A2_rho0']['updates']==2
         and checks['semantic_A2_rho0']['exact_trajectory_equal'] is True,'G3 exact semantic evidence is incomplete')
    need(set(summary['roles'])==set(ROLES) and all(summary['roles'][role]['returned_updates']==22
         and summary['roles'][role]['optimizer_updates_applied']==22
         and summary['roles'][role]['fresh_Base_SHA256']==protocol['matched_Base']['checkpoint']['sha256']
         and summary['roles'][role]['frozen_identity_versions_unchanged'] is True
         and summary['roles'][role]['mode_and_gradient_cleanup_passed'] is True for role in ROLES),
         'G3 roles are incomplete or did not use fresh matched Base')
    need(summary['diagnostic_optimizer_updates_attempted']==70 and summary['completed_step_returns']==70
         and summary['optimizer_calls_not_returned_partial_effect_unknown']==0,'G3 update accounting differs')
    return {'status':'passed_G3','run_dir':str(directory),'protocol_sha256':psha,
            'summary':binding(directory/'summary.json'),'status_file':binding(directory/'status.json'),
            'benchmark_rows':binding(directory/'benchmark_rows.json')}


def collect_protocol(args):
    mode=args.mode
    need(mode in {'base','resource','pilot','smoke'},'Unknown unified training mode')
    config=read(args.config)
    need(config.get('reproducibility',{}).get('mode')=='strict','Original strict math precision is mandatory')
    data=config['data']
    need([data.get(k) for k in ('num_frames','temporal_stride','height','width','scale')]==[33,1,128,224,4],
         'Keep accepted33stride1/128x224/x4 training geometry')
    need(not config.get('reference',{}).get('enabled',False),'Unified main line is NoRef')
    need(config.get('seed')==20261102,'Fixed data seed/order required')
    seed=config['training'].get('seed',config['seed'])
    need(type(seed) is int and seed in (20261102,20261103,20261104),'Use a declared Base training seed')
    need(mode=='base' or seed==20261102,'The resource gate and single-seed pilot use Base01')
    device=str(args.device)
    need(device.startswith('cuda:') and device[5:].isdigit(),'Production execution requires an explicit CUDA device')
    admission=read(args.gpu_admission)
    need(isinstance(admission.get('expected_gpu_uuid'),str) and admission['expected_gpu_uuid'],
         'Host GPU admission must pin its UUID')
    lock=Path(args.gpu_lock)
    need(lock.is_absolute(),'Use a shared absolute GPU lock path')
    cap=args.memory_cap_gib
    need(type(cap) in (int,float) and math.isfinite(cap) and cap>0,'Positive prospective memory ceiling required')
    cache,inputs=_cache_spec(args.cache_root)
    if mode=='smoke':inputs=inputs[:3]
    protocol={'schema':'vsr_training_protocol_v2','mode':mode,'sources':_source_bindings(),
        'config':config,'config_file':binding(args.config),'cache':cache,'inputs':inputs,
        'device':device,'admission':admission,'admission_file':binding(args.gpu_admission),
        'gpu_lock':str(lock),'memory_cap_bytes':int(cap*1024**3),
        'model_directory':str(Path(args.model_directory or config['model']['local_dir']).resolve()),
        'official_model_files':_official_bindings(args.model_directory or config['model']['local_dir']),
        'dependencies':_dependency_bindings(),
        'allowed_checkpoint_root':str(Path(args.allowed_checkpoint_root).resolve()),
        'method':METHOD,'optimizer':{k:config['training'][k] for k in ('learning_rate','weight_decay','max_grad_norm')},
        'fresh_only':True,'resume':False,'automatic_retry':False,
        'budget':{'scientific_updates':8192 if mode=='base' else 6144 if mode=='pilot' else 0,
                  'diagnostic_updates':70 if mode=='resource' else 0},
        'runtime':{'python':platform.python_version(),'torch':torch.__version__},
        'strict_environment':STRICT_ENVIRONMENT,
        'prepared_arguments':{key:str(getattr(args,key)) if isinstance(getattr(args,key),Path) else getattr(args,key)
                              for key in ('mode','config','cache_root','base_checkpoint','expected_base_sha',
                                  'allowed_checkpoint_root','g2_run','g3_run','device','gpu_admission','gpu_lock',
                                  'memory_cap_gib','g4_definitions','model_directory')}}
    if mode=='smoke':
        need(args.base_checkpoint and args.expected_base_sha==SMOKE_BASE_SHA,
             'Development smoke uses the audited matched Base01 only')
        protocol['matched_Base']=_base_spec(args.base_checkpoint,args.expected_base_sha,args.allowed_checkpoint_root,seed)
        need(protocol['matched_Base']['config']==config,'Smoke source config must equal the matched Base')
        need(not args.g2_run and not args.g3_run and not args.g4_definitions,
             'Development smoke must not consume or masquerade as scientific gates')
        protocol['development_only']=True
        protocol['budget']=dict(SMOKE_BUDGET)
        protocol['smoke']=_smoke_spec(protocol)
    elif mode=='base':
        need(config['training'].get('updates')==8192 and config['training'].get('variants')==['Base'],
             'A fresh Base is exactly8192 updates with variants=[Base]')
    else:
        need(args.base_checkpoint and args.expected_base_sha and args.g2_run,'Matched Base SHA and complete G2 evidence are mandatory')
        evaluation,evidence=_passed_g2(args.g2_run)
        protocol['G2_evidence']=evidence
        protocol['evaluation_protocol']=evaluation
        protocol['matched_Base']=_base_spec(args.base_checkpoint,args.expected_base_sha,args.allowed_checkpoint_root,seed)
        need(protocol['matched_Base']['config']==config,'Resource/pilot config must equal its matched Base source config')
        role=next((r for r in evaluation['roles'] if r['training_seed']==seed),None)
        need(role is not None and role['variant']=='Base' and role['checkpoint']==protocol['matched_Base']['checkpoint'],
             'Use the same completed Base checkpoint as G2')
        if mode=='pilot':
            need(args.g3_run and args.g4_definitions,'Pilot requires completed G3 and prospectively frozen G4 definitions')
            protocol['G3_evidence']=_passed_g3(args.g3_run,evidence)
            definition=binding(args.g4_definitions,'d552dbb136223d246e05af202f25b072a2c822ad4eeaecead0aa0d6360b1ca58')
            protocol['G4_definition_binding']={'definition':definition,'base_evaluation_protocol_sha256':evidence['protocol_sha256'],
                                              'definitions_modified':False,'binding_before_method_training':True}
    return protocol


def _safe_run(run_dir,run_root):
    path,root=Path(run_dir),Path(run_root)
    need(path.is_absolute() and root.is_absolute(),'Explicit absolute run directory/root required')
    path,root=path.resolve(),root.resolve()
    need(path.is_relative_to(root) and path!=root,'Run directory must be a new child of its output root')
    protected=[Path('/home/hugo/Projects/VSR'),Path('/home/hugo/Projects/ReprVSR-pretrained')]
    need(all(not path.is_relative_to(p.resolve()) for p in protected),'Historical/pretrained assets are read-only')
    return path,root


def prepare(args):
    run,root=_safe_run(args.run_dir,args.run_root)
    need(not run.exists(),'Fresh-only prepare; existing output directory is forbidden')
    protocol=collect_protocol(args)
    run.mkdir(parents=True)
    save(run/'protocol.json',protocol)
    sha=file_sha(run/'protocol.json')
    save(run/'status.json',{'status':'planned','protocol_sha256':sha,'scientific_training_updates':0,
                           'diagnostic_optimizer_updates_applied':0,'GPU_loaded':False})
    return {'status':'planned','run_dir':str(run),'protocol_sha256':sha,'budget':protocol['budget']}


def _new_engine(protocol,arm,*,rho=None):
    base=protocol['matched_Base']
    options={k:protocol['method'][k] for k in ('rho','span','gamma','student_layer','teacher_layer','hidden_dim',
                                             'ema_decay','feature_diagnostics_first','feature_diagnostics_every')}
    options.update(protocol['optimizer'])
    if rho is not None:options['rho']=rho
    engine=load_matched_base_continuation(base['checkpoint']['path'],expected_checkpoint_sha256=base['checkpoint']['sha256'],
        expected_training_seed=protocol['method']['training_seed'],allowed_root=protocol['allowed_checkpoint_root'],
        arm=arm,device=protocol['device'],model_directory=protocol['model_directory'],**options)
    need(engine.base_load_audit['config']==base['config'] and not engine.optimizer.state
         and all(x['foreach'] is False for x in engine.optimizer.param_groups),'Fresh matched Base/AdamW contract changed')
    return engine


def _load_inputs(protocol):
    from .data import PreparedTrainingData
    reader=PreparedTrainingData(protocol['cache']['root'],expected_seal_sha256=protocol['cache']['seal']['sha256'])
    need(reader.ids==protocol['cache']['selection']['training_ids'],'Training cache order changed')
    pairs=[reader[i] for i in range(len(protocol['inputs']))]
    text=reader.text_for(0)
    need(isinstance(text,torch.Tensor) and torch.isfinite(text).all(),'Fixed finite text cache required, no T5 fallback')
    return reader,pairs,text.to(protocol['device'])


def _method_config(base_config,arm):
    config=copy.deepcopy(base_config)
    config['restoration_architecture']='noref_base_v1'
    config['stage']='unified_protocol33_pilot'
    config['training']['variants']=[arm]
    config['training']['updates']=2048
    return _canonical_config(config)




def _smoke_config(base_config):
    """A bounded development artifact; never a formal pilot checkpoint."""
    config=copy.deepcopy(base_config)
    config['restoration_architecture']='noref_base_v1'
    config['stage']='development_smoke_5080'
    config['experiment_name']='development_smoke_only_not_formal'
    config['formal_training']={'enabled':False,'purpose':'development smoke, no scientific training RE'}
    config['training'].update(variants=['A2'],updates=3,evaluate_at=[],checkpoint_at=[3])
    return _canonical_config(config)


def _smoke_spec(protocol):
    """Freeze natural first3 native grid draws; do not select active steps."""
    from diffusers import UniPCMultistepScheduler
    scheduler=UniPCMultistepScheduler.from_pretrained(protocol['model_directory'],subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(1000,device='cpu')
    generator=torch.Generator(device='cpu').manual_seed(METHOD['training_seed']+2)
    indices=[int(torch.randint(0,1000,(),generator=generator)) for _ in range(3)]
    sigmas=[float(scheduler.sigmas[index]) for index in indices]
    need(indices==[23,954,808] and [value>SIGMA_BOUNDARY for value in sigmas]==[True,False,True],
         'The original natural development grid/window changed; do not select or retry')
    root=Path(protocol['cache']['root'])
    seal=read(protocol['cache']['seal']['path'])
    relative='clip_0/paired_rgb.pt'
    need(relative in seal['artifact_sha256'],'Original first training RGB must be sealed')
    return {'timestep_indices':indices,'sigmas':sigmas,'teacher_forward':[True,False,True],
            'training_clip_indices':[0,1,2],'inference_clip_index':0,'inference_seed':42,
            'input_RGB':binding(root/relative,seal['artifact_sha256'][relative]),
            'shape_HR':[1,3,33,128,224],'shape_LR':[1,3,33,32,56],
            'G2_G3_G4_passes_claimed':False,'checkpoint_reuse_for_formal_training':False}


def _run_development_smoke(protocol,run,reader,pairs,text,scheduler,vae,counter):
    """Exactly3 A2 diagnostic updates, atomic save/reload, one native restore."""
    from .model import tensor_state_hashes
    from .checkpoints import save_method_checkpoint,load_training_checkpoint
    from .infer import restore_video,verify_output_video
    from .evaluate import _crop_policy,_validate_restore,_save_pngs,_cleanup_preserving
    from .metrics import quantize_rgb
    from .data import write_video
    from PIL import Image
    import numpy as np
    need(protocol.get('development_only') is True and protocol['budget']==SMOKE_BUDGET
         and len(pairs)==3 and len(protocol['inputs'])==3,'The smoke budget is fixed, not a scientific gate bypass')
    config=_smoke_config(protocol['matched_Base']['config'])
    engine=model=None
    role='development_smoke_A2'
    saved_hash=None;training_rows=[];checkpoint=None;initialization=None
    reference=None
    try:
        engine=_new_engine(protocol,'A2')
        reference=weakref.ref(engine.model)
        counter.attach(engine.optimizer,role)
        initialization=engine.base_load_audit
        before=tensor_state_hashes(engine.model)
        frozen=_frozen_signature(engine.model)
        for ordinal in range(3):
            item=protocol['inputs'][ordinal]
            need(item['record_id']==reader.ids[ordinal] and engine.data_index(len(reader))==ordinal,
                 'Development must use the first3 fixed training clips in order')
            torch.cuda.synchronize()
            started=time.perf_counter()
            result=engine.step(pairs[ordinal]['clean'].to(protocol['device']),pairs[ordinal]['low'].to(protocol['device']),text,scheduler)
            torch.cuda.synchronize()
            counter.by_role[role]['returned_steps']+=1
            row=result|{'record_id':item['record_id'],'data_index':ordinal,'seconds':time.perf_counter()-started,
                        'feature_diagnostics_CPU_outside_step_timing':engine.drain_feature_diagnostics(),
                        'memory':_memory(torch,protocol['memory_cap_bytes'])}
            need(row['timestep_index']==protocol['smoke']['timestep_indices'][ordinal]
                 and row['sigma']==protocol['smoke']['sigmas'][ordinal]
                 and row['teacher_forward']==protocol['smoke']['teacher_forward'][ordinal],
                 'Development native RNG/window differs from the frozen first3 draws')
            need(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('fm','sd','total','grad_norm_before_clip'))
                 and frozen==_frozen_signature(engine.model),'Nonfinite development update or frozen backbone changed')
            training_rows.append(row);_append_row(run/'training.jsonl',row)
            save(run/'progress.json',{'mode':'smoke','phase':'diagnostic_update','completed_diagnostic_updates':ordinal+1,
                                      **_counts(counter,'smoke')},replace=True)
        need(engine.steps==engine.ema.num_updates==3 and sum(row['teacher_forward'] for row in training_rows)==2,
             'Incomplete3-step EMA/teacher development budget')
        saved_hash=tensor_state_hashes(engine.model)
        need(saved_hash['all_trainable']['sha256']!=before['all_trainable']['sha256'],'No student parameter update occurred')
        destination=run/'A2/checkpoint_000003.pt'
        checkpoint=save_method_checkpoint(destination,engine,config,allowed_root=run.parent,initialization=initialization)
        save(run/'initialization.json',initialization)
        save(run/'student_before_reload.json',{'before':before,'after3':saved_hash,'frozen_unchanged':True})
    finally:
        primary=sys.exc_info()[1]
        def release_engine():
            nonlocal engine
            engine=None
            _check_release(torch,reference,primary is None,counter,role)
        _cleanup_preserving(primary,[lambda:_cleanup_engine(torch,engine,successful=primary is None,counter=counter),release_engine])
    need(counter.snapshot()['diagnostic_optimizer_updates_applied']==3,'Exactly3 diagnostic AdamW calls required')
    reload_audit=None;sampling=None;pngs=None;video=None
    try:
        load_started=time.perf_counter()
        model,reload_audit=load_training_checkpoint(checkpoint['path'],device=protocol['device'],
            allowed_root=run.parent,variant='A2',model_directory=protocol['model_directory'])
        need(reload_audit['step']==3 and reload_audit['variant']=='A2'
             and reload_audit['checkpoint_sha256']==checkpoint['sha256']
             and reload_audit['training_only_state_restored'] is False and reload_audit['EMA_weights_selected'] is False,
             'Development reload must restore the student only')
        need(tensor_state_hashes(model)==saved_hash,'Saved and reloaded student parameter bytes differ')
        save(run/'reload_audit.json',reload_audit)
        model.eval()
        rgb_spec=protocol['smoke']['input_RGB']
        need(binding(rgb_spec['path'])==rgb_spec,'Original sealed LR input changed')
        rgb=torch.load(rgb_spec['path'],map_location='cpu',weights_only=True)
        low=rgb['lr'];need(list(low.shape)==protocol['smoke']['shape_LR'],'Development LRV geometry changed')
        metadata=reader.record(0);fps=metadata['output_fps']
        runtime_config=copy.deepcopy(config);runtime_config['model']['local_dir']=protocol['model_directory']
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
        with _crop_policy(vae) as policy:
            video,sampling=restore_video(model,vae,low,text,runtime_config,steps=20,seed=42,tile_hw=(256,448),overlap=64)
            _validate_restore(video,sampling,model,42)
            ledger=copy.deepcopy(policy.ledger)
        policy=None
        torch.cuda.synchronize();restore_seconds=time.perf_counter()-started
        measured=quantize_rgb(video)
        need(float(measured.std())>0,'Development restore is a constant image')
        pngs=_save_pngs(measured,run/'restoration/png')
        need(len(pngs)==33,'Development restore must save all33 PNGs')
        for index,entry in enumerate(pngs):
            actual=np.asarray(Image.open(entry['path']).convert('RGB')).copy()
            expected=torch.round(measured[0,:,index].permute(1,2,0)*255).byte().numpy()
            need(np.array_equal(actual,expected),'Saved PNG pixels do not reproduce the measured restore')
        torch.save(measured,run/'restoration/student_RGB.pt')
        preview=run/'restoration/student_preview.mp4';write_video(preview,measured,fps)
        container=verify_output_video(preview,frames=33,height=128,width=224,fps=fps)
        need(tensor_state_hashes(model)==saved_hash and all(p.grad is None for p in model.parameters()),
             'Inference changed the student or retained gradients')
        restore={'restores':1,'steps':20,'seed':42,'seconds':restore_seconds,'reload_and_restore_seconds':time.perf_counter()-load_started,
                 'output_shape':list(measured.shape),'finite':True,'nonconstant':True,'PNG_count':33,
                 'PNG_readback_exact':True,'PNG_artifacts':pngs,'preview':binding(preview),'preview_container':container,
                 'RGB_tensor':binding(run/'restoration/student_RGB.pt'),'sampling':sampling,'VAE_ledger':ledger,
                 'memory':_memory(torch,protocol['memory_cap_bytes']),
                 'input_record_id':reader.ids[0],'input_split':'train','quality_conclusion':False,
                 'teacher_used_for_inference':False,'HR_used_for_inference':False}
        save(run/'restoration.json',restore)
    finally:
        primary=sys.exc_info()[1]
        def release_restore():
            nonlocal model,video,measured,rgb,low
            model=video=measured=rgb=low=None
        _cleanup_preserving(primary,[lambda:model.clear_features() if model is not None else None,
                                     release_restore,gc.collect,torch.cuda.empty_cache])
    need(counter.snapshot()['diagnostic_optimizer_updates_attempted']==3
         and counter.snapshot()['completed_step_returns']==3,'Development update accounting differs')
    return {'status':'completed','mode':'smoke','development_only':True,'budget':dict(SMOKE_BUDGET),
            'training_rows':training_rows,'checkpoint':checkpoint,'student_exact_after_reload':True,
            'reload_audit':reload_audit,'restoration':restore,'formal_checkpoint_reuse_allowed':False,
            'G2_G3_G4_decisions_added':False,**counter.snapshot()}


def _counts(counter,mode):
    value=counter.snapshot()
    if mode not in {'resource','smoke'}:
        value['scientific_training_updates']=value.pop('diagnostic_optimizer_updates_applied')
        value['scientific_optimizer_updates_attempted']=value.pop('diagnostic_optimizer_updates_attempted')
    return value


def _append_row(path,row):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()


def _run_arm(protocol,run,arm,reader,text,scheduler,counter):
    from .checkpoints import save_method_checkpoint,save_base_checkpoint
    from .model import make_model,tensor_state_hashes
    mode=protocol['mode']
    engine=model=None
    successful=False
    try:
        if torch.device(protocol['device']).type=='cuda':torch.cuda.reset_peak_memory_stats()
        if mode=='base':
            config=_canonical_config(protocol['config'])
            runtime_config=copy.deepcopy(config)
            runtime_config['model']['local_dir']=protocol['model_directory']
            model,init=make_model(runtime_config,'Base',device=protocol['device'])
            options=protocol['optimizer']
            engine=NoRefContinuationStep(model,arm='flow_only',training_seed=config['training'].get('seed',config['seed']),**options)
            config_run=run
            updates=8192
        else:
            engine=_new_engine(protocol,arm)
            config=_method_config(protocol['matched_Base']['config'],arm)
            init=engine.base_load_audit
            config_run=run/'arm_runs'/arm
            config_run.mkdir(parents=True)
            updates=2048
        save(config_run/'config.json',config)
        save(config_run/'initialization.json',init)
        counter.attach(engine.optimizer,arm)
        first=tensor_state_hashes(engine.model)
        checkpoints=[]
        for i in range(updates):
            index=engine.data_index(len(reader))
            pair=reader[index]
            if torch.device(protocol['device']).type=='cuda':torch.cuda.synchronize()
            started=time.perf_counter()
            result=engine.step(pair['clean'].to(protocol['device']),pair['low'].to(protocol['device']),
                               reader.text_for(index).to(protocol['device']),scheduler)
            if torch.device(protocol['device']).type=='cuda':torch.cuda.synchronize()
            row={**result,'record_id':reader.ids[index],'data_index':index,'seconds':time.perf_counter()-started}
            counter.by_role[arm]['returned_steps']+=1
            row['feature_diagnostics_CPU_outside_step_timing']=engine.drain_feature_diagnostics()
            _append_row(config_run/'training.jsonl',row)
            save(run/'progress.json',{'mode':mode,'arm':arm,'returned_steps_in_arm':engine.steps,**_counts(counter,mode)},replace=True)
            if engine.steps%512==0 or engine.steps==updates:
                destination=config_run/arm/f'checkpoint_{engine.steps:06d}.pt'
                if mode=='base':
                    record=save_base_checkpoint(destination,engine.model,config,step=engine.steps,allowed_root=run.parent,initialization=init)
                else:
                    record=save_method_checkpoint(destination,engine,config,allowed_root=run.parent,initialization=init)
                checkpoints.append(binding(destination))
        need(engine.steps==updates,'Incomplete scientific training arm')
        result={'arm':arm,'updates':engine.steps,'checkpoint_files':checkpoints,'initialization':init,
                'trainable_hashes_before':first,'trainable_hashes_after':tensor_state_hashes(engine.model),
                'full_training_log':binding(config_run/'training.jsonl'),'memory':_memory(torch,protocol['memory_cap_bytes']),
                'optimizer_state_fresh_at_initialization':True,'optimizer_foreach':False,'diagnostic_updates':0}
        save(config_run/'training_complete.json',result)
        successful=True
        return result
    finally:
        if engine is not None:
            try:engine.optimizer.zero_grad(set_to_none=True);engine.model.clear_features()
            except BaseException as error:
                if successful:raise
                counter.cleanup_events.append({'cleanup_error':f'{type(error).__name__}: {error}',
                                               'original_exception_preserved':True})
        engine=model=None
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()




def _configure_runtime(config):
    """Configure/verify strict math before device context, H2D or model load."""
    from .model import configure_reproducibility
    need(not torch.cuda.is_initialized(),'Use a fresh process; CUDA was initialized before strict setup')
    result=configure_reproducibility(config)
    flags=result['flags']
    enabled=('deterministic_algorithms','cudnn_deterministic','sdp_math')
    disabled=('deterministic_warn_only','cudnn_benchmark','cudnn_allow_tf32','matmul_allow_tf32',
              'matmul_allow_fp16_reduced_precision_reduction','matmul_allow_bf16_reduced_precision_reduction',
              'sdp_flash','sdp_memory_efficient','sdp_cudnn','sdp_math_allow_fp16_bf16_reduction')
    need(result['mode']=='strict' and all(flags.get(key) is True for key in enabled)
         and all(flags.get(key) is False for key in disabled)
         and flags.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and flags.get('cuda_initialized') is False,
         'Strict deterministic math/TF32 flags differ before CUDA initialization')
    return result

def _training_scheduler(model_directory,device):
    """Construct the original explicitly configured1000-step training grid."""
    from diffusers import UniPCMultistepScheduler
    scheduler=UniPCMultistepScheduler.from_pretrained(model_directory,subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(1000,device=device)
    need(len(scheduler.timesteps)==1000 and len(scheduler.sigmas)>=1000,'Original1000-index scheduler required')
    return scheduler

def execute(args):
    """Explicit CUDA-only execution; no fallback, retry or resume is available."""
    from .checkpoints import gpu_lease
    run,root=_safe_run(args.run_dir,args.run_root)
    need(args.expected_protocol_sha and file_sha(run/'protocol.json')==args.expected_protocol_sha,
         'Exact prepared protocol SHA required')
    state=read(run/'status.json')
    need(state['status']=='planned' and state['protocol_sha256']==args.expected_protocol_sha,
         'Only a never-started planned run may execute')
    protocol=read(run/'protocol.json')
    reconstructed=argparse.Namespace(**protocol['prepared_arguments'])
    need(collect_protocol(reconstructed)==protocol,'Inputs, sources, environment or scientific prerequisites changed')
    need(protocol.get('strict_environment')==STRICT_ENVIRONMENT
         and all(os.environ.get(key)==value for key,value in STRICT_ENVIRONMENT.items()),
         'Strict/offline launch environment missing; reject before GPU admission')
    with (run/'execute.lock').open('x') as stream:stream.write(str(os.getpid())+'\n')
    mode=protocol['mode']
    counter=UpdateCounter();rows=[];roles={};semantics=None
    reader=pairs=text=scheduler=vae=None
    save(run/'status.json',{'status':'running','protocol_sha256':args.expected_protocol_sha,**_counts(counter,mode)},replace=True)
    def cleanup():
        nonlocal reader,pairs,text,scheduler,vae
        reader=pairs=text=scheduler=vae=None
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    def failure(error):
        record={'status':'failed_stop_no_retry','protocol_sha256':args.expected_protocol_sha,
                'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc(),
                'automatic_retry':False,'partial_benchmark_rows':rows,'completed_roles':list(roles),**_counts(counter,mode)}
        if not (run/'failure.json').exists():save(run/'failure.json',record)
        save(run/'status.json',{'status':'failed_stop_no_retry','protocol_sha256':args.expected_protocol_sha,
             'failure_sha256':file_sha(run/'failure.json'),'G3_passed':False,**_counts(counter,mode)},replace=True)
    try:
        with gpu_lease(protocol['gpu_lock'],protocol['device'],protocol['admission']) as admission:
            try:
                save(run/'GPU_admission.json',admission)
                runtime=_configure_runtime(protocol['config'])
                save(run/'runtime.json',runtime)
                from .vae import load_frozen_vae
                with torch.cuda.device(protocol['device']):
                    need(torch.cuda.is_available(),'The declared GPU is unavailable; no CPU fallback')
                    reader,pairs,text=_load_inputs(protocol)
                    model_dir=protocol['model_directory']
                    scheduler=_training_scheduler(model_dir,protocol['device'])
                    vae=load_frozen_vae(model_dir,device=protocol['device'])
                    if mode=='smoke':
                        summary=_run_development_smoke(protocol,run,reader,pairs,text,scheduler,vae,counter)
                        summary['protocol_sha256']=args.expected_protocol_sha
                    elif mode=='resource':
                        semantics=run_semantics(torch,protocol,pairs,text,scheduler,counter)
                        for role in ROLES:
                            roles[role]=run_benchmark_role(torch,protocol,role,pairs,text,scheduler,counter,rows,run)
                        gate=summarize_roles(rows)
                        save(run/'benchmark_rows.json',rows)
                        summary={'status':'completed','mode':mode,'protocol_sha256':args.expected_protocol_sha,
                            'semantics':semantics,'roles':roles,'resource_gate':gate,
                            'G3_passed':gate['status']=='passed_G3_resource_cost',
                            'benchmark_rows_sha256':file_sha(run/'benchmark_rows.json'),**counter.snapshot()}
                        need(summary['diagnostic_optimizer_updates_applied']==70,'Require exactly70 diagnostic updates')
                    else:
                        for arm in (('Base',) if mode=='base' else ROLES):
                            roles[arm]=_run_arm(protocol,run,arm,reader,text,scheduler,counter)
                        summary={'status':'completed','mode':mode,'protocol_sha256':args.expected_protocol_sha,
                                 'roles':roles,'G4_evaluated':False,**_counts(counter,mode)}
                        need(summary['scientific_training_updates']==protocol['budget']['scientific_updates'],
                             'Scientific update budget is incomplete')
                    cleanup()
                    save(run/'summary.json',summary)
                    save(run/'status.json',{'status':'completed','protocol_sha256':args.expected_protocol_sha,
                         'summary_sha256':file_sha(run/'summary.json'),'G3_passed':summary.get('G3_passed',False),
                         **_counts(counter,mode)},replace=True)
                    if mode=='resource' and not summary['G3_passed']:
                        raise RuntimeError('G3 performance gate failed; stop without retry or protocol changes')
                    return summary
            except BaseException as error:
                failure(error)
                try:cleanup()
                except BaseException as secondary:
                    save(run/'cleanup_failure.json',{'primary':str(error),'secondary':str(secondary),'primary_error_preserved':True})
                raise
    except BaseException as error:
        if not (run/'failure.json').exists():failure(error)
        raise



def _official_bindings(directory):
    directory=Path(directory).resolve()
    result={}
    for component in ('transformer','vae','scheduler'):
        base=directory/component
        need(base.is_dir(),'Official model component missing: '+str(base))
        configuration=base/('scheduler_config.json' if component=='scheduler' else 'config.json')
        need(configuration.is_file(),'Official component configuration missing: '+str(configuration))
        paths=sorted(p for p in base.rglob('*') if p.is_file())
        need(paths,'Official model component is empty: '+str(base))
        for path in paths:
            result[str(path.relative_to(directory))]=binding(path)
    need(any(name.endswith('.safetensors') for name in result if name.startswith('transformer/'))
         and any(name.endswith('.safetensors') for name in result if name.startswith('vae/')),
         'Official transformer/VAE safetensors are required')
    return result


def _dependency_bindings():
    from importlib import metadata,util
    result={}
    for name in ('torch','diffusers','transformers','peft','numpy','safetensors'):
        distribution=metadata.distribution(name)
        entry=util.find_spec(name)
        files={}
        if entry is not None and entry.origin and Path(entry.origin).is_file():
            files['package_entry']=binding(entry.origin)
        for item in distribution.files or []:
            if str(item).endswith(('.dist-info/METADATA','.dist-info/WHEEL','.dist-info/RECORD')):
                files[str(item)]=binding(distribution.locate_file(item))
        result[name]={'version':distribution.version,'files':files}
    import diffusers
    base=Path(diffusers.__file__).parent
    result['diffusers_native_sources']={}
    for name in ('models/transformers/transformer_wan.py','models/autoencoders/autoencoder_kl_wan.py',
                 'schedulers/scheduling_unipc_multistep.py'):
        result['diffusers_native_sources'][name]=binding(base/name)
    return result

def build_parser():
    parser=argparse.ArgumentParser(description='Unified NoRef training: prepare metadata, then explicit fresh CUDA execution.')
    sub=parser.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('prepare',help='Bind metadata and prerequisites; never load a GPU model')
    plan.add_argument('--mode',choices=['base','resource','pilot','smoke'],required=True)
    plan.add_argument('--model-directory',type=Path,help='Explicit migrated official model path; checkpoint config is not rewritten')
    plan.add_argument('--config',type=Path,required=True)
    plan.add_argument('--cache-root',type=Path,required=True)
    plan.add_argument('--base-checkpoint',type=Path)
    plan.add_argument('--expected-base-sha')
    plan.add_argument('--allowed-checkpoint-root',type=Path,required=True)
    plan.add_argument('--g2-run',type=Path)
    plan.add_argument('--g3-run',type=Path)
    plan.add_argument('--g4-definitions',type=Path)
    plan.add_argument('--device',required=True,help='Explicit cuda:N device; only execute initializes it')
    plan.add_argument('--gpu-admission',type=Path,required=True)
    plan.add_argument('--gpu-lock',type=Path,required=True)
    plan.add_argument('--memory-cap-gib',type=float,required=True)
    for command in [plan,sub.add_parser('execute',help='Execute a never-started sealed CUDA run')]:
        command.add_argument('--run-dir',type=Path,required=True)
        command.add_argument('--run-root',type=Path,required=True)
    sub.choices['execute'].add_argument('--expected-protocol-sha',required=True)
    for name in ('hr-prepare','hr-probe','hr-execute'):
        command=sub.add_parser(name,help='Independent paired HR VAE latent supervision study')
        command.add_argument('--run-dir',type=Path,required=True)
        if name=='hr-prepare':command.add_argument('--config',type=Path,required=True)
        else:command.add_argument('--expected-protocol-sha',required=True)
        if name=='hr-execute':command.add_argument('--resume',action='store_true')
    for name in ('ft-calibration-prepare','ft-calibrate','ft-prepare','ft-execute'):
        command=sub.add_parser(name,help='First-round fixed HR-VAE and HR-DINO supervision')
        command.add_argument('--run-dir',type=Path,required=True)
        if name in ('ft-prepare','ft-calibration-prepare'):command.add_argument('--config',type=Path,required=True)
        else:command.add_argument('--expected-protocol-sha',required=True)
        if name=='ft-execute':command.add_argument('--resume',action='store_true')
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.command=='ft-calibration-prepare':result=prepare_ft_calibration(args.config,args.run_dir)
    elif args.command=='ft-calibrate':result=execute_ft_calibration(args.run_dir,args.expected_protocol_sha)
    elif args.command=='ft-prepare':result=prepare_hr(args.config,args.run_dir)
    elif args.command=='ft-execute':result=execute_hr(args.run_dir,args.expected_protocol_sha,resume=args.resume)
    elif args.command=='hr-prepare':result=prepare_hr(args.config,args.run_dir)
    elif args.command=='hr-probe':result=probe_hr(args.run_dir,args.expected_protocol_sha)
    elif args.command=='hr-execute':result=execute_hr(args.run_dir,args.expected_protocol_sha,resume=args.resume)
    else:result=prepare(args) if args.command=='prepare' else execute(args)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result



# HR VAE study: an independent paired continuation, with no EMA or corruption.
class HRLatentStep(NoRefContinuationStep):
    def __init__(self, model, *, weight, training_seed=20261102, student_layer=7, teacher_kind='vae', **options):
        need(teacher_kind in ('vae','dino'),'HR teacher kind must be vae or dino')
        need(type(weight) in (int,float) and math.isfinite(weight) and weight>=0,'Nonnegative finite HR alignment weight required')
        super().__init__(model,arm='flow_only',training_seed=training_seed,**options)
        need(type(student_layer) is int and 0<=student_layer<len(model.transformer.blocks),'Invalid HR student block')
        self.weight,self.student_layer,self.teacher_kind=float(weight),student_layer,teacher_kind
        if self.weight>0:
            from .losses import HRLatentRegression,HRDINOCosineRegression
            dim=model.transformer.config.num_attention_heads*model.transformer.config.attention_head_dim
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator(device='cpu').manual_seed(training_seed+5).get_state())
                alignment=(HRLatentRegression(dim,latent_channels=model.transformer.config.in_channels,
                    patch_size=tuple(model.transformer.config.patch_size)) if teacher_kind=='vae'
                    else HRDINOCosineRegression(dim,teacher_dim=768))
                self.alignment=alignment.to(next(model.parameters()).device)
            named=[('restoration.'+name,p) for name,p in model.named_parameters() if p.requires_grad]
            named.extend(('student_projector.'+name,p) for name,p in self.alignment.projector.named_parameters())
            self._named_optimizer_parameters=tuple(named);self.parameters=[p for _,p in named]
            self.optimizer=torch.optim.AdamW(self.parameters,lr=options.get('learning_rate',1e-4),
                weight_decay=options.get('weight_decay',.01),foreach=False)
        need(not self.optimizer.state and self.ema is None,'HR study starts fresh and has no EMA')

    def step(self,clean,low,text,scheduler,*,teacher_target=None):
        if self.weight==0:
            value=super().step(clean,low,text,scheduler)
            sigma=value['sigma']
            return value|{'weight':0.,'align':0.,'student_layer':self.student_layer,'HR_target_detached':True,
                'teacher_kind':self.teacher_kind,'noise_bin':'low' if sigma<.33 else 'mid' if sigma<.67 else 'high'}
        from .teacher import capture_block_output
        need(torch.is_grad_enabled() and clean.shape==low.shape and clean.ndim==5 and clean.shape[2]==9
             and clean.dtype==low.dtype==torch.float32 and clean.device==low.device
             and not clean.requires_grad and not low.requires_grad
             and bool(torch.isfinite(clean).all()) and bool(torch.isfinite(low).all()),'Frozen finite matched nine-position latents required')
        if self.teacher_kind=='dino':
            patch=tuple(self.model.transformer.config.patch_size)
            need(all(size%stride==0 for size,stride in zip(clean.shape[2:],patch)), 'DINO target token grid requires divisible latent geometry')
            grid=tuple(size//stride for size,stride in zip(clean.shape[2:],patch))
            need(isinstance(teacher_target,torch.Tensor) and tuple(teacher_target.shape)==(clean.shape[0],math.prod(grid),768)
                 and teacher_target.dtype==torch.float32 and teacher_target.device==clean.device
                 and not teacher_target.requires_grad and bool(torch.isfinite(teacher_target).all()),
                 'Frozen finite FP32 DINO target must match B,T*H*W,768')
            alignment_target=teacher_target
        else:
            need(teacher_target is None,'VAE alignment uses the sealed clean HR latent directly')
            alignment_target=clean
        modes=_modes(self.model);head_modes=_modes(self.alignment);frozen=_frozen(self.model)
        self.optimizer.zero_grad(set_to_none=True);self.model.clear_features()
        try:
            self.model.train();self.alignment.train()
            noisy,target,timestep,sigma,index=self.rng.flow_inputs(clean,scheduler)
            flow,features=capture_block_output(self.model,self.student_layer,noisy,timestep,text,low)
            fm=flow_matching_loss(flow,target)
            align,diagnostic=self.alignment(features,alignment_target,tuple(self.model.token_grid))
            total=fm+self.weight*align
            need(bool(torch.isfinite(total)),'Nonfinite HR objective')
            total.backward()
            need(any(p.grad is not None for p in self.model.parameters() if p.requires_grad),'No student gradient')
            need(all(p.grad is None for p in self.model.parameters() if not p.requires_grad),'Frozen backbone gradient')
            _require_finite(('Nonfinite HR gradient: '+name,p.grad) for name,p in self._named_optimizer_parameters if p.grad is not None)
            norm=torch.nn.utils.clip_grad_norm_(self.parameters,self.max_grad_norm,error_if_nonfinite=True)
            self.optimizer.step();self.steps+=1
            need(frozen==_frozen(self.model),'Frozen backbone changed')
            return {'step':self.steps,'arm':'HR_DINO' if self.teacher_kind=='dino' else 'HR_VAE',
                'teacher_kind':self.teacher_kind,'weight':self.weight,'fm':float(fm.detach()),
                'align':float(align.detach()),'total':float(total.detach()),'grad_norm_before_clip':float(norm),
                'sigma':float(sigma),'timestep_index':index,'native_timestep':float(timestep[0]),
                'noise_bin':'low' if bool(sigma<.33) else 'mid' if bool(sigma<.67) else 'high',
                'student_layer':self.student_layer,'teacher_forward':False,'EMA_updates':0,
                'HR_target_detached':True,'alignment':diagnostic}
        finally:
            self.optimizer.zero_grad(set_to_none=True);self.model.clear_features()
            _restore_modes(modes);_restore_modes(head_modes)


def collect_hr_protocol(config_path):
    study=read(config_path)
    if study.get('schema')=='fixed_teacher_study_v1':return collect_fixed_teacher_protocol(config_path)
    need(study.get('schema')=='hr_vae_study_v1','Wrong HR study config')
    need(study['updates']==2048 and study['eval_updates']==[512,1024,2048]
         and study['checkpoint_updates']==[512,1024,1536,2048],'Keep the prospective small training budget')
    need(study['arms']==[{'label':'baseline','weight':0.0},{'label':'hr001','weight':.01},{'label':'hr01','weight':.1}],
         'Keep baseline and the two prospectively fixed weights')
    need(study['student_layer']==7 and study['smooth_l1_beta']==1. and study['training_seed']==20261102,
         'Keep block7, elementmean SmoothL1 beta1 and the single matched seed')
    config=read(study['base_config'])
    base=_base_spec(study['base_checkpoint'],SMOKE_BASE_SHA,study['allowed_base_root'],20261102)
    need(config==base['config'],'Use the unmodified Base source config')
    need(config.get('reproducibility',{}).get('mode')=='strict','Original strict math required')
    cache,inputs=_cache_spec(study['cache_root'])
    need(cache['config']==base['config_file'],'Use the complete original producer cache')
    model_dir=study['model_directory'];native=read(Path(model_dir)/'transformer/config.json')
    need(native['num_layers']==30 and native['in_channels']==16 and native['patch_size']==[1,2,2],
         'Expected original Wan thirty blocks and native latent patches')
    return {'schema':'hr_vae_training_protocol_v1','study':study,'study_file':binding(config_path),
        'sources':_source_bindings(),'base':base,'cache':cache,'inputs':inputs,
        'official_model_files':_official_bindings(model_dir),'dependencies':_dependency_bindings(),
        'admission':read(study['gpu_admission']),'admission_file':binding(study['gpu_admission']),
        'gpu_lock':study['gpu_lock'],'device':study['device'],'memory_cap_bytes':int(16*1024**3),
        'optimizer':{k:config['training'][k] for k in ('learning_rate','weight_decay','max_grad_norm')},
        'strict_environment':STRICT_ENVIRONMENT,'scientific_update_budget':6144,
        'online_teacher_forward':False,'target':'original frozen normalized posterior-mode clean HR latent',
        'inference_uses':'student only; no head or HR/teacher inputs','resume_policy':'same-protocol checkpoint, preserve attempt records'}


def prepare_hr(config_path,run_dir):
    protocol=collect_hr_protocol(config_path)
    run,root=_safe_run(Path(run_dir),Path(protocol['study']['run_root']))
    need(not run.exists(),'New HR study output directory required')
    run.mkdir(parents=True);save(run/'protocol.json',protocol)
    result={'status':'planned','protocol_sha256':file_sha(run/'protocol.json'),'scientific_updates':0,'GPU_loaded':False}
    save(run/'status.json',result)
    return result|{'run_dir':str(run)}


def _hr_protocol(run_dir,expected):
    run=Path(run_dir).resolve();protocol=read(run/'protocol.json')
    run,_=_safe_run(run,Path(protocol['study']['run_root']))
    need(file_sha(run/'protocol.json')==expected,'Exact HR protocol SHA required')
    need(protocol==collect_hr_protocol(protocol['study_file']['path']),'HR source/config/assets changed')
    need(all(os.environ.get(k)==v for k,v in STRICT_ENVIRONMENT.items()),'Strict/offline environment required')
    return run,protocol


def _hr_engine(protocol,weight,kind='vae'):
    from .checkpoints import load_training_checkpoint
    need(kind in ('vae','dino'),'HR teacher kind must be vae or dino')
    s=protocol['study'];b=protocol['base']['checkpoint']
    model,audit=load_training_checkpoint(b['path'],device=s['device'],allowed_root=s['allowed_base_root'],
                                        variant='Base',model_directory=s['model_directory'])
    need(audit['checkpoint_sha256']==b['sha256'] and audit['step']==8192
         and audit['config']==protocol['base']['config'],'Matched Base differs')
    kind_options={} if kind=='vae' else {'teacher_kind':kind}
    engine=HRLatentStep(model,weight=weight,training_seed=s['training_seed'],
                        student_layer=s['student_layer'],**kind_options,**protocol['optimizer'])
    engine.base_load_audit=audit
    return engine


def _hr_config(protocol,weight,kind='vae'):
    need(kind in ('vae','dino'),'HR teacher kind must be vae or dino')
    config=copy.deepcopy(protocol['base']['config'])
    config.pop('restoration_architecture',None)
    config['stage']='hr_vae_latent_single_seed' if kind=='vae' else 'hr_dino_single_seed'
    config['training'].update(variants=['Base'],updates=protocol['study']['updates'])
    if kind=='vae':
        config['auxiliary_supervision']={'target':'clean normalized HR VAE latent','weight':float(weight),
            'student_layer':7,'head':'Linear1536to64_FP32','loss':'SmoothL1','beta':1.,'reduction':'elementmean',
            'teacher_frozen':True,'EMA':False,'condition_corruption':False}
    else:
        config['auxiliary_supervision']={'target':'clean HR DINOv2-B/14 temporally pooled patch tokens',
            'family':'dino','teacher_kind':'dino','weight':float(weight),'student_layer':7,
            'head':'Linear1536to768_FP32','loss':'one_minus_cosine','normalization':'channel_L2',
            'eps':1e-12,'reduction':'tokenmean','teacher_frozen':True,'EMA':False,'condition_corruption':False}
    return _canonical_config(config)


def _hr_step(engine,pair,text,scheduler,device,*,teacher_target=None):
    start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();started=time.perf_counter();start_event.record()
    if teacher_target is None:
        value=engine.step(pair['clean'].to(device),pair['low'].to(device),text.to(device),scheduler)
    else:
        value=engine.step(pair['clean'].to(device),pair['low'].to(device),text.to(device),scheduler,
                          teacher_target=teacher_target.to(device))
    end_event.record();torch.cuda.synchronize()
    return value,time.perf_counter()-started,start_event.elapsed_time(end_event)/1000


def _hr_release(engine):
    if engine is not None:
        engine.optimizer.zero_grad(set_to_none=True);engine.model.clear_features()
    gc.collect()
    if torch.cuda.is_initialized():torch.cuda.empty_cache()


@contextmanager
def _hr_gpu_session(protocol,cleanup,on_error):
    from .checkpoints import gpu_lease
    with gpu_lease(protocol['gpu_lock'],protocol['device'],protocol['admission']) as admission:
        primary=None
        try:
            yield admission
        except BaseException as error:
            primary=error
            try:on_error(error)
            except BaseException as secondary:error.add_note('Failure record error: '+repr(secondary))
            raise
        finally:
            _hr_cleanup_preserving(cleanup,primary)


def _hr_cleanup_preserving(cleanup,primary):
    try:cleanup()
    except BaseException as secondary:
        if primary is None:raise
        primary.add_note('Cleanup error: '+repr(secondary))


def probe_hr(run_dir,expected):
    from .checkpoints import gpu_lease,save_hr_latent_checkpoint,restore_hr_latent_training_state,load_training_checkpoint
    from .data import PreparedTrainingData
    from .vae import load_frozen_vae,encode_video
    from .model import tensor_state_hashes
    run,p=_hr_protocol(run_dir,expected);s=p['study'];out=run/'probe'
    need(not out.exists(),'Probe is single attempt; preserve prior diagnostic artifacts')
    out.mkdir();engine=vae=reader=direct=reloaded=scheduler=None;rows=[];initials=[];updates=0
    def cleanup():
        nonlocal engine,vae,reader,direct,reloaded,scheduler
        _hr_release(engine);_hr_release(direct)
        if reader is not None:reader.close()
        engine=vae=reader=direct=reloaded=scheduler=None
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    def record_failure(error):
        if not (out/'failure.json').exists():
            save(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc(),'diagnostic_updates_returned':updates})
    try:
        with _hr_gpu_session(p,cleanup,record_failure) as admission:
            save(out/'GPU_admission.json',admission);runtime=_configure_runtime(p['base']['config']);save(out/'runtime.json',runtime)
            reader=PreparedTrainingData(p['cache']['root'],expected_seal_sha256=p['cache']['seal']['sha256'])
            vae=load_frozen_vae(s['model_directory'],device=s['device'])
            frozen_vae=_frozen(vae);encodings=[]
            for index in (0,1641):
                rgb_path=Path(p['cache']['root'])/f'clip_{index}'/'paired_rgb.pt'
                seal=read(Path(p['cache']['root'])/'cache_asset_seal.json')['artifact_sha256']
                binding(rgb_path,seal[f'clip_{index}/paired_rgb.pt'])
                rgb=torch.load(rgb_path,map_location='cpu',weights_only=True)
                latent_path=Path(p['cache']['root'])/f'clip_{index}'/'fresh_latents.pt'
                binding(latent_path,seal[f'clip_{index}/fresh_latents.pt'])
                latent=torch.load(latent_path,map_location='cpu',weights_only=True)
                encoded=encode_video(vae,rgb['hr'])
                need(torch.equal(encoded,latent['clean']),'Fresh HR VAE encoding differs from cached normalized target')
                encodings.append({'cache_index':index,'exact_equal':True,'shape':list(encoded.shape)})
            need(_frozen(vae)==frozen_vae and all(not x.requires_grad and x.grad is None for x in vae.parameters()),'Teacher VAE must remain frozen')
            scheduler=_training_scheduler(s['model_directory'],s['device']);pairs=[reader[i] for i in range(8)];text=reader.text_for(0)
            baseline_trajectory=[];resource={}
            for arm in s['arms']:
                engine=_hr_engine(p,arm['weight']);initials.append(tensor_state_hashes(engine.model))
                torch.cuda.reset_peak_memory_stats();role_rows=[]
                for i in range(8):
                    value,seconds,gpu_seconds=_hr_step(engine,pairs[i],text,scheduler,s['device']);updates+=1
                    row=value|{'role':arm['label'],'seconds':seconds,'gpu_event_seconds':gpu_seconds,'warmup':i<2}
                    rows.append(row);role_rows.append(row)
                    if arm['weight']==0 and i<3:baseline_trajectory.append(tensor_state_hashes(engine.model))
                resource[arm['label']]={'weight':arm['weight'],'all8_seconds':[x['seconds'] for x in role_rows],
                     'timed6_mean_seconds':statistics.mean(x['seconds'] for x in role_rows[2:]),
                     'timed6_sample_SD_seconds':statistics.stdev(x['seconds'] for x in role_rows[2:]),
                     'memory':_memory(torch,p['memory_cap_bytes'])}
                if arm['label']=='hr001':
                    destination=out/'save_reload'/'Base'/'checkpoint_000008.pt'
                    config=_hr_config(p,arm['weight'])
                    save_hr_latent_checkpoint(destination,engine,config,allowed_root=run,
                        protocol_sha256=expected,cumulative_training_seconds=sum(x['seconds'] for x in role_rows),
                        initialization=engine.base_load_audit)
                    expected_student=tensor_state_hashes(engine.model)
                    _hr_release(engine);engine=None;gc.collect();torch.cuda.empty_cache()
                    reloaded,audit=load_training_checkpoint(destination,device=s['device'],allowed_root=run,variant='Base',model_directory=s['model_directory'])
                    need(tensor_state_hashes(reloaded)==expected_student and reloaded.projector is None,'Inference reload changed student or loaded auxiliary head')
                    reloaded=None;gc.collect();torch.cuda.empty_cache()
                    engine=_hr_engine(p,arm['weight'])
                    restore_hr_latent_training_state(engine,destination,allowed_root=run,expected_protocol_sha256=expected,expected_config=config)
                    need(engine.steps==8 and tensor_state_hashes(engine.model)==expected_student,'Training resume failed exact student restore')
                _hr_release(engine);engine=None;gc.collect();torch.cuda.empty_cache()
            need(all(x==initials[0] for x in initials),'Probe arms did not start from the same student')
            direct=_hr_engine(p,0.)
            for i in range(3):
                # Call the original unmodified baseline Step directly.
                NoRefContinuationStep.step(direct,pairs[i]['clean'].to(s['device']),pairs[i]['low'].to(s['device']),text.to(s['device']),scheduler)
                updates+=1;need(tensor_state_hashes(direct.model)==baseline_trajectory[i],'Alignment off does not reproduce the original GPU baseline trajectory')
            _hr_release(direct);direct=None;gc.collect();torch.cuda.empty_cache()
            result={'status':'passed_hr_preflight','protocol_sha256':expected,'diagnostic_optimizer_updates':updates,
                'scientific_updates':0,'fresh_HR_encodings':encodings,'teacher_frozen':True,
                'alignment_off_exact_original_GPU_trajectory_updates':3,'save_inference_reload_training_resume_passed':True,
                'resources':resource,'relative_to_baseline':{k:v['timed6_mean_seconds']/resource['baseline']['timed6_mean_seconds'] for k,v in resource.items()},
                'all_rows':rows,'diagnostic_weights_reused_for_science':False}
            save(out/'summary.json',result)
            return result
    except BaseException as error:
        record_failure(error)
        raise
    finally:
        _hr_cleanup_preserving(cleanup,sys.exc_info()[1])


def execute_hr(run_dir,expected,*,resume=False):
    from .checkpoints import gpu_lease,save_hr_latent_checkpoint,restore_hr_latent_training_state
    from .data import PreparedTrainingData
    from .model import tensor_state_hashes
    from .vae import load_frozen_vae
    run,p=_hr_protocol(run_dir,expected);s=p['study'];state=read(run/'status.json')
    need(state['status']=='planned' or (resume and state['status'] in ('running','failed_recoverable')), 'Explicit checkpoint resume required for an interrupted study')
    fixed=p.get('schema')=='fixed_teacher_training_protocol_v1'
    expected_budget=p['scientific_update_budget'];expected_arms=len(s['arms'])
    need(expected_budget==s['updates']*expected_arms,'Scientific budget differs from the declared arm/update matrix')
    if fixed:
        cal=p['calibration'];cal_protocol=binding(cal['protocol']['path'],cal['protocol']['sha256'])
        cal_summary=binding(cal['summary']['path'],cal['summary']['sha256'])
        need(cal_protocol==cal['protocol'] and cal_summary==cal['summary'],'Fixed teacher preflight binding changed')
        preflight_protocol=read(cal_protocol['path']);probe=read(cal_summary['path'])
        need(probe.get('status')=='passed_fixed_teacher_preflight'
             and probe.get('protocol_sha256')==cal_protocol['sha256']
             and probe.get('sources')==p['sources'] and preflight_protocol.get('sources')==p['sources']
             and probe.get('scientific_updates')==0,'Require the passed same-source fixed-teacher preflight')
    else:
        probe=read(run/'probe/summary.json')
        need(probe['status']=='passed_hr_preflight' and probe['protocol_sha256']==expected,'A passed real preflight is required')
    engine=reader=dino_reader=vae=scheduler=None;completed={};effective=0;attempt_returns=0;started=time.perf_counter()
    attempt=len(list(run.glob('attempt_*.json')))
    save(run/f'attempt_{attempt:03d}.json',{'protocol_sha256':expected,'resume_requested':resume,'pid':os.getpid()})
    def cleanup():
        nonlocal engine,reader,dino_reader,vae,scheduler
        _hr_release(engine)
        if reader is not None:reader.close()
        if dino_reader is not None:dino_reader.close()
        engine=reader=dino_reader=vae=scheduler=None
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    def record_failure(error):
        failure={'status':'failed_recoverable','protocol_sha256':expected,'attempt':attempt,'error':str(error),
            'traceback':traceback.format_exc(),'completed_arms':list(completed),'effective_updates_completed_arms':effective,
            'current_arm_steps_returned':engine.steps if engine is not None else None,
            'wall_seconds_this_attempt':time.perf_counter()-started,'returned_updates_this_attempt':attempt_returns,
            'partial_optimizer_effects_if_unreturned':'unknown; restart only from verified completed checkpoint'}
        if not (run/f'failure_attempt_{attempt:03d}.json').exists():save(run/f'failure_attempt_{attempt:03d}.json',failure)
        save(run/'status.json',failure,replace=True)
    try:
        with _hr_gpu_session(p,cleanup,record_failure) as admission:
            save(run/f'GPU_admission_attempt_{attempt:03d}.json',admission);_configure_runtime(p['base']['config'])
            reader=PreparedTrainingData(p['cache']['root'],expected_seal_sha256=p['cache']['seal']['sha256'])
            if fixed:
                from .data import PreparedDINOData
                dino_reader=PreparedDINOData(p['dino_cache']['root'],p['cache']['root'],
                                           expected_seal_sha256=p['dino_cache']['seal']['sha256'])
                need(list(dino_reader.ids)==list(reader.ids) and len(dino_reader)==len(reader),
                     'DINO and HR training target order differs')
            vae=load_frozen_vae(s['model_directory'],device=s['device'])
            scheduler=_training_scheduler(s['model_directory'],s['device'])
            save(run/'status.json',{'status':'running','protocol_sha256':expected,'scientific_updates':0},replace=True)
            for arm in s['arms']:
                label,weight=arm['label'],arm['weight'];directory=run/'arm_runs'/label
                family,setting=(arm['family'],arm['setting']) if fixed else (None,None)
                dino=fixed and family=='hr_dino'
                directory.mkdir(parents=True,exist_ok=True)
                config=_hr_config(p,weight,kind='dino') if dino else _hr_config(p,weight)
                torch.cuda.reset_peak_memory_stats()
                engine=_hr_engine(p,weight,kind='dino') if dino else _hr_engine(p,weight)
                initial=tensor_state_hashes(engine.model)
                cumulative=0.;cp_records=[];log=directory/'training.jsonl'
                completion=directory/'completion.json'
                if completion.exists():
                    record=read(completion)
                    need(record['updates']==s['updates'] and record['weight']==weight and record['initial_student_hashes']==initial,'Completed arm differs')
                    if fixed:need(record.get('family')==family and record.get('setting')==setting,'Completed arm family/setting differs')
                    for cp in record['checkpoint_files']:binding(cp['path'],cp['sha256'])
                    need(binding(log)==record['full_training_log'],'Completed training log changed')
                    completed[label]=record;effective+=s['updates'];_hr_release(engine);engine=None
                    gc.collect();torch.cuda.empty_cache();continue
                checkpoints=sorted((directory/'Base').glob('checkpoint_*.pt')) if (directory/'Base').exists() else []
                if checkpoints:
                    need(resume,'Existing checkpoints require explicit resume')
                    last=checkpoints[-1]
                    audit=restore_hr_latent_training_state(engine,last,allowed_root=run,expected_protocol_sha256=expected,expected_config=config)
                    cumulative=audit['cumulative_training_seconds']
                    raw_log=log.read_bytes();raw_lines=raw_log.splitlines();lines=[];partial_tail=False
                    for ordinal,line in enumerate(raw_lines):
                        if not line.strip():continue
                        try:lines.append(json.loads(line))
                        except (ValueError,UnicodeDecodeError):
                            need(ordinal>=engine.steps,'Incomplete log inside the checkpoint prefix')
                            partial_tail=True;break
                    prefix=lines[:engine.steps]
                    need([x['step'] for x in prefix]==list(range(1,engine.steps+1)),'Resume log prefix missing')
                    previous=0.
                    for i,row in enumerate(prefix):
                        need(row['arm']==label and row['weight']==weight and row['data_index']==i%len(reader)
                             and row['record_id']==reader.ids[i%len(reader)],'Resume log role/data order differs')
                        if fixed:need(row.get('family')==family and row.get('setting')==setting,'Resume log family/setting differs')
                        need(all(type(row[k]) in (int,float) and math.isfinite(row[k])
                                 for k in ('fm','align','total','grad_norm_before_clip','sigma','seconds','cumulative_training_seconds')),
                             'Nonfinite resume log prefix')
                        need(row['seconds']>=0 and row['cumulative_training_seconds']==previous+row['seconds'],
                             'Resume cumulative time does not match committed step times')
                        previous=row['cumulative_training_seconds']
                    need(previous==cumulative,'Resume checkpoint time differs from committed prefix')
                    save(directory/f'resume_loader_audit_attempt_{attempt:03d}.json',engine.base_load_audit)
                    need(read(directory/'initialization.json')==audit['initialization'],'Original initialization sidecar differs')
                    engine.base_load_audit=audit['initialization']
                    if len(lines)>engine.steps or partial_tail:
                        preserved=directory/f'interrupted_log_attempt_{attempt:03d}.jsonl'
                        need(not preserved.exists(),'Attempt record exists')
                        preserved.write_bytes(raw_log)
                        discarded=lines[engine.steps:]
                        save(directory/f'rollback_attempt_{attempt:03d}.json',{'checkpoint_step':engine.steps,
                            'discarded_completed_step_returns':len(discarded),
                            'known_discarded_step_seconds':sum(float(x['seconds']) for x in discarded),
                            'unparsed_tail_present':partial_tail,'original_log':binding(preserved),
                            'counts_are_wasted_cost_not_effective_training_progress':True})
                        log.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in lines[:engine.steps]))
                    for cp in checkpoints:
                        cp_step=int(cp.stem.split('_')[-1]);row=lines[cp_step-1]
                        cp_records.append(binding(cp)|{'step':cp_step,'update':cp_step,'variant':'Base',
                           'cumulative_training_seconds':row['cumulative_training_seconds']})
                else:
                    need(not log.exists(),'Uncheckpointed attempt needs preserved restart review')
                    save(directory/'initialization.json',engine.base_load_audit)
                save(directory/'config.json',config) if not (directory/'config.json').exists() else need(read(directory/'config.json')==config,'Arm config changed')
                while engine.steps<s['updates']:
                    index=engine.data_index(len(reader));pair=reader[index]
                    if dino:
                        row,seconds,gpu_seconds=_hr_step(engine,pair,reader.text_for(index),scheduler,s['device'],
                                                         teacher_target=dino_reader[index])
                    else:
                        row,seconds,gpu_seconds=_hr_step(engine,pair,reader.text_for(index),scheduler,s['device'])
                    attempt_returns+=1
                    cumulative+=seconds
                    row=row|{'arm':label,'weight':weight,'record_id':reader.ids[index],'data_index':index,
                             'seconds':seconds,'cumulative_training_seconds':cumulative,'gpu_event_seconds':gpu_seconds}
                    if fixed:row.update(family=family,setting=setting)
                    _append_row(log,row)
                    _memory(torch,p['memory_cap_bytes'])
                    save(run/'progress.json',{'status':'running','arm':label,'step':engine.steps,
                         'scientific_updates':effective+engine.steps,'protocol_sha256':expected},replace=True)
                    if engine.steps in s['checkpoint_updates']:
                        destination=directory/'Base'/f'checkpoint_{engine.steps:06d}.pt'
                        record=save_hr_latent_checkpoint(destination,engine,config,allowed_root=run,protocol_sha256=expected,
                            cumulative_training_seconds=cumulative,initialization=engine.base_load_audit)
                        cp_records.append(binding(destination)|{'step':engine.steps,'update':engine.steps,'variant':'Base',
                              'cumulative_training_seconds':cumulative})
                record={'weight':weight,'updates':engine.steps,'checkpoint_files':cp_records,
                    'initialization':engine.base_load_audit,'full_training_log':binding(log),
                    'initial_student_hashes':initial,'final_student_hashes':tensor_state_hashes(engine.model),
                    'cumulative_training_seconds':cumulative,'teacher_is_frozen_cached_HR':True,
                    'head_parameters':sum(x.numel() for x in engine.alignment.parameters()) if engine.alignment is not None else 0,
                    'memory':_memory(torch,p['memory_cap_bytes'])}
                if fixed:record.update(family=family,setting=setting,teacher_kind='dino' if dino else 'vae')
                save(completion,record);completed[label]=record;effective+=engine.steps
                _hr_release(engine);engine=None;gc.collect();torch.cuda.empty_cache()
            need(effective==expected_budget and len(completed)==expected_arms,'Incomplete paired HR study')
            need(len({json.dumps(x['initial_student_hashes'],sort_keys=True) for x in completed.values()})==1,'Unequal student initialization')
            summary={'schema':'fixed_teacher_training_summary_v1' if fixed else 'hr_vae_training_summary_v1','status':'completed','protocol_sha256':expected,
                     'scientific_updates':effective,'arms':completed,'wall_seconds_this_attempt':time.perf_counter()-started,
                     'diagnostic_updates_in_this_training':0,'selection_or_evaluation_done':False}
            rollbacks=[read(path) for path in (run/'arm_runs').glob('*/rollback_attempt_*.json')]
            earlier=[read(path) for path in run.glob('failure_attempt_*.json')]
            summary['compute_cost']={'committed_training_step_seconds':sum(v['cumulative_training_seconds'] for v in completed.values()),
                'known_discarded_completed_step_seconds':sum(v['known_discarded_step_seconds'] for v in rollbacks),
                'known_discarded_completed_step_returns':sum(v['discarded_completed_step_returns'] for v in rollbacks),
                'wall_seconds_all_training_attempts':time.perf_counter()-started+sum(v.get('wall_seconds_this_attempt',0.) for v in earlier),
                'successful_step_returns_this_attempt':attempt_returns,
                'unreturned_failed_step_compute':'not in committed curve; included in attempt wall time, pure GPU duration unknown',
                'convergence_curve_time_basis':'committed synchronized training-step seconds; wasted work reported separately'}
            save(run/f'completion_attempt_{attempt:03d}.json',{'wall_seconds':time.perf_counter()-started,
                  'returned_updates_this_attempt':attempt_returns,'effective_updates_at_completion':effective})
            save(run/'summary.json',summary)
            save(run/'status.json',{'status':'completed','protocol_sha256':expected,'summary_sha256':file_sha(run/'summary.json'),
                                    'scientific_updates':effective},replace=True)
            return summary
    except BaseException as error:
        if not (run/f'failure_attempt_{attempt:03d}.json').exists():record_failure(error)
        raise
    finally:
        _hr_cleanup_preserving(cleanup,sys.exc_info()[1])


# Fixed-teacher study metadata and train-only calibration. No holdout score is
# read while selecting the scale of the auxiliary losses.
def _ft_common(config_path):
    from .data import PreparedDINOData
    study=read(config_path)
    config=read(study['base_config'])
    base=_base_spec(study['base_checkpoint'],SMOKE_BASE_SHA,study['allowed_base_root'],20261102)
    need(base['config']==config and config.get('reproducibility',{}).get('mode')=='strict','Use the exact strict matched Base config')
    cache,inputs=_cache_spec(study['cache_root'])
    need(cache['config']==base['config_file'],'Use the same full paired producer cache')
    root=Path(study['dino_cache_root']).resolve()
    dino=PreparedDINOData(root,study['cache_root'])
    try:
        need(dino.ids==cache['selection']['training_ids'],'DINO and VAE cache order differs')
        manifest=read(root/'manifest.json')
        dino_cache={'root':str(root),'manifest':binding(root/'manifest.json'),'seal':binding(root/'seal.json'),
                    'teacher':manifest['teacher'],'completion_receipt':binding(root/'completion_receipt.json'),
                    'extraction_receipt':binding(root/'extraction_receipt.json') if (root/'extraction_receipt.json').exists() else None}
    finally:dino.close()
    need(study['updates']==2048 and study['training_seed']==20261102 and study['student_layer']==7
         and study['eval_updates']==[512,1024,2048] and study['checkpoint_updates']==[512,1024,1536,2048],
         'Retain the prospectively declared single-seed small budget')
    need(study['smooth_l1_beta']==1. and study['head_depth']==1,'Keep meanSmoothL1 beta1 and comparable single-Linear heads')
    return {'study':study,'study_file':binding(config_path),'sources':_source_bindings(),'base':base,
        'cache':cache,'inputs':inputs,'dino_cache':dino_cache,
        'official_model_files':_official_bindings(study['model_directory']),'dependencies':_dependency_bindings(),
        'optimizer':{k:config['training'][k] for k in ('learning_rate','weight_decay','max_grad_norm')},
        'admission':read(study['gpu_admission']),'admission_file':binding(study['gpu_admission']),
        'gpu_lock':study['gpu_lock'],'device':study['device'],'memory_cap_bytes':16*1024**3,
        'strict_environment':STRICT_ENVIRONMENT,'ordinary_inference_weight_EMA':False,
        'learning_rate_schedule':{'type':'constant','separate_scheduler':None,'optimizer_groups_restored':True,
            'training_timestep_grid':'bound native UniPC configuration, deterministic set_timesteps(1000)'}}


def collect_ft_calibration_protocol(config_path):
    p=_ft_common(config_path)
    need(p['study']['schema']=='fixed_teacher_design_v1','Use the pre-score design for calibration')
    return p|{'schema':'fixed_teacher_calibration_protocol_v1','scientific_update_budget':0,
       'diagnostic_optimizer_budget':27,'calibration_sigmas':[.2,.5,.8],
       'calibration_training_index':0,'relative_auxiliary_gradient_strengths':[.1,1.],
       'selection_uses_validation_or_test':False}


def collect_fixed_teacher_protocol(config_path):
    p=_ft_common(config_path);s=p['study']
    need(s['schema']=='fixed_teacher_study_v1','Expected finalized fixed-teacher study')
    report=Path(s['calibration_report']).resolve();caldir=report.parent
    cal=read(report);calp=read(caldir/'protocol.json');calstate=read(caldir/'status.json')
    need(calstate.get('status')=='completed' and not (caldir/'failure.json').exists()
         and calstate.get('protocol_sha256')==file_sha(caldir/'protocol.json')
         and calstate.get('summary_sha256')==file_sha(report),'Preflight did not finish successfully through cleanup')
    need(cal['status']=='passed_fixed_teacher_preflight' and cal['scientific_updates']==0
         and cal['diagnostic_optimizer_updates']==27 and cal['sources']==p['sources']
         and cal['protocol_sha256']==file_sha(caldir/'protocol.json'),'Require a complete source-matched real fixed-teacher preflight')
    need(calp==collect_ft_calibration_protocol(calp['study_file']['path']),'Preflight inputs changed')
    for key in ('base','cache','inputs','dino_cache','official_model_files','dependencies','optimizer','admission','gpu_lock','device'):
        need(p[key]==calp[key],'Study differs from the preflight: '+key)
    for key in ('updates','student_layer','smooth_l1_beta','eval_updates','checkpoint_updates','evaluation','metric_weights','dinov2'):
        need(s[key]==calp['study'][key],'Final study changed the calibrated design: '+key)
    expected=[{'label':'baseline','family':'baseline','setting':'baseline','weight':0.}]
    for family,short,kind in (('hr_vae','vae','vae'),('hr_dino','dino','dino')):
        for strength,weight in zip(('low','high'),cal['calibrations'][kind]['weights']):
            expected.append({'label':short+'_'+strength,'family':family,'setting':strength,'weight':weight})
    need(s['arms']==expected,'Candidate settings must come only from train-gradient calibration')
    for name in ('validation_seal','validation_dataset','test_cohort_plan'):
        item=s['evaluation'][name];need(binding(item['path'],item['sha256'])==item,'Changed prespecified evaluation declaration')
    p.update(schema='fixed_teacher_training_protocol_v1',scientific_update_budget=5*s['updates'],
             calibration={'protocol':binding(caldir/'protocol.json'),'summary':binding(report),'status_file':binding(caldir/'status.json')})
    return p


def prepare_ft_calibration(config_path,run_dir):
    p=collect_ft_calibration_protocol(config_path);run,_=_safe_run(Path(run_dir),Path(p['study']['run_root']))
    need(not run.exists(),'Use a new calibration run')
    run.mkdir(parents=True);save(run/'protocol.json',p)
    result={'status':'planned','protocol_sha256':file_sha(run/'protocol.json'),'scientific_updates':0,'GPU_loaded':False}
    save(run/'status.json',result);return result|{'run_dir':str(run)}


def _ft_engine_snapshot(engine):
    from .checkpoints import _cpu_owned
    return _cpu_owned({'student':{n:v for n,v in engine.model.named_parameters() if v.requires_grad},
          'head':None if engine.alignment is None else engine.alignment.state_dict(),
          'optimizer':engine.optimizer.state_dict(),'steps':engine.steps,
          'RNG':{n:getattr(engine.rng,n).get_state() for n in ('noise','timestep','mask','corruption')}})


def execute_ft_calibration(run_dir,expected):
    from .data import PreparedTrainingData,PreparedDINOData
    from .teacher import FrozenDINOv2,capture_block_output
    from .vae import load_frozen_vae,encode_video,decode_latent
    from .metrics import quantize_rgb,paired_metrics,PerceptualMetrics
    from .losses import shared_gradient_norms,calibrated_auxiliary_weights
    from .checkpoints import save_hr_latent_checkpoint,restore_hr_latent_training_state,load_training_checkpoint,_hr_equal
    from .model import tensor_state_hashes
    run=Path(run_dir).resolve();p=read(run/'protocol.json');s=p['study']
    run,_=_safe_run(run,Path(s['run_root']))
    need(file_sha(run/'protocol.json')==expected and p==collect_ft_calibration_protocol(p['study_file']['path']),
         'Calibration source/input protocol changed')
    need(read(run/'status.json')['status']=='planned','One diagnostic attempt per calibration run')
    need(all(os.environ.get(k)==v for k,v in STRICT_ENVIRONMENT.items()),'Strict offline calibration environment required')
    engine=direct=reloaded=teacher=vae=reader=dino=scheduler=None;updates=0;rows=[]
    wall_started=time.perf_counter();gradient_seconds={}
    clean=low=embedding=target=noise=noisy=flow_target=sigma=timestep=flow=features=fm=aux=head_grads=None
    def cleanup():
        nonlocal engine,direct,reloaded,teacher,vae,reader,dino,scheduler
        nonlocal clean,low,embedding,target,noise,noisy,flow_target,sigma,timestep,flow,features,fm,aux,head_grads
        _hr_release(engine);_hr_release(direct)
        if teacher is not None:teacher.close()
        if reader is not None:reader.close()
        if dino is not None:dino.close()
        engine=direct=reloaded=teacher=vae=reader=dino=scheduler=None
        clean=low=embedding=target=noise=noisy=flow_target=sigma=timestep=flow=features=fm=aux=head_grads=None
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
    def failure(error):
        record={'status':'failed_calibration','protocol_sha256':expected,'error':str(error),
            'traceback':traceback.format_exc(),'diagnostic_optimizer_updates_returned':updates,'scientific_updates':0}
        if not (run/'failure.json').exists():save(run/'failure.json',record)
        save(run/'status.json',record,replace=True)
    try:
        with _hr_gpu_session(p,cleanup,failure) as admission:
            save(run/'GPU_admission.json',admission);save(run/'runtime.json',_configure_runtime(p['base']['config']))
            save(run/'status.json',{'status':'running','protocol_sha256':expected,'scientific_updates':0},replace=True)
            reader=PreparedTrainingData(p['cache']['root'],expected_seal_sha256=p['cache']['seal']['sha256'])
            dino=PreparedDINOData(p['dino_cache']['root'],p['cache']['root'],expected_seal_sha256=p['dino_cache']['seal']['sha256'])
            vae=load_frozen_vae(s['model_directory'],device=s['device']);vf=_frozen(vae)
            teacher=FrozenDINOv2(s['dinov2']['source_dir'],s['dinov2']['weights'],frame_batch_size=s['dinov2']['frame_batch_size'])
            teacher.prepare(s['device']);online=[];reconstruction=[]
            perceptual=PerceptualMetrics(s['metric_weights']['alexnet']['path'],s['metric_weights']['linear']['path'],device='cpu')
            seal=read(Path(p['cache']['root'])/'cache_asset_seal.json')['artifact_sha256']
            for index in (0,1):
                rgbpath=Path(p['cache']['root'])/f'clip_{index}'/'paired_rgb.pt'
                lpath=Path(p['cache']['root'])/f'clip_{index}'/'fresh_latents.pt'
                binding(rgbpath,seal[f'clip_{index}/paired_rgb.pt']);binding(lpath,seal[f'clip_{index}/fresh_latents.pt'])
                rgb=torch.load(rgbpath,map_location='cpu',weights_only=True)
                target=torch.load(lpath,map_location='cpu',weights_only=True)['clean']
                encoded=encode_video(vae,rgb['hr']);need(torch.equal(encoded,target),'Online VAE target differs from the original cache')
                feature=teacher.encode_hr(rgb['hr']);cached=dino.load(index)
                need(torch.equal(feature,cached),'Online DINO target differs from bound cache')
                decoded=decode_latent(vae,encoded)
                full=paired_metrics(quantize_rgb(decoded),quantize_rgb(rgb['hr']),perceptual,excluded_reference=0)
                reconstruction.append({'source_cache_index':index,'VAE_encoding_exact_cached':True,
                    'diagnosis_only_not_SR_upper_bound':True,'metrics':full['all_frames']})
                online.append({'source_cache_index':index,'DINO_online_exact_cached':True,'shape':list(feature.shape)})
            need(vf==_frozen(vae) and all(not x.requires_grad and x.grad is None for x in vae.parameters()),'VAE teacher changed')
            teacher.close();teacher=None;perceptual=None;gc.collect();torch.cuda.empty_cache()
            scheduler=_training_scheduler(s['model_directory'],s['device']);pair=reader[0];text=reader.text_for(0)
            calibrations={}
            for kind in ('vae','dino'):
                gradient_started=time.perf_counter()
                engine=_hr_engine(p,1.,kind=kind);before=tensor_state_hashes(engine.model);cal_rows=[]
                clean,low,embedding=(pair['clean'].to(s['device']),pair['low'].to(s['device']),text.to(s['device']))
                target=clean if kind=='vae' else dino[0].to(s['device'])
                for i,nominal in enumerate((.2,.5,.8)):
                    grid=scheduler.sigmas[:1000].detach().float().cpu()
                    index=int((grid-nominal).abs().argmin());sigma=scheduler.sigmas[index].to(clean.device).float()
                    timestep=scheduler.timesteps[index].to(clean.device).expand(clean.shape[0])
                    noise=torch.randn(clean.shape,device=clean.device,generator=torch.Generator(device=clean.device).manual_seed(s['training_seed']+101+i))
                    noisy=(1-sigma)*clean+sigma*noise;flow_target=noise-clean
                    engine.model.train();engine.alignment.train()
                    flow,features=capture_block_output(engine.model,7,noisy,timestep,embedding,low)
                    fm=flow_matching_loss(flow,flow_target);aux,info=engine.alignment(features,target,tuple(engine.model.token_grid))
                    aux_value,fm_value=float(aux.detach()),float(fm.detach())
                    head_grads=torch.autograd.grad(aux,list(engine.alignment.parameters()),retain_graph=True)
                    head_norm=math.sqrt(math.fsum(float(g.detach().double().square().sum().cpu()) for g in head_grads))
                    head_grads=None
                    norms=shared_gradient_norms(fm,aux,[v for v in engine.model.parameters() if v.requires_grad])
                    cal_rows.append({'sigma':nominal,'native_sigma':float(sigma),'native_timestep':float(timestep[0]),
                                     'native_timestep_index':index,'noise_seed':s['training_seed']+101+i,
                                     'fm':fm_value,'unit_aux':aux_value,'head_unit_aux_grad_norm':head_norm,**norms})
                    engine.model.clear_features();flow=features=fm=aux=noisy=flow_target=None
                need(before==tensor_state_hashes(engine.model) and engine.steps==0 and not engine.optimizer.state,
                     'Gradient calibration must not update the student')
                calibrations[kind]=calibrated_auxiliary_weights(cal_rows)|{'rows':cal_rows}
                gradient_seconds[kind]=time.perf_counter()-gradient_started
                _hr_release(engine);engine=None;clean=low=embedding=target=None;gc.collect();torch.cuda.empty_cache()
            resources={};initials=[];baseline_trajectory=[]
            for label,kind,weight in [('baseline','vae',0.),('hr_vae','vae',calibrations['vae']['weights'][1]),
                                      ('hr_dino','dino',calibrations['dino']['weights'][1])]:
                engine=_hr_engine(p,weight,kind=kind);initials.append(tensor_state_hashes(engine.model));timed=[]
                torch.cuda.reset_peak_memory_stats()
                for i in range(8):
                    row,seconds,event=_hr_step(engine,reader[i],reader.text_for(i),scheduler,s['device'],
                                              teacher_target=dino[i] if kind=='dino' else None)
                    updates+=1;entry=row|{'role':label,'seconds':seconds,'gpu_event_seconds':event,'warmup':i<2}
                    rows.append(entry);timed.append(seconds)
                    if label=='baseline' and i<3:baseline_trajectory.append(tensor_state_hashes(engine.model))
                resources[label]={'weight':weight,'head_parameters':0 if engine.alignment is None else sum(v.numel() for v in engine.alignment.parameters()),
                   'warmup2_and_timed6_seconds':timed,'timed6_mean_seconds':statistics.mean(timed[2:]),
                   'timed6_sample_SD_seconds':statistics.stdev(timed[2:]),'memory':_memory(torch,p['memory_cap_bytes'])}
                if weight>0:
                    checkpoint=run/'resume_checks'/label/'Base'/'checkpoint_000008.pt';cfg=_hr_config(p,weight,kind=kind)
                    snapshot=_ft_engine_snapshot(engine)
                    save_hr_latent_checkpoint(checkpoint,engine,cfg,allowed_root=run,protocol_sha256=expected,
                        cumulative_training_seconds=sum(timed),initialization=engine.base_load_audit)
                    _hr_release(engine);engine=None;gc.collect();torch.cuda.empty_cache()
                    reloaded,audit=load_training_checkpoint(checkpoint,device=s['device'],allowed_root=run,variant='Base',model_directory=s['model_directory'])
                    need(_hr_equal({n:v for n,v in reloaded.named_parameters() if v.requires_grad},snapshot['student']),
                         'Student-only inference reload differs')
                    need(audit['training_only_state_restored'] is False,'Auxiliary state leaked into inference')
                    reloaded=None;gc.collect();torch.cuda.empty_cache()
                    engine=_hr_engine(p,weight,kind=kind)
                    restore_hr_latent_training_state(engine,checkpoint,allowed_root=run,expected_protocol_sha256=expected,expected_config=cfg)
                    need(_hr_equal(_ft_engine_snapshot(engine),snapshot),'Head/AdamW/student/RNG exact GPU resume failed')
                _hr_release(engine);engine=None;gc.collect();torch.cuda.empty_cache()
            need(all(x==initials[0] for x in initials),'Preflight students had different initialization')
            direct=_hr_engine(p,0.)
            for i in range(3):
                pair=reader[i]
                NoRefContinuationStep.step(direct,pair['clean'].to(s['device']),pair['low'].to(s['device']),reader.text_for(i).to(s['device']),scheduler)
                updates+=1;need(tensor_state_hashes(direct.model)==baseline_trajectory[i],'Zero alignment failed original GPU trajectory reproduction')
            need(updates==27,'Unexpected diagnostic update budget')
            result={'status':'passed_fixed_teacher_preflight','protocol_sha256':expected,'sources':p['sources'],
                 'scientific_updates':0,'diagnostic_optimizer_updates':updates,'calibrations':calibrations,
                 'resources':resources,'relative_timed6_seconds_vs_baseline':{k:v['timed6_mean_seconds']/resources['baseline']['timed6_mean_seconds'] for k,v in resources.items()},
                 'online_cache_checks':online,'HR_VAE_reconstruction_diagnostics':reconstruction,
                 'all_step_rows':rows,'original_baseline_exact_GPU_updates':3,'both_heads_full_resume_exact':True,
                 'diagnostic_weights_reused':False,'validation_or_test_scores_used_for_weights':False,
                 'preflight_wall_seconds_including_metadata_io':time.perf_counter()-wall_started,
                 'gradient_calibration_wall_seconds_including_fresh_model_load':gradient_seconds}
            save(run/'summary.json',result);save(run/'status.json',{'status':'completed','protocol_sha256':expected,
                 'summary_sha256':file_sha(run/'summary.json'),'scientific_updates':0},replace=True)
            return result
    except BaseException as error:
        if not (run/'failure.json').exists():failure(error)
        raise
    finally:_hr_cleanup_preserving(cleanup,sys.exc_info()[1])

if __name__=='__main__':
    main()
