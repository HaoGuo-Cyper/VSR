"""Trainable-only EMA teacher and safe intermediate-feature capture.

Teacher execution finishes before building the student graph. Importing this
module never loads a model, initializes CUDA or starts an experiment.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import math
import torch

def _require(value, message):
    if not value:
        raise ValueError(message)


def _trainables(model):
    pairs = dict(model.named_parameters())
    result = {name: parameter for name, parameter in pairs.items() if parameter.requires_grad}
    _require(result, "Restoration model has no trainable parameters")
    return result


def _frozen_signature(model):
    return {name: (id(p), p.data_ptr(), p._version, tuple(p.shape), p.dtype, p.device)
            for name, p in model.named_parameters() if not p.requires_grad}


def _require_finite(entries):
    """Check every (failure message, tensor), with one verdict per device.

    Shape/name/dtype/device contracts belong to the caller. All finite-value
    reductions stay on their tensor's device until a single combined verdict
    is read. Only a failing device needs individual reads to locate the first
    offender. This changes synchronization, not the set of values checked.
    No tensor is mutated, so callers can validate before committing a change.
    """
    per_device = {}
    for message, value in entries:
        _require(isinstance(value, torch.Tensor), "Finite validation requires tensors")
        verdict = torch.isfinite(value.detach()).all()
        per_device.setdefault(value.device, []).append((message, verdict))
    for checks in per_device.values():
        if not bool(torch.stack([verdict for _, verdict in checks]).all()):
            for message, verdict in checks:
                _require(bool(verdict), message)


class TrainableEMA:
    """EMA of exactly the captured restoration trainables, initialized to Base.

    Floating dtypes/shapes/names are exact contracts. Frozen backbone parameters
    are never copied into the EMA or modified by it. update() is explicit; no
    optimizer operation occurs here. State export returns detached copies.
    """

    def __init__(self, model, decay=0.999):
        _require(type(decay) in (int, float) and math.isfinite(decay) and 0 <= decay < 1, "EMA decay must lie in [0,1)")
        self.decay, self.num_updates = float(decay), 0
        params = _trainables(model)
        self._spec = {name: (tuple(p.shape), p.dtype, p.device) for name, p in params.items()}
        _require(all(p.is_floating_point() for p in params.values()), "EMA initialization requires finite floating trainables")
        _require_finite(("EMA initialization requires finite floating trainables: " + name, p)
                        for name, p in params.items())
        self._shadow = {name: p.detach().clone() for name, p in params.items()}
        self._swapping = False

    @property
    def names(self):
        return tuple(self._spec)

    def _validate(self, model):
        params = _trainables(model)
        _require(set(params) == set(self._spec), "EMA trainable parameter names changed")
        _require(set(self._shadow) == set(self._spec), "EMA shadow parameter names changed")
        finite = []
        for name, p in params.items():
            _require((tuple(p.shape), p.dtype, p.device) == self._spec[name], "EMA shape/dtype/device changed: " + name)
            shadow = self._shadow[name]
            _require(isinstance(shadow, torch.Tensor) and (tuple(shadow.shape), shadow.dtype, shadow.device) == self._spec[name]
                     and not shadow.requires_grad, "Invalid EMA shadow: " + name)
            finite.extend((("Nonfinite student trainable: " + name, p),
                           ("Invalid EMA shadow: " + name, shadow)))
        _require_finite(finite)
        return params

    @torch.no_grad()
    def update(self, model):
        _require(not self._swapping, "Cannot update EMA during a temporary teacher swap")
        params = self._validate(model)  # validate every entry before mutation
        frozen = _frozen_signature(model)
        for name, p in params.items():
            self._shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        self.num_updates += 1
        _require(frozen == _frozen_signature(model), "Frozen parameters changed during EMA update")

    def state_dict(self):
        return {"schema": "vsr_trainable_only_ema_v1", "decay": self.decay,
                "num_updates": self.num_updates, "trainable_names": list(self.names),
                "shadow": {name: value.detach().clone() for name, value in self._shadow.items()}}

    def load_state_dict(self, state):
        _require(not self._swapping, "Cannot load EMA state during a teacher swap")
        _require(state.get("schema") == "vsr_trainable_only_ema_v1" and state.get("decay") == self.decay,
                 "EMA state schema/decay mismatch")
        _require(type(state.get("num_updates")) is int and state["num_updates"] >= 0, "Invalid EMA update count")
        _require(state.get("trainable_names") == list(self.names) and isinstance(state.get("shadow"), dict)
                 and set(state["shadow"]) == set(self._spec), "EMA state parameter names mismatch")
        for name, value in state["shadow"].items():
            _require(isinstance(value, torch.Tensor) and (tuple(value.shape), value.dtype, value.device) == self._spec[name]
                     and not value.requires_grad, "EMA state shape/dtype/device/value mismatch: " + name)
        _require_finite(("EMA state shape/dtype/device/value mismatch: " + name, value)
                        for name, value in state["shadow"].items())
        staged = {name: value.detach().clone() for name, value in state["shadow"].items()}
        self._shadow, self.num_updates = staged, state["num_updates"]

    @contextmanager
    def teacher_parameters(self, model):
        """Temporarily use EMA under eval/no_grad; always restore student/modes.

        Use before the student forward. Existing student gradients are retained.
        Caller-side modifications to frozen parameter versions are detected.
        """
        _require(not self._swapping, "Nested teacher swaps are forbidden")
        params = self._validate(model)
        saved = {name: p.detach().clone() for name, p in params.items()}
        flags = [(module, module.training) for module in model.modules()]
        frozen = _frozen_signature(model)
        self._swapping = True
        try:
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(self._shadow[name])
                model.eval()
                yield
        finally:
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(saved[name])
            for module, training in flags:
                module.training = training
            self._swapping = False
            _require(frozen == _frozen_signature(model), "Frozen parameters changed during the teacher call")

    def teacher_forward(self, model, call, *args, **kwargs):
        """Call a supplied forward/hook capture and return detached owned values."""
        def own(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, tuple):
                return tuple(own(item) for item in value)
            if isinstance(value, list):
                return [own(item) for item in value]
            if isinstance(value, dict):
                return {key: own(item) for key, item in value.items()}
            _require(value is None or type(value) in (bool, int, float, str), "Unsupported teacher result type")
            return value
        with self.teacher_parameters(model):
            return own(call(*args, **kwargs))


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def capture_block_output(model, block_index, noisy, timestep, text, low):
    """Observe one output, then remove the hook before checkpoint replay.

    Does not clear the model's LR feature graph. Student callers must retain it
    through backward and clear it afterward. A teacher caller may clear it as
    soon as its no-grad result has been obtained.
    """
    _require(type(block_index) is int and 0 <= block_index < len(model.transformer.blocks), "Invalid captured block")
    captured = []

    def hook(_module, _args, output):
        _require(isinstance(output, torch.Tensor) and output.ndim == 3, "Expected Wan block output [B,N,C]")
        captured.append(output)

    handle = model.transformer.blocks[block_index].register_forward_hook(hook)
    try:
        with _autocast(noisy.device):
            flow, projection = model(noisy, timestep, text, low)
        _require(projection is None and len(captured) == 1, "Base block must be observed exactly once on the forward pass")
        return flow, captured[0]
    finally:
        handle.remove()


DINOV2_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINOV2_WEIGHT_SHA256 = '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'


def dinov2_feature_spec():
    """This study's explicit adapter, not a claim about exact VAE support."""
    return {'input_shape':[1,3,33,128,224], 'input_range':'RGB_float32_0_1',
            'resize_hw':[112,196], 'resize':'bicubic_antialias_align_cornersFalse_clamp01',
            'spatial_scale':[7,8], 'crop':False, 'padding':False,
            'rgb_mean':[.485,.456,.406], 'rgb_std':[.229,.224,.225],
            'feature':'final_x_norm_patchtokens', 'CLS_included':False, 'registers':0,
            'patch_size':14, 'patch_grid':[8,14], 'channels':768,
            'time_groups':[[0],*[list(range(i,i+4)) for i in range(1,33,4)]],
            'time_reduction':'FP32_arithmetic_mean_before_channel_L2_in_loss',
            'token_order':'latent_time,H,W', 'output_shape':[1,1008,768],
            'exact_VAE_receptive_field_claimed':False,
            'attention':'torch_SDPA_math_only_XFORMERS_DISABLED'}


def dinov2_preprocess_hr(hr):
    """Keep the complete HR crop; 14 teacher pixels map to16 HR pixels."""
    import torch.nn.functional as F
    _require(isinstance(hr,torch.Tensor) and tuple(hr.shape)==(1,3,33,128,224)
             and hr.dtype==torch.float32 and not hr.requires_grad
             and bool(torch.isfinite(hr).all()) and float(hr.min())>=0 and float(hr.max())<=1,
             'DINO input must be frozen finite HR RGB01 [1,3,33,128,224]')
    frames=hr[0].permute(1,0,2,3).contiguous()
    with torch.autocast(hr.device.type,enabled=False):
        frames=F.interpolate(frames,size=(112,196),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        mean=frames.new_tensor([.485,.456,.406]).view(1,3,1,1)
        std=frames.new_tensor([.229,.224,.225]).view(1,3,1,1)
        return ((frames-mean)/std).detach()


def group_dinov2_time(tokens):
    """Map all33 frame feature maps to nine groups; do not normalize here."""
    _require(isinstance(tokens,torch.Tensor) and tuple(tokens.shape)==(33,112,768)
             and tokens.dtype==torch.float32 and bool(torch.isfinite(tokens).all()),
             'DINO must provide all33 finite FP32 8x14x768 patch maps')
    with torch.no_grad():
        groups=[tokens[0],*[tokens[i:i+4].mean(0) for i in range(1,33,4)]]
        result=torch.stack(groups).reshape(1,1008,768).detach().contiguous()
    _require(bool(torch.isfinite(result).all()),'Nonfinite grouped DINO targets')
    return result


def dinov2_teacher_binding(source_dir, checkpoint_path):
    """Metadata/hash verification only; never construct a model or use CUDA."""
    import hashlib
    import importlib.metadata
    from pathlib import Path
    import subprocess
    source,checkpoint=Path(source_dir).resolve(),Path(checkpoint_path).resolve()
    _require((source/'hubconf.py').is_file() and checkpoint.is_file(),'Explicit local DINO source/weights required')
    def git(*args):
        return subprocess.run(['git','-C',str(source),*args],check=True,capture_output=True).stdout
    _require(git('rev-parse','HEAD').decode().strip()==DINOV2_REVISION,'Wrong DINO source revision')
    _require(not git('status','--porcelain','--untracked-files=no').strip(),'DINO tracked source is modified')
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    weight_sha=sha(checkpoint)
    _require(weight_sha==DINOV2_WEIGHT_SHA256,'Wrong official non-register DINOv2-B/14 weights')
    files={}
    for raw in git('ls-files','-z').split(b'\0'):
        if raw:
            name=raw.decode();path=source/name
            _require(path.is_file() and path.resolve().is_relative_to(source),'DINO source file escapes source root')
            files[name]=sha(path)
    runtime={}
    for name in ('torch','torchvision','xformers'):
        try:runtime[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:runtime[name]=None
    return {'architecture':'dinov2_vitb14', 'source_revision':DINOV2_REVISION,
            'source_dir':str(source), 'source_files':files,
            'weights':{'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':weight_sha},
            'feature_spec':dinov2_feature_spec(), 'precision':'FP32_no_autocast', 'runtime':runtime}


def _verify_dinov2_imported_sources(binding):
    """Hash ordinary modules; constrain official namespace package search paths."""
    import sys
    import hashlib
    from pathlib import Path
    root=Path(binding['source_dir']).resolve()
    for name,module in tuple(sys.modules.items()):
        if name!='dinov2' and not name.startswith('dinov2.'):continue
        filename=getattr(module,'__file__',None)
        if filename is None:
            expected=root.joinpath(*name.split('.')).resolve()
            paths=[Path(p).resolve() for p in getattr(module,'__path__',())]
            relative=str(expected.relative_to(root))+'/' if expected.is_relative_to(root) else None
            _require(expected.is_dir() and paths==[expected] and relative is not None
                     and any(p.startswith(relative) for p in binding['source_files']),
                     'DINO namespace search path shadows bound source')
            continue
        path=Path(filename).resolve()
        _require(path.is_file() and path.is_relative_to(root),'Imported DINO module shadows bound local source')
        relative=str(path.relative_to(root))
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        _require(binding['source_files'].get(relative)==actual,'Imported DINO source changed or is untracked')


class FrozenDINOv2(torch.nn.Module):
    """Fixed local DINOv2-B/14; prepare explicitly inside the caller's lease."""
    def __init__(self,source_dir,checkpoint_path,*,frame_batch_size=8):
        super().__init__()
        _require(type(frame_batch_size) is int and 0<frame_batch_size<=33,'Positive bounded DINO frame batch required')
        self._binding=dinov2_teacher_binding(source_dir,checkpoint_path)
        self.frame_batch_size=frame_batch_size;self.model=None

    @property
    def binding(self):
        import copy
        return copy.deepcopy(self._binding)|{'frame_batch_size':self.frame_batch_size}

    def train(self,mode=True):
        return super().train(False)

    def prepare(self,device='cpu'):
        _require(self.model is None,'DINO teacher is already prepared')
        _require(dinov2_teacher_binding(self._binding['source_dir'],self._binding['weights']['path'])==self._binding,
                 'DINO source/weight/runtime changed before preparation')
        import os
        previous_xformers=os.environ.get('XFORMERS_DISABLED')
        os.environ['XFORMERS_DISABLED']='1'
        try:
            with torch.device('cpu'):
                model=torch.hub.load(self._binding['source_dir'],'dinov2_vitb14',source='local',pretrained=False)
        finally:
            if previous_xformers is None:os.environ.pop('XFORMERS_DISABLED',None)
            else:os.environ['XFORMERS_DISABLED']=previous_xformers
        import sys
        _verify_dinov2_imported_sources(self._binding)
        attention=sys.modules.get('dinov2.layers.attention')
        _require(attention is not None and attention.XFORMERS_AVAILABLE is False,
                 'DINO requires bound torch SDPA attention, never xFormers kernels')
        state=torch.load(self._binding['weights']['path'],map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        _require(model.num_register_tokens==0 and model.embed_dim==768
                 and tuple(model.patch_embed.patch_size)==(14,14),'DINO architecture/register contract differs')
        self.model=model.requires_grad_(False).eval().to(device=device,dtype=torch.float32)
        self.train(False)
        return self

    @torch.no_grad()
    def encode_hr(self,hr):
        _require(self.model is not None and not self.model.training
                 and all(not p.requires_grad and p.dtype==torch.float32 for p in self.model.parameters()),
                 'Prepare one frozen FP32 DINO teacher first')
        device=next(self.model.parameters()).device
        frames=dinov2_preprocess_hr(hr).to(device=device,dtype=torch.float32)
        output=[]
        with torch.autocast(device.type,enabled=False):
            for start in range(0,33,self.frame_batch_size):
                features=self.model.forward_features(frames[start:start+self.frame_batch_size])['x_norm_patchtokens']
                _require(features.shape==(min(self.frame_batch_size,33-start),112,768)
                         and features.dtype==torch.float32,'DINO patch token shape/dtype differs')
                output.append(features.detach().cpu())
        return group_dinov2_time(torch.cat(output))

    def close(self):
        import gc
        self.model=None;gc.collect()


class FP32TrainableTeacher(TrainableEMA):
    """Same-Wan frozen or EMA target; only restoration trainables are shadowed.

    Frozen backbone storage is shared through the inherited exception-safe swap.
    Shadows are FP32 regardless of the student's parameter dtype; the temporary
    teacher copy uses the student's execution dtype. The auxiliary head never
    belongs to the restoration model and is therefore never a teacher parameter.
    """
    def __init__(self, model, *, mode, decay=.999):
        _require(mode in ('frozen', 'ema'), 'Teacher mode must be frozen or ema')
        super().__init__(model, decay=decay)
        self.mode = mode
        self._shadow = {name:value.float() for name,value in self._shadow.items()}

    @property
    def shadow(self):
        return self._shadow

    def _validate(self, model):
        params = _trainables(model)
        _require(set(params) == set(self._spec) and set(self._shadow) == set(self._spec),
                 'Teacher trainable parameter names changed')
        _require(type(self.num_updates) is int and self.num_updates >= 0
                 and (self.mode != 'frozen' or self.num_updates == 0), 'Invalid teacher update count')
        checks = []
        for name,p in params.items():
            shape,dtype,device = self._spec[name]
            value = self._shadow[name]
            _require((tuple(p.shape),p.dtype,p.device) == (shape,dtype,device),
                     'Teacher source shape/dtype/device changed: '+name)
            _require(isinstance(value,torch.Tensor) and tuple(value.shape) == shape
                     and value.dtype == torch.float32 and value.device == device
                     and not value.requires_grad, 'Teacher shadow shape/dtype/device mismatch: '+name)
            checks.extend((('Nonfinite student trainable: '+name,p), ('Nonfinite teacher shadow: '+name,value)))
        _require_finite(checks)
        return params

    @torch.no_grad()
    def update(self, model):
        _require(self.mode == 'ema', 'Frozen teacher cannot be updated')
        _require(not self._swapping, 'Cannot update teacher during a temporary swap')
        params = self._validate(model)
        frozen = _frozen_signature(model)
        for name,p in params.items():
            self._shadow[name].mul_(self.decay).add_(p.detach().float(),alpha=1-self.decay)
        self.num_updates += 1
        _require_finite(('Nonfinite updated teacher shadow: '+name,value) for name,value in self._shadow.items())
        _require(frozen == _frozen_signature(model), 'Frozen parameters changed during teacher update')

    def state_dict(self):
        return {'schema':'vsr_fp32_trainable_teacher_v1','mode':self.mode,'decay':self.decay,
                'num_updates':self.num_updates,'trainable_names':list(self.names),
                'shadow':{name:value.detach().clone() for name,value in self._shadow.items()}}

    def load_state_dict(self,state):
        _require(not self._swapping, 'Cannot load teacher during a temporary swap')
        _require(isinstance(state,dict) and state.get('schema') == 'vsr_fp32_trainable_teacher_v1'
                 and state.get('mode') == self.mode and state.get('decay') == self.decay,
                 'Teacher state schema/mode/decay mismatch')
        count = state.get('num_updates')
        _require(type(count) is int and count >= 0 and (self.mode != 'frozen' or count == 0),
                 'Teacher state update count mismatch')
        _require(state.get('trainable_names') == list(self.names) and isinstance(state.get('shadow'),dict)
                 and set(state['shadow']) == set(self._spec), 'Teacher state parameter names mismatch')
        for name,value in state['shadow'].items():
            shape,_,device = self._spec[name]
            _require(isinstance(value,torch.Tensor) and (tuple(value.shape),value.dtype)
                     == (shape,torch.float32) and value.device in (torch.device('cpu'),device) and not value.requires_grad,
                     'Teacher state shape/dtype/device mismatch: '+name)
        _require_finite(('Nonfinite teacher state: '+name,value) for name,value in state['shadow'].items())
        staged = {name:value.detach().to(device=self._spec[name][2],copy=True) for name,value in state['shadow'].items()}
        self._shadow,self.num_updates = staged,count


def select_paired_teacher_timestep(scheduler, student_index, *, ratio=.5):
    """Nearest valid native timestep to half sigma, deterministic tie breaking.

    Only the first1000 paired sigma/timestep entries are eligible: an unpaired
    trailing sigma=0 is not a valid teacher time. Teacher sigma never exceeds
    the student sigma; ties prefer smaller sigma, then smaller native index.
    """
    _require(type(student_index) is int and 0 <= student_index < 1000
             and len(scheduler.timesteps) == 1000 and len(scheduler.sigmas) >= 1000,
             'Use a valid index from the paired native1000 training grid')
    _require(type(ratio) in (int,float) and math.isfinite(ratio) and 0 < ratio <= 1,
             'Teacher sigma ratio must lie in (0,1]')
    sigmas = scheduler.sigmas[:1000].detach().to(device='cpu',dtype=torch.float32)
    times = scheduler.timesteps.detach().to(device='cpu')
    _require(sigmas.ndim == times.ndim == 1 and bool(torch.isfinite(sigmas).all())
             and bool(torch.isfinite(times).all()) and bool(((sigmas >= 0) & (sigmas <= 1)).all()),
             'Native paired grid must be finite with sigmas in [0,1]')
    values = sigmas.tolist()
    student_sigma = values[student_index]
    target_sigma = student_sigma * float(ratio)
    eligible = [i for i,value in enumerate(values) if value <= student_sigma]
    teacher_index = min(eligible,key=lambda i:(abs(values[i]-target_sigma),values[i],i))
    teacher_sigma = values[teacher_index]
    return {'student_index':student_index,'student_sigma':student_sigma,
            'student_native_timestep':float(times[student_index]),
            'teacher_index':teacher_index,'teacher_sigma':teacher_sigma,
            'teacher_native_timestep':float(times[teacher_index]),
            'requested_sigma_ratio':float(ratio),'requested_teacher_sigma':target_sigma,
            'realized_sigma_ratio':teacher_sigma/student_sigma if student_sigma > 0 else None,
            'paired_grid_entries':1000,'tie_rule':'absolute_distance_then_smaller_sigma_then_smaller_index'}
