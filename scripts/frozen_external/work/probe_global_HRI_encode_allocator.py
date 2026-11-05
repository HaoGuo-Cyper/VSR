"""One declared allocator-only whole-HRI T1 encode attempt, not full SR QA.

Prepared worker: only --run-dir; root supplies an independent direct service.
No model offload, sampler, source pixels, quality score or automatic retry.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

P = Path('/home/hugo/Projects/VSR')
sys.path.insert(0, str(P))
COMMIT = '04228644605f5a3d96834ba05dde636636400817'
UNIT = 'vsr-global-hri-encode-allocator.service'
SOURCE_ENV = P/'runs/formal_hq_v2_reference_repeat_seed20261104_20261102T225815.432804Z/environment.json'
FAILED_RUN = P/'runs/native_resource_probe_cublas_envfix_20261103T002352.890851Z'
CHECKPOINT = P/'runs/formal_hq_v2_reference_20261102T171545.936007Z/Both/checkpoint_008192.pt'
CHECKPOINT_SHA = 'cfed75da0b5a4a6670c02796dc25e6e4dca5227ee06b128d3d598f468691d275'
INPUT_SHAPE = [1, 3, 1, 1088, 1920]
OUTPUT_SHAPE = [1, 16, 1, 136, 240]
INPUT_SHA = '888ad9cda27c0b58bf8bdb5e985991d69802a1fcce3062c0f525d503a8d49a4c'
SOURCE_FILES = {
    'protocol.json': 'da7c4736a97773c97ff165550966a355a57ec67f8971e71aa47f62386a4ceb1e',
    'status.json': 'cabee1ca0e4303280a6626377b35f46d7506454ec5c198c23bfe68323fcd4b6e',
    'failure_observations.json': '4b9ad140286e6bc514129b5d2fbaee9cd0a58dc50f2c087232de486c1b5345db',
    'reference_maximum_paired_on_input.json': '77970993e00a62e28404ff5bdac2a551d117d9b106a2d8aa6587a323659d7fa6',
}
ENV_REQUIRED = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'Expected JSON object: '+str(path))
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metadata_preflight():
    for key, expected in ENV_REQUIRED.items():
        require(os.environ.get(key) == expected, 'Set exact startup environment '+key+'='+expected)
    require(CHECKPOINT.resolve() == CHECKPOINT and CHECKPOINT.is_relative_to(P/'runs'), 'Not exact own HRI Both path')
    require(sha(CHECKPOINT) == CHECKPOINT_SHA, 'Own HRI Both checkpoint changed')
    environment = read(SOURCE_ENV)
    require(environment['git_commit'] == COMMIT and len(environment['code_sha256']) == 106, 'Expected frozen042/106 source record')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip() == COMMIT, 'Production HEAD changed')
    for name, expected in environment['code_sha256'].items():
        path = (P/name).resolve()
        require(path.is_relative_to(P.resolve()) and sha(path) == expected, 'Frozen source changed: '+name)
    for name, expected in SOURCE_FILES.items():
        require(sha(FAILED_RUN/name) == expected, 'Preserved failed source changed: '+name)
    source_protocol, source_status = read(FAILED_RUN/'protocol.json'), read(FAILED_RUN/'status.json')
    source_failure = read(FAILED_RUN/'failure_observations.json')
    require(source_status['status'] == 'failed' and source_protocol['checkpoints']['reference']['checkpoint_sha256'] == CHECKPOINT_SHA,
            'Expected failed co-resident whole-HRI source')
    call = source_failure['partial_restore_audit']['VAE_encode_calls'][0]
    require(call['shape'] == INPUT_SHAPE and call['pixels_sha256'] == INPUT_SHA
            and not source_failure['partial_restore_audit']['sample_tile_latent_calls'], 'Failure was not the exact pre-sampler global T1 input')
    config_path = CHECKPOINT.parent.parent/'config.json'
    config = read(config_path)
    require(config == source_protocol['checkpoints']['reference']['config']
            and config['model']['revision'] == '0fad780a534b6463e45facd96134c9f345acfa5b'
            and config['model']['repo_id'] == 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
            and config['reproducibility']['mode'] == 'strict' and config['reference']['enabled'] is True
            and config['seed'] == config['training'].get('seed', config['seed']) == 20261102,
            'HRI Both configuration differs from preserved failure')
    return config, {'run': str(FAILED_RUN), 'file_sha256': SOURCE_FILES,
                    'protocol_sha256': source_status['protocol_sha256'], 'normalized_input_sha256': INPUT_SHA,
                    'original_completed_NoRef_case_repeated': False}, environment


def serializable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return repr(value)


def allocator_observation(torch):
    """Observe actual settings/segment flags; never set allocator at runtime."""
    if not torch.cuda.is_initialized():
        return {'CUDA_initialized': False}
    result = {'CUDA_initialized': True, 'backend': torch.cuda.get_allocator_backend()}
    errors = []
    try:
        snapshot = torch.cuda.memory._snapshot()
        result['snapshot_api'] = 'torch.cuda.memory._snapshot'
        result['allocator_settings'] = serializable(snapshot.get('allocator_settings'))
        segments = snapshot['segments']
    except Exception as error:
        errors.append(f'private snapshot: {type(error).__name__}: {error}')
        try:
            segments = torch.cuda.memory_snapshot()
            result['snapshot_api'] = 'torch.cuda.memory_snapshot'
            result['allocator_settings'] = None
        except Exception as error:
            result['snapshot_errors'] = errors+[f'public snapshot: {type(error).__name__}: {error}']
            return result
    selected = [row for row in segments if row.get('device', 0) == 0]
    result['segments'] = [{key: row.get(key) for key in ('device', 'total_size', 'allocated_size', 'active_size', 'is_expandable')}
                          for row in selected]
    flags = [row.get('is_expandable') for row in selected]
    result.update(segment_count=len(selected), expandable_segment_count=sum(value is True for value in flags),
                  actual_expandable_segment_observed=any(value is True for value in flags), snapshot_errors=errors)
    stats = torch.cuda.memory_stats(0)
    keys = ('inactive_split_bytes.all.current', 'inactive_split_bytes.all.peak',
            'num_alloc_retries', 'num_ooms', 'allocated_bytes.all.current', 'reserved_bytes.all.current')
    result['memory_stats'] = {key: stats.get(key) for key in keys}
    return result


def device_dtype_map(network):
    return {kind+':'+name: {'device': str(value.device), 'dtype': str(value.dtype), 'shape': list(value.shape)}
            for kind, values in (('parameter', network.named_parameters()), ('buffer', network.named_buffers()))
            for name, value in values}


def cache_observation(vae):
    return {name: {'length': len(getattr(vae, name, [])),
                   'tensor_entries': sum(value is not None for value in getattr(vae, name, []))}
            for name in ('_feat_map', '_enc_feat_map')}


def run(run):
    started = time.perf_counter(); contract = None
    model = vae = raw = None; measurements = None
    phase = 'CPU_metadata_preflight'; observations = {}; warnings_seen = []
    run = Path(run).expanduser().resolve()
    require(run.is_relative_to((P/'runs').resolve()) and run != P/'runs', 'Run must be a new own VSR runs child')
    require(not run.exists() or not any(run.iterdir()), 'Preserve prior attempt; never reuse a nonempty run')
    run.mkdir(parents=True, exist_ok=True)
    # JSON output is available even for failure before importing the GPU library.
    def output(path, value):
        payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        temp = path.with_suffix(path.suffix+'.tmp'); temp.write_text(payload); temp.replace(path)
    gpu_memory = host_memory = None
    networks = flags = before_grads = before_state = rng_before = device_maps_before = None
    try:
        config, failure_source, environment = metadata_preflight()
        import torch
        from vsr.frozen_evaluation import _default_model, _default_vae, _digest, _json
        from vsr.native_resource_probe import _Measurements, _gpu_memory, _host_memory
        from vsr.native_qa_restore import (_network_state, _rng_snapshot, _rng_equal,
                                          _grads, _grads_equal, _contexts_empty, _tensor_sha)
        from vsr.reference_diagnostics import assert_gpu_idle
        from vsr.reproducibility import configure_reproducibility, reproducibility_flags
        output = _json; gpu_memory = _gpu_memory; host_memory = _host_memory
        require(not torch.cuda.is_initialized(), 'Fresh pre-CUDA worker required')
        guard = assert_gpu_idle(P)
        require(UNIT in guard['own_diagnostic_service_units'], 'Register this direct service unit before starting worker')
        reproducibility = configure_reproducibility(config)
        input_cpu = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device='cpu')
        require(_tensor_sha(input_cpu) == INPUT_SHA, 'Synthetic normalized zero bytes differ from preserved failing encode')
        protocol = {'schema': 'vsr_global_HRI_encode_allocator_probe_v1',
            'checkpoint': {'path': str(CHECKPOINT), 'sha256': CHECKPOINT_SHA, 'variant': 'Both', 'step': 8192},
            'config': config, 'official_revision': config['model']['revision'],
            'code_commit': COMMIT, 'code106_sha256': environment['code_sha256'],
            'source_environment_sha256': sha(SOURCE_ENV), 'worker_sha256': sha(Path(__file__)),
            'failure_source': failure_source, 'startup_environment': {key: os.environ.get(key) for key in ENV_REQUIRED},
            'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'strict_reproducibility': reproducibility, 'input_normalized_shape': INPUT_SHAPE,
            'input_dtype': 'torch.float32', 'input_normalized_sha256': INPUT_SHA,
            'input_source': 'synthetic all-zero normalized tensor, exact padded whole-HRI VAE input bytes from preserved failure',
            'expected_raw_latent_shape': OUTPUT_SHAPE, 'posterior': 'original VAE latent_dist.mode()',
            'VAE_device_dtype': 'cuda:0/torch.float32', 'inference_device': 'cuda:0',
            'model_co_resident': True, 'no_model_offload': True, 'training_updates': 0,
            'dataset_pixels_opened': False, 'quality_scores_retained': False,
            'full_resource_probe_passed': False, 'sampler_calls': 0, 'caption_encoder_or_embedding_load': False,
            'scope': 'one whole-HRI T1 encode only; no tiling, sampling, SR decode, MUSIQ or quality selection',
            'limits': ['fresh process also removes earlier NoRef allocation history; success does not uniquely prove fragmentation',
                       'single4GB attention matrix is a lower bound, not total encode peak/capacity upperbound',
                       'own task CPU state may be deserialized by frozen evaluation loader, but optimizer/scheduler/RNG are never restored',
                       'measurement/state hashing overhead included; no full resource, native QA or test success claim'],
            'GPU_guard': guard}
        contract = _digest(protocol)
        output(run/'protocol.json', protocol)
        output(run/'status.json', {'status': 'running', 'protocol_sha256': contract,
                                  'full_resource_probe_passed': False})
        measurements = _Measurements(run, [torch.device('cuda:0')])
        phase = 'pre_model_guard'
        output(run/'GPU_guard_before_load.json', assert_gpu_idle(P))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                phase = 'co_resident_model_load'
                model, loaded = measurements.call(phase, lambda: _default_model(CHECKPOINT, device='cuda', allowed_root=P/'runs'))
                require(loaded['checkpoint_sha256'] == CHECKPOINT_SHA and loaded['step'] == 8192
                        and loaded['variant'] == 'Both' and loaded['config'] == config, 'Actual model loader origin mismatch')
                output(run/'model_load_audit.json', loaded)
                phase = 'co_resident_FP32_VAE_load'
                vae = measurements.call(phase, lambda: _default_vae(config, device='cuda'))
                model.eval(); vae.requires_grad_(False).eval()
                require(not vae.use_tiling and not vae.use_slicing, 'Original dense VAE path required')
                require(all(p.device.type == 'cuda' for p in model.parameters())
                        and all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in vae.parameters()),
                        'Whole model+FP32 VAE must remain CUDA co-resident')
                observations['allocator_before_encode'] = allocator_observation(torch)
                output(run/'allocator_before_encode.json', observations['allocator_before_encode'])
                require(observations['allocator_before_encode']['backend'] == 'native'
                        and observations['allocator_before_encode'].get('actual_expandable_segment_observed') is True,
                        'Effective native expandable segments not observed; no silent allocator fallback')
                networks = [('model', model), ('vae', vae)]
                flags = [(module, module.training) for _, network in networks for module in network.modules()]
                before_grads = _grads(networks)
                before_state = {name: _network_state(network) for name, network in networks}
                device_maps_before = {name: device_dtype_map(network) for name, network in networks}
                rng_before = _rng_snapshot()
                require(_contexts_empty(model), 'Outstanding model context before isolated encode')
                observations['state_before'] = before_state
                observations['VAE_cache_before'] = cache_observation(vae)
                observations['devices_dtypes_before'] = device_maps_before
                phase = 'global_HRI_T1_encode'
                @torch.no_grad()
                def encode():
                    normalized = input_cpu.to(device='cuda:0', dtype=torch.float32)
                    require(_tensor_sha(normalized) == INPUT_SHA, 'GPU normalized input differs from bound bytes')
                    return vae.encode(normalized).latent_dist.mode()
                raw = measurements.call(phase, encode)
                require(list(raw.shape) == OUTPUT_SHAPE and raw.dtype == torch.float32
                        and raw.device.type == 'cuda' and not raw.requires_grad and bool(torch.isfinite(raw).all()),
                        'Original VAE raw latent is nonfinite or changed expected dtype/geometry')
                observations['raw_latent'] = {'shape': list(raw.shape), 'dtype': str(raw.dtype),
                    'device': str(raw.device), 'finite': True, 'sha256': _tensor_sha(raw)}
            finally:
                warnings_seen.extend({'message': str(item.message), 'category': item.category.__name__} for item in caught)
        phase = 'after_encode_audit'
        observations['state_after'] = {name: _network_state(network) for name, network in networks}
        observations['devices_dtypes_after'] = {name: device_dtype_map(network) for name, network in networks}
        checks = {'all_parameters_buffers_unchanged': before_state == observations['state_after'],
            'all_devices_dtypes_unchanged_no_offload': device_maps_before == observations['devices_dtypes_after'],
            'module_training_flags_unchanged': all(module.training == flag for module, flag in flags),
            'parameter_gradients_requires_grad_identity_unchanged': _grads_equal(before_grads, networks),
            'global_RNG_unchanged': _rng_equal(rng_before, _rng_snapshot()), 'model_contexts_empty': _contexts_empty(model)}
        observations.update(checks=checks, allocator_after_encode=allocator_observation(torch),
                            VAE_cache_after=cache_observation(vae), warnings=warnings_seen)
        require(all(checks.values()), 'State/RNG/flags/gradients/context changed during isolated encoding')
        require(all(row['tensor_entries'] == 0 for row in observations['VAE_cache_after'].values()),
                'Official successful encode left VAE temporary caches populated')
        require(sha(CHECKPOINT) == CHECKPOINT_SHA, 'Checkpoint bytes changed during probe')
        summary = {'status': 'passed', 'protocol_sha256': contract, 'observations': observations,
            'phase_observations_sha256': sha(run/'phase_observations.json'),
            'GPU_memory_after_and_encode_peak': _gpu_memory([torch.device('cuda:0')], peaks=True),
            'host_lifetime_memory': _host_memory(), 'wall_seconds': time.perf_counter()-started,
            'wall_scope': 'worker entry through final observation, including CPU metadata/load/state hashes; excludes interpreter/import startup and final serialization/cleanup',
            'training_updates': 0, 'dataset_pixels_opened': False, 'quality_scores_retained': False,
            'full_resource_probe_passed': False, 'model_co_resident': True, 'no_model_offload': True,
            'sampler_calls': 0, 'encode_feasible_in_this_fresh_process': True,
            'fragmentation_uniquely_proven': False, 'capacity_upperbound_claimed': False,
            'runtime_reproducibility_flags': reproducibility_flags()}
        output(run/'summary.json', summary)
        output(run/'status.json', {'status': 'passed', 'protocol_sha256': contract,
            'summary_sha256': sha(run/'summary.json'), 'full_resource_probe_passed': False})
        return summary
    except Exception as error:
        failure = {'status': 'failed', 'protocol_sha256': contract, 'phase': phase,
                   'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
                   'partial_observations': observations, 'warnings': warnings_seen,
                   'wall_seconds': time.perf_counter()-started, 'full_resource_probe_passed': False,
                   'training_updates': 0, 'dataset_pixels_opened': False, 'quality_scores_retained': False,
                   'no_model_offload': True, 'no_silent_fallback_or_retry': True}
        if gpu_memory is not None:
            failure['GPU_memory_before_cleanup_and_encode_peak'] = gpu_memory([torch.device('cuda:0')], peaks=True)
            failure['host_lifetime_memory'] = host_memory()
            try:
                failure['allocator_before_cleanup'] = allocator_observation(torch)
            except Exception as observe_error:
                failure['allocator_observation_error'] = str(observe_error)
        if vae is not None:
            failure['VAE_cache_before_failure_cleanup'] = cache_observation(vae)
            try:
                vae.clear_cache()  # official temporary-cache cleanup, never another encode
                failure['VAE_cache_after_failure_cleanup'] = cache_observation(vae)
            except Exception as cleanup_error:
                failure['VAE_cache_cleanup_error'] = str(cleanup_error)
        if networks is not None and rng_before is not None:
            try:
                after = {name: _network_state(network) for name, network in networks}
                failure['state_after_failure'] = after
                failure['cleanup_checks'] = {'all_parameters_buffers_unchanged': before_state == after,
                    'all_devices_dtypes_unchanged_no_offload': device_maps_before == {name: device_dtype_map(network) for name, network in networks},
                    'module_training_flags_unchanged': all(module.training == flag for module, flag in flags),
                    'parameter_gradients_requires_grad_identity_unchanged': _grads_equal(before_grads, networks),
                    'global_RNG_unchanged': _rng_equal(rng_before, _rng_snapshot()), 'model_contexts_empty': _contexts_empty(model)}
            except Exception as cleanup_error:
                failure['cleanup_audit_error'] = str(cleanup_error)
        output(run/'failure_observations.json', failure)
        output(run/'status.json', {'status': 'failed', 'protocol_sha256': contract,
            'error': failure['error'], 'phase': phase, 'wall_seconds': failure['wall_seconds'],
            'failure_observations_sha256': sha(run/'failure_observations.json'),
            'full_resource_probe_passed': False, 'no_silent_fallback_or_retry': True})
        raise
    finally:
        # Preserve all observations first. No retry, device/offload, precision,
        # SDPA/shape change, global marker or service operation occurs here.
        raw = model = vae = networks = flags = before_grads = None
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir)
    print(json.dumps({key: result[key] for key in ('status', 'protocol_sha256', 'full_resource_probe_passed')}))


if __name__ == '__main__':
    main()
