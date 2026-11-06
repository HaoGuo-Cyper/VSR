"""Explicit metadata-only external startup guard, full inventory/no filtering.

The frozen producer guard is NOT invoked. Marker/unit checks are adapted from
its SHA/AST-bound source; full three-column inventory is directly classified by
the pinned monitor classifier. Import does not query processes, GPU or units.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

W = Path(__file__).resolve().parents[1]
CLASSIFIER = W/'work/gpu_process_classification.py'
CLASSIFIER_SHA = '0f98c8b717c0f0528042d2abcbb38db0b7798b44906707b9b83133f7bd7d42d2'
ORIGINAL_SOURCE = Path('/home/hugo/Projects/VSR/vsr/reference_diagnostics.py')
ORIGINAL_SOURCE_SHA = 'aecbb4a5cec30ea24e2c4f75f82b257e762b0c5842bf18fc412e64c6a7a9cfa0'
ORIGINAL_FUNCTION_AST_SHA = '6430f9bee4188371cf3f1567ddfc027f7267349686cb7db3eee92b2f74b9bd6a'
SCHEMA = 'vsr_external_startup_GPU_guard_v2'
QUERY = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']
UNIT_PROPERTIES = ['LoadState', 'ActiveState', 'SubState', 'MainPID', 'InvocationID']
MARKERS = ('active_training_service.json', 'active_reference_training.json', 'active_training.json')
ACTIVE_STATES = {'active', 'activating', 'deactivating', 'reloading'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    value = json.loads(Path(path).read_bytes())
    require(isinstance(value, dict), 'Expected marker JSON object')
    return value


def binding():
    return {'helper': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'classifier': {'path': str(CLASSIFIER), 'sha256': CLASSIFIER_SHA}, 'schema': SCHEMA,
        'full_inventory_no_filter': True, 'legacy_producer_guard_invoked': False}


def verify_bindings(value):
    require(value == binding(), 'Root startup helper/classifier policy binding differs')
    require(sha(CLASSIFIER) == CLASSIFIER_SHA, 'Pinned strict classifier source changed')
    require(sha(ORIGINAL_SOURCE) == ORIGINAL_SOURCE_SHA, 'Original copied metadata reference source changed')
    original = next(n for n in ast.parse(ORIGINAL_SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'assert_gpu_idle')
    # Python3.14 changed ast.dump's default empty-list rendering. Preserve the
    # full3.12 form explicitly where supported; source bytes are pinned too.
    try:
        original_ast = ast.dump(original, include_attributes=False, show_empty=True)
    except TypeError:
        original_ast = ast.dump(original, include_attributes=False)
    require(hashlib.sha256(original_ast.encode()).hexdigest()
        == ORIGINAL_FUNCTION_AST_SHA, 'Original copied metadata reference AST changed')
    return value


def classifier_module():
    require(sha(CLASSIFIER) == CLASSIFIER_SHA, 'Pinned strict classifier source changed')
    spec = importlib.util.spec_from_file_location('external_startup_strict_classifier_v3', CLASSIFIER)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def query_inventory(classifier, *, runner, proc_reader, own_pid, own_verified, uid, unit):
    observation = {'argv': list(QUERY), 'stdout': None, 'stderr': None, 'returncode': None,
        'raw_rows': [], 'classification': None, 'exception': None}
    try:
        response = runner(list(QUERY), capture_output=True, text=True)
        observation.update(stdout=response.stdout, stderr=response.stderr, returncode=response.returncode)
        observation['raw_rows'] = response.stdout.splitlines()
        observation['classification'] = classifier.classify_inventory(observation['raw_rows'],
            own_pid=own_pid, own_worker_verified=own_verified, expected_uid=uid,
            expected_worker_unit=unit, proc_reader=proc_reader, query_success=response.returncode == 0)
    except Exception as error:
        observation['exception'] = type(error).__name__ + ': ' + str(error)
    observation['verified'] = bool(observation['classification'] and
        observation['classification']['all_inventory_processes_verified_allowed'])
    return observation


def base_observation(mode):
    return {'schema': SCHEMA, 'mode': mode, 'status': 'failed', 'startup_gpu_guard': binding(),
        'full_inventory_no_filter': True, 'legacy_producer_guard_invoked': False,
        'original_metadata_reference': {'path': str(ORIGINAL_SOURCE), 'sha256': ORIGINAL_SOURCE_SHA,
            'function': 'assert_gpu_idle', 'function_AST_sha256': ORIGINAL_FUNCTION_AST_SHA,
            'scope': 'marker/known-unit checks copied and adapted; original function not invoked'},
        'errors': [], 'inventory': None, 'own_worker_identity': None,
        'own_diagnostic_service_units': []}


def launcher_guard(*, runner=subprocess.run, proc_reader=None, expected_uid=None):
    """No own-PID allowance: only verified small desktop processes may remain."""
    started = time.perf_counter(); result = base_observation('launcher_no_model_allowed')
    try:
        verify_bindings(result['startup_gpu_guard']); classifier = classifier_module()
        uid = os.getuid() if expected_uid is None else expected_uid
        result['inventory'] = query_inventory(classifier, runner=runner,
            proc_reader=classifier.read_process_info if proc_reader is None else proc_reader,
            own_pid=None, own_verified=False, uid=uid, unit=None)
        require(result['inventory']['verified'], 'Full GPU inventory contains unverified/competing process or query failure')
        result['status'] = 'passed'
    except Exception as error:
        result['errors'].append(type(error).__name__ + ': ' + str(error))
    result['wall_seconds'] = time.perf_counter() - started
    result['wall_scope'] = 'metadata/source/proc/inventory guard only; no model or GPU computation'
    return result


def unit_query(unit, runner):
    argv = ['systemctl', '--user', 'show', unit]
    for key in UNIT_PROPERTIES:
        argv.extend(['-p', key])
    result = runner(argv, capture_output=True, text=True)
    props = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    return {'argv': argv, 'stdout': result.stdout, 'stderr': result.stderr,
        'returncode': result.returncode, 'properties': props}


def proc_bytes(pid, name):
    return (Path('/proc')/str(pid)/name).read_bytes()


def worker_guard(project_root, *, expected_unit, worker_path, worker_sha256, run_dir,
        phase, root_protocol_sha256, startup_environment, runner=subprocess.run,
        proc_reader=None, proc_bytes_reader=proc_bytes):
    """Verify this direct service using live identity, not late marker fields.

    A launching marker is already registered before systemd-run. Its actual PID
    and Invocation fields may legitimately appear only after this worker writes
    its protocol; obtain them from the real unit/proc now to avoid a deadlock.
    Dependencies are explicit for fake CPU tests; real worker passes no overrides.
    """
    started = time.perf_counter(); result = base_observation('registered_direct_worker')
    pid, uid, root = os.getpid(), os.getuid(), Path(project_root).resolve()
    marker_rows, units, unit_states = {}, {'vsr-training.service'}, {}
    result.update(training_markers_checked=[], known_units_checked=[], unit_states=unit_states)
    implementation_valid = False; classifier = None; service = None; identity = {'verified': False, 'PID': pid,
        'InvocationID': None, 'checks': {}, 'process_metadata': None}
    result['own_worker_identity'] = identity
    try:
        verify_bindings(result['startup_gpu_guard']); classifier = classifier_module(); implementation_valid = True
        require(Path(worker_path).resolve() == Path(worker_path) and sha(worker_path) == worker_sha256,
            'Actual declared new worker path/source changed')
        for name in MARKERS:
            path = root/'runs'/name
            if path.exists():
                row = read(path); marker_rows[name] = row
                if row.get('unit'): units.add(row['unit'])
                if row.get('status') == 'running' and (name != 'active_training_service.json' or not row.get('unit')):
                    result['errors'].append('Active '+name+'; reconcile training before external startup')
        service = marker_rows.get('active_training_service.json')
        result['training_markers_checked'] = sorted(marker_rows)
        require(isinstance(service, dict), 'Register the actual new direct service before worker startup')
        checks = identity['checks']
        checks['registered_launching_or_running'] = service.get('status') in {'launching', 'running'}
        checks['same_declared_unit_run_phase_and_source'] = (service.get('unit') == expected_unit
            and service.get('run') == str(Path(run_dir).resolve()) and service.get('phase') == phase
            and service.get('worker_sha256') == worker_sha256
            and service.get('root_protocol_file_sha256') == root_protocol_sha256
            and service.get('module') == 'workspace.'+Path(worker_path).stem)
        checks['same_declared_startup_environment'] = service.get('launch_environment') == startup_environment
        checks['same_declared_startup_guard_binding'] = service.get('startup_gpu_guard') == result['startup_gpu_guard']
        command = service.get('command', [])
        require(isinstance(command, list) and command.count(str(worker_path)) == 1, 'Launching command lacks exact new worker argv token')
        index = command.index(str(worker_path))
        require(index >= 2 and command[index-1] == '-u', 'Direct Python -u newworker command required')
        expected_argv = command[index-2:]
        for unit in sorted(units):
            observation = unit_query(unit, runner); unit_states[unit] = observation
            props = observation['properties']
            if observation['returncode'] and props.get('LoadState') != 'not-found':
                result['errors'].append('Cannot verify known service: '+unit)
            if not (props.get('LoadState') in {'loaded', 'not-found'}
                    and props.get('ActiveState') in ACTIVE_STATES | {'inactive', 'failed'}
                    and isinstance(props.get('MainPID'), str) and props['MainPID'].isdigit()
                    and (props['ActiveState'] in ACTIVE_STATES or props['MainPID'] == '0')):
                result['errors'].append('Incomplete/inconsistent known-service metadata: '+unit)
            if props.get('ActiveState') in ACTIVE_STATES:
                if not (unit == expected_unit and props.get('ActiveState') == 'active' and props.get('MainPID') == str(pid)):
                    result['errors'].append('Another/transitioning known GPU service is active: '+unit)
        result['known_units_checked'] = sorted(units)
        own = unit_states.get(expected_unit, {}).get('properties', {})
        invocation = own.get('InvocationID'); identity['InvocationID'] = invocation
        checks['live_loaded_active_running_direct_MainPID_and_Invocation'] = (own.get('LoadState') == 'loaded'
            and own.get('ActiveState') == 'active' and own.get('SubState') == 'running'
            and own.get('MainPID') == str(pid) and isinstance(invocation, str) and bool(invocation.strip()))
        info = (classifier.read_process_info if proc_reader is None else proc_reader)(pid)
        identity['process_metadata'] = info
        own_proof = classifier.verify_own_worker(info, expected_uid=uid, expected_unit=expected_unit)
        checks['actual_proc_metadata_PID_matches_this_process'] = type(info.get('pid')) is int and info['pid'] == pid
        checks['actual_proc_UID_python_inode_and_exact_unit_cgroup'] = own_proof['verified']
        raw_command = proc_bytes_reader(pid, 'cmdline')
        actual_argv = [part.decode() for part in raw_command.split(b'\0') if part]
        identity['actual_command_argv'] = actual_argv
        checks['exact_direct_command_and_new_worker_path'] = actual_argv == expected_argv
        initial = dict(part.split(b'=', 1) for part in proc_bytes_reader(pid, 'environ').split(b'\0') if b'=' in part)
        checks['actual_required_startup_environment'] = all(initial.get(key.encode()) == value.encode()
            for key, value in startup_environment.items())
        if 'actual_direct_worker_PID' in service:
            checks['published_PID_if_present_matches_live'] = service['actual_direct_worker_PID'] == pid
        if isinstance(service.get('properties'), dict) and service['properties'].get('InvocationID'):
            checks['published_Invocation_if_present_matches_live'] = service['properties']['InvocationID'] == invocation
        identity['verified'] = bool(checks) and all(checks.values()) and not result['errors']
        if not identity['verified']: result['errors'].append('Actual own direct-service identity not jointly verified')
        if identity['verified']: result['own_diagnostic_service_units'] = [expected_unit]
    except Exception as error:
        result['errors'].append(type(error).__name__ + ': ' + str(error))
    # Preserve complete inventory even when marker/unit/ownership failed. A
    # failed owner proof never grants this PID an exception in classification.
    if implementation_valid:
        result['inventory'] = query_inventory(classifier, runner=runner,
            proc_reader=classifier.read_process_info if proc_reader is None else proc_reader,
            own_pid=pid, own_verified=identity['verified'], uid=uid, unit=expected_unit)
        if not result['inventory']['verified']:
            result['errors'].append('Full GPU inventory contains unverified/competing process or query failure')
    result['status'] = 'passed' if identity['verified'] and not result['errors'] and result['inventory']['verified'] else 'failed'
    result['wall_seconds'] = time.perf_counter() - started
    result['wall_scope'] = 'new explicit marker/unit/direct-owner/proc/full-inventory guard; no legacy guard/model/GPU computation'
    return result


def ensure_passed(observation):
    inventory = observation.get('inventory') or {}
    classification = inventory.get('classification') or {}
    require(observation.get('schema') == SCHEMA and observation.get('status') == 'passed'
        and observation.get('full_inventory_no_filter') is True
        and observation.get('legacy_producer_guard_invoked') is False
        and inventory.get('verified') is True
        and classification.get('all_inventory_processes_verified_allowed') is True
        and (observation.get('mode') == 'registered_direct_worker'
            and (observation.get('own_worker_identity') or {}).get('verified') is True
            or observation.get('mode') == 'launcher_no_model_allowed'
            and classification.get('own_PID') is None
            and classification.get('own_worker_separately_verified') is False),
        'External startup guard failed; preserve full observation and reconcile')
