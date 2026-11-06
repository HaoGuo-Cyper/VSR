"""Fail-closed, monitor-only classification of real GPU inventory rows.

No GPU/service command is issued here. read_process_info reads OS metadata
only; fixtures inject fake /proc dictionaries. These three desktop categories
were verified on this host. Basenames alone never grant an allowance.
"""
from pathlib import Path
import csv
import math
import os
import re
import stat

DESKTOP_MEMORY_LIMIT_MIB = 256
DESKTOP_EXECUTABLES = {'/usr/bin/gjs-console': 'gnome_shell_gjs_console',
                       '/usr/bin/nautilus': 'gnome_nautilus',
                       '/usr/bin/ptyxis': 'gnome_ptyxis'}
OWN_PYTHON_EXE = '/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12'


def read_process_info(pid, *, proc_root=Path('/proc')):
    """Read executable identity, all four UIDs and unified cgroup, no values/RNG."""
    base = Path(proc_root)/str(pid)
    result = {'pid': pid}
    try:
        result['proc_directory_uid'] = base.stat().st_uid
        result['exe'] = os.readlink(base/'exe')
        uid_rows = [row for row in (base/'status').read_text().splitlines() if row.startswith('Uid:')]
        if len(uid_rows) != 1:
            raise ValueError('Expected one Uid row')
        uid_values = uid_rows[0].split()[1:]
        if len(uid_values) != 4 or not all(re.fullmatch(r'\d+', x) for x in uid_values):
            raise ValueError('Expected real/effective/saved/filesystem UIDs')
        result['uid_fields'] = [int(x) for x in uid_values]
        records = []
        for row in (base/'cgroup').read_text().splitlines():
            parts = row.split(':', 2)
            if len(parts) != 3:
                raise ValueError('Malformed cgroup metadata')
            records.append({'hierarchy': parts[0], 'controllers': parts[1], 'path': parts[2]})
        result['cgroups'] = records
        if result['exe'] in (*DESKTOP_EXECUTABLES, OWN_PYTHON_EXE):
            proc_exe, system_exe = (base/'exe').stat(), Path(result['exe']).stat()
            result['executable_identity'] = {
                'same_system_inode': (proc_exe.st_dev, proc_exe.st_ino) == (system_exe.st_dev, system_exe.st_ino),
                'system_uid': system_exe.st_uid, 'system_mode': stat.S_IMODE(system_exe.st_mode),
                'system_regular_file': stat.S_ISREG(system_exe.st_mode)}
        result['read_error'] = None
    except (OSError, ValueError) as error:
        result['read_error'] = type(error).__name__ + ': ' + str(error)
    return result


def _same_user(info, uid):
    return (isinstance(uid, int) and not isinstance(uid, bool) and uid >= 0
        and info.get('proc_directory_uid') == uid and info.get('uid_fields') == [uid]*4)


def _unified_cgroup(info):
    rows = info.get('cgroups')
    if (isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict) and rows[0].get('hierarchy') == '0'
            and rows[0].get('controllers') == '' and isinstance(rows[0].get('path'), str)):
        return rows[0]['path']
    return None


def _user_prefix(uid):
    return f'/user.slice/user-{uid}.slice/user@{uid}.service'


def verify_own_worker(info, *, expected_uid, expected_unit):
    """Additional metadata gate; helper separately keeps PID/cmdline/env/Invocation checks."""
    identity = info.get('executable_identity', {})
    identity = identity if isinstance(identity, dict) else {}
    checks = {'proc_readable': info.get('read_error') is None,
        'same_real_effective_saved_fs_and_directory_UID': _same_user(info, expected_uid),
        'exact_worker_python_executable': info.get('exe') == OWN_PYTHON_EXE,
        'actual_executable_inode_matches_declared_file': identity.get('same_system_inode') is True
            and identity.get('system_regular_file') is True,
        'exact_same_user_service_cgroup': _unified_cgroup(info) == _user_prefix(expected_uid)+'/app.slice/'+expected_unit}
    return {'verified': all(checks.values()), 'checks': checks, 'process': info,
        'scope': 'additional own-worker metadata; original MainPID/Invocation/cmdline/startup-env checks remain required'}


def _desktop_cgroup_matches(category, cgroup, uid, *, inventory_pid=None, process_pid=None):
    if not isinstance(cgroup, str):
        return False
    prefix = re.escape(_user_prefix(uid))
    if category == 'gnome_shell_gjs_console':
        # This exact GNOME Shell session unit was observed on this host.
        return re.fullmatch(prefix+r'/session\.slice/org\.gnome\.Shell@ubuntu\.service', cgroup) is not None
    if category == 'gnome_nautilus':
        # DBus endpoint/instance numbers are dynamic, but parent and child bus
        # endpoint must be the SAME. \x2d is a literal systemd path escape.
        pattern = (prefix+r'/app\.slice/app-dbus\\x2d:(?P<bus>\d+\.\d+)\\x2dorg\.gnome\.Nautilus\.slice/'
            r'dbus-:(?P=bus)-org\.gnome\.Nautilus@\d+\.service')
        return re.fullmatch(pattern, cgroup) is not None
    if category == 'gnome_ptyxis':
        # Exact observed GNOME xdg-terminal-exec scope, with literal systemd
        # \x2d escapes. The scope PID must equal both inventory and /proc PID.
        if (not isinstance(inventory_pid, int) or isinstance(inventory_pid, bool)
                or inventory_pid <= 0 or not isinstance(process_pid, int)
                or isinstance(process_pid, bool) or process_pid != inventory_pid):
            return False
        pattern = (prefix+r'/app\.slice/app-gnome-xdg\\x2dterminal\\x2dexec-'
            +str(inventory_pid)+r'\.scope')
        return re.fullmatch(pattern, cgroup) is not None
    return False


def classify_inventory(rows, *, own_pid, own_worker_verified, expected_uid,
                       proc_reader=read_process_info, query_success=True,
                       expected_worker_unit=None,
                       desktop_memory_limit_mib=DESKTOP_MEMORY_LIMIT_MIB):
    """Keep every raw row; unknown/unreadable/oversized processes block confirmation."""
    if desktop_memory_limit_mib != DESKTOP_MEMORY_LIMIT_MIB:
        raise ValueError('Desktop policy is fixed at256MiB; no runtime broadening')
    observations = []
    for raw in rows:
        if not raw.strip():
            continue
        observation = {'raw_inventory_row': raw, 'allowed': False, 'classification': 'unverified_inventory_row', 'reasons': []}
        observations.append(observation)
        try:
            fields = next(csv.reader([raw], skipinitialspace=True))
            if len(fields) != 3 or not re.fullmatch(r'[1-9]\d*', fields[0].strip()):
                raise ValueError('Expected positive PID,process_name,used_memory CSV row')
            pid = int(fields[0].strip()); name = fields[1].strip(); memory_text = fields[2].strip()
            memory = re.fullmatch(r'(\d+(?:\.\d+)?)\s+MiB', memory_text)
            if memory is None:
                raise ValueError('Used GPU memory must be explicitly readable inMiB')
            used = float(memory.group(1))
            if not math.isfinite(used):
                raise ValueError('Used GPU memory must be finite')
            observation.update(pid=pid, reported_process_name=name, used_memory_MiB=used)
            try:
                info = proc_reader(pid)
            except Exception as error:
                info = {'pid': pid, 'read_error': type(error).__name__+': '+str(error)}
            observation['process_metadata'] = info
            if not isinstance(info, dict) or info.get('read_error') is not None:
                observation['classification'] = 'unreadable_process_metadata'
                observation['reasons'].append('Actual /proc identity could not be verified')
                continue
            if not _same_user(info, expected_uid):
                observation['classification'] = 'unverified_or_other_UID'
                observation['reasons'].append('All process UIDs and /proc owner must match current user')
                continue
            if pid == own_pid:
                fresh = (verify_own_worker(info, expected_uid=expected_uid, expected_unit=expected_worker_unit)
                    if isinstance(expected_worker_unit, str) and expected_worker_unit else {'verified': False})
                observation['own_fresh_identity'] = fresh
                observation['allowed'] = own_worker_verified is True and fresh['verified'] is True
                observation['classification'] = 'own_VSR_worker' if observation['allowed'] else 'unverified_own_PID'
                if not observation['allowed']:
                    observation['reasons'].append('Own unit/PID/Invocation/cmdline/env/cgroup not jointly verified')
                continue
            category = DESKTOP_EXECUTABLES.get(info.get('exe'))
            identity = info.get('executable_identity', {})
            identity = identity if isinstance(identity, dict) else {}
            executable_ok = (category is not None and identity.get('same_system_inode') is True
                and identity.get('system_regular_file') is True and identity.get('system_uid') == 0
                and isinstance(identity.get('system_mode'), int) and not isinstance(identity.get('system_mode'), bool)
                and identity['system_mode'] & 0o022 == 0)
            cgroup_ok = _desktop_cgroup_matches(category, _unified_cgroup(info), expected_uid,
                inventory_pid=pid, process_pid=info.get('pid'))
            memory_ok = used <= DESKTOP_MEMORY_LIMIT_MIB
            observation['desktop_checks'] = {'exact_verified_system_executable': executable_ok,
                'same_user_known_GNOME_cgroup': cgroup_ok, 'at_most_256MiB': memory_ok}
            if executable_ok and cgroup_ok and memory_ok:
                observation.update(allowed=True, classification='verified_desktop_'+category)
            else:
                observation['classification'] = 'unverified_or_competing_process'
                observation['reasons'].extend(name for name, value in observation['desktop_checks'].items() if not value)
        except (ValueError, TypeError, IndexError, csv.Error) as error:
            observation['reasons'].append(type(error).__name__+': '+str(error))
    all_allowed = query_success is True and all(row['allowed'] for row in observations)
    return {'schema': 'vsr_monitor_GPU_process_classification_v1', 'query_success': query_success is True,
        'expected_UID': expected_uid, 'own_PID': own_pid, 'own_worker_separately_verified': own_worker_verified is True,
        'desktop_memory_limit_MiB': DESKTOP_MEMORY_LIMIT_MIB, 'observations': observations,
        'all_inventory_processes_verified_allowed': all_allowed,
        'no_concurrent_model_job_confirmed': all_allowed,
        'unknown_or_competing_inventory_requires_reconciliation': not all_allowed,
        'scope': 'monitor-only: own independently verified VSR PID or three verified small same-user GNOME categories; Ptyxis scope PID matches inventory and proc PID; no basename allowlist; frozen startup/worker guards unchanged'}
