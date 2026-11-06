"""PROPOSED ONLY: one pinned gjs plus original verified Nautilus/Ptyxis.

This scratch candidate is not imported by any frozen worker. Nothing runs on
import; CPU fixtures inject captured inventory and process metadata. Adoption
requires the human's choice and a fresh protocol/run, never retrying the failed
T2 run in place. The unchanged original classifier remains authoritative for
UID, exact system executable/inode, GNOME cgroup and <=256MiB per process.

Policy difference: the original guard requires EVERY returned row to match
the pinned gjs (and all([]) accepts an empty inventory). This candidate requires
EXACTLY ONE matching pinned gjs, unique PIDs, and permits additional rows only
when the original classifier already verifies Nautilus or Ptyxis. An unpinned
second gjs is rejected even if that monitor classifier permits its category.
"""
from __future__ import annotations

from pathlib import Path
import subprocess


PINNED_GJS = 'verified_desktop_gnome_shell_gjs_console'
EXTRA_DESKTOPS = frozenset(('verified_desktop_gnome_nautilus','verified_desktop_gnome_ptyxis'))


def process_start_ticks(pid):
    return int((Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19])


def gpu_idle_guard(desktop,classifier,*,runner=subprocess.run,proc_reader=None,start_reader=process_start_ticks):
    """Same query/classifier as the frozen guard, with the proposed row policy.

    No basename/process-name exemptions: all extra-process identity and memory
    evidence must come from the unchanged original classifier. This function
    only observes inventory; it never loads CUDA, changes or kills a process.
    """
    result = runner(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                     '--format=csv,noheader'],capture_output=True,text=True,timeout=20)
    classification = classifier.classify_inventory(result.stdout.splitlines(),own_pid=None,own_worker_verified=False,
        expected_uid=desktop['UID'],proc_reader=proc_reader or classifier.read_process_info,query_success=result.returncode==0)
    pinned,permitted,identifiers = [],[],[]
    for row in classification['observations']:
        try:
            matches = (row.get('allowed') is True and row.get('classification')==PINNED_GJS
                and row.get('pid')==desktop['PID'] and row.get('reported_process_name')==desktop['reported_name']
                and start_reader(row['pid'])==desktop['start_ticks']
                and row.get('process_metadata')==desktop['process_metadata'])
        except (OSError,ValueError):matches=False
        pinned.append(matches)
        permitted.append(matches or (row.get('allowed') is True and row.get('classification') in EXTRA_DESKTOPS))
        identifiers.append(row.get('pid'))
    unique_pids = (all(type(pid) is int and pid>0 for pid in identifiers)
                   and len(set(identifiers))==len(identifiers))
    anchor_count = sum(pinned)
    passed = (classification['all_inventory_processes_verified_allowed'] is True
              and classification['desktop_memory_limit_MiB']==256
              and desktop.get('memory_limit_MiB')==256
              and unique_pids and anchor_count==1 and all(permitted))
    return {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'classification':classification,
        'pinned_desktop_matches':pinned,'pinned_gjs_match_count':anchor_count,'unique_inventory_PIDs':unique_pids,
        'permitted_pinned_or_verified_extra_desktop':permitted,
        'original_gjs_only_rule_would_pass':classification['all_inventory_processes_verified_allowed'] and all(pinned),
        'passed':passed,'proposal_only':True,
        'policy':'exactly one complete pinned gjs; extras only original verified Nautilus/Ptyxis; reject duplicate PIDs, other gjs, unknown/ML/high-memory/unreadable rows'}
