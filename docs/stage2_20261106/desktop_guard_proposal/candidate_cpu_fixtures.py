"""CPU-only captured/fake fixtures; never query GPU or read live /proc."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import pinned_gjs_desktop_guard_candidate as candidate


ROOT = Path(__file__).resolve().parents[1]/'VSR-sync'
CLASSIFIER = ROOT/'scripts/frozen_external/work/gpu_process_classification.py'
CLASSIFIER_SHA = '0f98c8b717c0f0528042d2abcbb38db0b7798b44906707b9b83133f7bd7d42d2'
GJS,NAUTILUS,PTYXIS = 4816,1652088,1652090


def process(pid,exe,cgroup):
    return {'pid':pid,'proc_directory_uid':1000,'exe':exe,'uid_fields':[1000]*4,
        'cgroups':[{'hierarchy':'0','controllers':'','path':cgroup}],
        'executable_identity':{'same_system_inode':True,'system_uid':0,'system_mode':0o755,'system_regular_file':True},
        'read_error':None}


def metadata():
    prefix='/user.slice/user-1000.slice/user@1000.service'
    return {GJS:process(GJS,'/usr/bin/gjs-console',prefix+'/session.slice/org.gnome.Shell@ubuntu.service'),
        NAUTILUS:process(NAUTILUS,'/usr/bin/nautilus',prefix+'/app.slice/app-dbus\\x2d:1.1\\x2dorg.gnome.Nautilus.slice/dbus-:1.1-org.gnome.Nautilus@8.service'),
        PTYXIS:process(PTYXIS,'/usr/bin/ptyxis',prefix+f'/app.slice/app-gnome-xdg\\x2dterminal\\x2dexec-{PTYXIS}.scope')}


class CandidateGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(CLASSIFIER.read_bytes()).hexdigest()!=CLASSIFIER_SHA:
            raise AssertionError('Fixtures require the unchanged original classifier')
        spec=importlib.util.spec_from_file_location('original_frozen_classifier',CLASSIFIER)
        cls.classifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.classifier)

    def setUp(self):
        self.info=metadata()
        self.desktop={'PID':GJS,'reported_name':'gjs','UID':1000,'start_ticks':1771,
            'process_metadata':copy.deepcopy(self.info[GJS]),'memory_limit_MiB':256}

    def check(self,stdout,*,info=None,tick=1771,code=0):
        info=self.info if info is None else info
        calls=[]
        def runner(command,**kwargs):
            calls.append((command,kwargs))
            return subprocess.CompletedProcess(command,code,stdout,'captured stderr')
        result=candidate.gpu_idle_guard(self.desktop,self.classifier,runner=runner,
            proc_reader=lambda pid:info.get(pid,{'pid':pid,'read_error':'fixture missing metadata'}),
            start_reader=lambda _:tick)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['stdout'],stdout)
        self.assertEqual(result['stderr'],'captured stderr')
        return result

    def test_single_bound_gjs_is_still_allowed(self):
        result=self.check(f'{GJS}, gjs, 64 MiB\n')
        self.assertTrue(result['passed']);self.assertTrue(result['original_gjs_only_rule_would_pass'])
        self.assertEqual(result['pinned_gjs_match_count'],1)

    def test_verified_nautilus_and_ptyxis_are_only_additional_categories(self):
        for extra in (f'{NAUTILUS}, /usr/bin/nautilus, 72 MiB',f'{PTYXIS}, /usr/bin/ptyxis, 24 MiB',
                      f'{NAUTILUS}, /usr/bin/nautilus, 72 MiB\n{PTYXIS}, /usr/bin/ptyxis, 24 MiB'):
            with self.subTest(extra=extra):
                result=self.check(f'{GJS}, gjs, 64 MiB\n'+extra+'\n')
                self.assertTrue(result['classification']['all_inventory_processes_verified_allowed'])
                self.assertTrue(result['passed']);self.assertFalse(result['original_gjs_only_rule_would_pass'])

    def test_exact_captured_failed_T2_inventory_would_only_change_outer_policy(self):
        failed=json.loads((ROOT/'runs/stage2_base_eval_20261106T064330Z/GPU_idle_before_load.json').read_text())
        captured={row['pid']:row['process_metadata'] for row in failed['classification']['observations']}
        result=self.check(failed['stdout'],info=captured)
        self.assertTrue(failed['classification']['all_inventory_processes_verified_allowed'])
        self.assertFalse(failed['passed']);self.assertEqual(failed['pinned_desktop_matches'],[True,False])
        self.assertEqual(result['classification'],failed['classification'])
        self.assertTrue(result['passed']);self.assertTrue(result['proposal_only'])

    def test_unknown_python_unreadable_or_non_numeric_memory_is_rejected(self):
        python=1652111
        info=copy.deepcopy(self.info)
        info[python]=process(python,'/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python3.12',
            '/user.slice/user-1000.slice/user@1000.service/app.slice/unknown_model.service')
        for extra in (f'{python}, python, 20 MiB',f'8888, unknown, 10 MiB',f'{NAUTILUS}, /usr/bin/nautilus, N/A'):
            with self.subTest(extra=extra):
                self.assertFalse(self.check(f'{GJS}, gjs, 64 MiB\n'+extra+'\n',info=info)['passed'])

    def test_high_memory_is_rejected_for_anchor_and_extra(self):
        for stdout in (f'{GJS}, gjs, 257 MiB\n',
                       f'{GJS}, gjs, 64 MiB\n{NAUTILUS}, /usr/bin/nautilus, 257 MiB\n'):
            self.assertFalse(self.check(stdout)['passed'])

    def test_stale_or_missing_gjs_and_query_failure_are_rejected(self):
        self.assertFalse(self.check(f'{GJS}, gjs, 64 MiB\n',tick=1772)['passed'])
        for stdout in ('',f'{NAUTILUS}, /usr/bin/nautilus, 72 MiB\n'):
            result=self.check(stdout)
            self.assertFalse(result['passed']);self.assertEqual(result['pinned_gjs_match_count'],0)
        # Original all([]) is vacuously true; the candidate deliberately closes it.
        self.assertTrue(self.check('')['original_gjs_only_rule_would_pass'])
        self.assertFalse(self.check(f'{GJS}, gjs, 64 MiB\n',code=1)['passed'])

    def test_second_unpinned_gjs_or_any_duplicate_PID_is_rejected(self):
        second=1652222;info=copy.deepcopy(self.info);info[second]=copy.deepcopy(info[GJS]);info[second]['pid']=second
        result=self.check(f'{GJS}, gjs, 64 MiB\n{second}, gjs, 64 MiB\n',info=info)
        self.assertTrue(result['classification']['all_inventory_processes_verified_allowed'])
        self.assertFalse(result['passed'])
        for stdout in (f'{GJS}, gjs, 64 MiB\n{GJS}, gjs, 64 MiB\n',
            f'{GJS}, gjs, 64 MiB\n{NAUTILUS}, /usr/bin/nautilus, 72 MiB\n{NAUTILUS}, /usr/bin/nautilus, 72 MiB\n'):
            result=self.check(stdout);self.assertFalse(result['passed']);self.assertFalse(result['unique_inventory_PIDs'])

    def test_no_basename_UID_inode_cgroup_or_startup_binding_exemption(self):
        changes=[lambda x:x[NAUTILUS].update(uid_fields=[0]*4),
            lambda x:x[NAUTILUS].update(exe='/home/hugo/nautilus'),
            lambda x:x[NAUTILUS]['executable_identity'].update(same_system_inode=False),
            lambda x:x[NAUTILUS]['executable_identity'].update(system_uid=1000),
            lambda x:x[NAUTILUS]['executable_identity'].update(system_mode=0o777),
            lambda x:x[NAUTILUS]['cgroups'][0].update(path='/unverified'),
            lambda x:x[GJS]['executable_identity'].update(system_mode=0o700)]
        for change in changes:
            with self.subTest(change=change):
                info=copy.deepcopy(self.info);change(info)
                self.assertFalse(self.check(f'{GJS}, gjs, 64 MiB\n{NAUTILUS}, /usr/bin/nautilus, 72 MiB\n',info=info)['passed'])
        self.assertFalse(self.check(f'{GJS}, /usr/bin/gjs, 64 MiB\n')['passed'])
        self.desktop['memory_limit_MiB']=512
        self.assertFalse(self.check(f'{GJS}, gjs, 64 MiB\n')['passed'])

    def test_import_is_CPU_only_and_has_no_entrypoint(self):
        code=f"import sys;sys.dont_write_bytecode=True;sys.path.insert(0,{str(Path(__file__).parent)!r});import pinned_gjs_desktop_guard_candidate as m;assert 'torch' not in sys.modules;assert not hasattr(m,'main')"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
