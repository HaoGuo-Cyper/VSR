"""Synthetic complete matrices and fresh lifecycle rejection; CPU only."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from vsr import evaluate as ev
from vsr import metrics as mt


def seal_fixture():
    mask = [8 <= i < 16 for i in range(33)]
    return {'seal_digest':'a'*64,'config':{'training_seeds':list(ev.SEEDS),
             'inference':{'sampling_seeds':[42,43]},'conditions':list(ev.CONDITIONS[1:])},
            'cohort':[{'record':{'id':f'clip_{i}','source_id':f'source_{i}'},'mask':mask} for i in range(32)]}


def score_fixture(role, condition, *, pilot=False):
    index = 0 if pilot else ev.SEEDS.index(int(role))
    clean = condition == 'clean'
    lp = .1 + .01*index if clean else .3 + .01*index
    psnr = 30. if clean else 25.
    if pilot and not clean and role == 'A2': lp -= .05; psnr += 1
    mask = [8 <= i < 16 for i in range(33)]
    pools = {'masked':{'psnr_rgb_db':psnr,'ssim_rgb':.8,'lpips_alex_v0_1':lp},
             'neighbors':{'psnr_rgb_db':31.,'ssim_rgb':.9,'lpips_alex_v0_1':.07},
             'all_frames':{'psnr_rgb_db':30.,'ssim_rgb':.85,'lpips_alex_v0_1':lp}}
    return {'mask':mask,'pools':pools,'raw_harm_gap':{'psnr_rgb_db':31-psnr,'lpips_alex_v0_1':lp-.07},
            'temporal':{'mask_touch_tLP100':1.,'all28pairs_tLP100':1.}}


def rows_fixture(seal, *, pilot=False):
    roles = ev.ARMS if pilot else tuple(map(str,ev.SEEDS))
    hashes = {r:str(i+1+(3 if pilot else 0))*64 for i,r in enumerate(roles)}
    rows = []
    for role in roles:
        for item in seal['cohort']:
            for condition in ev.CONDITIONS:
                for sample in ev.SAMPLES:
                    row = {'training_seed':20261102 if pilot else int(role),'record_id':item['record']['id'],
                        'source_id':item['record']['source_id'],'condition':condition,'sampling_seed':sample,
                        'checkpoint_sha256':hashes[role],'cohort_seal_digest':seal['seal_digest'],
                        'score':score_fixture(role,condition,pilot=pilot)}
                    if pilot: row['arm'] = role
                    rows.append(row)
    return rows,hashes


class CompleteMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.seal=seal_fixture();cls.rows,cls.hashes=rows_fixture(cls.seal)

    def test_g2_is_byte_equivalent_to_archived_definition(self):
        archive=Path(__file__).resolve().parents[1]/'legacy/pre_refactor_20261106/scripts/stage2_metrics.py'
        namespace={}
        exec(compile(archive.read_text(),str(archive),'exec'),namespace)
        self.assertEqual(mt.summarize_g2(self.rows,self.seal),namespace['summarize_g2'](self.rows,self.seal))
        self.assertEqual(mt.summarize_g2(self.rows,self.seal)['status'],'passed_G2')

    def test_full1152_required_no_duplicates_wrong_source_or_hash(self):
        for mutate in ('missing','duplicate','source','hash','mask','type','seed'):
            rows=copy.deepcopy(self.rows)
            if mutate=='missing':rows.pop()
            elif mutate=='duplicate':rows[-1]=copy.deepcopy(rows[0])
            elif mutate=='source':rows[0]['source_id']='other'
            elif mutate=='hash':rows[0]['checkpoint_sha256']='f'*64
            elif mutate=='mask':rows[0]['score']['mask'][0]=True
            elif mutate=='type':rows[0]['condition']='selected_only'
            else:rows[0]['sampling_seed']=99
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                mt.index_observations(rows,self.seal,self.hashes,kind='bases')

    def test_g2_zero_effect_fails_and_partial_matrix_has_no_gate(self):
        rows=copy.deepcopy(self.rows)
        for row in rows:
            row['score']['raw_harm_gap']=score_fixture(str(row['training_seed']),'clean')['raw_harm_gap']
        self.assertEqual(mt.summarize_g2(rows,self.seal)['status'],'failed_G2_stop')
        with self.assertRaises(ValueError):mt.summarize_g2(rows[:228],self.seal)

    def test_g4_uses_frozen_point_gate_and_paired_PCG64_without_global_RNG(self):
        rows,hashes=rows_fixture(self.seal,pilot=True)
        rng_before=copy.deepcopy(np.random.get_state())
        report=mt.summarize_pilot(rows,self.rows,self.seal,hashes,self.hashes)
        self.assertEqual(report['status'],'passed_G4')
        self.assertAlmostEqual(report['recovery']['C1_minus_A2_masked_LPIPS'],.05)
        self.assertAlmostEqual(report['recovery']['Base_seed_sample_SD_ddof1_threshold'],.01)
        self.assertEqual(report['uncertainty']['draw_indices_sha256'],'138800fce3662d71660db5a7db379ce06635503115eabb5940772245208eebe8')
        self.assertFalse(report['uncertainty']['CI_is_additional_gate'])
        self.assertFalse(report['automatic_4096_continuation'])
        after=np.random.get_state()
        self.assertEqual(rng_before[0],after[0]);np.testing.assert_array_equal(rng_before[1],after[1])
        self.assertEqual(rng_before[2:],after[2:])

    def test_g4_clean_zero_tolerance_is_an_independent_guard(self):
        rows,hashes=rows_fixture(self.seal,pilot=True)
        for row in rows:
            if row['arm']=='A2' and row['condition']=='clean':
                row['score']['pools']['all_frames']['ssim_rgb']-=1e-12
        report=mt.summarize_pilot(rows,self.rows,self.seal,hashes,self.hashes)
        self.assertEqual(report['status'],'failed_G4_stop')
        self.assertTrue(report['recovery']['passes']);self.assertFalse(report['clean_guards']['ssim_rgb']['passes'])

    def test_quantization_and_rgb_identity_remain_finite(self):
        value=torch.linspace(0,1,3*2*8*8).reshape(1,3,2,8,8)
        actual=mt.quantize_rgb(value)
        torch.testing.assert_close(actual,torch.round(value*255).byte().float()/255,rtol=0,atol=0)
        report=mt.rgb_metrics(actual,actual)
        self.assertEqual(report['exact_match_frames'],2);self.assertEqual(report['psnr_rgb_db'],120.)


class EvaluateLifecycleTests(unittest.TestCase):
    def test_metadata_prepare_is_new_only_and_never_acquires_GPU(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'new'
            with patch.object(ev,'collect_protocol',return_value={'schema':'fixture'}),patch('vsr.checkpoints.gpu_lease') as lease:
                result=ev.prepare(Path(folder)/'config.json',run)
                self.assertEqual(result['status'],'planned');lease.assert_not_called()
                with self.assertRaisesRegex(ValueError,'Fresh-only'):ev.prepare(Path(folder)/'config.json',run)
        self.assertFalse(torch.cuda.is_initialized())

    def test_stopped_partial_or_wrong_protocol_cannot_execute_or_unlock_g2(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);ev.save(run/'protocol.json',{'schema':'vsr_evaluation_v2','kind':'bases','config':{'device':'cuda:0'}})
            psha=ev.sha(run/'protocol.json')
            for status in ('running','stopped_by_user','failed_stop','completed'):
                ev.save(run/'status.json',{'status':status,'protocol_sha256':psha,'restores':228,'training_updates':0},replace=True)
                with patch('vsr.checkpoints.gpu_lease') as lease,self.assertRaises(ValueError):ev.execute(run,psha)
                lease.assert_not_called()
                with self.assertRaises(ValueError):ev.validate_completed_base_evaluation(run)
            with self.assertRaises(ValueError):ev.execute(run,'0'*64)

    def test_cleanup_attempts_all_operations_and_preserves_primary(self):
        calls=[];primary=RuntimeError('primary failure')
        def first():calls.append(1);raise OSError('cleanup error')
        def second():calls.append(2)
        ev._cleanup_preserving(primary,[first,second])
        self.assertEqual(calls,[1,2]);self.assertIn('cleanup error',' '.join(primary.__notes__))
        calls.clear()
        with self.assertRaises(OSError):ev._cleanup_preserving(None,[first,second])
        self.assertEqual(calls,[1,2])

    def test_bound_input_change_rejected_without_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'input';file.write_bytes(b'original')
            b=ev.binding(file);file.write_bytes(b'changed')
            with self.assertRaises(ValueError):ev.verify_binding(b)

    def test_pilot_incomplete_summary_rejected_before_asset_or_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            ev.save(run/'protocol.json',{'schema':'vsr_training_protocol_v2','mode':'pilot'})
            ev.save(run/'status.json',{'status':'completed','protocol_sha256':ev.sha(run/'protocol.json'),'summary_sha256':'0'*64})
            ev.save(run/'summary.json',{'status':'completed','mode':'pilot','protocol_sha256':ev.sha(run/'protocol.json'),'scientific_training_updates':6144})
            with self.assertRaisesRegex(ValueError,'completion hash chain'):
                ev.validate_completed_pilot_training(run,[],{}, {})

    def test_all_public_help_paths_do_not_start_GPU(self):
        with self.assertRaises(SystemExit) as exit:
            ev.main(['--help'])
        self.assertEqual(exit.exception.code,0);self.assertFalse(torch.cuda.is_initialized())



class CompleteEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.folder=Path(cls.temp.name)
        cls.seal=seal_fixture();ev.save(cls.folder/'seal.json',cls.seal)
        rows,hashes=rows_fixture(cls.seal)
        roles=[{'role':str(seed),'training_seed':seed,'checkpoint':{'sha256':hashes[str(seed)]}}
               for seed in ev.SEEDS]
        cls.protocol={'schema':'vsr_evaluation_v2','kind':'bases','roles':roles,
                      'cohort_seal':ev.binding(cls.folder/'seal.json'),'config_binding':{'path':'synthetic.json'},'config':{}}
        ev.save(cls.folder/'protocol.json',cls.protocol);cls.psha=ev.sha(cls.folder/'protocol.json')
        raw=cls.folder/'fixture.bin';raw.write_bytes(b'CPU-only synthetic artifact')
        canonical=[]
        for row in rows:
            mask=row['score']['mask'];seed_index=ev.SEEDS.index(row['training_seed'])
            clean=row['condition']=='clean'
            frame=[{'psnr_rgb_db':(30. if clean else 25.) if flag else 31.,'ssim_rgb':.8 if flag else .9} for flag in mask]
            lp=[(.1 if clean else .3)+.01*seed_index if flag else .07 for flag in mask]
            full={'all_frames':{'per_frame':frame,'lpips_per_frame':lp,
                  'per_temporal_pair':[{'indices':[i,i+1],mt.TLP100_KEY:1.} for i in range(2,30)]}}
            row['score']=mt.score_masked(full,mask)
            directory=cls.folder/'results'/str(row['training_seed'])/row['record_id']/row['condition']/str(row['sampling_seed'])
            directory.mkdir(parents=True);(directory/'png').mkdir()
            artifacts=[]
            for index in range(33):
                path=directory/'png'/f'{index:06d}.png'
                import os
                os.link(raw,path);artifacts.append(ev.binding(path))
            detail={'status':'completed','training_updates':0,'observation':row,
                    'full_frame_pair_metrics':full,'PNG_artifacts':artifacts}
            ev.save(directory/'metrics.json',detail)
            ev.save(directory/'complete.json',{'protocol_sha256':cls.psha,'metrics_sha256':ev.sha(directory/'metrics.json'),
                                             'PNG_count':33,'resume_allowed':False})
            canonical.append(row|{'row_metrics_path':str(directory/'metrics.json'),'row_metrics_sha256':ev.sha(directory/'metrics.json')})
        ev.save(cls.folder/'observations.json',{'observations':canonical,'protocol_sha256':cls.psha})
        ev.save(cls.folder/'role_audits.json',[{'role':str(seed),'restores':384,'training_updates':0,
                 'inference_weights':'student','trainable_before':{'sha':'fixture'},'trainable_after':{'sha':'fixture'}} for seed in ev.SEEDS])
        summary=mt.summarize_g2(canonical,cls.seal);ev.save(cls.folder/'summary.json',summary)
        ev.save(cls.folder/'status.json',{'status':'completed','protocol_sha256':cls.psha,'restores':1152,'training_updates':0,
             'gate_status':summary['status'],'summary_sha256':ev.sha(cls.folder/'summary.json')})
        cls.rows=canonical

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def verified(self):
        with patch.object(ev,'_verify_protocol_inputs'),patch.object(ev,'collect_protocol',return_value=self.protocol):
            return ev.validate_completed_base_evaluation(self.folder)

    def test_complete1152_source_row_marker_and_artifact_chain_is_recomputed(self):
        _,evidence=self.verified()
        self.assertEqual(evidence['status'],'passed_G2');self.assertEqual(evidence['rows'],1152)
        self.assertEqual(evidence['protocol_sha256'],self.psha)

    def test_changed_row_marker_or_prediction_file_revokes_completion(self):
        path=Path(self.rows[0]['row_metrics_path']).parent/'complete.json';original=path.read_bytes()
        try:
            value=json.loads(original);value['PNG_count']=32;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'completion seal'):self.verified()
        finally:path.write_bytes(original)
        image=Path(self.rows[0]['row_metrics_path']).parent/'png/000000.png';raw=image.read_bytes()
        try:
            image.write_bytes(b'changed artifact')
            with self.assertRaisesRegex(ValueError,'Bound input changed'):self.verified()
        finally:image.write_bytes(raw)


class LeaseFailureTests(unittest.TestCase):
    def test_failure_receipt_is_written_before_the_GPU_lease_releases(self):
        from contextlib import contextmanager
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            protocol={'schema':'vsr_evaluation_v2','kind':'bases','config_binding':{'path':'fixture.json'},
                      'config':{'device':'cuda:0','gpu_lock':str(run/'gpu.lock'),'gpu_admission':{'expected_gpu_uuid':'GPU-fixture'}}}
            ev.save(run/'protocol.json',protocol);psha=ev.sha(run/'protocol.json')
            ev.save(run/'status.json',{'status':'planned','protocol_sha256':psha})
            events=[]
            @contextmanager
            def lease(*args):
                events.append('enter')
                try:yield {'GPU':'synthetic no device access'}
                finally:
                    self.assertTrue((run/'failure.json').is_file())
                    self.assertEqual(ev.read(run/'status.json')['status'],'failed_stop')
                    events.append('release')
            with patch.dict('os.environ',ev.STRICT_ENVIRONMENT),patch.object(ev,'_verify_protocol_inputs'),\
                 patch.object(ev,'collect_protocol',return_value=protocol),\
                 patch('vsr.checkpoints.gpu_lease',side_effect=lease),\
                 patch.object(ev,'_execute_loaded',side_effect=RuntimeError('simulated compute failure')):
                with self.assertRaisesRegex(RuntimeError,'simulated compute failure'):ev.execute(run,psha)
            self.assertEqual(events,['enter','release']);self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
