"""Ten-module boundary, immutable archive and real Base payload CPU checks."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch import nn
from vsr.checkpoints import load_training_checkpoint, OFFICIAL_REVISION

ROOT=Path(__file__).resolve().parents[1]
MODULES={'data','vae','model','teacher','losses','train','infer','evaluate','metrics','checkpoints'}


class CoreBoundaryTests(unittest.TestCase):
    def test_exact_ten_business_files_and_no_runtime_legacy_import(self):
        self.assertEqual({p.stem for p in (ROOT/'vsr').glob('*.py')},MODULES|{'__init__'})
        for path in (ROOT/'vsr').glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.level:
                    self.assertTrue(all(item.name in MODULES for item in node.names)) if node.module is None else self.assertIn(node.module.split('.')[0],MODULES)
                    if node.module is not None and path.stem in {'data','vae','model','teacher','losses','metrics','checkpoints','infer'}:
                        self.assertNotIn(node.module.split('.')[0],{'train','evaluate'})
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    value=ast.unparse(node)
                    self.assertNotIn('legacy',value)
            self.assertNotIn('spec_from_file_location',path.read_text())
            self.assertNotIn('sys.modules[',path.read_text())

    def test_archived_code_is_byte_identical_to_manifest(self):
        manifest=json.loads((ROOT/'legacy/pre_refactor_20261106/source_manifest.json').read_text())
        self.assertEqual(manifest['source_commit'],'60e2b4b178210806d32c00bb562eb9e8a4f3197e')
        self.assertEqual(len(manifest['files']),144)
        for item in manifest['files']:
            data=(ROOT/item['archive_path']).read_bytes()
            self.assertEqual(len(data),item['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(),item['sha256'])

    def test_all_modules_import_and_cli_help_with_CUDA_forbidden(self):
        script="""import importlib, torch
from unittest.mock import patch
with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden')):
    for name in %r: importlib.import_module('vsr.'+name)
    for name in ['data','train','infer','evaluate']:
        module=importlib.import_module('vsr.'+name)
        try: module.main(['--help'])
        except SystemExit as e: assert e.code==0
assert not torch.cuda.is_initialized()
""" % sorted(MODULES)
        process=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(process.returncode,0,process.stderr)

    def test_imports_and_CPU_fixtures_leave_CUDA_uninitialized(self):
        self.assertFalse(torch.cuda.is_initialized())


class RealBaseCPUCompatibilityTests(unittest.TestCase):
    def test_three_real_Base_payloads_restore_exactly_into_shape_only_CPU_fixture(self):
        """No Wan load: validate actual checkpoint schema/copy only, not GPU parity."""
        audit_path=ROOT/'docs/stage2_20261106/three_base_completion_audit.json'
        if not audit_path.is_file():self.skipTest('Private Base audit not present')
        audit=json.loads(audit_path.read_text())
        roots=Path('/home/hugo/Projects/VSR_/runs')
        declarations=[(20261102,'28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f'),
                      (20261103,'87f60ce812797d73b27b9d2d0bcb41abb45ea4659ca2b6eb6092d02f3b58061a'),
                      (20261104,'c6f0aedcefac965334851e081d72916dccfa9559b469dd1ad819d7139df65edd')]
        paths=[]
        for seed,digest in declarations:
            matches=list(roots.glob(f'stage2_protocol33_base_seed{seed}_*/Base/checkpoint_008192.pt'))
            if len(matches)!=1:self.skipTest('Private audited Base files not present on this host')
            paths.append((seed,digest,matches[0]))
        for seed,digest,path in paths:
            with self.subTest(seed=seed):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
                payload=torch.load(path,map_location='cpu',weights_only=True)
                state=payload['trainable_state']
                def factory(config,variant):
                    module=nn.Module()
                    for name,value in state.items():
                        node=module;parts=name.split('.')
                        for part in parts[:-1]:
                            if part not in node._modules:node.add_module(part,nn.Module())
                            node=node._modules[part]
                        node.register_parameter(parts[-1],nn.Parameter(torch.zeros_like(value,device='cpu')))
                    module.register_parameter('frozen_fixture_sentinel',nn.Parameter(torch.tensor([123.]),requires_grad=False))
                    return module,{'official_revision':OFFICIAL_REVISION,'fresh_official_initialization':True,
                                   'legacy_state_loaded':False,'test_only_shape_fixture':True}
                model,meta=load_training_checkpoint(path,device='cpu',allowed_root=roots,model_factory=factory,variant='Base')
                self.assertEqual(meta['checkpoint_sha256'],digest)
                self.assertEqual(meta['step'],8192)
                self.assertEqual(meta['restored_parameter_count'],6500992)
                self.assertEqual(len(state),488)
                self.assertFalse(meta['training_only_state_restored'])
                for name,p in model.named_parameters():
                    if p.requires_grad:torch.testing.assert_close(p.detach(),state[name],rtol=0,atol=0)
                self.assertEqual(float(model.frozen_fixture_sentinel),123.)
                del model,payload,state
        self.assertFalse(torch.cuda.is_initialized())


class RealDataMetadataCompatibilityTests(unittest.TestCase):
    def test_original_cache_and_val32_metadata_read_without_GPU_or_pixel_decoding(self):
        from vsr.data import PreparedTrainingData,FramewiseReader,validate_seal
        candidates=list(Path('/home/hugo/Projects/VSR_/runs').glob('stage2_protocol33_base_seed20261102_*/cache_asset_seal.json'))
        manifest=ROOT/'runs/stage2_val_assets_20261105T132924Z/dataset.json'
        seal_path=ROOT/'docs/stage2_eval_20261105/cohort_seal.json'
        if len(candidates)!=1 or not manifest.exists():self.skipTest('Private prepared assets absent')
        cache=PreparedTrainingData(candidates[0].parent,
                 expected_seal_sha256='aa4fdd751016ec6b03229f1f27581667e9f87bf91b6fedcf8fd03b2b9be19c73')
        self.assertEqual(len(cache),1641);self.assertEqual(len(cache.validation_ids),32)
        for index in (0,1640):
            pair=cache[index]
            self.assertEqual(tuple(pair['clean'].shape),(1,16,9,16,28))
            self.assertEqual(pair['clean'].device.type,'cpu')
            self.assertEqual(cache.record(index)['record_id'],cache.ids[index])
        seal=json.loads(seal_path.read_text());mapping={}
        for old_path,expected in seal['implementation_bindings'].items():
            current=Path(old_path)
            if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest()==expected:continue
            for prefix in ('vsr/','scripts/'):
                position=old_path.rfind('/'+prefix)
                if position<0:continue
                candidate=ROOT/'legacy/pre_refactor_20261106'/old_path[position+1:]
                if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest()==expected:
                    mapping[old_path]=str(candidate);break
        receipt=validate_seal(seal,path_map=mapping)
        self.assertTrue(receipt['source_hashes_verified'])
        reader=FramewiseReader(manifest,seal,path_map=mapping)
        self.assertEqual(len(reader.cohort),32)
        self.assertFalse(reader.metadata['pixel_decode_in_constructor'])
        self.assertFalse(reader.metadata['metrics_computed']);reader.close()
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
