import importlib.util,json,pathlib,tempfile,unittest
from unittest.mock import patch
P=pathlib.Path(__file__).with_name('export_videomaev2_text.py')
spec=importlib.util.spec_from_file_location('exporter',P);E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)

def fixture(root, payload=None):
    runs=root/'runs';run=runs/'study';repo=root/'repo';run.mkdir(parents=True);repo.mkdir()
    protocol={'schema':'vsr_videomaev2_training_protocol_v1','study':{'arms':[{'label':'baseline','weight':0.},{'label':'videomaev2','weight':.5}]}}
    if payload is not None:protocol['payload']=payload
    raw=(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n').encode();(run/'protocol.json').write_bytes(raw)
    (run/'status.json').write_text(json.dumps({'status':'running','protocol_sha256':E.sha(raw)}))
    return run,runs,repo

def destination(repo):return repo/'docs/videorepa_videomaev2_20261108/experiments/study'
def canonical(dest,rel):
    path=dest/(str(rel)+'.parts_manifest.json')
    return E.read_manifest_bytes(json.loads(path.read_text()),dest) if path.exists() else (dest/rel).read_bytes()

class ExportTests(unittest.TestCase):
    def test_large_protocol_unicode_long_line_exact_bounded_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,runs,repo=fixture(pathlib.Path(tmp),['完整🙂'*70000,'lastline'])
            E.export(run,runs,repo);dest=destination(repo)
            self.assertEqual(canonical(dest,'protocol.json'),(run/'protocol.json').read_bytes())
            manifest=json.loads((dest/'protocol.json.parts_manifest.json').read_text())
            self.assertGreater(len(manifest['parts']),1)
            for part in manifest['parts']:
                raw=(dest/part['path']).read_bytes();raw.decode();self.assertLessEqual(len(raw),400000)
            self.assertTrue(all(p.stat().st_size<=400000 for p in dest.rglob('*') if p.is_file()))

    def test_complete_line_chunks_and_append_immutable_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,runs,repo=fixture(pathlib.Path(tmp));arm=run/'arm_runs'/'videomaev2';arm.mkdir(parents=True)
            def row(i):return (json.dumps({'step':i,'arm':'videomaev2','value':'x'*19000})+'\n').encode()
            initial=b''.join(row(i) for i in range(1,25));log=arm/'training.jsonl';log.write_bytes(initial+b'{partial'+b'\xf0\x9f')
            E.export(run,runs,repo);dest=destination(repo);rel=pathlib.Path('arm_runs/videomaev2/training.jsonl')
            self.assertEqual(canonical(dest,rel),initial)
            parts={str(p.relative_to(dest)):p.read_bytes() for p in dest.rglob('part_*.txt')}
            log.write_bytes(initial+b''.join(row(i) for i in range(25,40)));E.export(run,runs,repo)
            self.assertEqual(canonical(dest,rel),log.read_bytes())
            for name,raw in parts.items():self.assertEqual((dest/name).read_bytes(),raw)
            for p in (dest/rel.parent/(rel.name+'.parts')).iterdir():
                for line in p.read_bytes().splitlines():json.loads(line)
            log.write_bytes(initial[:-1]+b'\n')
            with self.assertRaisesRegex(ValueError,'Append-only'):E.export(run,runs,repo)

    def test_binary_never_opened_and_allowlisted_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,runs,repo=fixture(pathlib.Path(tmp));binaries=[run/x for x in ['checkpoint.pt','prediction.mp4','sample.png','fresh_latents.pt']]
            for p in binaries:p.write_bytes(b'\0secret')
            original=pathlib.Path.read_bytes
            def guarded(p):
                if p in binaries:raise AssertionError('binary opened')
                return original(p)
            with patch.object(pathlib.Path,'read_bytes',guarded):E.export(run,runs,repo)
            self.assertFalse(any(p.suffix in ['.pt','.mp4','.png'] for p in destination(repo).rglob('*')))
            (run/'teacher_repeat.json').symlink_to(binaries[0])
            with self.assertRaisesRegex(ValueError,'Symlink'):E.export(run,runs,repo)

    def test_fixed_text_and_part_tamper_rejected_snapshots_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,runs,repo=fixture(pathlib.Path(tmp),'a'*500000);(run/'summary.json').write_text('{"status":"done"}')
            E.export(run,runs,repo);dest=destination(repo)
            state=json.loads((run/'status.json').read_text());state['status']='completed';(run/'status.json').write_text(json.dumps(state));E.export(run,runs,repo)
            self.assertEqual(json.loads((dest/'status.json').read_text())['status'],'completed')
            (run/'summary.json').write_text('{"status":"changed"}')
            with self.assertRaisesRegex(ValueError,'Immutable'):E.export(run,runs,repo)
            (run/'summary.json').write_text('{"status":"done"}')
            part=next((dest/'protocol.json.parts').iterdir());part.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Published chunk changed'):E.export(run,runs,repo)

    def test_wrong_schema_binding_arm_step_and_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,runs,repo=fixture(pathlib.Path(tmp));original=(run/'protocol.json').read_bytes()
            (run/'protocol.json').write_text('{"schema":"wan_trd_training_protocol_v1"}')
            with self.assertRaisesRegex(ValueError,'Not a VideoMAEv2'):E.export(run,runs,repo)
            (run/'protocol.json').write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'Protocol/status'):E.export(run,runs,repo)
            (run/'protocol.json').write_bytes(original);arm=run/'arm_runs/baseline';arm.mkdir(parents=True);(arm/'training.jsonl').write_text('{"step":2,"arm":"baseline"}\n')
            with self.assertRaisesRegex(ValueError,'prefix'):E.export(run,runs,repo)
            (arm/'training.jsonl').write_text('{"step":1,"arm":"videomaev2"}\n')
            with self.assertRaisesRegex(ValueError,'arm differs'):E.export(run,runs,repo)
            with self.assertRaisesRegex(ValueError,'Explicit child'):E.export(run,run,repo)

    def test_threshold_crossing_cannot_replace_fixed_small_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,runs,repo=fixture(pathlib.Path(tmp));(run/'teacher_repeat.json').write_text('{"ok":true}')
            E.export(run,runs,repo)
            (run/'teacher_repeat.json').write_text(json.dumps({'new':'x'*450000}))
            with self.assertRaisesRegex(ValueError,'Previously published fixed'):E.export(run,runs,repo)
            # One line over cap can be split, but normal lines remain intact.
            normal=(b'one\n'+b'two\n')*100001
            self.assertEqual(b''.join(E.utf8_chunks(normal)),normal)
            self.assertTrue(all(x.endswith(b'\n') for x in E.utf8_chunks(normal)))

if __name__=='__main__':unittest.main(verbosity=2)
