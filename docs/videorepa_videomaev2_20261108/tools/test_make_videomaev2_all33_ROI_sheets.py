"""Two CPU-only layout/early-gate fixtures; never use real test data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

HELPER = Path(__file__).with_name('make_videomaev2_all33_ROI_sheets.py')
spec = importlib.util.spec_from_file_location('vmae_roi_sheets', HELPER)
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


class LayoutTests(unittest.TestCase):
    def test_all33_original_center_pixels_chronology_grid_and_lossless_PNG(self):
        array = np.zeros((33, 172, 896, 3), dtype=np.uint8)
        array[:, :24] = 255; array[:, 152:] = 254
        for frame in range(33):
            for panel in range(4):
                y, x = np.mgrid[:128, :224]
                array[frame, 24:152, panel*224:(panel+1)*224] = np.stack([
                    np.full_like(x, frame*4+panel), x, y], axis=-1)
        sheet = v.make_sheet(array, source_indices=list(range(100,133)), sampling_seed=42, header=24)
        self.assertEqual(sheet.size, (1344, 902))
        saved = np.asarray(sheet)
        for frame in range(33):
            cy, cx = frame//3*82+18, frame%3*448
            for panel in range(4):
                expected = array[frame, 24+32:24+96, panel*224+56:panel*224+168]
                self.assertTrue(np.array_equal(saved[cy:cy+64,cx+panel*112:cx+(panel+1)*112], expected))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic.png'; sheet.save(path)
            with Image.open(path) as image: self.assertTrue(np.array_equal(np.asarray(image), saved))

    def test_incomplete_video_delivery_fails_before_manifest_or_RGB_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'video_compare_fixture'; folder.mkdir()
            (folder/'completion.json').write_text(json.dumps({'status':'running'}))
            opened=[]; original=v.read
            def guarded(path): opened.append(Path(path).name); return original(path)
            with patch.object(v,'OUTPUT_ROOT',root), patch.object(v,'read',guarded), self.assertRaisesRegex(ValueError,'completed first'):
                v.completed_display(folder)
            self.assertEqual(opened,['completion.json'])
            self.assertFalse((folder/'all33_center_ROI_manifest.json').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
