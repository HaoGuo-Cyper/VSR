#!/usr/bin/env python3
"""CPU display-only all33 center-ROI sheets, from original HR/LR and saved PNGs.

Run only after the existing six VideoMAEv2 video comparisons have completed.
This uses their saved native verification receipt, never a new restore or MP4
frame decode. Existing helper/core/parent files are imported read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = WORK.parent / 'outputs'
ROI_TLHW = (32, 56, 64, 112)
PANEL_ORDER = ('HR', 'LR_bicubic_x4', 'baseline', 'videomaev2')
LABEL_HEIGHT = 18
COLUMNS, ROWS = 3, 11
SHEET_HW = (902, 1344)


def need(value, message):
    if not value:
        raise ValueError(message)


def safe_path(value):
    path = Path(value)
    need(path.is_absolute() and '..' not in path.parts and str(path) == os.path.normpath(str(path)),
         'Use absolute normalized paths without traversal')
    for candidate in reversed((path, *path.parents)):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        need(not stat.S_ISLNK(mode), 'Symbolic links are not display inputs or outputs')
    return path


def read(path):
    return json.loads(safe_path(path).read_text(encoding='utf-8'))


def binding(path):
    path = safe_path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def checked(item):
    path = safe_path(item['path'])
    need(binding(path) == {k: item[k] for k in ('path', 'bytes', 'sha256')},
         'Bound presentation input changed: ' + str(path))
    return path


def completed_display(directory):
    """First gate, before importing the evaluator, reading RGB or quality rows."""
    directory = safe_path(directory)
    root = safe_path(OUTPUT_ROOT)
    need(directory != root and directory.is_relative_to(root)
         and directory.name.startswith('video_compare') and directory.is_dir(),
         'Use an existing video_compare directory inside original outputs')
    completion = read(directory / 'completion.json')
    need(completion.get('status') == 'completed_CPU_video_delivery'
         and completion.get('MP4_count') == completion.get('contact_sheet_count') == 6
         and completion.get('frames_per_MP4') == 33
         and completion.get('GPU_work_performed') is False and completion.get('new_inference_calls') == 0,
         'Existing six video comparisons must be completed first')
    need(completion['manifest']['path'] == str(directory / 'manifest.json'), 'Display manifest escaped its directory')
    checked(completion['manifest'])
    manifest = read(directory / 'manifest.json')
    need(manifest.get('schema') == 'videomaev2_video_delivery_v1'
         and manifest.get('status') == 'completed_CPU_display_only'
         and manifest.get('CPU_only') is True and manifest.get('display_only') is True
         and manifest.get('new_inference_calls') == manifest.get('new_training_updates') == 0
         and manifest.get('scores_changed') is False, 'Require the completed display-only VideoMAEv2 manifest')
    clips = manifest['fixed_video_clip_ids']
    need(len(clips) == len(set(clips)) == 3
         and all(isinstance(c, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,256}', c) for c in clips)
         and manifest['sampling_seeds'] == [42, 43], 'Fixed display clips or seeds differ')
    cases = manifest['cases']
    need(len(cases) == 6 and {(c['record_id'], c['sampling_seed']) for c in cases}
         == {(clip, seed) for clip in clips for seed in (42, 43)}, 'Missing or duplicate fixed display cases')
    receipt = manifest['complete_test_verification']
    need(receipt.get('status') == 'verified_complete_clean_evaluation'
         and receipt.get('restores') == 128 and receipt.get('training_updates') == 0
         and receipt.get('prediction_PNGs_verified') == 4224 and receipt.get('prediction_MP4s_verified') == 128
         and receipt.get('both_arms_retained') is True and receipt.get('training_only_checkpoint_roles_verified') is True
         and receipt.get('protocol_sha256') == manifest['test_protocol']['sha256'],
         'Existing successful native test128 verification receipt is required')
    target = safe_path(directory / 'all33_center_ROI_manifest.json')
    need(not target.exists(), 'Preserve any earlier ROI manifest; never overwrite')
    for case in cases:
        folder = safe_path(directory / case['record_id'] / f"seed_{case['sampling_seed']}")
        need(folder.is_dir(), 'Existing completed display case directory missing')
        need(not (folder / 'all33_center_ROI.png').exists(), 'Preserve earlier ROI PNG; never overwrite')
    return directory, completion, manifest


def make_sheet(combined, *, source_indices, sampling_seed, header):
    """Retain every original ROI pixel; only add an18-pixel frame label."""
    need(isinstance(combined, np.ndarray) and combined.dtype == np.uint8
         and combined.shape == (33, 172, 896, 3), 'Expected uncompressed native four-panel33 array')
    need(header == 24 and len(source_indices) == 33 and sampling_seed in (42, 43), 'Original display geometry/chronology required')
    top, left, height, width = ROI_TLHW
    need(top + height <= 128 and left + width <= 224, 'ROI must stay inside the original source pixels')
    sheet = Image.new('RGB', (SHEET_HW[1], SHEET_HW[0]), (18, 18, 18))
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for ordinal in range(33):
        cell_x, cell_y = ordinal % COLUMNS * 448, ordinal // COLUMNS * (64 + LABEL_HEIGHT)
        for panel, label in enumerate(('HR', 'LR', 'B', 'R')):
            crop = combined[ordinal, header + top:header + top + height,
                            panel * 224 + left:panel * 224 + left + width]
            need(crop.shape == (64, 112, 3), 'ROI shape or panel order changed')
            sheet.paste(Image.fromarray(crop), (cell_x + panel * width, cell_y + LABEL_HEIGHT))
            text = f'{ordinal:02d} {label}' if panel == 0 else label
            draw.text((cell_x + panel * width + 3, cell_y + 3), text, font=font, fill=(255, 255, 255))
    need(sheet.size == (1344, 902), 'ROI sheet geometry changed')
    return sheet


def generate(ev, display, *, display_dir):
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
         'Run in a fresh CPU process with CUDA_VISIBLE_DEVICES empty')
    directory, completion, manifest = completed_display(display_dir)
    # Text bindings and saved receipts replace a second native/model/media verify.
    for name in ('test_protocol', 'test_status', 'test_observations', 'study_file', 'dataset', 'cohort_seal', 'finals_receipt'):
        checked(manifest[name])
    for item in manifest['tool_sources'].values(): checked(item)
    for item in manifest['runtime_sources'].values(): checked(item)
    need(str(Path(ev.__file__).resolve()) == manifest['runtime_sources']['evaluate.py']['path'],
         'Use the same frozen evaluator source root as the completed display')
    need(manifest['tool_sources']['video_delivery']['path'] == str(Path(display.__file__).resolve()),
         'Use the same already-reviewed video delivery helper')
    run = safe_path(manifest['test_run'])
    need(manifest['test_protocol']['path'] == str(run / 'protocol.json')
         and manifest['test_status']['path'] == str(run / 'status.json')
         and manifest['test_observations']['path'] == str(run / 'observations.json'), 'Test metadata escaped its declared run')
    state, protocol = read(run / 'status.json'), read(run / 'protocol.json')
    psha = manifest['test_protocol']['sha256']
    need(state.get('status') == 'completed' and state.get('restores') == 128 and state.get('training_updates') == 0
         and state.get('protocol_sha256') == psha and protocol.get('schema') == ev.VMAE_EVALUATION_SCHEMA
         and protocol.get('phase') == 'test' and protocol['scope']['total_restores'] == 128,
         'Completed native VideoMAEv2 test128 must remain unchanged')
    cohort, clips = protocol['cohort'], protocol['fixed_video_clip_ids']
    expected = [c['record']['id'] for c in sorted(cohort, key=lambda c: c['record']['source_id'])[:3]]
    need(clips == expected == manifest['fixed_video_clip_ids'], 'Prospective display clips changed')
    roles = {r['family']: r for r in protocol['roles']}
    finals = protocol['validation_final_lock']['final_roles']
    need(len(protocol['roles']) == 2 and set(roles) == {'baseline', 'videomaev2'}
         and manifest['final_roles'] == finals, 'Final student lock changed')
    selected = {r['family']: r for r in finals}
    for family, weight in (('baseline', 0.), ('videomaev2', .5)):
        role = roles[family]
        need(role['arm'] == family and role['weight'] == weight and role['update'] == role['step'] == 2048
             and role['checkpoint'] == selected[family]['checkpoint'], 'ROI display changed a locked final checkpoint')
    observations = read(run / 'observations.json')['observations']
    index = {(r['role'], r['record_id'], r['sampling_seed']): r for r in observations}
    need(len(index) == len(observations) == 128 and set(index) == {
        (r['label'], c['record']['id'], seed) for r in roles.values() for c in cohort for seed in (42, 43)},
        'Original completed matrix identities changed')
    ev.verify_imported_runtime(protocol['runtime'])
    need(not torch.cuda.is_initialized(), 'Presentation import initialized CUDA')
    from vsr.data import FixedCleanReader
    cfg = protocol['config']
    reader = FixedCleanReader(cfg['dataset'], expected_split='test', cohort_seal=cfg['seal'],
                              path_map=cfg.get('path_map'), expected_dataset_sha256=protocol['dataset']['sha256'])
    cases = {(c['record_id'], c['sampling_seed']): c for c in manifest['cases']}
    images, records = [], []
    try:
        for clip in clips:
            item = next(c for c in cohort if c['record']['id'] == clip)
            source = reader.load(clip, condition='clean', num_frames=33, start=0)
            need(source['metadata']['split'] == 'test' and source['metadata']['source_frame_indices'] == item['source_indices']
                 and source['fps'] == item['cached_header']['fps'], 'Bound HR/LR source window or FPS changed')
            for seed in (42, 43):
                case = cases[clip, seed]
                need(case['panel_order'] == list(PANEL_ORDER) and case['native_panel_hw'] == [128, 224]
                     and case['source_frame_indices'] == item['source_indices'], 'Existing display panel or source mapping changed')
                for asset in case['HR_LR_assets'].values(): checked(asset)
                predictions, proofs = [], {}
                for family in ('baseline', 'videomaev2'):
                    frames, proof = display._prediction_frames(ev, run, psha, roles[family], item, seed, index)
                    need(proof == case['saved_prediction_inputs'][family], 'Original PNG display inputs changed')
                    predictions.append(frames); proofs[family] = proof
                combined = display._panels(source, predictions, clip_id=clip, source_indices=item['source_indices'],
                                           sample=seed, roles=roles, scale=1)
                sheet = make_sheet(combined, source_indices=item['source_indices'], sampling_seed=seed, header=display.HEADER)
                output = safe_path(directory / clip / f'seed_{seed}' / 'all33_center_ROI.png')
                images.append((output, sheet))
                records.append({'record_id': clip, 'source_id': item['record']['source_id'], 'sampling_seed': seed,
                    'frame_indices': list(range(33)), 'source_frame_indices': item['source_indices'],
                    'panel_order': list(PANEL_ORDER), 'ROI_top_left_height_width': list(ROI_TLHW),
                    'sheet_HW': list(SHEET_HW), 'frame_label_height': LABEL_HEIGHT, 'grid_columns': COLUMNS, 'grid_rows': ROWS,
                    'HR_LR_assets': case['HR_LR_assets'], 'saved_prediction_inputs': proofs,
                    'source_comparison_MP4_reference_not_decoded': case['comparison_MP4'], 'output_path': str(output)})
    finally:
        reader.close()
    need(len(images) == len(records) == 6 and not torch.cuda.is_initialized(), 'Expected six CPU-only all33 ROI sheets')
    for (path, image), record in zip(images, records):
        with path.open('xb') as stream: image.save(stream, format='PNG')
        with Image.open(path) as saved:
            need(saved.mode == 'RGB' and saved.size == (1344, 902)
                 and np.array_equal(np.asarray(saved), np.asarray(image)), 'Saved PNG must preserve all ROI pixels')
        record['all33_center_ROI_PNG'] = binding(path)
    result = {'schema': 'videomaev2_all33_center_ROI_presentation_v1', 'status': 'completed_CPU_display_only',
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'tool_source': binding(Path(__file__).resolve()),
        'existing_video_manifest': binding(directory / 'manifest.json'), 'existing_video_completion': binding(directory / 'completion.json'),
        'test_protocol': manifest['test_protocol'], 'existing_native_test_verification': manifest['complete_test_verification'],
        'fixed_video_clip_ids': clips, 'sampling_seeds': [42, 43], 'cases': records,
        'ROI_top_left_height_width': list(ROI_TLHW), 'frames_per_case': 33, 'sheet_count': 6,
        'display_only': True, 'CPU_only': True, 'new_inference_calls': 0, 'new_training_updates': 0,
        'scores_changed': False, 'MP4_decoded': False, 'native_verification_repeated': False,
        'source_pixels': 'Original bound HR/LR NPY and saved native inference PNG via reviewed display helper; never lossy MP4.',
        'publication_policy': {'text_metadata_may_be_public': True, 'ROI_PNG_must_not_be_uploaded': True}}
    target = directory / 'all33_center_ROI_manifest.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return {'status': result['status'], 'manifest': binding(target), 'ROI_sheets': 6,
            'sheet_HW': list(SHEET_HW), 'new_inference_calls': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dependency-root', type=Path)
    parser.add_argument('--display-dir', type=Path, default=OUTPUT_ROOT / 'video_compare_videomaev2_20261108')
    args = parser.parse_args(argv)
    completed_display(args.display_dir)  # Refuse incomplete delivery before helper/evaluator imports.
    sys.path.insert(0, str(WORK))
    spec = importlib.util.spec_from_file_location('videomaev2_roi_original_display', WORK / 'videomaev2_video_delivery.py')
    display = importlib.util.module_from_spec(spec); spec.loader.exec_module(display)
    ev = display.config_tools.load_evaluator(args.source_root, args.dependency_root)
    print(json.dumps(generate(ev, display, display_dir=args.display_dir), indent=2))


if __name__ == '__main__':
    main()
