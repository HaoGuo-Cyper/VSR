"""Descriptive analysis of a completed640-row clean HR-VAE validation study.

No model, checkpoint tensor, PNG, video or GPU is loaded.  This script is kept
outside the frozen10-module runtime.  Validation selection is not heldout-test
evidence, and the paired clip intervals are conditional on one training seed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
METRICS = ('lpips_alex_v0_1', 'psnr_rgb_db', 'ssim_rgb',
           'tLP100_alex_v0_1_TecoGAN_crop', 'tOF_Farneback_TecoGAN_crop')
UPDATES = (0, 512, 1024, 2048)
HIGHER_BETTER = {'psnr_rgb_db', 'ssim_rgb'}


def need(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def finite(value): return type(value) in (int, float) and math.isfinite(value)


def verify_file(item):
    path = Path(item['path']).resolve()
    need(path.is_file() and sha(path) == item['sha256'], 'Changed bound text input: ' + str(path))
    need('bytes' not in item or path.stat().st_size == item['bytes'], 'Bound text size changed')
    return path


def stats(vector):
    values = np.asarray(vector, dtype=np.float64)
    need(values.shape == (32,) and np.isfinite(values).all(), 'Need32 finite paired clip values')
    return {'mean': float(values.mean()), 'clip_sample_SD_ddof1': float(values.std(ddof=1)), 'clips': 32}


def first_target_crossings(curves, baseline_label, target):
    """Discrete observed checkpoints only; never infer an interpolated time."""
    initial = curves[baseline_label][0]['metrics']['lpips_alex_v0_1']['mean']
    if initial <= target:
        return {'status': 'uninterpretable_initial_already_meets_target', 'target_LPIPS': target,
                'reason': 'The shared initial model already meets baseline@2048; acceleration cannot be inferred.',
                'crossings': {arm: None for arm in curves}, 'speedups_vs_baseline': {arm: None for arm in curves}}
    crossings = {}
    for arm, points in curves.items():
        reached = [u for u in UPDATES if points[u]['metrics']['lpips_alex_v0_1']['mean'] <= target]
        if not reached:
            crossings[arm] = {'status': 'not_reached_at_measured_checkpoints', 'update': None,
                              'cumulative_training_seconds': None}
        else:
            update = reached[0]
            crossings[arm] = {'status': 'reached_at_observed_checkpoint', 'update': update,
                'cumulative_training_seconds': points[update]['cumulative_training_seconds']}
    baseline_time = crossings[baseline_label]['cumulative_training_seconds']
    speedups = {}
    for arm, value in crossings.items():
        seconds = value['cumulative_training_seconds']
        speedups[arm] = baseline_time / seconds if seconds is not None and seconds > 0 and baseline_time else None
    return {'status': 'discrete_checkpoint_comparison', 'target_LPIPS': target,
            'target_definition': 'measured baseline LPIPS at2048 updates', 'crossings': crossings,
            'speedups_vs_baseline': speedups, 'interpolation_used': False,
            'limitation': 'The first observed crossing is a checkpoint-resolution upper bound, not an exact convergence time.'}


def compute(run):
    run = Path(run).resolve()
    state = read(run / 'status.json')
    need(state.get('status') == 'completed' and state.get('restores') == 640
         and state.get('training_updates') == 0, 'Only a complete640-row evaluation can be analyzed')
    protocol = read(run / 'protocol.json')
    psha = sha(run / 'protocol.json')
    need(protocol.get('schema') == 'hr_vae_clean_evaluation_protocol_v1'
         and state['protocol_sha256'] == psha, 'Wrong study protocol or raw SHA')
    need(len(protocol['sources']) == 11, 'Bind all11 current package source files')
    for item in protocol['sources'].values(): verify_file(item)
    summary = read(run / 'summary.json')
    need(state['summary_sha256'] == sha(run / 'summary.json') and summary['protocol_sha256'] == psha
         and summary['restores'] == 640, 'Completion summary hash/row count differs')
    bundle = read(run / 'observations.json')
    rows = bundle['observations']
    need(bundle['protocol_sha256'] == psha and len(rows) == 640, 'Observation protocol/count differs')
    study_path = verify_file(protocol['study_file']); study = read(study_path)
    training = protocol['training_evidence']
    need(training['study_file'] == protocol['study_file'] and training['scientific_updates'] == 6144,
         'Evaluation must bind the completed6144-step study')
    for item in training['files'].values(): verify_file(item)
    seal_path = verify_file(protocol['cohort_seal']); seal = read(seal_path)
    cohort = seal['cohort']; records = [r['record']['id'] for r in cohort]
    source_by_id = {r['record']['id']: r['record']['source_id'] for r in cohort}
    need(len(records) == len(set(records)) == len(set(source_by_id.values())) == 32,
         'Require32 unique clips from32 independent sources')
    need(protocol['scope']['total_restores'] == 640 and protocol['scope']['roles'] == 10
         and protocol['scope']['conditions'] == ['clean'] and protocol['scope']['sampling_seeds'] == [42, 43]
         and protocol['scope']['primary_pool'] == 'all33_frames', 'Changed clean study scope')
    roles = {r['label']: r for r in protocol['roles']}
    need(len(roles) == 10, 'Require ten unique checkpoint roles')
    shared = [r for r in roles.values() if r['update'] == 0]
    need(len(shared) == 1 and shared[0]['cumulative_training_seconds'] == 0, 'Require one shared starting checkpoint')
    arms = study['arms']
    need(len(arms) == 3 and [a['weight'] for a in arms] == [0., .01, .1], 'Report all three prospective study arms')
    baseline = arms[0]['label']; methods = [a['label'] for a in arms[1:]]
    weight_by_arm = {a['label']: a['weight'] for a in arms}
    role_by_point = {(r['arm'], r['update']): r for r in roles.values() if r['update']}
    need(set(role_by_point) == {(a['label'], u) for a in arms for u in UPDATES[1:]}, 'Missing study intermediate/final checkpoint')
    indexed = {}
    for row in rows:
        key = row['role'], row['record_id'], row['sampling_seed']
        need(key not in indexed and key[0] in roles and key[1] in source_by_id and key[2] in (42, 43),
             'Duplicate or foreign study observation')
        role = roles[key[0]]
        need(row['condition'] == 'clean' and row['training_seed'] == 20261102
             and row['source_id'] == source_by_id[key[1]] and row['cohort_seal_digest'] == seal['seal_digest']
             and row['checkpoint_sha256'] == role['checkpoint']['sha256']
             and all(row[k] == role[k] for k in ('arm', 'weight', 'update', 'step', 'cumulative_training_seconds')),
             'Study row/source/checkpoint/time identity differs')
        metrics = row['metrics']
        need(all(finite(metrics[m]) for m in METRICS), 'Nonfinite study metric')
        path = Path(row['row_metrics_path']).resolve()
        need(path.is_relative_to(run / 'results') and sha(path) == row['row_metrics_sha256'], 'Row metric JSON SHA changed')
        raw = read(path)
        need(raw['observation'] == {k: v for k, v in row.items() if not k.startswith('row_metrics_')},
             'Canonical observation differs from completed metric row')
        indexed[key] = metrics
    need(set(indexed) == {(r, rec, s) for r in roles for rec in records for s in (42, 43)}, 'Incomplete640 matrix')
    vectors, curves = {}, {}
    for arm in weight_by_arm:
        curves[arm] = {}; previous = -1.
        for update in UPDATES:
            role = shared[0] if update == 0 else role_by_point[arm, update]
            need(update == 0 or role['weight'] == weight_by_arm[arm], 'Weight changed between checkpoints')
            seconds = role['cumulative_training_seconds']
            need(finite(seconds) and seconds >= 0 and seconds > previous, 'Checkpoint training times must increase')
            previous = seconds
            vectors[arm, update] = np.asarray([[np.mean([indexed[role['label'], rec, s][metric] for s in (42, 43)])
                                               for metric in METRICS] for rec in records], dtype=np.float64)
            curves[arm][update] = {'role': role['label'], 'weight': weight_by_arm[arm], 'update': update,
                'cumulative_training_seconds': seconds, 'metrics': {
                    metric: stats(vectors[arm, update][:, j]) for j, metric in enumerate(METRICS)}}
    indices = np.random.Generator(np.random.PCG64(20261102)).integers(0, 32, size=(10000, 32), dtype=np.int64)
    indices_sha = hashlib.sha256(indices.astype('<i8', copy=False).tobytes()).hexdigest()
    comparisons = {}
    for method in methods:
        comparisons[method] = {}
        for update in UPDATES:
            difference = vectors[method, update] - vectors[baseline, update]
            for j, metric in enumerate(METRICS):
                if metric not in HIGHER_BETTER: difference[:, j] *= -1
            intervals = np.quantile(difference[indices].mean(axis=1), [.025, .975], axis=0, method='linear')
            comparisons[method][update] = {}
            for j, metric in enumerate(METRICS):
                comparisons[method][update][metric] = stats(difference[:, j]) | {
                    'paired_clip_95pct_CI': list(map(float, intervals[:, j])),
                    'difference': 'method-minus-baseline' if metric in HIGHER_BETTER else 'baseline-minus-method',
                    'positive_favors_method': True, 'shared_initial_checkpoint': update == 0}
    final = {}
    for method in methods:
        lp = comparisons[method][2048]['lpips_alex_v0_1']; lo, hi = lp['paired_clip_95pct_CI']
        classification = 'single_training_seed_LPIPS_improvement' if lo > 0 else 'no_observed_gain' if hi <= 0 else 'insufficient_evidence'
        final[method] = {'weight': weight_by_arm[method], 'LPIPS_improvement_status': classification,
            'LPIPS_baseline_minus_method': lp,
            'PSNR_method_minus_baseline': comparisons[method][2048]['psnr_rgb_db'],
            'SSIM_method_minus_baseline': comparisons[method][2048]['ssim_rgb'],
            'PSNR_fidelity_tradeoff': comparisons[method][2048]['psnr_rgb_db']['mean'] < 0,
            'SSIM_fidelity_tradeoff': comparisons[method][2048]['ssim_rgb']['mean'] < 0}
    selected = min(methods, key=lambda a: (curves[a][2048]['metrics']['lpips_alex_v0_1']['mean'], weight_by_arm[a]))
    target = curves[baseline][2048]['metrics']['lpips_alex_v0_1']['mean']
    efficiency = first_target_crossings(curves, baseline, target)
    efficiency['final2048_training_seconds_ratio_method_over_baseline'] = {
        a: curves[a][2048]['cumulative_training_seconds'] / curves[baseline][2048]['cumulative_training_seconds'] for a in methods}
    return {
        'schema': 'hr_vae_clean_descriptive_analysis_v1', 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'run': str(run), 'protocol_sha256': psha, 'observations_sha256': sha(run / 'observations.json'),
        'summary_sha256': sha(run / 'summary.json'), 'study_file': protocol['study_file'],
        'analysis_script': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'runtime_sources_bound_by_study': protocol['sources'],
        'python_packages': {name: importlib.metadata.version(name) for name in ('numpy', 'matplotlib')},
        'method_training_seed': 20261102, 'sampling_seeds': [42, 43], 'paired_clips': records,
        'paired_source_ids': source_by_id, 'matrix_rows': 640, 'row_metric_JSON_SHA_reverified': 640,
        'training_scientific_updates': 6144, 'metric_names': list(METRICS),
        'aggregation': 'Average sampler42/43 within each clip, then equal32clips. Each clip has a distinct source; this equals source-macro. SD is across clips, not training seeds.',
        'curves': curves, 'paired_comparisons': comparisons, 'final_comparisons': final,
        'validation_selection': {'criterion': 'minimum mean all33 LPIPS at2048 among both method weights; exact tie chooses smaller weight',
            'selected_arm': selected, 'selected_weight': weight_by_arm[selected], 'all_candidates_reported': True,
            'heldout_test_claimed': False},
        'efficiency': efficiency,
        'bootstrap': {'draws': 10000, 'seed': 20261102, 'generator': 'PCG64', 'draw_indices_sha256': indices_sha,
                      'resampling_unit': 'paired clip after sampler mean', 'common_draws': True,
                      'interval': '95% percentile with linear quantile interpolation',
                      'scope': 'descriptive conditional on one method training seed', 'p_values_computed': False},
        'quality_definition': 'LPIPS at fixed updates and its paired descriptive clip interval; PSNR/SSIM tradeoffs and temporal metrics fully reported.',
        'efficiency_definition': 'Earliest measured checkpoint meeting baseline@2048 LPIPS versus recorded cumulative CUDA-synchronized training-step seconds; no interpolation.',
        'time_axis': 'Recorded cumulative_training_seconds; not pure CUDA kernel/NVML active time. Common8192-step initialization and offline preparation are excluded. Recoveries and discarded-work accounting follow the bound training records.',
        'limitations': ['The weights/checkpoints are selected using the already reused val32; this is development validation, not heldout test.',
            'Paired clip intervals and sampling repeats do not establish training-seed reproducibility or selection-adjusted/multiple-comparison significance.',
            'Only fixed full33-frame128x224 crops,x4 clean bicubic; these are not full-length source videos or a full-resolution/external test.',
            'Time-to-target uses only actually measured512/1024/2048 checkpoints; no exact/interpolated convergence speed claim.',
            'An LPIPS gain can coexist with negative PSNR/SSIM fidelity tradeoffs. No old EMA G2/G3/G4 criterion applies.'],
        'GPU_work_performed_by_analysis': False, 'loss_used_to_infer_quality': False,
    }


def render_markdown(report):
    arms = list(report['curves']); base = arms[0]
    lines = ['# HR-VAE clean ×4：质量与训练效率', '',
        '完整640条评价；3个训练arm共6144次更新，方法训练seed仅20261102。每clip先平均采样42/43，再等权平均32个不同来源clip。', '',
        '**表内 mean ± SD 是clip sample SD（ddof=1），不是训练seed SD。主指标为全部33帧LPIPS。**', '',
        '| Arm / λ | 更新 | 同步训练秒 | LPIPS↓ | PSNR↑ dB | SSIM↑ | tLP100↓ | tOF↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in arms:
        for update in UPDATES:
            point = report['curves'][arm][update]
            cells = [f'{point["metrics"][m]["mean"]:.5f} ± {point["metrics"][m]["clip_sample_SD_ddof1"]:.5f}' for m in METRICS]
            lines.append(f'| {arm} / {point["weight"]} | {update} | {point["cumulative_training_seconds"]:.3f} | '+' | '.join(cells)+' |')
    lines += ['', '## 方法相对baseline的配对结果', '',
              'LPIPS/tLP100/tOF取baseline−method，PSNR/SSIM取method−baseline；正值均有利于方法。区间为10000次PCG64配对clip bootstrap的95%linear-percentile描述区间，无p-value。', '',
              '| 方法 | 更新 | LPIPS收益 [95% CI] | PSNR差 [95% CI] | SSIM差 [95% CI] |',
              '| --- | ---: | ---: | ---: | ---: |']
    for arm, points in report['paired_comparisons'].items():
        for update in UPDATES[1:]:
            cells=[]
            for metric in METRICS[:3]:
                item=points[update][metric];lo,hi=item['paired_clip_95pct_CI']
                cells.append(f'{item["mean"]:.7f} [{lo:.7f}, {hi:.7f}]')
            lines.append(f'| {arm} | {update} | '+' | '.join(cells)+' |')
    lines += ['', '## 最终质量与保真取舍', '']
    translate = {'single_training_seed_LPIPS_improvement':'本训练seed观察到LPIPS改善',
                 'no_observed_gain':'未见LPIPS增益', 'insufficient_evidence':'clip区间证据不足'}
    for arm, item in report['final_comparisons'].items():
        lines.append(f'- {arm}（λ={item["weight"]}）：{translate[item["LPIPS_improvement_status"]]}；'
                     f'PSNR保真tradeoff={item["PSNR_fidelity_tradeoff"]}，SSIM保真tradeoff={item["SSIM_fidelity_tradeoff"]}。')
    selection=report['validation_selection']
    lines += ['', f'按事前规则，2048步val LPIPS最低的方法候选为 **{selection["selected_arm"]}（λ={selection["selected_weight"]}）**；精确并列优先λ=.01。两个候选均完整报告。该选择不等于通过独立test。', '', '## 效率', '']
    eff=report['efficiency']
    lines.append(f'固定目标：baseline在2048步的LPIPS={eff["target_LPIPS"]:.7f}。')
    if eff['status']=='uninterpretable_initial_already_meets_target':
        lines.append('**共享step0已经达到该目标，baseline本轮没有形成更难的质量目标，不能据此判断方法加速。**')
    else:
        lines += ['', '| Arm | 最早实测达标更新 | 同步训练秒 | 相对baseline耗时比的倒数 |', '| --- | ---: | ---: | ---: |']
        for arm, crossing in eff['crossings'].items():
            if crossing['update'] is None: lines.append(f'| {arm} | 已测checkpoint均未达到 | — | — |')
            else:
                factor=eff['speedups_vs_baseline'][arm]
                lines.append(f'| {arm} | {crossing["update"]} | {crossing["cumulative_training_seconds"]:.3f} | {factor:.4f} |')
        lines.append('\n仅比较实测checkpoint，不插值推断精确收敛时间。')
    lines += ['', '训练时间是绑定记录中的cumulative_training_seconds（CUDA同步训练step墙钟秒），不是纯kernel时间。共同Base初始化、离线准备不计入；故障恢复/丢弃工作的计费范围以训练记录为准。', '',
              '![质量随更新变化](quality_vs_updates.png)', '', '![质量随同步训练时间变化](quality_vs_training_seconds.png)', '',
              '## 范围与局限', '']
    lines += ['- '+x for x in report['limitations']]
    lines += ['', f'Protocol SHA：`{report["protocol_sha256"]}`；study SHA：`{report["study_file"]["sha256"]}`。完整5指标配对区间、source/CP/源码绑定见[机器结果](analysis.json)。']
    return '\n'.join(lines)+'\n'


def plots(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#222222', '#0072B2', '#D55E00');markers = ('o','s','^')
    panel_metrics = ('lpips_alex_v0_1','psnr_rgb_db','tLP100_alex_v0_1_TecoGAN_crop')
    titles = ('All33 LPIPS (lower better)','All33 RGB PSNR (higher better)','Original-pair tLP100 (lower better)')
    for time_axis, filename in ((False,'quality_vs_updates.png'),(True,'quality_vs_training_seconds.png')):
        fig, axes = plt.subplots(1,3,figsize=(13.5,4.2),constrained_layout=True)
        for i,(arm,points) in enumerate(report['curves'].items()):
            xs=[points[u]['cumulative_training_seconds'] if time_axis else u for u in UPDATES]
            for axis,metric,title in zip(axes,panel_metrics,titles):
                axis.plot(xs,[points[u]['metrics'][metric]['mean'] for u in UPDATES],
                          marker=markers[i],color=colors[i],label=f'{arm}, lambda={points[2048]["weight"]}',linewidth=1.7)
                axis.set_title(title,fontsize=10);axis.grid(alpha=.25)
                axis.set_xlabel('Recorded synchronized training-step seconds' if time_axis else 'Additional optimizer updates')
                if not time_axis:axis.set_xticks(UPDATES)
        axes[0].legend(fontsize=8)
        fig.suptitle('Clean x4 val32 | 1 training seed | mean over paired clip/sample measurements\nLines guide the eye; efficiency uses observed checkpoints only',fontsize=11)
        fig.savefig(destination/filename,dpi=180);plt.close(fig)


def analyze(run, output_dir, *, make_plots=True):
    report=compute(run);destination=Path(output_dir).resolve()
    need(not destination.exists(),'Use a new analysis output directory; preserve earlier reports')
    destination.mkdir(parents=True)
    (destination/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (destination/'report.md').write_text(render_markdown(report))
    if make_plots:plots(report,destination)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args(argv)
    destination=args.output_dir or WORKSPACE/'outputs'/('HRVAE_clean_'+args.run.name)
    result=analyze(args.run,destination,make_plots=not args.no_plots)
    print(json.dumps({'output':str(destination),'selected':result['validation_selection'],
                      'final_comparisons':result['final_comparisons'],'efficiency':result['efficiency']},
                     ensure_ascii=False,indent=2,allow_nan=False))
    return result


if __name__=='__main__':main()
