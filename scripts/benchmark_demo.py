"""Bounded, warm-cache CPU timing of the fixed seed-43 local demo."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_baseline as full
from recover_full_scoring import validate_scores
from audio_deepfake.resources import memory_snapshot
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = full.ROOT
PLAN = ROOT / 'docs/validation/demo_latency_plan.json'
RESULT = ROOT / 'docs/validation/demo_latency.json'
SOURCES = ('scripts/benchmark_demo.py', 'src/audio_deepfake/demo.py',
           'src/audio_deepfake/audio.py', 'src/audio_deepfake/features.py',
           'src/audio_deepfake/baseline.py', 'src/audio_deepfake/resources.py')


def write_new(path, value):
    """Never replace an earlier timing plan or result."""
    with Path(path).open('x', encoding='utf-8') as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write('\n')


def resource_check():
    snapshot = memory_snapshot()
    if (snapshot.get('system_available_ram_bytes') is None
            or snapshot['system_available_ram_bytes'] < 1024**3
            or snapshot['peak_rss_bytes'] > 2 * 1024**3
            or shutil.disk_usage(ROOT).free < 20 * 1024**3):
        raise RuntimeError('Benchmark resource gate: RAM, peak memory or disk budget')
    return snapshot


def summarize(records):
    if not records:
        raise ValueError('No measured records')
    seconds = np.asarray([row['wall_seconds'] for row in records], dtype=float)
    durations = np.asarray([row['duration_seconds'] for row in records], dtype=float)
    if (not np.all(np.isfinite(seconds)) or not np.all(np.isfinite(durations))
            or np.any(seconds < 0) or np.any(durations <= 0)):
        raise ValueError('Timing and duration must be finite, duration positive')
    rtfs = seconds / durations
    return {'observations': len(records), 'mean_ms': float(seconds.mean() * 1000),
            'p50_ms': float(np.percentile(seconds, 50) * 1000),
            'p95_ms': float(np.percentile(seconds, 95) * 1000),
            'mean_record_rtf': float(rtfs.mean()), 'p95_record_rtf': float(np.percentile(rtfs, 95)),
            'total_measured_seconds': float(seconds.sum()),
            'total_audio_seconds_with_repetitions': float(durations.sum()),
            'aggregate_rtf': float(seconds.sum() / durations.sum()),
            'percentile_method': 'NumPy percentile, linear interpolation'}


def build_plan():
    config_path = ROOT / 'configs/full_seed43.toml'
    config = tomllib.loads(config_path.read_text(encoding='utf-8'))
    full.validate_config(config)
    paths = config['paths']
    groups, pins = full.official_selection(ROOT / paths['protocol_dir'], ROOT / paths['selection_manifest'])
    selected = groups['dev'][:20]
    journal_path = ROOT / paths['audio_journal']
    journal = json.loads(journal_path.read_text(encoding='utf-8'))
    if journal.get('status') != 'ok' or journal.get('verified_records') != 50224:
        raise ValueError('Complete preparation journal is required')
    scores_path = ROOT / 'outputs/baseline-full-seed43/scores.csv'
    scores = dict(validate_scores(scores_path, groups['dev']))
    records = []
    for row in selected:
        resource_check()
        audit = journal['records'][row.audio_path]
        audio = ROOT / paths['audio_root'] / row.audio_path
        if full.digest(audio) != audit['sha256'] or audio.stat().st_size != audit['size_bytes']:
            raise ValueError('Audio differs from verified public dataset')
        records.append({**asdict(row), 'audio_sha256': audit['sha256'],
                        'duration_seconds': audit['duration_seconds'], 'saved_clean_score': scores[row.utterance_id]})
    model = ROOT / 'outputs/baseline-full-seed43/model.joblib'
    model_sha = full.digest(model)
    if model_sha != '08b77256906a546d5eabbe9fa2cc5bbef0430d7d5dc7b10fc5b9f9e8449a19d0':
        raise ValueError('Fixed seed-43 model hash differs')
    plan = {'schema_version': 1, 'created_utc': full.timestamp(), 'status': 'frozen',
            'purpose': 'Small engineering timing; not an accuracy experiment or production guarantee',
            'model_seed': 43, 'model_selection': 'Fixed seed 43 for future demo timing; no best-seed selection',
            'model_sha256': model_sha, 'model_path': model.relative_to(ROOT).as_posix(),
            'config_sha256': full.digest(config_path), 'input_sha256': {**pins,
                'audio_journal': full.digest(journal_path), 'clean_scores': full.digest(scores_path)},
            'source_sha256': {name: full.digest(ROOT / name) for name in SOURCES},
            'selection_policy': 'First 20 official CM dev protocol rows, unchanged order; no duration selection',
            'sample_labels': dict(Counter(row.label for row in selected)),
            'sample_speakers': sorted({row.speaker_id for row in selected}),
            'records': records, 'repetitions': 3, 'warmup': 'One complete score_audio call on the first selected recording',
            'measured_scope': 'Wall time around score_audio: bounded file read, decode, LFCC, GMM and return metadata; excludes model load and audio hash validation',
            'cache_policy': 'Warm process and warm OS file cache; hashes read each selected file before timing; not cold I/O',
            'model_load_scope': 'Separately timed trusted model hash validation, metadata checks and deserialization',
            'threads': 1, 'max_total_seconds': 120, 'clean_score_absolute_tolerance': 1e-10,
            'resource_limits': {'min_available_ram_bytes': 1024**3,
                                'max_process_peak_bytes': 2 * 1024**3, 'min_free_disk_bytes': 20 * 1024**3},
            'eval_enabled': False, 'transformed_audio': False}
    return plan, config


def execute(plan_path=PLAN, result_path=RESULT, machine_path=None):
    if Path(plan_path).exists() or Path(result_path).exists():
        raise ValueError('Existing benchmark plan/result: refusing overwrite')
    from audio_deepfake.demo import load_demo_model, score_audio
    plan, config = build_plan()
    machine = {'platform': platform.platform(), 'machine': platform.machine(),
               'processor': platform.processor(), 'python': platform.python_version()}
    if machine_path:
        # An explicitly supplied public machine audit must contain no private paths.
        machine['resource_audit'] = json.loads(Path(machine_path).read_text(encoding='utf-8'))
    plan['machine'] = machine
    write_new(plan_path, plan)
    initial = resource_check()
    started = time.perf_counter()
    observations = []
    with threadpool_limits(limits=1):
        load_start = time.perf_counter()
        model = load_demo_model()
        load_seconds = time.perf_counter() - load_start
        first = ROOT / config['paths']['audio_root'] / plan['records'][0]['audio_path']
        score_audio(model, first)
        for repetition in range(1, plan['repetitions'] + 1):
            for row in plan['records']:
                resource_check()
                if time.perf_counter() - started > plan['max_total_seconds']:
                    raise RuntimeError('Benchmark total time budget exceeded')
                audio = ROOT / config['paths']['audio_root'] / row['audio_path']
                tick = time.perf_counter()
                scored = score_audio(model, audio)
                wall = time.perf_counter() - tick
                if (not math.isfinite(scored['score'])
                        or abs(scored['score'] - row['saved_clean_score']) > plan['clean_score_absolute_tolerance']
                        or scored['duration_seconds'] != row['duration_seconds']):
                    raise ValueError('Demo score or decoded duration differs from saved clean result')
                observations.append({'utterance_id': row['utterance_id'], 'repetition': repetition,
                                     'duration_seconds': scored['duration_seconds'], 'wall_seconds': wall,
                                     'decode_seconds': scored['decode_seconds'],
                                     'inference_seconds': scored['inference_seconds'],
                                     'score': scored['score'], 'saved_score_absolute_error': abs(scored['score'] - row['saved_clean_score'])})
        pools = [{key: (Path(value).name if key == 'filepath' else value)
                  for key, value in pool.items()} for pool in threadpool_info()]
    final = resource_check()
    if any(pool['num_threads'] != 1 for pool in pools):
        raise ValueError('Numerical library was not restricted to one thread')
    if {name: full.digest(ROOT / name) for name in SOURCES} != plan['source_sha256']:
        raise ValueError('Benchmark source changed during measurement')
    result = {'schema_version': 1, 'status': 'complete', 'completed_utc': full.timestamp(),
              'plan_path': 'docs/validation/demo_latency_plan.json', 'plan_sha256': full.digest(Path(plan_path)),
              'model_seed': 43, 'model_sha256': plan['model_sha256'], 'machine': machine,
              'model_load_seconds': load_seconds, 'summary': summarize(observations),
              'observations': observations, 'clean_score_comparison': 'All 60 observations within 1e-10 of saved seed-43 scores',
              'resources': {'initial': initial, 'final': final}, 'threadpools': pools,
              'limitations': ['20 fixed dev recordings, three repeated passes; observations are dependent',
                              'First 20 official rows are bonafide from one speaker; timing does not cover attack or speaker diversity',
                              'Warm file cache and process; model load excluded from per-record timings',
                              'Specific local CPU and software; no production or hardware-general latency claim'],
              'eval_enabled': False, 'training_started': False}
    write_new(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--machine-audit', type=Path)
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(execute(machine_path=args.machine_audit)['summary'], indent=2))
    else:
        print(json.dumps(build_plan()[0], indent=2))


if __name__ == '__main__':
    main()
