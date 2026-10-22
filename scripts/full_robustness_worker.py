"""Verified, bounded inference workers for the frozen full-dev stress test."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_full_robustness as freeze
import run_full_baseline as full
from recover_full_scoring import validate_scores
from audio_deepfake.audio import load_audio
from audio_deepfake.baseline import load_model, score_features
from audio_deepfake.features import extract_lfcc
from audio_deepfake.transforms import narrowband_roundtrip
from threadpoolctl import threadpool_limits

ROOT = full.ROOT
CONTRACT_SHA256 = '806fd09f488449d6e276e4b36b60ddc016441be2b62e7a223d3bc74d7ea6d913'
MAX_CHUNK = 2000
EXECUTION_SOURCES = ('scripts/full_robustness_worker.py', 'scripts/finalize_full_robustness.py',
                     'scripts/full_robustness_job.py', 'scripts/start_full_robustness.ps1')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_contract():
    """Recompute every frozen pin, partition and calibration operating point."""
    if full.digest(freeze.AUDIT) != CONTRACT_SHA256:
        raise ValueError('Frozen robustness contract SHA-256 differs')
    saved = read_json(freeze.AUDIT)
    rebuilt, calibration, analysis, _ = freeze.build_plan()
    for key, value in rebuilt.items():
        if key != 'created_utc' and saved.get(key) != value:
            raise ValueError(f'Frozen robustness contract differs: {key}')
    for name, expected in saved['partition_file_sha256'].items():
        if full.digest(ROOT / saved['local_partition_dir'] / name) != expected:
            raise ValueError('Frozen partition file changed')
    return saved, calibration, analysis


def validate_execution(output):
    execution = read_json(output / 'execution.json')
    if (execution.get('status') != 'frozen' or execution.get('protocol_sha256') != CONTRACT_SHA256
            or execution.get('seeds') != [42, 43, 44] or execution.get('chunk_size') != MAX_CHUNK
            or execution.get('per_seed_budget_seconds') != 1800):
        raise ValueError('Execution plan differs')
    expected = {name: full.digest(ROOT / name) for name in EXECUTION_SOURCES}
    if execution.get('source_sha256') != expected:
        raise ValueError('Execution sources changed')
    return execution


def configuration(seed):
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError('Expected frozen model seed 42, 43 or 44')
    return tomllib.loads((ROOT / f'configs/full_seed{seed}.toml').read_text(encoding='utf-8'))


def current_inputs(config, groups_hashes):
    paths = config['paths']
    return {**groups_hashes, 'audio_journal': full.digest(ROOT / paths['audio_journal']),
            'asv_scores': full.digest(ROOT / paths.get('asv_scores', full.DEFAULT_ASV)),
            'reference_evaluator': full.digest(ROOT / paths.get('reference_evaluator', full.DEFAULT_REFERENCE))}


def frozen_inputs(output):
    execution = validate_execution(output)
    contract, calibration, analysis = validate_contract()
    plan = read_json(output / 'preflight.json')
    config = configuration(42)
    groups, hashes = full.official_selection(ROOT / config['paths']['protocol_dir'],
                                             ROOT / config['paths']['selection_manifest'])
    if (plan.get('status') != 'ok' or plan.get('protocol') != freeze.PROTOCOL_ID
            or plan.get('contract_sha256') != CONTRACT_SHA256
            or plan.get('execution_sha256') != full.digest(output / 'execution.json')
            or plan.get('execution_source_sha256') != execution['source_sha256']
            or plan.get('worker_source_sha256') != full.digest(Path(__file__))
            or plan.get('audio_records_verified') != 50224 or plan.get('eval_enabled') is not False
            or plan.get('input_sha256') != current_inputs(config, hashes)
            or plan.get('input_sha256') != contract['official_input_sha256']):
        raise ValueError('Preflight pins changed')
    return plan, contract, config, groups, calibration, analysis


def trusted_model(seed, contract, groups):
    config = configuration(seed)
    lfcc, gmm = full.validate_config(config)
    pin = contract['seeds'][str(seed)]
    path = ROOT / pin['model_path']
    if full.digest(path) != pin['model_sha256']:
        raise ValueError('Model SHA-256 changed; refusing unpickle')
    model = load_model(path)
    metadata = model.metadata
    if (model.lfcc_config != lfcc or model.gmm_config != gmm
            or metadata.get('lfcc_config') != asdict(lfcc) or metadata.get('gmm_config') != asdict(gmm)
            or metadata.get('training_ids') != sorted(row.utterance_id for row in groups['train'])
            or metadata.get('normalization') != 'none' or metadata.get('covariance_type') != 'diag'
            or metadata.get('format_version') != 1
            or metadata.get('versions') != pin['packages']
            or metadata.get('versions') != {name: version(name) for name in pin['packages']}
            or metadata.get('source_sha256') != {name: full.digest(ROOT / 'src/audio_deepfake' / name)
                                                   for name in ('baseline.py', 'features.py')}):
        raise ValueError('Model training IDs, configuration or implementation changed')
    return model, config, lfcc


def paired_record(row, clean, narrow, group):
    if not all(math.isfinite(value) for value in (clean, narrow)):
        raise ValueError('Non-finite paired score')
    delta = narrow - clean
    average = clean / 2 + narrow / 2
    if not all(math.isfinite(value) for value in (delta, average)):
        raise ValueError('Non-finite derived paired score')
    return {name: getattr(row, name) for name in ('utterance_id', 'split', 'speaker_id', 'label', 'attack_id')} | {
        's_clean': clean, 's_narrow': narrow, 's_average': average,
        'delta': delta, 'abs_delta': abs(delta), 'group': group}


def validate_range(start, stop):
    if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= 24844
            or stop - start > MAX_CHUNK):
        raise ValueError('Invalid full-dev chunk range')


def preflight(output):
    if (output / 'preflight.json').exists():
        raise ValueError('Preflight already exists')
    started = time.perf_counter()
    execution = validate_execution(output)
    contract, _, _ = validate_contract()
    checked = full.preflight(configuration(42), output, owned_output=True)
    if checked['input_sha256'] != contract['official_input_sha256']:
        raise ValueError('Full verified inputs differ from frozen contract')
    for seed in (42, 43, 44):
        trusted_model(seed, contract, checked['groups'])
        checked['guard'].check()
    plan = dict(status='ok', protocol=freeze.PROTOCOL_ID, created_utc=full.timestamp(),
                contract_sha256=CONTRACT_SHA256, worker_source_sha256=full.digest(Path(__file__)),
                execution_sha256=full.digest(output / 'execution.json'),
                execution_source_sha256=execution['source_sha256'], input_sha256=checked['input_sha256'],
                audio_records_verified=50224, eval_enabled=False,
                initial_memory=checked['initial_memory'], final_memory=checked['guard'].check(),
                preflight_seconds=time.perf_counter()-started)
    full.atomic_json(output / 'preflight.json', plan)
    return plan


def score_subset(output, seed, start, stop, *, reproduce_clean=False):
    started = time.perf_counter()
    plan, contract, _, groups, calibration, _ = frozen_inputs(output)
    model, config, lfcc = trusted_model(seed, contract, groups)
    root = ROOT / config['paths']['audio_root']
    guard = full.ResourceGuard(root)
    initial = guard.check(initial=True)
    guard.started = time.perf_counter()
    pin = contract['seeds'][str(seed)]
    clean = dict(validate_scores(ROOT / pin['clean_scores_path'], groups['dev']))
    journal = read_json(ROOT / config['paths']['audio_journal'])
    calibration_ids = {row.utterance_id for row in calibration}
    records, clean_checks = [], []
    with threadpool_limits(limits=1):
        for index, row in enumerate(groups['dev'][start:stop], start):
            guard.check()
            path, entry = root / row.audio_path, journal['records'][row.audio_path]
            if path.stat().st_size != entry['size_bytes'] or full.digest(path) != entry['sha256']:
                raise ValueError(f'Chunk audio hash/size differs: {row.utterance_id}')
            waveform = load_audio(path, sample_rate=lfcc.sample_rate)
            if reproduce_clean:
                value = score_features(model, extract_lfcc(waveform, lfcc))
                if not math.isclose(value, clean[row.utterance_id], rel_tol=0, abs_tol=1e-10):
                    raise ValueError('Smoke clean inference differs from saved score')
                clean_checks.append(dict(utterance_id=row.utterance_id, saved_score=clean[row.utterance_id],
                                         reproduced_score=value, absolute_difference=abs(value-clean[row.utterance_id])))
            narrow = score_features(model, extract_lfcc(narrowband_roundtrip(waveform, lfcc.sample_rate), lfcc))
            records.append(paired_record(row, clean[row.utterance_id], narrow,
                                         'calibration' if row.utterance_id in calibration_ids else 'analysis'))
            guard.check()
            if (index + 1) % 100 == 0:
                print(f'Seed {seed}: scored {index+1}/24844', flush=True)
    return dict(status='ok', protocol=freeze.PROTOCOL_ID, seed=seed, start=start, stop=stop, rows=records,
                clean_reproduction=clean_checks, processing_seconds=time.perf_counter()-started,
                initial_memory=initial, final_memory=guard.check(), contract_sha256=CONTRACT_SHA256,
                worker_source_sha256=full.digest(Path(__file__)), preflight_sha256=full.digest(output / 'preflight.json'),
                execution_sha256=full.digest(output / 'execution.json'), input_sha256=plan['input_sha256'],
                model_sha256=pin['model_sha256'])


def chunk(output, seed, start, stop):
    validate_range(start, stop)
    destination = output / 'chunks' / f'seed{seed}_{start:05d}_{stop:05d}.json'
    if destination.exists():
        raise ValueError('Chunk already exists; preserve recoverable results')
    result = score_subset(output, seed, start, stop)
    destination.parent.mkdir(parents=True, exist_ok=True)
    full.atomic_json(destination, result)
    return {key: value for key, value in result.items() if key not in ('rows', 'clean_reproduction')}


def smoke(output):
    destination = output / 'smoke.json'
    if destination.exists():
        raise ValueError('Smoke output already exists')
    result = dict(status='ok', protocol=freeze.PROTOCOL_ID, scope='fixed-first-20-dev-per-seed-engineering-smoke',
                  no_metric_tuning=True, seeds=[score_subset(output, seed, 0, 20, reproduce_clean=True)
                                               for seed in (42, 43, 44)])
    full.atomic_json(destination, result)
    return dict(status='ok', scope=result['scope'], seeds=[42, 43, 44])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('preflight', 'chunk', 'smoke'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--start', type=int)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()
    result = (preflight(args.output_dir) if args.mode == 'preflight' else smoke(args.output_dir)
              if args.mode == 'smoke' else chunk(args.output_dir, args.seed, args.start, args.stop))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
