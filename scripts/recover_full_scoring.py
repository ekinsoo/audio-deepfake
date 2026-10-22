"""Resume only missing dev scores from the pinned local seed-42 checkpoint.

Run preflight once, then each chunk in a fresh supervised process, then finalize.
All original artifacts are read-only; no fitting or evaluation-split access occurs.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time
import tomllib

# Permit both direct script execution and importlib-based tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_baseline as full
from audio_deepfake.baseline import load_model, score_features
from audio_deepfake.audio import load_audio
from audio_deepfake.features import extract_lfcc
from audio_deepfake.cli import environment_info
from threadpoolctl import threadpool_limits

ROOT = full.ROOT
ORIGINAL = ROOT / 'outputs/baseline-full-seed42'
PRIOR_JOB = ROOT / 'outputs/jobs/full-seed42-20261009T235915974Z/status.json'
CONFIG = ROOT / 'configs/full_seed42.toml'
MODEL_SHA256 = '701fc3d8107cc8b2696b5552dd9e1b4751e44d2d197cfd5d1dfe7e4814add101'
PREFIX_COUNT = 22815
TOTAL_DEV = 24844
MAX_CHUNK = 1000
RESOURCE_ORIGINAL_SHA256 = 'ebc6d201f0cc9203d9d935ab712054fce809bbd73c274be3961b5941ddefc615'
RESOURCE_FIXED_SHA256 = '9d68202f7335b17e83f2138600f0f1c7312adab46c3d73609d61751800282e7a'
FIELDS = ('utterance_id', 'split', 'label', 'attack_id', 'score')
CORE_SOURCES = ('scripts/run_full_baseline.py', 'src/audio_deepfake/baseline.py',
                'src/audio_deepfake/features.py', 'src/audio_deepfake/audio.py',
                'src/audio_deepfake/manifests.py', 'src/audio_deepfake/metrics.py',
                'src/audio_deepfake/resources.py', 'src/audio_deepfake/cli.py',
                'configs/full_seed42.toml')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_scores(path: Path, rows) -> list[tuple[str, float]]:
    """Require exact metadata, order, cardinality and finite values."""
    with path.open(encoding='utf-8', newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError('Score header differs')
        supplied = list(reader)
    if len(supplied) != len(rows):
        raise ValueError('Score count differs')
    result = []
    for record, row in zip(supplied, rows):
        if (set(record) != set(FIELDS) or any(record[name] != getattr(row, name) for name in FIELDS[:-1])):
            raise ValueError('Score metadata/order differs')
        value = float(record['score'])
        if not math.isfinite(value):
            raise ValueError('Non-finite score')
        result.append((row.utterance_id, value))
    return result


def validate_original(config: dict) -> dict:
    full.validate_config(config)
    if config['experiment']['seed'] != 42:
        raise ValueError('Recovery accepts seed 42 only')
    state = read_json(ORIGINAL / 'run_state.json')
    expected = dict(status='failed', stage='dev-scoring', train_records_processed=25380,
                    dev_records_processed=PREFIX_COUNT, model_sha256=MODEL_SHA256,
                    error_type='RuntimeError', error='Resource stop: process peak exceeds 2 GiB')
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError('Original failed-state provenance differs')
    prior = read_json(PRIOR_JOB)
    if (prior.get('status') != 'stopped' or prior.get('seed') != 42
            or prior.get('eval_enabled') is not False
            or prior.get('reason') != 'RuntimeError: child_peak_memory_budget_exceeded'):
        raise ValueError('Original supervisor provenance differs')
    hashes = {name: full.digest(ROOT / name) for name in CORE_SOURCES}
    resource_name = 'src/audio_deepfake/resources.py'
    if (prior.get('source_sha256', {}).get(resource_name) != RESOURCE_ORIGINAL_SHA256
            or hashes[resource_name] != RESOURCE_FIXED_SHA256):
        raise ValueError('Resource instrumentation fix hashes differ')
    if any(prior.get('source_sha256', {}).get(name) != value for name, value in hashes.items() if name != resource_name):
        raise ValueError('Training source/config hashes differ from current source')
    if read_json(ORIGINAL / 'config.json') != config:
        raise ValueError('Original configuration differs')
    if full.digest(ORIGINAL / 'model.joblib') != MODEL_SHA256:
        raise ValueError('Pinned model SHA-256 differs; refusing unpickle')
    return dict(source_sha256=hashes, original_source_sha256={name:prior['source_sha256'][name] for name in CORE_SOURCES},
                instrumentation_fix=dict(path=resource_name, original_sha256=RESOURCE_ORIGINAL_SHA256,
                                         current_sha256=RESOURCE_FIXED_SHA256,
                                         reason='Reuse ctypes classes to avoid pointer-type cache growth; scoring definition unchanged.'),
                original_artifact_sha256={
        name: full.digest(ORIGINAL / name) for name in ('model.joblib', 'scores.csv', 'config.json', 'run_state.json')},
        prior_supervisor_sha256=full.digest(PRIOR_JOB))


def trusted_model(groups, lfcc, gmm):
    # This independent check must precede every joblib load, including chunks.
    if full.digest(ORIGINAL / 'model.joblib') != MODEL_SHA256:
        raise ValueError('Pinned model SHA-256 differs; refusing unpickle')
    model = load_model(ORIGINAL / 'model.joblib')
    metadata = model.metadata
    if (model.lfcc_config != lfcc or model.gmm_config != gmm
            or metadata.get('lfcc_config') != asdict(lfcc)
            or metadata.get('gmm_config') != asdict(gmm)
            or metadata.get('training_ids') != sorted(row.utterance_id for row in groups['train'])
            or metadata.get('normalization') != 'none' or metadata.get('covariance_type') != 'diag'
            or metadata.get('format_version') != 1):
        raise ValueError('Saved model training IDs/configuration differ')
    expected_versions = {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'soundfile')}
    if metadata.get('versions') != expected_versions:
        raise ValueError('Saved model package versions differ')
    expected_sources = {name: full.digest(ROOT / 'src/audio_deepfake' / name) for name in ('baseline.py', 'features.py')}
    if metadata.get('source_sha256') != expected_sources:
        raise ValueError('Saved model source hashes differ')
    return model


def validate_overlap(model, checked, prefix):
    """Reproduce three saved prefix scores before permitting score reuse."""
    guard = checked['guard']
    results = []
    with threadpool_limits(limits=1):
        for index in (0, PREFIX_COUNT // 2, PREFIX_COUNT - 1):
            guard.check()
            row = checked['groups']['dev'][index]
            value = score_features(model, extract_lfcc(load_audio(checked['root'] / row.audio_path,
                                  sample_rate=checked['lfcc'].sample_rate), checked['lfcc']))
            original = prefix[index][1]
            if not math.isclose(value, original, rel_tol=0, abs_tol=1e-10):
                raise ValueError('Checkpoint overlap score differs')
            results.append(dict(index=index, utterance_id=row.utterance_id, original_score=original,
                                recovered_score=value, absolute_difference=abs(value-original)))
            guard.check()
    return results


def preflight(output: Path) -> dict:
    if output.exists():
        raise ValueError('Recovery output already exists')
    started = time.perf_counter()
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    provenance = validate_original(config)
    checked = full.preflight(config, output)
    prefix = validate_scores(ORIGINAL / 'scores.csv', checked['groups']['dev'][:PREFIX_COUNT])
    model = trusted_model(checked['groups'], checked['lfcc'], checked['gmm'])
    overlap = validate_overlap(model, checked, prefix)
    plan = dict(status='ok', scope='scoring-only-recovery-seed42', created_utc=full.timestamp(),
                prefix_count=PREFIX_COUNT, total_dev=TOTAL_DEV, max_chunk=MAX_CHUNK,
                model_sha256=MODEL_SHA256, configuration=config, provenance=provenance,
                input_sha256=checked['input_sha256'], model_metadata=model.metadata, overlap_validation=overlap,
                recovery_source_sha256=full.digest(Path(__file__)),
                preflight_seconds=time.perf_counter()-started)
    output.mkdir(parents=True)
    full.atomic_json(output / 'plan.json', plan)
    return plan


def frozen_plan(output: Path):
    plan = read_json(output / 'plan.json')
    if (plan.get('status') != 'ok' or plan.get('scope') != 'scoring-only-recovery-seed42'
            or plan.get('prefix_count') != PREFIX_COUNT or plan.get('total_dev') != TOTAL_DEV
            or plan.get('max_chunk') != MAX_CHUNK or plan.get('model_sha256') != MODEL_SHA256
            or plan.get('recovery_source_sha256') != full.digest(Path(__file__))):
        raise ValueError('Recovery plan pins differ')
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    if plan['configuration'] != config or validate_original(config) != plan['provenance']:
        raise ValueError('Recovery provenance changed')
    paths = config['paths']
    groups, hashes = full.official_selection(Path(paths['protocol_dir']), Path(paths['selection_manifest']))
    hashes.update(audio_journal=full.digest(Path(paths['audio_journal'])),
                  asv_scores=full.digest(Path(paths.get('asv_scores', full.DEFAULT_ASV))),
                  reference_evaluator=full.digest(Path(paths.get('reference_evaluator', full.DEFAULT_REFERENCE))))
    if hashes != plan['input_sha256']:
        raise ValueError('Recovery input hashes changed')
    validate_scores(ORIGINAL / 'scores.csv', groups['dev'][:PREFIX_COUNT])
    return plan, config, groups


def chunk(output: Path, start: int, stop: int) -> dict:
    if type(start) is not int or type(stop) is not int or not PREFIX_COUNT <= start < stop <= TOTAL_DEV or stop-start > MAX_CHUNK:
        raise ValueError('Invalid recovery chunk range')
    destination = output / f'chunk-{start}-{stop}.csv'
    record_path = destination.with_suffix('.json')
    temporary = destination.with_suffix('.csv.tmp')
    if any(path.exists() for path in (destination, record_path, temporary)):
        raise ValueError('Chunk output already exists')
    started = time.perf_counter()
    plan, config, groups = frozen_plan(output)
    root = Path(config['paths']['audio_root'])
    guard = full.ResourceGuard(root)
    initial = guard.check(initial=True)
    guard.started = time.perf_counter()
    lfcc, gmm = full.validate_config(config)
    model = trusted_model(groups, lfcc, gmm)
    if model.metadata != plan['model_metadata']:
        raise ValueError('Frozen model metadata changed')
    journal = read_json(config['paths']['audio_journal'])
    rows = groups['dev'][start:stop]
    with threadpool_limits(limits=1), temporary.open('x', encoding='utf-8', newline='') as target:
        writer = csv.writer(target, lineterminator='\n'); writer.writerow(FIELDS)
        for index, row in enumerate(rows, start+1):
            guard.check()
            path, entry = root / row.audio_path, journal['records'][row.audio_path]
            if path.stat().st_size != entry['size_bytes'] or full.digest(path) != entry['sha256']:
                raise ValueError(f'Chunk audio hash/size differs: {row.utterance_id}')
            value = score_features(model, extract_lfcc(load_audio(path, sample_rate=lfcc.sample_rate), lfcc))
            guard.check()
            writer.writerow((row.utterance_id, row.split, row.label, row.attack_id, value))
            if index % 100 == 0:
                print(f'Recovery scored {index}/{TOTAL_DEV}', flush=True)
        target.flush(); full.os.fsync(target.fileno())
    validate_scores(temporary, rows)
    memory = guard.check()
    temporary.replace(destination)
    result = dict(status='ok', start=start, stop=stop, count=stop-start,
                  model_sha256=MODEL_SHA256, score_sha256=full.digest(destination),
                  plan_sha256=full.digest(output / 'plan.json'), finished_utc=full.timestamp(),
                  recovery_seconds=time.perf_counter()-started, memory_start=initial, memory_after=memory)
    full.atomic_json(record_path, result)
    return result


def collect_chunks(output: Path, rows) -> tuple[list, list]:
    """Accept only a complete, non-overlapping, gap-free suffix."""
    records = []
    files = set(output.glob('chunk-*.csv'))
    for record_path in output.glob('chunk-*.json'):
        record = read_json(record_path)
        start, stop = record.get('start'), record.get('stop')
        if (type(start) is not int or type(stop) is not int or not PREFIX_COUNT <= start < stop <= TOTAL_DEV
                or stop-start > MAX_CHUNK or record.get('count') != stop-start or record.get('status') != 'ok'
                or record.get('model_sha256') != MODEL_SHA256
                or record.get('plan_sha256') != full.digest(output / 'plan.json')):
            raise ValueError('Invalid chunk record')
        path = output / f'chunk-{start}-{stop}.csv'
        if record_path != path.with_suffix('.json') or path not in files or full.digest(path) != record.get('score_sha256'):
            raise ValueError('Chunk artifact/hash differs')
        records.append(record)
    if len(files) != len(records):
        raise ValueError('Uncommitted chunk CSV exists')
    cursor, pairs = PREFIX_COUNT, []
    for record in sorted(records, key=lambda item: item['start']):
        if record['start'] != cursor:
            raise ValueError('Chunk gap/overlap')
        cursor = record['stop']
        pairs.extend(validate_scores(output / f"chunk-{record['start']}-{cursor}.csv", rows[record['start']:cursor]))
    if cursor != TOTAL_DEV:
        raise ValueError('Recovery chunks incomplete')
    return pairs, records


def finalize(output: Path) -> dict:
    if any((output / name).exists() for name in ('scores.csv', 'scores.csv.tmp', 'summary.json')):
        raise ValueError('Final output already exists')
    started = time.perf_counter()
    plan, config, groups = frozen_plan(output)
    suffix, chunks = collect_chunks(output, groups['dev'])
    # Recheck the complete original audio and reference inputs once more before metrics.
    checked = full.preflight(config, output, owned_output=True)
    if checked['input_sha256'] != plan['input_sha256']:
        raise ValueError('Final input hashes changed')
    guard = checked['guard']; guard.started = time.perf_counter()
    prefix = validate_scores(ORIGINAL / 'scores.csv', groups['dev'][:PREFIX_COUNT])
    pairs = prefix + suffix
    metrics = full.evaluate_scores(groups['dev'], pairs, checked['asv'], checked['reference'], evaluation_scope='full')
    temporary = output / 'scores.csv.tmp'
    with temporary.open('x', encoding='utf-8', newline='') as target:
        writer = csv.writer(target, lineterminator='\n'); writer.writerow(FIELDS)
        for row, (_, value) in zip(groups['dev'], pairs):
            writer.writerow((row.utterance_id, row.split, row.label, row.attack_id, value))
        target.flush(); full.os.fsync(target.fileno())
    validate_scores(temporary, groups['dev'])
    memory = guard.check()
    temporary.replace(output / 'scores.csv')
    summary = dict(status='ok', scope='full-official-train-dev-adaptation-scoring-recovery',
                   finished_utc=full.timestamp(), experiment=config['experiment'], configuration=config,
                   splits={key: full.summarize(value) for key, value in groups.items()}, metrics=metrics,
                   model_metadata=plan['model_metadata'], model_sha256=MODEL_SHA256,
                   input_sha256=plan['input_sha256'], provenance=plan['provenance'],
                   output_sha256={'scores.csv':full.digest(output/'scores.csv'), 'plan.json':full.digest(output/'plan.json')},
                   archive_provenance=checked['archive'], environment=environment_info(), memory_after=memory,
                   recovery=dict(original_prefix_count=PREFIX_COUNT, newly_scored_count=TOTAL_DEV-PREFIX_COUNT,
                                 chunks=chunks, overlap_validation=plan['overlap_validation'],
                                 preflight_seconds=plan['preflight_seconds'],
                                 chunk_seconds=sum(record['recovery_seconds'] for record in chunks),
                                 finalize_seconds=time.perf_counter()-started,
                                 timing_scope='Recovery stages only; original training/scoring runtime is not reconstructed.'),
                   source_sha256={**plan['provenance']['source_sha256'], 'scripts/recover_full_scoring.py':full.digest(Path(__file__))},
                   limitations=['Official development split only; eval remains closed.',
                                'Fixed sklearn adaptation, not exact official MATLAB reproduction.',
                                'Dev thresholds use dev labels; no deployment or held-out eval claim.',
                                'Original run exceeded its 2 GiB memory limit after saving the model.',
                                'Fresh scoring processes preserve the same model and score definition.',
                                'Original artifacts remain unchanged; recovery timing excludes the original run.'])
    full.atomic_json(output / 'summary.json', summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'chunk', 'finalize'), required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/scoring-recovery-seed42'))
    parser.add_argument('--start', type=int)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()
    try:
        if args.mode == 'chunk':
            result = chunk(args.output_dir, args.start, args.stop)
        elif args.start is not None or args.stop is not None:
            raise ValueError('start/stop are valid only in chunk mode')
        else:
            result = preflight(args.output_dir) if args.mode == 'preflight' else finalize(args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Recovery error: {error}\n')
    print(json.dumps({'status':result['status'], 'mode':args.mode}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
