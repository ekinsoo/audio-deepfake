"""Run the fixed full official train/dev adaptation, preserving eval closure.

Standalone resource checks are cooperative. Use the external supervisor to
enforce limits during native GMM fitting as well as between recordings.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tomllib

from threadpoolctl import threadpool_info, threadpool_limits

from audio_deepfake.audio import load_audio
from audio_deepfake.baseline import GMMConfig, save_model, score_features, train_feature_stream
from audio_deepfake.cli import environment_info
from audio_deepfake.features import LFCCConfig, extract_lfcc
from audio_deepfake.manifests import ManifestRow, check_audio_files, parse_protocol, summarize, validate_splits
from audio_deepfake.metrics import DEFAULT_REFERENCE, REFERENCE_SHA256, evaluate_scores
from audio_deepfake.resources import memory_snapshot

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = {'train': 'ASVspoof2019.LA.cm.train.trn.txt', 'dev': 'ASVspoof2019.LA.cm.dev.trl.txt'}
PROTOCOL_SHA256 = {'train': '65a289e4009400b42bb4f1e47ac60938d93f7f595452d1f220d35defc3de8ab8',
                   'dev': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7'}
COUNTS = {'train': {'count': 25380, 'labels': {'bonafide': 2580, 'spoof': 22800}},
          'dev': {'count': 24844, 'labels': {'bonafide': 2548, 'spoof': 22296}}}
ASV_SHA256 = 'fe8114cf608c3062c141df5979892eba9664aa37a8c99e97157f129d679d3389'
DEFAULT_ASV = ROOT / 'data/metadata/asvspoof2019/metrics/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt'
ARCHIVE_MD5 = '30c98f11d8b2bc21f2c257bfd78bb5c5'
ARCHIVE_SIZE = 7640952520
SELECTION_SHA256 = '37da16080ddbf0c0253b1f093a11917e1c463f12b2f88d322c270587abf17458'
GIB = 1024**3


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as target:
        target.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_config(config: dict) -> tuple[LFCCConfig, GMMConfig]:
    experiment, data = config['experiment'], config['data']
    seed = experiment['seed']
    if type(seed) is not int or seed not in (42, 43, 44) or experiment['stage'] != 'full-train-dev':
        raise ValueError('Full confirmation requires stage full-train-dev and seed 42, 43 or 44')
    expected_data = dict(dataset='ASVspoof2019-LA', version='Zenodo-6906306-v1', sample_rate=16000,
                         train_count=25380, dev_count=24844, eval_enabled=False)
    if data != expected_data or data['eval_enabled'] is not False:
        raise ValueError('Requires complete official train/dev counts, fixed dataset/version and eval_enabled=false')
    lfcc, gmm = LFCCConfig(**config['features']), GMMConfig(seed=seed)
    if lfcc != LFCCConfig():
        raise ValueError('LFCC configuration must match the fixed pilot definition')
    expected_model = dict(family='lfcc-gmm', n_components=32, covariance_type='diag',
                          max_total_training_frames=100000, max_iter=50, reg_covar=1e-6, tol=1e-3)
    if config['model'] != expected_model:
        raise ValueError('Model configuration must remain fixed for confirmation')
    expected_resources = dict(workers=1, min_start_ram_gib=3, min_free_disk_gib=20,
                              min_running_ram_gib=1, max_process_peak_gib=2, max_processing_seconds=1800)
    if config['resources'] != expected_resources:
        raise ValueError('Full confirmation requires the fixed resource stop budgets')
    return lfcc, gmm


def canonical_hash(rows: list[ManifestRow]) -> str:
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=tuple(ManifestRow.__dataclass_fields__), lineterminator='\n')
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    return hashlib.sha256(text.getvalue().encode('utf-8')).hexdigest()


def official_selection(protocol_dir: Path, selection: Path) -> tuple[dict, dict]:
    groups = {}
    for split in ('train', 'dev'):
        path = protocol_dir / PROTOCOL_FILES[split]
        if digest(path) != PROTOCOL_SHA256[split]:
            raise ValueError(f'Official protocol SHA-256 mismatch: {split}')
        groups[split] = parse_protocol(path, split)
        counts = summarize(groups[split])
        if counts['count'] != COUNTS[split]['count'] or counts['labels'] != COUNTS[split]['labels']:
            raise ValueError(f'Official natural class counts mismatch: {split}')
    validate_splits(groups)
    expected = groups['train'] + groups['dev']
    with selection.open(encoding='utf-8', newline='') as source:
        supplied = [ManifestRow(**record) for record in csv.DictReader(source)]
    if supplied != expected or digest(selection) != canonical_hash(expected) or digest(selection) != SELECTION_SHA256:
        raise ValueError('Full selection must exactly preserve all official metadata and protocol order')
    return groups, {'selection_manifest': digest(selection), **{f'{split}_protocol': PROTOCOL_SHA256[split] for split in groups}}


class ResourceGuard:
    """Cooperative gates; the external supervisor can interrupt native fitting."""
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.started = None

    def check(self, *, initial=False) -> dict:
        snapshot = memory_snapshot()
        available = snapshot.get('system_available_ram_bytes')
        if available is None:
            raise RuntimeError('Available system RAM measurement is required')
        floor = (3 if initial else 1) * GIB
        if available < floor:
            raise RuntimeError('Resource stop: insufficient available system RAM')
        if snapshot['peak_rss_bytes'] > 2 * GIB:
            raise RuntimeError('Resource stop: process peak exceeds 2 GiB')
        if shutil.disk_usage(self.data_root).free < 20 * GIB:
            raise RuntimeError('Resource stop: free disk below 20 GiB')
        if self.started is not None and time.perf_counter() - self.started > 1800:
            raise RuntimeError('Resource stop: processing exceeds 30 minutes')
        return snapshot


def preflight(config: dict, output_dir: Path, *, owned_output=False) -> dict:
    if output_dir.exists() and not owned_output:
        raise ValueError('Output already exists; use a new directory')
    lfcc, gmm = validate_config(config)
    paths = config['paths']
    if paths.get('selection_sha256') != SELECTION_SHA256:
        raise ValueError('Full selection SHA-256 pin differs from frozen official rows')
    root = Path(paths['audio_root']).resolve()
    guard = ResourceGuard(root)
    initial_memory = guard.check(initial=True)
    groups, hashes = official_selection(Path(paths['protocol_dir']), Path(paths['selection_manifest']))
    journal_path = Path(paths['audio_journal'])
    journal = json.loads(journal_path.read_text(encoding='utf-8'))
    rows = groups['train'] + groups['dev']
    total = sum(v['count'] for v in COUNTS.values())
    if (journal.get('status') != 'ok' or journal.get('expected_records') != total
            or journal.get('verified_records') != total or set(journal.get('records', {})) != {r.audio_path for r in rows}):
        raise ValueError('Requires a complete exact train/dev preparation journal')
    if journal.get('source_protocol_sha256') != PROTOCOL_SHA256:
        raise ValueError('Preparation journal protocol hashes differ')
    expected_manifests = {'full_train.csv': canonical_hash(groups['train']),
                          'full_dev.csv': canonical_hash(groups['dev']),
                          'full_train_dev.csv': hashes['selection_manifest']}
    if journal.get('source_manifest_sha256') != expected_manifests:
        raise ValueError('Preparation journal manifest hashes differ')
    archive = journal.get('archive', {})
    if archive.get('verified') is not True or archive.get('md5') != ARCHIVE_MD5 or archive.get('size_bytes') != ARCHIVE_SIZE:
        raise ValueError('Preparation journal must report the pinned complete archive verification')
    missing = check_audio_files(rows, root)
    if missing:
        raise ValueError(f'Missing full-split audio: {missing[0]}')
    for index, row in enumerate(rows):
        if index % 100 == 0:
            guard.check()
        path, entry = root / row.audio_path, journal['records'][row.audio_path]
        if (not 0 < path.stat().st_size <= 2 * 1024**2 or path.stat().st_size != entry['size_bytes']
                or digest(path) != entry['sha256']):
            raise ValueError(f'Audio SHA-256/size differs from journal: {row.utterance_id}')
        if entry.get('sample_rate') != 16000 or entry.get('channels') != 1 or entry.get('subtype') != 'PCM_16':
            raise ValueError('Journal audio format differs from original mono PCM16 16 kHz')
    asv = Path(paths.get('asv_scores', DEFAULT_ASV))
    reference = Path(paths.get('reference_evaluator', DEFAULT_REFERENCE))
    if digest(asv) != ASV_SHA256 or digest(reference) != REFERENCE_SHA256:
        raise ValueError('Official dev ASV/reference SHA-256 mismatch')
    hashes.update(audio_journal=digest(journal_path), asv_scores=digest(asv), reference_evaluator=digest(reference))
    guard.check(initial=True)
    return dict(groups=groups, root=root, guard=guard, initial_memory=initial_memory, lfcc=lfcc, gmm=gmm,
                input_sha256=hashes, archive=archive, asv=asv, reference=reference)


def run_experiment(config_path: Path, output_dir: Path) -> dict:
    config = tomllib.loads(config_path.read_text(encoding='utf-8'))
    if output_dir.exists():
        raise ValueError('Output already exists; use a new directory')
    output_dir.mkdir(parents=True)
    state = dict(status='running', stage='preflight', started_utc=timestamp(), updated_utc=timestamp(),
                 train_records_processed=0, dev_records_processed=0,
                 budget_enforcement='Cooperative runner checks; use external supervisor for native-fit interruption')
    state_path = output_dir / 'run_state.json'
    atomic_json(state_path, state)
    try:
        plan = preflight(config, output_dir, owned_output=True)
        atomic_json(output_dir / 'config.json', config)
    except BaseException as error:
        state.update(status='failed', failed_utc=timestamp(), updated_utc=timestamp(),
                     error_type=type(error).__name__, error=str(error))
        atomic_json(state_path, state)
        raise
    state['stage'] = 'training'
    state['processing_started_utc'] = timestamp()
    atomic_json(state_path, state)
    guard = plan['guard']
    guard.started = time.perf_counter()
    cpu = time.process_time()
    def progress():
        state['updated_utc'] = timestamp()
        atomic_json(state_path, state)
    def features(row):
        guard.check()
        waveform = load_audio(plan['root'] / row.audio_path, sample_rate=plan['lfcc'].sample_rate)
        result = extract_lfcc(waveform, plan['lfcc'])
        guard.check()
        if row.split == 'train':
            state['train_records_processed'] += 1
            if state['train_records_processed'] % 100 == 0:
                progress()
        return result
    try:
        with threadpool_limits(limits=1):
            pools = [{k:v for k,v in p.items() if k != 'filepath'} for p in threadpool_info()]
            before = guard.check()
            model = train_feature_stream(plan['groups']['train'], features, lfcc_config=plan['lfcc'], gmm_config=plan['gmm'])
            trained = time.perf_counter()
            # Preserve the expensive fitted model even if scoring subsequently fails.
            temporary_model = output_dir / 'model.joblib.tmp'
            save_model(model, temporary_model)
            temporary_model.replace(output_dir / 'model.joblib')
            state.update(stage='dev-scoring', model_sha256=digest(output_dir / 'model.joblib'), model_saved_utc=timestamp())
            progress()
            guard.check()
            pairs = []
            scoring_started = time.perf_counter()
            with (output_dir / 'scores.csv').open('x', encoding='utf-8', newline='') as destination:
                writer = csv.writer(destination, lineterminator='\n')
                writer.writerow(('utterance_id','split','label','attack_id','score'))
                for index, row in enumerate(plan['groups']['dev'], 1):
                    value = score_features(model, features(row))
                    guard.check()
                    pairs.append((row.utterance_id, value))
                    writer.writerow((row.utterance_id,row.split,row.label,row.attack_id,value))
                    state['dev_records_processed'] = index
                    if index % 100 == 0:
                        destination.flush()
                        os.fsync(destination.fileno())
                        progress()
            scored = time.perf_counter()
            state['stage'] = 'metrics'; progress()
            metrics = evaluate_scores(plan['groups']['dev'], pairs, plan['asv'], plan['reference'], evaluation_scope='full')
            after = guard.check()
            finished = time.perf_counter()
        summary = dict(status='ok', scope='full-official-train-dev-adaptation', experiment=config['experiment'],
                       started_utc=state['started_utc'], finished_utc=timestamp(),
                       splits={k:summarize(v) for k,v in plan['groups'].items()},
                       input_sha256={**plan['input_sha256'],'config':digest(config_path)},
                       archive_provenance=plan['archive'], configuration=config,
                       model_metadata=model.metadata, metrics=metrics, environment=environment_info(),
                       workers=1, thread_limit=1, threadpools=pools,
                       memory_start=plan['initial_memory'], memory_before=before, memory_after=after,
                       free_disk_bytes_after=shutil.disk_usage(plan['root']).free,
                       timings=dict(training_seconds=trained-guard.started,
                                    model_checkpoint_seconds=scoring_started-trained,
                                    dev_scoring_seconds=scored-scoring_started,
                                    metric_seconds=finished-scored,processing_wall_seconds=finished-guard.started,
                                    processing_cpu_seconds=time.process_time()-cpu),
                       source_sha256={name:digest(ROOT/name) for name in (
                           'scripts/run_full_baseline.py','src/audio_deepfake/baseline.py',
                           'src/audio_deepfake/features.py','src/audio_deepfake/audio.py',
                           'src/audio_deepfake/manifests.py','src/audio_deepfake/metrics.py','src/audio_deepfake/resources.py')},
                       limitations=['Complete official train/dev IDs and natural class proportions; eval remains closed.',
                                    'Fixed sklearn adaptation, not exact official MATLAB baseline reproduction.',
                                    'All train recordings contribute candidate frames; at most 50,000 frames per class are fitted.',
                                    'Dev EER/min t-DCF thresholds are selected on dev; no held-out eval or deployment claim.',
                                    'Resource enforcement is cooperative inside this runner; external supervision covers native fitting.',
                                    'Processing time excludes preflight hashes/imports; includes periodic progress and model checkpoint writes.',
                                    'Process peak RSS includes imports and filesystem cache is uncontrolled.',
                                    'Joblib artifacts are trusted local files only; archive verification is preparation-journal provenance.'])
        summary['output_sha256']={name:digest(output_dir/name) for name in ('model.joblib','scores.csv','config.json')}
        atomic_json(output_dir/'summary.json',summary)
        state.update(status='ok',stage='complete'); progress()
        return summary
    except BaseException as error:
        state.update(status='failed',failed_utc=timestamp(),error_type=type(error).__name__,error=str(error),
                     elapsed_processing_seconds=time.perf_counter()-guard.started)
        progress()
        raise


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/full_seed42.toml'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        summary=run_experiment(args.config,args.output_dir)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        parser.exit(1,f'Full baseline error: {error}\n')
    print(json.dumps({'status':summary['status'],'metrics':summary['metrics'],'timings':summary['timings']},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
