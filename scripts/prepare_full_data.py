"""Prepare complete official LA train/dev splits from a verified cached archive."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pilot_data import ROOT, SIZE, MD5, SOURCES, digest
from fetch_pilot_audio import inspect_audio, save_report
from audio_deepfake.manifests import parse_protocol, validate_splits, summarize
from audio_deepfake.resources import memory_snapshot

RESERVE_BYTES = 20 * 1024**3
MIN_RAM_BYTES = 3 * 1024**3
MIN_RUNNING_RAM_BYTES = 1024**3
MEMBER_CAP = 2 * 1024**2
SAVE_EVERY = 250
COMBINED_MANIFEST_SHA256 = '37da16080ddbf0c0253b1f093a11917e1c463f12b2f88d322c270587abf17458'
SPECS = {
    'train': {'filename': 'ASVspoof2019.LA.cm.train.trn.txt', 'sha256': '65a289e4009400b42bb4f1e47ac60938d93f7f595452d1f220d35defc3de8ab8', 'size_bytes': 840120,
              'count': 25380, 'bonafide': 2580, 'spoof': 22800, 'speakers': 20, 'bonafide_speakers': 20, 'spoof_speakers': 20, 'spoof_per_attack': 3800},
    'dev': {'filename': 'ASVspoof2019.LA.cm.dev.trl.txt', 'sha256': '2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7', 'size_bytes': 822400,
            'count': 24844, 'bonafide': 2548, 'spoof': 22296, 'speakers': 20, 'bonafide_speakers': 20, 'spoof_speakers': 10, 'spoof_per_attack': 3716},
}


def reject_symlinks(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Paths must not use symlinks')


def save_full_report(report, path):
    reject_symlinks(path)
    reject_symlinks(path.with_suffix('.json.tmp'))
    save_report(report, path)


def read_official_splits(protocol_dir: Path):
    splits, hashes = {}, {}
    for split, spec in SPECS.items():
        path = protocol_dir / spec['filename']
        reject_symlinks(path)
        if path.stat().st_size != spec['size_bytes'] or digest(path) != spec['sha256']:
            raise ValueError(f'Official protocol size/SHA-256 mismatch: {split}')
        rows = parse_protocol(path, split)
        counts = summarize(rows)
        if counts['count'] != spec['count'] or counts['labels'] != {'bonafide': spec['bonafide'], 'spoof': spec['spoof']}:
            raise ValueError('Official count or natural class counts differ')
        if counts['speakers'] != spec['speakers']:
            raise ValueError('Official speaker counts differ')
        for label in ('bonafide', 'spoof'):
            if len({row.speaker_id for row in rows if row.label == label}) != spec[label + '_speakers']:
                raise ValueError('Official class speaker counts differ')
        attacks = Counter(row.attack_id for row in rows if row.label == 'spoof')
        if attacks != {f'A{index:02d}': spec['spoof_per_attack'] for index in range(1, 7)}:
            raise ValueError('Official attack counts differ')
        splits[split], hashes[split] = rows, spec['sha256']
    validate_splits(splits)
    return splits, hashes


def manifest_payload(rows):
    # Preserve official protocol order and use explicit LF for this full manifest.
    import csv
    from dataclasses import asdict
    from audio_deepfake.manifests import ManifestRow
    target = io.StringIO(newline='')
    writer = csv.DictWriter(target, fieldnames=tuple(ManifestRow.__dataclass_fields__), lineterminator='\n')
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    return target.getvalue().encode('utf-8')


def manifest_payloads(splits):
    return {'full_train.csv': manifest_payload(splits['train']),
            'full_dev.csv': manifest_payload(splits['dev']),
            'full_train_dev.csv': manifest_payload(splits['train'] + splits['dev'])}


def member_map(source, rows):
    expected = {'LA/' + row.audio_path for row in rows}
    selected = {}
    extra = Counter()
    for info in source.infolist():
        if info.filename in expected:
            if info.filename in selected:
                raise ValueError('Duplicate official archive member')
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or not 0 < info.file_size <= MEMBER_CAP:
                raise ValueError('Unsafe or oversized official archive member')
            selected[info.filename] = info
        else:
            for split in ('train', 'dev'):
                if info.filename.startswith(f'LA/ASVspoof2019_LA_{split}/flac/') and info.filename.endswith('.flac'):
                    extra[split] += 1
    if set(selected) != expected:
        raise ValueError('Missing official archive member')
    return selected, dict(extra)


def audio_path(data_root, row):
    if row.split not in ('train', 'dev') or row.audio_path != f'ASVspoof2019_LA_{row.split}/flac/{row.utterance_id}.flac':
        raise ValueError('Only canonical train/dev audio paths are allowed; eval is forbidden')
    path = data_root / row.audio_path
    reject_symlinks(path)
    if not path.resolve().is_relative_to(data_root.resolve()):
        raise ValueError('Audio path escapes data root')
    return path


def plan_preparation(archive, protocol_dir, data_root, output_dir):
    for path in (archive, protocol_dir, data_root, output_dir):
        reject_symlinks(path)
    splits, protocol_hashes = read_official_splits(protocol_dir)
    rows = splits['train'] + splits['dev']
    if archive.stat().st_size != SIZE or digest(archive, 'md5') != MD5:
        raise ValueError('Cached archive fails published size/MD5; no download is performed')
    with zipfile.ZipFile(archive) as source:
        members, extra = member_map(source, rows)
        remaining = sum(members['LA/' + row.audio_path].file_size for row in rows if not audio_path(data_root, row).exists())
        total = sum(info.file_size for info in members.values())
    parent = data_root
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    memory = memory_snapshot()
    if free < remaining + RESERVE_BYTES:
        raise ValueError('Insufficient disk for missing audio plus 20 GiB reserve')
    if memory['system_available_ram_bytes'] is None or memory['system_available_ram_bytes'] < MIN_RAM_BYTES:
        raise ValueError('At least 3 GiB available system RAM is required')
    payloads = manifest_payloads(splits)
    manifest_hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    if manifest_hashes['full_train_dev.csv'] != COMBINED_MANIFEST_SHA256:
        raise ValueError('Canonical full manifest SHA-256 mismatch')
    return splits, payloads, {'scope': 'complete official train/dev only; eval closed', 'expected_records': len(rows),
        'splits': {split: summarize(group) for split, group in splits.items()},
        'source_protocol_sha256': protocol_hashes, 'source_manifest_sha256': manifest_hashes,
        'archive': {'size_bytes': SIZE, 'md5': MD5, 'verified': True},
        'official_flac_bytes': total, 'additional_flac_bytes': remaining,
        'excluded_unkeyed_flac_counts': extra, 'disk_free_bytes': free,
        'disk_required_bytes': remaining + RESERVE_BYTES, 'memory_preflight': memory}


def prepare_full(archive, protocol_dir, data_root, output_dir, *, pilot_journal=None, execute=False):
    splits, payloads, plan = plan_preparation(archive, protocol_dir, data_root, output_dir)
    if not execute:
        return plan
    rows = splits['train'] + splits['dev']
    journal_path = output_dir / 'prepare_report.json'
    if pilot_journal is not None and journal_path.resolve() == pilot_journal.resolve():
        raise ValueError('Full journal must not overwrite the pilot journal')
    lock = output_dir / 'prepare.lock'
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (journal_path, journal_path.with_suffix('.json.tmp'), lock,
                 output_dir / 'manifest_summary.json', output_dir / 'manifest_summary.json.tmp'):
        reject_symlinks(path)
    with lock.open('x') as target:
        target.write(str(os.getpid()))
    report = None
    started = time.monotonic()
    try:
        report = json.loads(journal_path.read_text()) if journal_path.exists() else {'records': {}, 'errors': []}
        if report.get('source_protocol_sha256', plan['source_protocol_sha256']) != plan['source_protocol_sha256'] or report.get('source_manifest_sha256', plan['source_manifest_sha256']) != plan['source_manifest_sha256']:
            raise ValueError('Full preparation journal selection changed')
        allowed = {row.audio_path for row in rows}
        if set(report['records']) - allowed:
            raise ValueError('Full journal contains extra or eval paths')
        old = {}
        if pilot_journal is not None and pilot_journal.exists():
            reject_symlinks(pilot_journal)
            old = json.loads(pilot_journal.read_text()).get('records', {})
            if set(old) - allowed:
                raise ValueError('Pilot journal contains extra or eval paths')
        for name, payload in payloads.items():
            path = output_dir / name
            reject_symlinks(path)
            if path.exists() and path.read_bytes() != payload:
                raise ValueError('Existing full manifest differs from official selection')
        for name, payload in payloads.items():
            path = output_dir / name
            if not path.exists():
                temporary = path.with_suffix('.csv.tmp')
                reject_symlinks(temporary)
                temporary.write_bytes(payload)
                temporary.replace(path)
        report.update({key: value for key, value in plan.items() if key not in ('memory_preflight', 'disk_free_bytes', 'disk_required_bytes')})
        report.update({'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'source_record': 'https://zenodo.org/records/6906306', 'source_urls': SOURCES,
            'license': 'ODC-By-1.0', 'format': 'FLAC',
            'hash_provenance': 'Cached full archive verified against published size/MD5; each selected audio SHA-256 is locally observed and compared with archive member.',
            'verified_records': 0})
        save_full_report(report, journal_path)
        with zipfile.ZipFile(archive) as source:
            members, _ = member_map(source, rows)
            for index, row in enumerate(rows, 1):
                path = audio_path(data_root, row)
                info = members['LA/' + row.audio_path]
                previous = report['records'].get(row.audio_path)
                pilot_previous = old.get(row.audio_path)
                checksum = hashlib.sha256()
                temporary = None
                try:
                    if path.exists():
                        if path.stat().st_size != info.file_size:
                            raise ValueError(f'Changed existing audio size: {row.audio_path}')
                        actual_hash = digest(path)
                        if previous is not None and actual_hash != previous.get('sha256'):
                            raise ValueError(f'Changed journaled audio: {row.audio_path}')
                        if pilot_previous is not None and actual_hash != pilot_previous.get('sha256'):
                            raise ValueError(f'Changed pilot audio: {row.audio_path}')
                        with source.open(info) as member:
                            for chunk in iter(lambda: member.read(1024**2), b''):
                                checksum.update(chunk)
                        if actual_hash != checksum.hexdigest():
                            raise ValueError(f'Existing audio differs from verified archive: {row.audio_path}')
                        observed = inspect_audio(path)
                    else:
                        if previous is not None or pilot_previous is not None:
                            raise ValueError(f'Journaled audio is missing: {row.audio_path}')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.part', delete=False) as target:
                            temporary = Path(target.name)
                            with source.open(info) as member:
                                for chunk in iter(lambda: member.read(1024**2), b''):
                                    target.write(chunk)
                                    checksum.update(chunk)
                            target.flush()
                            os.fsync(target.fileno())
                        observed = inspect_audio(temporary)
                        if observed['size_bytes'] != info.file_size or observed['sha256'] != checksum.hexdigest():
                            raise ValueError('Extracted audio integrity differs')
                        temporary.replace(path)
                    for known in (previous, pilot_previous):
                        if known is not None and any(observed[key] != known.get(key) for key in observed):
                            raise ValueError(f'Changed journaled audio metadata: {row.audio_path}')
                    report['records'][row.audio_path] = observed
                    report['verified_records'] = index
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                if index % SAVE_EVERY == 0 or index == len(rows):
                    report['elapsed_seconds'] = time.monotonic() - started
                    save_full_report(report, journal_path)
                    print(f'Full audio: verified {index:,}/{len(rows):,}', flush=True)
                    if memory_snapshot()['system_available_ram_bytes'] < MIN_RUNNING_RAM_BYTES:
                        raise ValueError('Available system RAM fell below 1 GiB; preparation stopped')
        report['status'] = 'ok'
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['elapsed_seconds'] = time.monotonic() - started
        report['memory_after'] = memory_snapshot()
        report['saved_audio_bytes'] = sum(item['size_bytes'] for item in report['records'].values())
        report['duration_seconds'] = sum(item['duration_seconds'] for item in report['records'].values())
        save_full_report(report, journal_path)
        summary = {key: value for key, value in report.items() if key not in ('records', 'errors')}
        save_full_report(summary, output_dir / 'manifest_summary.json')
        return report
    except BaseException as error:
        if report is not None:
            report['status'] = 'error'
            report['errors'].append({'type': type(error).__name__, 'message': str(error)})
            report['elapsed_seconds'] = time.monotonic() - started
            save_full_report(report, journal_path)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'data/archives/LA.zip')
    parser.add_argument('--protocol-dir', type=Path, default=ROOT / 'data/metadata/asvspoof2019')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/asvspoof2019/LA')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/full-data')
    parser.add_argument('--pilot-journal', type=Path, default=ROOT / 'outputs/pilot-data/prepare_report.json')
    parser.add_argument('--execute', action='store_true', help='Extract from cached archive only; no network')
    args = parser.parse_args(argv)
    report = prepare_full(args.archive, args.protocol_dir, args.data_root, args.output_dir,
                          pilot_journal=args.pilot_journal, execute=args.execute)
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
