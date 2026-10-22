"""Combine the 12 visible-output transport parts and independently verify extraction.

Offline only; no encoder load, fitting, downloads or browser operations. Audit is
written only after complete archive/hash/CRC/source/cache/resource validation.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from ssl_extract_worker import sha, digest, save, CONFIG_SHA
from ssl_extract_job import validate_all, GIB, cache_bytes

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_BYTES = 93337242
ARCHIVE_SHA = '5c45d276a5f749826438741c737bad2fb3da2db2016598e5225ae4f358f13845'
IDENTITY_SHA = '93a5ee49123949554a27c4acced1f75f4742425bde9c0ca866103018241568fa'
CHUNK_BYTES = 8 * 1024**2
PART_COUNT = 12


def combine(session):
    parts = [session / f'part-{i:02}.bin' for i in range(PART_COUNT)]
    if set(session.glob('part-*.bin')) != set(parts):
        raise ValueError('Require exactly all 12 numbered transport parts')
    for index, path in enumerate(parts):
        expected = min(CHUNK_BYTES, ARCHIVE_BYTES - index * CHUNK_BYTES)
        if path.stat().st_size != expected:
            raise ValueError(f'Part {index} size differs: expected {expected}')
    archive = session / 'ssl-extraction-v1.zip'
    if archive.exists():
        if archive.stat().st_size != ARCHIVE_BYTES or sha(archive) != ARCHIVE_SHA:
            raise ValueError('Existing archive differs; refuse replacement')
        return archive
    temporary = archive.with_suffix('.zip.tmp')
    with temporary.open('xb') as output:
        for path in parts:
            with path.open('rb') as stream:
                shutil.copyfileobj(stream, output, length=1024**2)
    if temporary.stat().st_size != ARCHIVE_BYTES or sha(temporary) != ARCHIVE_SHA:
        raise ValueError('Whole archive size/SHA-256 differs; retained temporary for inspection')
    temporary.replace(archive)
    return archive


def extract_checked(archive, target):
    if target.exists():
        raise ValueError('Require fresh result extraction directory')
    with zipfile.ZipFile(archive) as zipped:
        names = set()
        total = 0
        for info in zipped.infolist():
            name = info.filename
            relative = PurePosixPath(name)
            if (name in names or not name or relative.is_absolute() or '..' in relative.parts
                    or '\\' in name or ':' in name or '\x00' in name):
                raise ValueError('Unsafe or duplicate archive member')
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Archive symlinks are forbidden')
            names.add(name)
            total += info.file_size
        if total > 2 * GIB:
            raise ValueError('Excessive uncompressed archive size')
        bad = zipped.testzip()
        if bad is not None:
            raise ValueError(f'ZIP CRC failed: {bad}')
        target.mkdir(parents=True)
        for info in zipped.infolist():
            target_path = (target / info.filename).resolve()
            if not target_path.is_relative_to(target):
                raise ValueError('Archive path escapes fresh result directory')
            if info.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, target_path.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1024**2)
    return total


def verify(result, frozen):
    manifest_path = result / 'manifest.json'
    if sha(manifest_path) != sha(frozen / 'manifest.json'):
        raise ValueError('Recovered manifest differs from frozen v2 input')
    manifest = json.loads(manifest_path.read_text())
    for relative, wanted in manifest['files_sha256'].items():
        if sha(frozen / relative) != wanted:
            raise ValueError(f'Frozen input provenance differs: {relative}')
    if manifest['files_sha256']['config.json'] != CONFIG_SHA:
        raise ValueError('Frozen experiment configuration differs')
    for name in ('ssl_extract_worker.py', 'ssl_extract_job.py', 'localization_ssl_geometry.py'):
        if sha(result / 'sources' / name) != manifest['files_sha256'][name]:
            raise ValueError(f'Recovered executed source differs: {name}')
    run = result / 'run'
    identity = json.loads((run / 'identity.json').read_text())
    value = identity['identity']
    if identity['identity_sha256'] != IDENTITY_SHA or digest(value) != IDENTITY_SHA:
        raise ValueError('Extraction identity hash differs')
    if value['manifest_sha256'] != sha(manifest_path) or value['source_sha256'] != manifest['files_sha256']:
        raise ValueError('Identity input/source pins differ')
    for key, name in (('worker_sha256', 'ssl_extract_worker.py'), ('supervisor_sha256', 'ssl_extract_job.py')):
        if value[key] != manifest['files_sha256'][name]:
            raise ValueError('Identity executed source pin differs')
    config = json.loads((frozen / 'config.json').read_text())
    expected_model = {name: config['checkpoint']['metadata_sha256'][name]
                      for name in ('config.json', 'preprocessor_config.json')}
    expected_model['pytorch_model.bin'] = config['checkpoint']['sha256']
    if value['model_sha256'] != expected_model or value['device'] != 'cuda':
        raise ValueError('Encoder/processor/device identity differs')
    status = json.loads((run / 'status.json').read_text())
    worker = json.loads((run / 'worker-result.json').read_text())
    for record in (status, worker):
        if (record['status'] != 'complete' or record['record_count'] != 192
                or record['identity_sha256'] != IDENTITY_SHA):
            raise ValueError('Missing real supervised 192-record completion')
        if any(record.get(k) is not False for k in ('training_started', 'eval_accessed', 'native_frame_labels_opened')):
            raise ValueError('Extraction scope changed')
        if record.get('label_summary_metadata_opened') is not True:
            raise ValueError('Expected frozen selection-summary metadata scope')
    if status['exit_code'] != 0 or (run / 'active.lock').exists():
        raise ValueError('Worker exit or lock state differs')
    elapsed = status['cumulative_elapsed_seconds']
    if not math.isfinite(elapsed) or not 0 < elapsed <= 1800:
        raise ValueError('Cumulative timeout validation failed')
    if not 0 < status['peak_rss_bytes'] <= 3 * GIB:
        raise ValueError('External worker peak exceeds limit')
    if worker['resources_before_model_load']['system_available_ram_bytes'] < 4 * GIB:
        raise ValueError('Immediate pre-load available RAM gate failed')
    checked = validate_all(run, manifest, IDENTITY_SHA)
    validation_path = run / 'validation.json'
    if sha(validation_path) != status['validation_sha256']:
        raise ValueError('Supervisor validation receipt checksum differs')
    validation = json.loads(validation_path.read_text())
    if validation != {'identity_sha256': IDENTITY_SHA, 'records': checked}:
        raise ValueError('Independent per-record validation differs from supervisor')
    for row in manifest['records']:
        receipt = json.loads((run / 'cache' / row['split'] / (row['utterance_id'] + '.json')).read_text())
        resources = receipt['resources']
        if (resources['system_available_ram_bytes'] < 2 * GIB or resources['disk_free_bytes'] < 20 * GIB
                or resources['peak_rss_bytes'] > 3 * GIB):
            raise ValueError('Per-record resource receipt exceeds boundaries')
    if cache_bytes(run) > GIB:
        raise ValueError('Feature cache exceeds storage cap')
    return {'status': 'independently_verified_extraction_complete', 'strict_extraction_gate_pass': True,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': 'PartialSpoof v1.2',
            'selection': {'train': 128, 'dev': 64, 'eval': 0}, 'feature_dimension': 768,
            'aligned_bins': sum(r['coarse_count_20ms'] for r in manifest['records']),
            'identity_sha256': IDENTITY_SHA, 'manifest_sha256': sha(manifest_path),
            'training_started': False, 'classifier_fitted': False, 'official_benchmark': False,
            'native_frame_labels_opened': False, 'label_summary_metadata_opened': True,
            'status_sha256': sha(run / 'status.json'), 'validation_sha256': sha(validation_path),
            'cache_bytes_including_receipts': cache_bytes(run), 'supervisor': status,
            'worker': worker, 'identity': identity, 'verified_records': checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--session-dir', type=Path, default=ROOT / 'outputs/colab/extraction-session-v1')
    parser.add_argument('--frozen-dir', type=Path, default=ROOT / 'outputs/colab/extraction-frozen-inputs-v2')
    parser.add_argument('--audit', type=Path, default=ROOT / 'docs/validation/localization_ssl_extraction_v1.json')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'parts_required': PART_COUNT, 'archive_sha256': ARCHIVE_SHA}))
        return
    session, frozen, audit = args.session_dir.resolve(), args.frozen_dir.resolve(), args.audit.resolve()
    if not session.is_relative_to(ROOT / 'outputs') or not frozen.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Inputs/results must remain under ignored outputs/')
    if not audit.is_relative_to(ROOT / 'docs/validation') or audit.exists():
        raise ValueError('Require new repository validation audit path')
    archive = combine(session)
    result = session / 'result'
    uncompressed = extract_checked(archive, result)
    evidence = verify(result, frozen)
    evidence['archive'] = {'path': str(archive.relative_to(ROOT)).replace('\\', '/'),
                           'bytes': ARCHIVE_BYTES, 'sha256': ARCHIVE_SHA, 'crc_checked': True,
                           'uncompressed_bytes': uncompressed, 'transport_parts': PART_COUNT}
    evidence['result_directory'] = str((result / 'run').relative_to(ROOT)).replace('\\', '/')
    save(audit, evidence)
    print(json.dumps({'status': evidence['status'], 'record_count': len(evidence['verified_records']),
                      'audit': str(audit)}, indent=2))


if __name__ == '__main__':
    main()
