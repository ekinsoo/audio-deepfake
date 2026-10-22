"""Bounded, checksum-pinned PartialSpoof v1.2 metadata preparation.

No audio is downloaded. Archive members belonging to eval are never read.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audio_deepfake.resources import memory_snapshot

FILES = {
    'README_v1.2': (17948, '4fec0742e2a606d1b1186067d79862d6'),
    'database_protocols.tar.gz': (5389040, '699d81f020e4b7fa8f33747010e1cba8'),
    'database_segment_labels_v1.2.tar.gz': (76583019, 'c2bf6638e59ec7a5cf93c4a510fe4efe'),
    'database_vad.tar.gz': (10240345, '95e77e19a1bb4f2e79ed138fd35621ad'),
}
BUDGET = 110_000_000
JOURNAL = ROOT / 'docs/validation/partial_metadata.json'
DEST = ROOT / 'data/partialspoof'

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(value):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def resources():
    memory = memory_snapshot()
    free = shutil.disk_usage(ROOT).free
    if free < 20 * 1024**3 or memory['system_available_ram_bytes'] < 1024**3:
        raise RuntimeError('Metadata resource guard failed')
    return {'memory': memory, 'disk_free_bytes': free}

def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    return p

def allowed_member(name):
    p = safe_member(name)
    if any(x == 'eval' or x.startswith('eval_') or '.eval.' in x for x in p.parts):
        return False
    return (p.name in ('train_seglab_0.01.npy', 'dev_seglab_0.01.npy')
        or p.name in ('PartialSpoof.LA.cm.train.trl.txt', 'PartialSpoof.LA.cm.dev.trl.txt')
        or ('vad' in p.parts and any(x in ('train', 'dev') for x in p.parts) and p.suffix == '.vad'))

def select_protocols(journal):
    archive_path = DEST / 'database_protocols.tar.gz'
    if digest(archive_path, 'md5') != FILES[archive_path.name][1]:
        raise RuntimeError('Unverified protocol archive')
    result = {'version': 1, 'dataset': 'PartialSpoof v1.2', 'seed': 45,
        'selection_rule': 'Python Random(45) independently per split; sorted class pools; equal utterance classes; no score-dependent selection',
        'protocol_sha256': digest(archive_path), 'scoring_started': False,
        'duration_source': 'VAD endpoints pending; real audio duration is a separate alignment gate', 'splits': {}}
    with tarfile.open(archive_path, 'r:gz') as archive:
        for split, count in [('train', 128), ('dev', 64)]:
            name = 'database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.' + split + '.trl.txt'
            raw = archive.extractfile(name).read()
            records = []
            for line in raw.decode('utf-8').splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[4] not in ('bonafide', 'spoof'):
                    raise RuntimeError('Unexpected protocol row')
                records.append({'speaker_id': fields[0], 'utterance_id': fields[1],
                    'system_id': fields[3], 'utterance_class': fields[4]})
            if len({r['utterance_id'] for r in records}) != len(records):
                raise RuntimeError('Duplicate protocol ID')
            rng = random.Random(45)
            chosen = []
            class_counts = {}
            for label in ('bonafide', 'spoof'):
                pool = sorted([r for r in records if r['utterance_class'] == label], key=lambda r: r['utterance_id'])
                class_counts[label] = len(pool)
                chosen += rng.sample(pool, count // 2)
            chosen.sort(key=lambda r: r['utterance_id'])
            smoke = sorted([r['utterance_id'] for label in ('bonafide', 'spoof')
                for r in [x for x in chosen if x['utterance_class'] == label][:4]])
            result['splits'][split] = {'official_count': len(records), 'official_class_counts': class_counts,
                'protocol_text_sha256': hashlib.sha256(raw).hexdigest(),
                'selected_count': count, 'selected_records': chosen, 'smoke_ids': smoke,
                'timeline_ids': sorted([r['utterance_id'] for label in ('bonafide', 'spoof')
                    for r in [x for x in chosen if x['utterance_class'] == label][:5]]) if split == 'dev' else []}
    selection = ROOT / 'docs/validation/partial_selection_v1.json'
    if selection.exists():
        old = json.loads(selection.read_text())
        for split in ('train', 'dev'):
            if [x['utterance_id'] for x in old['splits'][split]['selected_records']] != [x['utterance_id'] for x in result['splits'][split]['selected_records']]:
                raise RuntimeError('Refuse to alter frozen selected IDs')
    selection.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    journal['selection_sha256'] = digest(selection)
    save(journal)

def fetch(journal):
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (size, md5) in FILES.items():
        path = DEST / name
        if path.exists():
            if path.stat().st_size != size or digest(path, 'md5') != md5:
                raise RuntimeError('Existing immutable artifact checksum mismatch: ' + name)
        else:
            temp = path.with_name(name + '.partial')
            url = 'https://zenodo.org/api/records/5766198/files/' + name + '/content'
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(url, timeout=45) as response, temp.open('wb') as f:
                        declared = response.headers.get('Content-Length')
                        if declared and int(declared) != size:
                            raise RuntimeError('Unexpected Content-Length')
                        written = 0
                        while True:
                            remaining = BUDGET - journal['transferred_bytes']
                            if remaining <= 0:
                                raise RuntimeError('Transfer budget exhausted')
                            block = response.read(min(1024 * 1024, remaining))
                            if not block:
                                break
                            written += len(block)
                            journal['transferred_bytes'] += len(block)
                            f.write(block)
                            save(journal)
                            if written > size:
                                raise RuntimeError('Artifact exceeded pinned size')
                            resources()
                    if written != size or digest(temp, 'md5') != md5:
                        raise RuntimeError('Publisher checksum/size mismatch')
                    temp.replace(path)
                    break
                except (OSError, TimeoutError) as exc:
                    journal.setdefault('interruptions', []).append({'file': name, 'attempt': attempt + 1, 'error': str(exc)})
                    save(journal)
                    if attempt == 1 or BUDGET - journal['transferred_bytes'] < size:
                        raise
                    time.sleep(1)
        journal['artifacts'][name] = {'size_bytes': size, 'published_md5': md5,
            'verified_md5': digest(path, 'md5'), 'sha256': digest(path),
            'url': 'https://zenodo.org/records/5766198/files/' + name + '?download=1'}
        save(journal)

def inspect(journal):
    inventory = {}
    for name in FILES:
        if not name.endswith('.tar.gz'):
            continue
        with tarfile.open(DEST / name, 'r:gz') as archive:
            members = archive.getmembers()
            for m in members:
                safe_member(m.name)
                if not (m.isdir() or m.isfile()):
                    raise RuntimeError('Archive contains non-regular member')
            allowed = [m.name for m in members if m.isfile() and allowed_member(m.name)]
            inventory[name] = {'member_count': len(members),
                'first_names': [m.name for m in members[:25]],
                'allowed_member_count': len(allowed), 'allowed_names_sample': allowed[:25],
                'eval_payload_parsed': False}
    journal['inventory'] = inventory
    save(journal)

def annotate_selection(journal):
    import numpy as np
    selection_path = ROOT / 'docs/validation/partial_selection_v1.json'
    selection = json.loads(selection_path.read_text())
    selected = {r['utterance_id']: r for split in selection['splits'].values()
                for r in split['selected_records']}
    # Only the published-MD5-verified NII archives are trusted for pickle.
    # Never load eval numpy members or unverified user files.
    for name in ('database_vad.tar.gz', 'database_segment_labels_v1.2.tar.gz'):
        if digest(DEST / name, 'md5') != FILES[name][1]:
            raise RuntimeError('Refuse to parse an unverified metadata archive')
    with tarfile.open(DEST / 'database_vad.tar.gz', 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or not allowed_member(member.name):
                continue
            uttid = PurePosixPath(member.name).stem
            if uttid not in selected:
                continue
            raw = archive.extractfile(member).read()
            target = DEST / 'verified_train_dev' / PurePosixPath(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            rows = []
            for line in raw.decode('utf-8').splitlines():
                start, end, label = line.split()
                start, end, label = float(start), float(end), int(label)
                if not np.isfinite([start, end]).all() or start < 0 or end <= start or label not in (0, 1, 2):
                    raise RuntimeError('Invalid VAD range')
                if rows and abs(start - rows[-1][1]) > 1e-8:
                    raise RuntimeError('VAD has a gap or overlap')
                rows.append((start, end, label))
            if not rows or rows[0][0] != 0:
                raise RuntimeError('VAD does not start at zero')
            rec = selected[uttid]
            rec.update({'vad_member': member.name, 'vad_sha256': hashlib.sha256(raw).hexdigest(),
                'vad_duration_seconds': rows[-1][1],
                'vad_spoof_seconds': sum(b-a for a,b,c in rows if c == 0),
                'vad_nonspeech_seconds': sum(b-a for a,b,c in rows if c == 2),
                'vad_ranges': rows,
                'audio_duration_verified': False})
    with tarfile.open(DEST / 'database_segment_labels_v1.2.tar.gz', 'r:gz') as archive:
        for split in ('train', 'dev'):
            basename = split + '_seglab_0.01.npy'
            members = [m for m in archive.getmembers() if m.isfile() and PurePosixPath(m.name).name == basename and allowed_member(m.name)]
            if len(members) != 1:
                raise RuntimeError('Missing/duplicate 10ms label member')
            member = members[0]
            extracted = DEST / 'verified_train_dev' / basename
            extracted.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, extracted.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
            resources()
            labels = np.load(extracted, allow_pickle=True).item()
            if not isinstance(labels, dict) or len(labels) != selection['splits'][split]['official_count']:
                raise RuntimeError('Unexpected native label dictionary')
            summary = selection['splits'][split]
            summary['label_member'] = member.name
            summary['label_member_sha256'] = digest(extracted)
            summary['label_dictionary_count'] = len(labels)
            tail_differences = []
            for rec in summary['selected_records']:
                arr = np.asarray(labels[rec['utterance_id']])
                if arr.ndim != 1 or not np.isin(arr, ['0', '1', 0, 1]).all():
                    raise RuntimeError('Unexpected native labels')
                if 'vad_duration_seconds' not in rec:
                    raise RuntimeError('Missing selected VAD record')
                binary = arr.astype(np.int8)
                n = len(binary)
                rec.update({'native_label_dtype': str(arr.dtype), 'native_label_count_10ms': n,
                    'native_spoof_bins_10ms': int((binary == 0).sum()),
                    'vad_minus_label_grid_seconds': rec['vad_duration_seconds'] - n * .01})
                # Inspect how published binary labels cover VAD nonspeech,
                # keeping mixed boundary bins explicit rather than relabelling.
                pure_nonspeech = [i for i in range(n) if any(a <= i*.01 + 1e-9 and b >= (i+1)*.01 - 1e-9
                    for a,b,c in rec['vad_ranges'] if c == 2)]
                rec['pure_nonspeech_bins_native_spoof'] = int((binary[pure_nonspeech] == 0).sum())
                rec['pure_nonspeech_bins_native_bonafide'] = int((binary[pure_nonspeech] == 1).sum())
                del rec['vad_ranges']
                tail_differences.append(rec['vad_minus_label_grid_seconds'])
                if (rec['utterance_class'] == 'bonafide') != bool(np.all(binary == 1)):
                    raise RuntimeError('Utterance and segment classes disagree')
            summary['selected_tail_difference_min_seconds'] = min(tail_differences)
            summary['selected_tail_difference_max_seconds'] = max(tail_differences)
            compact_path = DEST / 'verified_train_dev' / (split + '_selected_labels_10ms.npz')
            np.savez_compressed(compact_path, **{r['utterance_id']: np.asarray(labels[r['utterance_id']], dtype=np.int8)
                for r in summary['selected_records']})
            summary['selected_labels_file'] = compact_path.relative_to(ROOT).as_posix()
            summary['selected_labels_sha256'] = digest(compact_path)
            summary['selected_labels_conversion'] = 'Native 0/1 Unicode vectors converted exactly to int8; no temporal resampling or relabelling; load compact NPZ with allow_pickle=False'
            del labels
            import gc
            gc.collect()
            journal.setdefault('parse_memory_snapshots', []).append(memory_snapshot())
    selection['duration_source'] = 'Publisher VAD final endpoints; real audio durations remain unverified'
    selection['label_pickle_trust_scope'] = 'Only train/dev 10ms members from publisher-MD5-verified NII PartialSpoof v1.2 archive'
    selection['metadata_archive_sha256'] = {name: digest(DEST/name) for name in FILES}
    selection_path.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    journal['selection_sha256'] = digest(selection_path)
    journal['native_metadata_parsed'] = {'train': True, 'dev': True, 'eval': False}
    save(journal)

def export_selected_json(journal):
    import math
    import numpy as np
    selection_path = ROOT / 'docs/validation/partial_selection_v1.json'
    selection = json.loads(selection_path.read_text())
    compact = {'dataset': 'PartialSpoof v1.2', 'resolution_seconds': .01,
               'native_label_mapping': {'0': 'spoof', '1': 'bonafide'}, 'splits': {}}
    findings = {}
    for split, summary in selection['splits'].items():
        path = ROOT / summary['selected_labels_file']
        if digest(path) != summary['selected_labels_sha256']:
            raise RuntimeError('Selected vector export checksum mismatch')
        with np.load(path, allow_pickle=False) as arrays:
            compact['splits'][split] = {r['utterance_id']: arrays[r['utterance_id']].tolist()
                for r in summary['selected_records']}
        different = [r['utterance_id'] for r in summary['selected_records']
            if r['native_label_count_10ms'] != math.ceil(r['vad_duration_seconds']/.01)]
        findings[split] = {'selected_count': summary['selected_count'],
            'native_count_differs_from_ceil_vad_duration_ids': different,
            'native_nonspeech_bin_spoof_count': sum(r['pure_nonspeech_bins_native_spoof'] for r in summary['selected_records']),
            'native_nonspeech_bin_bonafide_count': sum(r['pure_nonspeech_bins_native_bonafide'] for r in summary['selected_records']),
            'vad_minus_native_grid_min_seconds': summary['selected_tail_difference_min_seconds'],
            'vad_minus_native_grid_max_seconds': summary['selected_tail_difference_max_seconds']}
    output = DEST / 'selected_labels_v1.json'
    output.write_text(json.dumps(compact, separators=(',', ':')) + '\n', encoding='utf-8')
    journal['selected_labels_export'] = {'path': output.relative_to(ROOT).as_posix(),
        'sha256': digest(output), 'size_bytes': output.stat().st_size, 'pickle_required': False}
    journal['alignment_metadata_findings'] = findings
    journal['binary_reference_rule'] = 'Use published native binary vectors; VAD label2 bins can be native0 or1. VAD0-only references do not reproduce native binary truth.'
    save(journal)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Parse already-verified local archives; no network')
    args = parser.parse_args()
    if not (args.execute or args.prepare_only):
        print(json.dumps({'files': FILES, 'transfer_budget_bytes': BUDGET, 'audio_download': False}, indent=2))
        return
    previous = json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {}
    journal = {'dataset': 'PartialSpoof v1.2', 'record': '5766198',
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'transfer_budget_bytes': BUDGET, 'transferred_bytes': previous.get('transferred_bytes', 0),
        'preflight': resources(), 'artifacts': {}, 'status': 'fetching',
        'audio_downloaded': False, 'eval_payload_parsed': False,
        'eval_metadata_extracted': False,
        'archive_scan_note': 'Published metadata containers include eval members. Gzip scans may decompress bytes while skipping them; eval member payloads are never extracted, deserialized or interpreted.'}
    save(journal)
    try:
        if not args.prepare_only:
            fetch(journal)
        else:
            journal['artifacts'] = previous.get('artifacts', {})
            for name, (size, md5) in FILES.items():
                if (DEST/name).stat().st_size != size or digest(DEST/name, 'md5') != md5:
                    raise RuntimeError('Invalid local archive')
        inspect(journal)
        select_protocols(journal)
        annotate_selection(journal)
        export_selected_json(journal)
        journal['status'] = 'metadata_verified_selection_annotated_audio_alignment_pending'
    except Exception as exc:
        journal['status'] = 'stopped'
        journal['error'] = str(exc)
        raise
    finally:
        journal['source_sha256'] = digest(Path(__file__))
        journal['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(journal)

if __name__ == '__main__':
    main()
