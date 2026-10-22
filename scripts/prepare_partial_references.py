"""Build explicitly derived 10 ms references, never claim publisher RTTM equivalence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/partialspoof/selected_references_v1'
sys.path.insert(0, str(ROOT/'src'))
from audio_deepfake.resources import memory_snapshot


def resource_gate():
    snapshot = memory_snapshot()
    if snapshot['system_available_ram_bytes'] < 2*1024**3 or snapshot['peak_rss_bytes'] > 1024**3:
        raise RuntimeError('Reference preparation RAM guard failed')
    if shutil.disk_usage(ROOT).free < 20*1024**3:
        raise RuntimeError('Reference preparation disk guard failed')
    return snapshot

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def spoof_ranges(labels, duration):
    if not labels or any(type(x) is not int or x not in (0, 1) for x in labels):
        raise ValueError('Require nonempty native numeric binary labels')
    if not math.isfinite(duration) or duration <= 0 or len(labels) != math.ceil(duration / .01 - 1e-10):
        raise ValueError('Exact actual-audio 10 ms count required')
    result = []
    begin = None
    for index, label in enumerate(labels + [1]):
        if label == 0 and begin is None:
            begin = index
        elif label == 1 and begin is not None:
            result.append((begin * .01, min(index * .01, duration)))
            begin = None
    return result

def scan_archive(path, seconds=90):
    start = time.monotonic()
    count = 0
    hits = []
    # Streaming mode retains no TarInfo list and never extracts any payload.
    with tarfile.open(path, 'r|gz') as stream:
        for member in stream:
            count += 1
            if count > 150000 or time.monotonic() - start > seconds:
                raise RuntimeError('Bounded archive header scan exceeded limits')
            if count % 1000 == 0:
                resource_gate()
            if 'rttm' in member.name.lower() or 'reco2dur' in member.name.lower():
                hits.append({'name': member.name, 'bytes': member.size})
    return {'members': count, 'reference_members': hits, 'seconds': time.monotonic()-start,
            'payload_extracted': False, 'eval_payload_interpreted': False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Plan: verify and stream train/dev audio archive headers; derive selected 10 ms references. No download.')
        return
    if (ROOT/'docs/validation/localization_references_v1.json').exists():
        raise ValueError('Preserve the existing reference audit; use a separately versioned recovery')
    preflight = resource_gate()
    labels_path = ROOT/'data/partialspoof/selected_labels_v1.json'
    labels = json.loads(labels_path.read_text())['splits']
    selection_path = ROOT/'docs/validation/partial_selection_v1.json'
    selection = json.loads(selection_path.read_text())
    alignment_path = ROOT/'docs/validation/localization_alignment.json'
    alignment = json.loads(alignment_path.read_text())
    if (not alignment['training_alignment_ready'] or alignment['selection_sha256'] != digest(selection_path)
            or alignment['compact_labels_sha256'] != digest(labels_path)):
        raise ValueError('Matching actual audio alignment audit required')
    audit = {'status': 'publisher_reference_unavailable_derived_native_bin_diagnostic_ready',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'reference_origin': 'Derived from publisher native numeric 10 ms labels, not publisher RTTM',
             'official_native_reference_gate_passed': False,
             'preflight': preflight, 'alignment_sha256': digest(alignment_path),
             'selection_sha256': digest(selection_path), 'labels_sha256': digest(labels_path),
             'script_sha256': digest(__file__), 'splits': {},
             'upstream_metric_commit': '847347aaec6f65c3c6d2f17c63515b826b94feb3',
             'upstream_inspection': {'metric/README.md': '0d2e524c6196d116b07fc32ac4ef814b60d2b8034eac81ee2dc50bc982d22c6c',
                 'metric/rttm_tool.py': 'c63a260013edba82243f5b47356e22675d60362771333599dedebd51d3b6dd71',
                 'metric/cal_EER.sh': '8379ad20ce81541f6e244dec406ad8dac2f48397d0b9f2c590ea485c9d0f5565'},
             'limitations': ['Downloaded v1.2 audio and protocol archives contain no publisher RTTM/reco2dur.',
                 'Pinned metric README expects rttm_2cls_0sil_spoof; a publisher reference is not available in the checked distribution.',
                 'rttm_tool.py converts RTTM to labels, not the inverse; silence treatment and sub-10 ms boundaries remain unverified.',
                 'Derived ranges use native 0 labels, not VAD 0. Final bins clip to actual audio duration.',
                 'Custom pilot development diagnostic only; not official benchmark or paper reproduction.']}
    for name, expected in audit['upstream_inspection'].items():
        evidence = ROOT/'.cache/partial-reference-upstream'/Path(name).name
        if digest(evidence) != expected:
            raise ValueError('Pinned upstream source evidence mismatch')
    audit['metadata_archive_inventory'] = alignment['native_rttm_inventory']
    for split in ('train', 'dev'):
        resource_gate()
        receipt_path = ROOT/f'data/partialspoof/audio_{split}_complete.json'
        receipt = json.loads(receipt_path.read_text())
        archive = ROOT/f'data/partialspoof/database_{split}.tar.gz'
        if receipt['selection_sha256'] != audit['selection_sha256'] or receipt['status'] != 'complete':
            raise ValueError('Matching complete audio receipt required')
        if digest(archive) != receipt['archive_sha256'] or digest(archive, 'md5') != receipt['archive_md5']:
            raise ValueError('Archive checksum mismatch')
        scan = scan_archive(archive)
        if scan['reference_members']:
            raise ValueError('Publisher reference found: inspect before using derived fallback')
        records = {r['utterance_id']: r for r in receipt['records']}
        ids = sorted(r['utterance_id'] for r in selection['splits'][split]['selected_records'])
        if set(ids) != set(records) or set(ids) != set(labels[split]):
            raise ValueError('Exact selected records required')
        rttm = []
        durations = []
        summaries = []
        for uid in ids:
            r = records[uid]
            duration = r['frames']/r['sample_rate']
            ranges = spoof_ranges(labels[split][uid], duration)
            for start, end in ranges:
                rttm.append(f'SPEAKER {uid} 1 {start:.8f} {end-start:.8f} <NA> <NA> spoof <NA> <NA>\n')
            durations.append(f'{uid} {duration:.8f}\n')
            summaries.append({'utterance_id': uid, 'duration_seconds': duration,
                              'spoof_seconds': sum(b-a for a,b in ranges), 'ranges': len(ranges)})
        out = DEST/split
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in [('reference.rttm', rttm), ('reco2dur', durations)]:
            (out/name).write_text(''.join(rows), encoding='utf-8')
        audit['splits'][split] = {'archive_scan': scan, 'archive_sha256': receipt['archive_sha256'],
            'archive_md5': receipt['archive_md5'], 'audio_receipt_sha256': digest(receipt_path),
            'reference_sha256': digest(out/'reference.rttm'), 'reco2dur_sha256': digest(out/'reco2dur'),
            'records': summaries}
    path = ROOT/'docs/validation/localization_references_v1.json'
    path.write_text(json.dumps(audit, indent=2)+'\n')
    print(path)

if __name__ == '__main__':
    main()
