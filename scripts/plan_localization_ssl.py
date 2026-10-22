"""Offline SSL geometry/resource preflight. Never downloads, imports torch or fits."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from audio_deepfake.localization_ssl_geometry import convolution_geometry, frame_mapping
from audio_deepfake.resources import memory_snapshot


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--write-audit', action='store_true')
    args = parser.parse_args()
    config_path = ROOT / 'configs/localization_ssl_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    for relative, digest in config['input_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Frozen input changed: {relative}')
    for name, digest in config['checkpoint']['metadata_sha256'].items():
        if sha(ROOT / '.cache/representation-sources' / name) != digest:
            raise ValueError(f'Pinned checkpoint metadata changed: {name}')
    upstream = json.loads((ROOT / '.cache/representation-sources/config.json').read_text())
    receptive, hop = convolution_geometry(upstream['conv_kernel'], upstream['conv_stride'])
    if (receptive, hop, upstream['hidden_size'], upstream['num_hidden_layers']) != (400, 320, 768, 12):
        raise ValueError('Unexpected encoder geometry')
    audit = json.loads((ROOT / 'docs/validation/localization_alignment.json').read_text())
    selection = json.loads((ROOT / 'docs/validation/partial_selection_v1.json').read_text())
    if not audit['training_alignment_ready']:
        raise ValueError('Actual audio/native label audit is required')
    summaries = {}
    for split, expected in [('train', 128), ('dev', 64)]:
        rows = audit['splits'][split]['records']
        ids = [r['utterance_id'] for r in rows]
        selected = [r['utterance_id'] for r in selection['splits'][split]['selected_records']]
        if len(ids) != expected or len(set(ids)) != expected or set(ids) != set(selected):
            raise ValueError('Unexpected selected IDs')
        source_frames = bins = 0
        for row in rows:
            if not row['candidate_match'] or row['sample_rate'] != 16000 or row['channels'] != 1:
                raise ValueError('Unsupported audited audio')
            mapping = frame_mapping(row['sample_count'])
            if len(mapping['frame_indices']) != row['coarse_count_20ms']:
                raise ValueError('Label grid changed')
            source_frames += mapping['source_frame_count']
            bins += len(mapping['frame_indices'])
        summaries[split] = {'recordings': len(rows), 'unaligned_encoder_frames': source_frames,
                            'aligned_bins': bins, 'raw_float32_cache_bytes': bins * 768 * 4,
                            'audio_seconds': sum(r['sample_count'] / 16000 for r in rows)}
    smoke = max(audit['splits']['train']['records'], key=lambda r: (r['sample_count'], r['utterance_id']))
    memory = memory_snapshot()
    free = shutil.disk_usage(ROOT).free
    source_paths = ['scripts/plan_localization_ssl.py', 'src/audio_deepfake/localization_ssl_geometry.py',
                    'docs/LOCALIZATION_SSL_PLAN.md', 'configs/localization_ssl_v1.json']
    result = {
        'status': 'offline_plan_geometry_passed_execution_not_implemented',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_downloaded': False, 'training_started': False, 'eval_accessed': False,
        'official_benchmark': False, 'checkpoint': config['checkpoint'],
        'source_sha256': {p: sha(ROOT / p) for p in source_paths},
        'input_sha256': config['input_sha256'], 'splits': summaries,
        'smoke_train_id': smoke['utterance_id'], 'smoke_sample_count': smoke['sample_count'],
        'resources': {'memory': memory, 'disk_free_bytes': free,
                      'gpu': 'CIM unavailable; previous Intel Iris Xe inventory, no CUDA assumed'},
        'resource_start_gate_passed_at_probe': (memory['system_available_ram_bytes'] is not None
            and memory['system_available_ram_bytes'] >= 4 * 1024**3 and free >= 20 * 1024**3),
        'execution_gates_pending': ['Isolated version-locked encoder environment',
            'Restricted weights-only loader and checkpoint hash verification',
            'Externally supervised single-recording shape/resource smoke',
            'Separate 768-dimensional head with matching sampled frame ID digest'],
        'limitations': ['Geometry is calculated from config, not validated on actual encoder output yet.',
            'Pretraining recording overlap is unverified.',
            'Small repeatedly inspected dev subset; no independent generalization claim.']}
    if args.write_audit:
        path = ROOT / 'docs/validation/localization_ssl_plan_v1.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
