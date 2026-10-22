"""Build the exact public 192-recording SSL input bundle locally; never upload.

No frame-label arrays, eval audio, weights, private audio or full checkout.
Frozen selection/alignment audits include utterance classes and label summaries.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def build(target):
    config_path = ROOT / 'configs/localization_ssl_v1.json'
    config = json.loads(config_path.read_text())
    for name, digest in config['input_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    smoke = json.loads((ROOT / 'docs/validation/localization_ssl_colab_smoke_v2.json').read_text())
    if not smoke['strict_resource_gate_pass'] or smoke['supervisor']['status'] != 'complete':
        raise ValueError('Successful supervised smoke required')
    alignment = json.loads((ROOT / 'docs/validation/localization_alignment.json').read_text())
    selection = json.loads((ROOT / 'docs/validation/partial_selection_v1.json').read_text())
    if not alignment['training_alignment_ready']:
        raise ValueError('Actual audio alignment gate failed')
    mapping = {
        'config.json': config_path,
        'partial_selection_v1.json': ROOT / 'docs/validation/partial_selection_v1.json',
        'localization_alignment.json': ROOT / 'docs/validation/localization_alignment.json',
        'localization_ssl_geometry.py': ROOT / 'src/audio_deepfake/localization_ssl_geometry.py',
        'ssl_extract_worker.py': ROOT / 'scripts/ssl_extract_worker.py',
        'ssl_extract_job.py': ROOT / 'scripts/ssl_extract_job.py',
        'PartialSpoof_README_v1.2': ROOT / 'data/partialspoof/README_v1.2',
        'PARTIALSPOOF_PROTOCOL.md': ROOT / 'docs/PARTIALSPOOF_PROTOCOL.md',
        'localization_ssl_colab_smoke_v2.json': ROOT / 'docs/validation/localization_ssl_colab_smoke_v2.json',
    }
    records = []
    seen = set()
    for split, count in [('train', 128), ('dev', 64)]:
        rows = alignment['splits'][split]['records']
        selected = {r['utterance_id'] for r in selection['splits'][split]['selected_records']}
        if len(rows) != count or {r['utterance_id'] for r in rows} != selected:
            raise ValueError('Require exact fixed train/dev selection')
        for row in sorted(rows, key=lambda r: r['utterance_id']):
            uid = row['utterance_id']
            if uid in seen or not uid.replace('_', '').isalnum():
                raise ValueError('Invalid or duplicate recording ID')
            seen.add(uid)
            path = (ROOT / row['audio_path']).resolve()
            expected = (ROOT / 'data/partialspoof/database' / split / 'con_wav' / f'{uid}.wav').resolve()
            if path != expected or sha(path) != row['audio_sha256']:
                raise ValueError('Audited public audio path/hash changed')
            relative = f'audio/{split}/{uid}.wav'
            mapping[relative] = path
            records.append({k: row[k] for k in ['utterance_id', 'sample_count', 'audio_sha256', 'coarse_count_20ms']} |
                           {'split': split, 'audio_file': relative})
    if sum(p.stat().st_size for p in mapping.values()) > 100_000_000:
        raise ValueError('Extraction input bundle exceeds 100 MB')
    manifest = {
        'version': 1, 'dataset': 'PartialSpoof v1.2', 'seed': 45,
        'config_file': 'config.json', 'selection_file': 'partial_selection_v1.json',
        'alignment_file': 'localization_alignment.json', 'model_dir': 'model',
        'smoke_audit_file': 'localization_ssl_colab_smoke_v2.json',
        'selection_sha256': sha(mapping['partial_selection_v1.json']),
        'alignment_sha256': sha(mapping['localization_alignment.json']),
        'frozen_plan_sha256': sha(config_path),
        'files_sha256': {name: sha(p) for name, p in mapping.items()}, 'records': records,
        'native_frame_labels_included': False, 'utterance_metadata_included': True,
        'eval_audio_included': False, 'private_audio_included': False,
        'public_audio_source': 'PartialSpoof v1.2; official train/dev splits; Zenodo 5766198',
        'attribution': 'CC BY 4.0; underlying ASVspoof 2019 ODC-By; docs/PARTIALSPOOF_PROTOCOL.md',
        'smoke_audit_sha256': sha(ROOT / 'docs/validation/localization_ssl_colab_smoke_v2.json'),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=2)+'\n')
        for name, path in mapping.items():
            z.write(path, name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None or set(z.namelist()) != set(mapping) | {'manifest.json'}:
            raise ValueError('Bundle inventory/CRC mismatch')
        for name, digest in manifest['files_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('Archived payload hash mismatch')
    return {'status': 'prepared_locally_not_uploaded', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'bundle_path': target.relative_to(ROOT).as_posix(), 'bytes': target.stat().st_size,
            'sha256': sha(target), 'manifest': manifest, 'training_started': False,
            'full_192_extraction_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/colab/ssl_extraction_inputs_v1.zip')
    parser.add_argument('--audit', type=Path, default=ROOT/'docs/validation/localization_ssl_extraction_bundle_v1.json')
    args = parser.parse_args()
    result = build(args.output.resolve())
    audit = args.audit.resolve()
    with audit.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['status', 'bundle_path', 'bytes', 'sha256']}, indent=2))


if __name__ == '__main__':
    main()
