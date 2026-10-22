"""Prepare an ignored Colab smoke bundle locally; never upload it."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    audit = json.loads((ROOT / 'docs/validation/localization_alignment.json').read_text())
    row = next(r for r in audit['splits']['train']['records'] if r['utterance_id'] == 'LA_T_1963829')
    config = json.loads((ROOT / 'configs/localization_ssl_v1.json').read_text())
    cpu = json.loads((ROOT / 'outputs/jobs/ssl-smoke-v1/smoke/result.json').read_text())
    mapping = {
        'LA_T_1963829.wav': ROOT / row['audio_path'],
        'cpu_reference.npy': ROOT / 'outputs/jobs/ssl-smoke-v1/smoke/LA_T_1963829.npy',
        'localization_ssl_geometry.py': ROOT / 'src/audio_deepfake/localization_ssl_geometry.py',
        'ssl_colab_smoke.py': ROOT / 'scripts/ssl_colab_smoke.py'}
    for name in ('config.json', 'preprocessor_config.json', 'README.md', 'pytorch_model.bin'):
        mapping['model/' + name] = ROOT / 'data/models/wav2vec2-base-v1' / name
    hashes = {name: sha(path) for name, path in mapping.items()}
    if hashes['LA_T_1963829.wav'] != row['audio_sha256'] or hashes['cpu_reference.npy'] != cpu['features_sha256']:
        raise ValueError('Smoke audio or preliminary CPU features changed')
    for name in ('config.json', 'preprocessor_config.json', 'README.md'):
        if hashes['model/' + name] != config['checkpoint']['metadata_sha256'][name]:
            raise ValueError('Checkpoint metadata changed')
    if hashes['model/pytorch_model.bin'] != config['checkpoint']['sha256']:
        raise ValueError('Checkpoint changed')
    manifest = {'smoke_id': row['utterance_id'], 'sample_count': row['sample_count'],
        'audio_file': 'LA_T_1963829.wav', 'audio_sha256': row['audio_sha256'],
        'checkpoint_dir': 'model', 'files_sha256': hashes,
        'cpu_reference_file': 'cpu_reference.npy', 'cpu_reference_sha256': cpu['features_sha256'],
        'cpu_reference_caveat': 'Preliminary CPU smoke v1 completed but lacked the immediate pre-load 4 GiB guard. Corrected v2 stopped before model load. This reference is a numerical comparison artifact, not a resource-gate pass.',
        'checkpoint_license': 'Apache-2.0 declared by official model card',
        'public_audio_source': 'PartialSpoof v1.2, Zenodo 5766198, official train split',
        'audio_terms': 'PartialSpoof CC BY 4.0 and underlying ASVspoof ODC-By attribution; see project protocol.',
        'expected_destination': 'Google Colab; upload awaits explicit user confirmation',
        'labels_included': False, 'dev_audio_included': False, 'eval_included': False}
    directory = ROOT / 'outputs/colab'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'ssl_colab_smoke_v1.zip'
    size = sum(path.stat().st_size for path in mapping.values())
    if size > 500_000_000:
        raise ValueError('Colab bundle size exceeds 500 MB')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
        for name, path in mapping.items():
            archive.write(path, name)
    with zipfile.ZipFile(target) as archive:
        if set(archive.namelist()) != set(mapping) | {'manifest.json'}:
            raise ValueError('Unexpected archive payload')
        for name, expected in hashes.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError('Archived payload hash changed')
    result = {'status': 'prepared_locally_upload_not_authorized_by_auto_review',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'bundle_path': str(target.relative_to(ROOT)),
        'bundle_bytes': target.stat().st_size, 'bundle_sha256': sha(target),
        'notebook_path': 'notebooks/ssl_colab_smoke.ipynb',
        'notebook_sha256': sha(ROOT / 'notebooks/ssl_colab_smoke.ipynb'), 'manifest': manifest,
        'uploaded': False, 'colab_executed': False,
        'notice': 'Google Colab notebook upload was rejected by automatic approval review: general Colab authorization did not specifically authorize sending local project payload. Request explicit approval before uploading notebook and bundle.'}
    with (ROOT / 'docs/validation/localization_ssl_colab_bundle_v1.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k: result[k] for k in ['status', 'bundle_bytes', 'bundle_sha256']}, indent=2))


if __name__ == '__main__':
    main()
