"""Fetch only the frozen official checkpoint, with bounded transfer and SHA-256."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audio_deepfake.resources import memory_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/localization_ssl_v1.json').read_text())
    pin = config['checkpoint']
    url = f"https://huggingface.co/{pin['repo_id']}/resolve/{pin['revision']}/{pin['file']}"
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'url': url, 'bytes': pin['size_bytes']}))
        return
    receipt = ROOT / 'docs/validation/localization_ssl_checkpoint_v1.json'
    directory = ROOT / 'data/models/wav2vec2-base-v1'
    target = directory / pin['file']
    partial = target.with_suffix('.bin.part')
    if receipt.exists() or target.exists() or partial.exists():
        raise ValueError('Existing checkpoint/attempt must be reviewed, never overwritten')
    if pin['size_bytes'] > config['budgets']['checkpoint_transfer_bytes']:
        raise ValueError('Checkpoint exceeds transfer budget')
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'preprocessor_config.json', 'README.md'):
        raw = (ROOT / '.cache/representation-sources' / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin['metadata_sha256'][name]:
            raise ValueError('Pinned model metadata changed')
        with (directory / name).open('xb') as out:
            out.write(raw)
    result = {'status': 'started', 'source_url': url, 'revision': pin['revision'],
              'started_utc': datetime.now(timezone.utc).isoformat(), 'received_bytes': 0,
              'training_started': False, 'checkpoint_sha256_expected': pin['sha256']}
    start = time.monotonic()
    digest = hashlib.sha256()
    try:
        if shutil.disk_usage(ROOT).free < config['budgets']['minimum_free_disk_bytes']:
            raise RuntimeError('Need at least 20 GiB free disk')
        result['memory_before'] = memory_snapshot()
        with urllib.request.urlopen(url, timeout=30) as response, partial.open('xb') as out:
            if response.status != 200:
                raise RuntimeError('Expected complete HTTP 200 response')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != pin['size_bytes']:
                raise RuntimeError('Unexpected content length')
            while True:
                if time.monotonic() - start > 600:
                    raise RuntimeError('Checkpoint transfer deadline exceeded')
                chunk = response.read(min(1024**2, pin['size_bytes'] - result['received_bytes'] + 1))
                if not chunk:
                    break
                result['received_bytes'] += len(chunk)
                if result['received_bytes'] > pin['size_bytes']:
                    raise RuntimeError('Received more than pinned size')
                if shutil.disk_usage(ROOT).free < config['budgets']['minimum_free_disk_bytes']:
                    raise RuntimeError('Free disk reserve reached')
                digest.update(chunk)
                out.write(chunk)
        result['sha256'] = digest.hexdigest()
        if result['received_bytes'] != pin['size_bytes'] or result['sha256'] != pin['sha256']:
            raise RuntimeError('Checkpoint length/hash verification failed')
        partial.rename(target)
        result['status'] = 'verified'
    except BaseException as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - start,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
