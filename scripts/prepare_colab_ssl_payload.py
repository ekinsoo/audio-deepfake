"""Verify a small approved upload and fetch the unchanged public checkpoint.

Transport fallback for slow browser uploads; no retry and no model execution.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

payload = Path('/content/ssl_colab_smoke_payload_v1.zip')
assert payload.stat().st_size == 1214355
assert hashlib.sha256(payload.read_bytes()).hexdigest() == '44bf452f97f4ce8bc0ccc2d9741dd9ca50704e2df1bca136d0ae8ee480dbcd2b'
bundle = Path('/content/ssl-smoke-bundle')
assert not bundle.exists(), 'Preserve existing bundle attempts'
assert shutil.disk_usage('/content').free >= 20 * 1024**3
bundle.mkdir()
with zipfile.ZipFile(payload) as archive:
    assert sum(i.file_size for i in archive.infolist()) < 5000000
    for item in archive.infolist():
        assert (bundle / item.filename).resolve().is_relative_to(bundle.resolve())
    archive.extractall(bundle)
url = 'https://huggingface.co/facebook/wav2vec2-base/resolve/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8/pytorch_model.bin'
target = bundle / 'model/pytorch_model.bin'
temporary = target.with_suffix('.partial')
receipt = {'url': url, 'status': 'started', 'bytes': 0, 'automatic_retry': False}
started = time.monotonic()
try:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=30) as response, temporary.open('xb') as stream:
        while block := response.read(1024 * 1024):
            if time.monotonic() - started > 180:
                raise TimeoutError('Checkpoint transfer exceeded 180 seconds')
            receipt['bytes'] += len(block)
            if receipt['bytes'] > 380267417:
                raise ValueError('Checkpoint exceeds pinned size')
            stream.write(block)
            digest.update(block)
    receipt['sha256'] = digest.hexdigest()
    assert receipt['bytes'] == 380267417
    assert receipt['sha256'] == '3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21'
    temporary.rename(target)
    receipt['status'] = 'verified'
except BaseException as error:
    receipt.update(status='failed', error=str(error))
    raise
finally:
    receipt['elapsed_seconds'] = time.monotonic() - started
    Path('/content/ssl_colab_checkpoint_transfer.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
