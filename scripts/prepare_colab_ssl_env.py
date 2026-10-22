"""Create an isolated Colab encoder environment, reusing runtime GPU wheels.

Does not change the encoder, install another torch build or modify global packages.
Run after preserving the original dependency resolution and failed pip check.
"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import venv

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    root = Path('/content/ssl-isolated-v1')
    assert not root.exists(), 'Preserve previous environment attempts'
    resolution = json.loads(Path('/content/ssl_resolve.json').read_text())
    owned = {canonicalize_name(i['metadata']['name']) for i in resolution['install']}
    assert 'torch' not in owned
    # Only the runtime torch dependency closure may enter the isolated environment.
    pending, reused = ['torch'], {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in owned or name in reused:
            continue
        dist = metadata.distribution(name)
        reused[name] = dist
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(root)
    python = root / 'bin/python'
    site = next((root / 'lib').glob('python*/site-packages'))
    links = {}
    provenance = {}
    for name, dist in sorted(reused.items()):
        provenance[name] = {'version': dist.version, 'root': str(dist.locate_file('')),
                            'metadata_hashes': {}}
        for file in dist.files or []:
            if not file.parts or file.parts[0] in ('..', '.'):
                continue
            source = Path(dist.locate_file(file.parts[0])).resolve()
            if file.name in ('RECORD', 'METADATA') and '.dist-info' in str(file):
                provenance[name]['metadata_hashes'][str(file)] = hashlib.sha256(
                    Path(dist.locate_file(file)).read_bytes()).hexdigest()
            target = site / file.parts[0]
            if target.is_symlink():
                assert target.resolve() == source, 'Conflicting reused package paths'
            elif target.exists():
                raise RuntimeError(f'Unexpected existing environment path: {target}')
            else:
                target.symlink_to(source, target_is_directory=source.is_dir())
            links[str(target.relative_to(root))] = str(source)
    receipt = {'runtime_torch_packages': {k: d.version for k, d in reused.items()},
               'provenance': provenance,
               'shared_paths': links, 'system_site_packages': False,
               'lock_sha256': hashlib.sha256(Path('/content/ssl-linux.lock').read_bytes()).hexdigest()}
    Path('/content/ssl_isolation.json').write_text(json.dumps(receipt, indent=2))
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
                    '--no-deps', '--require-hashes', '-r', '/content/ssl-linux.lock'], check=True)
    check = subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'check'],
                           capture_output=True, text=True)
    Path('/content/ssl_isolated_pip_check.txt').write_text(check.stdout + check.stderr)
    print(check.stdout + check.stderr, flush=True)
    check.check_returncode()
    probe = """import json, importlib.metadata as m, torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy, soundfile
assert torch.cuda.is_available()
x = torch.ones((2, 2), device='cuda')
assert (x @ x).sum().item() == 8
print(json.dumps({'versions': {d.metadata['Name']: d.version for d in m.distributions()},
                  'gpu': torch.cuda.get_device_name(0), 'cuda_matmul': 'passed'}, indent=2))
"""
    result = subprocess.run([str(python), '-c', probe], capture_output=True, text=True, timeout=90)
    Path('/content/ssl_isolated_probe.txt').write_text(result.stdout + result.stderr)
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()
    Path('/content/ssl_isolated_environment.json').write_text(result.stdout)
    print('Isolated encoder environment passed. Use /content/ssl-isolated-v1/bin/python for the worker.')


if __name__ == '__main__':
    main()
