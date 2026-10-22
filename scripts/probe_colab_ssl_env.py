"""Retry only the corrected environment probe; preserve attempt 1 evidence."""
import json
from pathlib import Path
import subprocess

probe = """import json, importlib.metadata as m, torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import numpy, soundfile
assert torch.cuda.is_available()
x = torch.ones((2, 2), device='cuda')
value = (x @ x).sum().item()
assert value == 8, value
print(json.dumps({'versions': {d.metadata['Name']: d.version for d in m.distributions()},
                  'gpu': torch.cuda.get_device_name(0), 'cuda_matmul_sum': value}, indent=2))
"""
result = subprocess.run(['/content/ssl-isolated-v1/bin/python', '-c', probe],
                        capture_output=True, text=True, timeout=90)
target = Path('/content/ssl_isolated_probe_v2.txt')
with target.open('x') as stream:
    stream.write(result.stdout + result.stderr)
print(result.stdout + result.stderr)
result.check_returncode()
with Path('/content/ssl_isolated_environment.json').open('x') as stream:
    stream.write(result.stdout)
print('Isolated CUDA and encoder imports passed. Encoder extraction has not started.')
