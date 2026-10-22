"""Offline frozen float32 encoder extraction; run through ssl_extract_job.py only."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import subprocess
import time

CONFIG_SHA = '44b1e5d8ae54a9f259fa64051b4ef7ca7267c476454116dc80cb7acaedd9e264'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def local(bundle, relative):
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle) or not path.is_file():
        raise ValueError('Payload path missing or escapes bundle')
    return path


def inputs(bundle, model):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    config_path = local(bundle, manifest['config_file'])
    if sha(config_path) != CONFIG_SHA:
        raise ValueError('Frozen configuration changed')
    config = json.loads(config_path.read_text())
    for relative, expected in manifest['files_sha256'].items():
        if 'eval' in Path(relative).parts or sha(local(bundle, relative)) != expected:
            raise ValueError('Payload source/hash differs or eval is present')
    for name in ('ssl_extract_worker.py', 'ssl_extract_job.py', 'localization_ssl_geometry.py'):
        if name not in manifest['files_sha256']:
            raise ValueError('Required source provenance is missing')
    for name in ('ssl_extract_worker.py', 'ssl_extract_job.py'):
        if sha(Path(__file__).with_name(name)) != manifest['files_sha256'][name]:
            raise ValueError('Executed source differs from frozen payload')
    if manifest['files_sha256']['localization_ssl_geometry.py'] != 'ddd6e4fe0de0111560281a3f7281220799f16a24554b9b11a4ea27618480cfe5':
        raise ValueError('Frozen geometry source differs')
    audits = []
    for key, frozen in (('selection_file', 'docs/validation/partial_selection_v1.json'),
                        ('alignment_file', 'docs/validation/localization_alignment.json')):
        path = local(bundle, manifest[key])
        if sha(path) != config['input_sha256'][frozen]:
            raise ValueError('Frozen selection/alignment changed')
        audits.append(json.loads(path.read_text()))
    selection, alignment = audits
    expected = {}
    for split, count in (('train', 128), ('dev', 64)):
        selected = selection['splits'][split]['selected_records']
        ids = {r['utterance_id'] for r in selected}
        if len(selected) != count or len(ids) != count:
            raise ValueError('Frozen split count differs')
        rows = {r['utterance_id']: r for r in alignment['splits'][split]['records']}
        for identifier in ids:
            row = rows[identifier]
            if not row['candidate_match'] or (row['sample_rate'], row['channels']) != (16000, 1):
                raise ValueError('Unsupported audited audio')
            expected[(split, identifier)] = row
    seen = set()
    for row in manifest['records']:
        key = (row['split'], row['utterance_id'])
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unselected record')
        seen.add(key)
        frozen = expected[key]
        for name in ('sample_count', 'audio_sha256', 'coarse_count_20ms'):
            if row[name] != frozen[name]:
                raise ValueError('Record differs from frozen alignment')
        if row['audio_file'] != f'audio/{key[0]}/{key[1]}.wav':
            raise ValueError('Audio path must identify its official split')
        if row['audio_file'] not in manifest['files_sha256']:
            raise ValueError('Audio provenance is absent from payload manifest')
        if sha(local(bundle, row['audio_file'])) != row['audio_sha256']:
            raise ValueError('Audio checksum differs')
    if seen != set(expected):
        raise ValueError('Incomplete frozen selection')
    model_hashes = {}
    for name in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin'):
        path = model / name
        wanted = config['checkpoint']['sha256'] if name.endswith('.bin') else config['checkpoint']['metadata_sha256'][name]
        if name.endswith('.bin') and path.stat().st_size != config['checkpoint']['size_bytes']:
            raise ValueError('Checkpoint size differs')
        model_hashes[name] = sha(path)
        if model_hashes[name] != wanted:
            raise ValueError('Checkpoint/processor checksum differs')
    return manifest, config, model_hashes


def identity(bundle, model, device):
    manifest, config, model_hashes = inputs(bundle, model)
    environment = sorted([d.metadata['Name'], d.version] for d in importlib.metadata.distributions())
    value = {'manifest_sha256': sha(bundle / 'manifest.json'), 'source_sha256': manifest['files_sha256'],
             'model_sha256': model_hashes, 'environment': environment, 'device': device,
             'python': sys.version, 'platform': platform.platform(),
             'host': platform.node(), 'cpu': platform.processor(),
             'worker_sha256': sha(__file__), 'supervisor_sha256': sha(Path(__file__).with_name('ssl_extract_job.py'))}
    if device == 'cuda':
        probe = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True, timeout=10)
        value['cuda_hardware'] = probe.stdout.strip()
        value['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    return manifest, config, {'identity_sha256': digest(value), 'identity': value}


def validate_cache(output, row, identity_sha, np):
    base = output / 'cache' / row['split'] / row['utterance_id']
    feature, receipt = base.with_suffix('.npy'), base.with_suffix('.json')
    if not feature.exists() and not receipt.exists():
        return False
    if not feature.is_file() or not receipt.is_file():
        raise ValueError('Partial cache requires explicit investigation; no automatic overwrite')
    record = json.loads(receipt.read_text())
    expected = {'identity_sha256': identity_sha, 'audio_sha256': row['audio_sha256'],
                'shape': [row['coarse_count_20ms'], 768], 'dtype': 'float32',
                'source_frame_count': 1 + (row['sample_count'] - 400) // 320,
                'recording_id': row['utterance_id'], 'split': row['split'], 'status': 'complete'}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Cache provenance/geometry differs')
    if sha(feature) != record['features_sha256'] or feature.stat().st_size != record['features_bytes']:
        raise ValueError('Cache checksum/size differs')
    values = np.load(feature, allow_pickle=False)
    if list(values.shape) != expected['shape'] or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Cache shape/dtype/finite check failed')
    return True


def worker(bundle, model_dir, output, device):
    from ssl_extract_job import guard, cache_bytes
    guard(output, initial=True, pid=os.getpid())
    manifest, config, ident = identity(bundle, model_dir, device)
    if json.loads((output / 'identity.json').read_text()) != ident:
        raise ValueError('Worker environment identity differs')
    version = importlib.metadata.version('torch').split('+')[0].split('.')
    if tuple(map(int, version[:2])) < (2, 10):
        raise RuntimeError('Torch >= 2.10 required for restricted loading')
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    sys.path.insert(0, str(bundle))
    from localization_ssl_geometry import convolution_geometry, frame_mapping
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(45)
    hardware = {'device': device, 'torch': torch.__version__, 'cuda_version': torch.version.cuda}
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback')
        torch.cuda.manual_seed_all(45)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        hardware['gpu_name'] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    pending = [r for r in manifest['records'] if not validate_cache(output, r, ident['identity_sha256'], np)]
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    if not processor.do_normalize or processor.sampling_rate != 16000:
        raise ValueError('Processor normalization differs')
    before_load = guard(output, initial=True, pid=os.getpid())
    model, loading = Wav2Vec2Model.from_pretrained(str(model_dir), local_files_only=True,
        weights_only=True, use_safetensors=False, attn_implementation='eager',
        output_loading_info=True, torch_dtype=torch.float32)
    if (loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs')
            or any(not k.startswith(('quantizer.', 'project_hid.', 'project_q.')) for k in loading.get('unexpected_keys', []))):
        raise ValueError('Encoder load mismatch')
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    if any(p.requires_grad or p.dtype != torch.float32 or p.device.type != device for p in model.parameters()):
        raise ValueError('Frozen float32 device invariant failed')
    if (*convolution_geometry(model.config.conv_kernel, model.config.conv_stride), model.config.hidden_size,
            model.config.num_hidden_layers) != (400, 320, 768, 12):
        raise ValueError('Model convolution geometry differs')
    for row in sorted(pending, key=lambda r: (r['split'], r['utterance_id'])):
        guard(output, pid=os.getpid())
        started = time.monotonic()
        path = local(bundle, row['audio_file'])
        if sha(path) != row['audio_sha256']:
            raise ValueError('Audio changed during run')
        audio, rate = sf.read(path, dtype='float32', always_2d=False)
        if audio.shape != (row['sample_count'],) or rate != 16000 or not np.isfinite(audio).all():
            raise ValueError('Actual waveform differs')
        mapping = frame_mapping(len(audio))
        inputs_ = processor(audio, sampling_rate=16000, return_tensors='pt', padding=False, return_attention_mask=False)
        if set(inputs_) != {'input_values'} or tuple(inputs_['input_values'].shape) != (1, len(audio)):
            raise ValueError('Audio padding/mask differs')
        with torch.inference_mode():
            states = model(input_values=inputs_['input_values'].to(device), output_hidden_states=False,
                           output_attentions=False, return_dict=True).last_hidden_state
        if device == 'cuda':
            torch.cuda.synchronize()
        if tuple(states.shape) != (1, mapping['source_frame_count'], 768):
            raise ValueError('Actual encoder geometry differs')
        features = states[0].cpu().numpy()[mapping['frame_indices']].copy()
        if features.shape != (row['coarse_count_20ms'], 768) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError('Aligned features invalid')
        base = output / 'cache' / row['split'] / row['utterance_id']
        base.parent.mkdir(parents=True, exist_ok=True)
        temporary = base.with_suffix('.npy.tmp')
        with temporary.open('xb') as stream:
            np.save(stream, features, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if cache_bytes(output) > config['budgets']['feature_cache_bytes']:
            raise RuntimeError('Feature storage cap reached')
        target = base.with_suffix('.npy')
        temporary.replace(target)
        receipt = {'status': 'complete', 'recording_id': row['utterance_id'], 'split': row['split'],
                   'identity_sha256': ident['identity_sha256'], 'audio_sha256': row['audio_sha256'],
                   'features_sha256': sha(target), 'features_bytes': target.stat().st_size,
                   'shape': list(features.shape), 'dtype': str(features.dtype),
                   'source_frame_count': mapping['source_frame_count'], 'extraction_seconds': time.monotonic() - started,
                   'resources': guard(output, pid=os.getpid())}
        save(base.with_suffix('.json'), receipt)
        del states, features, inputs_, audio
    if device == 'cuda':
        hardware.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save(output / 'worker-result.json', {'status': 'complete', 'identity_sha256': ident['identity_sha256'],
         'record_count': len(manifest['records']), 'training_started': False, 'eval_accessed': False,
         'native_frame_labels_opened': False, 'label_summary_metadata_opened': True,
         'hardware': hardware, 'resources_before_model_load': before_load,
         'resources_before_completion': guard(output, pid=os.getpid())})
