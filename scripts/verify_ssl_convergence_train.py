"""Independent no-fit train-artifact verification before fixed dev scoring."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_localization_ssl_head import read, sha, save


def verify(train, cache):
    import numpy as np
    from audio_deepfake.localization_ssl_head import SSLHead, sample_digest
    from full_run_job import base_python
    status, result, config = [read(train / p) for p in ('status.json', 'results.json', 'config.json')]
    state = read(train / 'worker-state.json')
    assert status['status'] == 'complete' and status['exit_code'] == 0
    assert result['status'] == 'complete_train_only' and result['convergence_verified'] is True
    assert result['dev_scores_produced'] is False and result['eval_accessed'] is False
    assert not (train / 'dev_scores.json').exists() and not (train / 'timelines').exists()
    for key, filename in [('results_sha256', 'results.json'), ('model_sha256', 'model.pkl'), ('config_sha256', 'config.json')]:
        assert status[key] == sha(train / filename)
    assert result['model_sha256'] == sha(train / 'model.pkl') and result['config_sha256'] == sha(train / 'config.json')
    assert status['worker_pid'] == state['worker_pid'] and Path(state['worker_executable']).samefile(base_python())
    assert status['command'][1:3] == ['-S', '-u'] and status['elapsed_seconds'] < 300
    assert 0 < status['peak_rss_bytes'] <= 3 * 1024**3
    assert status['peak_rss_bytes'] >= result['memory']['peak_rss_bytes']
    assert status['last_memory']['system_available_ram_bytes'] >= 2 * 1024**3
    for p, digest in config['source_sha256'].items():
        assert sha(ROOT / p) == digest, p
    identity = read(cache / 'identity.json')
    assert config['cache_identity'] == identity
    metadata = result['training']
    assert metadata['warnings'] == [] and metadata['max_iter'] == 3000
    assert metadata['training_ids'] == config['selected_ids']['train']
    assert all(0 < n < 3000 for n in metadata['iterations'])
    pairs = metadata['retained_sample_ids']
    assert sample_digest(pairs) == metadata['retained_sample_ids_sha256'] == config['lfcc_sampling_reference']['retained_sample_ids_sha256']
    assert len(pairs) == 9048
    arrays = {}
    for uid, _ in pairs:
        if uid not in arrays:
            p = cache / 'cache/train' / (uid + '.npy')
            assert sha(p) == config['cache_receipts']['train/' + uid]['features_sha256']
            arrays[uid] = np.load(p, allow_pickle=False)
    x = np.stack([arrays[uid][i] for uid, i in pairs])
    assert hashlib.sha256(x.tobytes()).hexdigest() == metadata['retained_feature_bytes_sha256']
    # Load only the locally generated, hash-checked model. No fit or scoring.
    with (train / 'model.pkl').open('rb') as f:
        model = pickle.load(f)
    assert isinstance(model, SSLHead) and model.metadata == metadata
    assert model.classifier.max_iter == 3000 and model.classifier.solver == 'lbfgs'
    assert model.classifier.C == 1.0 and model.classifier.tol == .0001 and model.classifier.random_state == 45
    assert np.array_equal(model.classifier.classes_, [0, 1])
    assert np.array_equal(model.classifier.n_iter_, metadata['iterations'])
    assert model.scaler.n_samples_seen_ == 9048
    xd = x.astype(np.float64)
    np.testing.assert_allclose(model.scaler.mean_, xd.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.scaler.var_, xd.var(axis=0), rtol=1e-12, atol=1e-12)
    assert np.isfinite(model.classifier.coef_).all() and np.isfinite(model.classifier.intercept_).all()
    return {'status': 'verified_train_only', 'protocol_id': config['protocol_id'],
        'train_dir': train.relative_to(ROOT).as_posix(), 'train_results_sha256': sha(train / 'results.json'),
        'train_status_sha256': sha(train / 'status.json'), 'train_config_sha256': sha(train / 'config.json'),
        'model_sha256': sha(train / 'model.pkl'), 'cache_identity_sha256': identity['identity_sha256'],
        'convergence_verified': True, 'dev_scores_absent': True, 'eval_enabled': False,
        'independent_checks': ['source/input hashes', 'real worker PID and resource limits',
            'ordered retained train feature bytes reconstructed', 'scaler mean/variance independently recomputed',
            'model optimizer, iterations and finite coefficients; no fit or decision_function'],
        'iterations': metadata['iterations'], 'peak_rss_bytes': status['peak_rss_bytes'],
        'elapsed_seconds': status['elapsed_seconds'], 'verifier_sha256': sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--train-run', type=Path)
    parser.add_argument('--cache-run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.execute:
        audit = verify(args.train_run.resolve(), args.cache_run.resolve())
        save(args.output.resolve(), audit)
        print(json.dumps(audit, indent=2))
    else:
        print(json.dumps({'status': 'plan_only', 'training_started': False, 'dev_scoring_enabled': False}))
