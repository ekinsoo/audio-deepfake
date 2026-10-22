"""Separate preregistered train-only convergence fit; plan only by default.

No dev scores, encoder execution, downloads, tuning or automatic retries.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'localization_ssl_convergence_v1'
CONFIG = 'configs/localization_ssl_convergence_v1.json'
PLAN = 'docs/LOCALIZATION_SSL_CONVERGENCE_PLAN.md'
sys.path.insert(0, str(ROOT / 'scripts'))
from run_localization_ssl_head import sha, read, save, worker_state


def protocol():
    followup = read(ROOT / CONFIG)
    original = read(ROOT / followup['original_config'])
    if sha(ROOT / followup['original_config']) != followup['original_config_sha256']:
        raise ValueError('Original frozen comparison changed')
    head = followup['head']
    expected = {'seed': 45, 'frames_per_class_cap': 50000, 'expected_retained_per_class': 4524,
                'C': 1.0, 'solver': 'lbfgs', 'max_iter': 3000, 'tol': 0.0001, 'numerical_threads': 1}
    if any(head.get(k) != v for k, v in expected.items()):
        raise ValueError('Preregistered followup optimizer differs')
    if followup['dev_scoring_enabled'] is not False or followup['eval_enabled'] is not False:
        raise ValueError('Train-only followup forbids dev/eval scoring')
    budgets = {'head_fit_timeout_seconds': 300, 'worker_peak_bytes': 3 * 1024**3,
               'minimum_runtime_available_ram_bytes': 2 * 1024**3, 'minimum_free_disk_bytes': 20 * 1024**3}
    if followup['budgets'] != budgets:
        raise ValueError('Preregistered resource caps differ')
    return followup, original


def run_worker(cache, output):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    from audio_deepfake.localization_model import TemporalConfig
    from audio_deepfake.localization_ssl_head import reference_sampling, train_ssl_head
    from audio_deepfake.partialspoof import labels_for_audio
    from audio_deepfake.resources import memory_snapshot
    from run_localization_pilot import validate_gate
    from run_localization_ssl_head import checked_cache

    started = time.monotonic()
    followup, original = protocol()
    baseline = read(ROOT / 'docs/validation/localization_pilot_v1.json')
    for package, expected in baseline['results']['training']['versions'].items():
        if version(package) != expected:
            raise ValueError('Numerical environment differs from original LFCC baseline')
    for path, expected in original['input_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen input changed')
    dependencies = ['src/audio_deepfake/localization_model.py', 'src/audio_deepfake/partialspoof.py',
                    'src/audio_deepfake/resources.py', 'scripts/run_localization_pilot.py']
    for path in dependencies:
        if sha(ROOT / path) != baseline['config']['source_sha256'][path]:
            raise ValueError('Frozen baseline dependency changed')
    source_paths = [CONFIG, PLAN, 'scripts/run_ssl_convergence_train.py', 'scripts/run_localization_ssl_head.py',
                    'src/audio_deepfake/localization_ssl_head.py', 'scripts/full_run_job.py',
                    'scripts/ssl_extract_worker.py', 'scripts/ssl_extract_job.py',
                    'src/audio_deepfake/localization_ssl_geometry.py', followup['original_config'],
                    'docs/validation/localization_ssl_sampling_preflight_v1.json',
                    'docs/validation/localization_ssl_extraction_v1.json',
                    'docs/validation/localization_ssl_head_convergence_v2.json',
                    *dependencies, *original['input_sha256']]
    pins = {p: sha(ROOT / p) for p in source_paths}
    selection = read(ROOT / 'docs/validation/partial_selection_v1.json')
    alignment = read(ROOT / 'docs/validation/localization_alignment.json')
    indexed = validate_gate(selection, alignment,
        original['input_sha256']['docs/validation/partial_selection_v1.json'],
        original['input_sha256']['data/partialspoof/selected_labels_v1.json'])
    identity, receipts = checked_cache(cache, indexed, original)
    if identity['identity_sha256'] != followup['extraction_identity_sha256']:
        raise ValueError('Followup extraction identity differs')
    # Shared compact JSON includes dev labels. Only train arrays are retained or used.
    train_labels = read(ROOT / 'data/partialspoof/selected_labels_v1.json')['splits']['train']

    def guard():
        row = memory_snapshot()
        if (row['system_available_ram_bytes'] is None or row['system_available_ram_bytes'] < 2 * 1024**3
                or row['peak_rss_bytes'] > 3 * 1024**3 or shutil.disk_usage(ROOT).free < 20 * 1024**3
                or time.monotonic() - started >= 300):
            raise RuntimeError('Train-only resource/time boundary reached')
        return row

    def truth(row):
        if row['split'] != 'train':
            raise ValueError('Only train labels may enter followup computations')
        uid = row['utterance_id']
        return labels_for_audio(train_labels[uid], sample_count=indexed['train'][uid]['sample_count'])

    def loader(row):
        guard()
        if row['split'] != 'train':
            raise ValueError('Only train features may enter fit')
        uid = row['utterance_id']
        path = cache / 'cache/train' / (uid + '.npy')
        if sha(path) != receipts['train/' + uid]['features_sha256']:
            raise ValueError('Training feature cache changed')
        return np.load(path, allow_pickle=False), truth(row)

    rows = [{'split': 'train', 'utterance_id': uid} for uid in sorted(indexed['train'])]
    temporal = TemporalConfig(seed=45, frames_per_class=50000, max_iter=3000, regularization_c=1.0)
    reference = reference_sampling(rows, truth, config=temporal)
    if (reference['frames_retained_per_class'] != 4524
            or reference['retained_sample_ids_sha256'] != followup['retained_sample_ids_sha256']):
        raise ValueError('Train-only sampling differs from preregistered original identities')
    save(output / 'config.json', {'status': 'frozen_before_train_only_fit', 'protocol_id': PROTOCOL,
         'created_utc': datetime.now(timezone.utc).isoformat(), 'followup_config_sha256': sha(ROOT / CONFIG),
         'source_sha256': pins, 'cache_identity': identity, 'cache_receipts': receipts,
         'lfcc_sampling_reference': reference, 'selected_ids': {s: sorted(indexed[s]) for s in indexed},
         'timeline_ids': selection['splits']['dev']['timeline_ids'], 'head': followup['head'],
         'official_benchmark': False, 'eval_enabled': False, 'dev_scoring_enabled': False,
         'shared_dev_label_json_parsed': True, 'dev_labels_used_in_computation': False, 'preflight': guard()})
    worker_state(output, phase='training_only', training_started=True, resources_before_fit=guard())
    model = train_ssl_head(rows, loader, config=temporal, expected_retained_per_class=4524,
                           expected_sample_digest=reference['retained_sample_ids_sha256'], guard=guard)
    if model.classifier.tol != followup['head']['tol'] or any(i >= 3000 for i in model.classifier.n_iter_):
        raise ValueError('Optimizer tolerance or convergence contract differs')
    if any(sha(ROOT / path) != expected for path, expected in pins.items()):
        raise ValueError('Source/input changed during train-only fit')
    final_identity, final_receipts = checked_cache(cache, indexed, original)
    if final_identity != identity or final_receipts != receipts:
        raise ValueError('Extraction cache changed during train-only fit')
    guard()
    with (output / 'model.pkl').open('xb') as stream:
        pickle.dump(model, stream, protocol=5)
    save(output / 'results.json', {'status': 'complete_train_only', 'protocol_id': PROTOCOL,
         'config_sha256': sha(output / 'config.json'), 'model_sha256': sha(output / 'model.pkl'),
         'training': model.metadata, 'convergence_verified': True, 'dev_scores_produced': False,
         'eval_accessed': False, 'official_benchmark': False, 'elapsed_seconds': time.monotonic() - started,
         'memory': guard(), 'limitations': ['Separate followup after the original fixed-300 convergence failure',
             'No development scores or accuracy claims; independent verification required before separate dev scoring']})
    worker_state(output, phase='complete_train_only', training_started=True, resources_after_completion=guard())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--cache-run', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'protocol_id': PROTOCOL, 'max_iter': 3000,
                          'training_started': False, 'dev_scoring_enabled': False, 'timeout_seconds': 300}))
        return
    if args.cache_run is None or args.output_dir is None:
        parser.error('Execution requires cache-run and fresh output-dir')
    cache, output = args.cache_run.resolve(), args.output_dir.resolve()
    if (not cache.is_relative_to(ROOT / 'outputs') or not output.is_relative_to(ROOT / 'outputs')
            or output == cache or output.is_relative_to(cache) or cache.is_relative_to(output)):
        raise ValueError('Require separate cache/output paths under ignored outputs')
    if args.worker:
        from full_run_job import configure_project_environment
        configure_project_environment()
        worker_state(output, phase='validating_inputs', training_started=False)
        try:
            run_worker(cache, output)
        except BaseException as error:
            worker_state(output, phase='failed', error_type=type(error).__name__, error=str(error))
            raise
        return
    protocol()
    if os.name != 'nt' or output.exists():
        raise ValueError('Require Windows external supervisor and fresh output')
    from full_run_job import base_python, windows_memory, stop_owned_child
    pins = {path: sha(ROOT / path) for path in (CONFIG, PLAN, 'scripts/run_ssl_convergence_train.py',
        'scripts/full_run_job.py', 'scripts/run_localization_ssl_head.py', 'src/audio_deepfake/localization_ssl_head.py')}
    output.mkdir(parents=True)
    command = [base_python(), '-S', '-u', str(Path(__file__).resolve()), '--worker', '--execute',
               '--cache-run', str(cache), '--output-dir', str(output)]
    status = {'status': 'stopped', 'protocol_id': PROTOCOL, 'training_started': False,
              'dev_scoring_enabled': False, 'eval_enabled': False, 'source_sha256': pins,
              'command': command, 'timeout_seconds': 300, 'peak_rss_bytes': 0, 'exit_code': None}
    child = None
    started = time.monotonic()

    def external_guard(final=False):
        memory = windows_memory(child.pid, process_handle=child._handle) if child else windows_memory()
        status['peak_rss_bytes'] = max(status['peak_rss_bytes'], memory.get('child_peak_rss_bytes', 0))
        status['last_memory'] = memory
        if (time.monotonic() - started >= 300 or status['peak_rss_bytes'] > 3 * 1024**3
                or memory['system_available_ram_bytes'] < 2 * 1024**3 or shutil.disk_usage(ROOT).free < 20 * 1024**3):
            raise RuntimeError('External train-only timeout/RAM/disk boundary reached')
        return memory

    try:
        status['prelaunch_resources'] = external_guard()
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        env['PYTHONNOUSERSITE'] = '1'
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            status['worker_pid'] = child.pid
            while child.poll() is None:
                external_guard()
                time.sleep(.1)
            status['exit_code'] = child.wait()
            external_guard(final=True)
        state = read(output / 'worker-state.json')
        status.update(training_started=state['training_started'], worker_phase=state['phase'],
                      worker_executable=state['worker_executable'], worker_state_sha256=sha(output / 'worker-state.json'))
        if state['worker_pid'] != child.pid or state['worker_executable'].lower() != base_python().lower():
            raise ValueError('Observed PID/executable differs from real base Python worker')
        if child.returncode != 0:
            raise RuntimeError('Train-only worker failed; inspect retained state/logs')
        result = read(output / 'results.json')
        if (result['status'] != 'complete_train_only' or result['dev_scores_produced'] is not False
                or result['eval_accessed'] is not False or result['model_sha256'] != sha(output / 'model.pkl')):
            raise ValueError('Missing valid train-only completion')
        if any(sha(ROOT / path) != expected for path, expected in pins.items()):
            raise ValueError('Supervisor/worker sources changed during fit')
        external_guard(final=True)
        status.update(status='complete', results_sha256=sha(output / 'results.json'),
                      model_sha256=sha(output / 'model.pkl'), config_sha256=sha(output / 'config.json'))
    except BaseException as error:
        status.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None:
            stop_owned_child(child)
            status['exit_code'] = child.returncode
        if (output / 'worker-state.json').exists():
            state = read(output / 'worker-state.json')
            status.update(training_started=state['training_started'], worker_phase=state['phase'],
                          worker_state_sha256=sha(output / 'worker-state.json'))
            if state.get('error_type'):
                status.update(worker_error_type=state['error_type'], worker_error=state['error'])
        status.update(elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        save(output / 'status.json', status)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
