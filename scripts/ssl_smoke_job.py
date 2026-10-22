"""Run the one-recording frozen encoder smoke under independent Windows limits."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import site
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from full_run_job import base_python, windows_memory, write_json, acquire_lock, stop_owned_child


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--job-dir', type=Path)
    args = parser.parse_args()
    if args.job_dir is not None:
        directory = args.job_dir.resolve()
        if not directory.is_relative_to(ROOT / 'outputs/jobs'):
            raise ValueError('Job must be inside outputs/jobs')
    if args.worker:
        if args.job_dir is None or not args.execute:
            raise ValueError('Worker requires explicit execution directory')
        packages = ROOT / '.cache/ssl-env/venv/Lib/site-packages'
        sys.path.insert(0, str(packages))
        site.addsitedir(str(packages))
        sys.path.insert(0, str(ROOT / 'src'))
        sys.argv = ['scripts/ssl_smoke_worker.py', '--execute', '--output-dir', str(directory / 'smoke')]
        runpy.run_path(str(ROOT / sys.argv[0]), run_name='__main__')
        return
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'training_started': False, 'worker_seconds': 180}))
        return
    if args.job_dir is None or directory.exists():
        raise ValueError('Require fresh --job-dir')
    config = json.loads((ROOT / 'configs/localization_ssl_v1.json').read_text())
    budgets = config['budgets']
    memory = windows_memory()
    if memory['system_available_ram_bytes'] < budgets['minimum_start_available_ram_bytes']:
        raise RuntimeError('Smoke start requires at least 4 GiB available RAM')
    if shutil.disk_usage(ROOT).free < budgets['minimum_free_disk_bytes']:
        raise RuntimeError('Smoke start requires at least 20 GiB free disk')
    directory.mkdir(parents=True)
    lock = ROOT / 'outputs/ssl_smoke.lock'
    source_paths = ['scripts/ssl_smoke_job.py', 'scripts/ssl_smoke_worker.py',
                    'src/audio_deepfake/localization_ssl_geometry.py', 'configs/localization_ssl_v1.json',
                    'requirements/ssl-windows.lock', 'scripts/full_run_job.py',
                    'src/audio_deepfake/resources.py']
    pins = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    acquire_lock(lock, {'pid': os.getpid(), 'job_dir': str(directory)})
    status = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'training_started': False, 'eval_enabled': False, 'peak_rss_bytes': 0,
              'source_sha256': pins, 'start_memory': memory, 'worker_seconds': 180}
    child = None
    start = time.monotonic()
    try:
        write_json(directory / 'status.json', status)
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[key] = '2'
        with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
            command = [base_python(), '-S', '-u', str(Path(__file__).resolve()), '--worker',
                       '--execute', '--job-dir', str(directory)]
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            status['child_pid'] = child.pid
            while child.poll() is None:
                if time.monotonic() - start >= budgets['smoke_timeout_seconds']:
                    raise RuntimeError('External smoke timeout')
                memory = windows_memory(child.pid, process_handle=child._handle)
                status['peak_rss_bytes'] = max(status['peak_rss_bytes'], memory['child_peak_rss_bytes'])
                status['last_memory'] = memory
                status['elapsed_seconds'] = time.monotonic() - start
                if status['peak_rss_bytes'] > budgets['worker_peak_bytes']:
                    raise RuntimeError('Worker peak RAM limit exceeded')
                if memory['system_available_ram_bytes'] < budgets['minimum_runtime_available_ram_bytes']:
                    raise RuntimeError('System available RAM stop boundary reached')
                if shutil.disk_usage(ROOT).free < budgets['minimum_free_disk_bytes']:
                    raise RuntimeError('Free disk stop boundary reached')
                write_json(directory / 'status.json', status)
                time.sleep(.1)
            status['exit_code'] = child.wait()
        if status['exit_code'] != 0:
            raise RuntimeError(f"Smoke worker exit {status['exit_code']}; inspect stderr.log")
        final_memory = windows_memory(child.pid, process_handle=child._handle)
        status['peak_rss_bytes'] = max(status['peak_rss_bytes'], final_memory['child_peak_rss_bytes'])
        if status['peak_rss_bytes'] > budgets['worker_peak_bytes']:
            raise RuntimeError('Final worker lifetime peak exceeds limit')
        for relative, digest in pins.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Source changed during smoke')
        result = json.loads((directory / 'smoke/result.json').read_text(encoding='utf-8'))
        if result['status'] != 'complete' or result['training_started']:
            raise RuntimeError('Missing real smoke completion')
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='stopped', error=str(error))
        raise
    finally:
        if child is not None:
            stop_owned_child(child)
        status.update(child_pid=None, elapsed_seconds=time.monotonic() - start,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / 'status.json', status)
        lock.unlink()


if __name__ == '__main__':
    main()
