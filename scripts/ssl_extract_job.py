"""Explicit, resumable 192-recording extraction; external cumulative 1800s limit.

Plan only by default. Stopped attempts require a separate --resume invocation.
Never starts fitting, downloads weights, accesses native frame-label arrays or opens eval audio.
Frozen selection metadata includes utterance classes and label-derived summaries.
"""
import argparse
import array
import ast
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

from ssl_extract_worker import sha, save, identity, worker, inputs

GIB = 1024 ** 3
SMOKE_AUDIT_SHA = '4eee8a7fb3686f01d38290e50460324fb6d31da9d7c782bd7733c816c0a81ac0'


class ProcessProbeUnavailable(RuntimeError):
    pass


if os.name == 'nt':
    from ctypes import wintypes
    class SystemMemory(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (n, ctypes.c_ulonglong) for n in ('total', 'available', 'page', 'available_page',
                                             'virtual', 'available_virtual', 'extended')]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ('peak', 'rss', 'qpp', 'qp', 'qnpp', 'qnp', 'page', 'peak_page')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(SystemMemory)]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]


def memory(pid=None):
    if os.name == 'nt':
        system = SystemMemory()
        system.length = ctypes.sizeof(system)
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        row = {'system_available_ram_bytes': system.available}
        if pid is not None:
            handle = kernel.OpenProcess(0x1000 | 0x10, False, pid)
            if not handle:
                raise ProcessProbeUnavailable('Cannot open owned process')
            try:
                process = ProcessMemory()
                process.cb = ctypes.sizeof(process)
                if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(process), process.cb):
                    raise ProcessProbeUnavailable('Cannot read owned process memory')
                row.update(rss_bytes=process.rss, peak_rss_bytes=process.peak)
            finally:
                kernel.CloseHandle(handle)
        return row
    if sys.platform != 'linux':
        raise RuntimeError('Resource supervisor supports Windows/Linux only')
    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].split()
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    row = {'system_available_ram_bytes': int(fields['MemAvailable'][0]) * 1024}
    if pid is not None:
        try:
            fields = {line.split(':', 1)[0]: line.split(':', 1)[1].split()
                      for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line}
            row.update(rss_bytes=int(fields['VmRSS'][0]) * 1024, peak_rss_bytes=int(fields['VmHWM'][0]) * 1024)
        except (FileNotFoundError, KeyError) as error:
            raise ProcessProbeUnavailable('Process memory disappeared during exit') from error
    return row


def cache_bytes(output):
    return sum(p.stat().st_size for p in (output / 'cache').rglob('*') if p.is_file())


def guard(output, initial=False, pid=None):
    row = memory(pid)
    row['disk_free_bytes'] = shutil.disk_usage(output).free
    if row['system_available_ram_bytes'] < (4 if initial else 2) * GIB:
        raise RuntimeError('Available system RAM boundary reached')
    if row.get('peak_rss_bytes', 0) > 3 * GIB:
        raise RuntimeError('Worker peak RSS exceeds 3 GiB')
    if row['disk_free_bytes'] < 20 * GIB:
        raise RuntimeError('Free disk below 20 GiB')
    if cache_bytes(output) > GIB:
        raise RuntimeError('Feature cache storage exceeds 1 GiB')
    return row


def validate_npy(path, shape):
    """Validate a small float32 NPY without importing a numerical framework."""
    with path.open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError('Invalid NPY magic')
        version = tuple(stream.read(2))
        if version == (1, 0):
            length = struct.unpack('<H', stream.read(2))[0]
        elif version == (2, 0):
            length = struct.unpack('<I', stream.read(4))[0]
        else:
            raise ValueError('Unsupported NPY version')
        if length > 4096:
            raise ValueError('Excessive NPY header')
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        if header != {'descr': '<f4', 'fortran_order': False, 'shape': tuple(shape)}:
            raise ValueError('NPY shape/dtype/order differs')
        expected = math.prod(shape) * 4
        data = stream.read(expected + 1)
        if len(data) != expected:
            raise ValueError('NPY size differs')
        values = array.array('f')
        values.frombytes(data)
        if sys.byteorder != 'little':
            values.byteswap()
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite feature cache')


def validate_all(output, manifest, identity_sha):
    expected_paths = {str(Path(row['split']) / (row['utterance_id'] + suffix))
                      for row in manifest['records'] for suffix in ('.npy', '.json')}
    actual_paths = {str(p.relative_to(output / 'cache'))
                    for p in (output / 'cache').rglob('*') if p.is_file()}
    if actual_paths != expected_paths:
        raise ValueError('Cache inventory contains missing, extra or partial files')
    checked = []
    for row in manifest['records']:
        base = output / 'cache' / row['split'] / row['utterance_id']
        path = base.with_suffix('.npy')
        receipt = json.loads(base.with_suffix('.json').read_text())
        expected = {'status': 'complete', 'identity_sha256': identity_sha, 'split': row['split'],
                    'recording_id': row['utterance_id'], 'audio_sha256': row['audio_sha256'],
                    'shape': [row['coarse_count_20ms'], 768], 'dtype': 'float32',
                    'source_frame_count': 1 + (row['sample_count'] - 400) // 320}
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('Receipt provenance differs')
        if sha(path) != receipt['features_sha256'] or path.stat().st_size != receipt['features_bytes']:
            raise ValueError('Feature hash/size differs')
        validate_npy(path, expected['shape'])
        checked.append({'split': row['split'], 'recording_id': row['utterance_id'],
                        'features_sha256': receipt['features_sha256'],
                        'receipt_sha256': sha(base.with_suffix('.json'))})
    return checked


def remaining_budget(previous):
    used = float(previous.get('cumulative_elapsed_seconds', 0))
    if not math.isfinite(used) or used < 0 or used >= 1800:
        raise RuntimeError('Cumulative extraction budget exhausted or invalid')
    return 1800 - used


def monitor(child, output, status, deadline, start, prior, clock=time.monotonic, sleep=time.sleep):
    missing_since = None
    while child.poll() is None:
        now = clock()
        if now >= deadline:
            raise RuntimeError('Cumulative 1800-second extraction deadline reached')
        system = guard(output)
        try:
            process = memory(child.pid)
        except ProcessProbeUnavailable as error:
            missing_since = now if missing_since is None else missing_since
            if now - missing_since >= 1:
                raise RuntimeError('Process memory unavailable while child is running') from error
            status['last_process_probe_error'] = str(error)
            try:
                return child.wait(timeout=min(.1, max(0, deadline - clock())))
            except subprocess.TimeoutExpired:
                continue
        missing_since = None
        status['peak_rss_bytes'] = max(status['peak_rss_bytes'], process['peak_rss_bytes'])
        if status['peak_rss_bytes'] > 3 * GIB:
            raise RuntimeError('Worker peak RSS exceeds 3 GiB')
        status['last_memory'] = {**system, **process}
        status['cumulative_elapsed_seconds'] = prior + clock() - start
        save(output / 'status.json', status)
        sleep(min(.1, max(0, deadline - clock())))
    return child.wait()


def stop(child):
    if child is None or child.poll() is not None:
        return
    try:
        if os.name == 'nt':
            child.terminate()
        else:
            os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == 'nt':
                child.kill()
            else:
                os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def supervise(bundle, model, output, device, resume=False):
    if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError('Output must be separate from immutable payload')
    if output == model or output.is_relative_to(model) or model.is_relative_to(output):
        raise ValueError('Output must be separate from immutable model')
    guard(output if output.exists() else output.parent, initial=True)
    manifest, config, ident = identity(bundle, model, device)
    gate = bundle / manifest['smoke_audit_file']
    if sha(gate) != SMOKE_AUDIT_SHA:
        raise ValueError('Require verified fresh supervisor-v2 smoke evidence')
    audit = json.loads(gate.read_text())
    if not audit['strict_resource_gate_pass'] or audit['supervisor']['status'] != 'complete':
        raise ValueError('Strict supervised smoke did not pass')
    if resume:
        previous = json.loads((output / 'status.json').read_text())
        if previous['status'] != 'stopped':
            raise ValueError('Explicit resume only accepts a stopped attempt')
        if json.loads((output / 'identity.json').read_text()) != ident:
            raise ValueError('Resume identity differs; do not mix caches')
        remaining = remaining_budget(previous)
    else:
        if output.exists():
            raise ValueError('Fresh execution requires new output directory')
        output.mkdir(parents=True)
        save(output / 'identity.json', ident)
        previous, remaining = {}, 1800
    lock = output / 'active.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    attempt = len(previous.get('attempts', [])) + 1
    status = {'status': 'running', 'identity_sha256': ident['identity_sha256'], 'record_count': 192,
              'training_started': False, 'eval_accessed': False, 'native_frame_labels_opened': False,
              'label_summary_metadata_opened': True,
              'manifest_sha256': ident['identity']['manifest_sha256'], 'peak_rss_bytes': 0,
              'attempts': previous.get('attempts', []), 'remaining_seconds_at_start': remaining}
    start = time.monotonic()
    prior = 1800 - remaining
    deadline = start + remaining
    child = None
    try:
        save(output / 'status.json', status)
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '2'
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--execute',
                   '--bundle-dir', str(bundle), '--model-dir', str(model), '--output-dir', str(output), '--device', device]
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
        with (output / f'stdout-{attempt}.log').open('xb') as stdout, (output / f'stderr-{attempt}.log').open('xb') as stderr:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, **options)
            status['worker_pid'] = child.pid
            status['exit_code'] = monitor(child, output, status, deadline, start, prior)
        if status['exit_code'] != 0:
            raise RuntimeError('Worker failed; retain logs and explicitly review before resume')
        result = json.loads((output / 'worker-result.json').read_text())
        if result['status'] != 'complete' or result['identity_sha256'] != ident['identity_sha256']:
            raise ValueError('Worker completion provenance differs')
        if result['record_count'] != 192 or any(result.get(k) is not False for k in ('training_started', 'eval_accessed', 'native_frame_labels_opened')):
            raise ValueError('Worker scope changed')
        if result['resources_before_model_load']['system_available_ram_bytes'] < 4 * GIB:
            raise RuntimeError('Immediate pre-load RAM gate failed')
        status['peak_rss_bytes'] = max(status['peak_rss_bytes'], result['resources_before_completion']['peak_rss_bytes'])
        if status['peak_rss_bytes'] > 3 * GIB:
            raise RuntimeError('Final worker lifetime memory exceeds limit')
        checked = validate_all(output, manifest, ident['identity_sha256'])
        _, _, final_identity = identity(bundle, model, device)
        if final_identity != ident:
            raise ValueError('Source/model/environment changed during extraction')
        guard(output)
        if time.monotonic() >= deadline:
            raise RuntimeError('Cumulative deadline reached during final validation')
        save(output / 'validation.json', {'identity_sha256': ident['identity_sha256'], 'records': checked})
        status.update(status='complete', validation_sha256=sha(output / 'validation.json'))
    except BaseException as error:
        status.update(status='stopped', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop(child)
        elapsed = time.monotonic() - start
        status['attempts'].append({'attempt': attempt, 'elapsed_seconds': elapsed, 'status': status['status'],
                                   'finished_utc': datetime.now(timezone.utc).isoformat()})
        status.update(worker_pid=None, cumulative_elapsed_seconds=prior + elapsed)
        save(output / f'attempt-{attempt}-status.json', status)
        if (output / 'worker-result.json').exists():
            save(output / f'attempt-{attempt}-worker-result.json',
                 json.loads((output / 'worker-result.json').read_text()))
        save(output / 'status.json', status)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--bundle-dir', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'record_count': 192, 'training_started': False,
                          'all_extraction_timeout_seconds': 1800, 'device': args.device}))
        return
    if any(v is None for v in (args.bundle_dir, args.model_dir, args.output_dir)):
        parser.error('Execution requires bundle/model/output directories')
    bundle, model, output = [p.resolve() for p in (args.bundle_dir, args.model_dir, args.output_dir)]
    if args.worker:
        worker(bundle, model, output, args.device)
    else:
        supervise(bundle, model, output, args.device, args.resume)


if __name__ == '__main__':
    main()
