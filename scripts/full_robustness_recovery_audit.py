"""Resume the unchanged narrowband experiment with explicit failed-attempt accounting."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import finalize_full_robustness as finalizer
import full_robustness_worker as worker
import run_full_baseline as full
from full_multiseed_job import public_stages
from recover_full_scoring import validate_scores

ROOT = full.ROOT
RECOVERY_SOURCES = ('scripts/full_robustness_recovery_audit.py',
                    'scripts/full_robustness_recovery_job.py',
                    'scripts/start_full_robustness_recovery.ps1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def smoke_checks(output, smoke, preflight, contract, groups, calibration):
    if (smoke.get('status') != 'ok' or smoke.get('protocol') != contract['protocol']
            or smoke.get('no_metric_tuning') is not True
            or [item.get('seed') for item in smoke.get('seeds', [])] != [42, 43, 44]):
        raise ValueError('Original smoke population changed')
    seconds = {}
    for item in smoke['seeds']:
        seed = str(item['seed'])
        expected = dict(status='ok', contract_sha256=preflight['contract_sha256'],
                        execution_sha256=preflight['execution_sha256'],
                        preflight_sha256=full.digest(output / 'preflight.json'), start=0, stop=20)
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError('Original smoke pins changed')
        clean = dict(validate_scores(ROOT / contract['seeds'][seed]['clean_scores_path'], groups['dev']))
        finalizer.validate_paired_rows(item['rows'], groups['dev'][:20], clean,
                                       {row.utterance_id for row in calibration})
        if len(item.get('clean_reproduction', [])) != 20:
            raise ValueError('Incomplete smoke reproduction')
        for check, row in zip(item['clean_reproduction'], groups['dev'][:20]):
            value = check.get('reproduced_score', float('nan'))
            if (check.get('utterance_id') != row.utterance_id
                    or check.get('saved_score') != clean[row.utterance_id]
                    or not math.isfinite(value) or abs(value-clean[row.utterance_id]) > 1e-10
                    or check.get('absolute_difference') != abs(value-clean[row.utterance_id])):
                raise ValueError('Original smoke reproduction changed')
        value = item.get('processing_seconds')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid smoke time')
        seconds[seed] = value
    return seconds


def validate_resume(output, original_job_dir, *, allow_new_chunks=False):
    output, original_job_dir = Path(output).resolve(), Path(original_job_dir).resolve()
    if (ROOT / 'outputs/jobs/full-robustness.lock').exists():
        # The recovery supervisor owns a different lock; the original must be absent.
        raise ValueError('Original job lock still exists')
    preflight, contract, _, groups, calibration, _ = worker.frozen_inputs(output)
    status = read(original_job_dir / 'status.json')
    if (status.get('status') != 'stopped' or status.get('exit_code') != 1
            or status.get('reason') != 'RuntimeError: child_exit_1'
            or status.get('source_sha256') != preflight['execution_source_sha256']):
        raise ValueError('Unexpected original stop or source pins')
    stages = status.get('stages', [])
    names = ['preflight', 'smoke'] + [f'seed{seed}_{start:05d}_{min(start+2000,24844):05d}'
                for seed in (42, 43) for start in range(0, 24844, 2000)]
    if ([item.get('name') for item in stages[:-1]] != names or not stages
            or stages[-1].get('name') != 'seed44_00000_02000'
            or stages[-1].get('status') != 'stopped' or stages[-1].get('exit_code') != 1
            or stages[-1].get('reason') != 'child_exit_1'):
        raise ValueError('Unexpected original stage population')
    stderr = original_job_dir / 'seed44_00000_02000.stderr.log'
    if 'Resource stop: insufficient available system RAM' not in stderr.read_text(encoding='utf-8'):
        raise ValueError('Original failure is not the verified RAM stop')
    smoke_seconds = smoke_checks(output, read(output / 'smoke.json'), preflight, contract, groups, calibration)
    chunks = {'44': []}
    for seed in (42, 43):
        clean = dict(validate_scores(ROOT / contract['seeds'][str(seed)]['clean_scores_path'], groups['dev']))
        _, chunks[str(seed)] = finalizer.collect_chunks(output, seed, groups['dev'], clean,
                         {row.utterance_id for row in calibration}, preflight, contract)
    expected_files = {item['file'] for seed in ('42', '43') for item in chunks[seed]}
    for path in (output / 'chunks').glob('*.json'):
        relative = path.relative_to(output).as_posix()
        if relative not in expected_files and (not allow_new_chunks or not path.name.startswith('seed44_')):
            raise ValueError('Unexpected completed chunk before resume')
    successful = public_stages(stages[:-1])
    success_wall = {str(seed): sum(item['elapsed_seconds'] for item in successful
                      if item['name'].startswith(f'seed{seed}_')) + smoke_seconds[str(seed)] for seed in (42,43,44)}
    ORIGINAL_SUPERVISION(dict(status='scoring_complete', source_sha256=preflight['execution_source_sha256'],
                  stages=successful, seed_smoke_seconds=smoke_seconds, seed_wall_seconds=success_wall),
                  preflight, chunks, smoke_seconds)
    failed = public_stages(stages[-1:])[0]
    elapsed = failed.get('elapsed_seconds')
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int,float))
            or not math.isfinite(elapsed) or not 0 <= elapsed <= failed['budget_seconds'] <= 1800):
        raise ValueError('Invalid failed-attempt processing time')
    failed.update(reason='child_exit_1', resource_failure='insufficient available system RAM',
                  stderr_sha256=full.digest(stderr))
    spent = dict(success_wall)
    spent['44'] += elapsed
    return dict(status='ok', original_job=original_job_dir.relative_to(ROOT).as_posix(),
                original_status_sha256=full.digest(original_job_dir / 'status.json'),
                success_stages=successful, failed_stages=[failed], seed_spent_seconds=spent,
                seed_smoke_seconds=smoke_seconds,
                chunk_sha256={item['file']: item['sha256'] for seed in ('42','43') for item in chunks[seed]})


def validate_recovery_execution(output, original_job_dir, *, allow_new_chunks=True):
    resume = validate_resume(output, original_job_dir, allow_new_chunks=allow_new_chunks)
    execution = read(output / 'recovery_execution.json')
    expected_sources = {name: full.digest(ROOT / name) for name in RECOVERY_SOURCES}
    if execution.get('status') != 'frozen' or execution.get('source_sha256') != expected_sources:
        raise ValueError('Recovery execution sources changed')
    for key in ('original_job', 'original_status_sha256', 'chunk_sha256'):
        if execution.get(key) != resume[key]:
            raise ValueError('Recovery original evidence changed: ' + key)
    for key, expected in dict(preflight_sha256=full.digest(output / 'preflight.json'),
                              smoke_sha256=full.digest(output / 'smoke.json'),
                              failure_history=resume['failed_stages']).items():
        if execution.get(key) != expected:
            raise ValueError('Recovery preflight, smoke or failure evidence changed: ' + key)
    return resume, execution


def verify(output, original_job_dir):
    started = time.perf_counter()
    resume, execution = validate_recovery_execution(output, original_job_dir, allow_new_chunks=False)
    if (output / 'recovery_preflight.json').exists():
        raise ValueError('Preserve existing recovery preflight')
    _, contract, config, _, _, _ = worker.frozen_inputs(output)
    checked = full.preflight(config, output, owned_output=True)
    if checked['input_sha256'] != contract['official_input_sha256']:
        raise ValueError('Recovery full audio verification changed')
    result = dict(status='ok', audio_records_verified=50224, input_sha256=checked['input_sha256'],
                  recovery_execution_sha256=full.digest(output / 'recovery_execution.json'),
                  original_status_sha256=resume['original_status_sha256'],
                  source_sha256=execution['source_sha256'], seconds=time.perf_counter()-started,
                  initial_memory=checked['initial_memory'], final_memory=checked['guard'].check())
    full.atomic_json(output / 'recovery_preflight.json', result)
    return result


def recovery_supervision(supervision, preflight, chunks, smoke_seconds, output, original_job_dir):
    resume, execution = validate_recovery_execution(output, original_job_dir)
    checked = read(output / 'recovery_preflight.json')
    if (checked.get('status') != 'ok' or checked.get('audio_records_verified') != 50224
            or checked.get('input_sha256') != preflight['input_sha256']
            or checked.get('recovery_execution_sha256') != full.digest(output / 'recovery_execution.json')
            or checked.get('original_status_sha256') != resume['original_status_sha256']
            or checked.get('source_sha256') != execution['source_sha256']):
        raise ValueError('Recovery preflight pins changed')
    expected = dict(recovered=True, recovery_execution_sha256=full.digest(output / 'recovery_execution.json'),
                    recovery_preflight_sha256=full.digest(output / 'recovery_preflight.json'),
                    failure_history=resume['failed_stages'], original_job=resume['original_job'],
                    original_status_sha256=resume['original_status_sha256'])
    if any(supervision.get(key) != value for key,value in expected.items()):
        raise ValueError('Recovery supervision evidence changed')
    original_count = len(resume['success_stages'])
    if supervision.get('stages', [])[:original_count] != resume['success_stages']:
        raise ValueError('Original successful supervision changed')
    successful_wall = {str(seed): sum(item['elapsed_seconds'] for item in supervision['stages']
                          if item['name'].startswith(f'seed{seed}_')) + smoke_seconds[str(seed)] for seed in (42,43,44)}
    success_only = {**supervision, 'seed_wall_seconds': successful_wall}
    ORIGINAL_SUPERVISION(success_only, preflight, chunks, smoke_seconds)
    for seed in (42,43,44):
        expected_time = successful_wall[str(seed)] + sum(item['elapsed_seconds'] for item in resume['failed_stages']
                                                       if item['name'].startswith(f'seed{seed}_'))
        actual = supervision['seed_wall_seconds'][str(seed)]
        if (isinstance(actual, bool) or not isinstance(actual, (int,float)) or not math.isfinite(actual)
                or not 0 <= actual <= 1800 or not math.isclose(actual, expected_time, rel_tol=0, abs_tol=1e-9)):
            raise ValueError('Recovery failed-attempt budget accounting differs')
    recovery_stages = supervision.get('recovery_stages', [])
    if (len(recovery_stages) != 1 or recovery_stages[0].get('name') != 'recovery_preflight'
            or recovery_stages[0].get('status') != 'complete' or recovery_stages[0].get('exit_code') != 0):
        raise ValueError('Missing successful recovery preflight supervision')
    stage = recovery_stages[0]
    elapsed, budget, peak = (stage.get(key) for key in ('elapsed_seconds','budget_seconds','peak_rss_bytes'))
    if (any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
            for value in (elapsed,budget,peak)) or not 0 <= elapsed <= budget <= 1800
            or not 0 <= peak <= 2*1024**3
            or stage.get('last_memory',{}).get('system_available_ram_bytes',0) < 1024**3):
        raise ValueError('Recovery preflight resource supervision differs')


ORIGINAL_SUPERVISION = finalizer.validate_supervision


def finalize(output, original_job_dir):
    validate_recovery_execution(output, original_job_dir)
    original = finalizer.validate_supervision
    finalizer.validate_supervision = lambda supervision, preflight, chunks, smoke_seconds: recovery_supervision(
                    supervision, preflight, chunks, smoke_seconds, output, original_job_dir)
    try:
        result = finalizer.finalize(output)
        validate_recovery_execution(output, original_job_dir)
        return result
    finally:
        finalizer.validate_supervision = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('verify','finalize'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--original-job-dir', type=Path, required=True)
    args = parser.parse_args()
    operation = verify if args.mode == 'verify' else finalize
    result = operation(args.output_dir.resolve(), args.original_job_dir.resolve())
    print(json.dumps({'status':result['status']}, indent=2))


if __name__ == '__main__':
    main()
