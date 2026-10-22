"""Bounded patched upstream RangeEER on explicitly derived native-bin references.

This is a custom pilot diagnostic, not a publisher-reference benchmark.
Dependencies remain in the separate metric environment. No training/network.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'.cache/partial-metric-env/upstream'
REFERENCES = ROOT/'data/partialspoof/selected_references_v1/dev'
REF_AUDIT = ROOT/'docs/validation/localization_references_v1.json'
UPSTREAM_SHA = 'f2736d6bef242970e0f73849a94eace6aed17bf3a4b65e587266b68814513677'
PILOTS = {'baseline': 'outputs/jobs/localization-pilot-20261010T163921940Z/pilot',
          'context': 'outputs/jobs/localization-context-20261010-v2/pilot'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_rates(scores, native, durations, threshold):
    """Independent duration arithmetic; native 10 ms truth, upstream <= tie rule."""
    positive = negative = missed = false_alarm = 0.
    for uid, row in scores.items():
        duration = durations[uid]
        for i, label in enumerate(native[uid]):
            width = min((i+1)*.01, duration)-i*.01
            predicted = row['scores'][i//2] <= threshold
            if label == 0:
                positive += width
                if not predicted:
                    missed += width
            else:
                negative += width
                if predicted:
                    false_alarm += width
    if min(positive, negative) <= 0:
        raise ValueError('Both duration classes required')
    return {'fpr': false_alarm/negative, 'fnr': missed/positive,
            'spoof_seconds': positive, 'bonafide_seconds': negative,
            'missed_spoof_seconds': missed, 'false_alarm_seconds': false_alarm}


def configure_metric():
    import site
    packages = ROOT/'.cache/partial-metric-env/venv/Lib/site-packages'
    sys.path.insert(0, str(packages))
    site.addsitedir(str(packages))
    sys.path.insert(0, str(CACHE))


def metric_worker(variant, output):
    configure_metric()
    import numpy as np
    if np.__version__ != '1.26.4':
        raise RuntimeError('Wrong isolated metric environment')
    from importlib.metadata import version
    (output/'environment.json').write_text(json.dumps({p: version(p) for p in (
        'numpy', 'pandas', 'scipy', 'pyannote.metrics', 'pyannote.core', 'pyannote.database', 'python-rle')}, indent=2))
    scores = json.loads((ROOT/PILOTS[variant]/'dev_scores.json').read_text())
    groups = np.empty(len(scores), dtype=object)
    for i, (uid, row) in enumerate(sorted(scores.items())):
        groups[i] = np.asarray([[uid, -score, score, label] for score, label in
                               zip(row['scores'], row['labels_20ms'])], dtype=object)
    np.save(output/'trusted_scores.npy', groups)
    sys.argv = ['RangeEER.bounded.py', '--hyp_sco_ali', str(output/'trusted_scores.npy'),
                '--ref_rttm', str(REFERENCES/'reference.rttm'),
                '--reco2dur_file', str(REFERENCES/'reco2dur'), '--scale', '2',
                '--save_dir', str(output/'curve')]
    runpy.run_path(str(CACHE/'RangeEER.bounded.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', choices=PILOTS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        metric_worker(args.worker, args.output)
        return
    if not args.execute:
        print('Plan only: two isolated 120-second metric workers, no training or download.')
        return
    sys.path.insert(0, str(ROOT/'scripts'))
    from full_run_job import base_python, windows_memory, stop_owned_child
    from localization_pilot_job import check_resources
    output = ROOT/'outputs/localization/range-derived-v1'
    audit_path = ROOT/'docs/validation/localization_range_results_v1.json'
    if output.exists() or audit_path.exists():
        raise ValueError('Preserve previous metric attempts; use a separately versioned recovery')
    refs = json.loads(REF_AUDIT.read_text())
    if refs['official_native_reference_gate_passed'] is not False:
        raise ValueError('This adapter explicitly expects derived references')
    if sha(CACHE/'RangeEER.bounded.py') != UPSTREAM_SHA:
        raise ValueError('Bounded evaluator pin mismatch')
    for path, expected in {
        'sandbox/eval_asvspoof.py': 'a7441cac67be01b1cd3322ec4319b850d761803bfd1f4b693fbb1e823f9944ca',
        'core_scripts/data_io/io_tools.py': '56ec6fca200a5d1a03a57f0eae16a71f0d71e36f8a052edfb82ae65eaa70fb3c'}.items():
        if sha(CACHE/path) != expected:
            raise ValueError('Pinned upstream dependency changed')
    for filename, key in [('reference.rttm', 'reference_sha256'), ('reco2dur', 'reco2dur_sha256')]:
        if sha(REFERENCES/filename) != refs['splits']['dev'][key]:
            raise ValueError('Derived reference changed')
    native = json.loads((ROOT/'data/partialspoof/selected_labels_v1.json').read_text())['splits']['dev']
    durations = {r['utterance_id']: r['duration_seconds'] for r in refs['splits']['dev']['records']}
    pins = {p: sha(ROOT/p) for p in ['scripts/evaluate_localization_ranges.py',
        'scripts/prepare_partial_references.py', 'scripts/full_run_job.py', 'scripts/localization_pilot_job.py',
        'docs/validation/localization_references_v1.json', 'data/partialspoof/selected_labels_v1.json',
        '.cache/partial-metric-env/upstream/RangeEER.bounded.py',
        '.cache/partial-metric-env/upstream/sandbox/eval_asvspoof.py',
        '.cache/partial-metric-env/upstream/core_scripts/data_io/io_tools.py']}
    output.mkdir(parents=True)
    audit = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Custom 64dev pilot: bounded patched upstream RangeEER, derived native10ms reference',
             'official_benchmark': False, 'publisher_reference_gate_passed': False,
             'prediction_step_seconds': .02, 'reference_step_seconds': .01,
             'tie_rule': 'upstream score <= threshold predicts spoof; project timelines use <',
             'source_sha256': pins, 'variants': {}, 'eval_enabled': False}
    child = None
    try:
        for name, pilot in PILOTS.items():
            directory = output/name
            directory.mkdir()
            result = json.loads((ROOT/pilot/'results.json').read_text())
            scores = json.loads((ROOT/pilot/'dev_scores.json').read_text())
            if result['status'] != 'complete' or sha(ROOT/pilot/'dev_scores.json') != result['dev_scores_sha256']:
                raise ValueError('Completed hash-matching dev predictions required')
            if set(scores) != set(durations) or len(scores) != 64:
                raise ValueError('Exact fixed 64 dev IDs required')
            if any(uid.startswith('CON') and not any(v == 0 for v in native[uid]) for uid in scores):
                raise ValueError('Upstream would skip a CON ID absent from spoof RTTM; review adapter')
            for uid, row in scores.items():
                if len(row['scores']) != (len(native[uid])+1)//2 or row['labels_20ms'] != [min(native[uid][i:i+2]) for i in range(0,len(native[uid]),2)]:
                    raise ValueError('Score/reference grid mismatch')
            check_resources(windows_memory(), shutil.disk_usage(ROOT).free)
            started = time.monotonic()
            peak = 0
            with (directory/'stdout.log').open('wb') as stdout, (directory/'stderr.log').open('wb') as stderr:
                child = subprocess.Popen([base_python(), '-S', '-u', str(Path(__file__).resolve()),
                    '--worker', name, '--output', str(directory)], cwd=ROOT, stdout=stdout, stderr=stderr,
                    stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                while child.poll() is None:
                    if time.monotonic()-started > 120:
                        raise RuntimeError('RangeEER worker exceeded 120 seconds')
                    try:
                        memory = windows_memory(child.pid, process_handle=child._handle)
                    except OSError:
                        if child.poll() is not None:
                            break
                        raise
                    peak = max(peak, memory['child_peak_rss_bytes'])
                    check_resources(memory, shutil.disk_usage(ROOT).free)
                    time.sleep(.5)
                if child.wait() != 0:
                    raise RuntimeError(f'{name} metric worker failed; see retained stderr')
            text = (directory/'stdout.log').read_text()
            row = json.loads(re.search(r'BoundedSearch selected_row=(\[[^\n]+\])', text).group(1))
            independent = direct_rates(scores, native, durations, row[0])
            if max(abs(row[1]-independent['fpr']), abs(row[2]-independent['fnr'])) > 1e-8:
                raise ValueError('Upstream rates disagree with independent native interval counts')
            audit['variants'][name] = {'range_eer_fraction': (row[1]+row[2])/2, 'selected_row': row,
                'independent_duration_check': independent, 'upstream_iterations': int(re.search(r'iterations=(\d+)', text).group(1)),
                'stop_reason': (re.search(r'BoundedSearch stop=([^\n]+)', text).group(1) if 'BoundedSearch stop=' in text else 'precision_or_bound'),
                'score_sha256': result['dev_scores_sha256'], 'worker_elapsed_seconds': time.monotonic()-started,
                'sampled_peak_rss_bytes': peak, 'stdout_sha256': sha(directory/'stdout.log'),
                'stderr_sha256': sha(directory/'stderr.log'),
                'packages': json.loads((directory/'environment.json').read_text())}
        if any(sha(ROOT/p) != value for p, value in pins.items()):
            raise ValueError('Metric input/source changed during evaluation')
        audit['status'] = 'complete_derived_reference_diagnostic'
    except BaseException as error:
        audit.update(status='stopped', error=str(error))
        raise
    finally:
        if child is not None:
            stop_owned_child(child)
        audit_path.write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({name: v['range_eer_fraction'] for name,v in audit['variants'].items()}))


if __name__ == '__main__':
    main()
