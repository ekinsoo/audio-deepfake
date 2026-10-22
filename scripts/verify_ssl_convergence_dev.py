"""No-fit replay and independent weighted ROC check of the fixed dev result."""
import json
from pathlib import Path
import pickle
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_localization_ssl_head import read, sha, save


def verify():
    import numpy as np
    from sklearn.metrics import roc_curve
    from audio_deepfake.localization_ssl_head import score_ssl_head
    pilot = ROOT / 'outputs/localization/ssl-convergence-dev-v1'
    cache = ROOT / 'outputs/colab/extraction-session-v1/result/run'
    diagnostic = ROOT / 'outputs/localization/ssl-convergence-diagnostics-v1'
    result, config, status, scores = [read(pilot / p) for p in ('results.json', 'config.json', 'status.json', 'dev_scores.json')]
    assert status['status'] == result['status'] == 'complete' and status['exit_code'] == 0
    assert status['training_started'] is False and status['monitor_pid'] == status['worker_pid']
    assert status['results_sha256'] == sha(pilot / 'results.json')
    for key, file in [('dev_scores_sha256', 'dev_scores.json'), ('config_sha256', 'config.json'), ('model_sha256', 'model.pkl')]:
        assert result[key] == sha(pilot / file)
    for p, digest in config['source_sha256'].items():
        assert sha(ROOT / p) == digest, p
    assert set(scores) == set(config['selected_ids']['dev']) and len(scores) == 64
    with (pilot / 'model.pkl').open('rb') as f:
        model = pickle.load(f)
    native = read(ROOT / 'data/partialspoof/selected_labels_v1.json')['splits']['dev']
    records = {r['utterance_id']: r for r in read(ROOT / 'docs/validation/localization_alignment.json')['splits']['dev']['records']}
    labels, values, weights = [], [], []
    for uid, row in sorted(scores.items()):
        x = np.load(cache / 'cache/dev' / (uid + '.npy'), allow_pickle=False)
        np.testing.assert_array_equal(score_ssl_head(model, x), row['scores'])
        truth = [min(native[uid][i:i+2]) for i in range(0, len(native[uid]), 2)]
        assert row['labels_20ms'] == truth
        n = records[uid]['sample_count']
        starts = np.arange(len(truth)) * .02
        ends = np.minimum((np.arange(len(truth))+1) * 320, n) / 16000
        np.testing.assert_allclose(row['bin_starts'], starts, rtol=0, atol=1e-14)
        np.testing.assert_allclose(row['bin_ends'], ends, rtol=0, atol=1e-14)
        labels.extend(truth); values.extend(row['scores']); weights.extend(ends-starts)
    labels, values, weights = map(np.asarray, (labels, values, weights))
    fpr, tpr, _ = roc_curve(labels, -values, pos_label=0, sample_weight=weights, drop_intermediate=False)
    i = np.argmin(abs(fpr - (1-tpr)))
    eer = float((fpr[i] + 1-tpr[i]) / 2)
    assert abs(eer - result['custom_duration_eer']['eer_fraction']) < 1e-12
    missed = float(weights[(labels == 0) & (values >= 0)].sum())
    false = float(weights[(labels == 1) & (values < 0)].sum())
    audit = read(diagnostic / 'results.json')
    assert audit['status'] == 'complete_derived_reference_diagnostic'
    for p, digest in audit['source_sha256'].items():
        assert sha(ROOT / p) == digest, p
    for key in ('error_worker', 'range_worker'):
        worker = audit[key]
        assert worker['status'] == 'complete' and worker['exit_code'] == 0
        assert worker['monitor_pid'] == worker['worker_pid'] and worker['peak_rss_bytes'] < 1024**3
    assert abs(audit['ssl']['fixed_zero_pooled20ms']['missed_spoof_seconds'] - missed) < 1e-8
    assert abs(audit['ssl']['fixed_zero_pooled20ms']['false_alarm_seconds'] - false) < 1e-8
    timeline_hashes = {}
    assert sorted(p.stem for p in (pilot / 'timelines').glob('*.svg')) == sorted(config['timeline_ids'])
    for p in (pilot / 'timelines').glob('*.svg'):
        xml = ET.parse(p).getroot()
        expected = []
        row = scores[p.stem]
        for value, start, end in zip(row['scores'], row['bin_starts'], row['bin_ends']):
            if value < 0:
                if expected and abs(expected[-1][1] - start) < 1e-9:
                    expected[-1][1] = end
                else:
                    expected.append([start, end])
        actual = [[float(e.attrib['data-start']), float(e.attrib['data-end'])] for e in xml.iter() if e.attrib.get('data-track') == 'prediction']
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        timeline_hashes[p.relative_to(ROOT).as_posix()] = sha(p)
    return {'status': 'verified_fixed_dev_followup', 'protocol_id': config['protocol_id'],
        'training_started': False, 'eval_enabled': False, 'official_benchmark': False,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), pilot / 'results.json', pilot / 'status.json', diagnostic / 'results.json']},
        'independent_checks': ['All64 saved scores reproduced exactly from verified model/caches without fit',
            'Native min-pooled labels and clipped audio-duration grids recomputed',
            'Weighted sklearn ROC independently matches custom duration EER',
            'Fixed-zero error durations recomputed', 'All10 timeline prediction spans checked against saved scores',
            'Diagnostic source pins and actual worker PID/resource receipts checked'],
        'ssl': audit['ssl'], 'frozen_lfcc': audit['frozen_lfcc'], 'timeline_sha256': timeline_hashes,
        'score_elapsed_seconds': status['elapsed_seconds'], 'score_peak_rss_bytes': status['peak_rss_bytes'],
        'comparison_caveat': 'Separate max3000 followup; original SSL300 failed; LFCC uses max300. Repeatedly inspected custom64dev, whole-recording context, unknown pretraining overlap.'}


if __name__ == '__main__':
    if sys.argv[1:] == ['--execute']:
        audit = verify()
        save(ROOT / 'docs/validation/localization_ssl_convergence_dev_v1.json', audit)
        print(json.dumps({k: audit[k] for k in ('status', 'ssl')}, indent=2))
    else:
        print(json.dumps({'status': 'plan_only', 'training_started': False}))
