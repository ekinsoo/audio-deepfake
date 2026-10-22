"""Verified SSL follow-up pages, separate from the frozen failed comparison."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ('CON_D_0000586', 'CON_D_0003090')


def read(name):
    return json.loads((ROOT / 'docs/validation' / name).read_text())


def timelines():
    audit = read('localization_ssl_convergence_dev_v1.json')
    alignment = read('localization_alignment.json')
    records = {r['utterance_id']: r for r in alignment['splits']['dev']['records']}
    rows = []
    for uid in EXAMPLES:
        tracks = {}
        for name, folder, expected in (
            ('SSL follow-up', 'ssl_convergence_v1', audit['timeline_sha256'][f'outputs/localization/ssl-convergence-dev-v1/timelines/{uid}.svg']),
            ('LFCC baseline', 'localization_pilot_v1', read('localization_pilot_review_v1.json')['output_sha256'][f'outputs/jobs/localization-pilot-20261010T163921940Z/pilot/timelines/{uid}.svg'])):
            path = ROOT / 'docs/figures' / folder / (uid + '.svg')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            xml = ET.parse(path).getroot()
            tracks[name] = [(float(e.attrib['data-start']), float(e.attrib['data-end'])) for e in xml.iter() if e.attrib.get('data-track') == 'prediction']
            reference = [(float(e.attrib['data-start']), float(e.attrib['data-end'])) for e in xml.iter() if e.attrib.get('data-track') == 'reference']
            if 'Native binary' in tracks:
                assert tracks['Native binary'] == reference
            tracks['Native binary'] = reference
        rows.append((uid, records[uid]['sample_count'] / 16000, tracks))
    return rows


def drawing(_):
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.lib import colors
    d = Drawing(720, 320)
    d.add(Rect(0, 0, 720, 320, fillColor=colors.white, strokeColor=None))
    for index, (uid, duration, tracks) in enumerate(timelines()):
        top = 300 - index * 155
        d.add(String(12, top, uid, fontName='Helvetica-Bold', fontSize=12))
        for j, name in enumerate(('Native binary', 'LFCC baseline', 'SSL follow-up')):
            y = top - 30 - j * 25
            d.add(String(12, y + 3, name, fontName='Helvetica', fontSize=10))
            d.add(Rect(125, y, 565, 14, fillColor=colors.HexColor('#EEEEEE'), strokeColor=None))
            for start, end in tracks[name]:
                d.add(Rect(125 + 565 * start / duration, y, 565 * (end-start) / duration, 14,
                           fillColor=colors.HexColor('#555555'), strokeColor=None))
        y = top - 90
        d.add(Line(125, y, 690, y, strokeColor=colors.HexColor('#777777'), strokeWidth=.6))
        for fraction in (0, .25, .5, .75, 1):
            d.add(String(125 + 565 * fraction, y-13, f'{duration*fraction:.2f}', fontName='Helvetica', fontSize=9, textAnchor='middle'))
    return d


def tex(_):
    from build_project_report import tex_escape
    lines = [r'\begin{center}\setlength{\unitlength}{1pt}\begin{picture}(480,220)']
    for index, (uid, duration, tracks) in enumerate(timelines()):
        top = 208 - index * 108
        lines.append(f'\\put(0,{top}){{\\makebox(0,0)[l]{{\\small\\bfseries {tex_escape(uid)}}}}}')
        for j, name in enumerate(('Native binary', 'LFCC baseline', 'SSL follow-up')):
            y = top - 23 - j * 18
            lines.append(f'\\put(0,{y+3}){{\\makebox(0,0)[l]{{\\scriptsize {name}}}}}')
            lines.append(f'\\put(83,{y}){{\\color{{lightgrid}}\\rule{{377pt}}{{10pt}}}}')
            for start, end in tracks[name]:
                lines.append(f'\\put({83+377*start/duration:.4f},{y}){{\\color[HTML]{{555555}}\\rule{{{377*(end-start)/duration:.4f}pt}}{{10pt}}}}')
        for fraction in (0, .25, .5, .75, 1):
            lines.append(f'\\put({83+377*fraction:.4f},{top-76}){{\\makebox(0,0){{\\scriptsize {duration*fraction:.2f}}}}}')
    return '\n'.join(lines + [r'\end{picture}\end{center}'])


def report_pages():
    dev = read('localization_ssl_convergence_dev_v1.json')
    train = read('localization_ssl_convergence_train_v1.json')
    assert dev['status'] == 'verified_fixed_dev_followup' and train['status'] == 'verified_train_only'
    ssl, lfcc = dev['ssl'], dev['frozen_lfcc']
    rows = [['Same custom 64-dev diagnostic', 'LFCC max 300', 'SSL max 3000 follow-up']]
    for key, label in [('custom_duration_eer_fraction', 'Pooled 20 ms duration EER'), ('derived_reference_range_eer_fraction', 'Derived native RangeEER')]:
        rows.append([label, f'{lfcc[key]*100:.2f}%', f'{ssl[key]*100:.2f}%'])
    for key, label in [('missed_spoof_seconds', 'Missed spoof at zero'), ('false_alarm_seconds', 'False alarm at zero')]:
        rows.append([label, f'{lfcc["fixed_zero_pooled20ms"][key]:.2f} s', f'{ssl["fixed_zero_pooled20ms"][key]:.2f} s'])
    rows.append(['Predicted spoof runs', str(lfcc['fragmentation20ms']['predicted_runs']), str(ssl['fragmentation20ms']['predicted_runs'])])
    return [
        ('A frozen encoder, with a separate convergence follow-up', [
            ('p', 'I extracted pinned wav2vec2-base features [13] for the same 128 train and 64 dev recordings. Its 768-dimensional final states were aligned to the existing 20 ms output bins. All 192 caches, source/checkpoint hashes and resource receipts passed verification. T4 extraction took 18.27 seconds, with 1.394 GiB externally measured peak RAM. No encoder fine-tuning was performed.'),
            ('h', 'Preserving the failed experiment'),
            ('p', 'The original SSL head used the same maximum 300 iterations as LFCC. It emitted a convergence warning and produced no accepted model or dev result. Its Windows monitor measured a launcher instead of the numerical child, so its reported RAM is invalid. I preserved that failure, repaired process supervision and tested the actual worker with a 64 MiB allocation.'),
            ('p', 'Before any SSL dev scores, I recorded a separate train-only maximum of 3,000 iterations. C=1, lbfgs, tolerance 0.0001, seed 45, one thread, features and ordered training samples stayed fixed. The new fit converged in 303 iterations. External duration was 7.67 seconds and peak RAM 316.9 MiB. An independent no-fit replay checked retained feature bytes, scaler statistics, parameters and real worker PID before dev scoring.'),
            ('table', rows),
            ('p', 'The custom duration EER decreased by 10.93 percentage points. At the fixed zero boundary, false-alarm duration decreased but missed spoof increased. This is a useful error trade-off, not evidence that every operating point improved. The SSL follow-up has a different optimization budget from LFCC 300; it is not an equal-training-budget comparison.'),
            ('small', 'Evidence: docs/validation/localization_ssl_convergence_train_v1.json and localization_ssl_convergence_dev_v1.json. All 64 saved scores were reproduced exactly without fitting; a separate weighted ROC calculation reproduced EER. The original fixed 300 failure remains in localization_ssl_head_convergence_v2.json.')]),
        ('SSL timelines and what the result cannot establish', [
            ('figure', 'ssl_timeline'),
            ('caption', 'The same two examples selected before scoring, with native reference, LFCC and SSL predictions. Dark spans indicate spoof. Both models use the fixed boundary zero; there is no smoothing or threshold selection for these tracks. Time is in seconds.'),
            ('p', 'SSL still produces 720 predicted runs for 109 pooled-reference runs; 423 predictions have no reference overlap. Thirteen reference runs have no prediction overlap, and 46 are split across multiple predictions. These examples and aggregate counts show remaining localization errors despite the lower EER.'),
            ('h', 'Development evidence, not a benchmark claim'),
            ('p', 'The same small dev subset has been inspected repeatedly. There is one model seed, no independent threshold test, and correlated frames within recordings. The whole-recording encoder uses future audio; a 20 ms score grid does not imply streaming latency. Encoder pretraining overlap is unknown, and PartialSpoof shares source material with ASVspoof.'),
            ('p', 'The range diagnostic still uses derived native 10 ms reference ranges and the same bounded patched upstream evaluator. Publisher RTTM parity is unverified. Its score <= threshold rule differs from the displayed score < 0 rule. Evaluation data remains closed. No further solver, seed or threshold search was performed.'),
            ('small', 'Next validation should use a separately fixed protocol and adequate independent data before making generalization or deployment claims. Public data only; no company data or internal results were used.')])]
