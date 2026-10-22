"""Public-audit localization pages and plain native vector timelines."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("CON_D_0000586", "CON_D_0003090")


def read(name):
    return json.loads((ROOT / "docs/validation" / name).read_text(encoding="utf-8"))


def timelines():
    import hashlib
    alignment = read("localization_alignment.json")
    review = read('localization_pilot_review_v1.json')
    records = {r["utterance_id"]: r for r in alignment["splits"]["dev"]["records"]}
    rows = []
    for uid in EXAMPLES:
        source = ROOT / "docs/figures/localization_pilot_v1" / f"{uid}.svg"
        key = f'outputs/jobs/localization-pilot-20261010T163921940Z/pilot/timelines/{uid}.svg'
        if hashlib.sha256(source.read_bytes()).hexdigest() != review['output_sha256'][key]:
            raise ValueError('Report timeline differs from the independently reviewed SVG')
        xml = ET.parse(source).getroot()
        spans = {name: [] for name in ("reference", "prediction")}
        for element in xml.iter():
            track = element.attrib.get("data-track")
            if track in spans:
                spans[track].append((float(element.attrib["data-start"]), float(element.attrib["data-end"])))
        rows.append((uid, records[uid]["sample_count"] / 16000, spans))
    return rows


def timeline_drawing(_):
    drawing = Drawing(720, 285)
    drawing.add(Rect(0, 0, 720, 285, fillColor=colors.white, strokeColor=None))
    for index, (uid, duration, spans) in enumerate(timelines()):
        top = 260 - index * 132
        drawing.add(String(18, top, uid, fontName="Helvetica-Bold", fontSize=12))
        for label, track, y in (("Native binary", "reference", top-28), ("Baseline", "prediction", top-55)):
            drawing.add(String(18, y+3, label, fontName="Helvetica", fontSize=10))
            drawing.add(Rect(125, y, 565, 15, fillColor=colors.HexColor("#EEEEEE"), strokeColor=None))
            for start, end in spans[track]:
                drawing.add(Rect(125+565*start/duration, y, 565*(end-start)/duration, 15,
                    fillColor=colors.HexColor("#555555"), strokeColor=None))
        axis = top-70
        drawing.add(Line(125, axis, 690, axis, strokeColor=colors.HexColor("#777777"), strokeWidth=.6))
        for fraction in (0, .25, .5, .75, 1):
            x = 125+565*fraction
            drawing.add(String(x, axis-14, f"{duration*fraction:.2f}", fontName="Helvetica", fontSize=9, textAnchor="middle"))
    drawing.add(String(18, 5, "Time in seconds. Dark spans indicate binary spoof labels or predictions; threshold fixed at zero.",
                       fontName="Helvetica", fontSize=9))
    return drawing


def timeline_tex(_):
    from build_project_report import tex_escape
    lines = [r"\begin{center}\setlength{\unitlength}{1pt}\begin{picture}(480,190)"]
    for index, (uid, duration, spans) in enumerate(timelines()):
        top = 178-index*88
        lines.append(f"\\put(0,{top}){{\\makebox(0,0)[l]{{\\small\\bfseries {tex_escape(uid)}}}}}")
        for label, track, y in (("Native binary", "reference", top-20), ("Baseline", "prediction", top-38)):
            lines.append(f"\\put(0,{y+3}){{\\makebox(0,0)[l]{{\\scriptsize {label}}}}}")
            lines.append(f"\\put(83,{y}){{\\color{{lightgrid}}\\rule{{377pt}}{{10pt}}}}")
            for start, end in spans[track]:
                x, width = 83+377*start/duration, 377*(end-start)/duration
                lines.append(f"\\put({x:.4f},{y}){{\\color[HTML]{{555555}}\\rule{{{width:.4f}pt}}{{10pt}}}}")
        for fraction in (0, .25, .5, .75, 1):
            lines.append(f"\\put({83+377*fraction:.3f},{top-55}){{\\makebox(0,0){{\\scriptsize {duration*fraction:.2f}}}}}")
    lines.append(r"\end{picture}\end{center}")
    return "\n".join(lines)


def context_values():
    context = read('localization_context_v1.json')
    if read('localization_context_review_v1.json')['status'] != 'passed':
        raise ValueError('Independent context review required')
    return [read('localization_pilot_v1.json')['results']['custom_duration_eer']['eer_fraction']*100,
            context['results']['custom_duration_eer']['eer_fraction']*100]


def context_drawing(_):
    drawing = Drawing(720, 235)
    drawing.add(Rect(0,0,720,235,fillColor=colors.white,strokeColor=None))
    drawing.add(String(25,215,'One fixed context experiment',fontName='Helvetica-Bold',fontSize=14))
    drawing.add(String(25,196,'Custom 20 ms duration EER (%) | same 64 dev recordings | lower is better',fontSize=10))
    for tick in (0,10,20,30,40,50):
        y=42+tick*2.6
        drawing.add(Line(80,y,670,y,strokeColor=colors.HexColor('#EEEEEE')))
        drawing.add(String(70,y-3,str(tick),fontSize=9,textAnchor='end'))
    for x,label,value in zip((245,510),('LFCC baseline','Recording median/MAD'),context_values()):
        drawing.add(Rect(x-35,42,70,value*2.6,fillColor=colors.HexColor('#555555'),strokeColor=None))
        drawing.add(String(x,52+value*2.6,f'{value:.2f}',fontName='Helvetica-Bold',fontSize=11,textAnchor='middle'))
        drawing.add(String(x,22,label,fontSize=10,textAnchor='middle'))
    return drawing


def context_tex(_):
    lines=[r'\begin{center}\setlength{\unitlength}{1pt}\begin{picture}(480,160)']
    for tick in (0,10,20,30,40,50):
        y=25+tick*2.1
        lines.append(f'\\put(50,{y:.3f}){{\\color{{lightgrid}}\\line(1,0){{400}}}}')
        lines.append(f'\\put(43,{y:.3f}){{\\makebox(0,0)[r]{{\\scriptsize {tick}}}}}')
    for x,label,value in zip((160,340),('LFCC baseline','Recording median/MAD'),context_values()):
        lines.append(f'\\put({x-23},25){{\\color[HTML]{{555555}}\\rule{{46pt}}{{{value*2.1:.4f}pt}}}}')
        lines.append(f'\\put({x},{35+value*2.1:.4f}){{\\makebox(0,0){{\\small {value:.2f}}}}}')
        lines.append(f'\\put({x},10){{\\makebox(0,0){{\\scriptsize {label}}}}}')
    lines.append(r'\end{picture}\end{center}')
    return '\n'.join(lines)


def report_pages():
    audit = read("localization_pilot_v1.json")
    result = audit["results"]
    review = read("localization_pilot_review_v1.json")
    if result["status"] != "complete" or review["status"] != "passed":
        raise ValueError("Completed and reviewed localization pilot required")
    metric = result["custom_duration_eer"]
    missed = sum(row["missed_spoof_seconds"] for row in result["per_record"])
    false = sum(row["false_alarm_seconds"] for row in result["per_record"])
    pages = [
        ("Finding manipulated time ranges", [
            ("p", "The second part of this project asks where fake audio occurs inside a recording. I used PartialSpoof v1.2 [8, 9], keeping its official splits. This first pilot uses 128 train and 64 dev recordings, balanced by utterance class and selected with seed 45 before fitting. Eval stays closed. PartialSpoof shares source material with ASVspoof, so this is not independent cross-dataset validation."),
            ("h", "Checking the time labels"),
            ("p", "All 192 selected WAVs passed individual hash and header checks. Each native binary vector has exactly ceil(N/160) labels for N samples at 16 kHz. No label was trimmed or padded. VAD timestamps can end up to 15 samples early, and VAD nonspeech can carry either binary class. I kept the publisher's binary truth and used VAD only for diagnostics."),
            ("h", "A small temporal model"),
            ("p", "I mapped the existing 60-dimensional LFCC features to 20 ms output intervals using the nearest window centre, choosing the earlier frame on ties. Native 10 ms labels are minimum-pooled in adjacent pairs; a final interval is clipped to the actual recording length. The model is StandardScaler plus logistic regression [11] (C=1, lbfgs, maximum 300 iterations). This is a new small baseline, not a reproduction of the authors' neural model."),
            ("p", "Normalization uses retained training frames only. The seed-45 reservoir is capped at 50,000 frames per binary class and balances to the smaller class. This pilot retained 4,524 frames per class and converged in 14 solver iterations without warnings. Output scores are uncalibrated and higher means bona fide. The classifier boundary zero was fixed for the displayed timelines."),
            ("table", [["Pilot measurement", "Observed value"],
                       ["Custom duration-weighted 20 ms EER", f"{metric['eer_fraction']*100:.2f}%"],
                       ["Missed spoof at threshold zero", f"{missed:.2f} / {metric['spoof_seconds']:.2f} s"],
                       ["False alarm at threshold zero", f"{false:.2f} / {metric['bonafide_seconds']:.2f} s"],
                       ["Computation / peak process RAM", f"{result['elapsed_seconds']:.2f} s / {result['memory']['peak_rss_bytes']/1024**2:.1f} MiB"]]),
            ("small", "The EER is our closest-grid duration diagnostic on pooled 20 ms labels, without interpolation. It is not official 10 ms SegmentEER or RangeEER. The threshold sweep uses this same small dev set. Computation excludes imports and data preparation. Independent output review reproduced the metric, durations, IDs, hashes and ten saved timelines.")]),
        ("What the first timelines show", [
            ("figure", "localization_timeline"),
            ("caption", "Two of ten dev examples fixed before scoring. The top recording contains substantial spoof; the second includes binary-spoof nonspeech. Gray background means the remaining duration, not a confidence interval. Predictions use the fixed zero threshold, with no smoothing."),
            ("p", "The first model makes many errors. At threshold zero, it marks a large part of bona fide duration as spoof and still misses real spoof ranges. The timelines are useful for inspecting those failures; they do not prove authenticity. Binary spoof reference spans include nonspeech where the publisher labelled it spoof."),
            ("h", "Limits of the pilot"),
            ("p", "There is one model seed and a deliberately small balanced utterance sample. Per-frame data are correlated within a recording, and there is no independent threshold test. The 20 ms output grid does not imply 20 ms acoustic context or streaming latency: LFCC windows and deltas use adjacent and future audio."),
            ("h", "Fragmented false alarms"),
            ("p", "At threshold zero, 109 pooled-reference spoof runs produce 1,349 prediction runs; 989 predictions have no reference overlap. The median predicted run lasts 60 ms. False alarms occur in speech as well as nonspeech. A VAD mask would change the reference target and would not solve the speech errors."),
            ("small", "Evidence: docs/validation/localization_alignment.json; localization_pilot_v1.json; localization_pilot_review_v1.json; localization_errors_v1.json. Native and pooled reference resolutions are kept separate in the error analysis.")])]
    context = read('localization_context_v1.json')
    comparison = context['baseline_comparison']
    ranges = read('localization_range_results_v1.json')
    if ranges['status'] != 'complete_derived_reference_diagnostic':
        raise ValueError('Completed derived-reference metric diagnostic required')
    pages.append(('One context change, and the reference limits', [
        ('p', "I tested one idea from the pilot plan: compare each frame with its own recording context. For every LFCC dimension, I subtracted the recording median and divided by max(1.4826 times MAD, 0.001). The full unlabelled recording supplies these statistics, including future audio. No dev labels enter the transform; train-only scaling and the classifier remain unchanged. This is offline and is not a new-method claim."),
        ('figure','localization_context'),
        ('caption','One seed and the same small dev subset. Bars use a zero-based axis; no confidence interval or generalization claim.'),
        ('table', [['Diagnostic', 'LFCC baseline', 'With context'],
                   ['Custom 20 ms duration EER', *[f'{v:.2f}%' for v in context_values()]],
                   ['Derived-reference RangeEER', *[f"{ranges['variants'][v]['range_eer_fraction']*100:.2f}%" for v in ('baseline','context')]],
                   ['Missed spoof at zero', f"{comparison['baseline_fixed_zero_missed_spoof_seconds']:.2f} s", f"{comparison['context_fixed_zero_missed_spoof_seconds']:.2f} s"],
                   ['False alarm at zero', f"{comparison['baseline_fixed_zero_false_alarm_seconds']:.2f} s", f"{comparison['context_fixed_zero_false_alarm_seconds']:.2f} s"]]),
        ('p', "Context normalization made the custom EER worse by 3.42 percentage points and increased both fixed-zero error durations. I kept the negative result and did not search for another MAD floor, seed or threshold. The first attempt stopped on a Windows status-file sharing error; an infrastructure-only retry used the same model and transform. Both attempts and pre-fit freezes are retained."),
        ('h','Why the range result is a diagnostic'),
        ('p', "The checked public train/dev archives contain no publisher RTTM reference. I converted the unchanged native 10 ms binary labels into contiguous spoof ranges and clipped the final interval to the actual WAV length. This is explicitly a derived reference; sub-bin boundaries and publisher silence treatment are unverified. I used the hash-pinned bounded patched upstream RangeEER [10] and pyannote.metrics [12], with separate dependencies and a 120-second external timeout. Independent duration arithmetic reproduced its selected operating point. It is not an unmodified official evaluator or paper reproduction."),
        ('small', "Range scoring uses the upstream rule score <= threshold for spoof; project timelines use score < threshold. The range metric uses native 10 ms truth, while the earlier custom metric uses pooled 20 ms truth. References, source hashes, solver state and failure logs are saved in the audits. These results support keeping the original baseline for now; a stronger representation needs a separately planned experiment.")]))
    return pages


if __name__ == '__main__':
    import hashlib
    from reportlab.graphics import renderPDF, renderSVG
    import pypdfium2 as pdfium
    output = ROOT/'docs/figures'
    stem = output/'localization_comparison'
    drawing = context_drawing(None)
    renderSVG.drawToFile(drawing, str(stem.with_suffix('.svg')))
    renderPDF.drawToFile(drawing, str(stem.with_suffix('.pdf')))
    document = pdfium.PdfDocument(str(stem.with_suffix('.pdf')))
    document[0].render(scale=2).to_pil().save(stem.with_suffix('.png'))
    evidence = {'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audits_sha256': {name: hashlib.sha256((ROOT/'docs/validation'/name).read_bytes()).hexdigest()
                         for name in ('localization_pilot_v1.json','localization_context_v1.json','localization_context_review_v1.json')},
        'outputs': {str(stem.with_suffix(suffix).relative_to(ROOT)): hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
                    for suffix in ('.svg','.pdf','.png')}}
    (output/'localization_comparison_manifest.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print('Saved localization comparison SVG, PDF and PNG')
