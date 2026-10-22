"""Build a shareable PDF and standalone LaTeX report from the checked audit.

Uses ReportLab for the PDF export. LaTeX is independently compilable and embeds
its plots as native picture commands; it needs no external images or data.
Run with Python that has ReportLab, after build_report_figures.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfgen.canvas import Canvas

from build_report_figures import detection_figure, reliability_figure, error_figure
from ssl_report_content import report_pages as ssl_pages, drawing as ssl_drawing, tex as ssl_tex
from localization_report_content import (report_pages as localization_pages, timeline_drawing,
                                         timeline_tex, context_drawing, context_tex)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/report"
AUDIT = ROOT / "docs/validation/full_robustness_results_v1.json"
AUTHOR = "Özgür Ekin Sönmez"
NAVY = "222222"
TEAL = "444444"
ORANGE = "BD642F"
BLUE = "5876A5"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pages(r):
    runs = r["runs"]
    det = r["detection_aggregate"]
    rank = r["ranking_aggregate"]
    fmt = lambda x: f"{x:.4f}"
    conditions = (("clean", "Original"), ("narrowband", "Narrowband"), ("average", "Fixed average"))
    detection_rows = [["Seed", "Original EER", "Narrow EER", "Average EER"]]
    cost_rows = [["Seed", "Original t-DCF", "Narrow t-DCF", "Average t-DCF"]]
    for run in runs:
        detection_rows.append([str(run["seed"])] + [f"{run['metrics'][k]['eer']*100:.2f}%" for k, _ in conditions])
        cost_rows.append([str(run["seed"])] + [fmt(run["metrics"][k]["min_tDCF"]) for k, _ in conditions])
    detection_rows.append(["Mean +/- SD"] + [f"{det[k]['eer']['mean']*100:.2f} +/- {det[k]['eer']['std']*100:.2f}%" for k, _ in conditions])
    cost_rows.append(["Mean +/- SD"] + [f"{det[k]['min_tDCF']['mean']:.4f} +/- {det[k]['min_tDCF']['std']:.4f}" for k, _ in conditions])
    reliability_rows = [["Seed / ranking", "AUROC", "Average precision", "Error prevalence"]]
    for run in runs:
        for k, label in (("abs_delta", "Score change"), ("negative_margin", "Score margin")):
            rr = run["reliability"]
            reliability_rows.append([f"{run['seed']} / {label}", fmt(rr["rankings"][k]["auroc"]),
                                     fmt(rr["rankings"][k]["average_precision"]), f"{rr['error_prevalence']*100:.2f}%"])
    failures = runs[0]["failure_examples"]["examples"][:3]
    failure_rows = [["Recording ID", "Attack", "Signed change", "Distance to threshold"]]
    for ex in failures:
        failure_rows.append([ex["utterance_id"], ex["attack_id"], f"{ex['delta']:+.4f}", f"{-ex['negative_margin']:.4f}"])
    report_pages = [
        ("A small detector, and a test of its errors", [
            ("kicker", "AUDIO DEEPFAKE DETECTION / DEVELOPMENT REPORT"),
            ("lead", "I tested whether a simple bandwidth change can reveal when a small speech detector is wrong. It did not work well in this experiment. The original score's distance from a fixed decision threshold was a much better error-ranking signal."),
            ("h", "What I built"),
            ("p", "This is an independent portfolio project using public ASVspoof 2019 Logical Access (LA) data [1, 2]. The first milestone is full-recording detection: one score for an entire recording. A separate small PartialSpoof localization pilot now adds scores for time ranges inside a recording; it is covered later in this report."),
            ("p", "I chose LFCC features [3] and a Gaussian mixture model (GMM) [6] as a small, inspectable baseline. Running on a laptop CPU is an engineering choice; the research question concerns detection errors. I trained three fixed model seeds, then reused those models for a narrowband stress test. The useful result is a clear failure of the proposed diagnostic, with the data and checks needed to repeat it."),
            ("table", [["Official split", "Recordings", "Bona fide", "Spoof"], ["Train", "25,380", "2,580", "22,800"], ["Development", "24,844", "2,548", "22,296"]]),
            ("small", "Only official keyed train/dev records are used. The 142 unkeyed dev archive members are excluded. Evaluation audio was not extracted, decoded or scored."),
            ("h", "Baseline in one pass"),
            ("p", "The reader requires finite, nonempty, mono 16 kHz audio. Features use a 20 ms window, 10 ms hop, 512-point FFT, 20 linear filters, 20 coefficients, pre-emphasis 0.97, and first- and second-order deltas: 60 dimensions per frame. There is no VAD, CMVN or implicit input resampling."),
            ("p", "Each class has a 32-component diagonal-covariance GMM. All train recordings are visited; a fixed random-priority reservoir retains at most 50,000 frames per class. This is 100,000 retained frames in total, not every training frame. Seeds are 42, 43 and 44; all six fitted mixtures converged without warnings."),
            ("formula", "score = mean over frames [log p(frame | bona fide) - log p(frame | spoof)]"),
            ("small", "Larger scores support bona fide speech. This is a Python adaptation; feature parity with the official MATLAB baseline has not been established.")]),
        ("The experiment and its boundaries", [
            ("h", "One fixed transformation"),
            ("p", "For each dev recording, I resampled 16 kHz to 8 kHz and back to 16 kHz. The implementation uses scipy.signal.resample_poly [7] with a Kaiser window (beta 5), constant zero padding, float64 intermediate samples and float32 output trimmed to the original length. No gain, clipping, noise or codec is added. Original audio stays unchanged."),
            ("p", "This is an approximate bandwidth stress test, not a telephone channel simulation. The clean and transformed versions are scored by the same saved model. I keep both scores, their fixed average, their signed difference and its absolute value."),
            ("formula", "average = clean / 2 + narrow / 2;  delta = narrow - clean"),
            ("h", "Detection metrics"),
            ("p", "EER balances false rejection and false acceptance. Minimum normalized t-DCF is the challenge's ASV-oriented detection cost. Both are calculated using the hash-pinned official 2019 v1 Python evaluator and matching dev ASV scores [4, 5]. Lower is better. Diagnostic thresholds are selected using dev labels."),
            ("p", "Original scores are official-dev adaptation results. Narrowband and average scores are custom diagnostics. The ASV reference is kept unchanged, so their t-DCF does not model a transformed channel for both systems."),
            ("h", "Calibration and error analysis"),
            ("table", [["Dev group", "Speaker IDs", "Recordings", "Bona fide / spoof"], ["Calibration", "8", "9,609", "945 / 8,664"], ["Analysis", "12", "15,235", "1,603 / 13,632"]]),
            ("p", "The CM manifest contains 20 distinct dev speaker IDs. A fixed hash ranking assigns eight IDs to calibration and twelve to analysis. Groups share no speakers or records and are disjoint from train. All six attacks occur in each group. The partition uses the IDs recorded in the CM key, including those attached to spoof recordings."),
            ("p", "For each model I sort the 945 clean calibration bona fide scores and use zero-based index 47 as the threshold. A clean score at or above it is accepted as bona fide. Each threshold accepts 898/945 calibration bona fide recordings (95.03%). An analysis error means disagreement between this clean prediction and the label."),
            ("table", [["Seed", "Frozen threshold", "Calibration accepted"], *[[str(run["seed"]), f"{run['pins']['calibration']['threshold']:.10f}", "898 / 945"] for run in runs]]),
            ("small", "The protocol was frozen before full transformed scoring, after clean dev and pilot results were seen. This is not a blind test or independent validation.")]),
        ("Detection under a bandwidth change", [
            ("figure", "detection"),
            ("caption", "Bars show means only. Individual seeds and population standard deviation (ddof=0) are retained in the tables; SD is not a confidence interval. All conditions use the same 24,844 dev recordings."),
            ("table", detection_rows),
            ("table", cost_rows),
            ("p", "Narrowband scoring increased mean EER from 6.34% to 22.17%. The fixed average reached 9.03%, also worse than the original score. Minimum t-DCF followed the same pattern: 0.1724 for original, 0.5298 for narrowband and 0.2012 for the average. I kept all three seeds; no run was replaced based on its result."),
            ("h", "What changed from the pilot?"),
            ("p", "The balanced pilot used 2,000 train and 1,000 dev recordings with seed 42. Its original EER was 8.8%; the fixed average gave 8.4%. That small apparent benefit did not carry over to the full experiment. Both experiments retained at most 50,000 frames per class, but training recordings, sampled frame contents, dev population, class balance and the number of seeds differ. This comparison does not isolate one cause."),
            ("small", "The pilot is part of the full dev population. It is not an independent dataset, and the full experiment is not a new held-out benchmark.")]),
        ("Can score change rank clean prediction errors?", [
            ("p", "I compared two fixed signals on the 15,235 analysis recordings: absolute clean-to-narrow score change, and negative absolute distance from the clean decision threshold. A larger signal ranks a recording as more likely to be an error. Neither signal was fitted to the analysis labels."),
            ("figure", "reliability"),
            ("caption", "Higher is better. AUROC 0.5 is the no-ranking reference. Mean error prevalence is 5.41%; each seed has its own prevalence. Bars show means only; the table retains each seed."),
            ("table", reliability_rows),
            ("p", f"Across seeds, absolute score change gave AUROC {rank['abs_delta']['auroc']['mean']:.3f} and average precision {rank['abs_delta']['average_precision']['mean']:.3f}. The margin baseline gave AUROC {rank['negative_margin']['auroc']['mean']:.3f} and AP {rank['negative_margin']['average_precision']['mean']:.3f}. Mean error prevalence was {r['error_prevalence_aggregate']['mean']*100:.2f}%."),
            ("p", "In this setup, a large response to bandwidth change is not a useful warning that the original prediction is wrong. The simple margin is much more informative. I did not invert the change signal, search averaging weights or tune a new rejection threshold after seeing this result. This is error ranking, not calibrated probability or a validated abstention policy.")]),
        ("What the errors look like", [
            ("figure", "errors"),
            ("caption", "Seed 42, analysis speakers only, at its frozen calibration threshold. Bona fide errors are rejected genuine recordings; spoof errors are accepted spoof recordings. Each attack has 2,272 analysis recordings; bona fide has 1,603."),
            ("p", "Seed 42 made 774 analysis errors: 144 bona fide rejections, 181 A04 acceptances and 449 A06 acceptances. There were no observed errors for A01, A02, A03 or A05 at this threshold. These counts use the calibration threshold, so they differ from the separate full-dev EER-threshold diagnostic."),
            ("h", "Three selected examples"),
            ("table", failure_rows),
            ("p", "All three examples are spoof recordings accepted by the original model. LA_D_1402046 is close to the threshold but changes little after resampling. LA_D_5369918 changes much more, yet is farther from the threshold. These examples illustrate why transformation response and clean-score margin measure different things; they do not establish a causal explanation for the errors."),
            ("p", "Examples follow the frozen deterministic rule: first ID in each delta-sign and top-quarter disagreement stratum, then fill to ten by descending absolute change and ID. The full audit saves ten examples per seed. This is an illustrative selection, not a random sample."),
            ("h", "What I would inspect next"),
            ("p", "A04 and A06 are useful starting points for listening and feature inspection. The current results do not show that the model detects new synthesis systems, real telephone recordings or partially manipulated clips. The local demo shows a raw score and measured latency. It cannot establish that a recording is authentic.")]),
        ("Reproduction, limitations and next work", [
            ("h", "What is saved"),
            ("p", "Configs, seeds, dataset version, official splits, frame-sampling rules, model and source hashes, paired scores, partitions, thresholds, package versions, metrics and failure examples are recorded. Public summaries stay in Git; downloaded audio, reference evaluator code, local score tables and models are ignored. Only public data was used."),
            ("p", "Environment: Python 3.12.14; NumPy 2.5.3; SciPy 1.18.1; scikit-learn 1.9.1; Joblib 1.6.0; SoundFile 0.13.1. The local machine has about 16 GiB RAM and Intel Iris Xe graphics. The experiments use CPU inference and one numerical thread."),
            ("p", "Two interruptions are preserved. The first clean run exposed a memory-probe bug; a scoring-only recovery reused its saved model. Later, narrowband seed 44 stopped when system available RAM fell below 1 GiB, with worker peak memory about 207 MiB. Recovery kept seeds 42/43, counted the failed attempt against the same seed budget and finished all remaining chunks. These runs are not an uninterrupted end-to-end timing benchmark."),
            ("h", "Limits of this result"),
            ("p", "All reported metrics are development diagnostics. Train/dev use the same attack taxonomy. Clean dev and pilot outcomes were already inspected; speaker separation inside dev does not remove that reuse. Three seeds measure variation on one population, not statistical independence or a significance test. The stress test changes only the countermeasure input. LFCC parity with the official MATLAB baseline has not been checked."),
            ("p", "The project does not yet demonstrate unseen-generator generalization, calibrated confidence, real-world channel robustness, reliable partial manipulation localization or production deployment. A negative result here applies to this transformation, model and diagnostic, not to all transformation-based methods."),
            ("h", "Next milestone"),
            ("p", "The local scoring demo, timing study and first PartialSpoof localization pilot are complete. The later pages cover its alignment checks, first errors and one fixed context experiment. A stronger representation and genuinely external data would need a separate protocol and measured compute budget."),
            ("h", "Sources and evidence"),
            ("refs", [
                ("[1] ASVspoof 2019 LA v1. Public dataset, Zenodo record 6906306.", "https://zenodo.org/records/6906306"),
                ("[2] Todisco et al. (2019). ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection. Interspeech, 1008-1012.", "https://www.isca-archive.org/interspeech_2019/todisco19_interspeech.html"),
                ("[3] Sahidullah, Kinnunen and Hanilci (2015). A comparison of features for synthetic speech detection. Interspeech, 2087-2091. Linked from the official baseline page.", "https://www.asvspoof.org/index2019.html"),
                ("[4] ASVspoof 2019 Evaluation Plan, v0.4.", "https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf"),
                ("[5] Official ASVspoof 2019 v1 Python evaluator. Pinned locally for EER and normalized t-DCF.", "https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip"),
                ("[6] scikit-learn documentation: GaussianMixture.", "https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html"),
                ("[7] SciPy documentation: scipy.signal.resample_poly.", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html")]),
            ("h", "Acknowledgment"),
            ("small", "I used an AI coding assistant for implementation support, debugging, tests and report editing. Parallel agents helped review code and result tables. Reported metrics were calculated from saved experiment outputs."),
            ("small", "Repository evidence: docs/validation/full_robustness_results_v1.json; docs/FULL_ROBUSTNESS_PLAN.md; docs/REPORT_EVIDENCE.md; configs/full_seed42.toml (and seeds 43/44); scripts/build_report_figures.py; scripts/build_project_report.py. The report and figure manifests record input and output SHA-256 hashes."),
            ("small", "Dataset: Zenodo-6906306-v1, ODC-By 1.0. The official evaluator has a separate CC BY-NC-SA 4.0 license and stays outside Git. Public data attribution and source/license checks are in docs/DATA_PROTOCOL.md and docs/METRICS.md.")]),
    ]

    title, blocks = report_pages.pop()
    split = next(i for i, block in enumerate(blocks) if block == ("h", "Sources and evidence"))
    report_pages.append((title, blocks[:split]))
    timing = json.loads((ROOT / "docs/validation/demo_latency.json").read_text(encoding="utf-8"))
    plan = ROOT / timing["plan_path"]
    if timing["status"] != "complete" or sha(plan) != timing["plan_sha256"]:
        raise ValueError("Completed hash-matching demo timing required")
    t = timing["summary"]
    report_pages.append(("Local demo and processing time", [
        ("p", "The demo loads the saved seed-43 detector and scores one WAV or FLAC recording. Seed 43 was fixed before timing, without choosing the best detector seed. The model hash, configuration, package versions and training metadata are checked before use."),
        ("h", "What the demo shows"),
        ("p", "The interface runs at http://127.0.0.1:8765. It shows a raw model score, audio duration and processing time. Higher scores support bona fide speech in this model; the value is not a probability. Uploads stay in memory. Inputs must be finite mono 16 kHz WAV/FLAC, at most 60 seconds and 8 MiB. The server listens only on the local computer."),
        ("h", "A small, fixed timing study"),
        ("p", "I selected the first 20 official development protocol rows in their original order before timing. After one explicit warm-up, I made three measured passes: 60 observations. All 60 scores matched the saved seed-43 clean scores exactly. Measurement covers validation, audio decoding, LFCC extraction and GMM scoring through the Python scoring function; browser and HTTP overhead are excluded."),
        ("table", [["Measurement", "Observed value"],
                   ["Mean per recording", f"{t['mean_ms']:.2f} ms"],
                   ["Median (p50)", f"{t['p50_ms']:.2f} ms"],
                   ["p95", f"{t['p95_ms']:.2f} ms"],
                   ["Total processing / audio duration", f"{t['aggregate_rtf']:.5f} RTF"],
                   ["Model load and verification (separate)", f"{timing['model_load_seconds']:.3f} s"],
                   ["Process peak memory", "189 MiB"]]),
        ("p", "Hardware: Intel Core i7-1195G7, four cores and eight logical threads; one numerical thread was used. The GPU was unused. A ratio of 0.00312 means the measured processing took about 0.31% of the combined audio duration. Model loading is outside the per-recording timer."),
        ("h", "Limits of this measurement"),
        ("p", "These 20 recordings happen to be bona fide speech from one speaker (LA_0069), lasting 2.555-5.927 seconds. Repeated measurements share the same model and warm operating-system file cache. They are dependent observations. This is a limited engineering check, not a claim about all audio, attack diversity, cold starts, streaming or production performance."),
        ("small", "Evidence: docs/validation/demo_latency_plan.json and demo_latency.json. The plan was written before measurement and records selection, source/model hashes, resource limits and timing scope. No new training or large download was needed.")]))
    report_pages.extend(localization_pages())
    report_pages.extend(ssl_pages())
    references = blocks[split+1:]
    for kind, value in references:
        if kind == "refs":
            value.extend([
                ("[8] PartialSpoof v1.2 public data. Zenodo record 5766198.", "https://zenodo.org/records/5766198"),
                ("[9] Zhang et al. The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance.", "https://arxiv.org/abs/2204.05177"),
                ("[10] Zhang et al. Range-Based Equal Error Rate for Spoof Localization. Interspeech 2023.", "https://www.isca-archive.org/interspeech_2023/zhang23v_interspeech.html"),
                ("[11] scikit-learn documentation: LogisticRegression.", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"),
                ("[12] Bredin (2017). pyannote.metrics: A Toolkit for Reproducible Evaluation, Diagnostic, and Error Analysis of Speaker Diarization Systems.", "https://pyannote.github.io/pyannote-metrics/"),
                ("[13] Meta AI. wav2vec2-base model card, pinned revision 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8.", "https://huggingface.co/facebook/wav2vec2-base/tree/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8")])
    references.append(('small', 'PartialSpoof v1.2 has separate CC BY 4.0 attribution requirements; its ASVspoof source has ODC-By terms. See docs/PARTIALSPOOF_PROTOCOL.md. Localization audit sources and derived references are documented separately from official benchmark results.'))
    report_pages.append(("References and acknowledgment", references))
    return report_pages



def tex_escape(s):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(ch, ch) for ch in s)


def picture_plot(r, kind):
    """Native mean bars with numeric labels, no external LaTeX assets."""
    if kind == "localization_timeline":
        return timeline_tex(r)
    if kind == "localization_context":
        return context_tex(r)
    if kind == "ssl_timeline":
        return ssl_tex(r)
    lines = [r"\begin{center}\setlength{\unitlength}{1pt}\begin{picture}(480,220)"]
    def put(x, y, text, align="c"):
        lines.append(f"\\put({x:.3f},{y:.3f}){{\\makebox(0,0)[{align}]{{\\scriptsize {tex_escape(text)}}}}}")
    if kind == "errors":
        groups = r["runs"][0]["reliability"]["by_attack_id"]
        datasets = [("Seed 42 analysis errors", [groups[k]["errors"] for k in ("-","A01","A02","A03","A04","A05","A06")],
                     ["Bona fide","A01","A02","A03","A04","A05","A06"], 500, 100, "errors")]
    else:
        datasets = []
        for metric, title, top, step in (("eer","EER (%)",30,10),("min_tDCF","Minimum normalized t-DCF",.65,.2)) if kind == "detection" else (("auroc","Error-ranking AUROC",1,.25),("average_precision","Average precision",1,.25)):
            keys = ("clean","narrowband","average") if kind == "detection" else ("abs_delta","negative_margin")
            aggregate = r["detection_aggregate"] if kind == "detection" else r["ranking_aggregate"]
            multiplier = 100 if metric == "eer" else 1
            values = [aggregate[k][metric]["mean"]*multiplier for k in keys]
            labels = ["Original","Narrow","Average"] if kind == "detection" else ["Score change","Score margin"]
            datasets.append((title, values, labels, top, step, metric))
    for panel, (title, values, labels, top, step, metric) in enumerate(datasets):
        multiplier = 100 if metric == "eer" else 1
        ox, oy, width, height = (43,48,400,140) if kind == "errors" else (32+panel*242,48,190,140)
        put(ox+width/2, 206, title)
        tick = 0
        while tick <= top+1e-6:
            yy = oy+height*tick/top
            lines.append(f"\\put({ox},{yy:.3f}){{\\color{{lightgrid}}\\line(1,0){{{width}}}}}")
            put(ox-6, yy, f"{tick:g}", "r")
            tick += step
        for i, (value, label) in enumerate(zip(values, labels)):
            center = ox+(i+.5)*width/len(values)
            bw = min(34, width/len(values)*.55)
            palette = ("666666","999999","BBBBBB") if kind == "detection" else ("666666","AAAAAA")
            color = ("666666" if i == 0 else "999999") if kind == "errors" else palette[i]
            hh = height*value/top
            lines.append(f"\\put({center-bw/2:.3f},{oy}){{\\color[HTML]{{{color}}}\\rule{{{bw:.3f}pt}}{{{hh:.3f}pt}}}}")
            put(center, oy-15, label)
            if kind == "errors":
                put(center, oy+hh+9, str(value))
            else:
                put(center, oy+hh+10, f"{value:.2f}" if kind == "detection" and metric == "eer" else f"{value:.3f}")
    lines.append(r"\end{picture}\end{center}")
    return "\n".join(lines)


def build_tex(report_pages, r):
    header = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{xcolor,booktabs,array,hyperref}
\definecolor{navy}{HTML}{222222}
\definecolor{teal}{HTML}{444444}
\definecolor{lightgrid}{HTML}{EEEEEE}
\hypersetup{colorlinks=true,urlcolor=teal,linkcolor=teal,pdftitle={Audio Deepfake Detection and Localization},pdfauthor={Özgür Ekin Sönmez}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\renewcommand{\arraystretch}{1.2}
\newcommand{\reporttitle}[1]{{\color{navy}\Large\bfseries #1}\par\vspace{5pt}}
\begin{document}
"""
    lines = [header]
    for index, (title, blocks) in enumerate(report_pages):
        if index:
            lines.append(r"\clearpage")
        else:
            lines.append(r"{\small Özgür Ekin Sönmez / Independent ML project / 11 October 2026}\par")
        lines.append(r"\reporttitle{"+tex_escape(title)+"}")
        for typ, value in blocks:
            if typ == "figure":
                lines.append(picture_plot(r,value))
            elif typ == "table":
                cols = len(value[0])
                lines.extend([r"\begin{center}\small", r"\begin{tabular}{@{}"+"l"*cols+r"@{}}\toprule"])
                for rowidx, row in enumerate(value):
                    lines.append(" & ".join(tex_escape(str(cell)) for cell in row)+r" \\")
                    if rowidx == 0:
                        lines.append(r"\midrule")
                lines.extend([r"\bottomrule\end{tabular}\end{center}"])
            elif typ == "refs":
                lines.append(r"{\small")
                for text,url in value:
                    lines.append(r"\href{"+url+"}{"+tex_escape(text)+r"}\par")
                lines.append("}")
            elif typ == "h":
                lines.append(r"\vspace{4pt}{\color{navy}\bfseries "+tex_escape(value)+r"}\par")
            elif typ in ("small","caption","kicker"):
                lines.append(r"{\small "+tex_escape(value)+r"}\par")
            elif typ == "formula":
                lines.append(r"{\small\ttfamily "+tex_escape(value)+r"}\par")
            elif typ == "lead":
                lines.append(r"{\color{navy}\large "+tex_escape(value)+r"}\par")
            else:
                lines.append(tex_escape(value)+r"\par")
    lines.append(r"\end{document}")
    return "\n\n".join(lines)+"\n"


def build_pdf(report_pages, r, target):
    ink = colors.HexColor("#"+NAVY)
    body = ParagraphStyle("body",fontName="Helvetica",fontSize=9.4,leading=13.4,textColor=ink,spaceAfter=7)
    styles = {"p":body,
              "lead":ParagraphStyle("lead",parent=body,fontSize=12.3,leading=17,spaceAfter=13),
              "small":ParagraphStyle("small",parent=body,fontSize=8,leading=11,textColor=colors.HexColor("#555555"),spaceAfter=7),
              "caption":ParagraphStyle("caption",parent=body,fontSize=8,leading=11,textColor=colors.HexColor("#555555"),spaceAfter=11),
              "kicker":ParagraphStyle("kicker",parent=body,fontName="Helvetica-Bold",fontSize=8,textColor=colors.HexColor("#"+TEAL),spaceAfter=10),
              "h":ParagraphStyle("h",parent=body,fontName="Helvetica-Bold",fontSize=11.5,leading=15,spaceBefore=8,spaceAfter=7),
              "formula":ParagraphStyle("formula",parent=body,fontName="Courier",fontSize=8,leading=11,spaceBefore=4,spaceAfter=13)}
    title_style = ParagraphStyle("title",parent=body,fontName="Helvetica-Bold",fontSize=18,leading=22,spaceAfter=15)
    doc = SimpleDocTemplate(str(target),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=52,bottomMargin=45,
                           title="Audio Deepfake Detection and Localization",author=AUTHOR)
    story=[]
    figure_builders={"detection":detection_figure,"reliability":reliability_figure,"errors":error_figure,
                     "localization_timeline":timeline_drawing, "localization_context":context_drawing, "ssl_timeline":ssl_drawing}
    for index,(title,blocks) in enumerate(report_pages):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(escape(title),title_style))
        if index == 0:
            story.append(Paragraph(escape(AUTHOR) + " / Independent ML project / 11 October 2026", styles["small"]))
        for typ,value in blocks:
            if typ == "figure":
                fig=figure_builders[value](r)
                scale=(A4[0]-92)/fig.width
                fig.scale(scale,scale)
                fig.width*=scale
                fig.height*=scale
                story.extend([fig,Spacer(1,5)])
            elif typ == "table":
                width=A4[0]-92
                column_widths = ([.34,.22,.22,.22] if len(value[0])==4 and "ranking" in value[0][0]
                                 else [.22]+[.78/(len(value[0])-1)]*(len(value[0])-1))
                cellstyle=ParagraphStyle("cell",parent=styles["small"],fontSize=8,leading=10,textColor=ink,spaceAfter=0)
                cells=[[Paragraph(escape(str(cell)).replace(" +/- ", "<br/>+/- ") if col > 0 else escape(str(cell)),cellstyle)
                        for col,cell in enumerate(row)] for row in value]
                table=Table(cells,colWidths=[x*width for x in column_widths],hAlign="LEFT")
                table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#F3F3F3")),
                                           ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#"+TEAL)),
                                           ("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#CCCCCC")),
                                           ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),6),
                                           ("BOTTOMPADDING",(0,0),(-1,-1),6)]))
                story.extend([table,Spacer(1,9)])
            elif typ == "refs":
                for text,url in value:
                    story.append(Paragraph(f'<link href="{escape(url)}" color="#{TEAL}">{escape(text)}</link>',styles["small"]))
            else:
                story.append(Paragraph(escape(value),styles.get(typ,body)))
    def footer(canvas,document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
        canvas.line(46,36,A4[0]-46,36)
        canvas.setFont("Helvetica",7.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(46,24,AUTHOR + " | Audio Deepfake | 11 October 2026")
        canvas.drawRightString(A4[0]-46,24,str(document.page))
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer,
              canvasmaker=lambda *args, **kwargs: Canvas(*args, **{**kwargs, "invariant": 1}))


def main():
    r=json.loads(AUDIT.read_text(encoding="utf-8"))
    if r.get("status")!="ok" or r.get("seeds") != [42,43,44] or r.get("eval_enabled") is not False:
        raise ValueError("Report requires complete fixed-seed dev audit")
    local=ROOT/"outputs/robustness-full-v1/summary.json"
    if sha(local)!=r["local_summary_sha256"]:
        raise ValueError("Local/public summary differs")
    OUT.mkdir(parents=True,exist_ok=True)
    content=pages(r)
    tex=OUT/"audio_deepfake_report.tex"
    pdf=OUT/"audio_deepfake_report.pdf"
    tex.write_text(build_tex(content,r),encoding="utf-8")
    build_pdf(content,r,pdf)
    manifest={"dataset":r["dataset"],"audit_path":AUDIT.relative_to(ROOT).as_posix(),"audit_sha256":sha(AUDIT),
              "generator_sha256":sha(Path(__file__)),"figure_generator_sha256":sha(ROOT/"scripts/build_report_figures.py"),
              "content_pages":len(content),"author":AUTHOR,
              "latency_audit_sha256":sha(ROOT/"docs/validation/demo_latency.json"),"pdf_renderer":"ReportLab; same report text/data, separate from LaTeX preview",
              "localization_content_sha256":sha(ROOT/"scripts/localization_report_content.py"),
              "ssl_content_sha256":sha(ROOT/"scripts/ssl_report_content.py"),
              "ssl_audit_sha256":{name:sha(ROOT/"docs/validation"/name) for name in ("localization_ssl_convergence_train_v1.json", "localization_ssl_convergence_dev_v1.json")},
              "localization_audit_sha256":sha(ROOT/"docs/validation/localization_pilot_v1.json"),
              "localization_review_sha256":sha(ROOT/"docs/validation/localization_pilot_review_v1.json"),
              "localization_inputs_sha256": {name: sha(ROOT/"docs/validation"/name) for name in (
                  "localization_context_v1.json", "localization_context_review_v1.json",
                  "localization_errors_v1.json", "localization_references_v1.json", "localization_range_results_v1.json")},
              "localization_timeline_inputs_sha256": {uid: sha(ROOT/"docs/figures/localization_pilot_v1"/(uid+".svg"))
                  for uid in ("CON_D_0000586", "CON_D_0003090")},
              "latex":"Standalone native vector charts; no external assets",
              "outputs":{path.name:sha(path) for path in (tex,pdf)}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ok","outputs":list(manifest["outputs"])}))


if __name__=="__main__":
    main()
