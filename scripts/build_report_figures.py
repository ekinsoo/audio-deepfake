"""Build report figures from the completed, public robustness audit.

Requires reportlab and pypdfium2. No audio, model files, or training are used.
The Drawing-returning functions also support embedding in a ReportLab report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#262626")
MUTED = colors.HexColor("#666666")
GRID = colors.HexColor("#E8E8E8")
PALETTE = [colors.HexColor("#555555")]*3


def _text(d, x, y, text, size=10, color=INK, anchor="start", bold=False):
    d.add(String(x, y, text, fontName="Helvetica-Bold" if bold else "Helvetica",
                 fontSize=size, fillColor=color, textAnchor=anchor))


def _canvas(title, subtitle, height=340):
    d = Drawing(720, height)
    d.add(Rect(0, 0, 720, height, fillColor=colors.white, strokeColor=None))
    _text(d, 28, height-28, title, 15, bold=True)
    _text(d, 28, height-47, subtitle, 10, MUTED)
    return d


def _panel(d, x, y, w, h, maximum, ticks, title, formatter):
    _text(d, x, y+h+16, title, 12, bold=True)
    for tick in ticks:
        yy = y+h*tick/maximum
        d.add(Line(x, yy, x+w, yy, strokeColor=GRID, strokeWidth=.7))
        _text(d, x-8, yy-3, formatter(tick), 9, MUTED, "end")
    d.add(Line(x, y, x, y+h, strokeColor=MUTED, strokeWidth=.7))
    d.add(Line(x, y, x+w, y, strokeColor=MUTED, strokeWidth=.7))


def _bar(d, cx, y, h, maximum, mean, color, label, precision=2):
    scale = h/maximum
    d.add(Rect(cx-21, y, 42, mean*scale, fillColor=color,
               strokeColor=None))
    _text(d, cx, y+mean*scale+10, f"{mean:.{precision}f}", 11, anchor="middle", bold=True)
    _text(d, cx, y-18, label, 10, anchor="middle")


def detection_figure(audit):
    """Three-seed detection metrics, shown as mean bars only."""
    d = _canvas("Detection results",
                "ASVspoof 2019 LA dev | 24,844 recordings per seed | seeds 42, 43, 44")
    keys = ["clean", "narrowband", "average"]
    labels = ["Clean", "Narrowband", "Score average"]
    for x, metric, maximum, ticks, title, factor, precision in [
        (58, "eer", 30, [0, 10, 20, 30], "EER (%) | lower is better", 100, 2),
        (402, "min_tDCF", .7, [0, .2, .4, .6], "min t-DCF | lower is better", 1, 3),
    ]:
        _panel(d, x, 88, 270, 168, maximum, ticks, title,
               lambda n: f"{n:g}")
        for index, key in enumerate(keys):
            summary = audit["detection_aggregate"][key][metric]
            _bar(d, x+45+index*88, 88, 168, maximum,
                 summary["mean"]*factor, PALETTE[index], labels[index], precision)
    _text(d, 28, 43, "Bars show the mean across seeds 42, 43 and 44. Per-seed values and SD are in the report tables.", 10, MUTED)
    _text(d, 28, 26, "Clean: official dev Python adaptation. Narrowband / average: custom stress with unchanged ASV scores.", 9, MUTED)
    return d


def reliability_figure(audit):
    """Error-ranking metrics on the fixed analysis speakers, full 0..1 axes."""
    d = _canvas("Error-ranking results",
                "15,235 analysis recordings | errors defined by clean scores at calibration-only thresholds")
    prevalence = sum(r["reliability"]["error_prevalence"] for r in audit["runs"])/len(audit["runs"])
    for x, metric, title, baseline_label in [
        (58, "auroc", "Error AUROC | higher is better", "Chance = 0.5"),
        (402, "average_precision", "Average precision | higher is better",
         f"Mean error prevalence = {prevalence:.3f}"),
    ]:
        _panel(d, x, 88, 270, 168, 1, [0, .25, .5, .75, 1], title,
               lambda n: f"{n:g}")
        for index, (key, label) in enumerate([("abs_delta", "Absolute score change"),
                                             ("negative_margin", "Threshold proximity")]):
            summary = audit["ranking_aggregate"][key][metric]
            _bar(d, x+68+index*134, 88, 168, 1, summary["mean"],
                 PALETTE[index+1], label, 3)
        _text(d, x, 58, baseline_label, 9, MUTED)
    _text(d, 28, 35, "Bars show the mean across seeds 42, 43 and 44. Per-seed values and SD are in the report tables.", 10, MUTED)
    _text(d, 28, 18, "Threshold proximity = -|clean score - threshold|. AP baseline is mean prevalence, not a fitted reference.", 9, MUTED)
    return d


def error_figure(audit):
    """Seed-42 clean errors by attack at the frozen calibration threshold."""
    run = next(r for r in audit["runs"] if r["seed"] == 42)
    reliability = run["reliability"]
    d = _canvas("Errors by attack",
                "Seed 42 | analysis speakers only | threshold fixed from calibration bona fide recordings", 320)
    _panel(d, 62, 84, 625, 152, 500, [0, 100, 200, 300, 400, 500],
           "Misclassified recordings (count)", lambda n: f"{n:g}")
    for index, attack in enumerate(["-", "A01", "A02", "A03", "A04", "A05", "A06"]):
        data = reliability["by_attack_id"][attack]
        cx = 105+index*88
        color = PALETTE[0] if attack == "-" else PALETTE[2]
        d.add(Rect(cx-23, 84, 46, data["errors"]/500*152,
                   fillColor=color, strokeColor=None))
        _text(d, cx, 94+data["errors"]/500*152, str(data["errors"]), 11, anchor="middle", bold=True)
        _text(d, cx, 66, "Bona fide" if attack == "-" else attack, 10, anchor="middle")
        _text(d, cx, 51, f"n = {data['count']:,}", 9, MUTED, "middle")
    _text(d, 28, 27, f"{reliability['errors']:,} / {reliability['count']:,} errors; clean threshold = {reliability['threshold']:.6f}.", 10, MUTED)
    _text(d, 28, 12, "Counts use a calibration-only decision rule; they are not errors at the full-dev EER threshold.", 9, MUTED)
    return d


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT/"docs/validation/full_robustness_results_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"docs/figures")
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit["status"] != "ok" or audit["seeds"] != [42, 43, 44] or audit["std_ddof"] != 0:
        raise ValueError("Expected the completed three-seed audit with population SD")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import pypdfium2
    outputs = {}
    for name, builder in [("detection_comparison", detection_figure),
                          ("reliability_comparison", reliability_figure),
                          ("error_concentration", error_figure)]:
        drawing = builder(audit)
        pdf = args.output_dir/f"{name}.pdf"
        renderPDF.drawToFile(drawing, str(pdf),
                             author="Özgür Ekin Sönmez", invariant=1)
        renderSVG.drawToFile(drawing, str(args.output_dir/f"{name}.svg"))
        document = pypdfium2.PdfDocument(str(pdf))
        page = document[0]
        bitmap = page.render(scale=2)
        bitmap.to_pil().save(args.output_dir/f"{name}.png")
        bitmap.close()
        page.close()
        document.close()
        outputs[name] = {suffix: digest(args.output_dir/f"{name}.{suffix}")
                         for suffix in ["svg", "png", "pdf"]}
    manifest = {
        "source": args.audit.name, "source_sha256": digest(args.audit),
        "generator": "scripts/build_report_figures.py", "generator_sha256": digest(Path(__file__)),
        "seeds": audit["seeds"], "std_ddof": 0,
        "display": "summary_bars_only",
        "summary": "Arithmetic mean across three seeds; per-seed values and population SD remain in report tables",
        "overlays": "None: no whiskers, seed dots, or reference lines",
        "error_figure": "Seed 42 analysis partition; frozen calibration threshold, not EER threshold",
        "artifacts_sha256": outputs,
    }
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "figures": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()
