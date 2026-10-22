# Project report

[PDF](audio_deepfake_report.pdf) · [LaTeX source](audio_deepfake_report.tex)

The report has 13 pages: the LFCC-GMM detection experiment, bandwidth/error
ranking, the local demo, PartialSpoof localization, context normalization,
and a separate frozen-wav2vec2 convergence follow-up. The final pages retain
the failed original SSL fit and the follow-up's different optimization budget.
The report also acknowledges AI assistance with implementation, tests and editing.

## Build

Use a separate reporting environment so the recorded model environment stays
unchanged. From the repository root on Windows:

```powershell
python -m venv .cache/report-env
.\.cache\report-env\Scripts\python.exe -m pip install -r requirements/reporting.txt
.\.cache\report-env\Scripts\python.exe scripts/build_report_figures.py
.\.cache\report-env\Scripts\python.exe scripts/localization_report_content.py
.\.cache\report-env\Scripts\python.exe scripts/build_project_report.py
```

The full report builder requires the matching ignored
`outputs/robustness-full-v1/summary.json`, checked against the saved public audit.
The saved PDF and standalone LaTeX remain readable without that local artifact.
The figure builder reads the public audit. Report building does not train or
score a model.

For LaTeX alone, run `pdflatex audio_deepfake_report.tex` in this directory or
open it in a LaTeX editor. Plots are embedded as native vectors; no dataset or
external images are needed. The ReportLab PDF and LaTeX share content, with
different typography.

Edit the generators before rebuilding:

- `scripts/build_project_report.py`: detection, demo and document layout.
- `scripts/localization_report_content.py`: LFCC localization and context.
- `scripts/ssl_report_content.py`: SSL follow-up and comparison timelines.

Direct edits to generated `.tex` are overwritten by the next build.
`manifest.json` records generator, evidence and output hashes. Original
experiment evidence should remain unchanged when editing report prose.

The [final SSL report review](../validation/localization_ssl_report_review_v1.json)
records the 13-page rendering check and successful LaTeX compilation.
