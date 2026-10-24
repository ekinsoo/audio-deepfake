"""Export the PDF report's shared text, tables and figures as Markdown."""
import json
from pathlib import Path

from build_project_report import pages

ROOT = Path(__file__).resolve().parents[1]


def main():
    from reportlab.graphics import renderSVG
    from localization_report_content import timeline_drawing
    from ssl_report_content import drawing as ssl_drawing

    for name, draw in [('localization_timelines', timeline_drawing), ('ssl_timelines', ssl_drawing)]:
        renderSVG.drawToFile(draw(None), str(ROOT / 'docs/figures' / (name + '.svg')))
    figures = {'detection': 'detection_comparison.svg', 'reliability': 'reliability_comparison.svg',
               'errors': 'error_concentration.svg', 'localization_context': 'localization_comparison.svg',
               'localization_timeline': 'localization_timelines.svg', 'ssl_timeline': 'ssl_timelines.svg'}
    audit = json.loads((ROOT / 'docs/validation/full_robustness_results_v1.json').read_text(encoding='utf-8'))
    lines = ['# Audio Deepfake Detection and Localization', '',
             'Özgür Ekin Sönmez', '',
             'Markdown edition of the [PDF report](docs/report/audio_deepfake_report.pdf). '
             'Both use the same report text, tables and figure generators.', '']
    for title, blocks in pages(audit):
        lines.extend(['## ' + title, ''])
        for kind, value in blocks:
            if kind == 'table':
                for i, row in enumerate(value):
                    lines.append('| ' + ' | '.join(str(c).replace('|', '\\|') for c in row) + ' |')
                    if i == 0:
                        lines.append('| ' + ' | '.join('---' for _ in row) + ' |')
            elif kind == 'figure':
                lines.append(f'![{title}](docs/figures/{figures[value]})')
            elif kind == 'refs':
                lines.extend(f'- [{label}]({url})' for label, url in value)
            elif kind == 'h':
                lines.append('### ' + value)
            elif kind == 'formula':
                lines.append('`' + value + '`')
            else:
                lines.append(value)
            lines.append('')
    (ROOT / 'reports.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
