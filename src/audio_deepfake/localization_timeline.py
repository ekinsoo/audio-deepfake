"""Plain SVG reference/prediction timelines for aligned localization scores.

Rendering returns text only: no file, audio, model or dataset is loaded. Scores
must already follow the project's time grid. This is not an official evaluator.
"""

from html import escape

import numpy as np

from .localization import scores_to_spoof_intervals, time_grid, validate_intervals


def render_timeline_svg(*, title, duration, reference_intervals, scores,
                        threshold, step=.02, include_curve=True):
    """Draw spoof spans and optionally the uncalibrated bin-score curve.

    Higher scores support bona fide; score < threshold predicts spoof and an
    exact tie predicts bona fide. This project tie convention may differ from
    official evaluation tools. Reference intervals are supplied by the caller,
    not derived from scores. No smoothing, merging across gaps or clipping of
    invalid annotations occurs. `step` is seconds and defaults to 20 ms.
    """
    if not isinstance(title, str) or any(ord(c) < 32 and c not in "\t\r\n" for c in title):
        raise ValueError("Title must be valid plain XML text")
    if not isinstance(include_curve, bool):
        raise ValueError("include_curve must be a boolean")
    reference = validate_intervals(reference_intervals, duration=duration)
    predicted = scores_to_spoof_intervals(
        scores, threshold=threshold, duration=duration, step=step)
    grid = time_grid(duration=duration, step=step)
    scores = np.asarray(scores, dtype=np.float64)
    duration = float(duration)
    threshold = float(threshold)
    width, height = 960, (360 if include_curve else 230)
    left, right = 140., 925.
    span = right - left

    def x(seconds):
        return left + float(seconds) / duration * span

    def line(x1, y1, x2, y2, **attrs):
        options = " ".join(f'{key.replace("_", "-")}="{escape(str(value), quote=True)}"'
                           for key, value in attrs.items())
        return (f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" '
                f'y2="{y2:.3f}" {options}/>')

    def text(value, tx, ty, **attrs):
        options = " ".join(f'{key.replace("_", "-")}="{escape(str(value), quote=True)}"'
                           for key, value in attrs.items())
        return f'<text x="{tx:.3f}" y="{ty:.3f}" {options}>{escape(str(value))}</text>'

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img">',
           f'<title>{escape(title)}</title>',
           '<desc>Reference and predicted spoof intervals. Higher scores support bona fide.</desc>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial, sans-serif" font-size="13" fill="#222">',
           text(title, 24, 30, font_size=18),
           text('Dark spans indicate spoof speech', 24, 53, font_size=12)]
    for label, intervals, y, fill in (
        ("Reference", reference, 78., "#555"),
        ("Prediction", predicted, 125., "#222"),
    ):
        svg += [text(label, 24, y + 19),
                f'<rect x="{left:.3f}" y="{y:.3f}" width="{span:.3f}" '
                'height="28" fill="#eee"/>']
        for start, end in intervals:
            svg.append(f'<rect data-track="{label.lower()}" data-start="{start:.12g}" '
                       f'data-end="{end:.12g}" x="{x(start):.3f}" y="{y:.3f}" '
                       f'width="{x(end)-x(start):.3f}" height="28" fill="{fill}">'
                       f'<title>{start:.6g}–{end:.6g} seconds</title></rect>')
    axis_y = 166.
    svg.append(line(left, axis_y, right, axis_y, stroke="#555"))
    for fraction in (0., .25, .5, .75, 1.):
        tx = left + span * fraction
        svg += [line(tx, axis_y, tx, axis_y + 5, stroke="#555"),
                text(f'{duration * fraction:.4g}', tx, axis_y + 20, text_anchor="middle")]
    svg.append(text("Time (seconds)", (left + right) / 2, axis_y + 39, text_anchor="middle"))

    if include_curve:
        top, bottom = 225., 305.
        # Scale before subtraction to keep even extreme finite inputs finite.
        scale = max(float(np.max(np.abs(scores))), abs(threshold), 1.)
        scaled = scores / scale
        cutoff = threshold / scale
        low, high = min(float(scaled.min()), cutoff), max(float(scaled.max()), cutoff)

        def y(value):
            fraction = .5 if high == low else (float(value) - low) / (high - low)
            return bottom - (bottom - top) * (.05 + .9 * fraction)

        svg += [text("Raw score", 24, (top + bottom) / 2),
                text(f'{max(float(scores.max()), threshold):.4g}', left - 8, top + 8,
                     text_anchor="end", font_size=11),
                text(f'{min(float(scores.min()), threshold):.4g}', left - 8, bottom,
                     text_anchor="end", font_size=11),
                line(left, y(cutoff), right, y(cutoff), stroke="#999", stroke_dasharray="4 3"),
                text(f'Threshold: {threshold:.4g}', right, top - 9, text_anchor="end", font_size=11)]
        centers = grid.mean(axis=1)
        points = " ".join(f'{x(center):.3f},{y(score):.3f}'
                          for center, score in zip(centers, scaled))
        svg.append(f'<polyline data-track="score" points="{points}" '
                   'fill="none" stroke="#222" stroke-width="1.2"/>')
        if len(scores) == 1:
            svg.append(f'<circle cx="{x(centers[0]):.3f}" cy="{y(scaled[0]):.3f}" '
                       'r="2" fill="#222"/>')
    svg += [text("Higher score supports bona fide. Spoof prediction: score < threshold; ties are bona fide.",
                 24, height - 15, font_size=11), '</g></svg>']
    return "\n".join(svg)
