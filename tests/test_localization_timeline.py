"""SVG content, safe escaping, alignment and threshold boundary checks."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from audio_deepfake.localization_timeline import render_timeline_svg


class LocalizationTimelineTests(unittest.TestCase):
    def render(self, **kwargs):
        arguments = dict(title="Synthetic test", duration=.085,
                         reference_intervals=[(.01, .03), (.06, .085)],
                         scores=[-.2, .5, .8, -.1, -.2], threshold=.5)
        arguments.update(kwargs)
        return render_timeline_svg(**arguments)

    def test_reference_prediction_ranges_and_equality_are_distinct(self):
        root = ET.fromstring(self.render())
        tracks = {name: [(float(node.attrib["data-start"]), float(node.attrib["data-end"]))
                         for node in root.iter() if node.attrib.get("data-track") == name]
                  for name in ("reference", "prediction")}
        self.assertEqual(tracks["reference"], [(.01, .03), (.06, .085)])
        self.assertEqual(tracks["prediction"], [(0, .02), (.06, .085)])
        self.assertIn("ties are bona fide", "".join(root.itertext()))

    def test_title_is_escaped_not_executed_and_svg_can_be_saved(self):
        title = '<script>alert("x")</script> & <Ekin>'
        svg = self.render(title=title)
        root = ET.fromstring(svg)
        self.assertNotIn("<script>", svg)
        self.assertEqual(root.find("{http://www.w3.org/2000/svg}title").text, title)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.svg"
            path.write_text(svg, encoding="utf-8")
            self.assertEqual(path.read_text(encoding="utf-8"), svg)

    def test_optional_curve_and_single_short_bin(self):
        root = ET.fromstring(self.render(include_curve=False))
        self.assertFalse(any(node.attrib.get("data-track") == "score" for node in root.iter()))
        root = ET.fromstring(self.render(duration=.0000625, scores=[0], threshold=0,
                                        reference_intervals=[]))
        self.assertTrue(any(node.tag.endswith("circle") for node in root.iter()))
        self.assertFalse(any(node.attrib.get("data-track") == "prediction" for node in root.iter()))

    def test_extreme_finite_constant_scores_have_finite_svg_coordinates(self):
        for scores, threshold in [([1e308] * 5, 1e308), ([-1e308] * 5, 1e308),
                                  ([0] * 5, 0), ([-1e308, 1e308, 0, 0, 0], 0)]:
            with self.subTest(scores=scores):
                root = ET.fromstring(self.render(scores=scores, threshold=threshold))
                for node in root.iter():
                    for key in ("x", "y", "x1", "y1", "x2", "y2", "cx", "cy"):
                        if key in node.attrib:
                            self.assertTrue(np.isfinite(float(node.attrib[key])))
                    if "points" in node.attrib:
                        for point in node.attrib["points"].split():
                            self.assertTrue(all(np.isfinite(float(v)) for v in point.split(",")))

    def test_invalid_audio_alignment_or_annotation_fails(self):
        cases = [dict(duration=0), dict(duration=np.nan), dict(step=0),
                 dict(scores=[0]), dict(scores=[np.nan] * 5),
                 dict(reference_intervals=[(0, .1)]),
                 dict(reference_intervals=[(.04, .06), (.05, .07)]),
                 dict(threshold=np.inf), dict(title="invalid\x00title"),
                 dict(title=1), dict(include_curve="false")]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.render(**changes)


if __name__ == "__main__":
    unittest.main()
