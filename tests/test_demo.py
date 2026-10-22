"""Validate bounded audio inputs, trusted artifacts, and loopback upload guards."""

import http.client
from http.server import HTTPServer
import io
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audio_deepfake import demo
from audio_deepfake.features import LFCCConfig


def wav(samples, rate=16000, subtype="PCM_16"):
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


MODEL = SimpleNamespace(lfcc_config=LFCCConfig(), gmm_config=SimpleNamespace(seed=43))


class DemoAudioTests(unittest.TestCase):
    def test_metadata_boundaries_reject_before_inference(self):
        cases = [
            (wav(np.zeros(160), 8000), "16000 Hz"),
            (wav(np.zeros((160, 2))), "mono"),
            (wav(np.zeros(0)), "empty"),
            (wav(np.array([0, np.nan]), subtype="FLOAT"), "non-finite"),
            (wav(np.zeros(60 * 16000 + 1)), "60 seconds"),
            (b"x" * (demo.MAX_BYTES + 1), "8 MiB"),
            (b"not an audio file", "decode"),
        ]
        for payload, message in cases:
            with self.subTest(message=message), patch.object(demo, "score_features") as score:
                with self.assertRaisesRegex(ValueError, message):
                    demo.score_audio(MODEL, payload)
                score.assert_not_called()

    def test_score_timing_and_score_direction(self):
        with patch.object(demo, "score_features", return_value=-2.5):
            result = demo.score_audio(MODEL, wav(np.zeros(1600)))
        self.assertEqual(result["score"], -2.5)
        self.assertEqual(result["duration_seconds"], .1)
        self.assertEqual(result["model_seed"], 43)
        self.assertEqual(result["sample_rate"], 16000)
        self.assertGreaterEqual(result["decode_seconds"], 0)
        self.assertGreaterEqual(result["inference_seconds"], 0)
        self.assertAlmostEqual(result["total_seconds"],
                               result["decode_seconds"] + result["inference_seconds"])
        self.assertIn("not a probability", result["score_direction"])

    def test_hash_mismatch_prevents_unpickling(self):
        with patch.object(demo, "_digest", side_effect=[demo.PROTOCOL_SHA256, "wrong"]), \
                patch.object(demo, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                demo.load_demo_model()
            load.assert_not_called()

    def test_changed_audit_prevents_unpickling(self):
        with patch.object(demo, "_digest", return_value="wrong"), \
                patch.object(demo, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "audit changed"):
                demo.load_demo_model()
            load.assert_not_called()


class DemoHTTPTests(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), demo.make_handler(MODEL, 0, "test-token"))
        self.port = self.server.server_port
        self.server.RequestHandlerClass = demo.make_handler(MODEL, self.port, "test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def valid_headers(self):
        return {"Origin": f"http://127.0.0.1:{self.port}",
                "X-Demo-Token": "test-token", "Content-Type": "application/octet-stream"}

    def test_page_has_no_probability_verdict(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"not a probability", body)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_unknown_host_denied(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.test"})[0], 403)

    def test_token_and_origin_required(self):
        headers = self.valid_headers()
        for key in ("Origin", "X-Demo-Token"):
            bad = dict(headers)
            bad.pop(key)
            self.assertEqual(self.request("POST", "/score", b"x", bad)[0], 403)
        bad = dict(headers, Origin="http://external.test")
        self.assertEqual(self.request("POST", "/score", b"x", bad)[0], 403)

    def test_oversize_and_chunked_denied_without_read(self):
        headers = self.valid_headers()
        headers["Content-Length"] = str(demo.MAX_BYTES + 1)
        self.assertEqual(self.request("POST", "/score", None, headers)[0], 413)
        headers = dict(self.valid_headers(), **{"Transfer-Encoding": "chunked"})
        self.assertEqual(self.request("POST", "/score", b"x", headers)[0], 400)

    def test_valid_upload_and_invalid_audio(self):
        with patch.object(demo, "score_features", return_value=.25):
            status, _, body = self.request("POST", "/score", wav(np.zeros(160)), self.valid_headers())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["score"], .25)
        self.assertEqual(self.request("POST", "/score", b"invalid", self.valid_headers())[0], 400)


if __name__ == "__main__":
    unittest.main()
