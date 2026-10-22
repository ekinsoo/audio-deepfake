"""Offline checks for bounded archive-member access and cache integrity."""

import importlib.util
import io
import tempfile
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
import soundfile as sf

from audio_deepfake.manifests import ManifestRow, write_manifest

SPEC = importlib.util.spec_from_file_location("fetch_pilot_audio", Path(__file__).resolve().parents[1] / "scripts/fetch_pilot_audio.py")
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class Response(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}

    def geturl(self):
        return fetch.SOURCE + "ASVspoof2019_LA_train/flac/LA_T_1.flac"


class Opener:
    def __init__(self, body):
        self.body = body
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        assert timeout == 30
        assert "/container/LA/ASVspoof2019_LA_" in request.full_url
        return Response(self.body)


class PilotAudioTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "outputs/test-pilot-temp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        self.row = ManifestRow("train", "speaker1", "LA_T_1", "-", "bonafide", "ASVspoof2019_LA_train/flac/LA_T_1.flac")
        buf = io.BytesIO()
        sf.write(buf, np.zeros(1600), 16000, format="FLAC", subtype="PCM_16")
        self.body = buf.getvalue()

    def tearDown(self):
        self.temp.cleanup()

    def test_download_validates_original_and_accounts_bytes(self):
        report = {"network_bytes": 0, "errors": []}
        dest = self.root / "audio.flac"
        result = fetch.fetch_one(self.row, dest, report, opener=Opener(self.body))
        self.assertEqual(result["duration_seconds"], 0.1)
        self.assertEqual(result["size_bytes"], report["network_bytes"])
        self.assertEqual(len(result["sha256"]), 64)
        self.assertTrue(dest.exists())

    def test_cap_and_invalid_audio_leave_no_destination(self):
        for body, spent in ((b"x" * (fetch.FILE_CAP + 1), 0), (self.body, fetch.TOTAL_CAP), (b"not flac", 0)):
            with self.subTest(spent=spent, size=len(body)):
                dest = self.root / "audio.flac"
                with self.assertRaises((ValueError, RuntimeError)):
                    fetch.fetch_one(self.row, dest, {"network_bytes": spent, "errors": []}, opener=Opener(body))
                self.assertFalse(dest.exists())
                self.assertEqual(list(self.root.glob("*.part")), [])

    def test_exact_file_and_total_boundary_succeeds_with_known_length(self):
        size = len(self.body)
        report = {"network_bytes": 100, "errors": []}
        with patch.object(fetch, "FILE_CAP", size), patch.object(fetch, "TOTAL_CAP", size + 100):
            result = fetch.fetch_one(self.row, self.root / "boundary.flac", report, opener=Opener(self.body))
        self.assertEqual(result["size_bytes"], size)
        self.assertEqual(report["network_bytes"], size + 100)

    def test_two_inflight_downloads_share_strict_total_budget(self):
        report = {"network_bytes": 0, "errors": []}
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        opener = Opener(self.body)
        def open_response(request, timeout):
            barrier.wait(timeout=5)
            return Response(self.body)
        size = len(self.body)
        with patch.object(opener, "open", side_effect=open_response), patch.object(fetch, "TOTAL_CAP", size + size // 2):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(fetch.fetch_one, self.row, self.root / f"{i}.flac", report, opener=opener, budget_lock=lock) for i in range(2)]
                results = []
                for future in futures:
                    try:
                        results.append(future.result())
                    except ValueError:
                        pass
        self.assertEqual(len(results), 1)
        self.assertEqual(report["network_bytes"], size + size // 2)
        self.assertEqual(len(list(self.root.glob("*.flac"))), 1)
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_eval_and_escape_rejected(self):
        with self.assertRaises(ValueError):
            fetch.safe_destination(self.root, ManifestRow("eval", "speaker", "LA_E_1", "A07", "spoof", "ASVspoof2019_LA_eval/flac/LA_E_1.flac"))
        with self.assertRaises(ValueError):
            fetch.safe_destination(self.root, ManifestRow("train", "speaker", "../bad", "-", "bonafide", "../bad"))

    def test_rate_limit_is_never_retried(self):
        opener = Opener(self.body)
        with patch.object(opener, "open", side_effect=HTTPError(fetch.SOURCE, 429, "Too Many Requests", {}, None)) as request, patch.object(fetch.time, "sleep") as sleep:
            report = {"network_bytes": 0, "errors": []}
            with self.assertRaises(HTTPError):
                fetch.fetch_one(self.row, self.root / "audio.flac", report, opener=opener)
            self.assertEqual(request.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(report["network_bytes"], 0)

    def test_wrong_sample_rate_and_subtype_are_rejected(self):
        for rate, subtype in ((8000, "PCM_16"), (16000, "PCM_24")):
            with self.subTest(rate=rate, subtype=subtype):
                path = self.root / "invalid.flac"
                sf.write(path, np.zeros(100), rate, subtype=subtype)
                with self.assertRaises(ValueError):
                    fetch.inspect_audio(path)

    def test_cached_audio_is_verified_and_never_silently_replaced(self):
        rows = [self.row]
        data = self.root / "data"
        manifests = self.root / "manifests"
        manifests.mkdir()
        for split in ("train", "dev"):
            (manifests / f"pilot_{split}.csv").write_text("source", encoding="utf-8")
        output = self.root / "output"
        with patch.object(fetch, "select_records", return_value=rows), patch.object(fetch, "build_opener", return_value=Opener(self.body)), patch.object(fetch.time, "sleep"):
            report = fetch.run(manifests, data, output, 1)
            self.assertEqual(len(report["records"]), 1)
            with patch.object(fetch, "fetch_one", side_effect=AssertionError("Unexpected network")):
                fetch.run(manifests, data, output, 1)
            (data / self.row.audio_path).write_bytes(b"changed")
            with self.assertRaises((ValueError, RuntimeError)):
                fetch.run(manifests, data, output, 1)
            self.assertEqual((data / self.row.audio_path).read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
