"""Offline checks for archive resume and selective pilot extraction."""

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

import numpy as np
import soundfile as sf

from audio_deepfake.manifests import ManifestRow, write_manifest

SPEC = importlib.util.spec_from_file_location("prepare_pilot_data", Path(__file__).resolve().parents[1] / "scripts/prepare_pilot_data.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class Response(io.BytesIO):
    def __init__(self, body, offset, size, status=206):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body)), "Content-Range": f"bytes {offset}-{size - 1}/{size}"}

    def geturl(self):
        return prepare.URL


class Opener:
    def __init__(self, response):
        self.response = response
        self.request = None

    def open(self, request, timeout):
        self.request = request
        return self.response


class PreparePilotTests(unittest.TestCase):
    def selection_fixture(self, root, balanced=True):
        hashes = {}
        for split, prefix, count in (("train", "T", 2000), ("dev", "D", 1000)):
            rows = []
            for index in range(count):
                label = "bonafide" if not balanced or index < count // 2 else "spoof"
                rows.append(ManifestRow(split, f"LA_{prefix}_speaker", f"LA_{prefix}_{index}",
                                        "-" if label == "bonafide" else "A01", label,
                                        f"ASVspoof2019_LA_{split}/flac/LA_{prefix}_{index}.flac"))
            path = root / f"pilot_{split}.csv"
            write_manifest(rows, path)
            hashes[path.name] = prepare.digest(path)
        return hashes

    def test_pinned_selection_and_balanced_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes = self.selection_fixture(root)
            with patch.object(prepare, "MANIFEST_SHA256", hashes):
                rows, observed = prepare.read_selection(root)
                self.assertEqual(len(rows), 3000)
                self.assertEqual(observed, hashes)
                with (root / "pilot_dev.csv").open("a") as source:
                    source.write("\n")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    prepare.read_selection(root)

    def test_matching_hash_does_not_bypass_class_balance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes = self.selection_fixture(root, balanced=False)
            with patch.object(prepare, "MANIFEST_SHA256", hashes), self.assertRaisesRegex(ValueError, "equal bonafide/spoof"):
                prepare.read_selection(root)

    def test_resume_and_verify_published_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LA.zip"
            path.with_suffix(".zip.part").write_bytes(b"abc")
            opener = Opener(Response(b"def", 3, 6))
            with patch.object(prepare, "SIZE", 6), patch.object(prepare, "MD5", hashlib.md5(b"abcdef").hexdigest()):
                prepare.download_archive(path, opener=opener)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual(opener.request.get_header("Range"), "bytes=3-5")
            self.assertFalse(path.with_suffix(".zip.part").exists())

    def test_datashare_resume_uses_pinned_source_and_same_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LA.zip"
            path.with_suffix(".zip.part").write_bytes(b"abc")
            response = Response(b"def", 3, 6)
            response.geturl = lambda: prepare.DATASHARE_CONTENT_URL
            opener = Opener(response)
            with patch.object(prepare, "SIZE", 6), patch.object(prepare, "MD5", hashlib.md5(b"abcdef").hexdigest()):
                prepare.download_archive(path, opener=opener, source="datashare")
                # A verified archive can be reused regardless of the selected mirror.
                prepare.download_archive(path, opener=Opener(None), source="zenodo")
            self.assertEqual(opener.request.full_url, prepare.DATASHARE_URL)
            self.assertEqual(opener.request.get_header("Range"), "bytes=3-5")
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_datashare_redirect_and_response_host_guards(self):
        for url in (prepare.DATASHARE_URL, prepare.DATASHARE_CONTENT_URL):
            prepare.validate_datashare_url(url)
        invalid = [prepare.DATASHARE_URL.replace("https:", "http:"),
                   prepare.DATASHARE_URL.replace("datashare.ed.ac.uk", "evil.example"),
                   prepare.DATASHARE_URL.replace("datashare.ed.ac.uk", "user@datashare.ed.ac.uk"),
                   prepare.DATASHARE_URL + "?x=1", "https://datashare.ed.ac.uk/unrelated"]
        handler = prepare.DataShareRedirectHandler()
        for url in invalid:
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "pinned HTTPS"):
                handler.redirect_request(Request(prepare.DATASHARE_URL), None, 302, "", {}, url)
        with tempfile.TemporaryDirectory() as temporary, patch.object(prepare, "SIZE", 3):
            response = Response(b"abc", 0, 3)
            with self.assertRaisesRegex(ValueError, "pinned HTTPS"):
                prepare.download_archive(Path(temporary) / "LA.zip", opener=Opener(response), source="datashare")

    def test_datashare_does_not_bypass_final_checksum(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(prepare, "SIZE", 3):
            path = Path(temporary) / "LA.zip"
            response = Response(b"abc", 0, 3)
            response.geturl = lambda: prepare.DATASHARE_CONTENT_URL
            with self.assertRaisesRegex(ValueError, "MD5 failed"):
                prepare.download_archive(path, opener=Opener(response), source="datashare")
            self.assertFalse(path.exists())

    def test_range_ignored_preserves_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LA.zip"
            partial = path.with_suffix(".zip.part")
            partial.write_bytes(b"abc")
            with patch.object(prepare, "SIZE", 6), self.assertRaisesRegex(ValueError, "exact range"):
                prepare.download_archive(path, opener=Opener(Response(b"abcdef", 0, 6, status=200)))
            self.assertEqual(partial.read_bytes(), b"abc")

    def test_wrong_checksum_never_promotes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LA.zip"
            with patch.object(prepare, "SIZE", 3), self.assertRaisesRegex(ValueError, "MD5 failed"):
                prepare.download_archive(path, opener=Opener(Response(b"abc", 0, 3)))
            self.assertFalse(path.exists())

    def fixture(self, root):
        row = ManifestRow("train", "LA_1", "LA_T_1", "-", "bonafide", "ASVspoof2019_LA_train/flac/LA_T_1.flac")
        waveform = io.BytesIO()
        sf.write(waveform, np.zeros(160, dtype=np.float32), 16000, format="FLAC", subtype="PCM_16")
        archive = root / "LA.zip"
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("LA/" + row.audio_path, waveform.getvalue())
            target.writestr("LA/ASVspoof2019_LA_eval/flac/LA_E_1.flac", waveform.getvalue())
        return row, archive

    def test_extract_only_selection_and_check_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, archive = self.fixture(root)
            journal = root / "journal.json"
            report = prepare.extract_selected(archive, [row], root / "audio", journal, {}, root / "missing.json")
            self.assertEqual(list(report["records"]), [row.audio_path])
            self.assertEqual(len(list((root / "audio").rglob("*.flac"))), 1)
            prepare.extract_selected(archive, [row], root / "audio", journal, {}, root / "missing.json")
            (root / "audio" / row.audio_path).write_bytes(b"broken")
            with self.assertRaises(Exception):
                prepare.extract_selected(archive, [row], root / "audio", journal, {}, root / "missing.json")

    def test_existing_audio_can_recover_from_verified_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, archive = self.fixture(root)
            destination = root / "audio" / row.audio_path
            destination.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive) as source:
                destination.write_bytes(source.read("LA/" + row.audio_path))
            report = prepare.extract_selected(archive, [row], root / "audio", root / "journal.json", {}, root / "missing.json")
            self.assertEqual(report["records"][row.audio_path]["sha256"], prepare.digest(destination))
            smoke = root / "smoke.json"
            smoke.write_text(json.dumps({"source_manifest_sha256": {}, "records": {row.audio_path: prepare.inspect_audio(destination)}}))
            report = prepare.extract_selected(archive, [row], root / "audio", root / "new.json", {}, smoke)
            self.assertEqual(len(report["records"]), 1)

    def test_interrupted_rename_before_journal_write_recovers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, archive = self.fixture(root)
            journal = root / "journal.json"
            with patch.object(prepare, "save_report", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                prepare.extract_selected(archive, [row], root / "audio", journal, {}, root / "missing.json")
            destination = root / "audio" / row.audio_path
            self.assertTrue(destination.exists())
            self.assertFalse(journal.exists())
            report = prepare.extract_selected(archive, [row], root / "audio", journal, {}, root / "missing.json")
            self.assertEqual(report["records"][row.audio_path]["sha256"], prepare.digest(destination))
            self.assertEqual(json.loads(journal.read_text())["records"], report["records"])
            self.assertEqual(report["last_chosen_archive_url"], prepare.URL)

    def test_unjournaled_audio_different_from_archive_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, archive = self.fixture(root)
            destination = root / "audio" / row.audio_path
            destination.parent.mkdir(parents=True)
            sf.write(destination, np.ones(160, dtype=np.float32) * 0.25, 16000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "differs from archive member"):
                prepare.extract_selected(archive, [row], root / "audio", root / "journal.json", {}, root / "missing.json")
            self.assertEqual(json.loads((root / "journal.json").read_text())["records"], {})

    def test_known_journal_mismatch_cannot_be_adopted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, archive = self.fixture(root)
            journal = root / "journal.json"
            report = prepare.extract_selected(archive, [row], root / "audio", journal, {}, root / "missing.json")
            report["records"][row.audio_path]["sha256"] = "0" * 64
            journal.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Changed journaled audio"):
                prepare.extract_selected(archive, [row], root / "audio", journal, {}, root / "missing.json")

    def test_eval_forbidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, archive = self.fixture(root)
            bad = ManifestRow("eval", "LA_1", "LA_E_1", "-", "bonafide", "ASVspoof2019_LA_eval/flac/LA_E_1.flac")
            with self.assertRaisesRegex(ValueError, "Eval is forbidden"):
                prepare.extract_selected(archive, [bad], root / "audio", root / "journal.json", {}, root / "missing.json")


if __name__ == "__main__":
    unittest.main()
