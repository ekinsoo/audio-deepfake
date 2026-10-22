"""Mocked resume, persistent budgets and archive extraction; no network."""

import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

SPEC = importlib.util.spec_from_file_location(
    "prepare_partial_audio", Path(__file__).resolve().parents[1] / "scripts/prepare_partial_audio.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Response:
    def __init__(self, payload=b"", status=200, headers=None, failure=None):
        self.body = io.BytesIO(payload)
        self.status = status
        self.headers = headers or {}
        self.failure = failure

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, amount):
        if self.failure:
            raise self.failure
        return self.body.read(amount)


class PartialAudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.dest = self.base / "data"
        self.dest.mkdir()
        for name, value in (("DEST", self.dest), ("AUDIT", self.base / "audit.json"),
                            ("READ_SIZE", 4)):
            context = patch.object(module, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.guard = patch.object(module, "guard", return_value={})
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.deadline = time.monotonic() + 100

    def fake_files(self, payload):
        mapping = {split: (len(payload), hashlib.md5(payload).hexdigest()) for split in ("train", "dev")}
        context = patch.object(module, "FILES", mapping)
        context.start()
        self.addCleanup(context.stop)

    def audit(self):
        value = {"charged_payload_bytes": 0, "known_received_bytes": 0,
                 "plan": module.plan(), "splits": {}}
        module.atomic_json(module.AUDIT, value)
        return value

    def test_dry_plan_preserves_official_split_pins_and_budget(self):
        self.assertEqual(set(module.plan()["files"]), {"train", "dev"})
        self.assertEqual(sum(v[0] for v in module.FILES.values()), 4_055_073_906)
        self.assertLess(sum(v[0] for v in module.FILES.values()), module.BUDGET)
        self.assertFalse(module.plan()["eval_access"])

    def test_verified_metadata_selection_and_export_gate(self):
        selection_path, metadata_path = self.base / "selection.json", self.base / "metadata.json"
        exported = self.dest / "labels.json"
        exported.write_text("{}", encoding="utf-8")
        selection = {"dataset": "PartialSpoof v1.2", "seed": 45, "splits": {}}
        for split, letter, count in (("train", "T", 128), ("dev", "D", 64)):
            selection["splits"][split] = {"selected_records": [
                {"utterance_id": f"CON_{letter}_{i:07d}"} for i in range(count)]}
        module.atomic_json(selection_path, selection)
        metadata = {"status": "metadata_verified_selection_annotated_audio_alignment_pending",
                    "eval_payload_parsed": False, "eval_metadata_extracted": False,
                    "native_metadata_parsed": {"train": True, "dev": True, "eval": False},
                    "selection_sha256": module.digest(selection_path),
                    "selected_labels_export": {"path": "data/labels.json", "size_bytes": 2,
                                               "sha256": module.digest(exported), "pickle_required": False}}
        module.atomic_json(metadata_path, metadata)
        with patch.object(module, "ROOT", self.base), patch.object(module, "SELECTION", selection_path), \
                patch.object(module, "METADATA", metadata_path):
            self.assertEqual(module.load_selection(), selection)
            exported.write_text('{"tampered":true}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                module.load_selection()
            exported.write_text("{}", encoding="utf-8")
            selection_path.write_text(selection_path.read_text() + " ", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                module.load_selection()

    def test_missing_cumulative_journal_cannot_reset_audio_history(self):
        selection_path, metadata_path = self.base / "selection.json", self.base / "metadata.json"
        selection_path.write_text("{}", encoding="utf-8")
        metadata_path.write_text("{}", encoding="utf-8")
        (self.dest / "audio_train_events.jsonl").write_text('{}\n', encoding="utf-8")
        with patch.object(module, "SELECTION", selection_path), patch.object(module, "METADATA", metadata_path):
            with self.assertRaises(RuntimeError):
                module.load_audit({})

    def test_validated_range_resume_and_checksum(self):
        payload = b"abcdefghij"
        self.fake_files(payload)
        (self.dest / "database_train.tar.gz.part").write_bytes(payload[:4])
        requests = []

        def opener(request, timeout):
            requests.append((request.get_header("Range"), timeout))
            return Response(payload[4:], 206, {"Content-Range": "bytes 4-9/10", "Content-Length": "6"})

        audit = self.audit()
        archive = module.download("train", audit, self.deadline, opener=opener)
        self.assertEqual(archive.read_bytes(), payload)
        self.assertEqual(requests, [("bytes=4-", 60)])
        self.assertEqual(audit["charged_payload_bytes"], 6)
        self.assertEqual(audit["known_received_bytes"], 6)
        self.assertFalse((self.dest / "database_train.tar.gz.part").exists())

    def test_resume_refuses_200_and_wrong_range_without_losing_partial(self):
        payload = b"abcdefghij"
        self.fake_files(payload)
        partial = self.dest / "database_train.tar.gz.part"
        partial.write_bytes(payload[:4])
        for response in (Response(payload), Response(payload[4:], 206, {"Content-Range": "bytes 0-5/10"})):
            audit = self.audit()
            with self.assertRaises(RuntimeError):
                module.download("train", audit, self.deadline, opener=lambda *a, **k: response)
            self.assertEqual(partial.read_bytes(), payload[:4])
            self.assertEqual(audit["charged_payload_bytes"], 0)

    def test_failed_read_is_charged_and_received_partial_is_retained(self):
        payload = b"abcdefghij"
        self.fake_files(payload)
        partial = self.dest / "database_train.tar.gz.part"
        partial.write_bytes(payload[:4])
        audit = self.audit()
        response = Response(status=206, headers={"Content-Range": "bytes 4-9/10"},
                            failure=http.client.IncompleteRead(b"ef", 4))
        with self.assertRaises(RuntimeError):
            module.download("train", audit, self.deadline, opener=lambda *a, **k: response)
        self.assertEqual(partial.read_bytes(), b"abcdef")
        saved = json.loads(module.AUDIT.read_text())
        self.assertEqual(saved["charged_payload_bytes"], 4)
        self.assertEqual(saved["known_received_bytes"], 2)

    def test_cumulative_budget_survives_restart(self):
        payload = b"abcdefghij"
        self.fake_files(payload)
        with patch.object(module, "BUDGET", 6):
            audit = self.audit()
            with self.assertRaises(RuntimeError):
                module.download("train", audit, self.deadline,
                                opener=lambda *a, **k: Response(payload))
            audit = json.loads(module.AUDIT.read_text())
            self.assertEqual(audit["charged_payload_bytes"], 4)
            with self.assertRaises(RuntimeError):
                module.download("train", audit, self.deadline, opener=lambda *a, **k:
                                Response(payload[4:], 206, {"Content-Range": "bytes 4-9/10"}))
            self.assertEqual((self.dest / "database_train.tar.gz.part").read_bytes(), payload[:4])

    def test_checksum_failure_preserves_full_partial_and_never_opens_tar(self):
        self.fake_files(b"abcdefghij")
        audit = self.audit()
        with self.assertRaises(RuntimeError):
            module.download("train", audit, self.deadline,
                            opener=lambda *a, **k: Response(b"wrongbytes"))
        self.assertTrue((self.dest / "database_train.tar.gz.part").exists())
        archive = self.dest / "bad.tar.gz"
        archive.write_bytes(b"wrongbytes")
        with patch.object(module.tarfile, "open") as opened, self.assertRaises(RuntimeError):
            module.extract_selected(archive, "train", [], self.deadline)
        opened.assert_not_called()

    def archive(self, members):
        archive = self.dest / "archive.tar.gz"
        with tarfile.open(archive, "w:gz") as container:
            for name, body, kind in members:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.size = len(body)
                if kind == tarfile.SYMTYPE:
                    member.linkname = "../../evil"
                container.addfile(member, io.BytesIO(body))
        payload = archive.read_bytes()
        self.fake_files(payload)
        return archive

    def test_only_selected_wav_written_and_duration_recorded(self):
        raw = io.BytesIO()
        sf.write(raw, np.zeros(321), 16000, format="WAV", subtype="PCM_16")
        archive = self.archive([
            ("database/train/con_wav/CON_T_0000000.wav", raw.getvalue(), tarfile.REGTYPE),
            ("database/train/con_wav/CON_T_0000001.wav", raw.getvalue(), tarfile.REGTYPE)])
        result = module.extract_selected(archive, "train", [{"utterance_id": "CON_T_0000001"}], self.deadline)
        self.assertEqual(result[0]["frames"], 321)
        self.assertEqual(result[0]["duration_seconds"], 321 / 16000)
        self.assertFalse((self.dest / "database/train/con_wav/CON_T_0000000.wav").exists())
        self.assertTrue((self.dest / "database/train/con_wav/CON_T_0000001.wav").exists())

    def test_tar_rejects_traversal_links_eval_and_oversized_selected_audio(self):
        for name, kind in (("../evil", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("database/train/evil", tarfile.SYMTYPE),
                           ("database/eval/con_wav/x.wav", tarfile.REGTYPE),
                           ("C:/evil", tarfile.REGTYPE)):
            with self.subTest(name=name):
                archive = self.archive([(name, b"evil", kind)])
                with self.assertRaises(ValueError):
                    module.extract_selected(archive, "train", [], self.deadline)
        archive = self.archive([("database/train/con_wav/CON_T_0000001.wav", b"12345", tarfile.REGTYPE)])
        with patch.object(module, "FILE_LIMIT", 4), self.assertRaises(ValueError):
            module.extract_selected(archive, "train", [{"utterance_id": "CON_T_0000001"}], self.deadline)


if __name__ == "__main__":
    unittest.main()
