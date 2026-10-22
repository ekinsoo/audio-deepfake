import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/fetch_protocols.py"
SPEC = importlib.util.spec_from_file_location("fetch_protocols", MODULE_PATH)
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class Response(io.BytesIO):
    def __init__(self, body, url="https://zenodo.org/files/test.txt", headers=None):
        super().__init__(body)
        self.url = url
        self.headers = {} if headers is None else headers

    def geturl(self):
        return self.url


class FetchProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "cache"
        self.body = b"public protocol\n"
        self.entry = {"filename": "test.txt", "url": "https://zenodo.org/files/test.txt",
                      "size_bytes": len(self.body), "sha256": hashlib.sha256(self.body).hexdigest()}
        self.config = self.root / "config.json"
        self.write_config()

    def write_config(self):
        self.config.write_text(json.dumps({"protocol_dir": "unused", "splits": {"train": self.entry},
                                         "metadata": {}, "archive": {"url": "MUST_NOT_FETCH"}}), encoding="utf-8")

    def run_fetch(self, **kwargs):
        return fetch.fetch_protocols(self.config, self.output, **kwargs)

    def test_download_verifies_and_matching_cache_skips_network(self):
        with patch.object(fetch, "build_opener") as build:
            build.return_value.open.return_value = Response(self.body)
            self.assertEqual(self.run_fetch(), {"test.txt": "downloaded"})
            self.assertEqual((self.output / "test.txt").read_bytes(), self.body)
            self.assertEqual(build.return_value.open.call_args.kwargs["timeout"], 30)
            build.reset_mock()
            self.assertEqual(self.run_fetch(), {"test.txt": "verified"})
            self.assertEqual(self.run_fetch(check_only=True), {"test.txt": "verified"})
            build.assert_not_called()

    def test_checksum_and_oversized_response_leave_no_partial_output(self):
        for content in [b"x" * len(self.body), self.body + b"extra"]:
            with self.subTest(content=content), patch.object(fetch, "build_opener") as build:
                build.return_value.open.return_value = Response(content)
                with self.assertRaises(ValueError):
                    self.run_fetch()
                self.assertEqual(list(self.output.iterdir()), [])

    def test_bad_cache_requires_overwrite_and_failed_download_preserves_it(self):
        self.output.mkdir()
        destination = self.output / "test.txt"
        destination.write_bytes(b"old")
        with patch.object(fetch, "build_opener") as build:
            with self.assertRaises(ValueError):
                self.run_fetch()
            with self.assertRaises(ValueError):
                self.run_fetch(check_only=True, overwrite=True)
            build.assert_not_called()
            build.return_value.open.return_value = Response(b"x" * len(self.body))
            with self.assertRaises(ValueError):
                self.run_fetch(overwrite=True)
            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(list(self.output.iterdir()), [destination])
            build.return_value.open.return_value = Response(self.body)
            self.run_fetch(overwrite=True)
            self.assertEqual(destination.read_bytes(), self.body)

    def test_check_only_missing_cache_never_uses_network(self):
        with patch.object(fetch, "build_opener") as build:
            with self.assertRaisesRegex(ValueError, "Missing"):
                self.run_fetch(check_only=True)
            build.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_invalid_sources_and_pins_fail_before_network(self):
        variants = [{"size_bytes": fetch.MAX_BYTES + 1}, {"filename": "../test.txt"},
                    {"filename": "LA.zip"}, {"url": "http://zenodo.org/test.txt"},
                    {"url": "https://evil.example/test.txt"}, {"sha256": "invalid"}]
        original = self.entry.copy()
        for changes in variants:
            self.entry = original | changes
            self.write_config()
            with self.subTest(changes=changes), patch.object(fetch, "build_opener") as build:
                with self.assertRaises(ValueError):
                    self.run_fetch()
                build.assert_not_called()

    def test_redirect_and_response_headers_are_checked(self):
        handler = fetch.ZenodoRedirectHandler()
        with self.assertRaises(ValueError):
            handler.redirect_request(Mock(), None, 302, "Found", {}, "https://evil.example/archive.zip")
        for response in [Response(self.body, url="http://zenodo.org/test.txt"),
                         Response(self.body, headers={"Content-Length": str(fetch.MAX_BYTES + 1)})]:
            with self.subTest(response=response), patch.object(fetch, "build_opener") as build:
                build.return_value.open.return_value = response
                with self.assertRaises(ValueError):
                    self.run_fetch()
                self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
