"""Offline integrity and cache checks for small official metric setup."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request
import zipfile

SPEC = importlib.util.spec_from_file_location('fetch_metric_reference', Path(__file__).resolve().parents[1] / 'scripts/fetch_metric_reference.py')
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class Response(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.status = 200
        self.headers = {'Content-Length': str(len(body))}
    def geturl(self):
        return fetch.ZIP_ENTRY['url']


class Opener:
    def __init__(self, body):
        self.body = body
    def open(self, request, timeout):
        return Response(self.body)


class ReferenceSetupTests(unittest.TestCase):
    def fixture(self):
        payloads = {'LICENSE': b'License notice', 'README.txt': b'Reference readme',
                    'eval_metrics.py': b'# synthetic source', 'scores/asv_dev.txt': b'target 1',
                    'scores/cm_dev.txt': b'bonafide 1'}
        members = {name: {'member': 'package/' + name, 'size_bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
                   for name, body in payloads.items()}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for name, body in payloads.items():
                archive.writestr(members[name]['member'], body)
            archive.writestr('../unselected-danger.py', b'never extract this')
        zip_bytes = buffer.getvalue()
        zip_entry = {**fetch.ZIP_ENTRY, 'size_bytes': len(zip_bytes), 'sha256': hashlib.sha256(zip_bytes).hexdigest()}
        asv_body = b'source target 1\n'
        asv_entry = {**fetch.ASV_ENTRY, 'size_bytes': len(asv_body), 'sha256': hashlib.sha256(asv_body).hexdigest()}
        return payloads, members, zip_bytes, zip_entry, asv_body, asv_entry

    def patches(self, fixture):
        return patch.multiple(fetch, MEMBERS=fixture[1], ZIP_ENTRY=fixture[3], ASV_ENTRY=fixture[5])

    def install(self, root, fixture):
        (root / fixture[3]['filename']).write_bytes(fixture[2])
        (root / fixture[5]['filename']).write_bytes(fixture[4])
        for name, body in fixture[0].items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)

    def test_cached_matching_files_and_check_only_never_use_network(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, self.patches(fixture):
            root = Path(temporary)
            self.install(root, fixture)
            with patch.object(fetch, 'download_zip', side_effect=AssertionError('network')), patch.object(fetch, 'download_entry', side_effect=AssertionError('network')):
                self.assertTrue(all(status == 'verified' for status in fetch.fetch_reference(root).values()))
                self.assertEqual(fetch.fetch_reference(root), fetch.fetch_reference(root, check_only=True))

    def test_check_only_missing_fails_without_writes(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, self.patches(fixture):
            root = Path(temporary) / 'missing'
            with patch.object(fetch, 'download_zip', side_effect=AssertionError('network')), self.assertRaisesRegex(ValueError, 'Missing cached'):
                fetch.fetch_reference(root, check_only=True)
            self.assertFalse(root.exists())

    def test_changed_cache_refused_then_explicit_overwrite_repairs(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, self.patches(fixture):
            root = Path(temporary)
            self.install(root, fixture)
            (root / 'eval_metrics.py').write_bytes(b'changed')
            with patch.object(fetch, 'download_zip', side_effect=AssertionError('network')), self.assertRaises(ValueError):
                fetch.fetch_reference(root)
            with patch.object(fetch, 'download_zip', side_effect=AssertionError('network')), patch.object(fetch, 'download_entry', side_effect=AssertionError('network')):
                statuses = fetch.fetch_reference(root, overwrite=True)
            self.assertEqual(statuses['eval_metrics.py'], 'extracted')
            self.assertEqual((root / 'eval_metrics.py').read_bytes(), fixture[0]['eval_metrics.py'])

    def test_fresh_setup_extracts_license_and_selected_members_only(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, self.patches(fixture):
            root = Path(temporary)
            def asv_download(entry, path):
                path.write_bytes(fixture[4])
            with patch.object(fetch, 'download_zip') as zip_mock:
                # Exercise the real ZIP downloader with an offline response.
                zip_mock.side_effect = lambda path: original_zip(path, opener=Opener(fixture[2]))
                with patch.object(fetch, 'download_entry', side_effect=asv_download):
                    statuses = fetch.fetch_reference(root)
            self.assertEqual(statuses['LICENSE'], 'extracted')
            self.assertEqual((root / 'LICENSE').read_bytes(), fixture[0]['LICENSE'])
            self.assertEqual((root / 'eval_metrics.py').read_bytes(), fixture[0]['eval_metrics.py'])
            self.assertFalse((root.parent / 'unselected-danger.py').exists())
            self.assertEqual(set(statuses), {fixture[3]['filename'], fixture[5]['filename'], *fixture[0]})

    def test_bad_zip_checksum_never_promotes(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, self.patches(fixture):
            path = Path(temporary) / fixture[3]['filename']
            body = b'x' * len(fixture[2])
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                fetch.download_zip(path, opener=Opener(body))
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])

    def test_zip_oversize_response_is_rejected(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, self.patches(fixture):
            with self.assertRaisesRegex(ValueError, 'response size mismatch'):
                fetch.download_zip(Path(temporary) / 'reference.zip', opener=Opener(fixture[2] + b'x'))

    def test_zip_body_cannot_exceed_pin_even_with_false_length_header(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, self.patches(fixture):
            response = Response(fixture[2] + b'x')
            response.headers['Content-Length'] = str(len(fixture[2]))
            opener = Opener(b'')
            opener.open = lambda request, timeout: response
            with self.assertRaisesRegex(ValueError, 'exceeds pinned size'):
                fetch.download_zip(Path(temporary) / 'reference.zip', opener=opener)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_member_hash_failure_writes_no_code_or_license(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, self.patches(fixture):
            root = Path(temporary)
            archive = root / fixture[3]['filename']
            archive.write_bytes(fixture[2])
            fetch.MEMBERS['eval_metrics.py']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'member SHA-256 mismatch'):
                fetch.extract_members(archive, root, ['LICENSE', 'eval_metrics.py'])
            self.assertFalse((root / 'LICENSE').exists())
            self.assertFalse((root / 'eval_metrics.py').exists())

    def test_redirect_guard_rejects_foreign_http_credentials_and_paths(self):
        guard = fetch.ReferenceRedirectHandler()
        for url in ['http://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip',
                    'https://evil.example/asvspoof2019/tDCF_python_v1.zip',
                    'https://user@www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip',
                    'https://www.asvspoof.org/other.zip']:
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, 'pinned ASVspoof HTTPS'):
                guard.redirect_request(Request(fetch.ZIP_ENTRY['url']), None, 302, '', {}, url)
        fetch.validate_reference_url(fetch.ZIP_ENTRY['url'])
        fetch.validate_reference_url(fetch.ZIP_ENTRY['url'].replace('www.asvspoof.org', 'asvspoof.org'))

    def test_unsafe_destination_is_refused(self):
        for name in ['../outside', '/absolute', 'C:/outside', 'scores\\outside']:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsafe reference destination'):
                fetch.destination(Path('unused'), name)


original_zip = fetch.download_zip
if __name__ == '__main__':
    unittest.main()
