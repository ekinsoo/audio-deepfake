"""Offline checks for bounded parallel archive downloads and recovery."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('download_archive_parallel', Path(__file__).resolve().parents[1] / 'scripts/download_archive_parallel.py')
parallel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel)
DOWNLOAD_CHUNK = parallel.download_chunk


class Response(io.BytesIO):
    def __init__(self, body, start, end, size, status=206):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Length': str(len(body)), 'Content-Range': f'bytes {start}-{end - 1}/{size}'}
    def geturl(self):
        return parallel.official.URL


class Opener:
    def __init__(self, response):
        self.response = response
    def open(self, request, timeout):
        self.request = request
        return self.response


class ParallelArchiveTests(unittest.TestCase):
    payload = b'abcdefghijkl'

    def constants(self):
        return patch.multiple(parallel.official, SIZE=len(self.payload), MD5=hashlib.md5(self.payload).hexdigest(), RESERVE=0)

    def fixture(self, root):
        archive = root / 'LA.zip'
        archive.with_suffix('.zip.part').write_bytes(self.payload[:3])
        return archive

    def chunk(self, cache, start, end, *, opener=None, source='zenodo'):
        response = Response(self.payload[start:end], start, end, len(self.payload))
        if source == 'datashare':
            response.geturl = lambda: parallel.official.DATASHARE_CONTENT_URL
        return DOWNLOAD_CHUNK(cache, start, end, opener=Opener(response), source=source)

    def test_datashare_cache_is_separate_and_old_cache_untouched(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            archive = self.fixture(Path(temporary))
            old_cache = parallel.cache_path(archive)
            old_cache.mkdir()
            (old_cache / 'run.lock').write_text('old process')
            with patch.object(parallel, 'download_chunk', side_effect=self.chunk):
                parallel.execute(archive, chunk_size=4, source='datashare')
            self.assertEqual(archive.read_bytes(), self.payload)
            self.assertEqual((old_cache / 'run.lock').read_text(), 'old process')
            mirror_cache = parallel.cache_path(archive, 'datashare')
            self.assertEqual(mirror_cache.name, 'LA.zip.datashare.chunks')
            self.assertEqual(json.loads((mirror_cache / 'plan.json').read_text())['url'], parallel.official.DATASHARE_URL)

    def test_datashare_chunk_uses_pinned_url_and_rejects_foreign_response(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            response = Response(b'defg', 3, 7, 12)
            response.geturl = lambda: parallel.official.DATASHARE_CONTENT_URL
            opener = Opener(response)
            DOWNLOAD_CHUNK(Path(temporary), 3, 7, opener=opener, source='datashare')
            self.assertEqual(opener.request.full_url, parallel.official.DATASHARE_URL)
            response = Response(b'hijk', 7, 11, 12)
            with self.assertRaisesRegex(ValueError, 'pinned HTTPS'):
                DOWNLOAD_CHUNK(Path(temporary), 7, 11, opener=Opener(response), source='datashare')

    def test_datashare_plan_url_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            archive = self.fixture(Path(temporary))
            with patch.object(parallel, 'download_chunk', side_effect=self.chunk), patch.object(parallel, 'assemble', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                parallel.execute(archive, chunk_size=4, source='datashare')
            plan_file = parallel.cache_path(archive, 'datashare') / 'plan.json'
            plan = json.loads(plan_file.read_text())
            plan['url'] = parallel.official.URL
            plan_file.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'Cached plan does not match'):
                parallel.execute(archive, chunk_size=4, source='datashare')

    def test_execute_reuses_prefix_and_promotes_only_verified_archive(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            archive = self.fixture(Path(temporary))
            with patch.object(parallel, 'download_chunk', side_effect=self.chunk) as fetch:
                parallel.execute(archive, workers=4, chunk_size=4)
            self.assertEqual(archive.read_bytes(), self.payload)
            self.assertEqual(sorted(call.args[1:] for call in fetch.call_args_list), [(3, 7), (7, 11), (11, 12)])
            self.assertFalse(archive.with_suffix('.zip.part').exists())
            parallel.execute(archive, chunk_size=4)

    def test_bad_range_leaves_no_completed_chunk(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            cache = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'exact chunk range'):
                parallel.download_chunk(cache, 3, 7, opener=Opener(Response(b'defg', 3, 7, 12, status=200)))
            self.assertEqual(list(cache.iterdir()), [])

    def test_short_response_is_not_cached(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            response = Response(b'def', 3, 7, 12)
            response.headers['Content-Length'] = '4'
            with self.assertRaisesRegex(ValueError, 'Interrupted chunk'):
                parallel.download_chunk(Path(temporary), 3, 7, opener=Opener(response))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_interrupted_assembly_replays_fixed_offsets_without_redownload(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            archive = self.fixture(Path(temporary))
            def interrupt(source, target, size):
                target.write(source.read(2))
                target.flush()
                raise KeyboardInterrupt
            with patch.object(parallel, 'download_chunk', side_effect=self.chunk), patch.object(parallel.shutil, 'copyfileobj', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                parallel.execute(archive, chunk_size=4)
            self.assertEqual(archive.with_suffix('.zip.part').read_bytes(), b'abcde')
            with patch.object(parallel, 'download_chunk', side_effect=AssertionError('unexpected download')):
                parallel.execute(archive, chunk_size=4)
            self.assertEqual(archive.read_bytes(), self.payload)

    def test_checksum_failure_preserves_cache_and_partial(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            archive = self.fixture(Path(temporary))
            with patch.object(parallel, 'download_chunk', side_effect=self.chunk), patch.object(parallel.official, 'MD5', '0' * 32), self.assertRaisesRegex(ValueError, 'Archive MD5 failed'):
                parallel.execute(archive, chunk_size=4)
            self.assertFalse(archive.exists())
            self.assertEqual(archive.with_suffix('.zip.part').read_bytes(), self.payload)
            self.assertEqual(len(list(archive.with_suffix('.zip.chunks').glob('*.chunk'))), 3)

    def test_corrupt_cached_chunk_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            archive = self.fixture(Path(temporary))
            with patch.object(parallel, 'download_chunk', side_effect=self.chunk), patch.object(parallel, 'assemble', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                parallel.execute(archive, chunk_size=4)
            chunk = next(archive.with_suffix('.zip.chunks').glob('*.chunk'))
            chunk.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'chunk checksum mismatch'):
                parallel.execute(archive, chunk_size=4)

    def test_changed_prefix_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            archive = self.fixture(Path(temporary))
            with patch.object(parallel, 'download_chunk', side_effect=self.chunk), patch.object(parallel, 'assemble', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                parallel.execute(archive, chunk_size=4)
            archive.with_suffix('.zip.part').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'prefix changed'):
                parallel.execute(archive, chunk_size=4)

    def test_plan_is_read_only_and_unknown_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary, self.constants():
            archive = Path(temporary) / 'LA.zip'
            plan = parallel.inspect_plan(archive, chunk_size=4)
            self.assertEqual(plan['prefix_size'], 0)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            cache = archive.with_suffix('.zip.chunks')
            cache.mkdir()
            (cache / 'unknown').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'without a plan'):
                parallel.execute(archive, chunk_size=4)

    def test_worker_cap(self):
        for workers in (0, 5):
            with self.assertRaisesRegex(ValueError, 'between 1 and 4'):
                parallel.execute(Path('unused.zip'), workers=workers)


if __name__ == '__main__':
    unittest.main()
