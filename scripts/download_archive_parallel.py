"""Download the official archive with at most four resumable range requests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.request import Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_pilot_data as official
from fetch_protocols import ZenodoRedirectHandler, validate_url

CHUNK_SIZE = 64 * 1024**2
BUFFER_SIZE = 1024**2


def prefix_digest(path, count):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while count:
            chunk = source.read(min(count, BUFFER_SIZE))
            if not chunk:
                raise ValueError('Archive prefix is truncated')
            result.update(chunk)
            count -= len(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as target:
        json.dump(value, target, indent=2)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def cache_path(archive, source='zenodo'):
    if source not in official.SOURCES:
        raise ValueError('Archive source must be zenodo or datashare')
    return archive.with_suffix('.zip.chunks' if source == 'zenodo' else '.zip.datashare.chunks')


def inspect_plan(archive, chunk_size=CHUNK_SIZE, *, source='zenodo'):
    """Read-only plan; never create a cache or touch an existing partial file."""
    partial = archive.with_suffix('.zip.part')
    cache = cache_path(archive, source)
    for path in (archive, partial, cache):
        if path.is_symlink():
            raise ValueError('Archive and cache must not be symlinks')
    if archive.exists():
        if archive.stat().st_size != official.SIZE or official.digest(archive, 'md5') != official.MD5:
            raise ValueError('Existing archive fails published size/MD5')
        return {'complete': True}
    have = partial.stat().st_size if partial.exists() else 0
    plan_file = cache / 'plan.json'
    if plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding='utf-8'))
        if plan.get('url') != official.SOURCES[source] or plan.get('size') != official.SIZE or plan.get('md5') != official.MD5 or plan.get('chunk_size') != chunk_size:
            raise ValueError('Cached plan does not match official archive or chunk size')
        if type(plan.get('prefix_size')) is not int or not 0 <= plan['prefix_size'] <= have <= official.SIZE:
            raise ValueError('Partial file size is inconsistent with cached plan')
        if not partial.exists() or prefix_digest(partial, plan['prefix_size']) != plan['prefix_sha256']:
            raise ValueError('Partial archive prefix changed')
    else:
        if cache.exists() and any(cache.iterdir()):
            raise ValueError('Unrecognized chunk cache without a plan')
        if not 0 <= have <= official.SIZE:
            raise ValueError('Partial archive exceeds published size')
        plan = {'url': official.SOURCES[source], 'size': official.SIZE, 'md5': official.MD5,
                'chunk_size': chunk_size, 'prefix_size': have,
                'prefix_sha256': prefix_digest(partial, have) if partial.exists() else hashlib.sha256(b'').hexdigest()}
    remaining = official.SIZE - plan['prefix_size']
    required = 2 * remaining + official.RESERVE
    parent = archive.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required:
        raise ValueError('Insufficient disk for archive, cached chunks and 22 GiB reserve')
    return {**plan, 'required_free_bytes': required, 'free_bytes': free, 'complete': False}


def ranges(plan):
    return [(start, min(start + plan['chunk_size'], plan['size']))
            for start in range(plan['prefix_size'], plan['size'], plan['chunk_size'])]


def cached_chunks(cache, plan):
    expected = set(ranges(plan))
    found = {}
    for path in cache.iterdir():
        if path.is_symlink():
            raise ValueError('Symlink in chunk cache')
        if path.name in ('plan.json', 'plan.tmp', 'run.lock'):
            continue
        match = re.fullmatch(r'(\d+)-(\d+)-([a-f0-9]{64})\.chunk', path.name)
        if not match:
            # Only our deterministic incomplete transfer filenames may be retried.
            incomplete = re.fullmatch(r'(\d+)-(\d+)\.download', path.name)
            if incomplete and tuple(map(int, incomplete.groups())) in expected:
                continue
            raise ValueError(f'Unrecognized chunk cache entry: {path.name}')
        start, end, checksum = int(match[1]), int(match[2]), match[3]
        if (start, end) not in expected or (start, end) in found:
            raise ValueError('Unexpected or duplicate cached range')
        if path.stat().st_size != end - start or official.digest(path) != checksum:
            raise ValueError('Cached chunk checksum mismatch')
        found[(start, end)] = path
    return found


def download_chunk(cache, start, end, *, opener=None, source='zenodo'):
    if source not in official.SOURCES:
        raise ValueError('Archive source must be zenodo or datashare')
    temporary = cache / f'{start}-{end}.download'
    handler = ZenodoRedirectHandler if source == 'zenodo' else official.DataShareRedirectHandler
    validate_source_url = validate_url if source == 'zenodo' else official.validate_datashare_url
    opener = opener or build_opener(handler())
    request = Request(official.SOURCES[source], headers={'User-Agent': 'audio-deepfake-pilot/0.1', 'Range': f'bytes={start}-{end - 1}'})
    checksum = hashlib.sha256()
    try:
        with opener.open(request, timeout=60) as response:
            validate_source_url(response.geturl())
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end - 1}/{official.SIZE}':
                raise ValueError('Server must honor exact chunk range')
            if int(response.headers.get('Content-Length', '-1')) != end - start:
                raise ValueError('Unexpected chunk response length')
            remaining = end - start
            with temporary.open('wb') as target:
                while remaining:
                    chunk = response.read(min(BUFFER_SIZE, remaining))
                    if not chunk:
                        raise ValueError('Interrupted chunk download')
                    target.write(chunk)
                    checksum.update(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise ValueError('Chunk response exceeds expected size')
                target.flush()
                os.fsync(target.fileno())
        destination = cache / f'{start}-{end}-{checksum.hexdigest()}.chunk'
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def assemble(archive, plan, chunks):
    """Fixed-offset writes allow replay after interruption; retain chunks on failure."""
    partial = archive.with_suffix('.zip.part')
    with partial.open('r+b') as target:
        for start, end in ranges(plan):
            path = chunks[(start, end)]
            if path.stat().st_size != end - start or official.digest(path) != path.name.split('-')[2].split('.')[0]:
                raise ValueError('Cached chunk changed before assembly')
            target.seek(start)
            with path.open('rb') as source:
                shutil.copyfileobj(source, target, BUFFER_SIZE)
            target.flush()
            os.fsync(target.fileno())
    if partial.stat().st_size != official.SIZE or official.digest(partial, 'md5') != official.MD5:
        raise ValueError('Archive MD5 failed; partial and chunk cache preserved')
    partial.replace(archive)


def execute(archive, workers=4, *, chunk_size=CHUNK_SIZE, source='zenodo'):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Workers must be between 1 and 4')
    plan = inspect_plan(archive, chunk_size, source=source)
    if plan['complete']:
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix('.zip.part')
    cache = cache_path(archive, source)
    cache.mkdir(exist_ok=True)
    lock = cache / 'run.lock'
    # A stale lock after a process crash requires explicit operator removal.
    with lock.open('x') as target:
        target.write(str(os.getpid()))
    try:
        if not (cache / 'plan.json').exists():
            if partial.exists() and partial.stat().st_size != plan['prefix_size']:
                raise ValueError('Another downloader changed the partial file')
            partial.touch(exist_ok=True)
            atomic_json(cache / 'plan.json', {key: plan[key] for key in ('url', 'size', 'md5', 'chunk_size', 'prefix_size', 'prefix_sha256')})
        chunks = cached_chunks(cache, plan)
        pending = iter([item for item in ranges(plan) if item not in chunks])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            def fill():
                while len(active) < workers:
                    item = next(pending, None)
                    if item is None:
                        break
                    active[pool.submit(download_chunk, cache, *item, source=source)] = item
            fill()
            last = time.monotonic()
            while active:
                done, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    item = active.pop(future)
                    chunks[item] = future.result()
                now = time.monotonic()
                if now - last >= 15 or not active:
                    have = plan['prefix_size'] + sum(end - start for start, end in chunks)
                    print(f'Archive chunks: {have:,}/{official.SIZE:,} bytes ({100 * have / official.SIZE:.1f}%)', file=sys.stderr, flush=True)
                    last = now
                fill()
        assemble(archive, plan, chunks)
        # Leave verified chunk files as a recoverable cache; operator may remove them.
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=official.ROOT / 'data/archives/LA.zip')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--source', choices=tuple(official.SOURCES), default='zenodo')
    parser.add_argument('--chunk-mib', type=int, choices=(8, 64), default=64)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    chunk_size = args.chunk_mib * 1024**2
    print(json.dumps(inspect_plan(args.archive, chunk_size, source=args.source), indent=2), flush=True)
    if args.execute:
        execute(args.archive, args.workers, chunk_size=chunk_size, source=args.source)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
