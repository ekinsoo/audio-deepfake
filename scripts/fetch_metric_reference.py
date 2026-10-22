"""Fetch the small, pinned ASVspoof 2019 evaluator and official LA dev ASV scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_protocols import download_entry, validate_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / 'data/metadata/asvspoof2019/metrics'
ZIP_ENTRY = {
    'filename': 'tDCF_python_v1.zip', 'size_bytes': 252290,
    'sha256': '0e60f66ac297cf6bd49539af5e7650d799e1aedb7b50496450aa12d4eb4adbaa',
    'url': 'https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip',
}
ASV_ENTRY = {
    'filename': 'ASVspoof2019.LA.asv.dev.gi.trl.scores.txt', 'size_bytes': 632703,
    'sha256': 'fe8114cf608c3062c141df5979892eba9664aa37a8c99e97157f129d679d3389',
    'url': 'https://zenodo.org/api/records/6906306/files/LA.zip/container/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt',
}
# Explicit original member names; no extractall, bytecode, or plotting driver.
MEMBERS = {
    'LICENSE': {'member': 'tDCF_python_v1/LICENSE', 'size_bytes': 308, 'sha256': '3c2ac94fd3175166a850a52da4fe67b6a5a17f57151c44c29fabb81a644a2211'},
    'README.txt': {'member': 'tDCF_python_v1/README.txt', 'size_bytes': 587, 'sha256': '2d58c9eb9bc5f5a99c47439553719d42d0c3d8d0bb4f30d0c64e77a24fe544aa'},
    'eval_metrics.py': {'member': 'tDCF_python_v1/eval_metrics.py', 'size_bytes': 9962, 'sha256': '93ea505282d329ed14537d977a74b2d1a25d161ac6767a16e0216a1a18391049'},
    'scores/asv_dev.txt': {'member': 'tDCF_python_v1/scores/asv_dev.txt', 'size_bytes': 446017, 'sha256': '43f8ed0380acc20cab28aca113ffd7813811132f092ceb5596c2cbd92f3101e6'},
    'scores/cm_dev.txt': {'member': 'tDCF_python_v1/scores/cm_dev.txt', 'size_bytes': 759398, 'sha256': '6c0b36e06845d59ef85eee7aa7f426a84a4ced4316ea6105315a2902fab67237'},
}
MAX_ZIP_BYTES = 2 * 1024**2
MAX_MEMBER_BYTES = 2 * 1024**2


def validate_reference_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in ('asvspoof.org', 'www.asvspoof.org')
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.path != '/asvspoof2019/tDCF_python_v1.zip' or parsed.query or parsed.fragment):
        raise ValueError('Reference URL must use the pinned ASVspoof HTTPS archive path')


class ReferenceRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_reference_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def destination(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(part in ('.', '..') for part in relative.parts) or '\\' in name or ':' in name:
        raise ValueError('Unsafe reference destination')
    path = directory.joinpath(*relative.parts)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError('Reference destination must not use symlinks')
    return path


def download_zip(path: Path, *, opener=None) -> None:
    validate_reference_url(ZIP_ENTRY['url'])
    if not 0 < ZIP_ENTRY['size_bytes'] <= MAX_ZIP_BYTES:
        raise ValueError('Reference archive exceeds size cap')
    opener = opener or build_opener(ReferenceRedirectHandler())
    temporary = None
    try:
        request = Request(ZIP_ENTRY['url'], headers={'User-Agent': 'audio-deepfake-metrics/0.1'})
        with opener.open(request, timeout=30) as response:
            validate_reference_url(response.geturl())
            if response.status != 200:
                raise ValueError('Expected complete reference archive response')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != ZIP_ENTRY['size_bytes']:
                raise ValueError('Reference archive response size mismatch')
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.reference-', suffix='.part', delete=False) as target:
                temporary = Path(target.name)
                total = 0
                while chunk := response.read(65536):
                    total += len(chunk)
                    if total > ZIP_ENTRY['size_bytes'] or total > MAX_ZIP_BYTES:
                        raise ValueError('Reference archive exceeds pinned size')
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        validate_file(temporary, ZIP_ENTRY)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_members(archive: Path, directory: Path, names: list[str]) -> None:
    validate_file(archive, ZIP_ENTRY)
    # Validate every requested member before writing any output.
    payloads = {}
    with zipfile.ZipFile(archive) as source:
        for name in names:
            entry = MEMBERS[name]
            destination(directory, name)
            member = PurePosixPath(entry['member'])
            if member.is_absolute() or '..' in member.parts or '\\' in entry['member'] or ':' in entry['member']:
                raise ValueError('Unsafe reference archive member')
            infos = [item for item in source.infolist() if item.filename == entry['member']]
            if len(infos) != 1:
                raise ValueError('Missing or duplicate pinned reference member')
            info = infos[0]
            if info.is_dir() or info.file_size != entry['size_bytes'] or not 0 < info.file_size <= MAX_MEMBER_BYTES:
                raise ValueError('Reference member size mismatch')
            payload = source.read(info)
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise ValueError('Reference member SHA-256 mismatch')
            payloads[name] = payload
    # LICENSE is first in the pinned mapping; never install code before its notice.
    for name in sorted(names, key=lambda item: (item != 'LICENSE', item)):
        path = destination(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.reference-', suffix='.part', delete=False) as target:
                temporary = Path(target.name)
                target.write(payloads[name])
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def fetch_reference(output_dir: Path = DEFAULT_DIR, *, overwrite=False, check_only=False) -> dict[str, str]:
    directory = Path(output_dir)
    entries = {ZIP_ENTRY['filename']: ZIP_ENTRY, **MEMBERS, ASV_ENTRY['filename']: ASV_ENTRY}
    statuses, needed = {}, []
    # Refuse altered existing files before doing any network or writes.
    for name, entry in entries.items():
        path = destination(directory, name)
        if path.exists():
            try:
                validate_file(path, {**entry, 'filename': name})
                statuses[name] = 'verified'
                continue
            except ValueError:
                if check_only or not overwrite:
                    raise
        elif check_only:
            raise ValueError(f'Missing cached reference: {name}')
        needed.append(name)
    if check_only or not needed:
        return statuses
    directory.mkdir(parents=True, exist_ok=True)
    archive = destination(directory, ZIP_ENTRY['filename'])
    if ZIP_ENTRY['filename'] in needed:
        download_zip(archive)
        statuses[ZIP_ENTRY['filename']] = 'downloaded'
    member_names = [name for name in MEMBERS if name in needed]
    if member_names:
        extract_members(archive, directory, member_names)
        statuses.update({name: 'extracted' for name in member_names})
    if ASV_ENTRY['filename'] in needed:
        download_entry(ASV_ENTRY, destination(directory, ASV_ENTRY['filename']))
        statuses[ASV_ENTRY['filename']] = 'downloaded'
    return statuses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--overwrite', action='store_true', help='Replace mismatched cached reference files')
    parser.add_argument('--check-only', action='store_true', help='Verify all cached files without network or writes')
    args = parser.parse_args(argv)
    try:
        statuses = fetch_reference(args.output_dir, overwrite=args.overwrite, check_only=args.check_only)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Reference setup error: {error}\n')
    print(json.dumps(statuses, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
