"""Fetch pinned ASVspoof metadata only; never download audio archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/datasets/asvspoof2019_la.json"
MAX_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 30


def validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "zenodo.org"
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError("Metadata URLs and redirects must use HTTPS on zenodo.org")


class ZenodoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_file(path: Path, entry: dict) -> None:
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"Size mismatch: {entry['filename']}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {entry['filename']}")


def read_entries(config: dict) -> list[dict]:
    entries = list(config["splits"].values()) + list(config["metadata"].values())
    names: set[str] = set()
    for entry in entries:
        name = entry["filename"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.txt", name):
            raise ValueError("Metadata filenames must be plain .txt filenames")
        if name in names:
            raise ValueError(f"Duplicate metadata filename: {name}")
        names.add(name)
        size = entry["size_bytes"]
        if type(size) is not int or not 0 < size <= MAX_BYTES:
            raise ValueError(f"Metadata size must be between 1 byte and 8 MiB: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"Invalid pinned SHA-256: {name}")
        validate_url(entry["url"])
    return entries


def download_entry(entry: dict, destination: Path) -> None:
    """Write to a sibling temporary file and replace only after verification."""
    temporary: Path | None = None
    try:
        opener = build_opener(ZenodoRedirectHandler())
        request = Request(entry["url"], headers={"User-Agent": "audio-deepfake-metadata/0.1"})
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            validate_url(response.geturl())
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != entry["size_bytes"]:
                raise ValueError(f"Response size mismatch: {entry['filename']}")
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                             suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                total = 0
                while chunk := response.read(65536):
                    total += len(chunk)
                    if total > MAX_BYTES or total > entry["size_bytes"]:
                        raise ValueError(f"Response exceeds pinned size or 8 MiB cap: {entry['filename']}")
                    output.write(chunk)
        validate_file(temporary, entry)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_protocols(config_path: str | Path = DEFAULT_CONFIG,
                    output_dir: str | Path | None = None, *,
                    overwrite: bool = False, check_only: bool = False) -> dict[str, str]:
    with Path(config_path).open(encoding="utf-8-sig") as source:
        config = json.load(source)
    entries = read_entries(config)
    directory = Path(output_dir) if output_dir is not None else REPO_ROOT / config["protocol_dir"]
    statuses: dict[str, str] = {}
    for entry in entries:
        destination = directory / entry["filename"]
        if destination.is_symlink():
            raise ValueError(f"Metadata destination must not be a symlink: {entry['filename']}")
        if destination.exists():
            try:
                validate_file(destination, entry)
                statuses[entry["filename"]] = "verified"
                continue
            except ValueError:
                if check_only or not overwrite:
                    raise
        elif check_only:
            raise ValueError(f"Missing cached metadata: {entry['filename']}")
        directory.mkdir(parents=True, exist_ok=True)
        download_entry(entry, destination)
        statuses[entry["filename"]] = "downloaded"
    return statuses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace mismatched cached metadata")
    parser.add_argument("--check-only", action="store_true", help="Verify cached files without network access")
    args = parser.parse_args(argv)
    try:
        statuses = fetch_protocols(args.config, args.output_dir,
                                  overwrite=args.overwrite, check_only=args.check_only)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Metadata error: {exc}\n")
    for name, status in statuses.items():
        print(f"{status}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
