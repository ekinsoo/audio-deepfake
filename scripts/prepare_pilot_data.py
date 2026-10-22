"""Plan or resume the official archive download and extract only the fixed pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from audio_deepfake.manifests import ManifestRow, validate_splits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_pilot_audio import ROOT, inspect_audio, safe_destination, save_report
from fetch_protocols import ZenodoRedirectHandler, validate_url

URL = "https://zenodo.org/records/6906306/files/LA.zip?download=1"
DATASHARE_URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
DATASHARE_CONTENT_URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
SOURCES = {"zenodo": URL, "datashare": DATASHARE_URL}
SIZE = 7_640_952_520
MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
RESERVE = 22 * 1024**3  # 20 GiB reserve plus 2 GiB extraction allowance.
CHUNK = 1024**2
MANIFEST_SHA256 = {
    "pilot_train.csv": "aa7a1ae65fbf02b99e587a57b53ee3ce1814649c9e3927290a25a6c7cafadc17",
    "pilot_dev.csv": "449228652db5c402522ae951aab6f84342e59f6d785049f7f00278a13d2ac6c8",
}


def validate_datashare_url(url: str) -> None:
    parsed = urlsplit(url)
    allowed_paths = {urlsplit(DATASHARE_URL).path, urlsplit(DATASHARE_CONTENT_URL).path}
    if (parsed.scheme != "https" or parsed.hostname != "datashare.ed.ac.uk"
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.path not in allowed_paths or parsed.query or parsed.fragment):
        raise ValueError("DataShare archive URLs must use pinned HTTPS host and bitstream paths")


class DataShareRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_datashare_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            result.update(chunk)
    return result.hexdigest()


def read_selection(directory: Path) -> tuple[list[ManifestRow], dict[str, str]]:
    splits, hashes = {}, {}
    for split, count in (("train", 2000), ("dev", 1000)):
        path = directory / f"pilot_{split}.csv"
        observed_hash = digest(path)
        if observed_hash != MANIFEST_SHA256[path.name]:
            raise ValueError(f"Fixed pilot manifest SHA-256 mismatch: {path.name}")
        with path.open(encoding="utf-8") as source:
            rows = [ManifestRow(**row) for row in csv.DictReader(source)]
        if len(rows) != count or {row.split for row in rows} != {split}:
            raise ValueError(f"Expected fixed {count} {split} rows")
        if sum(row.label == "bonafide" for row in rows) != count // 2 or sum(row.label == "spoof" for row in rows) != count // 2:
            raise ValueError(f"Expected equal bonafide/spoof counts in {split}")
        splits[split] = rows
        hashes[path.name] = observed_hash
    validate_splits(splits)
    return splits["train"] + splits["dev"], hashes


def preflight(archive: Path, data_root: Path) -> dict:
    archive.parent.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    if archive.is_symlink() or archive.with_suffix(".zip.part").is_symlink():
        raise ValueError("Archive path must not be a symlink")
    partial = archive.with_suffix(".zip.part")
    have = archive.stat().st_size if archive.exists() else partial.stat().st_size if partial.exists() else 0
    if have > SIZE:
        raise ValueError("Cached archive exceeds published size")
    free = shutil.disk_usage(archive.parent).free
    data_free = shutil.disk_usage(data_root).free
    required = SIZE - have + RESERVE
    if free < required or data_free < RESERVE:
        raise ValueError("Insufficient disk: archive remainder plus 22 GiB required")
    return {"free_bytes": free, "required_free_bytes": required, "cached_bytes": have}


def download_archive(path: Path, *, opener=None, source: str = "zenodo") -> None:
    """Keep incomplete bytes for resume; promote only after published MD5 passes."""
    if source not in SOURCES:
        raise ValueError("Archive source must be zenodo or datashare")
    archive_url = SOURCES[source]
    validate_source_url = validate_url if source == "zenodo" else validate_datashare_url
    redirect_handler = ZenodoRedirectHandler if source == "zenodo" else DataShareRedirectHandler
    if path.exists():
        if path.stat().st_size != SIZE or digest(path, "md5") != MD5:
            raise ValueError("Existing archive does not match published size/MD5")
        return
    partial = path.with_suffix(".zip.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > SIZE:
        raise ValueError("Partial archive exceeds published size")
    if offset < SIZE:
        opener = opener or build_opener(redirect_handler())
        headers = {"User-Agent": "audio-deepfake-pilot/0.1", "Range": f"bytes={offset}-{SIZE - 1}"}
        last_progress_bytes, last_progress_time = offset, time.monotonic()
        print(f"Archive: {offset:,}/{SIZE:,} bytes ({100 * offset / SIZE:.1f}%)", file=sys.stderr, flush=True)
        with opener.open(Request(archive_url, headers=headers), timeout=60) as response:
            validate_source_url(response.geturl())
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{SIZE - 1}/{SIZE}":
                raise ValueError("Server must honor exact range for bounded resume")
            if int(response.headers.get("Content-Length", "-1")) != SIZE - offset:
                raise ValueError("Unexpected archive response length")
            with partial.open("ab") as target:
                while offset < SIZE:
                    chunk = response.read(min(CHUNK, SIZE - offset))
                    if not chunk:
                        raise ValueError("Interrupted archive; rerun to resume")
                    target.write(chunk)
                    offset += len(chunk)
                    now = time.monotonic()
                    if offset - last_progress_bytes >= 32 * CHUNK or now - last_progress_time >= 15 or offset == SIZE:
                        print(f"Archive: {offset:,}/{SIZE:,} bytes ({100 * offset / SIZE:.1f}%)", file=sys.stderr, flush=True)
                        last_progress_bytes, last_progress_time = offset, now
    print("Verifying published archive MD5...", file=sys.stderr, flush=True)
    if partial.stat().st_size != SIZE or digest(partial, "md5") != MD5:
        raise ValueError("Archive MD5 failed; partial preserved for inspection")
    partial.replace(path)


def extract_selected(archive: Path, rows: list[ManifestRow], data_root: Path,
                     journal: Path, manifest_hashes: dict, smoke_journal: Path,
                     *, archive_source: str = "zenodo") -> dict:
    """Extract from an archive whose published size and MD5 already passed.

    Matching unjournaled files can recover an interruption after the file rename
    but before the journal write. Known journal mismatches remain fatal.
    """
    if archive_source not in SOURCES:
        raise ValueError("Archive source must be zenodo or datashare")
    report = json.loads(journal.read_text(encoding="utf-8")) if journal.exists() else {
        "source_record": "https://zenodo.org/records/6906306", "archive_md5": MD5,
        "archive_size_bytes": SIZE, "records": {}, "errors": [],
        "hash_provenance": "Archive verified against published MD5; member SHA-256 locally observed."}
    if report.get("source_manifest_sha256", manifest_hashes) != manifest_hashes:
        raise ValueError("Source pilot manifests changed")
    report["source_manifest_sha256"] = manifest_hashes
    report["last_chosen_archive_source"] = archive_source
    report["last_chosen_archive_url"] = SOURCES[archive_source]
    report["archive_source_note"] = "Last chosen source only; resumed prefixes or cached archives may come from a previous source. Complete archive verified against published MD5."
    old = json.loads(smoke_journal.read_text(encoding="utf-8")) if smoke_journal.exists() else {}
    if old and old.get("source_manifest_sha256") != manifest_hashes:
        raise ValueError("Smoke provenance refers to different pilot manifests")
    allowed = {row.audio_path for row in rows}
    if set(report["records"]) - allowed or set(old.get("records", {})) - allowed:
        raise ValueError("Journal contains records outside fixed pilot")
    started = time.monotonic()
    try:
        with zipfile.ZipFile(archive) as source:
            for row in rows:
                destination = safe_destination(data_root, row)
                member = source.getinfo("LA/" + row.audio_path)
                if member.is_dir() or not 0 < member.file_size <= 2 * 1024**2:
                    raise ValueError("Unexpected member size")
                previous = report["records"].get(row.audio_path) or old.get("records", {}).get(row.audio_path)
                if destination.exists():
                    observed = inspect_audio(destination)
                    if previous is not None and any(observed[k] != previous[k] for k in observed):
                        raise ValueError(f"Changed journaled audio: {row.audio_path}")
                    # Also recover a completed rename whose journal write was interrupted.
                    # Adoption requires an exact match to the checksum-verified archive.
                    with source.open(member) as audio:
                        if hashlib.sha256(audio.read()).hexdigest() != observed["sha256"]:
                            raise ValueError("Existing audio differs from archive member")
                else:
                    if previous is not None:
                        raise ValueError(f"Journaled audio missing: {row.audio_path}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as target:
                            temporary = Path(target.name)
                            with source.open(member) as audio:
                                shutil.copyfileobj(audio, target, CHUNK)
                        observed = inspect_audio(temporary)
                        temporary.replace(destination)
                    finally:
                        if temporary:
                            temporary.unlink(missing_ok=True)
                report["records"][row.audio_path] = observed
                save_report(report, journal)
    except Exception as exc:
        report["errors"].append(str(exc))
        raise
    finally:
        report["last_extract_seconds"] = time.monotonic() - started
        save_report(report, journal)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "outputs/manifests/asvspoof2019-la-seed42")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/asvspoof2019/LA")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/archives/LA.zip")
    parser.add_argument("--journal", type=Path, default=ROOT / "outputs/pilot-data/prepare_report.json")
    parser.add_argument("--smoke-journal", type=Path, default=ROOT / "outputs/audio-pilot/download_report.json")
    parser.add_argument("--source", choices=tuple(SOURCES), default="zenodo", help="Pinned official archive source")
    parser.add_argument("--execute", action="store_true", help="Download/resume archive and extract pilot; default is plan only")
    args = parser.parse_args()
    rows, hashes = read_selection(args.manifest_dir)
    plan = preflight(args.archive, args.data_root)
    plan.update({"archive_url": SOURCES[args.source], "archive_source": args.source, "archive_size_bytes": SIZE, "archive_md5": MD5,
                 "extract_train": 2000, "extract_dev": 1000, "extract_eval": 0,
                 "source_manifest_sha256": hashes, "execute": args.execute})
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        download_archive(args.archive, source=args.source)
        report = extract_selected(args.archive, rows, args.data_root, args.journal, hashes, args.smoke_journal, archive_source=args.source)
        print(f"Verified {len(report['records'])} fixed pilot files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
