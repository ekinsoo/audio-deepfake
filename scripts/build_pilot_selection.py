"""Export the checksum-pinned train/dev pilot as a canonical combined CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
import hashlib
import io
import json
from pathlib import Path
import sys

from audio_deepfake.manifests import ManifestRow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pilot_data import ROOT, read_selection


def build_selection(manifest_dir: Path, destination: Path) -> dict:
    """Preserve all original fields and row order; refuse changed source/output."""
    rows, source_hashes = read_selection(manifest_dir)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=[field.name for field in fields(ManifestRow)],
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    content = buffer.getvalue().encode("utf-8")
    if destination.is_symlink():
        raise ValueError("Selection destination must not be a symlink")
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError("Selection output already exists with different content")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            target.write(content)
    return {"selection_sha256": hashlib.sha256(content).hexdigest(),
            "source_manifest_sha256": source_hashes,
            "train_count": 2000, "dev_count": 1000, "eval_count": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path,
                        default=ROOT / "outputs/manifests/asvspoof2019-la-seed42")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pilot-data/selection3000.csv")
    args = parser.parse_args()
    try:
        report = build_selection(args.manifest_dir, args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Selection error: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
