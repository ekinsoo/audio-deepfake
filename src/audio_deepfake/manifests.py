"""Read official ASVspoof 2019 LA CM protocols without requiring audio."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


class ProtocolError(ValueError):
    """A protocol or manifest violates the expected benchmark structure."""


@dataclass(frozen=True)
class ManifestRow:
    split: str
    speaker_id: str
    utterance_id: str
    attack_id: str
    label: str
    audio_path: str


_SPLIT_PREFIXES = {"train": "T", "dev": "D", "eval": "E"}
_FIELDS = tuple(ManifestRow.__dataclass_fields__)


def _validate_row(row: ManifestRow) -> None:
    if row.split not in _SPLIT_PREFIXES:
        raise ProtocolError(f"Unknown split: {row.split!r}")
    prefix = _SPLIT_PREFIXES[row.split]
    if not re.fullmatch(rf"LA_{prefix}_[0-9]+", row.utterance_id):
        raise ProtocolError(f"Invalid {row.split} utterance ID: {row.utterance_id!r}")
    if not re.fullmatch(r"[A-Za-z0-9_]+", row.speaker_id):
        raise ProtocolError(f"Invalid speaker ID: {row.speaker_id!r}")
    if row.label not in {"bonafide", "spoof"}:
        raise ProtocolError(f"Invalid label: {row.label!r}")
    if row.label == "bonafide" and row.attack_id != "-":
        raise ProtocolError("Bonafide rows must use attack ID '-'")
    if row.label == "spoof" and not re.fullmatch(r"A(?:0[1-9]|1[0-9])", row.attack_id):
        raise ProtocolError(f"Invalid spoof attack ID: {row.attack_id!r}")
    if row.label == "spoof":
        attack_number = int(row.attack_id[1:])
        if (row.split == "eval" and attack_number < 7) or (row.split != "eval" and attack_number > 6):
            raise ProtocolError(f"Attack {row.attack_id} does not belong to the {row.split} split")
    expected = f"ASVspoof2019_LA_{row.split}/flac/{row.utterance_id}.flac"
    if row.audio_path != expected:
        raise ProtocolError("Audio path must be the canonical relative benchmark path")


def parse_protocol(path: str | Path, split: str) -> list[ManifestRow]:
    """Parse five-field CM lines; reject malformed lines and duplicate IDs.

    Empty lines are ignored. The original protocol order is preserved.
    """
    if split not in _SPLIT_PREFIXES:
        raise ProtocolError(f"Unknown split: {split!r}")
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                if len(fields) != 5:
                    raise ProtocolError("Expected five whitespace-separated fields")
                speaker, utterance, placeholder, attack, label = fields
                if placeholder != "-":
                    raise ProtocolError("Third field must be '-'")
                row = ManifestRow(split, speaker, utterance, attack, label,
                                  f"ASVspoof2019_LA_{split}/flac/{utterance}.flac")
                _validate_row(row)
                if utterance in seen:
                    raise ProtocolError(f"Duplicate utterance ID: {utterance}")
                seen.add(utterance)
                rows.append(row)
            except ProtocolError as exc:
                raise ProtocolError(f"Line {line_number}: {exc}") from exc
    if not rows:
        raise ProtocolError("Protocol contains no records")
    return rows


def validate_splits(splits: Mapping[str, Sequence[ManifestRow]]) -> None:
    """Reject ID reuse and speaker leakage across supplied official splits."""
    utterances: set[str] = set()
    speaker_splits: dict[str, str] = {}
    for split, rows in splits.items():
        if split not in _SPLIT_PREFIXES:
            raise ProtocolError(f"Unknown split: {split!r}")
        for row in rows:
            # Check duplicates first, including an ID copied under another split.
            if row.utterance_id in utterances:
                raise ProtocolError(f"Duplicate utterance ID: {row.utterance_id}")
            _validate_row(row)
            if row.split != split:
                raise ProtocolError("Manifest row split does not match its split group")
            utterances.add(row.utterance_id)
            previous = speaker_splits.setdefault(row.speaker_id, split)
            if previous != split:
                raise ProtocolError(f"Speaker overlap between {previous} and {split}: {row.speaker_id}")


def summarize(rows: Sequence[ManifestRow]) -> dict:
    """Return JSON-compatible record, label, attack and unique-speaker counts."""
    return {
        "count": len(rows),
        "labels": dict(sorted(Counter(row.label for row in rows).items())),
        "attacks": dict(sorted(Counter(row.attack_id for row in rows).items())),
        "speakers": len({row.speaker_id for row in rows}),
    }


def sample_pilot(rows: Sequence[ManifestRow], size: int, seed: int = 42) -> list[ManifestRow]:
    """Sample a single train/dev split with equal label quotas.

    Size must be positive and even. Half the rows are bonafide; spoof rows are
    allocated proportionally by attack using largest remainders. Insufficient
    class quotas or a zero attack quota raise an error, with no reallocation.
    Selection ranks SHA-256(seed, split, ID); output is sorted by ID. Input order
    does not affect the result. Eval records are never eligible.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 2:
        raise ProtocolError("Pilot size must be a positive even integer")
    if not rows or {row.split for row in rows} not in ({"train"}, {"dev"}):
        raise ProtocolError("Pilot sampling requires one train or dev split; eval is excluded")
    validate_splits({rows[0].split: rows})
    if size > len(rows):
        raise ProtocolError(f"Requested {size} pilot records but only {len(rows)} are available")
    groups: dict[str, list[ManifestRow]] = defaultdict(list)
    bonafide: list[ManifestRow] = []
    for row in rows:
        if row.label == "bonafide":
            bonafide.append(row)
        else:
            groups[row.attack_id].append(row)
    quota = size // 2
    spoof_count = sum(len(group) for group in groups.values())
    if len(bonafide) < quota or spoof_count < quota:
        raise ProtocolError(f"Each label needs {quota} records; available bonafide={len(bonafide)}, spoof={spoof_count}")
    keys = sorted(groups)
    allocations = {key: quota * len(groups[key]) // spoof_count for key in keys}
    remaining = quota - sum(allocations.values())
    ranked = sorted(keys, key=lambda key: (-(quota * len(groups[key]) % spoof_count), key))
    for key in ranked[:remaining]:
        allocations[key] += 1
    if any(count == 0 for count in allocations.values()):
        raise ProtocolError("Pilot size is too small to include every spoof attack at proportional quotas")

    def rank(row: ManifestRow) -> tuple[bytes, str]:
        value = f"{seed}\0{row.split}\0{row.utterance_id}".encode("utf-8")
        return hashlib.sha256(value).digest(), row.utterance_id

    selected = sorted(bonafide, key=rank)[:quota]
    for key in keys:
        selected.extend(sorted(groups[key], key=rank)[:allocations[key]])
    return sorted(selected, key=lambda row: row.utterance_id)


def write_manifest(rows: Sequence[ManifestRow], path: str | Path) -> None:
    """Export UTF-8 CSV with relative audio paths and no machine-specific root."""
    for row in rows:
        _validate_row(row)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def check_audio_files(rows: Sequence[ManifestRow], data_root: str | Path) -> list[str]:
    """Return missing relative paths; ``data_root`` contains LA split folders.

    Paths resolving outside the root (including via symlinks) are rejected.
    This checks existence only, not audio integrity or dataset authenticity.
    """
    root = Path(data_root).resolve()
    missing: list[str] = []
    for row in rows:
        _validate_row(row)
        candidate = root.joinpath(*PurePosixPath(row.audio_path).parts).resolve()
        if not candidate.is_relative_to(root):
            raise ProtocolError("Audio path resolves outside the data root")
        if not candidate.is_file():
            missing.append(row.audio_path)
    return missing
