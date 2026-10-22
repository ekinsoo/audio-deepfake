"""Audit selected PartialSpoof audio against published binary labels.

No network, pickle loading, label trimming or training. VAD timestamps remain
diagnostic: their final endpoint is not always the actual WAV duration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import soundfile as sf
from audio_deepfake.partialspoof import labels_for_audio, parse_vad

SELECTION = ROOT / "docs/validation/partial_selection_v1.json"
METADATA = ROOT / "docs/validation/partial_metadata.json"
LABELS = ROOT / "data/partialspoof/selected_labels_v1.json"
DEST = ROOT / "data/partialspoof"
OUTPUT = ROOT / "docs/validation/localization_alignment.json"
PINS = {
    "selection_sha256": "c7efeacc6a19e80557fed89183131ebbc64e476ce5c240b0a02fe151b6cfc374",
    "metadata_sha256": "1dd5f3394f076f5226891d9a89511c11b891c6470ab03b6f1a800dae69c44469",
    "compact_labels_sha256": "9c29d4eb387ca9b0ff2a7f524a03e73dfe592165b6969c6134a6274b184a0a4f",
}


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def native_rttm_inventory(metadata, *, deadline):
    """Inspect verified tar headers; never extract/read any eval member payload.

    gzip traversal necessarily decompresses skipped bytes. This inspection
    records filenames only, with no .npy deserialization or eval interpretation.
    """
    inventories = {}
    for name in ("database_protocols.tar.gz", "database_segment_labels_v1.2.tar.gz", "database_vad.tar.gz"):
        path = DEST / name
        artifact = metadata["artifacts"][name]
        if path.stat().st_size != artifact["size_bytes"] or digest(path, "md5") != artifact["published_md5"]:
            raise ValueError(f"Unverified metadata archive: {name}")
        count, rttm = 0, []
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Alignment audit time budget exhausted")
                count += 1
                # All payloads remain unopened, including any eval RTTM.
                if "rttm" in PurePosixPath(member.name).name.lower():
                    rttm.append(member.name)
        inventories[name] = {"archive_sha256": artifact["sha256"], "verified_md5": artifact["published_md5"],
                             "member_count": count, "rttm_member_names": rttm, "payloads_extracted": 0}
    return inventories


def inspect_record(selected, labels, audio, receipt):
    """Check audio hash/header and exact native-count contract independently of VAD."""
    uid = selected["utterance_id"]
    info = sf.info(audio)
    audio_hash = digest(audio)
    if (audio_hash != receipt["sha256"] or info.frames != receipt["frames"]
            or info.samplerate != 16000 or info.channels != 1):
        raise ValueError(f"Audio receipt/header mismatch: {uid}")
    vad = DEST / "verified_train_dev" / selected["vad_member"]
    if digest(vad) != selected["vad_sha256"]:
        raise ValueError(f"VAD checksum mismatch: {uid}")
    # Validate VAD on its OWN declared endpoint, never claim it covers actual WAV.
    segments = parse_vad(vad.read_text(encoding="utf-8"), duration=selected["vad_duration_seconds"])
    expected = (info.frames + 159) // 160
    try:
        coarse = labels_for_audio(labels, sample_count=info.frames)
        candidate_match, coarse_count, error = True, len(coarse), None
    except ValueError as exc:
        candidate_match, coarse_count, error = False, None, str(exc)
    return {"utterance_id": uid, "sample_count": info.frames, "sample_rate": info.samplerate,
            "channels": info.channels, "audio_sha256": audio_hash, "audio_path": str(audio.relative_to(ROOT)).replace("\\", "/"),
            "native_count_10ms": len(labels), "expected_ceil10ms": expected,
            "candidate_match": candidate_match, "coarse_count_20ms": coarse_count,
            "alignment_error": error, "vad_sha256": selected["vad_sha256"],
            "vad_segment_count": len(segments), "vad_duration_seconds": segments[-1].end,
            "audio_duration_seconds": info.frames / 16000,
            "vad_audio_delta_samples": round(info.frames - segments[-1].end * 16000, 9),
            "native_terminal_bin_end_seconds": len(labels) / 100,
            "last_bin_clipped_to_actual_audio": True,
            "labels_trimmed_or_padded": False}


def load_aligned_labels(record, label_vector, sample_count):
    """Replay checked native -> 20 ms min pooling, without repairing mismatch."""
    if record.get("candidate_match") is not True or record["sample_count"] != sample_count:
        raise ValueError("Audio samples differ from successful alignment audit")
    if len(label_vector) != record["native_count_10ms"]:
        raise ValueError("Label count differs from alignment audit")
    labels = labels_for_audio(label_vector, sample_count=sample_count)
    if len(labels) != record["coarse_count_20ms"]:
        raise ValueError("Coarse label count differs from alignment audit")
    return labels


def run(*, output=OUTPUT):
    started = time.monotonic()
    hashes = {key: digest(path) for key, path in (("selection_sha256", SELECTION), ("metadata_sha256", METADATA), ("compact_labels_sha256", LABELS))}
    if hashes != PINS:
        raise ValueError("Frozen metadata/selection/compact label pin mismatch")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    native = json.loads(LABELS.read_text(encoding="utf-8"))["splits"]
    audit = {"version": 1, "dataset": "PartialSpoof v1.2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             **hashes, "alignment_source_sha256": digest(Path(__file__)),
             "adapter_source_sha256": digest(ROOT / "src/audio_deepfake/partialspoof.py"),
             "training_alignment_ready": False, "native_rttm_metric_ready": False,
             "native_rttm_status": "No native binary RTTM reference has been validated; official metric gate remains pending. Inventory below covers metadata archives only, not the audio archives. Upstream default is database/dev/con_data/rttm_2cls_0sil; presence there remains unchecked.",
             "eval_payload_parsed": False, "splits": {},
             "terminal_policy": "Require native10ms_count=ceil(actual_samples/160); minimum over groups of2, including final partial group; clip final time interval to actual duration. Never trim or pad labels.",
             "vad_policy": "VAD endpoints are checked independently and are diagnostic only; binary labels including nonspeech are retained unchanged."}
    for split in ("train", "dev"):
        receipt_path = DEST / f"audio_{split}_complete.json"
        records = []
        split_audit = {"status": "pending_audio_receipt", "records": records}
        audit["splits"][split] = split_audit
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"] != "complete" or receipt["selection_sha256"] != hashes["selection_sha256"]:
            raise ValueError(f"Unverified audio selection receipt: {split}")
        by_id = {r["utterance_id"]: r for r in receipt["records"]}
        chosen = selection["splits"][split]["selected_records"]
        if len(by_id) != len(chosen) or set(by_id) != {r["utterance_id"] for r in chosen}:
            raise ValueError(f"Audio receipt IDs differ from fixed selection: {split}")
        for selected in chosen:
            uid = selected["utterance_id"]
            audio = DEST / "database" / split / "con_wav" / f"{uid}.wav"
            records.append(inspect_record(selected, native[split][uid], audio, by_id[uid]))
        deltas = [r["vad_audio_delta_samples"] for r in records]
        split_audit.update(status="passed" if all(r["candidate_match"] for r in records) else "failed_binary_alignment",
                           receipt_sha256=digest(receipt_path), record_count=len(records),
                           total_samples=sum(r["sample_count"] for r in records),
                           native_alignment_mismatch_ids=[r["utterance_id"] for r in records if not r["candidate_match"]],
                           vad_exact_audio_count=sum(d == 0 for d in deltas),
                           vad_audio_delta_samples_range=[min(deltas), max(deltas)])
    audit["native_rttm_inventory"] = native_rttm_inventory(metadata, deadline=started + 180)
    audit["training_alignment_ready"] = all(audit["splits"][s]["status"] == "passed" for s in ("train", "dev"))
    audit["status"] = "passed_binary_training_alignment" if audit["training_alignment_ready"] else "incomplete_or_failed"
    audit["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"output": str(args.output), "selection": str(SELECTION), "training": False, "network": False}))
    else:
        result = run(output=args.output)
        print(json.dumps({"status": result["status"], "training_alignment_ready": result["training_alignment_ready"], "native_rttm_metric_ready": False}))
        sys.exit(0 if result["training_alignment_ready"] else 2)
