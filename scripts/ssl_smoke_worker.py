"""Single train-recording frozen-encoder smoke; no downloads, labels or fitting.

Run only under an external 180-second, memory-bounded supervisor. The default
invocation prints a plan and imports no optional numerical/model packages.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
SMOKE_ID = "LA_T_1963829"
CONFIG = "configs/localization_ssl_v1.json"
PLAN_AUDIT = "docs/validation/localization_ssl_plan_v1.json"
MODEL_DIRECTORY = ROOT / "data/models/wav2vec2-base-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked_inputs():
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    preflight = json.loads((ROOT / PLAN_AUDIT).read_text(encoding="utf-8"))
    for relative, expected in preflight["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen planning source changed: {relative}")
    # The audit contains descriptive counts, not binary label payloads. Never
    # open selected_labels_v1.json, dev audio or an eval payload in this smoke.
    for relative in ("docs/validation/partial_selection_v1.json",
                     "docs/validation/localization_alignment.json"):
        if sha(ROOT / relative) != config["input_sha256"][relative]:
            raise ValueError(f"Frozen audio-selection input changed: {relative}")
    audit = json.loads((ROOT / "docs/validation/localization_alignment.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "docs/validation/partial_selection_v1.json").read_text(encoding="utf-8"))
    rows = audit["splits"]["train"]["records"]
    row = max(rows, key=lambda item: (item["sample_count"], item["utterance_id"]))
    selected = {item["utterance_id"] for item in selection["splits"]["train"]["selected_records"]}
    if (not audit["training_alignment_ready"] or row["utterance_id"] != SMOKE_ID
            or preflight["smoke_train_id"] != SMOKE_ID or SMOKE_ID not in selected
            or preflight["smoke_sample_count"] != row["sample_count"]):
        raise ValueError("The frozen longest selected train recording changed")
    if not row["candidate_match"] or row["sample_rate"] != 16000 or row["channels"] != 1:
        raise ValueError("Unsupported audited audio header")
    audio_path = (ROOT / row["audio_path"]).resolve()
    if not audio_path.is_relative_to((ROOT / "data/partialspoof/database/train").resolve()):
        raise ValueError("Smoke audio must be inside the train split")
    if sha(audio_path) != row["audio_sha256"]:
        raise ValueError("Smoke audio hash changed")
    hashes = {}
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        path = MODEL_DIRECTORY / name
        expected = (config["checkpoint"]["sha256"] if name == "pytorch_model.bin"
                    else config["checkpoint"]["metadata_sha256"][name])
        if name == "pytorch_model.bin" and path.stat().st_size != config["checkpoint"]["size_bytes"]:
            raise ValueError("Checkpoint size does not match the pinned source")
        hashes[name] = sha(path)
        if hashes[name] != expected:
            raise ValueError(f"Pinned model file changed: {name}")
    return config, row, audio_path, hashes


def run_smoke(output_dir):
    output = Path(output_dir).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()) or output.exists():
        raise ValueError("Require a fresh output directory inside ignored outputs/")
    output.mkdir(parents=True)
    started = time.perf_counter()
    result = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "recording_id": SMOKE_ID, "training_started": False, "labels_opened": False,
              "dev_audio_accessed": False, "eval_accessed": False, "model_downloaded_by_worker": False,
              "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"): sha(__file__)},
              "warnings": []}
    caught = []
    try:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "2"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        sys.path.insert(0, str(ROOT / "src"))
        from audio_deepfake.resources import memory_snapshot

        config, row, audio_path, model_hashes = checked_inputs()
        budgets = config["budgets"]

        def guard(initial=False):
            snapshot = memory_snapshot()
            minimum = budgets["minimum_start_available_ram_bytes"] if initial else budgets["minimum_runtime_available_ram_bytes"]
            if snapshot["system_available_ram_bytes"] is None or snapshot["system_available_ram_bytes"] < minimum:
                raise RuntimeError("SSL smoke available-RAM guard failed")
            if snapshot["peak_rss_bytes"] > budgets["worker_peak_bytes"]:
                raise RuntimeError("SSL smoke worker-memory guard failed")
            if shutil.disk_usage(ROOT).free < budgets["minimum_free_disk_bytes"]:
                raise RuntimeError("SSL smoke disk guard failed")
            return snapshot

        result["resources_before_import"] = guard(initial=True)
        result["model_sha256"] = model_hashes
        result["audio_sha256"] = row["audio_sha256"]
        result["config_sha256"] = sha(ROOT / CONFIG)
        result["geometry_source_sha256"] = sha(ROOT / "src/audio_deepfake/localization_ssl_geometry.py")
        result["environment"] = {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "numpy", "soundfile", "safetensors", "huggingface-hub")}
        version = result["environment"]["torch"].split("+", 1)[0].split(".")
        if tuple(int(part) for part in version[:2]) < (2, 10):
            raise RuntimeError("Restricted binary loading requires torch >= 2.10 (GHSA-63cw-57p8-fm3p)")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import numpy as np
            import soundfile as sf
            import torch
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            from audio_deepfake.localization_ssl_geometry import convolution_geometry, frame_mapping

            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.manual_seed(45)
            if torch.get_num_threads() != 2:
                raise RuntimeError("CPU thread count differs from the frozen plan")
            info = sf.info(audio_path)
            if (info.frames, info.samplerate, info.channels) != (row["sample_count"], 16000, 1):
                raise ValueError("Actual audio header differs from the alignment audit")
            audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if audio.ndim != 1 or len(audio) != row["sample_count"] or rate != 16000 or not np.isfinite(audio).all():
                raise ValueError("Invalid smoke waveform")
            guard()
            result["resources_before_model_load"] = guard(initial=True)
            load_started = time.perf_counter()
            processor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_DIRECTORY), local_files_only=True)
            if not processor.do_normalize or processor.sampling_rate != 16000:
                raise ValueError("Pinned waveform normalization differs")
            model, loading = Wav2Vec2Model.from_pretrained(
                str(MODEL_DIRECTORY), local_files_only=True, weights_only=True,
                use_safetensors=False, attn_implementation="eager", output_loading_info=True,
                torch_dtype=torch.float32)
            unexpected = loading.get("unexpected_keys", [])
            if (loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs")
                    or any(not key.startswith(("quantizer.", "project_hid.", "project_q.")) for key in unexpected)):
                raise ValueError(f"Unexpected encoder loading mismatch: {loading}")
            model.eval()
            model.requires_grad_(False)
            if any(parameter.dtype != torch.float32 or parameter.device.type != "cpu" or parameter.requires_grad
                   for parameter in model.parameters()):
                raise ValueError("Encoder must be frozen float32 CPU")
            receptive, hop = convolution_geometry(model.config.conv_kernel, model.config.conv_stride)
            if (receptive, hop, model.config.hidden_size, model.config.num_hidden_layers) != (400, 320, 768, 12):
                raise ValueError("Actual model geometry differs from the frozen plan")
            result["model_loading_info"] = loading
            result["model_load_seconds"] = time.perf_counter() - load_started
            result["resources_after_load"] = guard()
            extraction_started = time.perf_counter()
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt",
                               padding=False, return_attention_mask=False)
            if set(inputs) != {"input_values"} or tuple(inputs["input_values"].shape) != (1, len(audio)):
                raise ValueError("Waveform must remain unpadded with no attention mask")
            with torch.inference_mode():
                states = model(input_values=inputs["input_values"], output_hidden_states=False,
                               output_attentions=False, return_dict=True).last_hidden_state
            mapping = frame_mapping(len(audio))
            if tuple(states.shape) != (1, mapping["source_frame_count"], 768):
                raise ValueError("Real encoder frame count failed the frozen geometry check")
            features = states[0].cpu().numpy()[mapping["frame_indices"]].copy()
            if (features.dtype != np.float32 or features.shape != (row["coarse_count_20ms"], 768)
                    or not np.isfinite(features).all()):
                raise ValueError("Aligned features failed shape/dtype/finite checks")
            result["extraction_seconds"] = time.perf_counter() - extraction_started
            result["resources_after_extraction"] = guard()
            feature_path = output / f"{SMOKE_ID}.npy"
            with feature_path.open("xb") as stream:
                np.save(stream, features, allow_pickle=False)
            if feature_path.stat().st_size > budgets["feature_cache_bytes"]:
                raise RuntimeError("Feature cache exceeds the storage budget")
            result.update({"status": "complete", "audio_samples": len(audio),
                           "audio_seconds": len(audio) / 16000, "source_frame_count": mapping["source_frame_count"],
                           "aligned_shape": list(features.shape), "dtype": str(features.dtype),
                           "features_sha256": sha(feature_path), "features_bytes": feature_path.stat().st_size,
                           "real_time_factor": result["extraction_seconds"] / (len(audio) / 16000),
                           "offline_whole_recording_context": True})
    except Exception as error:
        result.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        result["warnings"] = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
        result["elapsed_seconds"] = time.perf_counter() - started
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_new(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "plan_only", "recording_id": SMOKE_ID,
                          "training_started": False, "external_supervisor_required": True}))
        return
    if args.output_dir is None:
        parser.error("--execute requires a fresh --output-dir")
    print(json.dumps(run_smoke(args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
