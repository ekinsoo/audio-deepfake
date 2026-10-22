"""Portable, externally bounded CUDA smoke for one public train recording.

No downloads, labels, development audio, classifier fit or encoder fine-tuning.
Bundle paths and their SHA-256 values are supplied by bundle/manifest.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import warnings

GIB = 1024 ** 3
SMOKE_ID = "LA_T_1963829"
WEIGHT_SHA = "3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21"
METADATA_SHA = {"config.json": "4937977e24d12d1bba70cdce8709c3c04807a8e4ae8ddac4229c48c436ae99ae",
                "preprocessor_config.json": "b225d617c025463b9e157e06afea8b90dc7078fc70b013c533328423e0486b4a"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def memory(pid=None):
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    row = {"system_available_ram_bytes": available}
    if pid is not None:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        for field, key in (("VmRSS:", "rss_bytes"), ("VmHWM:", "peak_rss_bytes")):
            row[key] = next(int(line.split()[1]) * 1024 for line in lines if line.startswith(field))
    return row


def guard(output, *, initial=False, pid=None):
    row = memory(pid)
    if row["system_available_ram_bytes"] < (4 if initial else 2) * GIB:
        raise RuntimeError("Available system RAM boundary reached")
    if row.get("peak_rss_bytes", 0) > 3 * GIB:
        raise RuntimeError("Worker peak RSS exceeds 3 GiB")
    if shutil.disk_usage(output).free < 20 * GIB:
        raise RuntimeError("Free disk below 20 GiB")
    return row


def local_path(bundle, relative):
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle) or not path.is_file():
        raise ValueError("Bundle path missing or escapes bundle directory")
    return path


def verify_bundle(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["smoke_id"] != SMOKE_ID or manifest["sample_count"] != 118059:
        raise ValueError("Smoke selection differs from the frozen longest train recording")
    model = (bundle / manifest["checkpoint_dir"]).resolve()
    if not model.is_relative_to(bundle):
        raise ValueError("Model directory escapes bundle")
    required = {str((model / name).relative_to(bundle)).replace("\\", "/") for name in
                ("config.json", "preprocessor_config.json", "pytorch_model.bin")}
    required.add("localization_ssl_geometry.py")
    if not required.issubset(manifest["files_sha256"]):
        raise ValueError("Required bundle file hashes are absent")
    for relative, expected in manifest["files_sha256"].items():
        if sha(local_path(bundle, relative)) != expected:
            raise ValueError(f"Bundle hash differs: {relative}")
    if sha(bundle / "localization_ssl_geometry.py") != "ddd6e4fe0de0111560281a3f7281220799f16a24554b9b11a4ea27618480cfe5":
        raise ValueError("Frozen geometry helper differs")
    weights = model / "pytorch_model.bin"
    if weights.stat().st_size != 380267417 or sha(weights) != WEIGHT_SHA:
        raise ValueError("Encoder checkpoint differs from the pinned source")
    for name, expected in METADATA_SHA.items():
        if sha(model / name) != expected:
            raise ValueError(f"Pinned encoder metadata differs: {name}")
    audio = local_path(bundle, manifest["audio_file"])
    reference = local_path(bundle, manifest["cpu_reference_file"])
    if sha(audio) != manifest["audio_sha256"] or sha(reference) != manifest["cpu_reference_sha256"]:
        raise ValueError("Audio or prior CPU reference hash differs")
    return manifest, model, audio, reference


def worker(bundle, output):
    start = time.perf_counter()
    result = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "recording_id": SMOKE_ID, "training_started": False, "labels_opened": False,
              "dev_audio_accessed": False, "eval_accessed": False, "downloaded_by_worker": False,
              "official_benchmark": False, "warnings": [], "source_sha256": sha(__file__),
              "manifest_sha256": sha(bundle / "manifest.json")}
    caught = []
    if (output / "result.json").exists():
        raise ValueError("Refuse to overwrite worker result")
    try:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        result["resources_before_import"] = guard(output, initial=True, pid=os.getpid())
        manifest, model_directory, audio_path, reference_path = verify_bundle(bundle)
        result["environment"] = {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "numpy", "soundfile", "safetensors", "huggingface-hub")}
        version = result["environment"]["torch"].split("+", 1)[0].split(".")
        if tuple(int(part) for part in version[:2]) < (2, 10):
            raise RuntimeError("Torch >= 2.10 required for restricted checkpoint loading")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import numpy as np
            import soundfile as sf
            import torch
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            sys.path.insert(0, str(bundle))
            from localization_ssl_geometry import convolution_geometry, frame_mapping

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; stop without automatic CPU fallback")
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.manual_seed(45)
            torch.cuda.manual_seed_all(45)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            device = torch.device("cuda:0")
            free_gpu, total_gpu = torch.cuda.mem_get_info(device)
            result["hardware"] = {"device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                                  "gpu_free_bytes_before_load": free_gpu, "gpu_total_bytes": total_gpu,
                                  "cuda_version": torch.version.cuda, "cpu_threads": 2,
                                  "tf32": False, "dtype": "float32"}
            info = sf.info(audio_path)
            if (info.frames, info.samplerate, info.channels) != (118059, 16000, 1):
                raise ValueError("Actual audio header differs from frozen smoke selection")
            audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if audio.shape != (118059,) or rate != 16000 or not np.isfinite(audio).all():
                raise ValueError("Invalid smoke waveform")
            result["resources_before_model_load"] = guard(output, initial=True, pid=os.getpid())
            loading_start = time.perf_counter()
            processor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_directory), local_files_only=True)
            if not processor.do_normalize or processor.sampling_rate != 16000:
                raise ValueError("Pinned waveform processor differs")
            model, loading = Wav2Vec2Model.from_pretrained(str(model_directory), local_files_only=True,
                weights_only=True, use_safetensors=False, attn_implementation="eager",
                output_loading_info=True, torch_dtype=torch.float32)
            if (loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs")
                    or any(not key.startswith(("quantizer.", "project_hid.", "project_q."))
                           for key in loading.get("unexpected_keys", []))):
                raise ValueError(f"Unexpected encoder loading mismatch: {loading}")
            model.eval().requires_grad_(False)
            model.to(device=device, dtype=torch.float32)
            if any(parameter.requires_grad or parameter.dtype != torch.float32 or parameter.device.type != "cuda"
                   for parameter in model.parameters()):
                raise ValueError("Encoder must be frozen CUDA float32")
            if (*convolution_geometry(model.config.conv_kernel, model.config.conv_stride),
                    model.config.hidden_size, model.config.num_hidden_layers) != (400, 320, 768, 12):
                raise ValueError("Actual encoder geometry differs")
            torch.cuda.synchronize(device)
            result["model_load_seconds"] = time.perf_counter() - loading_start
            result["model_loading_info"] = loading
            result["resources_after_load"] = guard(output, pid=os.getpid())
            torch.cuda.reset_peak_memory_stats(device)
            extraction_start = time.perf_counter()
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False,
                               return_attention_mask=False)
            if set(inputs) != {"input_values"} or tuple(inputs["input_values"].shape) != (1, 118059):
                raise ValueError("Unexpected padding or attention mask")
            with torch.inference_mode():
                states = model(input_values=inputs["input_values"].to(device), output_hidden_states=False,
                               output_attentions=False, return_dict=True).last_hidden_state
            torch.cuda.synchronize(device)
            mapping = frame_mapping(118059)
            if tuple(states.shape) != (1, 368, 768) or mapping["source_frame_count"] != 368:
                raise ValueError("Real encoder frame count differs from frozen geometry")
            features = states[0].cpu().numpy()[mapping["frame_indices"]].copy()
            if features.shape != (369, 768) or features.dtype != np.float32 or not np.isfinite(features).all():
                raise ValueError("Aligned features failed shape/dtype/finite checks")
            result["extraction_seconds"] = time.perf_counter() - extraction_start
            result["resources_after_extraction"] = guard(output, pid=os.getpid())
            result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            reference = np.load(reference_path, allow_pickle=False)
            if reference.shape != features.shape or reference.dtype != np.float32 or not np.isfinite(reference).all():
                raise ValueError("Invalid prior CPU reference")
            difference = features.astype(np.float64) - reference.astype(np.float64)
            result["cpu_reference_comparison"] = {"reference_sha256": manifest["cpu_reference_sha256"],
                "maximum_absolute_difference": float(np.max(np.abs(difference))),
                "root_mean_square_difference": float(np.sqrt(np.mean(difference ** 2))),
                "interpretation": "Numerical diagnostic only; no result-dependent tuning or performance claim"}
            target = output / f"{SMOKE_ID}.npy"
            with target.open("xb") as stream:
                np.save(stream, features, allow_pickle=False)
            result["resources_before_completion"] = guard(output, pid=os.getpid())
            result.update(status="complete", source_frame_count=368, aligned_shape=[369, 768],
                          features_sha256=sha(target), features_bytes=target.stat().st_size,
                          audio_sha256=manifest["audio_sha256"], audio_seconds=118059 / 16000,
                          real_time_factor=result["extraction_seconds"] / (118059 / 16000),
                          offline_whole_recording_context=True)
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["warnings"] = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
        result["elapsed_seconds"] = time.perf_counter() - start
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "result.json", result)


def supervise(bundle, output):
    if sys.platform != "linux":
        raise RuntimeError("The portable supervisor requires Linux /proc")
    if output.exists() or output == bundle or output.is_relative_to(bundle):
        raise ValueError("Require a fresh output directory outside the input bundle")
    guard(output.parent, initial=True)
    verify_bundle(bundle)
    output.mkdir(parents=True)
    status = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "training_started": False, "worker_seconds": 180, "peak_rss_bytes": 0,
              "source_sha256": sha(__file__), "manifest_sha256": sha(bundle / "manifest.json")}
    child = None
    start = time.monotonic()
    try:
        save(output / "status.json", status)
        environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "2"
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--execute",
                "--bundle-dir", str(bundle), "--output-dir", str(output)], env=environment,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
            status["worker_pid"] = child.pid
            while child.poll() is None:
                if time.monotonic() - start >= 180:
                    raise RuntimeError("External 180-second timeout reached")
                try:
                    probe = guard(output, pid=child.pid)
                except (FileNotFoundError, StopIteration):
                    if child.poll() is not None:
                        break
                    raise
                status["peak_rss_bytes"] = max(status["peak_rss_bytes"], probe["peak_rss_bytes"])
                status["last_memory"] = probe
                status["elapsed_seconds"] = time.monotonic() - start
                save(output / "status.json", status)
                time.sleep(.1)
            status["exit_code"] = child.wait()
        if status["exit_code"] != 0:
            raise RuntimeError("Worker failed; inspect stderr.log and result.json")
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        if result["status"] != "complete" or result["training_started"]:
            raise RuntimeError("Real worker completion is missing")
        status["peak_rss_bytes"] = max(status["peak_rss_bytes"],
                                        result["resources_before_completion"]["peak_rss_bytes"])
        if status["peak_rss_bytes"] > 3 * GIB:
            raise RuntimeError("Final worker lifetime peak exceeds 3 GiB")
        if sha(__file__) != status["source_sha256"] or sha(bundle / "manifest.json") != status["manifest_sha256"]:
            raise RuntimeError("Source or manifest changed during smoke")
        if sha(output / f"{SMOKE_ID}.npy") != result["features_sha256"]:
            raise RuntimeError("Feature artifact changed")
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="stopped", error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        status.update(worker_pid=None, elapsed_seconds=time.monotonic() - start,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "plan_only", "training_started": False, "cuda_required": True}))
        return
    if args.bundle_dir is None or args.output_dir is None:
        parser.error("Execution requires --bundle-dir and --output-dir")
    bundle, output = args.bundle_dir.resolve(), args.output_dir.resolve()
    (worker if args.worker else supervise)(bundle, output)


if __name__ == "__main__":
    main()
