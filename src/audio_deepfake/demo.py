"""Local recording scorer. This demo does not establish real-world accuracy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from time import perf_counter
import tomllib

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits

from .baseline import BaselineModel, load_model, score_features
from .features import extract_lfcc

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA256 = "08b77256906a546d5eabbe9fa2cc5bbef0430d7d5dc7b10fc5b9f9e8449a19d0"
PROTOCOL_SHA256 = "806fd09f488449d6e276e4b36b60ddc016441be2b62e7a223d3bc74d7ea6d913"
MAX_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 60


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_demo_model() -> BaselineModel:
    """Load only the audited local seed43 artifact, verifying hash before unpickling.

    Seed43 is fixed for the demo; it is not the best development-set seed.
    Downloaded or user-supplied joblib files are never accepted.
    """
    protocol_path = ROOT / "docs/validation/full_robustness_protocol_v1.json"
    if _digest(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("Frozen model audit changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    pin = protocol["seeds"]["43"]
    model_path = ROOT / "outputs/baseline-full-seed43/model.joblib"
    if pin["model_sha256"] != MODEL_SHA256 or _digest(model_path) != MODEL_SHA256:
        raise ValueError("Model SHA256 does not match the audited seed43 model")
    config_path = ROOT / "configs/full_seed43.toml"
    if _digest(config_path) != pin["config_sha256"]:
        raise ValueError("Training configuration changed")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    for package, expected in pin["packages"].items():
        if version(package) != expected:
            raise ValueError(f"Install the audited {package} version: {expected}")
    for relative in ("src/audio_deepfake/baseline.py", "src/audio_deepfake/features.py"):
        if _digest(ROOT / relative) != protocol["source_sha256"][relative]:
            raise ValueError("Audited inference implementation changed")
    selection = ROOT / config["paths"]["selection_manifest"]
    if _digest(selection) != config["paths"]["selection_sha256"]:
        raise ValueError("Audited training manifest changed")
    with selection.open(encoding="utf-8", newline="") as source:
        training_ids = sorted(row["utterance_id"] for row in csv.DictReader(source)
                              if row["split"] == "train")
    model = load_model(model_path)
    expected_gmm = {"n_components": 32, "seed": 43, "frames_per_class": 50000,
                    "reg_covar": 1e-6, "tol": .001, "max_iter": 50}
    if (asdict(model.lfcc_config) != config["features"]
            or asdict(model.gmm_config) != expected_gmm
            or model.metadata.get("training_ids") != training_ids
            or len(training_ids) != 25380
            or model.metadata.get("versions") != pin["packages"]
            or model.metadata.get("lfcc_config") != asdict(model.lfcc_config)
            or model.metadata.get("gmm_config") != expected_gmm):
        raise ValueError("Model metadata differs from the audited training run")
    for mixture in (model.bonafide, model.spoof):
        if (mixture.covariance_type != "diag" or mixture.n_components != 32
                or mixture.means_.shape != (32, 60)
                or not np.isfinite(mixture.means_).all()):
            raise ValueError("Unexpected model dimensions")
    return model


def score_audio(model: BaselineModel, path_or_bytes: str | Path | bytes) -> dict:
    """Score finite mono 16 kHz WAV/FLAC, without resampling or saving uploads.

    total_seconds includes input reading/decoding and inference, but excludes
    model loading. inference_seconds covers LFCC extraction and GMM scoring.
    """
    started = perf_counter()
    if isinstance(path_or_bytes, bytes):
        payload = path_or_bytes
    else:
        with Path(path_or_bytes).open("rb") as source:
            payload = source.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Audio file exceeds 8 MiB")
    try:
        with sf.SoundFile(io.BytesIO(payload)) as audio:
            if audio.format not in {"WAV", "WAVEX", "FLAC"}:
                raise ValueError("Use a WAV or FLAC file")
            if audio.samplerate != 16000:
                raise ValueError("Expected 16000 Hz audio; no automatic resampling")
            if audio.channels != 1:
                raise ValueError("Expected mono audio")
            if audio.frames == 0:
                raise ValueError("Audio is empty")
            if audio.frames > MAX_SECONDS * 16000:
                raise ValueError("Audio exceeds 60 seconds")
            waveform = audio.read(dtype="float32")
    except sf.LibsndfileError as error:
        raise ValueError("Could not decode WAV/FLAC audio") from error
    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains non-finite samples")
    decoded = perf_counter()
    with threadpool_limits(limits=1):
        score = score_features(model, extract_lfcc(waveform, model.lfcc_config))
    ended = perf_counter()
    return {"score": score, "duration_seconds": len(waveform) / 16000,
            "sample_rate": 16000, "decode_seconds": decoded - started,
            "inference_seconds": ended - decoded, "total_seconds": ended - started,
            "model_seed": model.gmm_config.seed, "model_sha256": MODEL_SHA256,
            "score_direction": "higher is more bonafide; not a probability"}


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audio deepfake detection</title><style nonce="TOKEN">
body{font:17px/1.6 system-ui;max-width:700px;margin:60px auto;padding:0 24px;color:#222}
button,input{font:inherit;margin:12px 8px 12px 0}pre{white-space:pre-wrap;background:#f3f3f3;padding:16px}
</style><h1>Audio deepfake detection</h1>
<p>A local demo of my LFCC–GMM speech detector. Upload one mono 16 kHz WAV or FLAC recording, up to 60 seconds and 8 MiB.</p>
<p>The output is a raw model score: higher scores are more bonafide-like. It is not a probability or a reliable authenticity verdict. Results were measured on ASVspoof 2019 LA development data; other recording conditions may behave differently.</p>
<input id="audio" type="file" accept=".wav,.flac"><button id="run">Score recording</button>
<pre id="result" role="status">No recording selected.</pre>
<p>Processing stays on this computer. Uploaded audio is held in memory and is not saved. Fixed model: seed 43. Processing time excludes loading the model.</p>
<script nonce="TOKEN">
document.getElementById('run').onclick=async()=>{
const f=document.getElementById('audio').files[0],r=document.getElementById('result');
if(!f){r.textContent='Select a WAV or FLAC recording.';return}
if(f.size>8388608){r.textContent='Audio file exceeds 8 MiB.';return}
r.textContent='Processing…';document.getElementById('run').disabled=true;
try{const response=await fetch('/score',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Demo-Token':'TOKEN'},body:f});
const x=await response.json();if(!response.ok)throw Error(x.error);
r.textContent='Raw score: '+x.score.toFixed(4)+'\\nRecording: '+x.duration_seconds.toFixed(2)+' s\\nProcessing: '+(1000*x.total_seconds).toFixed(1)+' ms\\nHigher is more bonafide-like; not a probability.';
}catch(e){r.textContent=e.message}finally{document.getElementById('run').disabled=false}
};</script></html>"""


def make_handler(model: BaselineModel, port: int, token: str):
    """Create a loopback-only, bounded, single-request-at-a-time upload handler."""
    expected_host = f"127.0.0.1:{port}"

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *_):
            pass  # Do not log file names or upload contents.

        def send_payload(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.send_header("Content-Security-Policy", "default-src 'none'; "
                             f"script-src 'nonce-{token}'; style-src 'nonce-{token}'; "
                             "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def error(self, status, message):
            self.send_payload(status, json.dumps({"error": message}).encode())

        def valid_host(self):
            return self.headers.get_all("Host") == [expected_host]

        def do_GET(self):
            if not self.valid_host():
                return self.error(403, "Use the local demo address")
            if self.path != "/":
                return self.error(404, "Not found")
            self.send_payload(200, PAGE.replace("TOKEN", token).encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if (not self.valid_host()
                    or self.headers.get("Origin") != f"http://{expected_host}"
                    or self.headers.get("X-Demo-Token") != token):
                return self.error(403, "Invalid local upload request")
            if self.path != "/score":
                return self.error(404, "Not found")
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
                return self.error(400, "A single Content-Length is required")
            length = int(lengths[0])
            if not 0 < length <= MAX_BYTES:
                return self.error(413, "Upload must be between 1 byte and 8 MiB")
            if self.headers.get("Content-Type") != "application/octet-stream":
                return self.error(415, "Expected a raw audio upload")
            try:
                payload = self.rfile.read(length)
                if len(payload) != length:
                    return self.error(400, "Incomplete upload")
                result = score_audio(model, payload)
            except ValueError as error:
                return self.error(400, str(error))
            except (TimeoutError, OSError):
                return self.error(400, "Could not read audio upload")
            self.send_payload(200, json.dumps(result, allow_nan=False).encode())

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Score one local WAV/FLAC recording")
    score.add_argument("audio", type=Path)
    serve = commands.add_parser("serve", help="Open the local upload demo")
    serve.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        model = load_demo_model()
        if args.command == "score":
            print(json.dumps(score_audio(model, args.audio), indent=2, allow_nan=False))
        else:
            if not 1 <= args.port <= 65535:
                raise ValueError("Port must be between 1 and 65535")
            server = HTTPServer((args.host, args.port), make_handler(model, args.port, secrets.token_hex(24)))
            print(f"Local demo: http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Demo error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
