"""Bounded synthetic conformance checks for the pinned, patched upstream metric.

Run with the separate metric environment. This never reads dataset audio.
Original sources stay in ignored cache; the patch text and hashes are audited.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/partial-metric-env/upstream"
COMMIT = "847347aaec6f65c3c6d2f17c63515b826b94feb3"
SOURCE = {
    "RangeEER.original.py": ("metric/RangeEER.py", "8b9c65c20c89443dc36aa77c4862ce9e89f62a2fb48227096f37d44e22dee194"),
    "sandbox/eval_asvspoof.py": ("project-NN-Pytorch-scripts.202102/sandbox/eval_asvspoof.py", "a7441cac67be01b1cd3322ec4319b850d761803bfd1f4b693fbb1e823f9944ca"),
    "core_scripts/data_io/io_tools.py": ("project-NN-Pytorch-scripts.202102/core_scripts/data_io/io_tools.py", "56ec6fca200a5d1a03a57f0eae16a71f0d71e36f8a052edfb82ae65eaa70fb3c"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch_sources():
    """Explicit network step; aggregate source budget 5 MiB, timeout 30 s/file."""
    total = 0
    for name, (remote, expected) in SOURCE.items():
        path = CACHE / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/nii-yamagishilab/PartialSpoof/{COMMIT}/{remote}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(5 * 1024**2 + 1)
            total += len(data)
            if total > 5 * 1024**2 or sha(data) != expected:
                raise ValueError("Source budget or SHA256 mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if sha(path.read_bytes()) != expected:
            raise ValueError(f"Untrusted source: {name}")


def compatibility_patch(original):
    """Fix two executed defects; leave threshold search/rate definitions intact."""
    start = original.index("def cal_eer_by_scolab(")
    end = original.index("\ndef get_rttm(", start)
    section = original[start:end]
    if len(re.findall(r"\bfrr\b", section)) != 3:
        raise ValueError("Unexpected upstream typo layout")
    patched = original[:start] + re.sub(r"\bfrr\b", "fnr", section) + original[end:]
    old = "        eer_ary = np.loadtxt(save_file).reshape(-1, 4)\n    eer_res = np.unique"
    new = "        eer_ary = np.loadtxt(save_file).reshape(-1, 4)\n    else:\n        eer_ary = np.asarray(eer_res).reshape(-1, 4)\n    eer_res = np.unique"
    if patched.count(old) != 1:
        raise ValueError("Unexpected upstream finalization layout")
    return patched.replace(old, new)


def bounded_search_patch(patched):
    """Explicit compatibility variant; original duration rates are unchanged."""
    marker = "    while(th_left < th_right and abs(fpr_mid-fnr_mid) > args.prec ):\n"
    guard = """    search_states = set()
    search_iterations = 0
    while(th_left < th_right and abs(fpr_mid-fnr_mid) > args.prec ):
        search_state = (th_left, th_right, th_left_per, th_right_per, th_mid, th_mid_per)
        if search_state in search_states or search_iterations >= 128:
            print("BoundedSearch stop={}".format("repeated_full_state" if search_state in search_states else "iteration_cap"))
            break
        search_states.add(search_state)
        search_iterations += 1
"""
    if patched.count(marker) != 1:
        raise ValueError("Unexpected search layout")
    result = patched.replace(marker, guard)
    old = "    if(os.path.exists(save_file)):\n        eer_ary = np.loadtxt(save_file).reshape(-1, 4)\n    else:\n        eer_ary = np.asarray(eer_res).reshape(-1, 4)"
    result = result.replace(old, "    print(\"BoundedSearch iterations={}\".format(search_iterations))\n    eer_ary = np.asarray(eer_res).reshape(-1, 4)")
    result = result.replace("    min_idx = np.argmin(eer_res[:,3])", "    min_idx = np.argmin(eer_res[:,3])\n    print(\"BoundedSearch selected_row={}\".format(eer_res[min_idx].tolist()))")
    return result


def prepare_sources():
    for name, (_, expected) in SOURCE.items():
        if sha((CACHE / name).read_bytes()) != expected:
            raise ValueError(f"Untrusted source: {name}")
    original = (CACHE / "RangeEER.original.py").read_text(encoding="utf-8")
    patched = compatibility_patch(original)
    difference = "".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                         fromfile="RangeEER.original.py", tofile="RangeEER.patched.py"))
    (CACHE / "RangeEER.patched.py").write_bytes(patched.encode())
    (CACHE / "compatibility.patch").write_bytes(difference.encode())
    bounded = bounded_search_patch(patched)
    bounded_diff = "".join(difflib.unified_diff(original.splitlines(True), bounded.splitlines(True),
                         fromfile="RangeEER.original.py", tofile="RangeEER.bounded.py"))
    (CACHE / "RangeEER.bounded.py").write_bytes(bounded.encode())
    (CACHE / "bounded_compatibility.patch").write_bytes(bounded_diff.encode())
    return patched, difference


@contextlib.contextmanager
def upstream_module(directory, patched=True, bounded=False):
    """Load the actual script, including its actual helper and pyannote imports."""
    ref = directory / "reference.rttm"
    dur = directory / "reco2dur"
    ref.write_text("SPEAKER u 1 0.000 0.040 <NA> <NA> spoof <NA> <NA>\n")
    dur.write_text("u 0.080\n")
    argv = sys.argv[:]
    path = sys.path[:]
    sys.argv = ["RangeEER", "--ref_rttm", str(ref), "--reco2dur_file", str(dur),
                "--save_dir", str(directory / "result"), "--scale", "2"]
    sys.path.insert(0, str(CACHE))
    try:
        name = "RangeEER.bounded.py" if bounded else "RangeEER.patched.py" if patched else "RangeEER.original.py"
        namespace = runpy.run_path(str(CACHE / name), run_name="upstream_range_eer")
        # Functions retain the execution globals, which runpy may return as a copy.
        yield namespace["get_detcos_by_th"].__globals__
    finally:
        sys.argv[:] = argv
        sys.path[:] = path


def scolab(scores, labels=None):
    import pandas as pd
    values = {"Uttid": ["u"] * len(scores), "Predict_Score_1": scores}
    if labels is not None:
        values["True_Class"] = labels
    return pd.DataFrame(values)


def set_reference(module, ranges, duration):
    annotation = module["Annotation"]()
    for start, end in ranges:
        annotation[module["Segment"](start, end)] = "spoof"
    module["ref_rttm"] = {"u": annotation} if ranges else {}
    module["reco2dur"] = {"u": str(duration)}


def invoke_main(metric, directory, scores, spoof_bins=2):
    import numpy as np
    score_path = directory / "scores.npy"
    rows = [["u", 1-score, score, 0. if index < spoof_bins else 1.]
            for index, score in enumerate(scores)]
    np.save(score_path, np.array([rows], dtype=object))
    metric["args"].hyp_sco_ali = str(score_path)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        metric["main"]()
    return output.getvalue()


def run_checks():
    import numpy as np
    patched, difference = prepare_sources()
    cases = []
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        directory = Path(temp)
        with upstream_module(directory, patched=False) as original:
            try:
                original["cal_eer_by_scolab"](scolab([.1, .9], [0., 1.]))
            except NameError as exc:
                original_failure = {"type": type(exc).__name__, "message": str(exc)}
            else:
                raise AssertionError("Expected upstream undefined frr failure")
            try:
                invoke_main(original, directory, [.1, .1, .9, .9])
            except UnboundLocalError as exc:
                original_finalization_failure = {"type": type(exc).__name__, "message": str(exc)}
            else:
                raise AssertionError("Expected upstream undefined eer_ary failure")
        with upstream_module(directory) as metric:
            for name, scores, ranges, duration, expected in [
                ("perfect", [.1, .1, .9, .9], [(0., .04)], .08, [0., 0.]),
                ("inverted", [.9, .9, .1, .1], [(0., .04)], .08, [1., 1.]),
                ("mixed", [.1, .9, .1, .9], [(0., .04)], .08, [.5, .5]),
                ("no_spoof", [.9, .9], [], .04, [0., 0.]),
                ("no_spoof_false_alarm", [.1, .9], [], .04, [.5, 0.]),
                ("all_spoof", [.1, .1], [(0., .04)], .04, [0., 0.]),
                ("threshold_ties_predict_spoof", [.5, .5, .5, .5], [(0., .04)], .08, [1., 0.]),
                ("duration_clipping", [.1, .9, .1], [(0., .02)], .05, [1/3, 0.]),
                ("scale2_is_20ms", [.9, .1, .9], [(.02, .04)], .06, [0., 0.]),
            ]:
                set_reference(metric, ranges, duration)
                actual = metric["get_detcos_by_th"](scolab(scores), .5)
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                cases.append({"case": name, "fpr": actual[0], "fnr": actual[1], "status": "passed"})
            segment_eer, _ = metric["cal_eer_by_scolab"](scolab([.1, .1, .9, .9], [0., 0., 1., 1.]))
            assert segment_eer == 0.
            # Exercise the actual search/finalization on an immediate-convergence case.
            set_reference(metric, [(0., .04)], .08)
            for name, scores, expected in [
                ("perfect", [.1, .1, .9, .9], 0.),
                ("inverted", [.9, .9, .1, .1], 1.),
                ("mixed", [.1, .9, .1, .9], .5),
                ("ties", [.5, .5, .5, .5], .5),
            ]:
                output = invoke_main(metric, directory, scores)
                match = re.search(r"RangeEER=([^,]+),", output)
                assert match is not None, output
                actual = float(match.group(1))
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                cases.append({"case": f"actual_main_{name}_no_search_rows", "range_eer": actual, "status": "passed"})
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        probe_dir = Path(temp)
        try:
            probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--probe-search", str(probe_dir)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
            search_probe = {"status": "completed" if probe.returncode == 0 else "failed", "returncode": probe.returncode}
        except subprocess.TimeoutExpired:
            search_probe = {"status": "timeout", "timeout_seconds": 8,
                            "diagnostic": "Upstream discrete percentile search did not terminate on this mixed synthetic fixture."}
        det_path = probe_dir / "result/RangeEER_det"
        if det_path.exists():
            with det_path.open("rb") as stream:
                stream.seek(max(0, det_path.stat().st_size - 4096))
                tail = stream.read().decode().splitlines()[-4:]
            search_probe["final_saved_rows"] = [[float(x) for x in row.split()] for row in tail]
        search_probe["fixture"] = {"score_1": [.1, .9, .8, .2, .3, .4, .7], "spoof_reference_seconds": [[0., .06]], "duration_seconds": .14}
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        directory = Path(temp)
        with upstream_module(directory, bounded=True) as metric:
            set_reference(metric, [(0., .06)], .14)
            output = invoke_main(metric, directory, [.1, .9, .8, .2, .3, .4, .7], spoof_bins=3)
            selected = json.loads(re.search(r"selected_row=(\[[^\n]+\])", output).group(1))
            iterations = int(re.search(r"iterations=(\d+)", output).group(1))
            reason = re.search(r"stop=([^\n]+)", output).group(1)
            # At threshold 0.4, 2/3 spoof bins are missed; 3/4 bona fide bins alarm.
            np.testing.assert_allclose(selected[1:3], [.75, 2/3], atol=1e-12)
            observed = np.loadtxt(directory/"result/RangeEER_det").reshape(-1, 4)
            assert selected[3] <= observed[:, 3].min() + 1e-12
            assert 5 < iterations <= 128  # Repeated threshold alone did not stop search.
            bounded_probe = {"status": "passed", "stop_reason": reason, "iterations": iterations,
                             "selected_row": selected, "range_eer": sum(selected[1:3])/2,
                             "hand_derived_fpr": .75, "hand_derived_fnr": 2/3,
                             "threshold_repeat_count": len(observed) - len(np.unique(observed[:, 0]))}
    return {
        "status": "bounded_patched_upstream_synthetic_conformance_passed", "date": "2026-10-10",
        "upstream_commit": COMMIT, "original_failure": original_failure,
        "original_finalization_failure": original_finalization_failure,
        "nontrivial_search_probe": search_probe,
        "bounded_search_variant": bounded_probe,
        "bounded_source_sha256": sha((CACHE/"RangeEER.bounded.py").read_bytes()),
        "bounded_patch_sha256": sha((CACHE/"bounded_compatibility.patch").read_bytes()),
        "bounded_patch_unified_diff": (CACHE/"bounded_compatibility.patch").read_text(),
        "sources": {name: {"repository_path": value[0], "sha256": value[1], "bytes": (CACHE/name).stat().st_size} for name, value in SOURCE.items()},
        "patched_source_sha256": sha(patched.encode()), "patch_sha256": sha(difference.encode()),
        "patch_unified_diff": difference, "dependency_bridge": "none; full pinned eval_asvspoof and io_tools modules",
        "python": sys.version.split()[0], "packages": dict(sorted(packages.items())),
        "harness_sha256": sha(Path(__file__).read_bytes()),
        "downloaded_dependency_archive_bytes": sum(p.stat().st_size for p in (CACHE.parent/"wheels").glob("*") if p.is_file()),
        "cases": cases, "scale": 2, "prediction_resolution_seconds": .02,
        "limitations": ["Synthetic checks only; no dataset accuracy result.",
            "Single-class duration rates follow upstream zero-denominator convention; EER is not meaningful without both classes.",
            "Discrete threshold ties need not yield equal FPR/FNR. No interpolation added.",
            "The typo-only upstream bisection timed out; the bounded compatibility variant is distinct from an unmodified official evaluator.",
            "Bounded variant uses original closest-observed-gap selection, including its initial row; no interpolated equal-rate point or exhaustive threshold sweep is introduced.",
            "Native official reference RTTM and real audio/label alignment are not verified by these synthetic checks.",
            "The --hyp_rttm branch remains unimplemented upstream.",
            "Trusted locally generated NumPy score files only: upstream uses allow_pickle=True."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch <=5 MiB pinned metric source only")
    parser.add_argument("--write-audit", action="store_true")
    parser.add_argument("--probe-search", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_search:
        prepare_sources()
        directory = args.probe_search
        with upstream_module(directory) as metric:
            set_reference(metric, [(0., .06)], .14)
            # Deliberately unequal class durations and no exact equal-rate step.
            invoke_main(metric, directory, [.1, .9, .8, .2, .3, .4, .7], spoof_bins=3)
        return
    if args.fetch:
        fetch_sources()
    audit = run_checks()
    if args.write_audit:
        output = ROOT / "docs/validation/range_eer_conformance.json"
        output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "cases": len(audit["cases"])}, indent=2))


if __name__ == "__main__":
    main()
