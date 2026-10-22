"""Plot a preselected metadata discrepancy; no model predictions or audio."""

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
ID = "CON_D_0003090"  # Fixed smoke ID; inspected before any model scores.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata_path = ROOT / "docs/validation/partial_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    selection_path = ROOT / "docs/validation/partial_selection_v1.json"
    if digest(selection_path) != metadata["selection_sha256"]:
        raise ValueError("Selection changed")
    selection = json.loads(selection_path.read_text())
    row = next(r for r in selection["splits"]["dev"]["selected_records"]
               if r["utterance_id"] == ID)
    label_path = ROOT / metadata["selected_labels_export"]["path"]
    if digest(label_path) != metadata["selected_labels_export"]["sha256"]:
        raise ValueError("Selected native labels changed")
    labels = json.loads(label_path.read_text())["splits"]["dev"][ID]
    vad_path = ROOT / "data/partialspoof/verified_train_dev" / row["vad_member"]
    if digest(vad_path) != row["vad_sha256"]:
        raise ValueError("VAD changed")
    vad = [(float(a), float(b), int(c)) for a, b, c in
           (line.split() for line in vad_path.read_text().splitlines() if line.strip())]
    end = max(len(labels) * .01, vad[-1][1])
    width, origin, span = 860, 190, 620
    x = lambda t: origin + span * t / end
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="300" viewBox="0 0 860 300">',
           '<rect width="860" height="300" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#222">',
           '<text x="28" y="30" font-size="18">Binary labels and nonspeech annotations</text>',
           f'<text x="28" y="52" font-size="12">{escape(ID)} | PartialSpoof v1.2 dev metadata | no audio or model scores</text>']
    tracks = [("Native binary spoof", [(i*.01, (i+1)*.01) for i, value in enumerate(labels) if value == 0]),
              ("VAD spoof (label 0)", [(a, b) for a, b, c in vad if c == 0]),
              ("VAD nonspeech (label 2)", [(a, b) for a, b, c in vad if c == 2])]
    for i, (title, intervals) in enumerate(tracks):
        y = 82 + 48*i
        svg.append(f'<text x="28" y="{y+17}" font-size="12">{title}</text>')
        svg.append(f'<rect x="{origin}" y="{y}" width="{span}" height="24" fill="#f1f1f1"/>')
        for a, b in intervals:
            svg.append(f'<rect x="{x(a):.4f}" y="{y}" width="{x(b)-x(a):.4f}" height="24" fill="#555"/>')
    for i in range(6):
        t = end * i / 5
        svg.append(f'<text x="{x(t):.2f}" y="226" font-size="11" text-anchor="middle">{t:.2f}s</text>')
    svg.extend([f'<text x="28" y="258" font-size="12">Native spoof: {sum(v==0 for v in labels)} bins at 10 ms; VAD spoof: {row["vad_spoof_seconds"]:.3f} s.</text>',
                '<text x="28" y="278" font-size="11">The annotation types differ. Audio duration and terminal-frame alignment still require verification.</text>',
                '</g></svg>'])
    output = ROOT / "docs/figures/partial_label_alignment.svg"
    output.write_text("\n".join(svg)+"\n", encoding="utf-8")
    evidence = {"kind": "metadata diagnostic, not predictions", "utterance_id": ID,
                "selection_sha256": digest(selection_path), "selected_labels_sha256": digest(label_path),
                "vad_sha256": digest(vad_path), "generator_sha256": digest(Path(__file__)),
                "figure_sha256": digest(output), "native_spoof_bins_10ms": sum(v==0 for v in labels),
                "vad_spoof_seconds": row["vad_spoof_seconds"], "audio_duration_verified": False}
    (ROOT / "docs/validation/partial_label_figure.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
