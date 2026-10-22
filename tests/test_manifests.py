import csv
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from audio_deepfake.manifests import (
    ManifestRow, ProtocolError, check_audio_files, parse_protocol,
    sample_pilot, summarize, validate_splits, write_manifest,
)


def row(index, split="train", label="bonafide", attack="-", speaker="LA_0001"):
    prefix = {"train": "T", "dev": "D", "eval": "E"}[split]
    utterance = f"LA_{prefix}_{index:07d}"
    return ManifestRow(split, speaker, utterance, attack, label,
                       f"ASVspoof2019_LA_{split}/flac/{utterance}.flac")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def protocol(self, content, split="train"):
        path = self.root / "protocol.txt"
        path.write_text(content, encoding="utf-8")
        return parse_protocol(path, split)

    def test_parse_bom_whitespace_summary_and_relative_csv(self):
        rows = self.protocol("\ufeffLA_0001  LA_T_1000001 - - bonafide\n\nLA_0002\tLA_T_1000002 - A01 spoof\n")
        self.assertEqual(summarize(rows), {
            "count": 2, "labels": {"bonafide": 1, "spoof": 1},
            "attacks": {"-": 1, "A01": 1}, "speakers": 2,
        })
        output = self.root / "nested" / "manifest.csv"
        write_manifest(rows, output)
        with output.open(encoding="utf-8", newline="") as source:
            exported = list(csv.DictReader(source))
        self.assertEqual(exported[0]["audio_path"], rows[0].audio_path)
        self.assertNotIn(str(self.root), output.read_text(encoding="utf-8"))
        self.assertFalse(Path(exported[0]["audio_path"]).is_absolute())

    def test_invalid_protocol_fields(self):
        invalid = [
            "LA_0001 LA_T_1000001 - bonafide",
            "LA_0001 LA_T_1000001 x - bonafide",
            "LA_0001 LA_D_1000001 - - bonafide",
            "LA_0001 LA_T_1000001 - - real",
            "LA_0001 LA_T_1000001 - A01 bonafide",
            "LA_0001 LA_T_1000001 - - spoof",
            "LA_0001 LA_T_1000001 - A20 spoof",
            "LA_0001 LA_T_1000001 - A07 spoof",
            "../../secret LA_T_1000001 - - bonafide",
            "LA_0001 ../LA_T_1000001 - - bonafide",
        ]
        for content in invalid:
            with self.subTest(content=content), self.assertRaisesRegex(ProtocolError, "Line 1"):
                self.protocol(content)
        with self.assertRaises(ProtocolError):
            self.protocol("\n")
        with self.assertRaises(ProtocolError):
            self.protocol("LA_0001 LA_E_1000001 - A01 spoof", "eval")
        self.assertEqual(self.protocol("LA_0001 LA_E_1000001 - A07 spoof", "eval")[0].attack_id, "A07")

    def test_duplicate_and_speaker_leakage(self):
        line = "LA_0001 LA_T_1000001 - - bonafide\n"
        with self.assertRaisesRegex(ProtocolError, "Duplicate"):
            self.protocol(line + line)
        with self.assertRaisesRegex(ProtocolError, "Duplicate"):
            validate_splits({"train": [row(1), row(1)]})
        with self.assertRaisesRegex(ProtocolError, "Speaker overlap"):
            validate_splits({"train": [row(1)], "dev": [row(2, "dev")]})
        validate_splits({"train": [row(1), row(2)], "dev": [row(3, "dev", speaker="LA_0002")]})
        with self.assertRaisesRegex(ProtocolError, "does not match"):
            validate_splits({"dev": [row(1)]})

    def test_balanced_pilot_deterministic_and_stratified(self):
        rows = [row(index) for index in range(20)]
        rows += [row(100 + index, label="spoof", attack="A01") for index in range(12)]
        rows += [row(200 + index, label="spoof", attack="A02") for index in range(8)]
        result = sample_pilot(rows, 20)
        self.assertEqual(result, sample_pilot(list(reversed(rows)), 20))
        self.assertNotEqual(result, sample_pilot(rows, 20, seed=43))
        self.assertEqual(Counter(item.label for item in result), {"bonafide": 10, "spoof": 10})
        self.assertEqual(Counter(item.attack_id for item in result), {"-": 10, "A01": 6, "A02": 4})
        self.assertEqual(len({item.utterance_id for item in result}), 20)

    def test_pilot_rejects_eval_mixed_invalid_sizes_and_shortfalls(self):
        rows = [row(1), row(2, label="spoof", attack="A01")]
        for size in [0, -1, 1, 3, True, 2.0, 4]:
            with self.subTest(size=size), self.assertRaises(ProtocolError):
                sample_pilot(rows, size)
        with self.assertRaises(ProtocolError):
            sample_pilot([row(1), row(2)], 2)
        with self.assertRaises(ProtocolError):
            sample_pilot([row(1, "eval"), row(2, "eval", "spoof", "A07")], 2)
        with self.assertRaises(ProtocolError):
            sample_pilot([row(1), row(2, "dev", "spoof", "A01")], 2)
        with self.assertRaisesRegex(ProtocolError, "every spoof attack"):
            sample_pilot([row(1), row(2, label="spoof", attack="A01"),
                          row(3, label="spoof", attack="A02")], 2)

    def test_existence_check_and_path_traversal_rejection(self):
        rows = [row(1), row(2)]
        existing = self.root / rows[0].audio_path
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"test")
        self.assertEqual(check_audio_files(rows, self.root), [rows[1].audio_path])
        for unsafe in ["../private.flac", "C:/private.flac", "/private.flac"]:
            invalid = replace(rows[0], audio_path=unsafe)
            with self.subTest(path=unsafe), self.assertRaises(ProtocolError):
                check_audio_files([invalid], self.root)
            with self.subTest(path=unsafe), self.assertRaises(ProtocolError):
                write_manifest([invalid], self.root / "bad.csv")


if __name__ == "__main__":
    unittest.main()
