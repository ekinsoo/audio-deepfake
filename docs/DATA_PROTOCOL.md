# ASVspoof 2019 LA data protocol

Verified on 2026-10-09. This project starts with the Logical Access (LA)
countermeasure task: one real-valued score for each recording. A higher score
means stronger support for bona fide speech.

## Source and local state

Use the official [v1 dataset record](https://zenodo.org/records/6906306),
[challenge page](https://www.asvspoof.org/index2019.html), and
[evaluation plan v0.4](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf).
The record publishes `LA.zip` (about 7.6 GB) with MD5
`30c98f11d8b2bc21f2c257bfd78bb5c5`. The complete 7,640,952,520-byte archive is
now cached and verified against that published MD5. The
[official DataShare release](https://doi.org/10.7488/ds/2555), linked by the
challenge website, supplied the remaining chunks after a source switch.
The existing prefix was reused and may contain bytes from both official sources;
the final complete archive passed the same published checksum. Only the fixed
train/dev pilot was extracted; eval bytes remain inside the unopened archive
members. The [public pilot audit](validation/pilot_data_seed42.json) records an
independent local checksum, selected-file checks and prefix provenance.

Three original CM key files and three README/license text files were
downloaded into ignored `data/metadata/asvspoof2019/` (4,058,293 bytes total).
The small official baseline source archive was also inspected locally; it
does not contain the full CM keys. The official 2019 metric source and LA dev
ASV scores are also cached for evaluation checks; see [metrics](METRICS.md).

The audio cache contains the complete fixed seed-42 pilot: 2,000 train and 1,000
dev recordings, each split balanced between bona fide and spoof. The original
FLAC files total 187,197,762 bytes and 10,434.510625 seconds of audio (about 2.90
hours). All 3,000 files passed 16 kHz mono PCM16 FLAC, nonempty finite-waveform,
and SHA-256 checks against both the extraction journal and verified archive
members. Actual cached paths exactly match the selected manifests. The manifests
also reproduce the original seeded selection from the official protocol keys.

| Selected split | Recordings | Bona fide | Spoof | All speakers | Spoof speakers | Audio bytes | Duration seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2,000 | 1,000 | 1,000 | 20 | 20 | 124,326,860 | 6,911.3770625 |
| Dev | 1,000 | 500 | 500 | 20 | 10 | 62,870,902 | 3,523.1335625 |

Train and dev share no speaker or utterance IDs. Both selected splits contain
all six spoof attack IDs A01-A06. No eval audio was extracted, decoded or scored.
[Pilot audio access](PILOT_DATA.md) documents the preparation workflow.
The existing smoke downloader retains its 100-file cap. Its earlier feature
profile used only 50 train / 50 dev recordings; those timing and memory numbers
do not measure training or full-pilot inference.
No pretrained models were downloaded. Availability and results of later training
runs are tracked in the experiment reports rather than inferred from these files.

The [configuration](../configs/datasets/asvspoof2019_la.json) records the exact
source URLs, byte sizes, SHA256 hashes, and measured split counts. The keys
come directly from individual members of the official Zenodo archive, not
from a repackaged dataset. For example:

```text
https://zenodo.org/api/records/6906306/files/LA.zip/container/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
```

Downloading this member returns only the protocol text. Do not fall back to
downloading the full archive automatically if a member URL fails.

## Format

The original [LA readme](https://zenodo.org/api/records/6906306/files/LA.zip/container/LA/README.LA.txt)
defines five whitespace-separated fields, with no header:

```text
speaker_id utterance_id unused attack_id label
LA_0079 LA_T_1138215 - - bonafide
```

The third field must be `-`. The fourth is `-` for `bonafide`, or an attack ID
for `spoof`. Keep these values as supplied. The readme's numbered field
description is confusing, but its schema line and the actual files agree on
this ordering.

Audio paths are resolved relative to the extracted `LA` directory:
`ASVspoof2019_LA_<split>/flac/<utterance_id>.flac`. The readme specifies
16 kHz, 16-bit FLAC. Protocol availability does not imply audio availability.

## Measured split audit

These counts were computed from the downloaded original CM keys, without
opening audio or running a detector.

| Split | Filename | Rows | Bona fide | Spoof | All speaker IDs | Spoof speaker IDs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | `ASVspoof2019.LA.cm.train.trn.txt` | 25,380 | 2,580 | 22,800 | 20 | 20 |
| Dev | `ASVspoof2019.LA.cm.dev.trl.txt` | 24,844 | 2,548 | 22,296 | 20 | 10 |
| Eval | `ASVspoof2019.LA.cm.eval.trl.txt` | 71,237 | 7,355 | 63,882 | 67 | 48 |

The catalogue's 20/10/48 speaker summary should not be used as an assertion
about all CM speaker IDs. Dev contains 20 bona fide speaker IDs; eval contains
67. The spoof rows use 10 and 48 respectively. Validate the actual CM keys.
All three pairs of splits have zero shared speaker IDs and zero shared
utterance IDs.

Train and dev spoof rows contain `A01` through `A06`. Each attack has 3,800
train and 3,716 dev recordings. Eval contains `A07` through `A19`, each with
4,914 recordings. Different attack IDs are not proof that every synthesis
method is unrelated: the official database paper describes related systems
and attack families. See the [database paper](https://arxiv.org/abs/1911.01601).

| Key | SHA256 |
| --- | --- |
| Train | `65a289e4009400b42bb4f1e47ac60938d93f7f595452d1f220d35defc3de8ab8` |
| Dev | `2fedaebf1ba3ae1a9bb7d3ec916804ac237b4eee063a27fa92e6549f03d129c7` |
| Eval | `84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036` |

## Evaluation rules

The primary 2019 metric is minimum normalized t-DCF; EER is secondary. Scores
must be finite and continuous, with higher values supporting bona fide.
Use the matching official ASV scores and the 2019 cost/prior settings for
t-DCF. CM labels alone are insufficient. Do not silently use another year's
t-DCF variant. The [official Python package](https://www.asvspoof.org/asvspoof2019/tDCF_python_v1.zip)
is linked from the challenge page. The adapter now delegates to its hash-pinned
metric module, and reference parity checks passed, including the supplied
numerical fixture. See [metrics](METRICS.md) for version, hashes, tie handling,
and ASV requirements. This verifies evaluation plumbing; it does not establish
detector performance or make a pilot-subset score an official benchmark result.

Train fits the model; dev selects settings and any operational threshold.
Eval metadata was audited for format and split integrity only. Keep eval
audio and model scores closed until the experiment choices are fixed.
Do not use attack labels as model inputs. Store seeded pilot manifests
separately and call their results pilot results. Report narrowband stress
tests separately from the clean official benchmark.

## Attribution and limitations

The [dataset README](https://zenodo.org/records/6906306/files/README.txt?download=1)
states ODC-By. The [supplied license](https://zenodo.org/records/6906306/files/LICENSE_text.txt?download=1)
requires attribution and preserving notices when conveying the database.
Keep source data outside Git and link readers to the original distribution.
Published outputs should name ASVspoof 2019, link the dataset and license,
and cite the challenge paper. The database license does not itself clear
every individual-content or personality right; see sections 2.4 and 4.

LA is a controlled English speech benchmark based on VCTK. It does not
establish performance on modern public voice generators, arbitrary channels,
or partial edits. ASVspoof 2019 and PartialSpoof share source material, so
cross-testing them is not independent evidence of new-domain generalization.
