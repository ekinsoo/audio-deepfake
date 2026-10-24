# Pilot audio access

The fixed pilot contains 2,000 train and 1,000 dev recordings, sampled with seed 42.
The first smoke check uses 50 recordings from each split. Eval is not used here.
Run all commands below from the repository root. Set up the environment,
protocol manifests and metric reference first; see the
[README](../../README.md#prepare-metadata). Downloading the 100-record smoke sample
is optional for the full-pilot workflow. Its existing files are verified when
present, rather than downloaded again.

## Current state

On 2026-10-09, the full 7,640,952,520-byte archive passed its published MD5 check.
All 3,000 selected recordings were extracted and checked against the archive
and audio journal: 2,000 train and 1,000 dev, 16 kHz mono PCM16 FLAC. Saved audio
totals 187,197,762 bytes. Train and dev share no selected speakers or recording
IDs. No eval audio was extracted, decoded or scored.

The [saved data audit](../validation/pilot_data_seed42.json) records source URLs,
checksums, split counts, formats and provenance. The reused download prefix may
contain bytes from both official sources; the completed archive was verified
as a whole. The baseline and narrowband diagnostic have also completed; see
[BASELINE.md](BASELINE.md#measured-pilot-result) for results. The following
commands describe how to prepare the same selection again.

## Source and download decision

The official [Zenodo record](https://zenodo.org/records/6906306) publishes `LA.zip`:

- Size: 7,640,952,520 bytes.
- MD5: `30c98f11d8b2bc21f2c257bfd78bb5c5`.
- License: [ODC-By 1.0](https://opendatacommons.org/licenses/by/1.0/).

On 2026-10-09, an 8 MiB bounded probe returned HTTP 206 with the expected byte
range and total size. It took 6.33 seconds including connection overhead.
Extrapolating that probe gives about 96 minutes for the full archive, but this is
not a reliable estimate of sustained throughput. The probe bytes were discarded.
The full archive was not downloaded as part of this probe.

Fetching 3,000 members separately would involve many slow requests. A verified,
resumable archive is simpler. It includes eval bytes, but the preparation command
extracts only the fixed train/dev pilot. Eval files are never opened for modeling.

## Plan first

```powershell
.venv\Scripts\python.exe scripts/prepare_pilot_data.py
```

The default prints a plan. It checks the fixed manifest sizes and pinned SHA-256
hashes, equal bona fide/spoof counts in each split, split integrity, and free disk.
The archive disk must have space for the remaining archive bytes,
2 GiB of extraction allowance, and a 20 GiB reserve. The data disk also needs the
22 GiB allowance/reserve. All data and journals stay in Git-ignored directories.

## Prepare the full pilot

```powershell
.venv\Scripts\python.exe scripts/prepare_pilot_data.py --execute
```

This is a 7.64 GB download, so run it when the time and network budget fit.
Interrupted bytes remain in `data/archives/LA.zip.part`. Run the same command to
resume. The server must honor the exact range; a full response during resume is
rejected. The archive is promoted to `LA.zip` only after size and published MD5
match. A checksum failure preserves the partial file for inspection.
Download progress prints received bytes and percentage to stderr every 32 MiB
or 15 seconds, whichever comes first between completed chunks.

Extraction checks original PCM16 FLAC, 16 kHz, mono, nonempty finite samples,
and records member SHA-256 hashes in `outputs/pilot-data/prepare_report.json`.
The original 100 files must match their smoke journal and the verified archive.
Changed journaled files stop the run. If extraction was interrupted after a file
rename but before its journal write, a rerun adopts the unjournaled file only when
its SHA-256 exactly matches the member in the verified archive. An unjournaled
file that differs also stops the run. Cached files are never silently replaced.
This recovery requires the published archive size and MD5 checks to pass first;
the command verifies both before calling extraction, including on resumed runs.
Source manifest hashes are stored so a resumed run cannot silently change selection.

The existing `fetch_pilot_audio.py` remains limited to 100 records and 32 MiB.
The archive preparation is a separate explicit workflow, not a raised smoke cap.

## Official DataShare mirror

The [ASVspoof 2019 website](https://www.asvspoof.org/index2019.html) also links the
[Edinburgh DataShare release](https://doi.org/10.7488/ds/2555).
Its [LA archive](https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download)
can be selected explicitly when Zenodo is slow:

```powershell
.venv\Scripts\python.exe scripts/prepare_pilot_data.py --source datashare
.venv\Scripts\python.exe scripts/prepare_pilot_data.py --source datashare --execute
```

Stop any existing downloader before switching sources. The command reuses the
existing contiguous `.part` bytes and accepts only the pinned DataShare bitstream
URL or its known content redirect, both on HTTPS at `datashare.ed.ac.uk`.
The exact HTTP 206 range, total size, and final published MD5 checks stay the same.
A source switch never bypasses complete archive verification.

The preparation journal records the last chosen source and URL. This describes
the latest invocation, not necessarily every downloaded byte: the prefix or a
complete cached archive may have come from the other official source.
Zenodo remains the default; arbitrary download URLs are not accepted.

## Optional parallel archive download

If a single connection is slow, the archive can be downloaded with up to four
connections. Stop any existing archive downloader before using this command.
The two downloaders must never write the same partial archive at the same time.

```powershell
.venv\Scripts\python.exe scripts/download_archive_parallel.py
.venv\Scripts\python.exe scripts/download_archive_parallel.py --execute --workers 4
.venv\Scripts\python.exe scripts/prepare_pilot_data.py --execute
```

For the official DataShare source with smaller 8 MiB chunks:

```powershell
.venv\Scripts\python.exe scripts/download_archive_parallel.py --source datashare --chunk-mib 8
.venv\Scripts\python.exe scripts/download_archive_parallel.py --source datashare --chunk-mib 8 --workers 4 --execute
.venv\Scripts\python.exe scripts/prepare_pilot_data.py --source datashare --execute
```

Stop the single connection downloader before this switch. DataShare uses a
separate `LA.zip.datashare.chunks` cache; the Zenodo chunk cache is left untouched.
The plan pins the selected source URL and chunk size. Keep those options the same
when resuming a cache. `--chunk-mib` accepts 8 or 64; the default remains 64.

The first command only prints a plan. The second reuses the current contiguous
`LA.zip.part` prefix, pins its size and local SHA-256 in
`data/archives/LA.zip.chunks/plan.json`, and downloads the remaining bytes in
64 MiB chunks by default. Each request requires the exact HTTP 206 range and
length on the selected official HTTPS source, with its strict redirect guard.
A completed chunk is flushed and atomically named with its observed
SHA-256; incomplete transfers are retried on the next run. Existing completed
chunks must pass size and SHA-256 checks before reuse. Changed prefixes, unknown
cache entries, and corrupt completed chunks stop the run.

Assembly writes chunks at their fixed offsets, so an interrupted assembly can
replay without appending duplicate bytes. Chunk files remain available until and
after the final archive passes the published size and MD5 checks. Local chunk
hashes provide resume integrity; the published MD5 checks the complete download.
The third command rechecks the archive and extracts only the fixed pilot.
The parallel downloader does not extract audio or start training.

This approach temporarily keeps both the remaining archive bytes and a chunk
copy. Its conservative disk check requires twice the remaining archive size plus
the existing 22 GiB allowance/reserve. Memory stays bounded by four transfer
buffers; entire chunks are never loaded into RAM. Progress reports completed
chunk bytes about every 15 seconds, so active partial chunks are not counted yet.

`run.lock` prevents concurrent instances of the parallel downloader. After a hard
process crash, remove a stale lock only after confirming no downloader is running.
Keep the original partial, plan, and completed chunks together when resuming.
After successful verification, the chunk cache can be removed to reclaim space.
More connections can help throughput, but the gain depends on the server and
network; no speedup is assumed by the workflow.

## Export the selection and run experiments

After preparation reports `Verified 3000 fixed pilot files`, export the exact
combined selection and run the baseline:

```powershell
.venv\Scripts\python.exe scripts/build_pilot_selection.py
.venv\Scripts\python.exe scripts/run_baseline.py --config configs/pilot.toml --output-dir outputs/baseline-pilot-seed42
```

The exporter reads the pinned `pilot_train.csv` and `pilot_dev.csv` in
`outputs/manifests/asvspoof2019-la-seed42`. It writes
`outputs/pilot-data/selection3000.csv` with fixed LF line endings and original
row fields/order. An identical existing selection is reused; different content
is refused. `configs/pilot.toml` pins its SHA-256, the audio root
`data/asvspoof2019/LA`, and `outputs/pilot-data/prepare_report.json`.

The baseline checks all selected files against the journal before fitting.
It requires all 2,000 train and 1,000 dev recordings; missing files do not trigger
a smaller run. It does not use the archive's eval recordings. Keep the archive
and chunk caches separate from model outputs.

Once the full baseline run succeeds, use its saved model and clean scores for
the frozen narrowband experiment:

```powershell
.venv\Scripts\python.exe scripts/run_robustness.py --baseline-run-dir outputs/baseline-pilot-seed42 --output-dir outputs/robustness-pilot-seed42
```

This experiment requires the same pilot configuration and checked audio. It
does not start another download or retrain the model. Both run commands protect
existing output directories; choose new names for repeat runs and pass the
matching baseline directory to robustness. See [BASELINE.md](BASELINE.md) for
outputs and interpretation.
