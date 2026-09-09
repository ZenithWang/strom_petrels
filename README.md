# Storm Petrel Acoustic Classification Pipeline

[中文版本](README_zh.md)

This project builds a binary Storm Petrel acoustic detector from BirdNET-style detections. It prepares 3-second audio clips, creates stratified grouped nested-CV folds, trains BirdNET-Analyzer classifiers, and summarizes fold-level predictions and metrics.

The current binary task is:

- Positive class: `storm_petrel`
- Negative class: `not_storm_petrel`
- Negative subtypes are preserved in metadata through `negative_type`, for example `environment_noise` and `other_bird`

## Repository Layout

```text
project_config.py                         Shared path configuration
data_processing/run_pipeline.py           End-to-end data preparation entrypoint
data_processing/prepare_dataset.py
data_processing/build_birdnet_fold_dataset.py
model_training/run_nested_cv.py           Nested-CV BirdNET training and evaluation
model_training/train_deployment_model.py
long_recording_analysis/                  Streaming inference and candidate extraction
comp_chick_adult/                         Adult/chick date-proxy acoustic comparison
chick_screening/                          Dual-model exploratory chick-call screening
results_analysis/summarize_nested_cv.py
workspace/                                Generated clips, folds, model outputs, summaries
```

Notebook files are kept for inspection, but the reproducible workflow is script-based.

Local and server installations use the same repository-relative layout, regardless of where the
repository itself is stored:

```text
<project-root>/project_config.py
<project-root>/data_processing/
<project-root>/model_training/
<project-root>/long_recording_analysis/
<project-root>/comp_chick_adult/
<project-root>/results_analysis/
<project-root>/workspace/
```

`results_analysis/` is needed for the final summary step.

To synchronize the complete workflow while preserving this repository-relative layout, preview first
and then apply. Replace `username`, `server`, and `/path/to/...` throughout this README with values for
your own machine:

```bash
python sync_project.py username@server:/path/to/strom_petrels
python sync_project.py username@server:/path/to/strom_petrels --apply
```

Only configuration, environment documentation, and the five functional code directories are synchronized.
`workspace`, `.env`, databases, audio, models, caches, tests, documents, exploratory notebooks, and local
IDE/agent files stay machine-local. Use `--apply --delete` only to remove stale files inside the five
synchronized code directories.

## Server Quick Start: Adult/Chick Comparison

After synchronizing from the local machine, log in and run:

```bash
ssh username@server
cd /path/to/strom_petrels
conda activate strompetrel

ls workspace/models/deployment/model.tflite
ls workspace/models/deployment/model_Labels.txt

python comp_chick_adult/run_analysis.py \
  --recordings-dir /path/to/recordings \
  --analysis-name adult_proxy_vs_chick_proxy \
  --threshold 0.80 \
  --adult-proxy-end 2021-05-31 \
  --chick-proxy-start 2021-07-23
```

No general long-recording run is required first, and no project/workspace environment variables are
needed when `project_config.py` and `workspace/` are both under the project root. The first run scans and
caches the selected date phases under
`workspace/comp_chick_adult/adult_proxy_vs_chick_proxy/scan/`; later identical runs reuse those scores.
Use `--rescan` only after changing the recordings, deployment model, or date boundaries.
If the printed resolved paths point to an unintended location, remove stale `STORM_PETREL_*` exports
from the current shell or `~/.bashrc`, because shell values override `.env` and defaults.

If the database or audio root is also needed by other workflows, create a machine-local `.env` in the
project root:

```dotenv
STORM_PETREL_DB_PATH=/path/to/storm_petrel_2021_full.db
STORM_PETREL_AUDIO_ROOT=/path/to/recordings
```

The `.env` file is intentionally not synchronized because paths may differ between machines.

## Python Environments

The root `environment.yml` contains the dependencies for data preparation, model training, and notebooks. Create the environment with:

```bash
conda env create -f environment.yml
conda activate strompetrel
python verify_environment.py
```

Update an existing environment with:

```bash
conda env update -n strompetrel -f environment.yml
```

The environment uses Python 3.11 and the BirdNET-Analyzer 2.4.0 training extra. Acoustic comparison
plots additionally use `matplotlib`; it is included explicitly in `environment.yml`.

### PyCharm

Open `File | Settings | Project | Python Interpreter`, select `Add Interpreter | Add Local Interpreter | Conda Environment | Existing environment`, and choose the interpreter from the environment you created. Typical locations are:

```powershell
<conda-root>\envs\strompetrel\python.exe
```

On Linux or macOS, the equivalent is `<conda-root>/envs/strompetrel/bin/python`. The exact Conda root
is installation-specific; `conda info --base` prints it.

PyCharm will load the shared `Verify Environment`, `Data Pipeline`, `Nested CV Training`, and `Summarize Results` run configurations from `.run/`. Before running the data pipeline, add the real `STORM_PETREL_DB_PATH` and `STORM_PETREL_AUDIO_ROOT` values to the `Data Pipeline` configuration; personal data paths are intentionally not stored in the shared files.

## Path Configuration

All scripts read paths from the root-level `project_config.py`. Local and server installations keep the
same repository-relative structure, while their absolute locations can differ. When the code folders and
`workspace/` are under the project root, no project/workspace environment variables are required; use
overrides only for data or output stored elsewhere.

Runtime path overrides can be stored once in a root-level `.env` file instead of exporting them for every
shell session:

```bash
cp .env.example .env
# Edit .env and replace the example absolute paths.
```

`project_config.py` loads `.env` automatically. Existing shell variables take precedence, and `.env` is
ignored by Git. Set `STORM_PETREL_ENV_FILE` only when a different env-file location is required.

Important variables:

```bash
cd /path/to/strom_petrels
export STORM_PETREL_PROJECT_ROOT="$PWD"
export STORM_PETREL_DB_PATH=/path/to/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/path/to/recordings
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_MODEL_ROOT="$STORM_PETREL_WORKSPACE_DIR/models/nested_cv_balanced"
export STORM_PETREL_DEPLOYMENT_MODEL_DIR="$STORM_PETREL_WORKSPACE_DIR/models/deployment"
export STORM_PETREL_LONG_RECORDING_ANALYSIS_DIR="$STORM_PETREL_WORKSPACE_DIR/long_recording_analysis"
export STORM_PETREL_COMP_CHICK_ADULT_DIR="$STORM_PETREL_WORKSPACE_DIR/comp_chick_adult"
export STORM_PETREL_CHICK_SCREENING_DIR="$STORM_PETREL_WORKSPACE_DIR/chick_screening"
```

Meaning:

- `STORM_PETREL_PROJECT_ROOT`: repository root on the current machine
- `STORM_PETREL_DB_PATH`: SQLite database with BirdNET-style detections
- `STORM_PETREL_AUDIO_ROOT`: folder containing the original full recordings; files can be nested in subfolders
- `STORM_PETREL_WORKSPACE_DIR`: where generated clips, manifests, folds, and fold datasets are written
- `STORM_PETREL_MODEL_ROOT`: where trained models, predictions, and metric summaries are written
- `STORM_PETREL_DEPLOYMENT_MODEL_DIR`: all-clips deployment model output
- `STORM_PETREL_LONG_RECORDING_ANALYSIS_DIR`: long-recording scores, detections, and candidate WAV outputs
- `STORM_PETREL_COMP_CHICK_ADULT_DIR`: adult/chick proxy tables, figures, reports, and representative audio
- `STORM_PETREL_CHICK_SCREENING_DIR`: date-specific models, dual-model scores, and chick-screening analysis

If a variable is not set, defaults are resolved relative to the project folder:

```text
DB_PATH       = data_processing/storm_petrel_2021_full.db
AUDIO_ROOT    = download/data
WORKSPACE_DIR = workspace
MODEL_ROOT    = workspace/models/nested_cv
DEPLOYMENT_MODEL_DIR = workspace/models/deployment
LONG_RECORDING_ANALYSIS_DIR = workspace/long_recording_analysis
COMP_CHICK_ADULT_DIR = workspace/comp_chick_adult
CHICK_SCREENING_DIR = workspace/chick_screening
```

The training script prints the resolved project, dataset, and model paths before validating the fold directories.

## Expected Input Data

Before data preparation, arrange the required inputs in locations of your choice and configure them
through `.env` or environment variables:

```text
<project-root>/                         Project code
<database-path>                         Detection database
<recordings-root>/                      Full source recordings
```

The audio root should contain files referenced by the database recording names. Files are matched by filename or stem, including inside nested directories.

## Complete End-to-End Workflow

Run every command from the project root. The core reproducible pipeline and the two biological research
tracks share configuration, but they produce separate outputs.

### 1. One-time environment and path setup

```bash
conda env create -f environment.yml
conda activate strompetrel
cp .env.example .env
# Edit .env with the real database, recording, project, and workspace paths.
python verify_environment.py
```

Always run `python verify_environment.py` after changing or updating the environment.

### 2. Core dataset, nested-CV, and paper metrics

```bash
python data_processing/run_pipeline.py
python model_training/run_nested_cv.py \
  --outer-folds 01 02 03 04 05 \
  --epochs 50 \
  --threads 8
python results_analysis/summarize_nested_cv.py
```

This is the complete reproducible path for the 2,000-clip binary dataset, grouped nested-CV models,
out-of-fold predictions, error summary, and the recording-clustered bootstrap used in Table 5.3.

### 3. Deployment model and long-recording detections

```bash
python model_training/train_deployment_model.py --epochs 50 --threads 8
python long_recording_analysis/run_pipeline.py \
  /path/to/recordings \
  --run-name deployment_scan \
  --threshold 0.80 \
  --batch-size 32
```

### 4. Independent adult/chick date-proxy comparison

```bash
python comp_chick_adult/run_analysis.py \
  --recordings-dir /path/to/recordings \
  --analysis-name adult_proxy_vs_chick_proxy \
  --threshold 0.80 \
  --adult-proxy-end 2021-05-31 \
  --chick-proxy-start 2021-07-23 \
  --bootstrap-iterations 5000 \
  --seed 2021
```

This track uses the deployment model. It performs its own date-filtered scan and cache, so the general
long-recording command in step 3 is not a prerequisite.

### 5. Exploratory dual-model chick-call screening

```bash
python chick_screening/run_pipeline.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00" \
  --recordings-dir /path/to/chick_proxy_recordings \
  --epochs 50 \
  --threads 8 \
  --adult-threshold 0.80 \
  --chick-threshold 0.80
```

This separate exploratory track builds date-specific training datasets, trains adult-period and
chick-period models, scores the same recordings with both models, and analyzes their discrepancies.
Its date-proxy labels do not establish the biological age of the caller.

### Execution order and restart rules

```text
environment/config
    -> data preparation
        -> grouped nested-CV training -> results summary/Table 5.3
        -> deployment-model training -> long-recording scan
                                  \-> adult/chick proxy comparison
        -> date-specific datasets -> two date models -> dual scan -> chick screening analysis
```

- Run data preparation again only when the database, source recordings, selection constraints, or fold logic changes.
- Use `--evaluate-only` when final nested-CV models exist and only outer predictions/metrics need rebuilding.
- Change a long-recording threshold by rerunning candidate extraction; model inference does not need to repeat.
- Use `--rescan` for adult/chick analysis only when recordings, model, or proxy-date boundaries changed.
- `--force` options replace existing outputs. Read the printed target path before using them.

## Data Processing

Run the data preparation pipeline from the project root.

Windows example:

```powershell
$env:STORM_PETREL_PROJECT_ROOT="C:\path\to\strom_petrels"
$env:STORM_PETREL_DB_PATH="C:\path\to\storm_petrel_2021_full.db"
$env:STORM_PETREL_AUDIO_ROOT="C:\path\to\recordings"
$env:STORM_PETREL_WORKSPACE_DIR="$env:STORM_PETREL_PROJECT_ROOT\workspace"
$env:STORM_PETREL_MODEL_ROOT="$env:STORM_PETREL_WORKSPACE_DIR\models\nested_cv_balanced"

python data_processing\run_pipeline.py
```

Linux/server example:

```bash
cd /path/to/strom_petrels

export STORM_PETREL_PROJECT_ROOT="$PWD"
export STORM_PETREL_DB_PATH=/path/to/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/path/to/recordings
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_MODEL_ROOT="$STORM_PETREL_WORKSPACE_DIR/models/nested_cv_balanced"

python data_processing/run_pipeline.py
```

This produces or rebuilds:

```text
$STORM_PETREL_WORKSPACE_DIR/download_files.csv
$STORM_PETREL_WORKSPACE_DIR/clips/storm_petrel/
$STORM_PETREL_WORKSPACE_DIR/clips/not_storm_petrel/
$STORM_PETREL_WORKSPACE_DIR/binary_labels.csv
$STORM_PETREL_WORKSPACE_DIR/nested_cv_folds.csv
$STORM_PETREL_WORKSPACE_DIR/fold_balance_report.csv
$STORM_PETREL_WORKSPACE_DIR/nested_cv_datasets/outer_01 ... outer_05
```

The selected dataset contains 1,000 Storm Petrel clips and 1,000 negative clips. The negative class contains 500 environment-noise and 500 other-bird clips. Selection maximizes positive confidence subject to these constraints:

- every selected three-second window from the same recording is non-overlapping
- each `source_file` contributes the same number of positive and negative clips
- each `source_file` contributes at most 73 clips per class
- each of the four configured negative labels contributes exactly 250 clips

The 73-pair source limit ensures that three maximum-sized source groups still fit inside the largest permitted outer test fold (`3 * 73 * 2 = 438`).

Fold assignment uses deterministic mixed-integer optimization:

- `source_file` is used as the grouping key
- the same `source_file` will not appear in both train and held-out test for the same outer fold
- every outer and inner train, validation, and test partition contains exactly 50% positive clips
- held-out fold sizes must be within 10% of their ideal size
- held-out sample counts are optimized toward equal fold distributions

Candidate availability and fold feasibility are checked before old generated clips are removed. If 1,000 balanced non-overlapping pairs cannot be selected, or grouped folds cannot meet the constraints, the pipeline stops with an error. Successful runs write the statistics to `fold_balance_report.csv`.

## Rebuilding Folds From Existing Clips

If clips and `binary_labels.csv` already exist but original full recordings are not available locally, you can rebuild only the folds and BirdNET folder layout:

```bash
python -c "import csv, sys; from pathlib import Path; sys.path.insert(0, str(Path('data_processing').resolve())); import prepare_dataset, build_birdnet_fold_dataset; rows=list(csv.DictReader(open(prepare_dataset.BINARY_LABELS_PATH, newline='', encoding='utf-8'))); prepare_dataset.write_folds(rows); build_birdnet_fold_dataset.main()"
```

This is useful for validating split logic, but every source in the existing labels must already contain equal positive and negative counts. Full data preparation from raw recordings still requires `STORM_PETREL_AUDIO_ROOT`.

## Model Training

Run full nested-CV training after data processing has created `nested_cv_datasets`.

Windows example:

```powershell
$env:STORM_PETREL_PROJECT_ROOT="C:\path\to\strom_petrels"
$env:STORM_PETREL_WORKSPACE_DIR="$env:STORM_PETREL_PROJECT_ROOT\workspace"
$env:STORM_PETREL_MODEL_ROOT="$env:STORM_PETREL_WORKSPACE_DIR\models\nested_cv_balanced"
python model_training\run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
```

Linux/server example:

```bash
cd /path/to/strom_petrels
export STORM_PETREL_PROJECT_ROOT="$PWD"
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_MODEL_ROOT="$STORM_PETREL_WORKSPACE_DIR/models/nested_cv_balanced"

python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
```

Use a new model directory for the balanced dataset. Do not point this run at the legacy `nested_cv_stratified` directory: existing `metrics.csv` files can skip folds, and existing `.npz` caches contain embeddings from the earlier dataset.

Important options:

```text
--outer-folds 01 02 03 04 05   Which outer folds to run
--epochs 50                    BirdNET training epochs
--threads 8                    BirdNET training threads
--force                        Re-run a fold even if metrics.csv already exists
--evaluate-only                Recompute predictions/metrics from existing final models without retraining
```

Training outputs are written under `STORM_PETREL_MODEL_ROOT`:

```text
outer_01/inner_search/.../model.tflite
outer_01/final_model/model.tflite
outer_01/selected_hyperparameters.csv
outer_01/predictions.csv
outer_01/metrics.csv
...
outer_05/...
```

The script also sets `NUMBA_CACHE_DIR`, `TMP`, and `TEMP` under the workspace to avoid Windows temp/cache issues during audio loading and BirdNET training.

## Deployment Model and Long-recording Detection

Train one deployment classifier from all 1,000 positive and 1,000 negative clips with the selected numeric best hyperparameters:

```bash
python model_training/train_deployment_model.py --epochs 50 --threads 8
```

The fixed settings are learning rate `0.0001`, 256 hidden units, dropout `0.3`, batch size 32, and a random 20% internal validation split. `STORM_PETREL_DEPLOYMENT_MODEL_DIR` receives `model.tflite`, its labels and parameter/sample-count files, and `training_config.json`. Existing output is only replaced with `--force`.

Recursively scan long WAV/FLAC recordings in non-overlapping three-second windows and extract
high-confidence candidates:

```bash
python long_recording_analysis/run_pipeline.py \
  /path/to/long_recordings \
  --run-name all_recordings_step3 \
  --threshold 0.80
```

Both the default window and step are three seconds. A final tail shorter than three seconds is ignored.
Every thresholded window becomes one
strictly three-second, 48 kHz mono candidate; adjacent hits are not merged and no context is added.
Outer-CV evaluation follows the same complete-window policy: a test clip shorter than three seconds
raises an error instead of being padded, because it indicates a damaged or incomplete prepared dataset.
`detections.csv` retains its established columns with `window_count=1`. Runs produced by the earlier
1.5-second stepping workflow are not converted automatically: use a new run name and rescan.

Changing the threshold does not repeat inference. Re-run extraction against the saved scores:

```bash
python long_recording_analysis/extract_candidates.py \
  workspace/long_recording_analysis/all_recordings_step3/all_windows.csv \
  --threshold 0.80 \
  --force
```

## Adult-proxy versus chick-proxy comparison

The `comp_chick_adult` workflow runs independently from general long-recording detection and compares
high-confidence model hits using date proxies:
`adult_proxy` is on or before 2021-05-31, `chick_proxy` is on or after 2021-07-23, and the intervening
incubation period is excluded. It can scan the raw recordings directly:

```bash
python comp_chick_adult/run_analysis.py \
  --recordings-dir /path/to/recordings \
  --analysis-name adult_proxy_vs_chick_proxy \
  --threshold 0.80 \
  --adult-proxy-end 2021-05-31 \
  --chick-proxy-start 2021-07-23
```

Before inference, filenames are date-filtered so only adult- and chick-proxy recordings are scanned.
Transition-period and unparseable filenames are
recorded under `scan/excluded_recordings.csv` without consuming model inference. Non-overlapping
three-second scores and their configuration are cached under `scan/`; later runs reuse a matching cache
and automatically refresh derived analysis outputs. If the recordings, model, or date boundaries change,
the script stops and asks for an explicit `--rescan`.

Existing scores remain supported without invoking the independent scanner:

```bash
python comp_chick_adult/run_analysis.py \
  workspace/long_recording_analysis/RUN_NAME/all_windows.csv \
  --analysis-name adult_proxy_vs_chick_proxy \
  --threshold 0.80
```

One or more CSVs may be supplied, and a directory input is searched recursively for `all_windows.csv`.
The script deduplicates by recording and start time, reads complete windows from the original recording,
converts them to 48 kHz mono, and never lowers the requested threshold automatically.

Outputs are written below
`workspace/comp_chick_adult/adult_proxy_vs_chick_proxy/`: `scan/`, `analysis_windows.csv`,
`recording_summary.csv`, `feature_comparison.csv`, `threshold_yield.csv`, `excluded_windows.csv`,
`analysis_report.md`, `representative_audio/`, and `figures/`. Formal feature analysis uses only the
configured threshold (default 0.80); the yield table separately reports 0.20, 0.50, and 0.80. Features
are summarized per recording before Mann–Whitney U, Cliff's delta, recording-level bootstrap intervals,
and BH-FDR. With fewer than two independent recordings in either phase, only descriptive statistics are
reported.

The model score is an uncalibrated sigmoid output: 0.80 is a high-confidence filter, not an 80%
biological probability. The phase labels are date proxies rather than confirmed caller identities, so
the report describes associations and does not make a causal breeding-stage claim. The default boundary
dates are literature proxies informed by [UK bioacoustic research](https://pmc.ncbi.nlm.nih.gov/articles/PMC12336414/)
and the [Skokholm breeding research archive](https://ora.ox.ac.uk/objects/uuid%3A24973c21-ab78-42f1-a3c1-a7665b776bbf);
override them when site-specific 2021 field records are available.

## Result Summary

After training, summarize all completed outer folds:

```bash
python results_analysis/summarize_nested_cv.py
```

The summary script reads from `STORM_PETREL_MODEL_ROOT` and writes:

```text
$STORM_PETREL_MODEL_ROOT/nested_cv_summary.csv
$STORM_PETREL_MODEL_ROOT/negative_type_error_summary.csv
$STORM_PETREL_MODEL_ROOT/clustered_bootstrap_ci.csv
$STORM_PETREL_MODEL_ROOT/model_metrics.tex
```

`nested_cv_summary.csv` contains fold-level and mean/std metrics:

- precision at 0.5
- recall at 0.5
- F1 at 0.5
- optimized-threshold precision/recall/F1
- AUPRC
- AUROC

`negative_type_error_summary.csv` summarizes false positives by negative subtype.
`clustered_bootstrap_ci.csv` and `model_metrics.tex` reproduce the pooled estimates and 95% intervals
in Table 5.3 by resampling whole source recordings 5,000 times with seed 2021. This formal bootstrap
requires predictions from all five outer folds. For an intentionally partial smoke test, run the summary
with `--skip-clustered-bootstrap`.

## Exploratory Dual-model Chick Screening

The `chick_screening` workflow is separate from the deployment-model adult/chick comparison above. It
trains one model on the adult-proxy period and another on the chick-proxy period, applies both models to
the same recordings, and divides windows into four score quadrants.

Run the complete workflow:

```bash
python chick_screening/run_pipeline.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00" \
  --recordings-dir /path/to/chick_proxy_recordings \
  --epochs 50 --threads 8
```

Run its stages independently:

```bash
# Prepare both date-specific datasets.
python chick_screening/prepare_date_dataset.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00"

# Train both date-specific models.
python chick_screening/train_date_models.py --epochs 50 --threads 8

# Score recordings with both models.
python chick_screening/dual_model_scan.py /path/to/recordings \
  --output-csv workspace/chick_screening/dual_scores.csv \
  --batch-size 32

# Reanalyze saved scores without rescanning.
python chick_screening/analyze_discrepancies.py \
  --scores-csv workspace/chick_screening/dual_scores.csv \
  --adult-threshold 0.80 \
  --chick-threshold 0.80 \
  --bootstrap-iterations 5000 \
  --seed 2021
```

Use `--adult-only` or `--chick-only` on preparation/training commands when only one period needs to be
rebuilt. The detailed output tree and interpretation cautions are documented in
`chick_screening/README.md`.

## Command Reference: Run Each Function Separately

The table below lists every production command-line entry point. Use `python <script> --help` for the
authoritative option list installed with the current code.

| Function | Command | Primary output |
|---|---|---|
| Verify Python and packages | `python verify_environment.py` | terminal pass/fail report |
| Preview/apply server sync | `python sync_project.py USER@HOST:PATH [--apply] [--delete]` | synchronized source tree |
| Select clips, extract WAVs, build folds and datasets | `python data_processing/run_pipeline.py` | `workspace/clips`, manifests, folds, datasets |
| Select/extract clips and write folds only | `python data_processing/prepare_dataset.py` | clips and fold CSV files |
| Rebuild BirdNET directory layout from fold CSV | `python data_processing/build_birdnet_fold_dataset.py` | `workspace/nested_cv_datasets` |
| Train/evaluate nested CV | `python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05` | fold models, predictions, metrics |
| Re-evaluate existing nested-CV models | `python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --evaluate-only --force` | refreshed predictions and metrics |
| Train all-clips deployment model | `python model_training/train_deployment_model.py` | `workspace/models/deployment` |
| Scan and extract long-recording candidates | `python long_recording_analysis/run_pipeline.py INPUT_DIR` | named long-recording run directory |
| Run long-recording inference only | `python long_recording_analysis/scan_long_recordings.py INPUT_DIR --output-csv FILE` | `all_windows.csv` |
| Apply/reapply threshold and export WAVs | `python long_recording_analysis/extract_candidates.py ALL_WINDOWS_CSV` | detections CSV and candidate WAVs |
| Analyze adult/chick proxies from raw recordings | `python comp_chick_adult/run_analysis.py --recordings-dir INPUT_DIR` | cached scan, tables, figures, report |
| Analyze adult/chick proxies from saved scores | `python comp_chick_adult/run_analysis.py ALL_WINDOWS_CSV [...]` | tables, figures, report |
| Run all chick-screening stages | `python chick_screening/run_pipeline.py --recordings-dir INPUT_DIR` | date datasets, models, scores, analysis |
| Prepare chick-screening datasets | `python chick_screening/prepare_date_dataset.py` | adult/chick clip datasets |
| Train chick-screening models | `python chick_screening/train_date_models.py` | adult/chick TFLite models |
| Run dual-model inference | `python chick_screening/dual_model_scan.py INPUT_DIR` | `dual_scores.csv` |
| Analyze dual-model scores | `python chick_screening/analyze_discrepancies.py` | quadrant tables, figures, report |
| Summarize nested CV and Table 5.3 | `python results_analysis/summarize_nested_cv.py` | summary CSVs and LaTeX table |

## Validation, Smoke Tests, and Troubleshooting

Run the complete lightweight automated test suite:

```bash
python -m pytest -q
```

Useful smoke tests that do not claim final model quality:

```bash
python model_training/run_nested_cv.py --outer-folds 04 --epochs 5 --threads 4
python results_analysis/summarize_nested_cv.py --skip-clustered-bootstrap
python chick_screening/run_pipeline.py --epochs 5 --threads 4 --skip-scan --skip-analysis
```

Check these conditions first when a run fails:

- The resolved paths printed at startup point to the intended database, recordings, workspace, and model directory.
- `python verify_environment.py` passes and reports Python 3.11 and BirdNET-Analyzer 2.4.0.
- `STORM_PETREL_MODEL_ROOT` is `models/nested_cv_balanced`, not a legacy model/cache directory.
- All five `outer_*/predictions.csv` files exist before running the formal clustered bootstrap.
- Prepared CV clips are complete three-second, 48 kHz files; evaluation deliberately refuses to pad short clips.
- A changed long-recording model, inventory, or proxy-date boundary requires a new scan or explicit `--rescan`.
- On Windows, Numba and temporary paths should resolve under `workspace`; the scripts configure these automatically.

## Generated Files and Git

Do not commit databases, recordings, extracted clips, caches, trained models, or anything under
`workspace/`. Commit only source, lightweight configuration, notebooks intended for inspection, README
files, and small project documentation. The `.gitignore` already covers the main generated formats.

## Typical Server Run

A full server run usually looks like this:

```bash
cd /path/to/strom_petrels

export STORM_PETREL_PROJECT_ROOT="$PWD"
export STORM_PETREL_DB_PATH=/path/to/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/path/to/recordings
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_MODEL_ROOT="$STORM_PETREL_WORKSPACE_DIR/models/nested_cv_balanced"

python data_processing/run_pipeline.py
python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
python results_analysis/summarize_nested_cv.py
```

For a quick single-fold smoke run on the server:

```bash
python model_training/run_nested_cv.py --outer-folds 04 --epochs 5 --threads 4
python results_analysis/summarize_nested_cv.py --skip-clustered-bootstrap
```

## Current Data Notes

The current generated dataset contains 1,000 positive and 1,000 negative clips with exactly balanced grouped folds. Train new models under `workspace/models/nested_cv_balanced`.

`workspace/models/nested_cv_stratified`, when present, contains legacy results and caches from the earlier 500-positive/1,000-negative dataset. Do not reuse that directory for the balanced run.
