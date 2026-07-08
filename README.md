# Storm Petrel Acoustic Classification Pipeline

[中文版本](README_zh.md)

This project builds a binary Storm Petrel acoustic detector from BirdNET-style detections. It prepares 3-second audio clips, creates stratified grouped nested-CV folds, trains BirdNET-Analyzer classifiers, and summarizes fold-level predictions and metrics.

The current binary task is:

- Positive class: `storm_petrel`
- Negative class: `not_storm_petrel`
- Negative subtypes are preserved in metadata through `negative_type`, for example `environment_noise` and `other_bird`

## Repository Layout

```text
scripts/project_config.py                 Shared path configuration for local and server runs
scripts/data_processing/run_pipeline.py   End-to-end data preparation entrypoint
scripts/data_processing/prepare_dataset.py
scripts/data_processing/build_birdnet_fold_dataset.py
scripts/model_training/run_nested_cv.py   Nested-CV BirdNET training and evaluation
scripts/results_analysis/summarize_nested_cv.py
workspace/                                Generated clips, folds, model outputs, summaries
```

Notebook files are kept for inspection, but the reproducible workflow is script-based.

On the server, the working directory is expected to use this flatter layout:

```text
/mnt/home/alyzw89/working/project_config.py
/mnt/home/alyzw89/working/data_processing/
/mnt/home/alyzw89/working/model_training/
/mnt/home/alyzw89/working/results_analysis/
/mnt/home/alyzw89/working/workspace/
```

`results_analysis/` is needed for the final summary step. If it is not currently under `/mnt/home/alyzw89/working`, copy it there before running `python results_analysis/summarize_nested_cv.py`.

## Python Environments

Data preparation needs Python with audio and table libraries:

```bash
pip install librosa soundfile pandas scikit-learn
```

Model training needs Python 3.11 with BirdNET-Analyzer training support:

```bash
pip install "birdnet-analyzer[train]" pandas scikit-learn tensorflow soundfile
```

On this Windows machine the working training interpreter is:

```powershell
D:\Users\93508\anaconda3\envs\strompetrel\python.exe
```

On a server, use the Python executable from the server environment that has the packages above installed.

## Path Configuration

All scripts read paths from `scripts/project_config.py`. Do not hard-code server paths inside the scripts. Set environment variables before running the pipeline.

Important variables:

```bash
export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR=/mnt/home/alyzw89/working/workspace
export STORM_PETREL_MODEL_ROOT=/mnt/home/alyzw89/working/workspace/models/nested_cv_stratified
```

Meaning:

- `STORM_PETREL_PROJECT_ROOT`: repository folder containing `scripts/`
- `STORM_PETREL_DB_PATH`: SQLite database with BirdNET-style detections. The README assumes the file is `/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db`; change only the filename if the database has a different name.
- `STORM_PETREL_AUDIO_ROOT`: folder containing the original full recordings; files can be nested in subfolders
- `STORM_PETREL_WORKSPACE_DIR`: where generated clips, manifests, folds, and fold datasets are written
- `STORM_PETREL_MODEL_ROOT`: where trained models, predictions, and metric summaries are written

If a variable is not set, defaults are resolved relative to the project folder:

```text
DB_PATH       = scripts/data_processing/storm_petrel_2021_full.db
AUDIO_ROOT    = download/data
WORKSPACE_DIR = workspace
MODEL_ROOT    = workspace/models/nested_cv
```

Every script prints the resolved paths at startup. Check those lines first when running on a server.

## Expected Input Data

Before data preparation, the server should have:

```text
/mnt/home/alyzw89/working/             Project code
/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db  Detection database
/mnt/home/alyzw89/data/2021/recordings/  Full source recordings
```

The audio root should contain files referenced by the database recording paths or names. The lookup supports common audio extensions such as `.flac`, `.wav`, `.mp3`, `.ogg`, `.m4a`, and `.webm`.

## Data Processing

Run the data preparation pipeline from the project root.

Windows example:

```powershell
$env:STORM_PETREL_PROJECT_ROOT="D:\Users\93508\Documents\StromPetrels"
$env:STORM_PETREL_DB_PATH="D:\Users\93508\Documents\StromPetrels\scripts\data_processing\storm_petrel_2021_full.db"
$env:STORM_PETREL_AUDIO_ROOT="D:\path\to\recordings"
$env:STORM_PETREL_WORKSPACE_DIR="D:\Users\93508\Documents\StromPetrels\workspace"
$env:STORM_PETREL_MODEL_ROOT="D:\Users\93508\Documents\StromPetrels\workspace\models\nested_cv_stratified"

D:\Users\93508\anaconda3\envs\strompetrel\python.exe scripts\data_processing\run_pipeline.py
```

Linux/server example:

```bash
cd /mnt/home/alyzw89/working

export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR=/mnt/home/alyzw89/working/workspace
export STORM_PETREL_MODEL_ROOT=/mnt/home/alyzw89/working/workspace/models/nested_cv_stratified

python data_processing/run_pipeline.py
```

This produces or rebuilds:

```text
$STORM_PETREL_WORKSPACE_DIR/download_files.csv
$STORM_PETREL_WORKSPACE_DIR/clips/storm_petrel/
$STORM_PETREL_WORKSPACE_DIR/clips/not_storm_petrel/
$STORM_PETREL_WORKSPACE_DIR/binary_labels.csv
$STORM_PETREL_WORKSPACE_DIR/nested_cv_folds.csv
$STORM_PETREL_WORKSPACE_DIR/nested_cv_datasets/outer_01 ... outer_05
```

The fold split uses `StratifiedGroupKFold`:

- `target_label` is used for stratification
- `source_file` is used as the grouping key
- the same `source_file` will not appear in both train and held-out test for the same outer fold
- exact class balance is not guaranteed because a `source_file` is never split across folds

The script prints fold quality information such as positive count, negative count, total samples, source file count, and positive ratio. If a fold has only one class, the script fails.

## Rebuilding Folds From Existing Clips

If clips and `binary_labels.csv` already exist but original full recordings are not available locally, you can rebuild only the folds and BirdNET folder layout:

```bash
python -c "import csv, sys; from pathlib import Path; sys.path.insert(0, str(Path('scripts/data_processing').resolve())); import prepare_dataset, build_birdnet_fold_dataset; rows=list(csv.DictReader(open(prepare_dataset.BINARY_LABELS_PATH, newline='', encoding='utf-8'))); prepare_dataset.write_folds(rows); build_birdnet_fold_dataset.main()"
```

This is useful for validating split logic, but full data preparation from raw recordings still requires `STORM_PETREL_AUDIO_ROOT`.

## Model Training

Run full nested-CV training after data processing has created `nested_cv_datasets`.

Windows example:

```powershell
$env:STORM_PETREL_MODEL_ROOT="D:\Users\93508\Documents\StromPetrels\workspace\models\nested_cv_stratified"
D:\Users\93508\anaconda3\envs\strompetrel\python.exe scripts\model_training\run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
```

Linux/server example:

```bash
export STORM_PETREL_MODEL_ROOT=/mnt/home/alyzw89/working/workspace/models/nested_cv_stratified
python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
```

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

## Result Summary

After training, summarize all completed outer folds:

```bash
python results_analysis/summarize_nested_cv.py
```

The summary script reads from `STORM_PETREL_MODEL_ROOT` and writes:

```text
$STORM_PETREL_MODEL_ROOT/nested_cv_summary.csv
$STORM_PETREL_MODEL_ROOT/negative_type_error_summary.csv
```

`nested_cv_summary.csv` contains fold-level and mean/std metrics:

- precision at 0.5
- recall at 0.5
- F1 at 0.5
- optimized-threshold precision/recall/F1
- AUPRC
- AUROC

`negative_type_error_summary.csv` summarizes false positives by negative subtype.

## Typical Server Run

A full server run usually looks like this:

```bash
cd /mnt/home/alyzw89/working

export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR=/mnt/home/alyzw89/working/workspace
export STORM_PETREL_MODEL_ROOT=/mnt/home/alyzw89/working/workspace/models/nested_cv_stratified

python data_processing/run_pipeline.py
python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
python results_analysis/summarize_nested_cv.py
```

For a quick single-fold smoke run on the server:

```bash
python model_training/run_nested_cv.py --outer-folds 04 --epochs 5 --threads 4
python results_analysis/summarize_nested_cv.py
```

## Current Local Notes

On the current Windows workspace, original full recordings are not under the default `download/data` folder. The existing `workspace/clips` and `workspace/binary_labels.csv` are available, and the latest stratified training results were written to:

```text
workspace/models/nested_cv_stratified
```

For a true end-to-end data rebuild on a server, set `STORM_PETREL_AUDIO_ROOT` to the real folder containing full recordings.



