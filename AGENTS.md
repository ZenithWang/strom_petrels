# AGENTS.md

## Project Overview

This repository contains a Storm Petrel acoustic classification pipeline. The project prepares 3-second audio clips from BirdNET-style detections, builds stratified grouped nested-CV folds, trains BirdNET-Analyzer classifiers, and summarizes fold-level predictions and metrics.

Current binary task:

- Positive class: `storm_petrel`
- Negative class: `not_storm_petrel`
- Negative subtype metadata is preserved in `negative_type`, including values such as `environment_noise` and `other_bird`

## Repository Structure

- `README.md` and `README_zh.md`: user-facing project documentation
- `scripts/project_config.py`: shared path configuration for local and server runs
- `scripts/data_processing/`: reproducible data preparation scripts
- `scripts/model_training/`: nested-CV training scripts and training notebook
- `scripts/results_analysis/`: result summarization scripts and analysis notebook
- `notebooks/`: exploratory/reporting notebooks kept for inspection
- `documents/`: project documentation and writeups
- `workspace/`: generated clips, folds, model outputs, predictions, summaries, caches, and other local artifacts
- `archive/`: local archived material

The reproducible workflow is script-based. Treat notebooks as inspection/reporting material unless the user explicitly asks to edit them.

## Important Rules

- Do not commit large data files, databases, audio files, trained models, cached model outputs, or generated workspace outputs.
- Keep `workspace/`, `archive/`, `.codex/`, `.agents/`, `.idea/`, Python caches, and IDE state out of Git.
- Keep data processing, model training, and result analysis separated by directory.
- Prefer editing existing scripts in place rather than creating duplicate or alternate versions.
- Do not hard-code machine-specific paths inside scripts. Use `scripts/project_config.py` and environment variables.
- Preserve existing notebook structure unless the user explicitly asks for a rewrite.
- Be clear whether a check was structural validation, a smoke test, or a full end-to-end run.

## Python Environment

On this Windows machine, the current project README identifies the working training interpreter as:

```powershell
D:\Users\93508\anaconda3\envs\strompetrel\python.exe
```

Data preparation requires Python packages such as:

```bash
librosa soundfile pandas scikit-learn
```

Model training requires BirdNET-Analyzer training support and related dependencies:

```bash
birdnet-analyzer[train] pandas scikit-learn tensorflow soundfile
```

If audio processing appears to hang on Windows, first check cache and temporary directory behavior before changing model code.

## Path Configuration

All main scripts read paths from `scripts/project_config.py`. Environment variables can override defaults.

Important variables:

- `STORM_PETREL_PROJECT_ROOT`
- `STORM_PETREL_DB_PATH`
- `STORM_PETREL_AUDIO_ROOT`
- `STORM_PETREL_WORKSPACE_DIR`
- `STORM_PETREL_MODEL_ROOT`

Default local paths are resolved relative to the repository:

- Database: `scripts/data_processing/storm_petrel_2021_full.db`
- Audio root: `download/data`
- Workspace: `workspace`
- Model root: `workspace/models/nested_cv`

Every script prints resolved paths at startup. Check those lines first when debugging path issues.

## Common Commands

Run data preparation from the repository root:

```powershell
D:\Users\93508\anaconda3\envs\strompetrel\python.exe scripts\data_processing\run_pipeline.py
```

Run nested cross-validation training:

```powershell
D:\Users\93508\anaconda3\envs\strompetrel\python.exe scripts\model_training\run_nested_cv.py
```

Summarize nested-CV results:

```powershell
D:\Users\93508\anaconda3\envs\strompetrel\python.exe scripts\results_analysis\summarize_nested_cv.py
```

When running on a server, use the server Python environment that has the required packages installed and set the `STORM_PETREL_*` environment variables before running scripts.

## Git Workflow

Before committing:

```powershell
git status
```

Commit source code, notebooks, README files, and lightweight project documentation.

Do not commit:

- `.db`, `.sqlite`, `.sqlite3`
- `.h5`, `.keras`, `.tflite`, `.pkl`, `.joblib`
- audio files such as `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`
- `workspace/`
- `archive/`
- generated model outputs
- local caches
- PDF paper collections unless explicitly requested

Use normal sync commands:

```powershell
git pull
git add .
git commit -m "Describe the change"
git push
```

## Verification Guidance

For script changes, run the smallest relevant command first. Prefer targeted checks before expensive training.

Suggested checks:

- For path/config changes, run the relevant script far enough to inspect printed resolved paths.
- For data-processing changes, run `scripts\data_processing\run_pipeline.py` only when required inputs are available.
- For training changes, avoid claiming final model quality unless training and evaluation actually ran.
- For result-analysis changes, run `scripts\results_analysis\summarize_nested_cv.py` against existing model outputs when available.

When reporting results, state what actually ran and what did not run.
