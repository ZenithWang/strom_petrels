# Notebooks

These notebooks are for inspection, visualization, manual review, training, and result analysis.

Current workflow:

```powershell
python scripts\data_processing\run_pipeline.py
python scripts\data_processing\build_birdnet_fold_dataset.py
```

Then open the Jupyter notebooks:

- `scripts/model_training/nested_cv_training.ipynb`: main nested-CV BirdNET training workflow.
- `scripts/results_analysis/nested_cv_analysis.ipynb`: nested-CV metrics and error analysis.
- `01_dataset_overview.ipynb`: inspect labels, dataset split sizes, and class balance.
- `02_model_results_analysis.ipynb`: legacy model-results exploration notebook.
- `03_error_review_false_positives.ipynb`: inspect false positives and false negatives.

`model_training` and `results_analysis` are notebook-only directories. Data preparation remains script-based under `scripts/data_processing`.
