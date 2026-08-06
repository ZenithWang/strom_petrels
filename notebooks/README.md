# Notebooks

These notebooks are for inspection, visualization, manual review, training, and result analysis.

Current workflow:

```powershell
python data_processing\run_pipeline.py
python data_processing\build_birdnet_fold_dataset.py
```

Then open the Jupyter notebooks:

- `model_training/nested_cv_training.ipynb`: main nested-CV BirdNET training workflow.
- `results_analysis/nested_cv_analysis.ipynb`: nested-CV metrics and error analysis.
- `01_dataset_overview.ipynb`: inspect labels, dataset split sizes, and class balance.
- `02_model_results_analysis.ipynb`: legacy model-results exploration notebook.
- `03_error_review_false_positives.ipynb`: inspect false positives and false negatives.

The reproducible scripts and their related notebooks share the same top-level functional directories.
