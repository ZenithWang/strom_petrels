from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import BINARY_LABELS_PATH, MODEL_ROOT, print_config

SUMMARY_PATH = MODEL_ROOT / "nested_cv_summary.csv"
ERROR_SUMMARY_PATH = MODEL_ROOT / "negative_type_error_summary.csv"
METRIC_FIELDS = ["precision_0_5", "recall_0_5", "f1_0_5", "precision_opt", "recall_opt", "f1_opt", "auprc", "auroc"]


def read_table(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def summarize_metrics() -> pd.DataFrame:
    metric_rows = []
    for outer_dir in sorted(MODEL_ROOT.glob("outer_*")):
        metrics = read_table(outer_dir / "metrics.csv")
        if metrics.empty:
            continue
        row = metrics.iloc[0].to_dict()
        row["fold"] = outer_dir.name.replace("outer_", "")
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        print(f"No metrics.csv files found under {MODEL_ROOT}.")
        return metrics_df

    summary_rows = []
    fields = [field for field in METRIC_FIELDS if field in metrics_df.columns]
    for _, row in metrics_df.iterrows():
        summary_rows.append({"fold": row["fold"], **{field: row.get(field, "") for field in fields}})
    summary_rows.append({"fold": "mean", **{field: metrics_df[field].mean() for field in fields}})
    summary_rows.append({"fold": "std", **{field: metrics_df[field].std(ddof=1) if len(metrics_df) > 1 else 0.0 for field in fields}})
    summary = pd.DataFrame(summary_rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    print(summary.to_string(index=False))
    return metrics_df


def summarize_errors() -> None:
    labels = read_table(BINARY_LABELS_PATH)
    if labels.empty:
        print(f"No labels found at {BINARY_LABELS_PATH}; skipping error summary.")
        return

    lookup = {f"{row.clip_id}.wav": row._asdict() for row in labels.itertuples(index=False)}
    prediction_rows = []
    for outer_dir in sorted(MODEL_ROOT.glob("outer_*")):
        predictions = read_table(outer_dir / "predictions.csv")
        metrics = read_table(outer_dir / "metrics.csv")
        if predictions.empty:
            continue
        threshold = float(metrics.iloc[0].get("optimal_threshold", 0.5)) if not metrics.empty else 0.5
        predictions = predictions.copy()
        predictions["outer_fold"] = outer_dir.name.replace("outer_", "")
        predictions["threshold"] = threshold
        prediction_rows.append(predictions)

    if not prediction_rows:
        print(f"No predictions.csv files found under {MODEL_ROOT}; skipping error summary.")
        return

    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    predictions_df["file_name"] = predictions_df["file"].map(lambda value: Path(str(value)).name)
    predictions_df["score"] = predictions_df["score"].astype(float)
    predictions_df["pred_opt"] = predictions_df["score"] >= predictions_df["threshold"]
    predictions_df["pred_0_5"] = predictions_df["score"] >= 0.5
    meta = predictions_df["file_name"].map(lookup)
    predictions_df["negative_type"] = meta.map(lambda item: item.get("negative_type", "unknown") if isinstance(item, dict) else "unknown")

    negatives = predictions_df[predictions_df["true_label"].astype(str) == "0"].copy()
    error_summary = (
        negatives.groupby("negative_type")
        .agg(
            negative_samples=("file", "count"),
            false_positives_opt=("pred_opt", "sum"),
            false_positives_0_5=("pred_0_5", "sum"),
        )
        .reset_index()
    )
    error_summary["false_positive_rate_opt"] = error_summary["false_positives_opt"] / error_summary["negative_samples"]
    error_summary["false_positive_rate_0_5"] = error_summary["false_positives_0_5"] / error_summary["negative_samples"]
    ERROR_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    error_summary.to_csv(ERROR_SUMMARY_PATH, index=False, encoding="utf-8")
    print(f"Wrote {ERROR_SUMMARY_PATH}")
    print(error_summary.to_string(index=False))


def main() -> None:
    print_config()
    summarize_metrics()
    summarize_errors()


if __name__ == "__main__":
    main()
