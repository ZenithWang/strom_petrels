"""汇总 nested-CV 指标、错误类型和录音级 clustered bootstrap。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import BINARY_LABELS_PATH, MODEL_ROOT, print_config

SUMMARY_PATH = MODEL_ROOT / "nested_cv_summary.csv"
ERROR_SUMMARY_PATH = MODEL_ROOT / "negative_type_error_summary.csv"
CLUSTERED_BOOTSTRAP_PATH = MODEL_ROOT / "clustered_bootstrap_ci.csv"
CLUSTERED_BOOTSTRAP_TEX_PATH = MODEL_ROOT / "model_metrics.tex"
METRIC_FIELDS = ["precision_0_5", "recall_0_5", "f1_0_5", "precision_opt", "recall_opt", "f1_opt", "auprc", "auroc"]
PROBABILITY_THRESHOLD = 0.50
BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 2021
EXPECTED_OUTER_FOLDS = {f"{fold:02d}" for fold in range(1, 6)}


def read_table(path: Path) -> pd.DataFrame:
    """读取存在的 CSV；缺失路径返回空表供可选汇总使用。"""

    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    """验证数据表包含计算所需的全部字段。"""

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def collect_oof_predictions() -> pd.DataFrame:
    """汇集五个 outer fold 的 OOF 预测并关联来源录音。"""

    labels = read_table(BINARY_LABELS_PATH)
    if labels.empty:
        raise FileNotFoundError(f"Binary labels are required for clustered bootstrap: {BINARY_LABELS_PATH}")
    require_columns(labels, {"clip_id", "source_file"}, str(BINARY_LABELS_PATH))
    labels = labels.copy()
    labels["file_name"] = labels["clip_id"].astype(str) + ".wav"
    if labels["file_name"].duplicated().any():
        raise ValueError(f"Binary labels contain duplicate clip ids: {BINARY_LABELS_PATH}")

    parts = []
    completed_folds = set()
    for outer_dir in sorted(MODEL_ROOT.glob("outer_*")):
        path = outer_dir / "predictions.csv"
        if not path.is_file():
            continue
        part = pd.read_csv(path)
        require_columns(part, {"file", "true_label", "score"}, str(path))
        fold = outer_dir.name.replace("outer_", "")
        part = part.copy()
        part["outer_fold"] = fold
        parts.append(part)
        completed_folds.add(fold)

    if completed_folds != EXPECTED_OUTER_FOLDS:
        missing = sorted(EXPECTED_OUTER_FOLDS - completed_folds)
        unexpected = sorted(completed_folds - EXPECTED_OUTER_FOLDS)
        raise FileNotFoundError(
            "Clustered bootstrap requires predictions from outer folds 01-05; "
            f"missing={missing}, unexpected={unexpected}, model_root={MODEL_ROOT}"
        )

    oof = pd.concat(parts, ignore_index=True)
    oof["raw_logit"] = pd.to_numeric(oof["score"], errors="raise")
    oof["probability"] = expit(oof["raw_logit"].to_numpy(float))
    oof["true_label"] = pd.to_numeric(oof["true_label"], errors="raise").astype(int)
    oof["file_name"] = oof["file"].astype(str).map(lambda value: Path(value).name)
    if oof["file_name"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate clip file names across outer folds")

    oof = oof.merge(
        labels[["file_name", "source_file"]],
        on="file_name",
        how="left",
        validate="one_to_one",
    )
    if oof["source_file"].isna().any():
        missing_files = sorted(oof.loc[oof["source_file"].isna(), "file_name"].astype(str).tolist())
        raise ValueError(f"OOF predictions could not be matched to binary labels: {missing_files[:10]}")
    if set(oof["true_label"].unique()) != {0, 1}:
        raise ValueError("OOF predictions must contain both binary classes")
    return oof


def pooled_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """计算固定 0.5 阈值下的 pooled OOF 指标。"""

    y_true = frame["true_label"].to_numpy(int)
    probability = frame["probability"].to_numpy(float)
    predicted = probability >= PROBABILITY_THRESHOLD
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def clustered_bootstrap_ci(
    oof: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute percentile intervals by resampling source recordings as whole clusters."""
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")
    require_columns(oof, {"source_file", "true_label", "probability"}, "OOF predictions")

    source_codes, sources = pd.factorize(oof["source_file"], sort=True)
    if len(sources) < 2:
        raise ValueError("Clustered bootstrap requires at least two source recordings")
    y_array = oof["true_label"].to_numpy(int)
    probability = oof["probability"].to_numpy(float)
    predicted = probability >= PROBABILITY_THRESHOLD
    descending = np.argsort(-probability, kind="mergesort")
    sorted_y = y_array[descending]
    sorted_probability = probability[descending]
    threshold_endpoints = np.r_[
        np.flatnonzero(np.diff(sorted_probability) != 0),
        len(sorted_probability) - 1,
    ]

    def weighted_metrics(cluster_multiplicity: np.ndarray) -> dict[str, float]:
        """按一次录音簇抽样的重复次数计算加权指标。"""

        weights = cluster_multiplicity[source_codes].astype(float)
        total_positive = float(np.sum(weights * y_array))
        total_negative = float(np.sum(weights * (1 - y_array)))
        if total_positive == 0 or total_negative == 0:
            raise ValueError(
                "A bootstrap replicate contains only one class; source clusters must support both classes"
            )

        true_positive = float(np.sum(weights * y_array * predicted))
        false_positive = float(np.sum(weights * (1 - y_array) * predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / total_positive
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        sorted_weights = weights[descending]
        cumulative_tp = np.cumsum(sorted_weights * sorted_y)[threshold_endpoints]
        cumulative_fp = np.cumsum(sorted_weights * (1 - sorted_y))[threshold_endpoints]
        curve_precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
        curve_recall = cumulative_tp / total_positive
        auprc = float(np.sum(np.diff(np.r_[0.0, curve_recall]) * curve_precision))
        true_positive_rate = np.r_[0.0, cumulative_tp / total_positive]
        false_positive_rate = np.r_[0.0, cumulative_fp / total_negative]
        auroc = float(np.trapezoid(true_positive_rate, false_positive_rate))
        return {"auroc": auroc, "auprc": auprc, "precision": precision, "recall": recall, "f1": f1}

    rng = np.random.default_rng(seed)
    cluster_draws = rng.multinomial(
        len(sources),
        np.repeat(1 / len(sources), len(sources)),
        size=iterations,
    )
    bootstrap = pd.DataFrame(weighted_metrics(draw) for draw in cluster_draws)
    estimates = pooled_metrics(oof)
    rows = []
    for metric in bootstrap.columns:
        low, high = np.quantile(bootstrap[metric], [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "estimate": estimates[metric],
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_unit": "source_file",
                "iterations": iterations,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def write_clustered_bootstrap() -> pd.DataFrame:
    """计算并写出表 5.3 的 CSV 与 LaTeX 结果。"""

    clustered_ci = clustered_bootstrap_ci(collect_oof_predictions())
    CLUSTERED_BOOTSTRAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    clustered_ci.to_csv(CLUSTERED_BOOTSTRAP_PATH, index=False, encoding="utf-8")
    CLUSTERED_BOOTSTRAP_TEX_PATH.write_text(
        clustered_ci.to_latex(index=False, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )
    print(f"Wrote {CLUSTERED_BOOTSTRAP_PATH}")
    print(f"Wrote {CLUSTERED_BOOTSTRAP_TEX_PATH}")
    print(clustered_ci.to_string(index=False))
    return clustered_ci


def summarize_metrics() -> pd.DataFrame:
    """汇总每个 outer fold 的指标以及 mean/std。"""

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
    """按负类子类型汇总 held-out false positives。"""

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


def parse_args() -> argparse.Namespace:
    """解析正式 bootstrap 与部分 fold smoke test 选项。"""

    parser = argparse.ArgumentParser(description="Summarize grouped nested-CV results.")
    parser.add_argument(
        "--skip-clustered-bootstrap",
        action="store_true",
        help="Skip the formal five-fold clustered bootstrap, for example during a partial-fold smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    """运行完整的 nested-CV 结果汇总入口。"""

    args = parse_args()
    print_config()
    summarize_metrics()
    summarize_errors()
    if args.skip_clustered_bootstrap:
        print("Skipping clustered bootstrap by explicit request.")
    else:
        write_clustered_bootstrap()


if __name__ == "__main__":
    main()
