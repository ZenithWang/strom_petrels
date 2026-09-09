# ============================================================
# analyze_discrepancies.py
# 分析成鸟/雏鸟双模型分数差异，表征 chick 候选窗口声学特征。
# ============================================================
"""分析成鸟期与雏鸟期模型在相同窗口上的分数差异。"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import CHICK_SCREENING_DIR, NUMBA_CACHE_DIR, WORKSPACE_DIR

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE_DIR / "matplotlib_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.stats import mannwhitneyu

from comp_chick_adult.analysis import (
    _bh_fdr,
    bootstrap_median_difference,
    cliffs_delta,
)
from comp_chick_adult.data_processing import (
    FEATURE_COLUMNS,
    SAMPLE_RATE,
    compute_features,
    load_audio_window,
)

QUADRANT_LABELS = {
    "chick_candidate": "Chick candidate (chick↑ adult↓)",
    "general_storm_petrel": "General storm petrel (both↑)",
    "adult_biased": "Adult biased (adult↑ chick↓)",
    "background": "Background (both↓)",
}

YIELD_THRESHOLDS = (0.20, 0.50, 0.80)


def classify_quadrant(
    prob_adult: float,
    prob_chick: float,
    adult_threshold: float,
    chick_threshold: float,
) -> str:
    """依据两个模型阈值把窗口分入四个解释象限。"""

    adult_high = prob_adult >= adult_threshold
    chick_high = prob_chick >= chick_threshold
    if chick_high and not adult_high:
        return "chick_candidate"
    if chick_high and adult_high:
        return "general_storm_petrel"
    if adult_high and not chick_high:
        return "adult_biased"
    return "background"


def build_threshold_yield(scores: pd.DataFrame) -> pd.DataFrame:
    """统计多个阈值组合下各模型的过阈值窗口数量。"""

    rows = []
    for t in YIELD_THRESHOLDS:
        for model in ("adult", "chick"):
            col = f"prob_{model}"
            n = int((scores[col] >= t).sum())
            n_recordings = scores.loc[scores[col] >= t, "recording"].nunique()
            rows.append({
                "threshold": t,
                "model": model,
                "window_count": n,
                "recording_count": n_recordings,
            })
    return pd.DataFrame(rows)


def build_quadrant_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """汇总每个分数象限的窗口数、录音数和模型分数。"""

    counts = scores.groupby("quadrant").agg(
        window_count=("delta_prob", "count"),
        recording_count=("recording", "nunique"),
        mean_delta=("delta_prob", "mean"),
        median_delta=("delta_prob", "median"),
    ).reset_index()
    counts["label"] = counts["quadrant"].map(QUADRANT_LABELS)
    return counts


def build_recording_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """把双模型窗口分数汇总到录音与象限层级。"""

    rows = []
    for rec, group in scores.groupby("recording", sort=True):
        chick_cand = group[group["quadrant"] == "chick_candidate"]
        general = group[group["quadrant"] == "general_storm_petrel"]
        rows.append({
            "recording": rec,
            "total_windows": len(group),
            "chick_candidate_count": len(chick_cand),
            "general_storm_petrel_count": len(general),
            "adult_biased_count": int((group["quadrant"] == "adult_biased").sum()),
            "background_count": int((group["quadrant"] == "background").sum()),
            "mean_prob_adult": float(group["prob_adult"].mean()),
            "mean_prob_chick": float(group["prob_chick"].mean()),
            "median_delta": float(group["delta_prob"].median()),
            "mean_delta": float(group["delta_prob"].mean()),
        })
    return pd.DataFrame(rows)


def extract_window_features(scores: pd.DataFrame) -> pd.DataFrame:
    """为完整双模型窗口读取音频并计算声学特征。"""

    feature_rows = []
    log_mels: dict[str, list[np.ndarray]] = {
        "chick_candidate": [],
        "general_storm_petrel": [],
    }
    for _, row in scores.iterrows():
        quadrant = row["quadrant"]
        if quadrant not in log_mels:
            continue
        try:
            audio = load_audio_window(
                Path(str(row["recording"])), float(row["start_seconds"])
            )
            features, log_mel = compute_features(audio)
        except (ValueError, RuntimeError, OSError):
            continue
        feature_rows.append({
            "recording": row["recording"],
            "start_seconds": row["start_seconds"],
            "quadrant": quadrant,
            "prob_adult": row["prob_adult"],
            "prob_chick": row["prob_chick"],
            "delta_prob": row["delta_prob"],
            **features,
        })
        log_mels[quadrant].append(log_mel)
    return pd.DataFrame(feature_rows), log_mels


def compare_quadrant_features(
    features: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    """比较 chick-candidate 与 general-storm-petrel 象限的录音级特征。"""

    chick = features[features["quadrant"] == "chick_candidate"]
    general = features[features["quadrant"] == "general_storm_petrel"]
    enough = len(chick) >= 2 and len(general) >= 2
    rng = np.random.default_rng(seed)
    rows = []
    for feat in FEATURE_COLUMNS:
        adult_vals = general[feat].dropna().to_numpy(float)
        chick_vals = chick[feat].dropna().to_numpy(float)
        row: dict[str, object] = {
            "feature": feat,
            "general_storm_petrel_count": len(adult_vals),
            "chick_candidate_count": len(chick_vals),
            "general_median": float(np.median(adult_vals)) if len(adult_vals) else np.nan,
            "chick_median": float(np.median(chick_vals)) if len(chick_vals) else np.nan,
            "median_difference": (
                float(np.median(chick_vals) - np.median(adult_vals))
                if len(adult_vals) and len(chick_vals)
                else np.nan
            ),
            "mann_whitney_u": np.nan,
            "p_value": np.nan,
            "cliffs_delta": np.nan,
            "bootstrap_ci95_low": np.nan,
            "bootstrap_ci95_high": np.nan,
        }
        if enough and len(adult_vals) >= 2 and len(chick_vals) >= 2:
            test = mannwhitneyu(chick_vals, adult_vals, alternative="two-sided")
            low, high = bootstrap_median_difference(
                adult_vals, chick_vals, bootstrap_iterations, rng
            )
            row.update({
                "mann_whitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
                "cliffs_delta": cliffs_delta(adult_vals, chick_vals),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
            })
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison["p_value_bh_fdr"] = _bh_fdr(comparison["p_value"])
    return comparison


# ---- Plotting ----

def _save_empty_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_probability_scatter(scores: pd.DataFrame, adult_t: float, chick_t: float, out: Path) -> None:
    """绘制双模型概率散点图和阈值象限。"""

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {
        "chick_candidate": "#d62728",
        "general_storm_petrel": "#2ca02c",
        "adult_biased": "#1f77b4",
        "background": "#aaaaaa",
    }
    for quad, group in scores.groupby("quadrant"):
        ax.scatter(
            group["prob_adult"], group["prob_chick"],
            c=colors.get(quad, "#000000"), alpha=0.3, s=4,
            label=QUADRANT_LABELS.get(quad, quad),
        )
    ax.axvline(adult_t, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(chick_t, color="gray", linestyle="--", alpha=0.5)
    ax.set(xlabel="Adult model probability", ylabel="Chick model probability",
           title=f"Dual-model probability (thresholds: adult={adult_t}, chick={chick_t})")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_delta_distribution(scores: pd.DataFrame, out: Path) -> None:
    """绘制 chick-minus-adult 模型分数差分布。"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for quad, group in scores.groupby("quadrant"):
        axes[0].hist(group["delta_prob"], bins=40, alpha=0.4, label=QUADRANT_LABELS.get(quad, quad))
    axes[0].set(xlabel="delta = prob_chick − prob_adult", ylabel="Window count",
                title="Delta distribution by quadrant")
    axes[0].legend(fontsize=7)
    axes[1].hist(scores["delta_prob"], bins=60, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set(xlabel="delta = prob_chick − prob_adult", ylabel="Window count",
                title="Overall delta distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_average_spectrograms(
    log_mels: dict[str, list[np.ndarray]], out_dir: Path
) -> None:
    """绘制目标象限的平均频谱及差异频谱。"""

    averages = {
        quad: np.mean(np.stack(vals), axis=0) if vals else None
        for quad, vals in log_mels.items()
    }
    valid = [img for img in averages.values() if img is not None]
    if not valid:
        _save_empty_figure(out_dir / "average_log_mel_spectrograms.png",
                           "Average log-mel spectrograms", "No windows")
        _save_empty_figure(out_dir / "difference_log_mel_spectrogram.png",
                           "Chick candidate − general storm petrel", "Both groups required")
        return
    vmin = min(float(np.min(img)) for img in valid)
    vmax = max(float(np.max(img)) for img in valid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, quad in zip(axes, ["general_storm_petrel", "chick_candidate"]):
        img = averages.get(quad)
        if img is None:
            ax.text(0.5, 0.5, "No windows", ha="center", va="center")
            ax.set_axis_off()
        else:
            display = librosa.display.specshow(
                img, sr=SAMPLE_RATE, hop_length=512,
                x_axis="time", y_axis="mel", fmax=SAMPLE_RATE / 2,
                ax=ax, vmin=vmin, vmax=vmax,
            )
            fig.colorbar(display, ax=ax, format="%+2.0f dB")
        ax.set_title(f"{quad} (n={len(log_mels.get(quad, []))})")
    fig.tight_layout()
    fig.savefig(out_dir / "average_log_mel_spectrograms.png", dpi=160)
    plt.close(fig)

    if averages.get("general_storm_petrel") is None or averages.get("chick_candidate") is None:
        _save_empty_figure(out_dir / "difference_log_mel_spectrogram.png",
                           "Chick candidate − general storm petrel", "Both groups required")
        return
    diff = averages["chick_candidate"] - averages["general_storm_petrel"]
    limit = max(float(np.max(np.abs(diff))), np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(8, 4))
    display = librosa.display.specshow(
        diff, sr=SAMPLE_RATE, hop_length=512,
        x_axis="time", y_axis="mel", fmax=SAMPLE_RATE / 2,
        ax=ax, cmap="coolwarm", vmin=-limit, vmax=limit,
    )
    ax.set_title("Mean log-mel difference: chick candidate − general storm petrel")
    fig.colorbar(display, ax=ax, label="dB difference")
    fig.tight_layout()
    fig.savefig(out_dir / "difference_log_mel_spectrogram.png", dpi=160)
    plt.close(fig)


def plot_feature_boxplots(features: pd.DataFrame, out: Path) -> None:
    """绘制目标象限之间的声学特征箱线图。"""

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for ax, feat in zip(axes.ravel(), FEATURE_COLUMNS):
        vals = [
            features.loc[features["quadrant"] == "general_storm_petrel", feat].dropna().to_numpy(),
            features.loc[features["quadrant"] == "chick_candidate", feat].dropna().to_numpy(),
        ]
        if any(len(v) for v in vals):
            ax.boxplot(vals, tick_labels=["general", "chick_cand"], showfliers=True)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(feat)
    fig.suptitle("Acoustic features: general storm petrel vs chick candidate")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def export_representative_audio(
    scores: pd.DataFrame, output_dir: Path, count: int
) -> pd.DataFrame:
    """导出每个象限的代表窗口音频及清单。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for quad in ["chick_candidate", "general_storm_petrel"]:
        candidates = scores[scores["quadrant"] == quad].sort_values(
            ["delta_prob", "prob_chick"], ascending=[False, False]
        )
        first_per_rec = candidates.drop_duplicates("recording", keep="first")
        selected = first_per_rec.head(count)
        quad_dir = output_dir / quad
        quad_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            rec = Path(str(row["recording"]))
            audio = load_audio_window(rec, float(row["start_seconds"]))
            digest = hashlib.sha1(str(rec).encode()).hexdigest()[:8]
            start_ms = round(float(row["start_seconds"]) * 1000)
            name = f"{rank:02d}_{rec.stem}_{digest}_{start_ms:010d}.wav"
            out_path = quad_dir / name
            sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")
            manifest.append({
                "quadrant": quad, "rank": rank,
                "prob_adult": row["prob_adult"],
                "prob_chick": row["prob_chick"],
                "delta_prob": row["delta_prob"],
                "recording": row["recording"],
                "start_seconds": row["start_seconds"],
                "output_path": str(out_path.resolve()),
            })
    return pd.DataFrame(manifest)


def write_report(
    path: Path,
    scores: pd.DataFrame,
    quadrant_summary: pd.DataFrame,
    feature_comparison: pd.DataFrame,
    adult_threshold: float,
    chick_threshold: float,
) -> None:
    """写出双模型筛选结果和解释限制的 Markdown 报告。"""

    total = len(scores)
    chick_cand_n = int((scores["quadrant"] == "chick_candidate").sum())
    general_n = int((scores["quadrant"] == "general_storm_petrel").sum())
    lines = [
        "# Dual-model chick vocalisation screening",
        "",
        "## Scope",
        "",
        f"- Adult model threshold: `{adult_threshold:.2f}`",
        f"- Chick model threshold: `{chick_threshold:.2f}`",
        "- Model scores are uncalibrated sigmoid outputs, not biological probabilities.",
        "- 'Chick candidate' = chick model high + adult model low.",
        "- Results are associative; chick-proxy recordings may also contain adult calls.",
        "",
        "## Quadrant summary",
        "",
        f"| Quadrant | Windows | Recordings | Mean delta |",
        f"|---|---:|---:|---:|",
    ]
    for _, row in quadrant_summary.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['window_count'])} | "
            f"{int(row['recording_count'])} | {float(row['mean_delta']):.4f} |"
        )
    lines.extend([
        "",
        f"Total windows: {total}",
        f"Chick candidates: {chick_cand_n} ({100*chick_cand_n/total:.1f}%)" if total else "",
        f"General storm petrel: {general_n} ({100*general_n/total:.1f}%)" if total else "",
        "",
        "## Acoustic feature comparison (chick candidate vs general storm petrel)",
        "",
        "| Feature | General n | General median | Chick n | Chick median | Diff | p | BH-FDR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    def fmt(value: object) -> str:
        """统一格式化报告中的数值与缺失值。"""

        return "" if pd.isna(value) else f"{float(value):.6g}"

    for _, row in feature_comparison.iterrows():
        lines.append(
            f"| {row['feature']} | {int(row['general_storm_petrel_count'])} | "
            f"{fmt(row['general_median'])} | {int(row['chick_candidate_count'])} | "
            f"{fmt(row['chick_median'])} | {fmt(row['median_difference'])} | "
            f"{fmt(row['p_value'])} | {fmt(row['p_value_bh_fdr'])} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `quadrant_summary.csv`, `recording_summary.csv`, `threshold_yield.csv`",
        "- `feature_comparison.csv`, `window_features.csv`",
        "- `figures/`: scatter, delta histogram, spectrograms, feature boxplots",
        "- `representative_audio/`: top chick-candidate and general storm-petrel examples",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_discrepancies(
    scores_csv: Path,
    output_dir: Path,
    adult_threshold: float = 0.80,
    chick_threshold: float = 0.80,
    representative_count: int = 12,
    bootstrap_iterations: int = 5000,
    seed: int = 2021,
) -> Path:
    """从 dual_scores.csv 生成全部统计、图表、音频和报告。"""

    if not scores_csv.is_file():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")

    scores = pd.read_csv(scores_csv)
    required = {"recording", "start_seconds", "prob_adult", "prob_chick"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing columns in scores CSV: {sorted(missing)}")

    scores["quadrant"] = scores.apply(
        lambda r: classify_quadrant(
            float(r["prob_adult"]), float(r["prob_chick"]),
            adult_threshold, chick_threshold,
        ),
        axis=1,
    )

    figures_dir = output_dir / "figures"
    rep_dir = output_dir / "representative_audio"
    for d in (figures_dir, rep_dir):
        if d.exists():
            shutil.rmtree(d)
    figures_dir.mkdir(parents=True)
    rep_dir.mkdir(parents=True)

    # Tables
    threshold_yield = build_threshold_yield(scores)
    quadrant_summary = build_quadrant_summary(scores)
    recording_summary = build_recording_summary(scores)

    # Features
    window_features, log_mels = extract_window_features(scores)
    feature_comparison = compare_quadrant_features(window_features, bootstrap_iterations, seed)

    # Audio
    rep_manifest = export_representative_audio(scores, rep_dir, representative_count)

    # Write CSVs
    threshold_yield.to_csv(output_dir / "threshold_yield.csv", index=False)
    quadrant_summary.to_csv(output_dir / "quadrant_summary.csv", index=False)
    recording_summary.to_csv(output_dir / "recording_summary.csv", index=False)
    window_features.to_csv(output_dir / "window_features.csv", index=False)
    feature_comparison.to_csv(output_dir / "feature_comparison.csv", index=False)
    rep_manifest.to_csv(rep_dir / "manifest.csv", index=False)

    # Plots
    plot_probability_scatter(scores, adult_threshold, chick_threshold,
                             figures_dir / "probability_scatter.png")
    plot_delta_distribution(scores, figures_dir / "delta_distribution.png")
    plot_average_spectrograms(log_mels, figures_dir)
    if not window_features.empty:
        plot_feature_boxplots(window_features, figures_dir / "feature_boxplots.png")

    # Report
    write_report(output_dir / "analysis_report.md", scores, quadrant_summary,
                 feature_comparison, adult_threshold, chick_threshold)

    print(f"Wrote analysis to {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    """解析双模型分歧分析的输入、阈值和 bootstrap 参数。"""

    parser = argparse.ArgumentParser(
        description="Analyze dual-model score discrepancies for chick screening."
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=CHICK_SCREENING_DIR / "dual_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to workspace/chick_screening/<auto_name>",
    )
    parser.add_argument("--adult-threshold", type=float, default=0.80)
    parser.add_argument("--chick-threshold", type=float, default=0.80)
    parser.add_argument("--representative-count", type=int, default=12)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def main() -> None:
    """运行双模型分歧分析命令行入口。"""

    args = parse_args()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else CHICK_SCREENING_DIR / "analysis"
    )
    analyze_discrepancies(
        args.scores_csv.expanduser().resolve(),
        output_dir,
        args.adult_threshold,
        args.chick_threshold,
        args.representative_count,
        args.bootstrap_iterations,
        args.seed,
    )


if __name__ == "__main__":
    main()
