"""汇总日期代理窗口，并生成统计比较、图表、音频和报告。"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import mannwhitneyu

from project_config import NUMBA_CACHE_DIR, WORKSPACE_DIR

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE_DIR / "matplotlib_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import librosa
import librosa.display
import matplotlib.pyplot as plt

from comp_chick_adult.data_processing import (
    FEATURE_COLUMNS,
    PHASES,
    SAMPLE_RATE,
    load_audio_window,
)

YIELD_THRESHOLDS = (0.20, 0.50, 0.80)
COMPARISON_COLUMNS = FEATURE_COLUMNS + ["high_confidence_hits_per_hour"]
DERIVED_FILES = [
    "analysis_windows.csv",
    "recording_summary.csv",
    "feature_comparison.csv",
    "threshold_yield.csv",
    "excluded_windows.csv",
    "analysis_report.md",
]


def build_threshold_yield(eligible: pd.DataFrame) -> pd.DataFrame:
    """统计预设阈值下两个日期阶段的候选窗口数量。"""

    rows: list[dict[str, object]] = []
    for threshold in YIELD_THRESHOLDS:
        for phase in PHASES:
            if eligible.empty:
                selected = eligible
            else:
                selected = eligible[
                    (eligible["phase"] == phase) & (eligible["probability"] >= threshold)
                ]
            rows.append(
                {
                    "threshold": threshold,
                    "phase": phase,
                    "window_count": len(selected),
                    "recording_count": selected["recording"].nunique() if not selected.empty else 0,
                }
            )
    return pd.DataFrame(rows)


def build_recording_summary(included: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """把窗口级特征汇总为每条录音、每个代理阶段的中位数。"""

    columns = [
        "recording",
        "recording_date",
        "phase",
        "recording_duration_seconds",
        "eligible_window_count",
        "high_confidence_window_count",
        "high_confidence_hits_per_hour",
        "median_probability",
        *FEATURE_COLUMNS,
    ]
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for recording, all_group in eligible.groupby("recording", sort=True):
        high_group = (
            included[included["recording"] == recording] if not included.empty else included
        )
        duration = float(all_group["recording_duration_seconds"].iloc[0])
        row: dict[str, object] = {
            "recording": recording,
            "recording_date": all_group["recording_date"].iloc[0],
            "phase": all_group["phase"].iloc[0],
            "recording_duration_seconds": duration,
            "eligible_window_count": len(all_group),
            "high_confidence_window_count": len(high_group),
            "high_confidence_hits_per_hour": (
                len(high_group) / (duration / 3600.0) if duration > 0 else np.nan
            ),
            "median_probability": (
                float(high_group["probability"].median()) if not high_group.empty else np.nan
            ),
        }
        for feature in FEATURE_COLUMNS:
            row[feature] = (
                float(high_group[feature].median()) if not high_group.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def cliffs_delta(adult: np.ndarray, chick: np.ndarray) -> float:
    """计算 chick-minus-adult 方向的 Cliff's delta。"""

    differences = chick[:, None] - adult[None, :]
    return float(
        (np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0))
        / differences.size
    )


def bootstrap_median_difference(
    adult: np.ndarray,
    chick: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """以录音为单位 bootstrap 两组中位数差的百分位区间。"""

    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        adult_sample = rng.choice(adult, size=len(adult), replace=True)
        chick_sample = rng.choice(chick, size=len(chick), replace=True)
        differences[index] = np.median(chick_sample) - np.median(adult_sample)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    count = len(valid)
    if not count:
        return result
    adjusted = (valid.to_numpy() * count / np.arange(1, count + 1))[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def compare_features(
    recording_summary: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, bool]:
    """计算特征差异、效应量、检验、bootstrap 区间和 BH-FDR。"""

    high_confidence_recordings = (
        recording_summary[recording_summary["high_confidence_window_count"] > 0]
        if not recording_summary.empty
        else recording_summary
    )
    phase_counts = (
        high_confidence_recordings.groupby("phase")["recording"].nunique()
        if not high_confidence_recordings.empty
        else {}
    )
    enough_recordings = all(int(phase_counts.get(phase, 0)) >= 2 for phase in PHASES)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for feature in COMPARISON_COLUMNS:
        adult = (
            recording_summary.loc[
                recording_summary["phase"] == "adult_proxy", feature
            ]
            .dropna()
            .to_numpy(float)
        )
        chick = (
            recording_summary.loc[
                recording_summary["phase"] == "chick_proxy", feature
            ]
            .dropna()
            .to_numpy(float)
        )
        row: dict[str, object] = {
            "feature": feature,
            "adult_recording_count": len(adult),
            "chick_recording_count": len(chick),
            "adult_median": float(np.median(adult)) if len(adult) else np.nan,
            "adult_q1": float(np.percentile(adult, 25)) if len(adult) else np.nan,
            "adult_q3": float(np.percentile(adult, 75)) if len(adult) else np.nan,
            "chick_median": float(np.median(chick)) if len(chick) else np.nan,
            "chick_q1": float(np.percentile(chick, 25)) if len(chick) else np.nan,
            "chick_q3": float(np.percentile(chick, 75)) if len(chick) else np.nan,
            "median_difference_chick_minus_adult": (
                float(np.median(chick) - np.median(adult))
                if len(adult) and len(chick)
                else np.nan
            ),
            "mann_whitney_u": np.nan,
            "p_value": np.nan,
            "cliffs_delta_chick_minus_adult": np.nan,
            "bootstrap_ci95_low": np.nan,
            "bootstrap_ci95_high": np.nan,
        }
        if enough_recordings and len(adult) >= 2 and len(chick) >= 2:
            test = mannwhitneyu(chick, adult, alternative="two-sided")
            low, high = bootstrap_median_difference(
                adult, chick, bootstrap_iterations, rng
            )
            row.update(
                {
                    "mann_whitney_u": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "cliffs_delta_chick_minus_adult": cliffs_delta(adult, chick),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                }
            )
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison["p_value_bh_fdr"] = _bh_fdr(comparison["p_value"])
    return comparison, enough_recordings


def select_representatives(
    included: pd.DataFrame, phase: str, count: int
) -> pd.DataFrame:
    """按分数为每个日期代理阶段选择代表窗口。"""

    if included.empty:
        return included
    candidates = included[included["phase"] == phase].sort_values(
        ["probability", "recording", "start_seconds"], ascending=[False, True, True]
    )
    first_per_recording = candidates.drop_duplicates("recording", keep="first")
    selected_indices = list(first_per_recording.head(count).index)
    if len(selected_indices) < count:
        remaining = candidates.drop(index=selected_indices)
        selected_indices.extend(remaining.head(count - len(selected_indices)).index)
    return candidates.loc[selected_indices].sort_values("probability", ascending=False)


def export_representative_audio(
    included: pd.DataFrame, output_dir: Path, count: int
) -> pd.DataFrame:
    """导出代表窗口的音频文件和可追踪清单。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for phase in PHASES:
        selected = select_representatives(included, phase, count)
        phase_dir = output_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            recording = Path(str(row["recording"]))
            audio = load_audio_window(recording, float(row["start_seconds"]))
            digest = hashlib.sha1(str(recording).encode("utf-8")).hexdigest()[:8]
            start_ms = round(float(row["start_seconds"]) * 1000)
            name = f"{rank:02d}_{recording.stem}_{digest}_{start_ms:010d}.wav"
            output_path = phase_dir / name
            sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")
            manifest.append(
                {
                    "phase": phase,
                    "rank": rank,
                    "probability": row["probability"],
                    "recording": row["recording"],
                    "start_seconds": row["start_seconds"],
                    "output_path": str(output_path.resolve()),
                }
            )
    return pd.DataFrame(
        manifest,
        columns=[
            "phase",
            "rank",
            "probability",
            "recording",
            "start_seconds",
            "output_path",
        ],
    )


def _save_empty_figure(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_average_spectrograms(
    log_mels: dict[str, list[np.ndarray]], figures_dir: Path
) -> None:
    """绘制两个代理阶段的平均 log-mel 频谱图。"""

    averages = {
        phase: np.mean(np.stack(values), axis=0) if values else None
        for phase, values in log_mels.items()
    }
    valid_images = [image for image in averages.values() if image is not None]
    if not valid_images:
        _save_empty_figure(
            figures_dir / "average_log_mel_spectrograms.png",
            "Average log-mel spectrograms",
            "No high-confidence windows",
        )
        _save_empty_figure(
            figures_dir / "difference_log_mel_spectrogram.png",
            "Chick proxy minus adult proxy",
            "Both phases are required",
        )
        return
    value_min = min(float(np.min(image)) for image in valid_images)
    value_max = max(float(np.max(image)) for image in valid_images)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, phase in zip(axes, PHASES):
        image = averages[phase]
        if image is None:
            axis.text(0.5, 0.5, "No windows", ha="center", va="center")
            axis.set_axis_off()
        else:
            display = librosa.display.specshow(
                image,
                sr=SAMPLE_RATE,
                hop_length=512,
                x_axis="time",
                y_axis="mel",
                fmax=SAMPLE_RATE / 2,
                ax=axis,
                vmin=value_min,
                vmax=value_max,
            )
            figure.colorbar(display, ax=axis, format="%+2.0f dB")
        axis.set_title(f"{phase} (n={len(log_mels[phase])})")
    figure.tight_layout()
    figure.savefig(figures_dir / "average_log_mel_spectrograms.png", dpi=160)
    plt.close(figure)

    if averages["adult_proxy"] is None or averages["chick_proxy"] is None:
        _save_empty_figure(
            figures_dir / "difference_log_mel_spectrogram.png",
            "Chick proxy minus adult proxy",
            "Both phases are required",
        )
        return
    difference = averages["chick_proxy"] - averages["adult_proxy"]
    limit = max(float(np.max(np.abs(difference))), np.finfo(float).eps)
    figure, axis = plt.subplots(figsize=(8, 4))
    display = librosa.display.specshow(
        difference,
        sr=SAMPLE_RATE,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=SAMPLE_RATE / 2,
        ax=axis,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_title("Mean log-mel difference: chick proxy minus adult proxy")
    figure.colorbar(display, ax=axis, label="dB difference")
    figure.tight_layout()
    figure.savefig(figures_dir / "difference_log_mel_spectrogram.png", dpi=160)
    plt.close(figure)


def plot_representative_spectrograms(
    manifest: pd.DataFrame, figures_dir: Path
) -> None:
    """绘制已导出代表音频的频谱图面板。"""

    if manifest.empty:
        _save_empty_figure(
            figures_dir / "representative_spectrograms.png",
            "Representative high-confidence windows",
            "No representative windows",
        )
        return
    shown = pd.concat(
        [manifest[manifest["phase"] == phase].head(3) for phase in PHASES],
        ignore_index=True,
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for axis in axes.ravel():
        axis.set_axis_off()
    for phase_index, phase in enumerate(PHASES):
        phase_rows = shown[shown["phase"] == phase]
        for column_index, (_, row) in enumerate(phase_rows.iterrows()):
            axis = axes[phase_index, column_index]
            audio, _ = librosa.load(row["output_path"], sr=SAMPLE_RATE, mono=True)
            mel = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=audio, sr=SAMPLE_RATE, n_mels=128, fmax=SAMPLE_RATE / 2
                ),
                ref=1.0,
                top_db=100,
            )
            librosa.display.specshow(
                mel,
                sr=SAMPLE_RATE,
                hop_length=512,
                x_axis="time",
                y_axis="mel",
                ax=axis,
            )
            axis.set_axis_on()
            axis.set_title(f"{phase}, p={float(row['probability']):.3f}")
    figure.tight_layout()
    figure.savefig(figures_dir / "representative_spectrograms.png", dpi=160)
    plt.close(figure)


def plot_feature_distributions(
    recording_summary: pd.DataFrame, figures_dir: Path
) -> None:
    """绘制录音级声学特征分布。"""

    figure, axes = plt.subplots(3, 3, figsize=(14, 12))
    for axis, feature in zip(axes.ravel(), FEATURE_COLUMNS):
        values = [
            recording_summary.loc[
                recording_summary["phase"] == phase, feature
            ]
            .dropna()
            .to_numpy()
            for phase in PHASES
        ]
        if any(len(value) for value in values):
            axis.boxplot(values, tick_labels=["adult", "chick"], showfliers=True)
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_title(feature)
    figure.suptitle("Recording-level median acoustic features")
    figure.tight_layout()
    figure.savefig(figures_dir / "feature_distributions.png", dpi=160)
    plt.close(figure)


def plot_probability_distributions(
    eligible: pd.DataFrame, threshold: float, figures_dir: Path
) -> None:
    """绘制两个代理阶段的模型分数分布。"""

    figure, axis = plt.subplots(figsize=(8, 4))
    for phase in PHASES:
        values = (
            eligible.loc[eligible["phase"] == phase, "probability"]
            if not eligible.empty
            else []
        )
        axis.hist(values, bins=np.linspace(0, 1, 31), alpha=0.5, label=phase)
    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        label=f"analysis threshold {threshold:.2f}",
    )
    axis.set(
        xlabel="Model score",
        ylabel="Window count",
        title="Probability distribution by date proxy",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "probability_distributions.png", dpi=160)
    plt.close(figure)


def plot_date_activity(included: pd.DataFrame, figures_dir: Path) -> None:
    """绘制纳入窗口随录音日期变化的活动图。"""

    if included.empty:
        _save_empty_figure(
            figures_dir / "date_activity.png",
            "High-confidence detections by date",
            "No high-confidence windows",
        )
        return
    activity = (
        included.groupby(["recording_date", "phase"])
        .size()
        .rename("count")
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    for phase in PHASES:
        values = activity[activity["phase"] == phase]
        axis.plot(
            pd.to_datetime(values["recording_date"]),
            values["count"],
            marker="o",
            label=phase,
        )
    axis.set(
        xlabel="Recording date",
        ylabel="High-confidence window count",
        title="Date activity",
    )
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(figures_dir / "date_activity.png", dpi=160)
    plt.close(figure)


def write_report(
    path: Path,
    csv_paths: Sequence[Path],
    threshold: float,
    adult_proxy_end: date,
    chick_proxy_start: date,
    included: pd.DataFrame,
    recording_summary: pd.DataFrame,
    excluded: pd.DataFrame,
    comparison: pd.DataFrame,
    enough_recordings: bool,
    excluded_recordings: pd.DataFrame | None = None,
) -> None:
    """写出包含输入、限制、统计和输出说明的 Markdown 报告。"""

    phase_windows = included.groupby("phase").size() if not included.empty else {}
    phase_recordings = (
        included.groupby("phase")["recording"].nunique() if not included.empty else {}
    )
    lines = [
        "# Adult-proxy versus chick-proxy acoustic comparison",
        "",
        "## Scope",
        "",
        f"- High-confidence threshold: `{threshold:.2f}` (an uncalibrated sigmoid model score, not a biological probability).",
        f"- Adult proxy: recordings dated on or before `{adult_proxy_end.isoformat()}`.",
        f"- Chick proxy: recordings dated on or after `{chick_proxy_start.isoformat()}`.",
        "- Dates between the two boundaries are excluded as the incubation/transition period.",
        "- These are date-based breeding-stage proxies, not manually confirmed caller identities. Results are associative, not causal.",
        f"- Input score files: {len(csv_paths)}.",
        "",
        "## Included high-confidence windows",
        "",
    ]
    for phase in PHASES:
        lines.append(
            f"- `{phase}`: {int(phase_windows.get(phase, 0))} windows from "
            f"{int(phase_recordings.get(phase, 0))} recordings."
        )
    lines.extend(["", "## Exclusions", ""])
    if excluded.empty and (excluded_recordings is None or excluded_recordings.empty):
        lines.append("- None.")
    else:
        if not excluded.empty:
            for reason, count in (
                excluded["exclusion_reason"].value_counts().sort_index().items()
            ):
                lines.append(f"- Window `{reason}`: {count}")
        if excluded_recordings is not None and not excluded_recordings.empty:
            for reason, count in (
                excluded_recordings["exclusion_reason"]
                .value_counts()
                .sort_index()
                .items()
            ):
                lines.append(f"- Recording `{reason}` (not scanned): {count}")
    lines.extend(["", "## Statistical interpretation", ""])
    if enough_recordings:
        lines.extend(
            [
                "- Acoustic features were reduced to a median per source recording before inference.",
                "- The table uses two-sided Mann–Whitney U tests, Cliff's delta (positive means chick proxy is larger), recording-level bootstrap 95% intervals for the median difference, and Benjamini–Hochberg FDR.",
            ]
        )
    else:
        lines.extend(
            [
                "> **Insufficient independent recordings for inferential statistics.** At least one phase has fewer than two source recordings at the analysis threshold.",
                "",
                "Only descriptive statistics are reported. Add recordings or explicitly choose a lower threshold and rerun; the threshold was not lowered automatically.",
            ]
        )
    lines.extend(
        [
            "",
            "## Descriptive feature results",
            "",
            "| feature | adult n | adult median | chick n | chick median | chick-adult | p | BH-FDR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    def fmt(value: object) -> str:
        """把报告中的数值统一格式化，并保留缺失标记。"""

        return "" if pd.isna(value) else f"{float(value):.6g}"

    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['feature']} | {int(row['adult_recording_count'])} | {fmt(row['adult_median'])} | "
            f"{int(row['chick_recording_count'])} | {fmt(row['chick_median'])} | "
            f"{fmt(row['median_difference_chick_minus_adult'])} | {fmt(row['p_value'])} | "
            f"{fmt(row['p_value_bh_fdr'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `analysis_windows.csv`: included high-confidence three-second windows and their acoustic features.",
            "- `recording_summary.csv`: recording-level feature medians and high-confidence hits per hour.",
            "- `feature_comparison.csv`: descriptive and, when supported, inferential recording-level comparisons.",
            "- `threshold_yield.csv`: phase yield at model-score thresholds 0.20, 0.50, and 0.80.",
            "- `excluded_windows.csv`: records excluded from the formal analysis and the reason.",
            "- `scan/`: independently generated window scores, scan configuration, and pre-scan recording exclusions when raw recordings were supplied.",
            "- `representative_audio/`: highest-confidence, recording-diverse three-second WAV examples.",
            "- `figures/`: mean, difference, representative, feature, probability, and date plots.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def refresh_derived_output_directories(output_dir: Path) -> tuple[Path, Path]:
    """重建可再生的图表和代表音频目录，同时保留扫描缓存。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in DERIVED_FILES:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    figures_dir = output_dir / "figures"
    representative_dir = output_dir / "representative_audio"
    for path in (figures_dir, representative_dir):
        if path.exists():
            shutil.rmtree(path)
    figures_dir.mkdir()
    return figures_dir, representative_dir
