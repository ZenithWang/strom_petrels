from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import (
    COMP_CHICK_ADULT_DIR,
    DEPLOYMENT_MODEL_DIR,
    print_config,
)

# This module remains the stable public facade and CLI entry point. Re-export
# the existing helpers so imports used before the internal split keep working.
from comp_chick_adult.analysis import *  # noqa: F403
from comp_chick_adult.data_processing import *  # noqa: F403

_refresh_derived_output_directories = refresh_derived_output_directories  # noqa: F405


def run_analysis(
    inputs: Sequence[Path],
    output_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,  # noqa: F405
    adult_proxy_end: date = DEFAULT_ADULT_PROXY_END,  # noqa: F405
    chick_proxy_start: date = DEFAULT_CHICK_PROXY_START,  # noqa: F405
    representative_count: int = 12,
    bootstrap_iterations: int = 5000,
    seed: int = 2021,
    force: bool = False,
    excluded_recordings: pd.DataFrame | None = None,
) -> Path:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if representative_count < 0:
        raise ValueError("representative_count must not be negative")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be greater than zero")
    if adult_proxy_end >= chick_proxy_start:
        raise ValueError("adult_proxy_end must be earlier than chick_proxy_start")
    csv_paths = discover_csvs(inputs)  # noqa: F405
    # Derived analysis artifacts are reproducible and refresh automatically.
    # The independent scan cache under output_dir/scan is deliberately preserved.
    del force
    figures_dir, representative_dir = refresh_derived_output_directories(  # noqa: F405
        output_dir
    )

    included, eligible, excluded, log_mels = validate_windows(  # noqa: F405
        csv_paths, threshold, adult_proxy_end, chick_proxy_start
    )
    threshold_yield = build_threshold_yield(eligible)  # noqa: F405
    recording_summary = build_recording_summary(included, eligible)  # noqa: F405
    comparison, enough_recordings = compare_features(  # noqa: F405
        recording_summary, bootstrap_iterations, seed
    )
    representative_manifest = export_representative_audio(  # noqa: F405
        included, representative_dir, representative_count
    )

    included.to_csv(output_dir / "analysis_windows.csv", index=False)
    recording_summary.to_csv(output_dir / "recording_summary.csv", index=False)
    comparison.to_csv(output_dir / "feature_comparison.csv", index=False)
    threshold_yield.to_csv(output_dir / "threshold_yield.csv", index=False)
    excluded.to_csv(output_dir / "excluded_windows.csv", index=False)
    representative_manifest.to_csv(
        representative_dir / "manifest.csv", index=False
    )

    plot_average_spectrograms(log_mels, figures_dir)  # noqa: F405
    plot_representative_spectrograms(  # noqa: F405
        representative_manifest, figures_dir
    )
    plot_feature_distributions(recording_summary, figures_dir)  # noqa: F405
    plot_probability_distributions(eligible, threshold, figures_dir)  # noqa: F405
    plot_date_activity(included, figures_dir)  # noqa: F405
    write_report(  # noqa: F405
        output_dir / "analysis_report.md",
        csv_paths,
        threshold,
        adult_proxy_end,
        chick_proxy_start,
        included,
        recording_summary,
        excluded,
        comparison,
        enough_recordings,
        excluded_recordings,
    )
    print(f"Wrote adult/chick proxy analysis to {output_dir}")
    return output_dir


def run_recording_analysis(
    recordings_dir: Path,
    output_dir: Path,
    model_path: Path = DEPLOYMENT_MODEL_DIR / "model.tflite",
    threshold: float = DEFAULT_THRESHOLD,  # noqa: F405
    adult_proxy_end: date = DEFAULT_ADULT_PROXY_END,  # noqa: F405
    chick_proxy_start: date = DEFAULT_CHICK_PROXY_START,  # noqa: F405
    representative_count: int = 12,
    bootstrap_iterations: int = 5000,
    seed: int = 2021,
    batch_size: int = 32,
    rescan: bool = False,
    predictor: BatchPredictor | None = None,  # noqa: F405
) -> tuple[Path, bool]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    scores_path, excluded_recordings, reused = prepare_scan_cache(  # noqa: F405
        recordings_dir,
        output_dir,
        model_path,
        adult_proxy_end,
        chick_proxy_start,
        batch_size,
        rescan,
        predictor,
    )
    result = run_analysis(
        [scores_path],
        output_dir,
        threshold,
        adult_proxy_end,
        chick_proxy_start,
        representative_count,
        bootstrap_iterations,
        seed,
        excluded_recordings=excluded_recordings,
    )
    return result, reused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare high-confidence adult-proxy and chick-proxy Storm Petrel detections."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Existing all_windows.csv files or directories to search.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        help="Raw WAV/FLAC directory to date-filter, scan, cache, and analyze independently.",
    )
    parser.add_argument(
        "--model", type=Path, default=DEPLOYMENT_MODEL_DIR / "model.tflite"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Rebuild the independent model-score cache.",
    )
    parser.add_argument("--analysis-name", default="adult_proxy_vs_chick_proxy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the configured analysis output directory.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)  # noqa: F405
    parser.add_argument(  # noqa: F405
        "--adult-proxy-end",
        type=parse_iso_date,
        default=DEFAULT_ADULT_PROXY_END,
    )
    parser.add_argument(  # noqa: F405
        "--chick-proxy-start",
        type=parse_iso_date,
        default=DEFAULT_CHICK_PROXY_START,
    )
    parser.add_argument("--representative-count", type=int, default=12)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for compatibility; derived outputs are refreshed automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_name = validate_name(args.analysis_name)  # noqa: F405
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else COMP_CHICK_ADULT_DIR / analysis_name
    )
    print_config()
    print(f"Analysis output: {output_dir}")
    if args.recordings_dir is not None:
        if args.inputs:
            raise ValueError(
                "--recordings-dir cannot be combined with all_windows.csv inputs"
            )
        run_recording_analysis(
            args.recordings_dir,
            output_dir,
            args.model,
            args.threshold,
            args.adult_proxy_end,
            args.chick_proxy_start,
            args.representative_count,
            args.bootstrap_iterations,
            args.seed,
            args.batch_size,
            args.rescan,
        )
    else:
        if not args.inputs:
            raise ValueError(
                "Supply --recordings-dir or at least one all_windows.csv input"
            )
        if args.rescan:
            raise ValueError("--rescan is only valid with --recordings-dir")
        run_analysis(
            args.inputs,
            output_dir,
            args.threshold,
            args.adult_proxy_end,
            args.chick_proxy_start,
            args.representative_count,
            args.bootstrap_iterations,
            args.seed,
            args.force,
        )


if __name__ == "__main__":
    main()
