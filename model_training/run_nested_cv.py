"""运行 grouped nested cross-validation 的 BirdNET 训练、推理和评估。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import DATASET_ROOT, FOLDS_PATH, MODEL_ROOT, NUMBA_CACHE_DIR, PROJECT_ROOT, TMP_DIR
SAMPLE_RATE = 48_000
SAMPLE_LENGTH = SAMPLE_RATE * 3
POSITIVE_CLASS = "storm_petrel"


@dataclass(frozen=True)
class HyperParameters:
    """保存一组可复现的 BirdNET 分类头超参数。"""

    name: str
    learning_rate: str
    hidden_units: str
    dropout: str

# 预定义的超参数组合列表，用于在嵌套交叉验证中进行搜索和比较。
HYPERPARAMETERS = [
    HyperParameters("baseline", "0.0001", "0", "0.0"),
    HyperParameters("hidden256_dropout02", "0.00005", "256", "0.2"),
    HyperParameters("hidden256_dropout03", "0.0001", "256", "0.3"),
    HyperParameters("hidden512_dropout03", "0.00005", "512", "0.3"),
]

# 这个方法是用来配置Windows系统下的Numba缓存和临时目录的。它会创建指定的缓存目录和临时目录（如果不存在），并将这些目录的路径设置为环境变量，以便Numba在运行时使用这些目录进行缓存和临时文件存储。
def configure_windows_cache() -> None:
    """把 Numba 与临时文件目录固定到项目 workspace。"""

    NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_DIR)
    os.environ["TMP"] = str(TMP_DIR)
    os.environ["TEMP"] = str(TMP_DIR)

#这个方法是用来解析命令行参数的。它使用了argparse模块来定义和处理命令行选项。具体来说，它定义了以下参数：
# --outer-folds: 一个可选参数，允许用户指定要运行的外部折叠ID列表，默认值为["04"]。用户可以通过空格分隔的方式提供多个折叠ID。
# --epochs: 一个可选参数，指定训练的轮数，默认值从环境变量BIRDNET_EPOCHS获取，如果未设置则默认为"50"。
# --threads: 一个可选参数，指定训练时使用的线程数，默认值从环境变量BIRDNET_THREADS获取，如果未设置则默认为"8"。
# --force: 一个布尔标志，如果设置了这个标志，程序将重新运行折叠，即使metrics.csv文件已经存在。
# --evaluate-only: 一个布尔标志，如果设置了这个标志，程序将不会进行训练，而是从现有的final_model/model.tflite文件中重新生成外部测试预测和指标。
def parse_args() -> argparse.Namespace:
    """解析 nested-CV 训练与仅评估模式的命令行参数。"""

    parser = argparse.ArgumentParser(description="Run grouped nested-CV BirdNET training.")
    parser.add_argument(
        "--outer-folds",
        nargs="+",
        default=["04"],
        help="Outer fold ids to run, for example: --outer-folds 04 or --outer-folds 01 02 03 04 05.",
    )
    parser.add_argument("--epochs", default=os.environ.get("BIRDNET_EPOCHS", "50"))
    parser.add_argument("--threads", default=os.environ.get("BIRDNET_THREADS", "8"))
    parser.add_argument("--force", action="store_true", help="Re-run folds even when metrics.csv already exists.")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Do not train; regenerate outer test predictions and metrics from existing final_model/model.tflite files.",
    )
    return parser.parse_args()


def normalize_fold(value: str) -> str:
    """把 fold 编号规范化为两位字符串。"""

    return f"{int(value):02d}"


def available_outer_folds() -> list[str]:
    """返回当前数据集目录中可用的 outer fold 编号。"""

    return sorted(path.name.replace("outer_", "") for path in DATASET_ROOT.glob("outer_*") if path.is_dir())

#检查输入的外部折叠目录是否存在，并检查折叠清单文件是否存在。如果缺少任何外部折叠目录或折叠清单文件，将引发FileNotFoundError异常。
def validate_inputs(outer_folds: list[str]) -> None:
    """在训练前验证请求的 fold 目录和 fold 清单。"""

    missing = [fold for fold in outer_folds if not (DATASET_ROOT / f"outer_{fold}").is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing outer fold dataset directories: {missing}")
    if not FOLDS_PATH.is_file():
        raise FileNotFoundError(f"Missing fold manifest: {FOLDS_PATH}")

#这里的run_birdnet_training函数是用来运行BirdNET训练的。它接受训练和验证数据的目录、输出前缀、超参数、缓存文件路径、缓存模式、训练轮数和线程数作为参数。
# 函数首先检查训练和验证目录是否存在，如果不存在则跳过训练。然后，它构建一个命令列表，用于调用BirdNET的训练模块，并传递相应的参数。
# 接着，它创建输出目录和缓存目录（如果不存在），并打印出要运行的训练命令。最后，它使用subprocess.run()执行命令，并返回训练是否成功（即返回码是否为0）。
def run_birdnet_training(
    train_dir: Path,
    validation_dir: Path,
    output_prefix: Path,
    params: HyperParameters,
    cache_file: Path,
    cache_mode: str,
    epochs: str,
    threads: str,
) -> bool:
    """调用 BirdNET-Analyzer 训练命令，并返回进程是否成功。"""

    if not train_dir.is_dir() or not validation_dir.is_dir():
        print(f"Skip training; missing directory: train={train_dir}, validation={validation_dir}")
        return False

    command = [
        sys.executable,
        "-m",
        "birdnet_analyzer.train",
        str(train_dir),
        "-o",
        str(output_prefix),
        "--test_data",
        str(validation_dir),
        "--epochs",
        str(epochs),
        "--threads",
        str(threads),
        "--learning_rate",
        params.learning_rate,
        "--hidden_units",
        params.hidden_units,
        "--dropout",
        params.dropout,
        "--model_save_mode",
        "replace",
        "--model_format",
        "tflite",
        "--cache_mode",
        cache_mode,
        "--cache_file",
        str(cache_file),
    ]
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    print("Running training command:")
    print(" ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    print(f"Training return code: {result.returncode}")
    return result.returncode == 0


def metric_from_evaluation(path: Path) -> float:
    """严格读取 BirdNET 评估 CSV 中 storm-petrel 类的 F1 指标。"""

    if not path.is_file():
        raise FileNotFoundError(f"Missing BirdNET evaluation file: {path}")
    rows = pd.read_csv(path)
    if rows.empty:
        raise ValueError(f"BirdNET evaluation file is empty: {path}")
    if "Class" in rows.columns:
        selected = rows[rows["Class"] == POSITIVE_CLASS]
        if selected.empty:
            raise ValueError(f"BirdNET evaluation file has no {POSITIVE_CLASS!r} row: {path}")
        row = selected.iloc[0]
    else:
        row = rows.iloc[0]
    for key in ["F1 Score (opt)", "F1 Score (0.5)", "f1_opt", "f1_0_5"]:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    raise ValueError(f"BirdNET evaluation file has no supported F1 metric: {path}")


def run_inner_search(outer_fold: str, epochs: str, threads: str) -> HyperParameters:
    """运行一个 outer fold 的内层超参数搜索并返回最佳配置。"""

    outer_dataset = DATASET_ROOT / f"outer_{outer_fold}"
    outer_model_dir = MODEL_ROOT / f"outer_{outer_fold}"
    summary_rows = []
    inner_dirs = sorted(path for path in outer_dataset.glob("inner_*") if path.is_dir())
    if not inner_dirs:
        print(f"outer {outer_fold} has no inner folds; using baseline hyperparameters.")
        return HYPERPARAMETERS[0]

    for params in HYPERPARAMETERS:
        scores = []
        for inner_dir in inner_dirs:
            output_prefix = outer_model_dir / "inner_search" / params.name / inner_dir.name / "model"
            cache_file = outer_model_dir / "cache" / f"{inner_dir.name}.npz"
            cache_mode = "save" if not cache_file.is_file() else "load"
            ok = run_birdnet_training(
                inner_dir / "train",
                inner_dir / "validation",
                output_prefix,
                params,
                cache_file,
                cache_mode,
                epochs,
                threads,
            )
            if not ok:
                raise RuntimeError(
                    f"BirdNET inner training failed for outer={outer_fold}, "
                    f"inner={inner_dir.name}, params={params.name}"
                )
            score = metric_from_evaluation(output_prefix.with_name(f"{output_prefix.name}_evaluation.csv"))
            scores.append(score)
            summary_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_dir.name.replace("inner_", ""),
                    "params": params.name,
                    "f1_opt": score,
                }
            )
        summary_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": "mean",
                "params": params.name,
                "f1_opt": sum(scores) / len(scores) if scores else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = outer_model_dir / "inner_search_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    best_name = summary[summary["inner_fold"] == "mean"].sort_values("f1_opt", ascending=False).iloc[0]["params"]
    best = next(item for item in HYPERPARAMETERS if item.name == best_name)
    print(f"Selected hyperparameters for outer {outer_fold}: {best}")
    return best


def load_audio(path: Path) -> np.ndarray | None:
    """读取严格三秒、48 kHz 的评估音频；不对短片段补零。"""

    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        print(f"Skipping sample-rate mismatch: {path} ({sample_rate})")
        return None
    if len(audio) < SAMPLE_LENGTH:
        raise ValueError(
            f"Evaluation clip is shorter than the required 3-second window; padding is disabled: "
            f"{path} ({len(audio)}/{SAMPLE_LENGTH} samples)"
        )
    return audio[:SAMPLE_LENGTH].astype("float32")


def predict_tflite(model_path: Path, data_dir: Path) -> list[dict[str, str]]:
    """使用 TFLite 模型为 BirdNET 文件夹数据集生成逐 clip 预测。"""

    if not model_path.is_file() or not data_dir.is_dir():
        print(f"No model or data directory to predict: model={model_path}, data={data_dir}")
        return []

    from tensorflow import lite as tflite

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    labels_path = model_path.with_name(f"{model_path.stem}_Labels.txt")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if POSITIVE_CLASS not in labels:
        raise ValueError(f"{POSITIVE_CLASS!r} is not in model labels: {labels_path}")
    positive_index = labels.index(POSITIVE_CLASS)
    rows: list[dict[str, str]] = []

    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        true_label = 1 if folder.name == POSITIVE_CLASS else 0
        for path in sorted(folder.glob("*.wav")):
            audio = load_audio(path)
            if audio is None:
                continue
            interpreter.resize_tensor_input(input_details["index"], [1, SAMPLE_LENGTH])
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details["index"], audio.reshape(1, SAMPLE_LENGTH))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
            score = float(output[positive_index])
            rows.append(
                {
                    "file": str(path),
                    "class_folder": folder.name,
                    "true_label": str(true_label),
                    "score": f"{score:.8f}",
                }
            )
    return rows


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """在固定网格上寻找 F1 最高的分类阈值。"""

    from sklearn.metrics import f1_score

    best = (0.5, -1.0)
    for threshold in np.linspace(0.05, 0.95, 91):
        score = f1_score(y_true, scores >= threshold, zero_division=0)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def write_evaluation(rows: list[dict[str, str]], output_prefix: Path, model_path: Path, data_dir: Path) -> dict[str, str]:
    """写出逐 clip 预测和二分类评估指标。"""

    if not rows:
        print("No predictions; skipping metric write.")
        return {}

    from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_prefix.with_name(f"{output_prefix.name}_predictions.csv")
    metrics_path = output_prefix.with_name(f"{output_prefix.name}_metrics.csv")
    pd.DataFrame(rows).to_csv(predictions_path, index=False, encoding="utf-8")

    y_true = np.array([int(row["true_label"]) for row in rows])
    scores = np.array([float(row["score"]) for row in rows])
    threshold, _ = best_threshold(y_true, scores)
    y_pred_default = scores >= 0.5
    y_pred_opt = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt, labels=[0, 1]).ravel()
    metrics = {
        "model": str(model_path),
        "data_dir": str(data_dir),
        "samples": len(rows),
        "precision_0_5": precision_score(y_true, y_pred_default, zero_division=0),
        "recall_0_5": recall_score(y_true, y_pred_default, zero_division=0),
        "f1_0_5": f1_score(y_true, y_pred_default, zero_division=0),
        "precision_opt": precision_score(y_true, y_pred_opt, zero_division=0),
        "recall_opt": recall_score(y_true, y_pred_opt, zero_division=0),
        "f1_opt": f1_score(y_true, y_pred_opt, zero_division=0),
        "auprc": average_precision_score(y_true, scores),
        "auroc": roc_auc_score(y_true, scores),
        "optimal_threshold": threshold,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8")
    print(pd.DataFrame([metrics]).to_string(index=False))
    return metrics


def run_outer_fold(outer_fold: str, epochs: str, threads: str, force: bool) -> None:
    """完成一个 outer fold 的内层搜索、最终训练和 held-out 评估。"""

    outer_dataset = DATASET_ROOT / f"outer_{outer_fold}"
    outer_model_dir = MODEL_ROOT / f"outer_{outer_fold}"
    if (outer_model_dir / "metrics.csv").is_file() and not force:
        print(f"outer {outer_fold} already has metrics.csv; skipping. Use --force to rerun.")
        return

    best_params = run_inner_search(outer_fold, epochs, threads)
    selected = pd.DataFrame([{**vars(best_params), "outer_fold": outer_fold}])
    selected_path = outer_model_dir / "selected_hyperparameters.csv"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_path, index=False, encoding="utf-8")

    final_prefix = outer_model_dir / "final_model" / "model"
    cache_file = outer_model_dir / "cache" / "final_train.npz"
    cache_mode = "save" if not cache_file.is_file() else "load"
    ok = run_birdnet_training(
        outer_dataset / "final_train",
        outer_dataset / "final_train",
        final_prefix,
        best_params,
        cache_file,
        cache_mode,
        epochs,
        threads,
    )
    if not ok:
        print(f"outer {outer_fold} final training failed; skipping test evaluation.")
        return

    test_dir = outer_dataset / "test"
    model_path = final_prefix.with_suffix(".tflite")
    output_prefix = outer_model_dir / "outer_test"
    rows = predict_tflite(model_path, test_dir)
    metrics = write_evaluation(rows, output_prefix, model_path, test_dir)
    if metrics:
        shutil.copy2(output_prefix.with_name(f"{output_prefix.name}_predictions.csv"), outer_model_dir / "predictions.csv")
        shutil.copy2(output_prefix.with_name(f"{output_prefix.name}_metrics.csv"), outer_model_dir / "metrics.csv")


def evaluate_outer_fold(outer_fold: str) -> None:
    """复用已有最终模型，重新生成一个 outer fold 的预测和指标。"""

    outer_dataset = DATASET_ROOT / f"outer_{outer_fold}"
    outer_model_dir = MODEL_ROOT / f"outer_{outer_fold}"
    model_path = outer_model_dir / "final_model" / "model.tflite"
    test_dir = outer_dataset / "test"
    output_prefix = outer_model_dir / "outer_test"
    rows = predict_tflite(model_path, test_dir)
    metrics = write_evaluation(rows, output_prefix, model_path, test_dir)
    if metrics:
        shutil.copy2(output_prefix.with_name(f"{output_prefix.name}_predictions.csv"), outer_model_dir / "predictions.csv")
        shutil.copy2(output_prefix.with_name(f"{output_prefix.name}_metrics.csv"), outer_model_dir / "metrics.csv")


def main() -> None:
    """执行命令行请求的 outer folds。"""

    args = parse_args()
    configure_windows_cache()
    outer_folds = [normalize_fold(value) for value in args.outer_folds]
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Model root: {MODEL_ROOT}")
    print(f"Available outer folds: {available_outer_folds()}")
    print(f"Requested outer folds: {outer_folds}")
    print(f"Epochs: {args.epochs}; threads: {args.threads}")
    validate_inputs(outer_folds)
    for outer_fold in outer_folds:
        print()
        print(f"=== Running outer fold {outer_fold} ===")
        if args.evaluate_only:
            evaluate_outer_fold(outer_fold)
        else:
            run_outer_fold(outer_fold, args.epochs, args.threads, args.force)


if __name__ == "__main__":
    main()
