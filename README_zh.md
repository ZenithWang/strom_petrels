# Storm Petrel 声学分类项目

[English version](README.md)

这个项目用于训练一个二分类的 Storm Petrel 声音检测模型。流程包括：从 BirdNET 风格的检测数据库中截取 3 秒音频片段，生成分层分组的 nested CV 数据集，训练 BirdNET-Analyzer 分类器，并汇总每个 fold 的预测结果和指标。

当前二分类任务是：

- 正类：`storm_petrel`
- 负类：`not_storm_petrel`
- 负类子类型会保留在元数据里，例如 `environment_noise` 和 `other_bird`

## 项目结构

```text
scripts/project_config.py                 本地和服务器共用的路径配置
scripts/data_processing/run_pipeline.py   数据处理总入口
scripts/data_processing/prepare_dataset.py
scripts/data_processing/build_birdnet_fold_dataset.py
scripts/model_training/run_nested_cv.py   nested CV 训练和评估脚本
scripts/results_analysis/summarize_nested_cv.py
workspace/                                生成的 clips、folds、模型输出和汇总结果
```

notebook 文件主要用于检查和分析。可重复运行的正式流程以脚本为准。

服务器上的 working 目录按下面这种平铺结构使用：

```text
/mnt/home/alyzw89/working/project_config.py
/mnt/home/alyzw89/working/data_processing/
/mnt/home/alyzw89/working/model_training/
/mnt/home/alyzw89/working/results_analysis/
/mnt/home/alyzw89/working/workspace/
```

`results_analysis/` 是最后汇总结果需要的文件夹。如果 `/mnt/home/alyzw89/working` 下面现在没有它，需要先复制过去，再运行 `python results_analysis/summarize_nested_cv.py`。

## Python 环境

数据处理环境需要音频和表格处理库：

```bash
pip install librosa soundfile pandas scikit-learn
```

模型训练环境需要 Python 3.11 和 BirdNET-Analyzer 的训练依赖：

```bash
pip install "birdnet-analyzer[train]" pandas scikit-learn tensorflow soundfile
```

当前 Windows 本机可用的训练解释器是：

```powershell
D:\Users\93508\anaconda3\envs\strompetrel\python.exe
```

服务器上使用已经安装好上述依赖的 Python 即可。

## 路径配置

所有脚本都从 `scripts/project_config.py` 读取路径。不要在各个脚本里手写服务器路径；服务器运行时通过环境变量覆盖。

常用环境变量：

```bash
export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR=/mnt/home/alyzw89/working/workspace
export STORM_PETREL_MODEL_ROOT=/mnt/home/alyzw89/working/workspace/models/nested_cv_stratified
```

含义：

- `STORM_PETREL_PROJECT_ROOT`：项目代码所在文件夹，里面应该有 `scripts/`
- `STORM_PETREL_DB_PATH`：包含 BirdNET 检测结果的 SQLite 数据库。README 默认文件名是 `/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db`；如果服务器上的数据库文件名不同，只需要改文件名。
- `STORM_PETREL_AUDIO_ROOT`：原始完整录音所在文件夹，可以包含多层子文件夹
- `STORM_PETREL_WORKSPACE_DIR`：生成 clips、清单、folds 和 fold 数据集的位置
- `STORM_PETREL_MODEL_ROOT`：训练模型、预测结果和指标汇总的输出位置

如果不设置环境变量，默认路径相对于项目根目录解析：

```text
DB_PATH       = scripts/data_processing/storm_petrel_2021_full.db
AUDIO_ROOT    = download/data
WORKSPACE_DIR = workspace
MODEL_ROOT    = workspace/models/nested_cv
```

每个脚本启动时都会打印最终解析到的路径。服务器运行时先检查这些打印结果，确认没有指到本机路径或错误目录。

## 输入数据要求

完整数据处理前，服务器上需要有：

```text
/mnt/home/alyzw89/working/             项目代码
/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db  检测数据库
/mnt/home/alyzw89/data/2021/recordings/  原始完整录音
```

`STORM_PETREL_AUDIO_ROOT` 需要能找到数据库里引用的录音文件。脚本会搜索常见音频格式，例如 `.flac`、`.wav`、`.mp3`、`.ogg`、`.m4a` 和 `.webm`。

## 数据处理

从项目根目录运行数据处理总脚本。

Windows 示例：

```powershell
$env:STORM_PETREL_PROJECT_ROOT="D:\Users\93508\Documents\StromPetrels"
$env:STORM_PETREL_DB_PATH="D:\Users\93508\Documents\StromPetrels\scripts\data_processing\storm_petrel_2021_full.db"
$env:STORM_PETREL_AUDIO_ROOT="D:\path\to\recordings"
$env:STORM_PETREL_WORKSPACE_DIR="D:\Users\93508\Documents\StromPetrels\workspace"
$env:STORM_PETREL_MODEL_ROOT="D:\Users\93508\Documents\StromPetrels\workspace\models\nested_cv_stratified"

D:\Users\93508\anaconda3\envs\strompetrel\python.exe scripts\data_processing\run_pipeline.py
```

Linux/服务器示例：

```bash
cd /mnt/home/alyzw89/working

export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR=/mnt/home/alyzw89/working/workspace
export STORM_PETREL_MODEL_ROOT=/mnt/home/alyzw89/working/workspace/models/nested_cv_stratified

python data_processing/run_pipeline.py
```

运行后会生成或重建：

```text
$STORM_PETREL_WORKSPACE_DIR/download_files.csv
$STORM_PETREL_WORKSPACE_DIR/clips/storm_petrel/
$STORM_PETREL_WORKSPACE_DIR/clips/not_storm_petrel/
$STORM_PETREL_WORKSPACE_DIR/binary_labels.csv
$STORM_PETREL_WORKSPACE_DIR/nested_cv_folds.csv
$STORM_PETREL_WORKSPACE_DIR/nested_cv_datasets/outer_01 ... outer_05
```

fold 分配使用 `StratifiedGroupKFold`：

- 用 `target_label` 做类别分层
- 用 `source_file` 做分组
- 同一个 `source_file` 不会同时出现在同一个 outer fold 的 train 和 test 中
- 因为 `source_file` 不能拆开，所以类别比例不保证完全精确

脚本会打印每个 fold 的质量信息，包括正类数量、负类数量、总数、`source_file` 数量和正类比例。如果某个 fold 只有单一类别，脚本会报错。

## 从已有 clips 重建 folds

如果本地已经有 `workspace/clips` 和 `workspace/binary_labels.csv`，但没有原始完整录音，可以只重建 fold 分配和 BirdNET 文件夹结构：

```bash
python -c "import csv, sys; from pathlib import Path; sys.path.insert(0, str(Path('scripts/data_processing').resolve())); import prepare_dataset, build_birdnet_fold_dataset; rows=list(csv.DictReader(open(prepare_dataset.BINARY_LABELS_PATH, newline='', encoding='utf-8'))); prepare_dataset.write_folds(rows); build_birdnet_fold_dataset.main()"
```

这个命令适合检查分组逻辑。但如果要从原始录音重新截取 clips，仍然必须设置正确的 `STORM_PETREL_AUDIO_ROOT`。

## 模型训练

数据处理完成后，运行 nested CV 训练。

Windows 示例：

```powershell
$env:STORM_PETREL_MODEL_ROOT="D:\Users\93508\Documents\StromPetrels\workspace\models\nested_cv_stratified"
D:\Users\93508\anaconda3\envs\strompetrel\python.exe scripts\model_training\run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
```

Linux/服务器示例：

```bash
export STORM_PETREL_MODEL_ROOT=/mnt/home/alyzw89/working/workspace/models/nested_cv_stratified
python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
```

常用参数：

```text
--outer-folds 01 02 03 04 05   要运行哪些 outer folds
--epochs 50                    BirdNET 训练轮数
--threads 8                    BirdNET 训练线程数
--force                        即使 metrics.csv 已存在也重新训练
--evaluate-only                不重新训练，只用已有 final_model 重新计算预测和指标
```

训练结果会写入 `STORM_PETREL_MODEL_ROOT`：

```text
outer_01/inner_search/.../model.tflite
outer_01/final_model/model.tflite
outer_01/selected_hyperparameters.csv
outer_01/predictions.csv
outer_01/metrics.csv
...
outer_05/...
```

训练脚本会自动把 `NUMBA_CACHE_DIR`、`TMP` 和 `TEMP` 指到 workspace 里面，避免 Windows 下音频加载或 BirdNET 训练时卡在系统临时目录/cache。

## 结果汇总

训练完成后，运行汇总脚本：

```bash
python results_analysis/summarize_nested_cv.py
```

汇总脚本读取 `STORM_PETREL_MODEL_ROOT`，并写出：

```text
$STORM_PETREL_MODEL_ROOT/nested_cv_summary.csv
$STORM_PETREL_MODEL_ROOT/negative_type_error_summary.csv
```

`nested_cv_summary.csv` 包含每个 outer fold 的指标和 mean/std：

- threshold=0.5 的 precision
- threshold=0.5 的 recall
- threshold=0.5 的 F1
- 最优阈值下的 precision/recall/F1
- AUPRC
- AUROC

`negative_type_error_summary.csv` 汇总负类子类型中的 false positives。

## 服务器完整运行示例

服务器上一次完整运行通常是：

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

服务器上快速 smoke test 可以只跑一个 fold 和少量 epochs：

```bash
python model_training/run_nested_cv.py --outer-folds 04 --epochs 5 --threads 4
python results_analysis/summarize_nested_cv.py
```

## 当前本地状态

当前 Windows workspace 里，默认的 `download/data` 下没有原始完整录音。因此本机不能直接从 raw recordings 完整重建 clips，除非设置 `STORM_PETREL_AUDIO_ROOT` 指向真实录音目录。

当前本机已有：

```text
workspace/clips
workspace/binary_labels.csv
workspace/nested_cv_folds.csv
workspace/nested_cv_datasets
workspace/models/nested_cv_stratified
```

最新正式训练结果在：

```text
workspace/models/nested_cv_stratified
```

如果要在服务器上做真正端到端重建，请确认 `STORM_PETREL_AUDIO_ROOT` 指向完整录音所在目录。





