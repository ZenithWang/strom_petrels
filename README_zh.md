# Storm Petrel 声学分类项目

[English version](README.md)

这个项目用于训练一个二分类的 Storm Petrel 声音检测模型。流程包括：从 BirdNET 风格的检测数据库中截取 3 秒音频片段，生成分层分组的 nested CV 数据集，训练 BirdNET-Analyzer 分类器，并汇总每个 fold 的预测结果和指标。

当前二分类任务是：

- 正类：`storm_petrel`
- 负类：`not_storm_petrel`
- 负类子类型会保留在元数据里，例如 `environment_noise` 和 `other_bird`

## 项目结构

```text
project_config.py                         本地和服务器共用的路径配置
data_processing/run_pipeline.py           数据处理总入口
data_processing/prepare_dataset.py
data_processing/build_birdnet_fold_dataset.py
model_training/run_nested_cv.py           nested CV 训练和评估脚本
model_training/train_deployment_model.py
long_recording_analysis/                  长录音流式推理与候选截取
comp_chick_adult/                         成鸟/雏鸟日期代理声学比较
chick_screening/                          双模型雏鸟叫声探索性筛选
results_analysis/summarize_nested_cv.py
workspace/                                生成的 clips、folds、模型输出和汇总结果
```

notebook 文件主要用于检查和分析。可重复运行的正式流程以脚本为准。

本地和服务器都使用下面同一种物理目录结构：

```text
/mnt/home/alyzw89/working/project_config.py
/mnt/home/alyzw89/working/data_processing/
/mnt/home/alyzw89/working/model_training/
/mnt/home/alyzw89/working/long_recording_analysis/
/mnt/home/alyzw89/working/comp_chick_adult/
/mnt/home/alyzw89/working/results_analysis/
/mnt/home/alyzw89/working/workspace/
```

`results_analysis/` 是最后汇总结果需要的文件夹。

完整流程按相同物理结构同步时，先预览再执行：

```bash
python sync_project.py alyzw89@strange:~/working
python sync_project.py alyzw89@strange:~/working --apply
```

只同步配置、环境说明和五个功能代码目录。`workspace`、`.env`、数据库、音频、模型、缓存、tests、
documents、顶层探索 notebook 以及本地 IDE/agent 文件都保留在各自机器。只有需要清理五个代码
目录中的陈旧文件时才使用 `--apply --delete`。

## 服务器快速运行：成鸟/雏鸟比较

本地同步完成后，登录服务器并运行：

```bash
ssh alyzw89@strange
cd ~/working
conda activate strompetrel-birdnet

ls workspace/models/deployment/model.tflite
ls workspace/models/deployment/model_Labels.txt

python comp_chick_adult/run_analysis.py \
  --recordings-dir /mnt/home/alyzw89/data/2021/recordings \
  --analysis-name adult_proxy_vs_chick_proxy \
  --threshold 0.80 \
  --adult-proxy-end 2021-05-31 \
  --chick-proxy-start 2021-07-23
```

不需要先运行通用长录音分析。当 `project_config.py` 和 `workspace/` 都在 `~/working` 下时，
也不需要设置项目或 workspace 环境变量。第一次运行会扫描两个日期阶段，并把分数缓存到
`workspace/comp_chick_adult/adult_proxy_vs_chick_proxy/scan/`；相同配置再次运行会直接复用。
只有录音、部署模型或日期边界改变后才添加 `--rescan`。
如果启动时打印的解析路径仍指向 `~/working` 之外，需要清除当前 shell 或 `~/.bashrc` 中遗留的
`STORM_PETREL_*`，因为 shell 变量优先于 `.env` 和默认值。

如果其他完整流程还需要数据库和音频根目录，可以在服务器创建仅属于该机器的 `~/working/.env`：

```dotenv
STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
```

`.env` 不参与同步，因为两台机器的实际路径可能不同。

## Python 环境

仓库根目录的 `environment.yml` 同时包含数据处理、模型训练和 notebook 所需依赖。首次创建环境：

```powershell
conda env create -f environment.yml
conda activate strompetrel
python verify_environment.py
```

如果 `strompetrel` 环境已经存在，更新即可：

```powershell
conda env update -n strompetrel -f environment.yml
```

环境固定使用 Python 3.11 和 BirdNET-Analyzer 2.4.0 的训练扩展。声学比较图还使用
`matplotlib`，已经显式加入 `environment.yml`。Linux 和服务器使用相同命令，只需把脚本路径中的反斜杠改成正斜杠。

### PyCharm

打开仓库后进入 `File | Settings | Project | Python Interpreter`，选择 `Add Interpreter | Add Local Interpreter | Conda Environment | Existing environment`，然后选择：

```powershell
D:\Users\93508\anaconda3\envs\strompetrel\python.exe
```

当前 Linux/Miniforge 环境的解释器路径是：

```text
/home/zenith/miniforge3/envs/strompetrel/bin/python
```

选择解释器后，PyCharm 的运行配置列表中会自动出现：

- `Verify Environment`：检查 Python 和所有关键依赖
- `Data Pipeline`：运行数据准备流程
- `Nested CV Training`：默认训练 outer fold 04
- `Summarize Results`：汇总已有训练结果

运行数据处理前，需要在 `Data Pipeline` 的环境变量中补充真实的 `STORM_PETREL_DB_PATH` 和 `STORM_PETREL_AUDIO_ROOT`。共享配置不会写入个人数据路径。

## 路径配置

所有脚本都从根目录的 `project_config.py` 读取路径，本地和服务器目录在物理上完全一致。
如果代码目录和 `workspace/` 都在同一个 working 目录下，不需要设置项目或 workspace 环境变量；
只有数据或输出位于其他位置时才需要覆盖。

运行路径可以一次写入根目录 `.env`，不需要每次打开终端都执行 `export`：

```bash
cp .env.example .env
# 编辑 .env，把示例替换成当前机器的绝对路径。
```

`project_config.py` 会自动读取 `.env`；shell 中已有的变量优先级更高，真实 `.env` 已加入 Git 忽略。
只有需要把 env 文件放在别处时才设置 `STORM_PETREL_ENV_FILE`。

常用环境变量：

```bash
cd /mnt/home/alyzw89/working
export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_MODEL_ROOT="$STORM_PETREL_WORKSPACE_DIR/models/nested_cv_balanced"
export STORM_PETREL_DEPLOYMENT_MODEL_DIR="$STORM_PETREL_WORKSPACE_DIR/models/deployment"
export STORM_PETREL_LONG_RECORDING_ANALYSIS_DIR="$STORM_PETREL_WORKSPACE_DIR/long_recording_analysis"
export STORM_PETREL_COMP_CHICK_ADULT_DIR="$STORM_PETREL_WORKSPACE_DIR/comp_chick_adult"
export STORM_PETREL_CHICK_SCREENING_DIR="$STORM_PETREL_WORKSPACE_DIR/chick_screening"
```

含义：

- `STORM_PETREL_PROJECT_ROOT`：本地仓库根目录，或服务器上的平铺 working 目录
- `STORM_PETREL_DB_PATH`：包含 BirdNET 检测结果的 SQLite 数据库。README 默认文件名是 `/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db`；如果服务器上的数据库文件名不同，只需要改文件名。
- `STORM_PETREL_AUDIO_ROOT`：原始完整录音所在文件夹，可以包含多层子文件夹
- `STORM_PETREL_WORKSPACE_DIR`：生成 clips、清单、folds 和 fold 数据集的位置
- `STORM_PETREL_MODEL_ROOT`：训练模型、预测结果和指标汇总的输出位置
- `STORM_PETREL_DEPLOYMENT_MODEL_DIR`：使用全部 clips 训练的部署模型输出位置
- `STORM_PETREL_LONG_RECORDING_ANALYSIS_DIR`：长录音窗口分数、检测结果和候选 WAV 输出位置
- `STORM_PETREL_COMP_CHICK_ADULT_DIR`：成鸟/雏鸟代理比较的表格、图、报告和代表音频输出位置
- `STORM_PETREL_CHICK_SCREENING_DIR`：日期专用模型、双模型分数和雏鸟筛选分析输出位置

如果不设置环境变量，默认路径相对于项目根目录解析：

```text
DB_PATH       = data_processing/storm_petrel_2021_full.db
AUDIO_ROOT    = download/data
WORKSPACE_DIR = workspace
MODEL_ROOT    = workspace/models/nested_cv
DEPLOYMENT_MODEL_DIR = workspace/models/deployment
LONG_RECORDING_ANALYSIS_DIR = workspace/long_recording_analysis
COMP_CHICK_ADULT_DIR = workspace/comp_chick_adult
CHICK_SCREENING_DIR = workspace/chick_screening
```

训练脚本会在检查 fold 目录之前自动打印项目、数据集和模型输出路径。

## 输入数据要求

完整数据处理前，服务器上需要有：

```text
/mnt/home/alyzw89/working/             项目代码
/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db  检测数据库
/mnt/home/alyzw89/data/2021/recordings/  原始完整录音
```

`STORM_PETREL_AUDIO_ROOT` 需要能找到数据库里引用的录音文件。脚本会递归扫描目录，并按文件名或不带扩展名的 stem 匹配。

## 从零开始完整运行

所有命令都应从项目根目录执行。核心可复现管线和两个生物学研究分支共用路径配置，但输出彼此独立。

### 1. 一次性配置环境和路径

```bash
conda env create -f environment.yml
conda activate strompetrel
cp .env.example .env
# 编辑 .env，填写真实的数据库、录音、项目和 workspace 路径。
python verify_environment.py
```

已有服务器环境可以改为执行 `conda activate strompetrel-birdnet`。每次新建或更新环境后，都应运行
`python verify_environment.py`。

### 2. 核心数据集、nested CV 和论文指标

```bash
python data_processing/run_pipeline.py
python model_training/run_nested_cv.py \
  --outer-folds 01 02 03 04 05 \
  --epochs 50 \
  --threads 8
python results_analysis/summarize_nested_cv.py
```

这条完整路径会生成 2,000 个 clip 的二分类数据集、grouped nested-CV 模型、OOF 预测、错误汇总，
以及表 5.3 使用的录音级 clustered bootstrap。

### 3. 部署模型和长录音检测

```bash
python model_training/train_deployment_model.py --epochs 50 --threads 8
python long_recording_analysis/run_pipeline.py \
  /absolute/path/to/recordings \
  --run-name deployment_scan \
  --threshold 0.80 \
  --batch-size 32
```

### 4. 独立的成鸟/雏鸟日期代理比较

```bash
python comp_chick_adult/run_analysis.py \
  --recordings-dir /absolute/path/to/recordings \
  --analysis-name adult_proxy_vs_chick_proxy \
  --threshold 0.80 \
  --adult-proxy-end 2021-05-31 \
  --chick-proxy-start 2021-07-23 \
  --bootstrap-iterations 5000 \
  --seed 2021
```

该分支使用部署模型，并维护自己的日期过滤扫描缓存，所以不要求先运行第 3 步的通用长录音扫描。

### 5. 探索性双模型雏鸟叫声筛选

```bash
python chick_screening/run_pipeline.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00" \
  --recordings-dir /absolute/path/to/chick_proxy_recordings \
  --epochs 50 \
  --threads 8 \
  --adult-threshold 0.80 \
  --chick-threshold 0.80
```

这个独立分支会准备日期专用训练集、分别训练成鸟期和雏鸟期模型、用两个模型扫描相同录音，
并分析分数差异。日期代理标签不能证明发声个体的真实年龄。

### 执行顺序与断点重跑规则

```text
环境/路径配置
    -> 数据准备
        -> grouped nested-CV 训练 -> 结果汇总/表 5.3
        -> 部署模型训练 -> 长录音扫描
                        \-> 成鸟/雏鸟代理比较
        -> 日期专用数据集 -> 两个日期模型 -> 双模型扫描 -> 雏鸟筛选分析
```

- 只有数据库、原始录音、筛选约束或 fold 逻辑改变时，才需要重新运行数据准备。
- 已有 final nested-CV 模型而只需重建 outer 预测/指标时，使用 `--evaluate-only`。
- 长录音只改变阈值时，只重跑候选截取，不需要再次推理。
- 只有录音、模型或代理日期边界改变时，成鸟/雏鸟比较才需要 `--rescan`。
- `--force` 会替换既有输出；执行前先核对脚本打印的目标路径。

## 数据处理

从项目根目录运行数据处理总脚本。

Windows 示例：

```powershell
$env:STORM_PETREL_PROJECT_ROOT="D:\Users\93508\Documents\StromPetrels"
$env:STORM_PETREL_DB_PATH="D:\Users\93508\Documents\StromPetrels\data_processing\storm_petrel_2021_full.db"
$env:STORM_PETREL_AUDIO_ROOT="D:\path\to\recordings"
$env:STORM_PETREL_WORKSPACE_DIR="D:\Users\93508\Documents\StromPetrels\workspace"
$env:STORM_PETREL_MODEL_ROOT="D:\Users\93508\Documents\StromPetrels\workspace\models\nested_cv_balanced"

D:\Users\93508\anaconda3\envs\strompetrel\python.exe data_processing\run_pipeline.py
```

Linux/服务器示例：

```bash
cd /mnt/home/alyzw89/working

export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_MODEL_ROOT="$STORM_PETREL_WORKSPACE_DIR/models/nested_cv_balanced"

python data_processing/run_pipeline.py
```

运行后会生成或重建：

```text
$STORM_PETREL_WORKSPACE_DIR/download_files.csv
$STORM_PETREL_WORKSPACE_DIR/clips/storm_petrel/
$STORM_PETREL_WORKSPACE_DIR/clips/not_storm_petrel/
$STORM_PETREL_WORKSPACE_DIR/binary_labels.csv
$STORM_PETREL_WORKSPACE_DIR/nested_cv_folds.csv
$STORM_PETREL_WORKSPACE_DIR/fold_balance_report.csv
$STORM_PETREL_WORKSPACE_DIR/nested_cv_datasets/outer_01 ... outer_05
```

数据集目标为 1000 个 Storm Petrel 正类 clips 和 1000 个负类 clips。负类中环境噪声和其他鸟类各 500 个。采样在以下约束下优先选择置信度最高的正类：

- 同一录音中所有入选的 3 秒窗口互不重叠
- 每个 `source_file` 的正类和负类数量严格相等
- 每个 `source_file` 每类最多贡献 73 个 clips
- 四种负类标签分别严格选择 250 个 clips

73 对的来源上限保证三个最大来源组仍能放进允许的最大 outer test fold（`3 * 73 * 2 = 438`）。

fold 使用确定性混合整数优化分配：

- 用 `source_file` 做分组
- 同一个 `source_file` 不会同时出现在同一个 outer fold 的 train 和 test 中
- 所有 outer/inner 的 train、validation 和 test 正类比例严格为 50%
- held-out fold 样本数必须在理想大小的 ±10% 内
- held-out 样本数继续向相等分布优化

候选数量和 fold 可行性会在删除旧 clips 之前检查。如果不足 1000 对来源内平衡且互不重叠的样本，或 grouped folds 无法满足约束，脚本会停止并报错。成功时会把统计写入 `fold_balance_report.csv`。

## 从已有 clips 重建 folds

如果本地已经有 `workspace/clips` 和 `workspace/binary_labels.csv`，但没有原始完整录音，可以只重建 fold 分配和 BirdNET 文件夹结构：

```bash
python -c "import csv, sys; from pathlib import Path; sys.path.insert(0, str(Path('data_processing').resolve())); import prepare_dataset, build_birdnet_fold_dataset; rows=list(csv.DictReader(open(prepare_dataset.BINARY_LABELS_PATH, newline='', encoding='utf-8'))); prepare_dataset.write_folds(rows); build_birdnet_fold_dataset.main()"
```

这个命令适合检查分组逻辑，但已有标签中每个来源的正负类数量必须已经相等。如果要从原始录音重新截取 clips，仍然必须设置正确的 `STORM_PETREL_AUDIO_ROOT`。

## 模型训练

数据处理完成后，运行 nested CV 训练。

Windows 示例：

```powershell
$env:STORM_PETREL_PROJECT_ROOT="D:\Users\93508\Documents\StromPetrels"
$env:STORM_PETREL_WORKSPACE_DIR="$env:STORM_PETREL_PROJECT_ROOT\workspace"
$env:STORM_PETREL_MODEL_ROOT="$env:STORM_PETREL_WORKSPACE_DIR\models\nested_cv_balanced"
D:\Users\93508\anaconda3\envs\strompetrel\python.exe model_training\run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
```

Linux/服务器示例：

```bash
cd /mnt/home/alyzw89/working
export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_MODEL_ROOT="$STORM_PETREL_WORKSPACE_DIR/models/nested_cv_balanced"

python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
```

平衡数据集必须使用新的模型目录。不要指向旧的 `nested_cv_stratified`：其中的 `metrics.csv` 会导致 fold 被跳过，已有 `.npz` 缓存也属于旧数据集。

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

## 部署模型与长录音检测

使用已经选定的数值最优超参数，从全部 1000 个正类和 1000 个负类 clips 训练单个部署分类器：

```bash
python model_training/train_deployment_model.py --epochs 50 --threads 8
```

固定参数为学习率 `0.0001`、256 个 hidden units、dropout `0.3`、batch size 32，以及随机 20% 内部验证集。`STORM_PETREL_DEPLOYMENT_MODEL_DIR` 会包含 `model.tflite`、标签、参数和样本数文件，以及 `training_config.json`。只有显式加入 `--force` 才会覆盖已有部署输出。

以互不重叠的 3 秒窗口递归扫描长 WAV/FLAC 录音并截取高置信候选：

```bash
python long_recording_analysis/run_pipeline.py \
  /path/to/long_recordings \
  --run-name all_recordings_step3 \
  --threshold 0.80
```

窗口和默认步长都是 3 秒，录音末尾不足 3 秒的部分会忽略。每个过阈值窗口独立生成一个严格
3 秒、48 kHz 单声道候选；相邻命中不合并，
也不添加上下文。`detections.csv` 保留原字段，且每行 `window_count=1`。旧的 1.5 秒步长结果
不会自动转换，必须换一个新的 run name 重新扫描。
Outer-CV 评估采用相同的完整窗口政策：测试 clip 不足 3 秒时直接报错，不做 padding，
因为这意味着准备好的评估数据已经损坏或不完整。

调整阈值不会重复模型推理，直接对保存的分数重新提取：

```bash
python long_recording_analysis/extract_candidates.py \
  workspace/long_recording_analysis/all_recordings_step3/all_windows.csv \
  --threshold 0.80 \
  --force
```

## 成鸟代理与雏鸟代理比较

`comp_chick_adult` 与通用长录音检测完全独立运行，并按日期代理比较高置信模型命中：
2021-05-31 及以前为 `adult_proxy`，2021-07-23 及以后为 `chick_proxy`，中间孵卵/过渡期排除。
它可以直接扫描原始录音：

```bash
python comp_chick_adult/run_analysis.py \
  --recordings-dir /mnt/home/alyzw89/data/2021/recordings \
  --analysis-name adult_proxy_vs_chick_proxy \
  --threshold 0.80 \
  --adult-proxy-end 2021-05-31 \
  --chick-proxy-start 2021-07-23
```

推理前会先按文件名日期筛选，只扫描成鸟代理和雏鸟代理阶段；过渡期和无法解析日期的录音不消耗
模型推理，并写入 `scan/excluded_recordings.csv`。
互不重叠的三秒窗口分数和扫描配置保存在 `scan/` 中；后续运行默认复用匹配缓存，并自动刷新
派生表格、图和代表音频。如果录音、模型或日期边界发生变化，脚本会停止并要求显式添加
`--rescan`。

仍可直接复用已有窗口分数，不触发独立扫描：

```bash
python comp_chick_adult/run_analysis.py \
  workspace/long_recording_analysis/RUN_NAME/all_windows.csv \
  --analysis-name adult_proxy_vs_chick_proxy \
  --threshold 0.80
```

可以传入一个或多个 CSV，也可以传入目录并递归查找 `all_windows.csv`。脚本按录音路径和开始
时间去重，从原始录音读取完整窗口并统一为 48 kHz 单声道；它不会因为样本不足自动降低阈值。

结果写入 `workspace/comp_chick_adult/adult_proxy_vs_chick_proxy/`，包括 `scan/`、
`analysis_windows.csv`、`recording_summary.csv`、`feature_comparison.csv`、
`threshold_yield.csv`、`excluded_windows.csv`、`analysis_report.md`、
`representative_audio/` 和 `figures/`。正式特征分析只使用指定阈值（默认 0.80），产量表另外
列出 0.20、0.50 和 0.80。特征先按录音汇总，再计算 Mann–Whitney U、Cliff's delta、
录音级 bootstrap 区间和 BH-FDR；任一阶段少于 2 条独立录音时只输出描述统计。

模型分数是未经校准的 sigmoid 输出：0.80 是高置信筛选，并不代表真实的 80% 生物学概率。
阶段标签是日期代理，不是人工确认的叫声身份，因此报告只描述关联，不作繁殖阶段导致声学变化的因果结论。
默认边界参考了[英国生物声学研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12336414/)和
[Skokholm 繁殖研究档案](https://ora.ox.ac.uk/objects/uuid%3A24973c21-ab78-42f1-a3c1-a7665b776bbf)；
取得当地 2021 年实地记录后应通过 CLI 覆盖。

## 结果汇总

训练完成后，运行汇总脚本：

```bash
python results_analysis/summarize_nested_cv.py
```

汇总脚本读取 `STORM_PETREL_MODEL_ROOT`，并写出：

```text
$STORM_PETREL_MODEL_ROOT/nested_cv_summary.csv
$STORM_PETREL_MODEL_ROOT/negative_type_error_summary.csv
$STORM_PETREL_MODEL_ROOT/clustered_bootstrap_ci.csv
$STORM_PETREL_MODEL_ROOT/model_metrics.tex
```

`nested_cv_summary.csv` 包含每个 outer fold 的指标和 mean/std：

- threshold=0.5 的 precision
- threshold=0.5 的 recall
- threshold=0.5 的 F1
- 最优阈值下的 precision/recall/F1
- AUPRC
- AUROC

`negative_type_error_summary.csv` 汇总负类子类型中的 false positives。
`clustered_bootstrap_ci.csv` 和 `model_metrics.tex` 通过按来源录音整簇抽样 5,000 次（seed 2021），
复现表 5.3 的 pooled 指标和 95% 置信区间。正式 bootstrap 要求五个 outer fold 的预测全部存在；
若只是有意运行单 fold smoke test，请给汇总命令加 `--skip-clustered-bootstrap`。

## 探索性双模型雏鸟筛选

`chick_screening` 与上面的部署模型成鸟/雏鸟比较是两个不同流程。它分别用成鸟代理期和雏鸟代理期
训练模型，然后对相同录音运行两个模型，并按分数划分四个象限。

完整运行：

```bash
python chick_screening/run_pipeline.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00" \
  --recordings-dir /path/to/chick_proxy_recordings \
  --epochs 50 --threads 8
```

逐阶段运行：

```bash
# 准备两个日期阶段的数据集。
python chick_screening/prepare_date_dataset.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00"

# 训练两个日期专用模型。
python chick_screening/train_date_models.py --epochs 50 --threads 8

# 用两个模型扫描录音。
python chick_screening/dual_model_scan.py /path/to/recordings \
  --output-csv workspace/chick_screening/dual_scores.csv \
  --batch-size 32

# 复用已有分数重新分析，不重复扫描。
python chick_screening/analyze_discrepancies.py \
  --scores-csv workspace/chick_screening/dual_scores.csv \
  --adult-threshold 0.80 \
  --chick-threshold 0.80 \
  --bootstrap-iterations 5000 \
  --seed 2021
```

只需重建一个时期时，可在准备/训练命令中使用 `--adult-only` 或 `--chick-only`。完整输出目录和解释
限制见 `chick_screening/README.md`。

## 命令索引：单独运行每项功能

下表列出全部生产命令入口。当前代码支持的准确参数始终以 `python <脚本> --help` 为准。

| 功能 | 命令 | 主要输出 |
|---|---|---|
| 检查 Python 和依赖 | `python verify_environment.py` | 终端通过/失败报告 |
| 预览/执行服务器同步 | `python sync_project.py USER@HOST:PATH [--apply] [--delete]` | 同步后的源码目录 |
| 选样、截取 WAV、构建 folds 和数据集 | `python data_processing/run_pipeline.py` | `workspace/clips`、清单、folds、数据集 |
| 只选样、截取并写 folds | `python data_processing/prepare_dataset.py` | clips 和 fold CSV |
| 从 fold CSV 重建 BirdNET 目录 | `python data_processing/build_birdnet_fold_dataset.py` | `workspace/nested_cv_datasets` |
| 训练/评估 nested CV | `python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05` | fold 模型、预测和指标 |
| 复用 nested-CV 模型重新评估 | `python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --evaluate-only --force` | 刷新的预测和指标 |
| 训练全 clips 部署模型 | `python model_training/train_deployment_model.py` | `workspace/models/deployment` |
| 扫描长录音并截取候选 | `python long_recording_analysis/run_pipeline.py INPUT_DIR` | 命名的长录音运行目录 |
| 只运行长录音推理 | `python long_recording_analysis/scan_long_recordings.py INPUT_DIR --output-csv FILE` | `all_windows.csv` |
| 应用/更换阈值并导出 WAV | `python long_recording_analysis/extract_candidates.py ALL_WINDOWS_CSV` | detections CSV 和候选 WAV |
| 从原始录音分析日期代理 | `python comp_chick_adult/run_analysis.py --recordings-dir INPUT_DIR` | 扫描缓存、表、图和报告 |
| 从已有分数分析日期代理 | `python comp_chick_adult/run_analysis.py ALL_WINDOWS_CSV [...]` | 表、图和报告 |
| 运行全部雏鸟筛选阶段 | `python chick_screening/run_pipeline.py --recordings-dir INPUT_DIR` | 日期数据集、模型、分数和分析 |
| 准备雏鸟筛选数据集 | `python chick_screening/prepare_date_dataset.py` | 成鸟/雏鸟 clip 数据集 |
| 训练雏鸟筛选模型 | `python chick_screening/train_date_models.py` | 成鸟/雏鸟 TFLite 模型 |
| 运行双模型推理 | `python chick_screening/dual_model_scan.py INPUT_DIR` | `dual_scores.csv` |
| 分析双模型分数 | `python chick_screening/analyze_discrepancies.py` | 象限表、图和报告 |
| 汇总 nested CV 和表 5.3 | `python results_analysis/summarize_nested_cv.py` | 汇总 CSV 和 LaTeX 表 |

## 验证、Smoke Test 与排错

运行完整的轻量自动测试：

```bash
python -m pytest -q
```

不代表最终模型质量的快速 smoke test：

```bash
python model_training/run_nested_cv.py --outer-folds 04 --epochs 5 --threads 4
python results_analysis/summarize_nested_cv.py --skip-clustered-bootstrap
python chick_screening/run_pipeline.py --epochs 5 --threads 4 --skip-scan --skip-analysis
```

运行失败时优先检查：

- 启动时打印的数据库、录音、workspace 和模型路径是否指向预期位置。
- `python verify_environment.py` 是否通过，并显示 Python 3.11 和 BirdNET-Analyzer 2.4.0。
- `STORM_PETREL_MODEL_ROOT` 是否为 `models/nested_cv_balanced`，而不是历史模型/cache 目录。
- 正式 clustered bootstrap 前，五个 `outer_*/predictions.csv` 是否全部存在。
- CV clips 是否为完整的 3 秒、48 kHz 文件；评估会有意拒绝 padding 短片段。
- 长录音模型、文件清单或日期边界变化后，是否新建扫描或显式使用了 `--rescan`。
- Windows 下 Numba 与临时目录是否位于 `workspace`；脚本会自动配置这些路径。

## 生成文件与 Git

不要提交数据库、录音、截取的 clips、缓存、训练模型或 `workspace/` 下的任何文件。只提交源码、
轻量配置、用于检查的 notebook、README 和小型项目文档。`.gitignore` 已覆盖主要生成格式。

## 服务器完整运行示例

服务器上一次完整运行通常是：

```bash
cd /mnt/home/alyzw89/working

export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_MODEL_ROOT="$STORM_PETREL_WORKSPACE_DIR/models/nested_cv_balanced"

python data_processing/run_pipeline.py
python model_training/run_nested_cv.py --outer-folds 01 02 03 04 05 --epochs 50 --threads 8
python results_analysis/summarize_nested_cv.py
```

服务器上快速 smoke test 可以只跑一个 fold 和少量 epochs：

```bash
python model_training/run_nested_cv.py --outer-folds 04 --epochs 5 --threads 4
python results_analysis/summarize_nested_cv.py --skip-clustered-bootstrap
```

## 当前数据状态

当前生成的数据集包含 1000 个正类和 1000 个负类，所有 grouped folds 均严格保持 50:50。新模型统一写入 `workspace/models/nested_cv_balanced`。

如果存在 `workspace/models/nested_cv_stratified`，其中是旧版 500 正类 / 1000 负类数据集的历史结果和缓存，不能用于本次平衡数据训练。
