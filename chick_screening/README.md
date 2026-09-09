# Chick Screening — 双模型雏鸟叫声筛选

用成鸟期和雏鸟期分别训练的 BirdNET 模型，对雏鸟期长录音进行双模型推理，
通过分数差异筛选候选雏鸟特有叫声，并结合声学特征分析。

## 原理

```
成鸟期 DB 候选 → MILP 选取 → adult_clips → 训练成鸟模型
雏鸟期 DB 候选 → MILP 选取 → chick_clips → 训练雏鸟模型
                                                ↓
长录音 → 双模型并行推理 → 分数合并 → 四象限分类 → 声学特征对比 → 报告
```

- **Chick 候选**：雏鸟模型高分 + 成鸟模型低分 → 候选雏鸟特有叫声
- **通用海燕**：两个模型都高分 → 跨生命阶段的通用特征
- **成鸟偏向**：成鸟模型高分 + 雏鸟模型低分
- **背景**：两个模型都低分

## 目录结构

```
chick_screening/
├── run_pipeline.py              # 总入口
├── prepare_date_dataset.py      # 步骤 1: 按日期从 DB 选取数据
├── train_date_models.py         # 步骤 2: 训练两个 BirdNET 模型
├── dual_model_scan.py           # 步骤 3: 双模型并行推理
├── analyze_discrepancies.py     # 步骤 4: 分歧分析 + 可视化
└── README.md
```

输出目录：`workspace/chick_screening/`

## 本地运行

### 前置条件

```bash
conda activate strompetrel
python verify_environment.py
```

需要数据库文件（`STORM_PETREL_DB_PATH`）和音频文件（`STORM_PETREL_AUDIO_ROOT`）。
如果 `project_config.py` 和 `workspace/` 都在项目根目录下，不需要额外设置环境变量。

### 完整流程

```bash
python chick_screening/run_pipeline.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00" \
  --recordings-dir /path/to/chick_proxy_recordings \
  --epochs 50 --threads 8
```

### 分步运行

```bash
# 步骤 1: 只准备数据
python chick_screening/prepare_date_dataset.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00"

# 步骤 2: 只训练模型
python chick_screening/train_date_models.py --epochs 50 --threads 8

# 步骤 3: 只做推理
python chick_screening/dual_model_scan.py /path/to/recordings

# 步骤 4: 只做分析（不需要重新扫描）
python chick_screening/analyze_discrepancies.py \
  --scores-csv workspace/chick_screening/dual_scores.csv \
  --adult-threshold 0.80 --chick-threshold 0.80
```

### Smoke test（快速验证）

```bash
python chick_screening/run_pipeline.py \
  --epochs 5 --threads 4 \
  --skip-scan --skip-analysis
```

## 服务器运行

项目通过 `sync_project.py` 同步到服务器。服务器目录结构与本地一致。

### 同步代码

```bash
# 预览
python sync_project.py alyzw89@strange:~/working

# 执行同步
python sync_project.py alyzw89@strange:~/working --apply
```

### 服务器环境设置

```bash
ssh alyzw89@strange
cd ~/working
conda activate strompetrel-birdnet

# 设置环境变量（如需要）
export STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
export STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
export STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
export STORM_PETREL_WORKSPACE_DIR="$STORM_PETREL_PROJECT_ROOT/workspace"
export STORM_PETREL_CHICK_SCREENING_DIR="$STORM_PETREL_WORKSPACE_DIR/chick_screening"
```

或者创建 `~/working/.env` 文件（避免每次 export）：

```dotenv
STORM_PETREL_PROJECT_ROOT=/mnt/home/alyzw89/working
STORM_PETREL_DB_PATH=/mnt/home/alyzw89/data/2021/storm_petrel_2021_full.db
STORM_PETREL_AUDIO_ROOT=/mnt/home/alyzw89/data/2021/recordings
STORM_PETREL_WORKSPACE_DIR=${STORM_PETREL_PROJECT_ROOT}/workspace
STORM_PETREL_CHICK_SCREENING_DIR=${STORM_PETREL_WORKSPACE_DIR}/chick_screening
```

### 服务器完整运行

```bash
cd /mnt/home/alyzw89/working

python chick_screening/run_pipeline.py \
  --adult-end "2021-05-31T23:59:59+00:00" \
  --chick-start "2021-07-23T00:00:00+00:00" \
  --recordings-dir /mnt/home/alyzw89/data/2021/recordings \
  --epochs 50 --threads 8
```

### 服务器 smoke test

```bash
python chick_screening/run_pipeline.py \
  --epochs 5 --threads 4 \
  --skip-scan --skip-analysis
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--adult-end` | `2021-05-31T23:59:59+00:00` | 成鸟期截止时间（ISO 格式） |
| `--chick-start` | `2021-07-23T00:00:00+00:00` | 雏鸟期起始时间（ISO 格式） |
| `--target-positive` | `None`（自动） | 每时期目标正例数，按比例缩放负例 |
| `--epochs` | 50 | BirdNET 训练轮数 |
| `--threads` | 8 | BirdNET 训练线程数 |
| `--adult-threshold` | 0.80 | 成鸟模型高置信阈值 |
| `--chick-threshold` | 0.80 | 雏鸟模型高置信阈值 |
| `--force` | false | 覆盖已有输出 |
| `--skip-*` | false | 跳过指定步骤 |

## 输出

```
workspace/chick_screening/
├── adult_clips/                  # 成鸟期训练数据
│   ├── storm_petrel/*.wav
│   ├── not_storm_petrel/*.wav
│   └── labels.csv
├── chick_clips/                  # 雏鸟期训练数据
│   ├── storm_petrel/*.wav
│   ├── not_storm_petrel/*.wav
│   └── labels.csv
├── models/
│   ├── adult_model/              # 成鸟期训练的模型
│   │   ├── model.tflite
│   │   ├── model_Labels.txt
│   │   └── training_config.json
│   └── chick_model/              # 雏鸟期训练的模型
│       ├── model.tflite
│       ├── model_Labels.txt
│       └── training_config.json
├── dual_scores.csv               # 双模型推理合并分数
└── analysis/                     # 分歧分析结果
    ├── quadrant_summary.csv
    ├── recording_summary.csv
    ├── feature_comparison.csv
    ├── window_features.csv
    ├── threshold_yield.csv
    ├── figures/
    │   ├── probability_scatter.png
    │   ├── delta_distribution.png
    │   ├── average_log_mel_spectrograms.png
    │   ├── difference_log_mel_spectrogram.png
    │   └── feature_boxplots.png
    ├── representative_audio/
    └── analysis_report.md
```

## 重要注意事项

1. **雏鸟期数据量取决于录音覆盖**：可以通过调整 `--chick-start` 来扩大雏鸟期定义，以增加训练数据
2. **模型分数不可直接比较**：两个模型在不同数据上训练，sigmoid 输出未经校准，使用 per-model 阈值和 delta 分析
3. **雏鸟录音中可能包含成鸟叫声**：chick 候选不等于确认的雏鸟叫声，结果应视为探索性分析
4. **日期是代理标签**：模型对生命阶段的区分能力受限于训练数据的日期代理性质
