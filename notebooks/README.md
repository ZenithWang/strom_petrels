# 论文分析 Notebooks

`notebooks/01`–`05` 是五个彼此独立、可从干净 kernel 完整执行的论文分析模块。解释文字使用中文，图表标签使用英文；所有生成结果统一写入 `workspace/paper_analysis/<module>/`，不应提交到 Git。

## 推荐顺序与统计层级

| Notebook | 主要输入 | 主要统计层级 | 输出目录 |
|---|---|---|---|
| `01_dataset_overview.ipynb` | `binary_labels.csv`, `nested_cv_folds.csv`, `fold_balance_report.csv` | 三秒片段描述；`source_file` 分组验证 | `01_dataset_overview/` |
| `02_model_results_analysis.ipynb` | `models/nested_cv_balanced/outer_*/predictions.csv` | 2,000 个 OOF 片段；置信区间按来源录音 clustered bootstrap | `02_model_results_analysis/` |
| `03_error_review_false_positives.ipynb` | 平衡 nested-CV OOF predictions、片段音频与标签元数据 | 错误片段描述；来源与 fold 分层；人工试听 | `03_error_review/` |
| `04_adult_chick_proxy_analysis.ipynb` | `comp_chick_adult/adult_proxy_vs_chick_proxy/` 及日期边界敏感性产物 | 来源录音；声学特征先取录音内中位数 | `04_adult_chick_proxy/` |
| `05_chick_screening_analysis.ipynb` | adult/chick 训练标签、`dual_scores.csv`、`window_features.csv` | 来源录音内配对；窗口级结果仅描述 | `05_chick_screening/` |

建议按 01→05 执行，便于先核对数据拆分，再阅读模型、错误和两个专项分析。每个 notebook 顶部集中设置阈值、bootstrap 次数、随机种子及输出目录，并在分析前验证必需字段。

## 统一分析约定

- 正式 nested-CV 只读取 `workspace/models/nested_cv_balanced`；历史 `nested_cv_stratified` 不进入主分析。
- BirdNET/TFLite 原始输出按 logits 处理并经 sigmoid 转为 0–1 分数。固定分类阈值为 0.50，不使用 outer test 上搜索的最优阈值。
- Adult/chick 日期比较的主分析为 `adult_proxy_vs_chick_proxy`；`early_vs_possible_chick_0716_p080` 仅作日期边界敏感性分析。
- Chick screening 主阈值为 adult/chick = 0.80/0.80。与任一训练集来源重叠的 9 条扫描录音从论文主结果排除，全量结果只作敏感性对照。
- 模型 sigmoid 分数未做概率校准，不能解释为真实生物学概率；日期分组是非因果代理变量。
- 多声学特征推断使用 BH-FDR；bootstrap 固定 5,000 次、随机种子 2021。

## 输出与执行

每个模块至少导出 300 dpi PNG、矢量 PDF、汇总 CSV 和 LaTeX 表格。批量执行示例：

```bash
for notebook in notebooks/0{1..5}_*.ipynb; do
  conda run -n strompetrel python -m nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 "$notebook"
done
```

`03_error_review_false_positives.ipynb` 默认关闭批量音频播放器；人工复核时将 `ENABLE_AUDIO_PLAYBACK` 改为 `True`，再调用 `listen_to_error`。

训练、数据准备与正式结果汇总仍以脚本为准：`results_analysis/summarize_nested_cv.py` 是表 5.3
录音级 clustered bootstrap 的权威可重跑实现；`02_model_results_analysis.ipynb` 镜像该计算用于检查。
`model_training/nested_cv_training.ipynb` 和 `results_analysis/nested_cv_analysis.ipynb` 同样只作检查材料，不替代生产脚本。
