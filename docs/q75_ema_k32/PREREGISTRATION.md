# 预登记：EMA×k32 四 seed 评估（2026-10-03）

## 背景

- 第三轮确认 `K32`（`tabm_k` 16→32）是唯一在两个目标、两个折上同向改善的配置，且四 seed 时长增量融合过门（+0.0046、LCB95 +0.0022）。
- EMA（训练轨迹权重滑动平均，`beta=0.99`，实现见 `src/bf_tap_r2/component_regularization.py`）是本项目**平台兑现最大单项收益**的机制（Q75 时长 = V32 时长 + 0.75×(EMA − V7)）。
- 两者都是方差缩减机制，可能互补也可能重叠。

## 已完成的初步筛选（**未预登记**，如实记录，不用于选择）

在 `split_seed=42` fold 0/1 上跑了 2 臂 × 2 折 = 4 次拟合（`local/runs/q75-ema-k32-20261002/screen-r2`），仅作机制探查：

| 目标 | EMA16 均值 | EMA32 均值 | Δ（EMA32−EMA16） | 两折同向 |
| --- | ---: | ---: | ---: | :--: |
| 时长 | 0.038020 | 0.037579 | **−0.000440** | **是** |
| 铁量 | 0.036626 | 0.036640 | +0.000014 | 否 |

（另记：首次启动因 settings 键名写错在任何拟合前失败，`screen-r1/ABORTED.json` 保留。）

该 4 次拟合在预登记之前执行，属流程偏差；本文件登记事实，且**不把该筛选结果当作晋级证据**——下表四 seed 阶段独立评估。

## 冻结的四 seed 阶段

- 配置：`ComponentRegressor(recipe={tabm, frequency 0.01}, arm="EMA", mechanisms={ema_beta: 0.99})`，训练设置取 `configs/strong_component_regularization/SPEC.yaml` 的 `training.tap_time_len`，只把 `tabm_k` 设为 32。
- seed：42、3407、271828、314159 × 5 折 = **20 次正式拟合**（每次 2 个 optimizer）；逐折预测持久化。
- 参考列：`local/runs/q75-combination-review-20261002/review-r1/combination-s{seed}.npz` 的 `q75`（时长）与 `iron`（铁量）；K32 四 seed 列已存在于 `local/runs/q75-k32-confirmation-20261002/development-r1`。
- 评价（全部只用时长列；铁量在本阶段仅作描述）：
  1. 单列 WMAPE 与残差相关（对 Q75 时长列、对 K32 列）；
  2. `Q75 + w×(EMA32 − Q75)`：权重由其余三个 seed 选出，留出 seed 评分；
  3. 三成分 `Q75 + w1×(K32 − Q75) + w2×(EMA32 − Q75)`：两个权重由其余三个 seed 的网格搜索选出，留出 seed 评分。
- 判读：某形式四 seed 全正且 seed 层 LCB95 > 0 才登记为可进入另行冻结的全量/包阶段。
- 预算：**20 次正式拟合**；无全量、无包、无桌面、无上传。
- 资源：单数值线程、worker ≤ 6、峰值内存 1536 MiB/worker；不设时间预算。
