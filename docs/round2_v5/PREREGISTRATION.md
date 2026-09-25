# Round2 V5 预登记：误差协方差选成员、组合形状、评估分辨率与平台噪声底

日期：2026-09-25  
分支：`round2-v5-error-covariance-resolution`（base `4a05321`）  
状态：**PRE_REGISTERED — 本文件在任何 V5 评测执行之前写入**  
机器可读契约：`configs/round2_v5/SPEC.yaml`，由 `src/bf_tap_r2/v5_spec.py` 硬校验

## 0. 范围与目标口径

用户本轮目标原文为**平台 93.5**，按仓库既有处置**保留原文、不擅自改写**（`configs/round2_v5/SPEC.yaml: targets.user_requested_platform_score_verbatim`）。
该数值当前已被远超：用户回传最佳为 `V36_USER_REQUESTED_OUTER_FAILED = 96.2734`（用户报告，未独立账号核验）。
因此本轮的**operational 目标**沿用仓库现行下一阶段目标：**平台 > 96.3**；本地工作门槛 `96.25`。

本轮不做模型族扩张，只做三件事：

1. 用**零拟合**方式把“误差协方差选成员”真正做一次（此前 V3.4 的 “diversity” 是结构去重，不是误差相关筛选）；
2. 把本地判据从噪声里挪出来（判据单位从“两 seed 均值 ≥0.005”改为 **split seed 级配对下置信界**）；
3. 用一次平台提交测**平台噪声底**（同配方只换训练 seed）。

## 1. 阈值依据：执行前实测（只读）

以下数字在预登记前由只读脚本测得，是 SPEC 中阈值的依据，不是候选选择结果。

### 1.1 现有 OOF 池清点（修正“482 文件 / 444 trial”前提）

| 来源 | 文件 | 覆盖 | 训练约定 |
|---|---:|---:|---|
| `coarse-*-r1` | 400 | seed 42，**仅 folds 0/1（1102/2754 行）** | 开 early stopping |
| `refine-catboost-r1` | 48 | 24 trial × seed 42/3407 × 5 折 | 关 ES |
| `refine-all-r1` | 16 | 8 trial × 2 seed × 5 折 | 关 ES |
| `refine-xgboost-r1` | 12 | 6 trial × 2 seed × 5 折 | 关 ES |
| `direct-pair-confirm-r1` | 2 | seed 2026 包级向量 | — |
| `smoke-r1` | 4 | 与 coarse 同 id 重复 | — |

去重后 **400 个 trial id**；真正 2 seed × 5 折全覆的只有 **38 个 trial**（CatBoost 12+12、XGBoost 3+3、其余各 1+1）。

同一 `trial_id` 在不同目录是**不同拟合**：用 coarse ledger 中 `v3-catboost-tap_iron-0034` 规格重跑，开 ES 得 fold0 WMAPE `0.038854`（≈ledger `0.039112`），关 ES 得 `0.038706`，与 `refine-catboost-r1/pred-42-...` **精确一致**。故库键必须为 `(目录, seed, trial_id)`。

### 1.2 零拟合可用库

每目标在 2 seed × 5 折上全覆的候选列：**铁量 95 / 时长 91**（V36 完整开发缓存 20 专家 + `refine-*` 38 trial + `round2-v2*` OOF 列）。所有 v2 OOF 的 `fold` 列与 `folds-{seed}.csv` **逐行一致**（已验证）。

### 1.3 相关性与组合形状

| 量 | 铁量 | 时长 |
|---|---|---|
| V36 列权重 | 0.604 A + 0.360 D-0029 + 0.036 N-0005 | 0.781 A + 0.090 O-0057 + 0.129 D-0048 |
| 留一成员 Δscore（重归一） | drop N-0005 −0.0003；drop D-0029 −0.0066；drop A −0.0172 | drop O-0057 −0.0006；**drop D-0048 −0.0011**（seed 3407 为 +0.0002）；drop A −0.0426 |
| 与基座残差 ρ 范围 | 0.73–0.99（最低 JM1/K1/KW2/KWT/KR1） | 0.77–0.96（最低 JM1 0.770） |
| 现有池最优“加 1 成员”嵌套增益 | `+0.000070` WMAPE ≈ **+0.0035pt** | `+0.000070` WMAPE ≈ **+0.0035pt** |
| coarse 池最低相关候选（ρ≈0.27–0.33） | 单独 0.066–0.076 vs 基座 0.037，嵌套 2 折增益 ≈ −0.0006 | 同左 |

结论：**“低相关”单独会捞到噪声**，必须联合“精度损失有界”。SPEC 取 `ρ ≤ 0.95` 且 `单独 WMAPE ≤ 1.25 × 基座`：恰好纳入 N-0005(1.11)/JM1(1.13)/KW2(1.16)，排除 2× 的垃圾。

### 1.4 分辨率

N2 完整覆盖增益 `+0.014374`、8/10 折为正、逐折跨度 −0.009~+0.028，**折级配对 LCB95 ≈ +0.008 > 0**——即折级下置信界**无法拒绝**这个已知坏候选。故 SPEC 把晋级单位定为 **split seed**：≥4 个独立 seed 的 seed 级配对 LCB95 > 0（`resolution.seed_level`）。

## 2. 阶段与 stop rule

### Stage 0 — 预登记（0 拟合）
写 SPEC + 本文件 + `v5_spec.py`，落盘提交。

### Stage 1 — 零新拟合诊断（0 拟合）
统一库 → 协方差/留一/替换诊断 → 嵌套选成员（α 在 fit seed 选、held seed 评）→ **回溯校准**：新规则必须拒绝 ≥3/4 已知坏候选（V42 N2、N4、N5、R4），并保留已知好对照。
- 若无候选同时满足准入规则且 held-seed 配对为正 → 判定“现有 2-seed 池已榨干”，**不封包**，转 Stage 2。
- 若回溯校准失败 → **停止，不消耗 Stage 2 拟合**。

### Stage 2 — 新拟合（预算受 SPEC 约束）
- **2a 时长列 N 家族补全**：对 48 个 N 时长 coarse trial 按准入规则取 top-16，补到 seed 42/3407 × 5 折（上限 190 fit slot），top-3 再补 seed 2026。
- **2b 独立 split seed 复现**：派生 seed 7777、12011（`splits.make_folds`，确定性 group-safe；派生 seed 已在此预声明），对最终 1–2 个候选做 seed 级配对 LCB。
- 晋级：≥4 seed 的 seed 级 LCB95 > 0 且两 seed 初值同向，且本地完整包 ≥ 96.25。

### Stage 3 — 交付与平台实验（用户上传）
- 单列隔离包（未改列**逐字节**透传父包 CSV 字段文本；父包复现核对 + 冷推理一致性检查）。
- **平台噪声实验包**：V34_A 生产配方 + V36 专家在全量训练集上用预登记 training-seed 偏移重拟合；平台差 = seed 敏感度 + 平台分辨率。
- 判据：平台差 ≈0.02 → 停止千分位追逐；≈0.005 → 继续误差协方差方向并重估 N2 倒挂归因。
- 助手不执行上传、不推算剩余额度：用户自行上传并回传。

## 3. 明确不再投入

`closed_routes.do_not_reopen`：NODE 逐深度、ODST 核心、TabR 完整融合、残差修正器、按误差删样本/降权/手改预测、0.25 以外密集 α 网格。
`closed_routes.unavailable_not_untried`：时序/滞后特征（`复赛_train/train_features.csv` 只有 `sample_id` + 21 个瞬时特征）。

## 4. 声明

- 不修改冻结契约：CatBoost 参数/迭代/目标/损失/特征窗口/后处理/验收阈值均不动；`baseline-v0.1` 标签与历史包不改写。
- 私有产物只写 `local/runs/round2-v5-*`；模型、预测、报告、提交包不入 Git。
- 本轮所有“候选”在跑出 Stage 2 之前都只是 `candidate_pool`，不构成平台收益承诺。
