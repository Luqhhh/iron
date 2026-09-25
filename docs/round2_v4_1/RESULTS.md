# Round2 V4.1：强基线增量诊断与受控结构修正执行结果

日期：2026-09-25  
分支：`round2-v4.1-strong-increment`  
状态：**EXECUTED_NO_PROMOTION**  
目标：平台严格 **> 96.3**

## 0. 执行摘要

本轮已按 `V4_1_TASKBOOK.md` 执行 D0、R0、C 线、E 线，并补齐对应源码、CLI 与测试。
真实旧 OOF 诊断已运行；C 线 10 个 method/target 单元、E 线 8 个 method/target 单元均产生了本地证据。
**没有任何方向或修正器满足“两个 seed 同向为正、平均完整包增量 ≥0.02”的继续条件。**
本轮不进入完整开发五折，不生成平台 ZIP，不上传，不消费新的 outer seed。

当前平台最佳仍为用户回传 `V36_USER_REQUESTED_OUTER_FAILED = 96.2734`，距严格大于 96.3 还差
**0.0266**。完整 `B_star = 01_V34_NE3_CAP50` 仍未在本 checkout 恢复；本轮所有增量均为
A-frozen 相对证据，不声称 B_star 级或平台级结论。

## 1. D0：真实 V4 OOF 零模型训练诊断

运行：

```bash
.venv/bin/python scripts/round2_v4_1/diagnose_v4_oof.py \
  --root . \
  --source local/runs/round2-v4-mechanism-search/coarse-r2 \
  --output local/runs/round2-v4.1-strong-increment/diagnostic-r1
```

结果：

| 项 | 值 |
|---|---:|
| 可评分 seed-unit 事件 | 76 |
| blocked seed-unit 事件 | 4 |
| 新基础模型训练 | 0 |
| 同标签标量优化 | 76 |
| 两 seed oracle 增益均为正的旧方向 | 0 |
| `worth_nested_retest_NOT_PROMOTION` | 0 |

按 `oracle_mean_delta_SAME_LABELS` 排序的最高旧方向如下；全部显式为 `SAME_LABELS / NOT_VALIDATION`：

| method_id | target | single_mean_delta | half_blend_mean_delta | oracle_mean_delta_same_labels | two seeds positive |
|---|---|---:|---:|---:|---:|
| v4-F-006 | tap_iron | -0.272420 | -0.077364 | 0.000291 | 0 |
| v4-S-019 | tap_time_len | -1.695191 | -0.563578 | 0.000271 | 0 |
| v4-J-034 | tap_iron | -0.353605 | -0.107154 | 0.000035 | 0 |
| v4-J-033 | tap_iron | -0.622550 | -0.179270 | 0.000002 | 0 |

D0 决策：无方向入围嵌套复测；不扩参数，不把 0.0003 级同标签包络作为候选。

产物：

- `local/runs/round2-v4.1-strong-increment/diagnostic-r1/direction_diagnostics.csv`
- `local/runs/round2-v4.1-strong-increment/diagnostic-r1/diagnostic_shortlist_NOT_PROMOTION.csv`
- `local/runs/round2-v4.1-strong-increment/diagnostic-r1/manifest.json`

## 2. R0：可调用的强基线工厂

新增 `src/bf_tap_r2/v4_1_reference.py`。工厂 `FrozenAReferenceFactory` 读取 V3.4 私有恢复脚本和
V3.1/V3.4 冻结权重，只在给定训练帧上重训 21 个 L1 base member 和 A 所需 V3.4 专家，再按
A 的冻结 deployment 权重组合。协议记为 `A_frozen_deployment_weights_v1`。

在 `seed=42, outer_fold=0` 上与现有 A OOF 参照对比：

| target | A_frozen factory WMAPE | A_dev OOF WMAPE | 差异 |
|---|---:|---:|---:|
| tap_iron | 0.03645232 | 0.03644502 | +0.00000730 |
| tap_time_len | 0.03817201 | 0.03814127 | +0.00003074 |

差异来源是冻结 deployment 权重与历史逐折选择的重放差异。该工厂是**固定冻结权重基线**，
不宣称已经实现“历史 A 在每个训练子集内重新选权”的完整嵌套过程；因为它不在 T/V 上做权重选择，
所以不会用 V 标签选择超参数。B_star 仍未恢复，P4/B_star 级声明继续 blocked。

## 3. C 线：spout、强 EBM 父配方与显式 pair/triple 控制

新增 `src/bf_tap_r2/v4_1_paired_terms.py`，CLI 为 `src/bf_tap_r2/v4_1_run.py`。
C 线共 10 个 method/target 单元：C0 复用 V4 S5 旧结果，C1–C4 为新拟合，共 32 次外层 fold fit。

两 seed、folds0/1 的平均 A-relative 完整包增量：

| method | 内容 | tap_iron 平均 delta | tap_time_len 平均 delta |
|---|---|---:|---:|
| C0 | V4 S5 21 数值列（复用） | -0.110199 | -0.744502 |
| C1 | 同 S5 + 正确 nominal spout_no | -0.109698 | -0.278077 |
| C2 | 恢复 V3.4 强 EBM 父配方（类别、log1p、group-safe bags） | **-0.018102** | **-0.105639** |
| C3 | C2 pair 集合显式重放 | -0.019110 | -0.121697 |
| C4 | C3 + 最多 2 个训练内部三元项 | -0.024639 | -0.132219 |

观察：

1. **spout_no 是真实覆盖缺口，但不是当前增量答案。** C1 相对 C0 在时长上改善约 +0.4664，
   在铁量上仅 +0.0005，仍全部为负。
2. **低容量代理与强父配方差距很大。** C2 明显强于 C0/C1，铁量平均接近 A（-0.0181），
   时长仍 -0.1056；但没有一个 seed/target 达到两 seed 同向为正。
3. **显式 pair 控制保住了 pair 集合。** C3 实际 `term_features_` 与 C2 的 pair 集合逐 term 一致；
   但显式化本身没有带来正增量。
4. **C4 三元项没有增量。** 训练内部三元候选只由训练部分的 pair/single term contribution 选出；
   加入后两目标平均都变差，且铁量 seed42、时长 seed42/3407 均为负。
5. 时长目标的 C2 父配方使用 `max_interaction_bins=64`；直接显式三元在该分箱下出现不可接受的多维
   分区耗时，因此按任务书“两个控制一起采用较粗交互分箱并报告对 C2 损失”的例外，时长 C3/C4
   统一改用 `max_interaction_bins=16`，并保留 C2 full-bin 结果作为损失参照。

C 线产物：

- `local/runs/round2-v4.1-strong-increment/paired-terms-r1/fit_ledger.jsonl`
- `local/runs/round2-v4.1-strong-increment/paired-terms-r1/paired_terms_summary.csv`
- `local/runs/round2-v4.1-strong-increment/paired-terms-r1/manifest.json`

## 4. E 线：嵌套强基线残差修正

新增：

- `src/bf_tap_r2/v4_1_residuals.py`：E0 常数中位数、E1 LAD 低自由度样条、E2 LightGBM linear leaf、
  E3 prediction-state 局部加权中位数。
- `src/bf_tap_r2/v4_1_nested.py`：严格按任务书伪代码执行 U/W meta-fold、U 内部 base OOF、
  meta alpha 选择、T 内部 base OOF 和最终 B_T + alpha*h_T。

协议常量：2 个 group-safe meta folds，每个 U 内 2 个 group-safe subfolds。每个
seed/outer-fold base cache 需要 9 次 A_frozen 基础拟合，4 个 cache 共 **36 次基础拟合**。
E0 无正确器模型拟合；E1/E2/E3 每个 seed/outer-fold/target 做 3 次正确器拟合。
alpha 仅在训练部分 meta folds 上求取，范围 [0,1]，含零回退；外层 V 只在全部选择完成后评分。

两 seed、folds0/1 的平均 nested 单目标完整包增量：

| method | 结构 | tap_iron 平均 delta | tap_time_len 平均 delta | 两 seed 双正 |
|---|---|---:|---:|---:|
| E0 | 全局残差中位数 | +0.001645 | +0.000005 | 否 |
| E1 | LAD 低自由度样条 | **+0.001693** | **+0.000824** | 否 |
| E2 | LightGBM linear leaf | +0.000417 | -0.001956 | 否 |
| E3 | prediction-state 局部加权中位数 | -0.002568 | -0.003007 | 否 |

单事件最高值为 E0/E1 铁量 seed42 fold0 `+0.005916`，E2 时长 seed3407 fold1 `+0.005342`；
但相同方法在另一 seed/fold 常回到 0 或为负。没有任何 E 单元达到“两个完整开发切分都为正”，
更未达到平均 `+0.02`。因此不追加后续配方，也不进入最终 outer 复验。

E 线产物：

- `local/runs/round2-v4.1-strong-increment/nested-residual-r1/fit_ledger.jsonl`
- `local/runs/round2-v4.1-strong-increment/nested-residual-r1/nested_residual_summary.csv`
- `local/runs/round2-v4.1-strong-increment/nested-residual-r1/nested-base-cache/*.npz`
- `local/runs/round2-v4.1-strong-increment/nested-residual-r1/manifest.json`

## 5. 最终决策

| 项 | 结果 |
|---|---|
| D0 旧 OOF 诊断 | 已执行；0 方向入围 |
| A-frozen 工厂验证 | 已执行；A-relative，非 B_star 级 |
| C 线新单元 | 8/8 已执行；无两 seed 正增量 |
| E 线新单元 | 8/8 已执行；无两 seed 正增量 |
| 完整开发五折 | 未进入 |
| 最终 outer 新切分 | 未提出；23003 已消费，不可复用 |
| 平台封包 / 上传 | 0 / 0 |
| 新增 B_star 级声明 | 无 |

结论：按实际测试范围关闭本批 C/E 实现。该负结果只约束本轮实现和协议，不升级为模型家族或
理论极限结论。后续若继续，必须先恢复/实现带训练子集内部选权的完整 A/B_star，再重新预登记预算。

## 6. 测试与交付文件

- D0 合成测试：`tests/test_increment_diagnostic.py`，30 passed。
- V4.1 契约测试：`tests/test_round2_v4_1.py`，10 passed。
- 与 V4 相关既有测试联合运行：50 passed。
- D0 脚本：`scripts/round2_v4_1/increment_diagnostic.py`、`scripts/round2_v4_1/diagnose_v4_oof.py`。
- 任务书：`docs/round2_v4_1/V4_1_TASKBOOK.md`。
- 实现：`src/bf_tap_r2/v4_1_reference.py`、`v4_1_paired_terms.py`、`v4_1_residuals.py`、`v4_1_nested.py`、`v4_1_run.py`。
- 私有逐样本证据全部留在 `local/runs/round2-v4.1-strong-increment/`，未提交。

---

## 7. 2026-09-25 strong-base C2/C3/C4 later execution

The self-contained V4.1 strong-base C2/C3/C4 path was later implemented and
executed as a descriptive frozen-V3.6 development screen.  It did not promote a
candidate.  See `docs/round2_v4_1/STRONG_INCREMENT_RESULTS.md` for the first
round and complete iron follow-up tables, blocked time-C4 units, real fit
counts, and the zero-submission decision.
