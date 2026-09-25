# Round2 V4.3：真实强基座上的完全交叉拟合时长残差校准

日期：2026-09-25
分支：`round2-v4.3-strong-base-time-residual`
基提交：`6106299d4c098e668d5b7126c0203d978dcf9333`
状态：**预登记完成，smoke 执行中；无封包、无上传**

## 0. 一句话目标

把 V4.1 正交残差校准从「公开可重建的 V34 EBM 锚点」搬到**真实冻结父包**
（当前用户回传平台最佳的 V36 组合），并在该父包上按**完全交叉拟合**重跑同一
候选池，逐行对齐父包，用预登记门槛判定是否值得动用平台名额。

## 1. 为什么要重做

V4.1（远端 `round2-v4.1-orthogonal-residual-search`）得到 `HGB_L2_L15_A050`
时长残差修正：单目标贡献 **+0.02570552**、9/10 折改善、两 seed 同向为正，
但它明确声明：

- 锚点是**公开重建**的 V34 EBM 配方，不是 V34_A/V36 的真实重放；
- 校准后锚点时长 WMAPE 0.0396395108，**仍差于**真实父包的时长侧；
- 该机器缺少 V34_A/V36 私有 OOF、模型与 test 预测缓存，
  因此不能声称 +0.0257 可叠加到 96.2734。

本 checkout **具备**这些私有缓存（V3.4 l1-oof、refine-r1、V3.6 complete-dev-combined、
V34_A 与 V36 包），所以可以在真实父包上重做，并直接给出对父包的增量。

## 2. 父包身份核验（可复现）

父包 = `V36_USER_REQUESTED_OUTER_FAILED` 的开发侧组合，定义在
`local/runs/round2-v3.6-loss-training-and-numeric-encoding/v36-summary.json`
的 `complete_development_composition`。

用私有缓存逐行重建后与记录值**逐位一致**（测试
`tests/test_round2_v4_3.py::test_recorded_parent_identity_when_private_caches_exist`
在缓存存在时强制校验）：

| 项 | 重建值 | 记录值 |
|---|---:|---:|
| 铁量 WMAPE seed 42 | 0.03749915940318959 | — |
| 铁量 WMAPE seed 3407 | 0.037585782769457295 | — |
| 时长 WMAPE seed 42 | 0.03847465422923957 | — |
| 时长 WMAPE seed 3407 | 0.03828990083841439 | — |
| 每 seed 包分 42 | 96.20130931837855 | 96.20130931837855 |
| 每 seed 包分 3407 | 96.20621581960641 | 96.20621581960641 |
| 平均包分 | **96.20376256899247** | **96.20376256899247** |

包分公式：`100 - 50 * (WMAPE_iron + WMAPE_time)`。
开发侧不做 0 截断（父包开发预测最小值为 64.28，无负值）；导出侧仍按包规则截断。

## 3. 组合定义（冻结）

```
parent_time = w0 * A_dev + w1 * v36-s1-O-0057 + w2 * v36-s1-D-0048
w           = [0.7810993469207348, 0.09014959669794487, 0.1287510563813204]

A_dev_time  = c0 * L1 + c1 * v34-s1-ebm_boundary-0089 + c2 * v34-s1-global_spout_shrink-0136
c           = [0.6192377408552066, 0.28691289439355067, 0.09384936475124295]

L1          = LP-simplex 加权（lp_simplex_weights，目标 WMAPE）五个成员：
              v31-s1-time-0021-0050 / -0031 / -0039 / v3-catboost-tap_time_len-0021 / -0002
```

铁量侧不在本轮范围内，保持父包原样；因此时长侧单目标增量直接等于整包增量
（×50 换算后）。

## 4. 完全交叉拟合协议

对每个外层折 k（seed ∈ {42, 3407}，fold ∈ {0..4}）：

1. `T = {folds != k}`；`inner_seed = 9700 + seed*10 + k`；
2. 用 `group_safe_inner_folds(T, n_splits=3, seed=inner_seed)` 做重复组隔离的 3 折；
3. 对每个 inner 折 m，把**全部 9 个成员**（5 个 L1 池成员 + 2 个 A 专家 + 2 个父包专家）
   在 `T \ inner_m` 上重训、在 `inner_m` 上预测 → 得到每个成员在 T 上的
   cross-fitted 预测；
4. 用 T 上的 inner-OOF 矩阵解一次 LP-simplex 得到 L1 权重（只用 T 标签），
   再按第 3 节权重合成 `parent_oof(T)`；
5. 残差目标 `r = y_time - parent_oof(T)`；候选修正器在 `(features, spout_no, parent_oof)` 上拟合；
6. 外层验证行的基座预测取**已记录的父包开发 OOF**：折 k 的记录预测正是
   「在除 k 以外全部折上拟合的成员」的合成，也就是 T 上的父包重训结果，
   因此验证基座与父包逐行对齐，无需额外拟合；
7. 候选预测 `= clip(parent_val + correction, 0)`，在外层验证行上按 WMAPE 评分，
   `package_delta_single = 50 * (parent_wmape - candidate_wmape)`。

**已知并声明的近似**：L1 的 simplex 权重在 T 的 inner-OOF 上求解，因此某个训练行的
权重向量见过该行标签。这与 V3.4 记录父包时使用的构造完全一致（栈式 OOF 的标准形式），
此处明确登记而非隐藏；该行自身的基座预测仍是 out-of-fold。

## 5. 候选池（与 V4.1 完全一致，冻结）

每目标 12 个：`shrink_bins` 4 个、`lightgbm_residual` 4 个、`histgb_residual` 4 个，
超参与 V4.1 相同（见 `configs/round2_v4_3/experiment.yaml`）。
本轮只跑时长目标（`tap_time_len`），共 12 个候选。

复用同一候选池的目的是**隔离“基座替换”这一个变量**：V4.3 与 V4.1 的唯一差异是
锚点从公开重建换成真实父包。

## 6. 门槛与决策规则（先登记后执行）

- 粗筛：seeds {42, 3407} × folds {0, 1}；
- 继续门槛：**两 seed 均严格为正** 且 **平均整包增量 ≥ 0.005**；
- 完整开发：仅对通过继续门槛的候选跑 folds {0..4}，要求至少 2 个完整开发切分为正；
- 本地工作门槛：`96.20376256899247 + 平均增量 ≥ 96.25`，即需要 **≥ +0.04623743100753**；
- 平台目标：严格 `> 96.3`（当前用户回传最佳 96.2734，差 0.0266）；
- outer seed 23003 已消费，无替代 outer，本轮**止步于完整开发覆盖**；
- 不生成平台包、不上传；失败候选记录为负证据，不做回改。

## 7. 计算预算（实测）

成员单次拟合（2203 行，含 V4.2 运行竞争 CPU）：

| 成员 | 秒 |
|---|---:|
| v34-s1-ebm_boundary-0089 | 18.07 |
| v34-s1-global_spout_shrink-0136 | 19.64 |
| v36-s1-O-0057 | 16.49 |
| v36-s1-D-0048 | 16.32 |
| v3-catboost-tap_time_len-0021 | 20.22 |
| v31-s1-time-0021-0050 / -0031 / -0039 / -0002 | 24.98 / 11.07 / 14.05 / 12.72 |

单遍 9 成员合计 ≈ **175 s**；每个 `(seed, fold)` 需 3 个 inner 折 = 27 fits ≈ **9 min**，
外加 12 个修正器拟合。粗筛 4 个任务、完整 10 个任务，按 worker 并行。

## 8. 声明边界

- 所有增量都是**相对真实 V36 开发组合**的本地开发重放证据，不是平台分、不是平台预测；
- 不声称与 V34_A/V36 的平台成绩可加；平台名额决策仍以本地工作门槛为准；
- 铁量侧未复核：若时长残差通过门槛，整包结论仍受“铁量侧保持父包原样”这一前提约束；
- 本轮的负结果同样是有价值证据：它区分了「公开弱锚点上有效」与「真实强基座上有效」。

## 9. 私有证据位置

```text
local/runs/round2-v4.3-strong-base-time-residual/smoke-r1      # 协议 smoke（seeds × fold 0）
local/runs/round2-v4.3-strong-base-time-residual/coarse-r1     # 粗筛（seeds × folds 0,1）
local/runs/round2-v4.3-strong-base-time-residual/complete-r1   # 完整开发（仅通过门槛时）
```

均为 append-only；模型、预测、ledger、报告不进入 Git。
