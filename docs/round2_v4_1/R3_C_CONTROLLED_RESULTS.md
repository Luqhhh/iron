# Round2 V4.1-r3：C2/C3/C4 受控交互执行结果

日期：2026-09-25  
分支：`round2-v4.1-strong-increment`  
HEAD：`193a48549ab8ca561608d85299fe2c4255123974`  
状态：**EXECUTED_NO_PROMOTION**

## 0. 输入与范围

输入包：

```text
D:\Edge\iron_round2_v41_r3_c_controlled_bundle.zip
SHA256: aef460e331ffab205a490102e3fa61ad2fdfff268bdb17e65a5cd7417efcca32
```

本包只实现原 C 预算内的 C2/C3/C4 两目标六个单元：

- 铁量父专家：`v36-s1-D-0029`
- 时长父专家：`v36-s1-O-0057`
- C2：原 V36 父专家 EBM 精确重训；
- C3：保留父配方二元集合的显式重放；
- C4：在 C3 二元集合上加入内部训练/验证选出的至多两个三元项。

未实现 C0/C1 的新逻辑，未实现完整 V36 `fit/predict_state` 后端，未进入 E/F 最终整包路线。
所有输出为 `DESCRIPTIVE_FROZEN_V36_DEVELOPMENT_SCREEN`，不是独立外层验证，不是平台预测。

## 1. 工程验证

### 1.1 合成测试

使用原仓库 `.venv`（Python 3.12.12，InterpretML 0.6.10）：

```bash
cd /tmp/iron_r3_bundle/iron_v41_r3_bundle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/lux1/iron/.venv/bin/python -m pytest -q --disable-warnings
```

结果：`82 passed, 1 warning`。

包内 `BUILD_REPORT.json` 记录的 `81 passed, 1 skipped` 是在未安装 InterpretML 的构建环境；
本仓库环境已安装 `interpret-core 0.6.10`，因此原 native 项也通过。

### 1.2 native preflight

```bash
.venv/bin/python /tmp/iron_r3_bundle/iron_v41_r3_bundle/c41_controlled.py --native-preflight
```

结果：

```text
NATIVE_SYNTHETIC_PREFLIGHT_PASSED
interpret_core: 0.6.10
```

证明原生显式 pair / triple term 与 bag 接口可在当前锁定环境工作；未使用赛事数据。

### 1.3 只读 V4 方向诊断

```bash
.venv/bin/python /tmp/iron_r3_bundle/iron_v41_r3_bundle/diagnose_v4_against_v36.py \
  --root . \
  --output local/runs/round2-v4.1-strong-increment/diagnostic-v36-r3
```

结果与 r2 相同：`76` 个可评分 seed-unit，`4` 个 blocked，`model_training_fits=0`，
无 V4 方向满足嵌套复测条件。

## 2. 时长高维分区的工程阻塞与受控例外

首次按 r3 包原始代码执行两 seed × folds 0/1：

```text
local/runs/round2-v4.1-strong-increment/c-controlled-r1
```

首个 job 完成后，第二个 job（`tap_time_len`）在原生 EBM 中卡死超过 40 分钟。
`gdb` 栈显示停留在：

```text
PartitionMultiDimensionalTree
GenerateTermUpdate
libebm_linux_x64.so
```

原因：`v36-s1-O-0057` 的精确父配方为 `max_bins=256`、`max_interaction_bins=64`、
`interactions=10`、`outer_bags=4`、`max_rounds=6000`。在该精确网格上加入显式三元项后，
原生多维分区不可接受，属于此前 V4.1 已记录过的时长高维分区计算例外。

处理方式：

- C2 始终保持精确父配方；
- 仅 `tap_time_len` 的显式 C3/C4 拟合使用 `max_interaction_bins=16`；
- 铁量保持精确父配方；
- 该变体记录为 `c41_controlled_coarse_time.py`，SHA256：
  `91b88c1abb46780ea683739d1ad95561f937ff3477c5230c2456cd953f5c69ad`
- 不把 C3/C4 时长结果宣称为完整精确父配方的同计算量控制。

## 3. folds 0/1 粗筛

受控执行：

```bash
.venv/bin/python /tmp/iron_r3_bundle/iron_v41_r3_bundle/c41_controlled_coarse_time.py \
  --root . \
  --seeds 42 3407 --folds 0 1 \
  --output local/runs/round2-v4.1-strong-increment/c-controlled-coarse-time-r1
```

单目标完整包增量（`50 × ΔWMAPE`，另一目标固定为同折 V36）：

| target | variant | seed42 | seed3407 | 平均 | 两 seed 同正 |
|---|---|---:|---:|---:|---:|
| tap_iron | C2_parent | 0.000000 | 0.000000 | 0.000000 | 否 |
| tap_iron | C3_same_pairs | 0.000000 | 0.000000 | 0.000000 | 否 |
| tap_iron | C4_pairs_plus_triples | +0.000505 | +0.000869 | +0.000687 | 是 |
| tap_time_len | C2_parent | 0.000000 | 0.000000 | 0.000000 | 否 |
| tap_time_len | C3_same_pairs | +0.000430 | -0.001631 | -0.000600 | 否 |
| tap_time_len | C4_pairs_plus_triples | +0.000825 | -0.000858 | -0.000016 | 否 |

结论：只有 `tap_iron / C4` 两 seed 同正，进入完整五折补齐；其余不纳入。

## 4. 完整开发五折

对 seed 42/3407 的 folds 0–4 执行同一受控流程：

```bash
.venv/bin/python /tmp/iron_r3_bundle/iron_v41_r3_bundle/c41_controlled_coarse_time.py \
  --root . \
  --seeds 42 3407 --folds 0 1 2 3 4 \
  --output local/runs/round2-v4.1-strong-increment/c-controlled-full-r1
```

按 seed 跨样本池化 WMAPE，再做 seed 平均：

| target | variant | seed42 pooled Δ | seed3407 pooled Δ | seed 平均 | 两 seed 同正 |
|---|---|---:|---:|---:|---:|
| tap_iron | C2_parent | 0.000000 | 0.000000 | 0.000000 | 否 |
| tap_iron | C3_same_pairs | 0.000000 | 0.000000 | 0.000000 | 否 |
| tap_iron | C4_pairs_plus_triples | +0.000004 | +0.001034 | **+0.000519** | 是 |
| tap_time_len | C2_parent | 0.000000 | 0.000000 | 0.000000 | 否 |
| tap_time_len | C3_same_pairs | -0.000525 | -0.001684 | -0.001104 | 否 |
| tap_time_len | C4_pairs_plus_triples | -0.000191 | -0.001595 | -0.000893 | 否 |

关键观察：

- `tap_iron / C4` 的完整五折 seed 平均增量仅 `+0.000519`，远低于 `+0.02`；
- `tap_time_len / C3` 与 `C4` 完整五折均负；
- 没有目标达到继续追加候选或取得提交资格的条件；
- 当前 V36 本地完整开发回放仍为 `96.20376256899247`，`96.25` 工作门槛未达到。

## 5. 最终决策

| 项 | 结果 |
|---|---|
| native preflight | 通过 |
| r3 合成测试 | 82 passed |
| 只读 V4 诊断 | 0 方向入围 |
| C2/C3/C4 folds 0/1 粗筛 | 完成 |
| 完整五折 | 进对 `tap_iron/C4` 完成后，全目标失败 |
| `+0.02` 追加触发 | 0 |
| 最终整包候选 | 0 |
| 新 outer seed | 未消费 |
| 平台 ZIP / 上传 | 0 / 0 |
| 原始 V36 包 | 未修改 |
| 新增完整 V36 后端 | 未新增，仍属后续工作 |

最终结论：**V4.1-r3 的 C2/C3/C4 受控路线在当前实现和受控时长分区例外下不产生可继续候选。**
该负结果只约束本包实现、铁口/父配方与评分口径，不升级为模型家族或全局最优结论。

## 6. 产物

私有证据：

```text
local/runs/round2-v4.1-strong-increment/diagnostic-v36-r3
local/runs/round2-v4.1-strong-increment/c-controlled-r1                  # 高维分区阻塞失败证据
local/runs/round2-v4.1-strong-increment/c-controlled-coarse-time-r1      # folds 0/1 受控粗筛
local/runs/round2-v4.1-strong-increment/c-controlled-full-r1             # seeds 42/3407 folds 0-4 完整五折
```

完整五折关键文件：

- `manifest.json`
- `fit_log.json`：125 次 EBM fit
- `ledger.jsonl`
- `fold_metrics.csv`
- `pooled_metrics.csv`
- `development_predictions.csv`

模型、预测、逐样本标签、ledger 和 manifest 均不进入 Git，不自动上传。
