# Round2 V4.1-r2：V36 固定配方残差诊断执行结果

日期：2026-09-25  
分支：`round2-v4.1-strong-increment`  
HEAD：`3f32d242093cb1cde8eac1cd38223001c470f262`  
状态：**EXECUTED_NO_PROMOTION**

## 0. 范围与声明

本轮执行的是 `iron_round2_v41_r2_v36_bundle` 的 V4.1-r2 路线：

- 只读 V36 冻结开发回放 + V4 旧方向重诊断；
- E0 / E1-state / E3-state 的 V36 嵌套残差粗筛；
- E2 的 r2-state 适配版本，显式记为 `E2_state_linear_leaf`；
- C0–C4 相对 V36 冻结开发回放的粗筛重估。

协议声明：

1. 使用的 V36 后端是 **fixed-recipe**：
   - A_dev 由 `FrozenAReferenceFactory` 在当前训练部分重训；
   - V36 四个选定专家 `D-0029 / N-0005 / O-0057 / D-0048` 在当前训练部分重训；
   - 冻结 V36 顶层权重直接复用，未在 T/V 内重新执行顶层约束选权。
2. E2 原始 V4.1 实现使用 21 个 raw features；r2 core 的 state 不携带 raw features。
   本轮 E2 使用同样的 LightGBM linear-tree 机制，但输入为 r2 prediction state，
   并显式命名为 `E2_state_linear_leaf`，不是原 raw-feature E2 的重放。
3. 外层评价标签只在 `predict_variants` 完成后用于评分；没有按外层标签扫描 alpha。
4. 没有完整开发五折，没有消费新 outer seed，没有平台 ZIP，没有上传。

当前平台最佳仍是用户回传的：

```text
V36_USER_REQUESTED_OUTER_FAILED = 96.2734
```

这是用户回传，未独立核验；距离严格 `> 96.3` 还差 `0.0266`。

## 1. 测试与只读诊断

### 1.1 r2 bundle 合成测试

```bash
cd /tmp/iron_bundle/iron_v41_r2_bundle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/lux1/iron/.venv/bin/python -m pytest -q --disable-warnings
```

结果：`54 passed`。

### 1.2 只读 V36 重放 + V4 方向诊断

```bash
cd /home/lux1/iron
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python /tmp/iron_bundle/iron_v41_r2_bundle/diagnose_v4_against_v36.py \
  --root /home/lux1/iron
```

输出目录：

```text
local/runs/round2-v4.1-strong-increment/diagnostic-v36-r1
```

结果：

| 项 | 值 |
|---|---:|
| V36 冻结回放均值 | 96.20376256899247 |
| seed 42 | 96.20130931837855 |
| seed 3407 | 96.20621581960641 |
| 可评分 seed-unit 事件 | 76 |
| blocked seed-unit 事件 | 4 |
| method-target units | 38 |
| 新基础模型训练 | 0 |
| 满足嵌套复测条件的 V4 方向 | 0 |

`up_to_6_retest_requests_NOT_PROMOTION.csv` 为空。最高同标签 oracle 方向也只有
`v4-J-034/tap_iron = 0.000571`，且两 seed 不同时为正，不构成候选。

## 2. V36 fixed-recipe 接口检查

先执行一个合法外层折：

```bash
.venv/bin/python scripts/round2_v4_1/run_r2_v36_nested.py \
  --root . \
  --output local/runs/round2-v4.1-strong-increment/r2-v36-interface-r1 \
  --seeds 42 --outer-folds 0 --workers 16
```

检查结果：

| 项 | 值 |
|---|---:|
| `base_fit_calls` | 13 |
| `corrector_fits` | 24 |
| `scalar_optimizations` | 6 |
| 训练/评价组交叉 | 无 |
| baseline 折内 package score | 96.26441297591434 |
| E0 package delta | +0.0061823865 |
| E1-state package delta | +0.0065593658 |
| E3-state package delta | +0.0002378520 |

接口检查通过后，才继续 4 个粗筛外层折。

## 3. E 线粗筛结果

粗筛配置：seed `42/3407`，outer folds `0/1`，共 4 个外层折。
共享 fixed-recipe V36 基础拟合 `52` 次。

E 线的任务书口径是“单目标完整包增量”：候选只替换该目标，另一目标固定为同折 V36。
下表因此报告每个 method-target unit 的单目标 package delta，而不是两个目标同时修正后的联合值。

| method | target | seed42 平均 | seed3407 平均 | 两 seed 同正 | 平均单目标 package delta |
|---|---|---:|---:|---:|---:|
| E0_median | tap_iron | +0.003051 | +0.000038 | 是 | +0.001545 |
| E0_median | tap_time_len | +0.000064 | -0.000603 | 否 | -0.000270 |
| E1_state_spline_median | tap_iron | +0.003051 | +0.000038 | 是 | +0.001545 |
| E1_state_spline_median | tap_time_len | +0.000045 | -0.000404 | 否 | -0.000179 |
| E2_state_linear_leaf | tap_iron | -0.000378 | -0.000128 | 否 | -0.000253 |
| E2_state_linear_leaf | tap_time_len | -0.002082 | +0.001263 | 否 | -0.000410 |
| E3_state_local_median | tap_iron | -0.004131 | +0.003915 | 否 | -0.000108 |
| E3_state_local_median | tap_time_len | -0.000883 | -0.001076 | 否 | -0.000980 |

逐外层折的单目标 package delta：

| method | target | seed42 fold0 | seed42 fold1 | seed3407 fold0 | seed3407 fold1 |
|---|---|---:|---:|---:|---:|
| E0_median | tap_iron | +0.006054 | +0.000049 | -0.000259 | +0.000335 |
| E0_median | tap_time_len | +0.000128 | 0.000000 | -0.001206 | 0.000000 |
| E1_state_spline_median | tap_iron | +0.006054 | +0.000049 | -0.000259 | +0.000335 |
| E1_state_spline_median | tap_time_len | +0.000505 | -0.000415 | -0.000841 | +0.000034 |
| E2_state_linear_leaf | tap_iron | +0.000045 | -0.000802 | -0.000161 | -0.000095 |
| E2_state_linear_leaf | tap_time_len | -0.002981 | -0.001184 | 0.000000 | +0.002525 |
| E3_state_local_median | tap_iron | +0.000072 | -0.008334 | +0.007287 | +0.000542 |
| E3_state_local_median | tap_time_len | +0.000166 | -0.001932 | +0.000442 | -0.002594 |

结论：

- E0/E1 的铁量 unit 两个 seed 平均为正，但平均单目标 package delta 仅约 `+0.001545`，远低于 `+0.02`；
- 没有任何 E method-target unit 同时满足“两个开发 seed 平均为正”和“平均单目标 package delta >= 0.02”；
- E 线不进入完整开发五折；
- 不追加参数邻域；
- 不把折内单点正增益升级为路线结论。

如需复核联合修正结果，原始事件级 jointly-corrected package delta 保留在
`r2-v36-interface-r1/summary.json` 与 `r2-v36-e2-state-r1/summary.json`；
单目标列表另存于 `r2-v36-e-per-target-events.csv` 和 `r2-v36-final-summary.json`。

## 4. C 线 V36 相对重估

C 线仍按原 V4.1 C0–C4 配方拟合，只把参考替换为 V36 冻结开发回放：

```bash
.venv/bin/python scripts/round2_v4_1/run_r2_v36_c_line.py \
  --root . \
  --output local/runs/round2-v4.1-strong-increment/r2-v36-c-line-r1 \
  --seeds 42 3407 --folds 0 1
```

20 个 method-target unit 的 seed 平均 package delta：

| method | target | seed42 | seed3407 | 两 seed 同正 |
|---|---|---:|---:|---:|
| C0 | tap_iron | -0.117892 | -0.114599 | 否 |
| C0 | tap_time_len | -0.700415 | -0.783363 | 否 |
| C1 | tap_iron | -0.110346 | -0.121144 | 否 |
| C1 | tap_time_len | -0.230499 | -0.320430 | 否 |
| C2 | tap_iron | -0.032272 | -0.016025 | 否 |
| C2 | tap_time_len | -0.103354 | -0.102700 | 否 |
| C3 | tap_iron | -0.034454 | -0.015858 | 否 |
| C3 | tap_time_len | -0.113971 | -0.124197 | 否 |
| C4 | tap_iron | -0.041171 | -0.020200 | 否 |
| C4 | tap_time_len | -0.111568 | -0.147645 | 否 |

结论：所有 C 线 unit 相对 V36 均为负，且没有两 seed 同正。C2 是最接近 A/V36 的父配方，
但仍不足以构成完整包增量。

## 5. fixed-recipe baseline 与 V36 回放的折内对照

| seed | outer fold | fixed-recipe baseline | V36 replay | delta |
|---:|---:|---:|---:|---:|
| 42 | 0 | 96.264413 | 96.265499 | -0.001086 |
| 42 | 1 | 96.309684 | 96.308857 | +0.000827 |
| 3407 | 0 | 96.254945 | 96.260986 | -0.006041 |
| 3407 | 1 | 96.055904 | 96.054176 | +0.001728 |

该对照只证明 fixed-recipe adapter 与冻结回放折内分数接近，不代表 full top-level reselection、
独立验证或平台预测。

## 6. 最终决策

| 项 | 结果 |
|---|---|
| D0 旧方向诊断 | 0 方向入围 |
| fixed-recipe 接口检查 | 通过；13 / 24 / 6 |
| E0/E1/E3/E2 粗筛 | 均已执行；无两 seed 同正且 >=0.02 |
| C0–C4 V36 相对重估 | 20/20 unit 无两 seed 同正 |
| 完整开发五折 | 未进入 |
| 最终 outer 新切分 | 未消费 |
| 平台 ZIP / 上传 | 0 / 0 |
| 原始 V36 包 | 未修改 |
| 平台最佳 | V36 96.2734（用户回传） |

本批结论为负：在 fixed-recipe V36 与当前实现范围内，没有 E/C unit 达到继续条件。
E0/E1 铁量单目标两 seed 同正，但平均只有约 +0.001545，未达到 +0.02 门槛；其余 E unit 和全部 C unit 均未满足两 seed 同正。
该负结果只约束本轮实现、协议和 fixed-recipe 声明，不能升级为模型家族或理论极限结论。

## 7. 代码与私有证据

新增/修改的公共代码：

- `src/bf_tap_r2/v4_1_r2_v36.py`
- `src/bf_tap_r2/v4_1_r2_e2.py`
- `src/bf_tap_r2/v4_1_r2_nested.py`
- `src/bf_tap_r2/v4_1_r2_increment.py`
- `scripts/round2_v4_1/run_r2_v36_nested.py`
- `scripts/round2_v4_1/run_r2_v36_c_line.py`
- `src/bf_tap_r2/v4_1_reference.py`：`fit_predict` 额外暴露 `l1` 与 `experts`，不改变既有预测逻辑。

私有证据目录：

```text
local/runs/round2-v4.1-strong-increment/diagnostic-v36-r1
local/runs/round2-v4.1-strong-increment/r2-v36-interface-r1
local/runs/round2-v4.1-strong-increment/r2-v36-e2-state-r1
local/runs/round2-v4.1-strong-increment/r2-v36-c-line-r1
local/runs/round2-v4.1-strong-increment/r2-v36-final-summary.json
```

模型、预测、报告、ledger 和本地证据均不进入 Git，不自动上传。
