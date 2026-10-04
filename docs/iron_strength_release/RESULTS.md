# 铁量强度 q=1.0 零拟合发布（交付记录）

日期：2026-10-05。协议：[PREREGISTRATION.md](PREREGISTRATION.md)。规范：`configs/iron_strength_release/SPEC.json`。私有目录：`local/runs/iron-strength-release-20261005/release-r1`。

## 候选

`DE3_IRON_STRENGTH_100` = 当前最佳 `DE3_IRON_EMA_MEAN3_Q100` 的时长列原字符串 + 铁量列改为 `mean(3 DE3 full-data members)`。

包：`local/runs/iron-strength-release-20261005/release-r1/DE3_IRON_STRENGTH_100/Luqhhh_bf_tap_predict_round2.zip`
ZIP SHA256：`3334e9479387b1db78ecbf1337f5709ef386e26fb7b0042e11a98f7c98b26414`

## 零拟合实现方式（执行前修订，已记录）

原协议要求对两个已保存的 `refit.pt` 做冷推理以取得成员预测。执行前发现发布配方是一个**精确仿射替换**：

```
package_iron(q=0.5) = V32_iron + 0.5 * (mean3 − J42)
⇒ mean3 = J42 + 2 * (package_iron(q=0.5) − V32_iron)
```

其中 `J42` 是 V12 原生成员的 full-data 铁量预测，`V32_iron` 是父包铁量列，`mean3` 正是 q=1.0 列。因此不需要加载任何模型。修订已写入 `SPEC.json` 的 `amendment` 字段（在任何包生成之前），预算相应改为 0 次冷推理。

## 验证链（全部为冻结输入，SHA256 逐一核对）

| 检查 | 结果 |
| --- | ---: |
| 当前最佳包铁量字段 == DE3 发布包铁量字段（逐字符串） | 相同 |
| 由 q=0.5 配方反解出的列重建 q=0.5 == 当前最佳包铁量 | **最大差 0.0** |
| 反解出的 `mean3` == 原发布**独立冷回放记录** `ensemble/cold.npy` | **最大差 1.1368683772161603e−13** |
| 冷回放记录的相对尺度 `cold.json.relative_scale` = 801.093281776908 vs 本列最大值 | 801.0933 ✓ |

第三项是关键：`ensemble/cold.npy` 是原发布阶段对三个 full-data 成员做的独立冷回放产物，**不是**本次计算出来的。两条独立路径吻合到 float64 舍入，说明本列确实等于原配方里的三成员均值。

## 独立进程冷回读审计

`independent-audit.json`（新进程，全部从冻结输入重算）：ZIP 只含 `result.csv`、322 行 322 唯一 ID、官方模板顺序、CRC 通过、有限非负、时长列字符串零改动、铁量列与独立推导逐字符串相同、且确实不同于父包；`passed = true`。终态对账：新拟合 0、optimizer 0、模型加载 0、包 1、桌面写入 0、助手上传 0。

## 上传前筛选证据（已在上传前记录）

| 项目 | 值 |
| --- | --- |
| 改动列 | `pred_tap_iron` |
| `rho` | 0.005923 |
| `bias_pct` | −0.128% |
| `slope` | **−0.004239**（非扩张） |
| 冻结额度门 | **通过**（非扩张且 rho ≤ 0.01） |
| `fitted-weights.npz` 泛函预测 | **+0.003976（正）**，状态：未验证的回顾拟合 |

注意：泛函给的是 +0.0040，而 `iron_strength_bound_20261004` 已独立复核的条件平台上界是 **96.40025（即 +0.0024）**。泛函超出该上界，说明它在此处**高估**——这正是它仍未经验证的直接证据，两者都保留。

## G1 与预期

- 本地证据：两切分 +0.0038 / +0.0022，合并 +0.0030，8/10 折改善（`docs/iron_strength_curve/RESULTS.md`）。
- 条件平台上界：96.40025；**本候选最多把分数推到 96.4003，跨过 96.4，不可能接近 96.45**。
- 分类：探索性两切分强度探针，**非**四 seed 正式晋级；原 q=0.5 决定与历史门槛不变。

## 边界

不预报平台分数；桌面写入与上传均未执行（桌面写入需用户明确要求）；原包、原失败证据与全部历史决定保留。
