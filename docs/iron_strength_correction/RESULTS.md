# 铁量强度曲线更正（零拟合）

日期：2026-10-05。规范：`configs/iron_strength_correction/SPEC.json`。实施：`scripts/iron_strength_correction.py`。私有证据：`local/runs/iron-strength-correction-20261005/review-r1/report.json`。0 拟合 / 0 新包 / 0 上传。

## 更正什么

`docs/iron_strength_curve/RESULTS.md` 曾判定"incumbent 的铁量放大系数 q=0.5 不是局部最优，两个 split 的最优都在 q=1.0"，并据此发布了 `DE3_IRON_STRENGTH_100`。**该判定错误。**

错因是基准取错：

- 那条曲线用的是 `v12_iron + q × (mean3 − v12_iron)`，其中 `v12_iron` 是**原生 V12 联合成员**（逐折 `reference-*/v12_iron`，即 seed 42 的预测）；
- 而 incumbent 铁量列是 `V32_iron + 0.5 × (mean3 − J42)`，它的基准 `V32_iron` 是**已发布的 V12 混合列**（V12_IRON_JOINT_PLR001_A50），不是原生联合成员；
- 两个基准的 RMS 差约 **4.0**（本更正中 `base_minus_native_rms` = 3.954 / 4.018），远大于被比较的增益量级。

所以原曲线的 q=0.5 点根本不是 incumbent，两条族只共享 `mean3` 这一端点；原结论从未把 incumbent 放进被比较的族里。

## 正确的族

用折级记录重建 incumbent 所在的仿射线：

```
base = incumbent_iron − 0.5 × (mean3 − J42)
iron(q) = base + q × (mean3 − J42)
```

| q | split 42 相对 incumbent | split 3407 相对 incumbent |
| ---: | ---: | ---: |
| 0.00 | −0.0044 | −0.0026 |
| 0.25 | −0.0013 | −0.0006 |
| **0.50（incumbent）** | **0.0000** | **0.0000** |
| 0.75 | +0.0001 | −0.0009 |
| 1.00 | −0.0008 | −0.0035 |
| 1.25 | −0.0029 | −0.0074 |

**incumbent 的 q=0.5 就在最优点**：split 3407 最优在 0.5，split 42 的 0.75 只领先 +0.0001（噪声量级），而 q=1.0 在两个 split 都更差（−0.0008 / −0.0035，均值 −0.0021）。

## 对已发布包的判定

已生成的 `DE3_IRON_STRENGTH_100`（铁量 = 三成员普通均值 `mean3`）相对 incumbent：

| split | 相对 incumbent 总分 |
| --- | ---: |
| 42 | **−0.0173** |
| 3407 | **−0.0163** |

**两个 split 都显著更差，均值 −0.0168。该包必须撤回，不得上传。** 包文件、审计与全部收据按"不覆盖失败证据"的规则保留在 `local/runs/iron-strength-release-20261005/release-r1`，并加撤回标记。

## 为什么之前的筛选没有拦住它

冻结的额度门只检查"改动方向是否扩张、扰动是否过大"（本包 rho 0.0059、slope −0.0042，确实通过），它**不检查改动的方向本身是否指向更差的列**——而这正是四 seed 本地门要负责的事。本包只有两切分曲线证据，而那条曲线又用错了基准，两层防线同时失效。

教训（已写入结果）：**任何"某个系数不是最优"的结论，必须先证明被比较的族里确实包含 incumbent 这一列**；用邻近但不相同的列作基准会把结论整体带偏。后续同类曲线在冻结前必须先做"基准一致性检查"：把参考列代入族内应当精确复原 incumbent。

## G0

单进程、单数值线程；只读已记录 OOF 与逐折预测；10 折 × 6 个 q 值 + 两条直接对照；无模型构造、无 optimizer、无新包、无桌面写入、无上传。
