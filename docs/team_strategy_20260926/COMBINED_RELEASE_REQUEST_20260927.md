# 恢复优化与组合交付请求（2026-09-27）

## 当前结论与证据边界

用户明确要求“深入研究或优化，给我提交包”，恢复此前 V16 后暂停的工作；随后回复两个已报分原包“不在我的电脑上”。本机安全查找未发现 V12/V7 原 ZIP，实际组合提交包尚未生成。此处交付的是已验证的工具和执行说明，不是比赛 ZIP。

Material Passport：`ANALYZED`；academic-research-suite experiment-agent `validate / implementation`；版本 `original_column_composition_v1`；日期 2026-09-27。工程验证使用合成输入，真实原包组合及其新进程审计尚待输入。新增正式模型拟合、全量训练、实际比赛提交包、桌面写入和平台上传均为 0。

当前最佳仍是用户回传 **V12_IRON_JOINT_PLR001_A50 = 96.3526**；V7_TIME_PLR001_A50 = 96.3519。两者均已报分，不作为未测试候选重新推荐。

## 优先策略：原列组合，不重训或重新选权

拟交付候选 `V12_A50_IRON_V7_A50_TIME_COMBINED`：按官方 sample_id 对齐，取 V12 原始铁量列与 V7 原始时长列，逐字段保留原数值字符串，不重新训练、不增加 alpha 搜索。

理由：V12 相对共同 A35 基准只改铁量，V7 只改时长；原列组合保留两个已经分别获得平台改善的分量，不引入新模型、选权或数值重格式化的额外不确定性。它是合理的低成本候选，但不是已测最优组合。

在同一隐藏样本/标签、既定等权 WMAPE、原列身份和报分归属正确的条件下，算术为 `96.3526 + 96.3519 - 96.3366 = 96.3679`，保留 ±0.0003 的报分精度余量。**不是平台实测、不是保证，仍不能承诺 >96.4。** 固定端点增权的条件界限及精度敏感性见 [反馈后复核](STATUS_UPDATE_20260927.md)，不把该局部界限推广为所有路线的上界。

## 为什么本机不能直接重建原包

- 工作区、相关桌面/下载/文档位置未找到 V7/V12/A35 原 ZIP；仓库登记的是队友 `lqh22` 电脑上的路径，而非本机 `28639`。
- 本机发现的旧 V4.1 包属于 `V41_PUBLIC_ANCHOR_HGB_L2_L15_A050_TIME`，记录本地分约 96.119985，不是 A35/V36/V12/V7，不能冒充当前最佳或新提升包。
- 从代码完整重建还依赖缺失的 V3.6 冻结 summary、权重/训练账本、A35 原包和 V7/V12 确认 OOF/回放材料。公开源码不足以复现已经报过分的原始预测；重新训练也不能冒充原件。
- 精确原列组合只需要两个原 ZIP，不必把模型、预测或私有缓存提交到 Git。

## 在存有原包的电脑上生成 ZIP

拉取本分支代码后，在仓库根目录运行 [独立封包工具](../../scripts/compose_v12_v7_release.py)。工具只用 Python 标准库（Python 3.10+），不需要重新安装训练依赖。下面路径来自已有团队登记；若文件已移动，替换为实际路径：

```powershell
python scripts/compose_v12_v7_release.py build --iron-zip "C:/Users/lqh22/Desktop/submission/V12_IRON_JOINT_PLR001_A50/Luqhhh_bf_tap_predict_round2.zip" --time-zip "C:/Users/lqh22/Desktop/submission/V7_TIME_PLR001_A50/Luqhhh_bf_tap_predict_round2.zip"
```

WSL 下应使用对应 `/mnt/c/Users/lqh22/...` 输入路径。原 ZIP SHA-256 必须分别为：

| 原件 | SHA-256 |
| --- | --- |
| V12 铁量 A50 | `a1c205a6722da3976a12e258458b649967c7c25130a2c55d840c5ecb2a1dc669` |
| V7 时长 A50 | `4382523c7bd688974f87eab2f42502bf8f54b2b330008b36e797ae7672490299` |

成功后生成：

```text
local/runs/round2-v12-v7-combined/release-r1/
  Luqhhh_bf_tap_predict_round2.zip  # ZIP 内只有 result.csv
  result.csv
  manifest.json
  verification.json
  README.txt
```

**这是运行成功后的路径，本机现在没有该实际 ZIP。** 只有命令成功退出、核验记录通过且没有 `FAILED.json` 时才可交付 ZIP；README/审计记录不放入 ZIP。输出已存在或审计失败时保留原证据，下次改用新的 `--output local/runs/round2-v12-v7-combined/release-r2`，不覆盖、不删除旧目录。用户自行上传；不据此假设账户额度已恢复。

## 封包防护与已验证范围（G0）

- 原 ZIP 哈希固定；不接受任意重训包或角色互换。限长读取、检查单一 ZIP 成员/CRC，无路径解压。
- 官方模板固定 322 个 V2 测试 ID 及顺序；源包须同 ID 集，无缺失/重复/额外行，三列顺序正确，数值有限且非负。
- 按 ID 组合而非盲按行拼接；CSV 回读后逐字段比较原数值字符串，源包哈希再次检查。
- 仅新建工作目录 `local/` 下的输出，不写桌面、不读取训练数据、不拟合、不上传。原件不修改。
- 生成后另起进程重新从原件组合并检查 ZIP/CSV 字节、哈希及字段一致性；失败保留产物及 `FAILED.json`，不标记通过。此为同实现的新进程审计，不冒充独立专家审核。
- 23 项专项测试在锁定 Python 3.12 和 3.11 下均通过，覆盖上述边界、合成精度字符串和确定性 ZIP。Python 3.12 完整回归 **1162 passed / 11 skipped / 3 warnings**；跳过项为可选 RealMLP 与缺失私有缓存，不算真实模型再现成功。
- **真实原包的正向组合、新进程审计和平台成绩仍未验证。** 详细证据见 `EVIDENCE_STATUS.json -> round2_user_resumed_combined_release_2026_09_27`。

## 继续模型研究还缺什么

先取得当前参照的私有冻结材料，验证模型、权重、数据 ID、OOF/确认覆盖和原 ZIP 身份；否则无法可靠判断是否优于当前方案。再把真实模型接入已经验证的样本级嵌套核心，让预处理、辅助目标和早停严格限制在当前训练折。新一轮候选、完整覆盖与真实训练成本需要预先登记，保留四 seed 确认和原发布门槛。不能用旧参照、开发分数或跨 seed 稳定性代替样本级独立增益。

V14/V15/V16 的失败证据和资源限制保留；不据少量冻结配方否定整个模型家族，也不继续无界搜索来迎合交付请求。资料恢复后，优先对时长侧做小范围、预登记、完整覆盖的增量验证；同时在相同覆盖下比较当前 V12 和原列组合，而非偷偷改历史权重或门槛。

## 推断审查（11/11）

Simpson / Ecological：不把总体包分直接推到样本/子群改善；Berkson / Base-rate：不把筛选后的少量配方当作家族胜率；Collider / Reverse-causality：不按隐藏误差筛样本、不用结果反推泄露特征；Regression-to-mean / Survivorship：保留 V15 确认失败与所有失败记录；Look-elsewhere / Forking-paths：不追加权重扫点或追认门槛；Correlation-causation：分列加性只作条件指标算术，不声称机制因果或保证实际得分。
