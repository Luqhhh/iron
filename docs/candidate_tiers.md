# 后续候选双轨分类（candidate-tiers-v1）

从下一轮候选评估开始使用，不改变 V2.1/V2.2 的冻结配置、历史选择、已交付包或当前平台基准。分类不新增模型拟合。

- 正式晋级 `formal`：沿用两组完整 OOF 均改善、至少 7/10 折改善、任一铁口 WMAPE 退化不超过 0.001 的门槛。每目标最多推荐一个，按平均 WMAPE 选择；1e-6 内按事先登记的低成本优先顺序选择。
- 探索候选 `exploration`：平均 OOF 相对当前目标参照改善超过数值容差 1e-12，但未通过一项或多项稳定性门槛。完整保留未通过条件、两组收益、折改善数和最差铁口退化，不授予正式晋级资格。
- 不入围 `not_shortlisted`：平均 OOF 没有改善。保留记录，不标注为“已证明无效”。身份错误、缺失指标、非有限指标属于证据错误，直接停止，不能降级成探索候选。

每轮探索推荐最多 **1 个（跨两个目标合计）**。按平均改善降序，再按最小跨切分改善降序，完全相同时按目标名、候选名排序；其余探索候选保留在报告中。正式候选优先，按最小跨切分改善排序。没有正式候选时，探索候选仍可作为待验证方案；额度不足时顺延，不为用完额度自动发布。

分类报告是开发短名单，不自动生成提交包、不修改 incumbent、不调用平台。发布仍需数据/模型摘要核对、同验证折 OOF 检查、独立回读复核和冷推理检查；未修改的目标列复制当前父包的 CSV 字符串。探索候选只做单目标平台验证，平台回传后再决定是否替换当前包。用户自行上传并回传，每天上限 5 次，不重复询问上传方式或额度。

配对重采样仅描述冻结 OOF 对样本组成的敏感性，不作为新增淘汰门槛。探索机制降低误淘汰风险，但不能保证捕获所有平台小幅收益，也不能证明候选有效。

## 接入下一轮

下一轮必须在看结果前冻结候选池、每目标当前参照和低成本平局优先顺序。沿用 metrics 的 `target -> route -> split -> wmape/by_fold/by_spout` 格式。阶段 YAML 包含：

```yaml
split_seeds: [42, 3407]
folds: 5
reference_by_target:
  tap_iron: CURRENT_IRON
  tap_time_len: CURRENT_TIME
candidates:
  tap_iron: [IRON_CANDIDATE]
  tap_time_len: [TIME_CANDIDATE]
tie_preference_by_target:
  tap_iron: [IRON_CANDIDATE]
  tap_time_len: [TIME_CANDIDATE]
```

使用真实冻结阶段文件替换以下占位路径：

```bash
uv run --locked --python 3.12 --extra round2 python -m bf_tap_r2.candidate_tiers \
  --metrics local/runs/NEXT/selection/summary.json \
  --spec configs/NEXT/experiment.yaml \
  --policy configs/candidate_tiers.yaml \
  --output local/runs/NEXT/selection/candidate-tiers.json
```

命令只读取已生成的指标，不训练；输出限制在 `local/`，已存在文件拒绝覆盖。新阶段评估器也可直接调用 `classify_candidates`。旧 `choose_c2()` 和 `choose_targetwise()` 保留历史含义，不自动切换规则。

## 工程验证

2026-09-22：锁定 Python 3.12 环境，新增定向测试 7 项通过，完整测试集 647 项通过。覆盖正式/探索排序、跨目标探索上限、当前目标参照、风险原因、数值容差与无效证据拒绝。G0 工程检查通过；G1 未新增赛事拟合或平台成绩，当前基准及已交付包不变。
