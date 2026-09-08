# OPT-07B 旧校准证据边界

快照的 submission_log 记载 1.68610975 来自 O202406–O202408 拟合、
O202409–O202410 检查。其后 experiment.yaml 在全部 origin 使用同一常数，
旧 run 没有逐 origin 的校准来源可用性约束。这限制全流水线严格 OOF 解释。

本轮尚未恢复该常数的精确拟合样本身份，状态为 EXACT_FIT_SAMPLE_PROVENANCE_UNRESOLVED。
不补造 sample IDs、来源 run、权重或拟合时间。E16 仍是已有发布参数和历史平台
参照；旧 G1 是原有配置下的判定，不改写成绩。

这不构成发现 test_a 或 November 标签泄漏。新 E12-CVcal 每 origin 重新拟合，
同样本比较 E12-raw；它不是平台 82.9918 对应的原 E16。严格时序生成只提高
流水线回放的证据强度，不使多轮使用的开发月份重新成为 untouched holdout。

## 本轮后续模型校准审计

`local/runs/optimization-v0.3-followup-r1/calibration_provenance.json` 保存
CB-CVcal、LG-CVcal、CB-FB-CVcal 在七个运行分组上的 21 份来源记录。
每块 266–284 个唯一校准样本，均达到 100 下限；未触发零回退。
calibration_fit_overlap 与 calibration_available_after_origin 均为 false。
两旧 DEV 折与对应网格 origin 可能共享训练截止点及校准块，不能视作独立拟合数据。

CB-CVcal 在五个网格 origin 的 median(pred_time−actual_time) 分别约为
1.9173、0.4753、1.2974、11.1011、5.0349 分钟；它们是内部时间块上的
算法拟合结果，不是平台反推量，也不作为下一模型的通用常数。
铁量保持零校准。该算法虽改善自身 raw，整体 J 仍未满足相对 C_ref 的
新增晋级幅度；完整结果见 RESULTS_SUMMARY.md。
