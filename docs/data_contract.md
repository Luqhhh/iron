# 数据契约状态

审阅日期：2026-09-06。

仓库内已经存在正式 CSV 与 `data_dictionary.xlsx`，原实施包中“数据未提供”的描述已过期。字典确认了公开列名、数据类型、单位、来源字段、样本记录数和允许使用参考时刻及之前数据的总原则。

字典没有额外给出独立的“发布时间”列。为消除不可执行状态，baseline-v0.1 对赛事发布表冻结以下操作口径：

- `operation_hourly`：`event_time = available_at = clock`；只纳入 `clock <= reference_time` 的官方行；
- `burden_change`：`event_time = available_at = cal_time`；只纳入 `cal_time <= reference_time` 的官方事件；
- `tap_history_train` 与监督目标：`available_at = tap_end_time`；只有堵口完成且满足场景冻结起点的历史结果可用；
- 同一时间戳的全字段完全重复行记录后去重，值冲突继续失败关闭。

依据是 PDF 对“参考时刻及其之前数据”的统一约束，以及字典为各公开源提供的唯一业务时间字段。这是可复现的赛事数据操作解释，不虚构不存在的报送时间；若官方后续说明小时窗口或报送延迟，应新建契约版本并使旧缓存失效。样本主键采用官方唯一 `sample_id`；`tap_no` 作为额外唯一性审计字段，不作为数值模型特征。

字典自身还有一处阶段清单不一致：`PackageFiles` 将 A/B 合计描述为 657 行，并把 548 行标成 B，未单列 C；实际主表为 A=335、B=322、C=548。赛题 PDF 第 2 页与实际文件一致，因此实现按 A/B/C 三阶段处理，不使用字典汇总行数作为逐文件强约束。字典 `Summary` 中的 `final.zip` 两阶段命名同样是旧口径；PDF 第 8 页明确为 `prelim`、`round2`、`semifinal`，打包器采用 PDF 口径。

开发流程不得读取 2024 年 11 月目标值。`read_development_labels` 先只读 ID/时间列，再在 CSV 解析层跳过保护行，避免把保护目标加载进开发进程。
