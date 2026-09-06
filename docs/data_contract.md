# 数据契约状态

审阅日期：2026-09-06。

仓库内已经存在正式 CSV 与 `data_dictionary.xlsx`，原实施包中“数据未提供”的描述已过期。字典确认了公开列名、数据类型、单位、来源字段、样本记录数和允许使用参考时刻及之前数据的总原则。

字典没有额外给出独立的“发布时间”列。baseline-v0.1 因此冻结以下条件性操作口径，其机器可读定义位于 `configs/data_contract.yaml`：

- `operation_hourly`：`event_time = available_at = clock`；只纳入 `clock <= reference_time` 的官方行；
- `burden_change`：`event_time = available_at = cal_time`；只纳入 `cal_time <= reference_time` 的官方事件；
- `tap_history_train` 与监督目标：`available_at = tap_end_time`；只有堵口完成且满足场景冻结起点的历史结果可用；
- 同一时间戳的全字段完全重复行记录后去重，值冲突继续失败关闭。

`evidence_status=ASSUMED`。依据是 PDF 对“参考时刻及其之前数据”的统一约束，以及字典为各公开源提供的唯一业务时间字段；这些材料不足以证明业务时间就是完整结果的最早发布时间。当前结果只能表述为基于该约定的条件性结果，不能写成可用性语义已经官方验证。若官方后续说明小时窗口或报送延迟，应新建契约版本并使旧缓存及产物失效，不覆盖旧 run。样本主键采用官方唯一 `sample_id`；`tap_no` 作为额外唯一性审计字段，不作为数值模型特征。

字典自身还有一处阶段清单不一致：`PackageFiles` 将 A/B 合计描述为 657 行，并把 548 行标成 B，未单列 C；实际主表为 A=335、B=322、C=548。赛题 PDF 第 2 页与实际文件一致，因此实现按 A/B/C 三阶段处理，不使用字典汇总行数作为逐文件强约束。字典 `Summary` 中的 `final.zip` 两阶段命名同样是旧口径；PDF 第 8 页明确为 `prelim`、`round2`、`semifinal`，打包器采用 PDF 口径。

开发流程不得读取 2024 年 11 月目标值。保护边界不再从普通 fold 推导，而由带摘要的 `configs/protection.yaml` 独立提供。`read_development_labels` 会在解析标签前验证请求范围，再在 CSV 解析层跳过保护行。保护标签只允许在 `holdout_scoring` 或 `final_training` 生命周期中，凭冻结 manifest 摘要写入本地访问账本后读取。

历史表先校验 `reference_time <= tap_end_time <= available_at`，再与样本主表核对 ID、铁次号、铁口号、参考时刻和 DEV 授权范围内的两个标签。11 月仅核对不含目标的元数据。
