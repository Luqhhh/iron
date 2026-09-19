# v0.32：同铁口优先的 OOB 响应条件化

状态：冻结实施方案；真实执行结果由本轮不可覆盖的本地 run 证据登记。

审阅日期：2026-09-19。基点：`optimization-v0.31-oob-occurrence-pooling@656be40fc19283972f6c757917d8caf460d75cd4`。执行分支：`optimization-v0.32-same-spout-oob-responses`。

## 1. 决策与候选

v0.31 两项出现次数池化的用户回传为 83.3123、83.3116，均低于 V30A 的 83.3175。v0.32 恢复 V30A 的 256 棵源森林、v0.29 OOB selected members 和每树等权，不继续扩展 v0.31 权重规则。

- A `V32I_SAME_SPOUT_OOB_BLEND`：只对铁量 QRF 的原 selected members 做同铁口优先条件化；保留 V26A 完整 CatBoost 铁量端点、50/50 `mean6` 和 V30A 时长。
- B `V32T_SAME_SPOUT_OOB_TIME`：只对时长 QRF 做同样条件化；保留 V30A 铁量、原 v0.15 支持度 gate、V21 历史中位数与 0.25 收缩。

两个候选、最终预测和 ZIP 必须在首次本轮平台反馈前同时冻结。A 的反馈不得改变 B；不生成第三个组合候选。

## 2. 唯一允许的成员变化

对每棵树的原 v0.29 selected set `A_b(x)`，以及按冻结训练 ID 对齐的原始铁口 token `s_i`，计算：

`C_b(x) = {i in A_b(x): s_i = s_x}`。

- 查询铁口在冻结预处理器 vocabulary 中，且 `C_b` 非空：使用 `C_b`，一个成员合法。
- 查询铁口已知但 `C_b` 为空：精确保留原 `A_b`；不得去 full leaf 另找同铁口行。
- 查询铁口缺失或未知：全部树旁路条件化，保留原 `A_b`。

条件化结果必须是原集合的非空子集。禁止重训、借邻叶、跳树、引入最低计数阈值、改 bootstrap 重数或把 `"1.0"` 归一成 `"1"`。原 OOB 空叶回退和本轮同铁口空交集回退是两个不同标志，分别报告。

## 3. 权重、预测链与数值身份

继续使用原 `distribution_weights` 与 `lower_median`：每树总质量相等、树内成员等权，取较小加权中位数。精确 Fraction 回退必须接收条件化后的成员集合。禁止 v0.31 occurrence pooling、recency 权重及 estimator 默认预测。

A 从 V26A 完整铁量端点和新 QRF 铁量重新组成原 50/50 `mean6`；不得把已含一次融合的 V30A 铁量再与新 QRF 平均。A 的时长 CSV 字符串逐行复制 V30A。

B 的新原始时长中位数先做原六位 round-trip，再运行被冻结的 `spout=1 and effective_neighbors_v015<500` gate 与 V21 60 日历史中位数收缩。gate 支持度不得改用条件化后的有效邻居数。B 的铁量 CSV 字符串逐行复制 V30A。

禁用条件化时，两个目标都必须精确恢复 V30A 完整输出。

## 4. 身份、预算与评价

训练及查询 `spout_no` 使用冻结输入中的原字符串 token，并按模型 `training_ids`、预处理器 `training_ids` 和查询 `sample_id` 逐行核验。只允许重排元数据以匹配已冻结模型顺序；不得重排树、响应、矩阵或 selected-member 位置。

开发仍为 June–November 六个 cutoff，最终为 2024-12-01 01:44:00+08:00、2,754 训练行和 335 个 test_a 查询。新增模型、树、预处理、LAD、beta、lambda、bias 拟合均为 0；两目标各读取七份条件化协议及七份既有 OOB 附件，最多生成两个候选包和两次显式平台测试。自动上传、恢复上传、桌面覆盖和公开推送均为 0。

历史评价使用同一实际样本、未舍入标签与六位预测，主口径为 origin 宏平均；exposure pooled 只作诊断。报告 V30A_PARENT、V1、V21_REPLAY、V28I、V29I、V29T。逐 cell 与 macro 核验：A 的时长变化为 0 且 `Delta E = 0.5 Delta W_iron`；B 的铁量变化为 0 且 `Delta E = 0.5 Delta W_time`。

按目标、cutoff、查询铁口报告原/新跨铁口质量、同铁口非空树比例、空交集回退、原 OOB 回退、新集合大小、不同响应行数、有效邻居数、预测改变量和实际改动行数。跨铁口质量不增加是实现不变量，不是提分证据。

## 5. 冷审计与提交边界

七个 cutoff 全部从独立进程恢复，覆盖 full/reverse/chunk/subset/single 推理；核验源树、随机状态、响应、bootstrap 与 v0.29 attachment 不变。真实 fit 拦截计数必须为 0。

若某候选在提交精度下与父包全同，登记 `NO_OP_FINAL_PAYLOAD` 并跳过，不扰动数值凑实验。否则封装独立 `result.csv`/ZIP 并在反馈前冻结。当前保底 V30A 用户回传 83.3175，ZIP SHA256 为 `fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986`，payload SHA256 为 `97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a`。

本轮 agent 自动上传为 0。任何平台操作前须由用户核对账号当前生效条目、当日额度、截止时间及恢复高分包的机会。G0 工程状态、已消费历史 G1 和用户回传平台结果分开登记。
