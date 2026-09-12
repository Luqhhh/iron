# 复赛接入清单与后续实验协议草案

2026-09-12。OPT-29 不登记模型、阈值或训练预算。当前 V1 与 R2 不变，V6T 仍为已关闭的 recency60 研究对照，不做最终训练或封包。

## 正式版数据接入前置清单

正式复赛包尚未到官方开放时间 2026-09-21 11:00。已有 test_b 只作 PREVIEW_ENGINEERING_ONLY。正式包到达后，新建唯一只读身份核查目录和 manifest，不覆盖 preview 或原包。

| 检查 | 必需记录 | 当前状态 |
| --- | --- | --- |
| 获取身份 | 官方账号下载时间、官方包名称/摘要、授权来源 | 待正式开放；不把旧同名包冒充正式版 |
| 文件版本 | 每个 CSV/XLSX 的 SHA-256、bytes、schema/dtypes | preview 留存；正式版待核验 |
| 样本身份 | 唯一完整 ID、行数、输入顺序、与训练 IDs 的交集 | preview 通过；正式版待核验 |
| 时间范围 | min/max reference_time、北京时间转换、实际月份和 cutoff 相对位置 | preview 记录；正式版重算 |
| 无标签配置 | 仅当前 stage metadata 与三类无标签源路径 | wrapper 拒绝额外标签/外部历史 |
| source_contract | operation/burden 的完整身份与旧契约逐项对比 | preview 沿用原契约；正式版待核验 |
| 事件/报送语义 | 真实 event/available 字段、官方边界/报送延迟说明 | 原 ASSUMED 不改写；新说明需独立契约 |
| 源变化 | 追加、修订、截断均登记旧/新身份 | 若变化 BLOCKED_SOURCE_CONTRACT_CHANGE；本轮不授权迁移代码 |
| 历史冻结 | cutoff 和原训练/历史摘要、参考时刻前可用性 | 不补造 December 真值或递归使用测试预测 |
| 冷推理 | 新正式 manifest 下原 stage comparator、全量/反序/分块/子集 | 正式版待核验 |
| 评分/平台 | quality_evaluated 与 platform_verified 独立证据 | 当前均 false，工程 PASS 不代替 |

正式封包需另行明确授权，文件 result.csv 仅三列 sample_id/pred_tap_iron/pred_tap_time_len，UTF-8、唯一覆盖 ID、六位小数及 ZIP 回读。当前仅内部 CSV，没有新 ZIP 或上传。

## 阶段映射

以实际 bundle cutoff `2024-12-01 01:44:00+08:00` 为锚。原定义 H1 为同一日历月，H2 为下一日历月；January preview 因而对应 H2。这是原规则下暂定映射，不是正式包 identity_verified。正式数据涉及多个实际月份时逐样本列示 horizon，禁止只凭文件名推断或移动旧 cutoff。

## 下一轮真正训练前必须另行冻结

先完成 OPT-27 的具体可检验缺口和 OPT-28 正式源身份核验。拟定复赛主指标为匹配正式阶段的 origin 等权 mean E，暂定 H2；旧 18-cell J 和 DEV_LONG/SHORT 保留可比诊断。正式映射未确认时不得宣称协议覆盖正式复赛。

下一阶段独立注册必须明确：唯一主要干预及合理输入；原 V1 完整对照；候选目标隔离；每个 fold 原训练身份/label availability/as-of；先前可用 OOF 与系数来源；新预测先冻结后评分；阶段主指标和最低实用收益；两个被改目标各自防退化；跨 horizon/DEV 的工程与质量守门；明确训练预算；失败即关闭、不追加搜索；独立冷推理/来源契约和唯一选择规则。

本草案**不规定新的数值门槛**，不从已见 V6T H2 收益反向设置阈值，不授予任何条件训练额度。一次只引入一个主要变化，不叠加已经失败的 trajectory/recency/ratio。所有旧已消费标签继续标记 retrospective，不创建“未触碰验证”叙述。

当前没有足够证据把误差症状单独归因为漂移、burden 滞后、LAD 或标签错误。先呈交数值证据，再按正式源与因果约束另行注册。
