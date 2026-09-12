# 当前实施范围

截至 optimization-v0.14，baseline-v0.1-reproducible 保持不可变。当前活动发布 V1（用户回传 83.0319），R2（83.0207）回退不变；关闭的 V2–V6 保持历史 FAIL。

本阶段只改变时长结构系数的校准预报跨度：七份真实 H2 bank、同 ID/同日期/同标签 H1 对照；原模型/特征/history/cutoff/source_contract 不变。V7 与诊断 D1 各六个 constrained 单时长 LAD，铁量直接复制完整 V1。开发预算完成 0 新基础模型、12 时长 LAD；最终拟合/ZIP/上传/旧发布覆盖均 0。结果 FAIL_CLOSE_V7_RETAIN_V1，D1 永不晋级。

校准标签仅来自各 outer cutoff 的认证原历史，先冻结 manifest/注册/来源/保护契约并追加新账本。所有 outer 输出和摘要保存后才读取评分归档。November 已消费，所有评价是回溯开发；早期 outer 标签可用于后期合法 OOF 系数，不能称 untouched holdout。不读取测试真值/分布来调整候选或阈值，禁止追加系数、偏置、路由、参数或历史递归搜索。

G0 独立冷工程通过，G1 历史完整门槛 失败并关闭固定 V7，两者分开。新候选序列化和顺序/分块/子集/单样本 exact equality；原组件 tolerance 1e-10，E/J tolerance 1e-12 未放宽。342 项 Python 3.12 锁定测试通过。正式数据身份、时间语义、最终系数、challenger 和平台核验状态分别记录；旧 B 仍 preview，不能移动旧 cutoff 或补造 December 标签。

旧 run/模型/配置/失败报告/账本/发布哈希不改写，逐样本产物仅 local/。v0.13 在随后用户授权下已推送，旧“当时仅本地”记录保留。v0.14 仅本地提交，不公开推送、不自动改 visibility/force-push/覆盖桌面包；数据历史独立处置尚未完成。

[冻结计划](optimization_v0_14/PLAN.md) · [执行结果](optimization_v0_14/RESULTS.md) · [新增维护观察](optimization_v0_14/MAINTENANCE_20260912.md) · [历史工程基线](review/FREEZE_REPORT.md)
