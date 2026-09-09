# 当前实施范围

截至 optimization-v0.10，用户已授权并完成独立优化阶段。`baseline-v0.1-reproducible`
仍是不可变工程基线；800 轮、MAE、原特征及保护入口的冻结约束只描述该 baseline，
不把已登记的新模型误称为“未改变 baseline 参数”的同一模型。

## 当前结论

当前发布为 V1（用户回传 83.0319），原 R2 保留回退。v0.10 的 V4 完成 12 次开发
residual fit，G0 通过、G1 全部失败；没有 final fit 或 challenger。v0.9 ratio 扩展已关闭。
下一轮模型没有登记，不追加本轮参数、缩放系数或目标消融扫描。

## 标签与时间范围

November 在早期 r2 授权生命周期中已消费，真实 holdout scoring 和 final training
已经执行。后续阶段的 18-cell/H1–H4 是已消费回溯验证，不能声称独立 holdout。
冻结 baseline development 读取器仍拒绝 November；新优化入口必须根据本阶段明确
授权、`configs/protection.yaml`、冻结 manifest 摘要及追加访问账本读取合法范围。
v0.10 用各 outer cutoff 的已核验历史提供 residual 标签，使用此前 causal OOF 预测，
先生成全部外层预测再评分；没有读取测试标签或依据 test_a 调参。

字段可用性仍采用 `competition-timestamp-contract-v1 / ASSUMED`。窗口和报送时点
没有新的官方确认记录。语义变化必须创建新 contract ID，并保留旧 run。

## 产物与验收

使用新 run ID，保留模型、失败证据与摘要，禁止覆盖冻结历史记录。G0 工程与 G1
质量分别报告。现存 V1/R2 配置和 ZIP 不因失败实验改变；本轮不自动上传平台。
历史报告中的“未消费、尚未训练、当前候选”按其阶段解释，见 [文档索引](INDEX.md)。

实现和门槛见 [v0.10 冻结计划](optimization_v0_10/PLAN.md)，验收见
[v0.10 结果](optimization_v0_10/RESULTS.md)。历史 baseline 修复见 [冻结报告](review/FREEZE_REPORT.md)。
