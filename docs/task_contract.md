# 当前实施范围

截至 optimization-v0.13，用户已授权并完成独立优化阶段。`baseline-v0.1-reproducible`
仍是不可变工程基线；800 轮、MAE、原特征及保护入口的冻结约束只描述该 baseline，
不把已登记的新模型误称为“未改变 baseline 参数”的同一模型。

## 当前结论

当前发布为 V1（用户回传 83.0319），原 R2 保留回退。v0.13 为零训练误差诊断、旧包 stage 适配与正式复赛协议草案；G0 诊断/preview 通过，正式包、模型质量与平台回执分别待核验。固定 recency60 与 V2–V6 均已关闭。
本轮新模型/LAD/校准/候选/challenger/上传/旧包覆盖均 0，不预注册下一训练或设为旧失败候选定制的新阈值。

## 标签与时间范围

November 在早期 r2 授权生命周期中已消费，真实 holdout scoring 和 final training
已经执行。后续阶段的 18-cell/H1–H4 是已消费回溯验证，不能声称独立 holdout。
冻结 baseline development 读取器仍拒绝 November；新优化入口必须根据本阶段明确
授权、`configs/protection.yaml`、冻结 manifest 摘要及追加访问账本读取合法范围。
v0.13 仅读取已消费归档误差及匹配哈希的原模型/历史；冻结源码、来源、分组与账本后诊断，核验原训练/OOF/系数身份，不训练，不读取测试真值或依据测试分布挑模型。

字段可用性仍采用 `competition-timestamp-contract-v1 / ASSUMED`。窗口和报送时点
没有新的官方确认记录。语义变化必须创建新 contract ID，并保留旧 run。

## 产物与验收

使用新 run ID，保留模型、失败证据与摘要，禁止覆盖冻结历史记录。G0 工程与 G1
质量分别报告。现存 V1/R2 配置和 ZIP 不因失败实验改变；本轮不自动上传平台。
历史报告中的“未消费、尚未训练、当前候选”按其阶段解释，见 [文档索引](INDEX.md)。

本轮范围见 [v0.13 冻结计划](optimization_v0_13/PLAN.md)，工程/诊断验收及缺失项见
[v0.13 结果](optimization_v0_13/RESULTS.md)。历史 baseline 修复见 [冻结报告](review/FREEZE_REPORT.md)。

本阶段仅本地提交，数据历史处置完成前不公开推送；没有自动修改仓库可见性或重写历史。
当前 public 状态、元数据核查与日志/artifact 内容权限限制见
[数据与发布边界](optimization_v0_12/DATA_PUBLICATION_REVIEW.md)。
