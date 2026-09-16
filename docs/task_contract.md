# 当前实施范围

截至 2026-09-16，v0.22 已按冻结规格完成开发：G0/G1 均 PASS，状态
`DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY`；H2 平均 ΔE -0.0021665373、5/5 origins
改善，J Δ -0.0011855502。用户本轮明确允许在缺少 V21 原 ZIP 字节时继续，但该
waiver 不等于原包核验，也不放宽 V21 规则回放、来源、预算或质量门槛。实际预算为
2 forest + 2 preprocessor + 512 树、12 个 lambda 槽和 16 个中位数统计；无铁量
fit。开发验收后，用户另行授权以 0 个最终森林、2 个最终 lambda 和 2 个最终中位数
生成一个 test_a ZIP 并写入桌面；上传为 0、发布指针不变。正式复赛身份仍待独立核验。
[v0.22 结果](optimization_v0_22/RESULTS.md)。

该包最新用户回传为 83.1166，比 V21 低 0.1209、比 V10 低 0.0785；V22 test_a
候选据此关闭。反馈不回写原开发 G0/G1，也不授权阈值、窗口、铁口子集或逐行搜索。

截至 2026-09-15，项目已补录用户在测试平台回传的最新 test_a 成绩 **83.2375**：候选为 V21 `T_GATE_V10_QRF_MEDIAN_RECENT60_SPOUT1_ONLY`，仅对 spout=1 且 `effective_neighbors < 500` 的 32 行做 25% 向训练集同喷嘴近 60 天中位数收缩；未重新拟合模型、预处理器或 LAD。该实验包的 ZIP SHA-256 为 `1a1d34ba96501da1391d2f97b237630f25661b718439efd070c7e52186d589a6`，反馈记录见 [`platform_feedback.json`](../local/runs/optimization-v0.21-time-gate-spout1-r1/platform_feedback.json)。该分数由用户手工提交回传，项目未独立登录平台核验；正式发布指针仍保持 V1，V21 作为当前最高的实验性用户反馈保留。

此前 V11_V6I_IRON_QRF_MEAN_TIME 用户回传 **83.1806**，比 V10 低 0.0145 分；已保留原 ZIP 身份并关闭该均值探针。[独立 V11 反馈](../local/runs/platform-probes-r2-v11-feedback-r1/platform_feedback.json)、[交付与推送回执](../local/runs/platform-probes-r2-publication-r1/publication_receipt.json)。

本轮 platform-probes-r2 完成 V11 零拟合探针和 V10 独立模型推理；随后 v0.18–v0.21 完成不读取 test target 的时间门控实验。旧 G1、发布指针与原包不变。[实施规格](platform_probes_r2/PLAN.md)、[V21 反馈](../local/runs/optimization-v0.21-time-gate-spout1-r1/platform_feedback.json)。

截至 optimization-v0.15，baseline-v0.1-reproducible保持冻结。当前V1活动发布、R2回退与用户成绩证据等级不变；V2–V8的固定失败设计和D1/D2诊断均不获得发布授权。

用户后续指令“生成提交包放到桌面”“是v8”另行授权 V8 初赛实验提交，在独立 `local/runs/optimization-v0.15-v8-user-test-a-r1/` 冻结源码/输入/保护契约、追加独立账本，完成一次最终 QRF 和一次预处理、335 行冷验及桌面交付。用户回传 83.1636，比 V1 高 0.1317 分，绑定新包摘要；本次成绩登记不新增 fit。该后续授权及平台反馈不回写原开发 FAIL，不自动变更活动发布。[独立反馈记录](../local/runs/optimization-v0.15-v8-user-test-a-r1/platform_feedback_r1.json) 仅存 local。

本次 V10 用户回传 **83.1951** 的登记只核对原组合包及两列来源，新增拟合/标签读取/封包/上传均 0；将最高初赛用户回传记录更新为 V10，保留原高分 ZIP、原开发 FAIL、发布登记与历史账本。分数与原先预期在显示四位小数上一致；不宣称独立账号核验或复赛质量确认。[V10 反馈](../local/runs/optimization-v0.15-v10-platform-feedback-r1/platform_feedback.json) 及维护快照仅存 local。

用户本次明确授权提交推送当前成绩维护，以及删除桌面提交包；仅删除五份已与 local 原包核验一致的桌面 ZIP，保留全部本地原始产物。推送范围仅五份可维护当前摘要，不包含模型/预测/账本/提交包或新赛事数据。平台测试评估只登记可选 D2 与第二梯队 V2，不授权新拟合、封包或上传；本次上述数量均 0。旧失败、旧目录与发布配置保持原样。[本次范围](../local/runs/optimization-v0.15-feedback-push-cleanup-r1/PLAN.md)。

以下为原 OPT-32–33 注册与开发收口范围，最终预算 0 描述该原阶段。

用户后续指定优先测试 V6I/V6T/D1，并明确 V6I 超过 V1 时组合其铁量与 V8 时长。这项独立实验授权已用于 2 次单目标最终 CatBoost＋3 次标量 LAD，三包交付与冷审计完成；不改判原开发 FAIL 或 D1 诊断身份。用户更正 V6I 为 83.0634，高于 V1 0.0315 分，满足组合条件。本次新增拟合/标签读取/平台上传均 0，直接按 ID 复制原 CSV 列，交付独立 V10 桌面包；封包时理论预期 83.1951 与其后同值用户回传分开登记，不改判历史质量。V8 原包与 83.1636 原最高回传记录保留，V6T 随后回传 82.9852、D1 随后回传 82.9993，v0.16 在该阶段暂缓。见 [新阶段注册](../local/runs/optimization-v0.15-v10-target-composition-r1/PLAN.md)；不覆盖旧包，不自动改变发布指针。

本次 V6T 成绩登记只核验原包身份及未改铁量列、保存用户回传证据与维护快照；所有拟合、标签读取、封包和上传均为 0。比较结果为 V6T 比 V1 低 0.0467、比 V8 低 0.1784 分，初赛时长选择继续保留 V8；不据此改写旧回溯结果或复赛选择协议。[新增反馈](../local/runs/optimization-v0.15-v6t-platform-feedback-r1/platform_feedback.json) 及新账本仅存 local。

本次 D1 成绩登记只核对原包身份、ID 与未改铁量列，保存用户回传 82.9993 及维护快照；原 D1 诊断角色不改，所有拟合/标签读取/新包/上传为 0。三条回收实验已完成用户成绩登记，初赛保留 V6I 铁量收益与 V8 时长，其后已收到 V10 的 83.1951 用户回传，不追加候选搜索。[D1 反馈](../local/runs/optimization-v0.15-d1-platform-feedback-r1/platform_feedback.json) 仅存 local。

本轮唯一 V8_QRF_TIME：原完整V1铁量exact复制，原E09/R2 as-of特征经train-only填充/one-hot进入独立QRF worker，full-training-row叶分布取固定较小中位数。6 forest + 6 preprocessor、1536树预算完成；无新CatBoost/E04/rate/q、LAD或后校准。模型/原矩阵/历史/cutoff/source_contract/旧包不改。原P0元数据入口失败保留，工程修复只恢复认证OOF元数据和既有交接，无额外fit。

G0独立冷审计通过，G1十项质量门槛失败、FAIL_CLOSE_V8_RETAIN_V1。H2 delta E +0.0025314055，J +0.0032074898；H2 0/5改善。根锁定Python3.12.12测试364项，独立worker25项，分别报告；推理所有fit尝试0。D2加权均值只诊断，不晋级。失败不授权叶子/树数/分位数/种子/融合/路由/偏置/旧recency变体搜索，条件最终预算不使用。

manifest/源码/锁/输入/旧证据/保护契约在访问标签前冻结并追加新ledger。训练仅认证cutoff history，全部新输出摘要先落盘再读outer评分归档；P0旧V1已消费指标重建提前获准。November已消费，不是untouched holdout。固定J贡献和每cell半时长增量恒等式在1e-12内重建；原组件1e-10、候选exact等容差未放宽。后续固定诊断只读冻结errors，数字完成时ledger快照与追加前缀证据分别绑定，旧ledger不变。

final model/preprocessor、新challenger ZIP、平台上传、active pointer/桌面覆盖均0。官方新版身份/source语义/质量/平台状态独立记录；旧包保留原README发布身份，与09-21版本关系待核验。保持原2754行与2024-12-01 01:44+08 cutoff定义，不移动训练历史或虚构December标签。当前固定QRF已关闭，正式版兼容也不授权失败模型最终训练。

所有新模型及响应/叶成员、逐样本目标/预测/误差、账本与私有证据仅local。v14已随后按用户指令推送f940f9f并成功CI；v15随后按用户明确指令推送代码6d19a29，实验提交产物未进入Git。未自动修改visibility、force-push或清史。旧阶段报告/配置/目录/失败/回执保持原文与摘要。后续训练需新明确阶段登记。

[冻结计划](optimization_v0_15/PLAN.md) · [执行结果](optimization_v0_15/RESULTS.md) · [新增维护](optimization_v0_15/MAINTENANCE_20260912.md)
