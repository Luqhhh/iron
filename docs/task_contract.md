# 当前实施范围

后续按用户指令提交推送 platform-probes-r2，并已把原 V11 ZIP 字节一致地复制到桌面：`Luqhhh_bf_tap_predict_prelim_V11_V6I_IRON_QRF_MEAN_TIME.zip`，SHA-256 `f11dc1216c4c0742955a7222b16d65bc9ca4c1fd3b11d1c38fb7416f920b5efa`。原开发阶段桌面写入 0 的冻结记录保持原意；本次独立交付新增桌面副本 1，未重训、未重新封包、未上传平台。V11 成绩待反馈，当前最高用户回传仍是 V10 83.1951。[交付与推送回执](../local/runs/platform-probes-r2-publication-r1/publication_receipt.json)。

本轮 platform-probes-r2 已完成唯一 V11 零拟合探针的本地准备，以及 V10 独立模型推理：旧 test_a 335 行全精度分支和原 result.csv 字节精确复现；旧 test_b 322 行仅工程预演，全量/反序/分块/子集/单样本一致性及 root/worker 拟合保护通过。新模型、预处理、LAD 拟合均 0；V11 铁量原字段保持 V10，时长为同一最终 QRF 的加权均值，六位提交精度改变 335 行。根 Python 3.12 锁定测试 382 项、独立 worker 32 项分别通过。V11 未上传、无平台分数，桌面写入 0；当前最高用户回传仍是 V10 83.1951，旧 G1、发布指针与原包不变，V2/v0.16 暂停。平台剩余次数未知，仅本地准备；手动探针需剩余至少两次。[实施规格](platform_probes_r2/PLAN.md)、[本地完成记录](../local/runs/platform-probes-r2-r1/completion.json)。

截至 optimization-v0.15，baseline-v0.1-reproducible保持冻结。当前V1活动发布、R2回退与用户成绩证据等级不变；V2–V8的固定失败设计和D1/D2诊断均不获得发布授权。

用户后续指令“生成提交包放到桌面”“是v8”另行授权 V8 初赛实验提交，在独立 `local/runs/optimization-v0.15-v8-user-test-a-r1/` 冻结源码/输入/保护契约、追加独立账本，完成一次最终 QRF 和一次预处理、335 行冷验及桌面交付。用户回传 83.1636，比 V1 高 0.1317 分，绑定新包摘要；本次成绩登记不新增 fit。该后续授权及平台反馈不回写原开发 FAIL，不自动变更活动发布。[独立反馈记录](../local/runs/optimization-v0.15-v8-user-test-a-r1/platform_feedback_r1.json) 仅存 local。

本次 V10 用户回传 **83.1951** 的登记只核对原组合包及两列来源，新增拟合/标签读取/封包/上传均 0；将最高初赛用户回传记录更新为 V10，保留原高分 ZIP、原开发 FAIL、发布登记与历史账本。分数与原先预期在显示四位小数上一致；不宣称独立账号核验或复赛质量确认。[V10 反馈](../local/runs/optimization-v0.15-v10-platform-feedback-r1/platform_feedback.json) 及维护快照仅存 local。

用户本次明确授权提交推送当前成绩维护，以及删除桌面提交包；仅删除五份已与 local 原包核验一致的桌面 ZIP，保留全部本地原始产物。推送范围仅五份可维护当前摘要，不包含模型/预测/账本/提交包或新赛事数据。平台测试评估只登记可选 D2 与第二梯队 V2，不授权新拟合、封包或上传；本次上述数量均 0。旧失败、旧目录与发布配置保持原样。[本次范围](../local/runs/optimization-v0.15-feedback-push-cleanup-r1/PLAN.md)。

以下为原 OPT-32–33 注册与开发收口范围，最终预算 0 描述该原阶段。

用户后续指定优先测试 V6I/V6T/D1，并明确 V6I 超过 V1 时组合其铁量与 V8 时长。这项独立实验授权已用于 2 次单目标最终 CatBoost＋3 次标量 LAD，三包交付与冷审计完成；不改判原开发 FAIL 或 D1 诊断身份。用户更正 V6I 为 83.0634，高于 V1 0.0315 分，满足组合条件。本次新增拟合/标签读取/平台上传均 0，直接按 ID 复制原 CSV 列，交付独立 V10 桌面包；封包时理论预期 83.1951 与其后同值用户回传分开登记，不改判历史质量。V8 原包与 83.1636 原最高回传记录保留，V6T 随后回传 82.9852、D1 随后回传 82.9993，v0.16 暂缓。见 [新阶段注册](../local/runs/optimization-v0.15-v10-target-composition-r1/PLAN.md)；不覆盖旧包，不自动改变发布指针。

本次 V6T 成绩登记只核验原包身份及未改铁量列、保存用户回传证据与维护快照；所有拟合、标签读取、封包和上传均为 0。比较结果为 V6T 比 V1 低 0.0467、比 V8 低 0.1784 分，初赛时长选择继续保留 V8；不据此改写旧回溯结果或复赛选择协议。[新增反馈](../local/runs/optimization-v0.15-v6t-platform-feedback-r1/platform_feedback.json) 及新账本仅存 local。

本次 D1 成绩登记只核对原包身份、ID 与未改铁量列，保存用户回传 82.9993 及维护快照；原 D1 诊断角色不改，所有拟合/标签读取/新包/上传为 0。三条回收实验已完成用户成绩登记，初赛保留 V6I 铁量收益与 V8 时长，其后已收到 V10 的 83.1951 用户回传，不追加候选搜索。[D1 反馈](../local/runs/optimization-v0.15-d1-platform-feedback-r1/platform_feedback.json) 仅存 local。

本轮唯一 V8_QRF_TIME：原完整V1铁量exact复制，原E09/R2 as-of特征经train-only填充/one-hot进入独立QRF worker，full-training-row叶分布取固定较小中位数。6 forest + 6 preprocessor、1536树预算完成；无新CatBoost/E04/rate/q、LAD或后校准。模型/原矩阵/历史/cutoff/source_contract/旧包不改。原P0元数据入口失败保留，工程修复只恢复认证OOF元数据和既有交接，无额外fit。

G0独立冷审计通过，G1十项质量门槛失败、FAIL_CLOSE_V8_RETAIN_V1。H2 delta E +0.0025314055，J +0.0032074898；H2 0/5改善。根锁定Python3.12.12测试364项，独立worker25项，分别报告；推理所有fit尝试0。D2加权均值只诊断，不晋级。失败不授权叶子/树数/分位数/种子/融合/路由/偏置/旧recency变体搜索，条件最终预算不使用。

manifest/源码/锁/输入/旧证据/保护契约在访问标签前冻结并追加新ledger。训练仅认证cutoff history，全部新输出摘要先落盘再读outer评分归档；P0旧V1已消费指标重建提前获准。November已消费，不是untouched holdout。固定J贡献和每cell半时长增量恒等式在1e-12内重建；原组件1e-10、候选exact等容差未放宽。后续固定诊断只读冻结errors，数字完成时ledger快照与追加前缀证据分别绑定，旧ledger不变。

final model/preprocessor、新challenger ZIP、平台上传、active pointer/桌面覆盖均0。官方新版身份/source语义/质量/平台状态独立记录；旧包保留原README发布身份，与09-21版本关系待核验。保持原2754行与2024-12-01 01:44+08 cutoff定义，不移动训练历史或虚构December标签。当前固定QRF已关闭，正式版兼容也不授权失败模型最终训练。

所有新模型及响应/叶成员、逐样本目标/预测/误差、账本与私有证据仅local。v14已随后按用户指令推送f940f9f并成功CI；v15随后按用户明确指令推送代码6d19a29，实验提交产物未进入Git。未自动修改visibility、force-push或清史。旧阶段报告/配置/目录/失败/回执保持原文与摘要。后续训练需新明确阶段登记。

[冻结计划](optimization_v0_15/PLAN.md) · [执行结果](optimization_v0_15/RESULTS.md) · [新增维护](optimization_v0_15/MAINTENANCE_20260912.md)
