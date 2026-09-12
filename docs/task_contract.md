# 当前实施范围

截至 optimization-v0.15，baseline-v0.1-reproducible保持冻结。当前V1活动发布、R2回退与用户成绩证据等级不变；V2–V8的固定失败设计和D1/D2诊断均不获得发布授权。

用户后续指令“生成提交包放到桌面”“是v8”另行授权 V8 初赛实验提交，在独立 `local/runs/optimization-v0.15-v8-user-test-a-r1/` 冻结源码/输入/保护契约、追加独立账本，完成一次最终 QRF 和一次预处理、335 行冷验及桌面交付。用户回传 83.1636，比 V1 高 0.1317 分，绑定新包摘要；本次成绩登记不新增 fit。该后续授权及平台反馈不回写原开发 FAIL，不自动变更活动发布。[独立反馈记录](../local/runs/optimization-v0.15-v8-user-test-a-r1/platform_feedback_r1.json) 仅存 local。

以下为原 OPT-32–33 注册与开发收口范围，最终预算 0 描述该原阶段。

本轮唯一 V8_QRF_TIME：原完整V1铁量exact复制，原E09/R2 as-of特征经train-only填充/one-hot进入独立QRF worker，full-training-row叶分布取固定较小中位数。6 forest + 6 preprocessor、1536树预算完成；无新CatBoost/E04/rate/q、LAD或后校准。模型/原矩阵/历史/cutoff/source_contract/旧包不改。原P0元数据入口失败保留，工程修复只恢复认证OOF元数据和既有交接，无额外fit。

G0独立冷审计通过，G1十项质量门槛失败、FAIL_CLOSE_V8_RETAIN_V1。H2 delta E +0.0025314055，J +0.0032074898；H2 0/5改善。根锁定Python3.12.12测试364项，独立worker25项，分别报告；推理所有fit尝试0。D2加权均值只诊断，不晋级。失败不授权叶子/树数/分位数/种子/融合/路由/偏置/旧recency变体搜索，条件最终预算不使用。

manifest/源码/锁/输入/旧证据/保护契约在访问标签前冻结并追加新ledger。训练仅认证cutoff history，全部新输出摘要先落盘再读outer评分归档；P0旧V1已消费指标重建提前获准。November已消费，不是untouched holdout。固定J贡献和每cell半时长增量恒等式在1e-12内重建；原组件1e-10、候选exact等容差未放宽。后续固定诊断只读冻结errors，数字完成时ledger快照与追加前缀证据分别绑定，旧ledger不变。

final model/preprocessor、新challenger ZIP、平台上传、active pointer/桌面覆盖均0。官方新版身份/source语义/质量/平台状态独立记录；旧包保留原README发布身份，与09-21版本关系待核验。保持原2754行与2024-12-01 01:44+08 cutoff定义，不移动训练历史或虚构December标签。当前固定QRF已关闭，正式版兼容也不授权失败模型最终训练。

所有新模型及响应/叶成员、逐样本目标/预测/误差、账本与私有证据仅local。v14已随后按用户指令推送f940f9f并成功CI；v15随后按用户明确指令推送代码6d19a29，实验提交产物未进入Git。未自动修改visibility、force-push或清史。旧阶段报告/配置/目录/失败/回执保持原文与摘要。后续训练需新明确阶段登记。

[冻结计划](optimization_v0_15/PLAN.md) · [执行结果](optimization_v0_15/RESULTS.md) · [新增维护](optimization_v0_15/MAINTENANCE_20260912.md)
