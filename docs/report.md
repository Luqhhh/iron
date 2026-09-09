# 项目实施报告

截至 2026-09-09，optimization-v0.10 / OPT-23 已完成并推送至 `fcda5e0`。
当前发布仍为 **V1_RATE_STRUCTURAL = 83.0319**（用户回传），R2 = 83.0207 保留回退。

- 最新 G0：执行和独立冷进程通过；221 项 Python 3.12 锁定测试通过，六个 origin 预测最大差为 0。
- 最新 G1：固定 V4 九项门槛全部失败，J 相对 V1 退化 0.0053128；12 次 residual 开发训练后关闭。
- v0.9 V2/V3 同样未通过严格门槛；两阶段均无 final fit、无 challenger。
- November 已经授权消费；18-cell、六个 H1 origins、DEV_LONG/SHORT 已执行。旧 baseline 的保护入口仍按原契约运行。
- 桌面为 V1 原包，V1/R2 摘要保持不变；平台成绩未经独立回执核验。

[完整 v0.10 结果](optimization_v0_10/RESULTS.md) · [当前发布](optimization_v0_8/CURRENT_RELEASE.md) · [文档索引](INDEX.md)

## 历史 baseline 修复记录（非当前项目状态）

以下为当时记录，其中测试数、保护集、未执行项和提交动作只适用于该次修复。

当时状态：修复后的工程链路已在本地 Python 3.12 锁定环境验收通过；冻结模型质量未通过。

剩余两个 P1 已收口：bundle v3 强绑定公共 process source 身份，baseline/feature/semantic 配置以 canonical SHA-256 全量冻结。新 v3 DEV 相对既有 baseline 原始预测差为 0；详见 `docs/review/FREEZE_REPORT.md`。

- G0：`PASS_LOCAL_LOCKED_ENVIRONMENT`；48 项测试、真实 DEV 双重训、同 bundle 双独立进程推理、结构审计与提交打包合成集成测试通过。
- G1：`FAIL_DEV_LONG`；DEV_LONG CatBoost E=0.185646，B1 E=0.176101。DEV_SHORT E=0.171456，通过。
- 保护集：`holdout_consumed=false`；H1–H4 未执行，未创建保护访问账本。
- 契约：`competition-timestamp-contract-v1 / ASSUMED`，字段级依据与资产摘要见 `configs/data_contract.yaml`。
- 详细命令、摘要、批次和未解决项见 `docs/review/REPAIR_REPORT.md`。
- baseline 冻结身份和延期 P2 见 `docs/review/FREEZE_REPORT.md`。
- test_a / prelim 平台成绩：`81.4554`，由用户报告，未由本次代理独立核验；提交包身份见 `docs/submission_log.md`。

修复代码提交的 GitHub Actions run `34037523200` 已成功。本状态没有排行榜提交、正式最终训练或 GitHub 分支保护变更。
