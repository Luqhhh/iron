# 项目实施报告

截至 2026-09-11，optimization-v0.11 / OPT-24 已完成本地执行，审计完成代码为 `e7952d5`，尚未推送。
当前发布仍为 **V1_RATE_STRUCTURAL = 83.0319**（用户回传），R2 = 83.0207 保留回退。

- 最新 G0：开发与修正后的独立冷进程审计通过；235 项 Python 3.12 锁定测试通过，八个 OOF fold 和六个 origin 预测最大差均为 0，铁量逐样本完全相等。
- 最新 G1：固定 V5 的 J、H1 时长、H1 改善 origin 数、H4 四项门槛失败；J 相对 V1 退化 0.0001383591，H1 时长 WMAPE 退化 0.0002481035。
- 预算：8 次直接时长 CatBoost + 6 次时长 LAD；0 次新铁量/E04/rate/q/residual fit；无 final fit、无 challenger、无上传。
- 两次工程失败及修正证据保留：旧 bundle 的 registry 身份兼容；CSV 时区类型规范化。未重训或重估系数，未改候选、门槛或预测数值容差。
- V2/V3/V4 保持关闭。November 已消费，所有评价均为已消费回溯开发；bootstrap 区间不作独立确认或平台收益预测。
- V1/R2 原包、发布指针和桌面包保持不变；平台成绩未经独立核验。

[完整 v0.11 结果](optimization_v0_11/RESULTS.md) · [当前发布](optimization_v0_8/CURRENT_RELEASE.md) · [文档索引](INDEX.md)

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
