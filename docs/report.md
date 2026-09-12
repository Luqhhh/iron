# 项目实施报告

v0.14 / OPT-30–31 已完成 H2 匹配时长校准：G0 通过，G1 失败、关闭固定 V7；V1 与 R2 回退保持不变。
当前 V1 83.0319、R2 83.0207 为用户回传，非独立平台核验。

- G0：七份真实 H2 与配对 H1、八个旧月度 OOF、六个原 outer 复现和独立冷审计通过；342 项锁定测试通过，铁量完全不变。
- G1：FAIL_CLOSE_V7_RETAIN_V1；H2 delta E +0.0001752136，delta J -0.0001585477；H2 改善 3/5，最近 2/3。
- 预算：新基础模型 0，V7/D1 各 6 时长 LAD；最终拟合/封包/上传/原包覆盖 0。
- 数据：已消费回溯，时间契约 ASSUMED；正式复赛身份与平台有效回执均待核验。
- 维护：v0.13 已随后按用户指令推送 2db6d5f，其 CI 成功；旧报告不回写。本阶段本地提交，无公开推送或历史清理。

[完整 v0.14 结果](optimization_v0_14/RESULTS.md) · [维护观察](optimization_v0_14/MAINTENANCE_20260912.md) · [文档索引](INDEX.md)

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
