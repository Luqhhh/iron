# 项目实施报告

截至 2026-09-11，optimization-v0.12 / OPT-25–26 已完成，评分前冻结实施提交 `e6dea84`。本阶段仅本地提交，暂停公开推送。
当前发布仍为 **V1_RATE_STRUCTURAL = 83.0319**（用户回传），R2 = 83.0207 保留回退，未独立核验平台成绩。

- G0：预检、开发与独立冷进程审计通过；255 项 Python 3.12 锁定测试通过，八个 OOF folds 和六个 origins 的新候选冷预测最大差为 0。
- G1：FAIL_NO_RELEASE。V6I 未通过 J、H1 E/改善 origins、最近改善数、H3 E/铁量和 DEV_LONG；V6T 未通过 H1 E；V6B 未通过 J 与 H3 铁量。
- Delta J（候选减 V1）：V6I +0.0002275856，V6T -0.0006288865，V6B -0.0004013009。部分改善不能替代完整门槛。
- 预算：16 次新直接目标 CatBoost + 12 次 LAD；旧模型和推理 0 fit；最终 0 fit、无 challenger、无上传。
- 原 V1/R2 包、发布指针和桌面包不变；固定 recency60 关闭，V2–V5 保持关闭。November 已消费，不作独立 holdout 或平台收益推断。
- 仓库仍为 public，数据历史处置待单独实施；已私下保存 Git 历史和摘要，未更改可见性或清史。Actions 日志/artifact 内容因下载权限未核验。

[完整 v0.12 结果](optimization_v0_12/RESULTS.md) · [数据与发布边界](optimization_v0_12/DATA_PUBLICATION_REVIEW.md) · [当前发布](optimization_v0_8/CURRENT_RELEASE.md) · [文档索引](INDEX.md)

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
