# 项目实施报告

截至 2026-09-12，optimization-v0.13 / OPT-27–29 的零训练诊断和旧包 stage 工程预演已完成，预注册实施提交 `fcbf896`。正式复赛数据和平台有效提交回执仍待核验。本阶段不公开推送。
当前发布仍为 **V1_RATE_STRUCTURAL = 83.0319**（用户回传），R2 = 83.0207 保留回退，未独立核验平台资格或成绩。

- G0：原 E/J/目标分母和逐预测 J 贡献重建通过，最大误差 2.78e-17；286 项 Python 3.12 锁定测试通过。
- 冷工程：test_a 335 行与旧 v8 exact equality；旧 test_b 322 行全量/反序/分块/子集/单样本一致，R2 stage 对照最大差 0；只作 PREVIEW_ENGINEERING_ONLY。
- G1：本阶段未评估新模型质量；v0.12 三个 recency60 候选 FAIL_NO_RELEASE 保持关闭。
- 诊断：V1 最高 10% 唯一铁次约贡献铁量/时长 J 误差的 32.32%/33.72%；H1 时长结构修正略扩大改进，H2/H3 则削弱 base 改进，不能统一归因于 LAD。
- 预算：新增模型、LAD/校准、候选、challenger、上传和旧包覆盖均 0；原模型/历史/发布身份和账本不变。
- 维护：v0.12 62c62cc 已在此前授权后推送且 CI 成功；旧报告保留当时未推送口径。仓库仍 public，历史处置待单独实施。
- 平台回执已自行检索，只有原用户回传记录；动态平台需要账户会话，未取得有效提交证明。正式复赛数据 09-21 开放，现有同名文件不代表正式版。

[完整 v0.13 结果](optimization_v0_13/RESULTS.md) · [维护观察](optimization_v0_13/MAINTENANCE_20260912.md) · [正式接入协议](optimization_v0_13/SECOND_ROUND_PROTOCOL.md) · [文档索引](INDEX.md)

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
