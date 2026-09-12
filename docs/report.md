# 项目实施报告

v0.15 / OPT-32–33 已完成固定 QRF 时长分支：G0通过，G1 FAIL_CLOSE_V8_RETAIN_V1。原 V1/R2不变，平台83.0319/83.0207仅用户回传。

随后用户明确要求生成 V8 初赛包并放到桌面，在独立 local 登记中完成原 2754 行与 `2024-12-01 01:44:00+08:00` cutoff 的一次最终 QRF、一次预处理，335 行包通过冷进程及 ZIP 回读检查。用户回传 **83.1636，比 V1 高 0.1317 分**，绑定 ZIP SHA256 `409218c364dac20f59169b08f95af54c1a9e206eb19b71a6115f05ce4889b402`。原开发 FAIL 保留，活动发布仍为 V1，平台有效回执未独立核验。[独立反馈记录](../local/runs/optimization-v0.15-v8-user-test-a-r1/platform_feedback_r1.json) 及完整证据仅存 local。此次成绩登记新增 fit 为 0。

当前新增 V10 用户回传 **83.1951**，比 V8 高 **0.0315**、比 V1 高 **0.1632**、比 R2 高 **0.1744** 分，登记为初赛最高用户回传包；与封包前预期在显示四位小数上一致。核验原桌面 ZIP SHA256 `be83f1f623c12f4ef03479f2fcacdfb756920d41c4e09d1c4f6b55a2279761ac`、335 行覆盖、V6I 铁量列和 V8 时长列精确一致。本次拟合、标签读取、封包、上传均 0。[V10 独立反馈](../local/runs/optimization-v0.15-v10-platform-feedback-r1/platform_feedback.json) 保留用户回传证据等级，未获得账号回执；旧开发 FAIL、原发布登记与原包不变，不据此宣称复赛质量通过。

本次用户要求提交推送成绩维护并清理桌面包。已核验五份桌面 ZIP 与 local 原包字节一致后删除桌面副本，V10 高分原包、模型与旧回执/账本均保留；[清理记录](../local/runs/optimization-v0.15-feedback-push-cleanup-r1/cleanup_receipt.json) 单独登记。平台测试复核无必须追加项，D2 现有最终森林均值为第一可选零新增训练对照，V2 需单独登记最终 q＋LAD、留作第二梯队。本次拟合/标签读取/新包/平台上传均 0，只提交当前维护摘要。

以下预算及验收描述属于原 OPT-32–33 开发收口；用户后续指定的实验提交单独记账。

用户后续指定优先回收 V6I/V6T/D1，独立完成 2 个最终单目标 CatBoost＋3 个标量 LAD，335 行三包通过冷审计并交付；ID 对齐修复不增加拟合，原失败证据保留。用户更正 V6I 成绩为 **83.0634，比 V1 高 0.0315 分**，绑定原桌面 ZIP 摘要。按用户事先指定的正交组合条件，已直接复制 V6I 铁量与 V8 时长的原 CSV 字符串，零拟合交付 V10；独立进程验证 ID 覆盖、反序/分块/子集/单行组合、列精确一致与 ZIP 回读。封包时理论预期为 **83.1951**，其后收到同值用户回传；最高回传现为 V10。D1 随后回传 82.9993，v0.16 暂缓，原质量结论与发布指针不改。[新增完成记录](../local/runs/optimization-v0.15-v10-target-composition-r1/completion.json) 仅存 local。

V6T 随后用户回传 **82.9852**，比 V1 低 **0.0467**、比 V8 低 **0.1784** 分。核验原桌面 ZIP SHA256 `7fde26ec525b1c9fdc70b9162095f5dd38819084ccc852144ce2db48d7deb1b7`、335 行覆盖与铁量列逐字一致 V1，未重新训练、读标签或生成新包。这条初赛反馈支持保留 V8 时长，不外推复赛；V10 原封包时的预期记录保持不变，后续反馈另行登记。[独立 V6T 反馈](../local/runs/optimization-v0.15-v6t-platform-feedback-r1/platform_feedback.json) 未独立登录平台核验；旧完成回执及快照不修改。

D1 随后用户回传 **82.9993**，比 V1 低 **0.0326**、比 V8 低 **0.1643** 分；原 ZIP 摘要、335 个 ID 与铁量列逐字一致 V1 核验通过。[独立 D1 反馈](../local/runs/optimization-v0.15-d1-platform-feedback-r1/platform_feedback.json) 保留其原诊断角色，新增拟合/标签读取/封包/上传均为 0。三条优先回收测试均已有回传，仅 V6I 超过 V1；初赛保留 V8 时长，V10 原封包时预期 83.1951，后续收到同值用户回传，不外推复赛。

- G0：原六个训练矩阵与V1/R2发布回读、旧E/J/分母重建、独立QRF冷进程和反序/分块/子集/单行exact equality通过；推理fit尝试0，铁量exact。
- G1：十项质量门槛失败；H2 delta E +0.0025314055、0/5改善，delta J +0.0032074898。固定QRF关闭，D2加权均值只诊断。
- 预算：6 forest + 6 preprocessor、1536内部树；新旧CatBoost/E04/rate/q/LAD/铁量模型fit均0，final/ZIP/上传/覆盖0。
- 测试：根Python3.12.12原锁364 passed，独立worker锁25 passed，计数和合成fit分开记录；原P0与较早测试失败保留。
- 来源：已消费回溯5651 exposures/1865唯一ID，原时间契约ASSUMED；正式新版身份与平台有效回执未核验。
- 维护：已核实v14远端f940f9f及34690189315成功CI；v15随后按用户指令推送6d19a29，本阶段远端CI未在此次登记中另行核验；旧报告及历史保持不变。

[完整v15结果](optimization_v0_15/RESULTS.md) · [工程恢复](optimization_v0_15/ENGINEERING_REPAIR.md) · [维护记录](optimization_v0_15/MAINTENANCE_20260912.md) · [文档索引](INDEX.md)

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
