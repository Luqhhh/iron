# 项目实施报告

v0.15 / OPT-32–33 已完成固定 QRF 时长分支：G0通过，G1 FAIL_CLOSE_V8_RETAIN_V1。原 V1/R2不变，平台83.0319/83.0207仅用户回传。

- G0：原六个训练矩阵与V1/R2发布回读、旧E/J/分母重建、独立QRF冷进程和反序/分块/子集/单行exact equality通过；推理fit尝试0，铁量exact。
- G1：十项质量门槛失败；H2 delta E +0.0025314055、0/5改善，delta J +0.0032074898。固定QRF关闭，D2加权均值只诊断。
- 预算：6 forest + 6 preprocessor、1536内部树；新旧CatBoost/E04/rate/q/LAD/铁量模型fit均0，final/ZIP/上传/覆盖0。
- 测试：根Python3.12.12原锁364 passed，独立worker锁25 passed，计数和合成fit分开记录；原P0与较早测试失败保留。
- 来源：已消费回溯5651 exposures/1865唯一ID，原时间契约ASSUMED；正式新版身份与平台有效回执未核验。
- 维护：已核实v14远端f940f9f及34690189315成功CI；v15本地提交，无本阶段远端CI，不改写旧报告或历史。

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
