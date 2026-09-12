# v0.15 / OPT-32–33 执行结果

2026-09-12：**G0 PASS，G1 FAIL_CLOSE_V8_RETAIN_V1**。固定 `V8_QRF_TIME` 关闭；十项质量门槛全部失败，铁量与工程门槛通过。原 V1 活动发布和 R2 回退保持不变，无最终模型、challenger ZIP、上传或桌面覆盖。失败不授权叶子/树数/分位数/融合/校准/种子搜索。

起点 `f940f9f4ed09382b7948f84145ceccd3f542cdd7`；本地分支 `optimization-v0.15-qrf-time`。执行前登记 `5de6523`；P0 工程修复登记 `b515046`，保留原失败与源码，见 [修复记录](ENGINEERING_REPAIR.md)。本文件为真实执行报告，[PLAN](PLAN.md) 为执行前冻结规格。唯一运行目录 `local/runs/optimization-v0.15-opt32-r1/`，所有模型、响应、逐样本预测/误差、访问账本均不进入 Git。

## 固定分支与预算

铁量复制原 V1 完整全精度数组；时长直接取原 E09/R2 as-of 特征下的 QRF 条件较小中位数。协议 `QRF_FULLTRAIN_LEAF_v1`：256 树、每树全原始训练行各一次投叶、等树质量，bootstrap 重数只用于构树；无 E04 混合、rate/LAD 修正或后校准。原 schema/列序/dtype/每行值与旧 v0.12 审计摘要一致。独立 worker 训练中位数填充和 spout 词表，raw/transformed schema/hash 分开。

| 项目 | attempt | completed |
| --- | ---: | ---: |
| 开发 forest fit | 6 | 6 |
| 开发 preprocessor fit | 6 | 6 |
| 内部树 | 1536 | 1536 |
| 新 CatBoost/LightGBM/E04/rate/q/铁量模型 | 0 | 0 |
| LAD/偏置/后校准 | 0 | 0 |
| 最终 forest/preprocessor | 0 | 0 |
| 新候选 / 新 challenger ZIP | 1 / 0 | 1 / 0 |

June–November 原身份行数为 888、1180、1490、1803、2091、2424。每个 cutoff 的训练 reference < cutoff、标签 availability ≤ cutoff，原逐行 as-of 历史与过程逻辑不变。DEV_LONG/SHORT 分别复用 July/September 模型，不补训 April/May 或旧资产。每次 fit 前持久 intent，完成后保存并摘要校验。

P0 检索 109 份既有 registry/manifest 元数据，无 QRF 同定义匹配。六个 origin 及 V1/R2 原发布的旧推理最大差均为 0，原 E/J 和分母按 1e-12 重建通过。六个新 origin 的 V8/D2/V1 与输入/模型摘要全部落盘后才读取本轮评分归档；P0 仅提前重建既有 V1 误差。无官方 train/history 目标列直接读取或测试分布选择。

## 完整质量门槛

统一 delta=V8−V1，负值为改善。J(V1)=0.1657325366，J(V8)=0.1689400264。H2 时长 WMAPE 变化为 +0.0050628110；H2 五个 origins 全部退化，最近三个目标月份 September/October/November 0/3 改善。

| 门槛 | 注册要求 | 实际结果 | 判定 |
| --- | --- | --- | --- |
| DEV_LONG_guardrail | ≤ +0.0007 | +0.0010034018 | FAIL |
| DEV_SHORT_guardrail | ≤ +0.0007 | +0.0031102416 | FAIL |
| H1_guardrail | ≤ +0.0005 | +0.0007829000 | FAIL |
| H2_improved_origins | ≥ 4/5 | 0/5 | FAIL |
| H2_mean_E | ≤ -0.0005 | +0.0025314055 | FAIL |
| H2_single_origin | ≤ +0.0010 | +0.0052869312 | FAIL |
| H3_guardrail | ≤ +0.0005 | +0.0042868510 | FAIL |
| H4_guardrail | ≤ +0.0005 | +0.0052288029 | FAIL |
| J_guardrail | ≤ +0.0002 | +0.0032074898 | FAIL |
| engineering_causal_budget | 全部通过 | PASS | PASS |
| iron_exact | 逐样本完全一致 | PASS | PASS |
| recent_H2_improved | ≥ 2/3 | 0/3 | FAIL |

| horizon | V1 mean E | V8 mean E | delta E |
| --- | ---: | ---: | ---: |
| H1 | 0.1566406476 | 0.1574235476 | +0.0007829000 |
| H2 | 0.1652028078 | 0.1677342133 | +0.0025314055 |
| H3 | 0.1728381905 | 0.1771250415 | +0.0042868510 |
| H4 | 0.1682485004 | 0.1734773033 | +0.0052288029 |

## 18-cell 全网格

铁量全部 exact equality，因此各 cell 的 delta E 等于该 cell 时长 delta WMAPE 的一半。数值验收仍用原全精度 1e-12，不用下表打印值替代。

| cell | V1 time WMAPE | V8 time WMAPE | delta E |
| --- | ---: | ---: | ---: |
| O202406_H1 | 0.1259303578 | 0.1268674539 | +0.0004685480 |
| O202406_H2 | 0.1427738077 | 0.1438027089 | +0.0005144506 |
| O202406_H3 | 0.1838048733 | 0.1952818165 | +0.0057384716 |
| O202406_H4 | 0.1679008417 | 0.1760123718 | +0.0040557651 |
| O202407_H1 | 0.1449771933 | 0.1407104523 | -0.0021333705 |
| O202407_H2 | 0.1828349768 | 0.1842796577 | +0.0007223404 |
| O202407_H3 | 0.1689409209 | 0.1692250415 | +0.0001420603 |
| O202407_H4 | 0.1793549902 | 0.1898419280 | +0.0052434689 |
| O202408_H1 | 0.1752888710 | 0.1742398627 | -0.0005245041 |
| O202408_H2 | 0.1616407272 | 0.1640128301 | +0.0011860514 |
| O202408_H3 | 0.1765568065 | 0.1865768334 | +0.0050100134 |
| O202408_H4 | 0.1591804283 | 0.1719547775 | +0.0063871746 |
| O202409_H1 | 0.1499677464 | 0.1521278424 | +0.0010800480 |
| O202409_H2 | 0.1679754068 | 0.1778699145 | +0.0049472538 |
| O202409_H3 | 0.1513656203 | 0.1638793373 | +0.0062568585 |
| O202410_H1 | 0.1663940710 | 0.1727908785 | +0.0031984037 |
| O202410_H2 | 0.1515080382 | 0.1620819006 | +0.0052869312 |
| O202411_H1 | 0.1562662051 | 0.1614827550 | +0.0026082749 |

`D2_QRF_WEIGHTED_MEAN_TIME` 只作同分布均值诊断：delta J=+0.0033047686，H2 delta E=+0.0023569321。它不是备选可发布模型，未据其分数新增组合。

## 冷审计、测试与诊断

独立 root/worker 进程恢复六份可信摘要模型，原 V1 回读差为 0；QRF median/mean 全量、反序、分块、子集、单行及持久化复现 exact equality，最大差 0。模型、预处理、CatBoost/LAD fit 尝试数全为 0。原 bundle/history/系数/source_contract/ZIP/旧 ledger 与旧回执摘要不变。原时间契约 `competition-timestamp-contract-v1` 仍 ASSUMED，没有新官方时钟确认。

根 Python 3.12.12 / 原 uv.lock：**364 passed**，JUnit `local/reports/pytest-v015-root-locked-r4.xml`。新 worker Python 3.12.12 / 独立 uv.lock：**25 passed**，JUnit `local/reports/pytest-qrf-v015-worker-synthetic-r4.xml`；worker 该次合成 fit 单独记录为 1 forest、3 preprocessor，比赛 fit 不计入。附件19项、旧342项、两套环境测试不合并。较早合成失败和 root r3 源码变化保护失败保留于各回执/修复记录。

worker 实际版本：sklearn 1.8.0、NumPy 2.2.6、scipy 1.18.1、joblib 1.6.0、threadpoolctl 3.6.0。根 pyproject/uv.lock 不变；worker lock SHA256 `a71f304c8a03e4ddccfd47ffa4c229152745805f00467c2a25f57b2495a790cb`。

共享 calendar-week paired bootstrap：1000 次、seed=2026，有效975、无效25，不补抽。V8 的 H2 delta E 区间 [+0.0013119563, +0.0037441046]；J delta 区间 [+0.0022059168, +0.0040834863]。仅描述已消费回溯稳定性，不是独立复赛或平台确认，不作普遍算法否定。

每候选 18-cell 为5651次 exposure、1865个唯一铁次。按既定权重归一化的误差贡献重建 J/delta J、20个单元的冻结铁量与半时长增量恒等式均在 1e-12 内通过。最高1%/5%/10%唯一 ID 原始误差与 J 贡献、目标/月/铁口/horizon signed mean/median、median/mean 配对、预测差分位数、邻居有效数/最大权重/训练月质量、响应支持/边界/填充/未知类别均保存私有诊断，不删除样本或改预测。

六个新模型合计11,845,186字节。worker 实际开发查询的加载计时约0.30–0.73秒、QRF预测核约0.49–1.92秒；fit计时含预处理、构树、成员建立和落盘。进程 resource 峰值观测保存在 `post_score/model_performance.json`。这些不含原 as-of 构建/全阶段 I/O，不声称正式复赛端到端延迟。

## 不可变发布与交付

V1 ZIP SHA256 `fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa`；R2 ZIP SHA256 `e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf`。用户回传83.0319/83.0207保持原证据等级；账号有效提交回执仍未独立核验。本轮不重建或覆盖桌面副本。

核心交付：manifest、P0/registry/旧分数重建、features私有SHA交接、六份model/preprocessor与intent、models_complete、predictions_complete、metrics/summary、acceptance、bootstrap、cold_validation、fit_counts、completion。补充 `post_score/` 输出 error_budget、分组误差、median_mean_pairing、neighbor_locality、model_performance，以及数字完成时 ledger 快照和追加后前缀证明；没有新标签来源或额外 fit。完成回执 `local/reports/optimization-v0.15-completion-r1.json` 绑定核心与补充证据。

状态分别为 engineering_valid=true、historical_quality_passed=false、official_data_identity_verified=false、final_model_fitted=false、ready_challenger=false、platform_verified=false。固定设计已关闭，所以即便正式包后来兼容，也不授权该失败模型最终拟合。旧包保留其 README 原发布身份，与09-21版本关系待独立核验。维护远端事实见 [新增维护记录](MAINTENANCE_20260912.md)，旧 v14 报告未改写。本轮代码仅本地提交，无自动上传、公开推送、可见性修改或清史。
