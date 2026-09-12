# optimization-v0.14 / OPT-30–31 执行结果

执行日期 2026-09-12。精确起点 2db6d5faae9c8568e01f5d75d145c17fc8404fb8；本地分支 optimization-v0.14-h2-time，评分前登记提交 4f855eddf66165dc617791065114fa10be1ce130。工程读取修复登记 abf94452f859ad6a134c5abc6a3abe2f5e199172，旧 U1 名称映射修复登记 4d76025cdb0f9fa6130d36430dd69c932381b4f5；原三个 v0.14 源码先按冻结摘要归档，旧 manifest 不覆盖。本报告是执行结果；冻结计划、旧 V1/R2 和 v0.13 原证据未修改。

**结论：FAIL_CLOSE_V7_RETAIN_V1。** G0 独立冷审计/来源/因果/身份通过；G1 完整质量门槛失败，关闭固定 V7。V1 活动发布保持不变，用户回传初赛 83.0319；R2 83.0207 保留回退，平台有效提交未独立核验。

相对 V1：H2 mean delta E=+0.0001752136，delta J=-0.0001585477；H2 严格改善 3/5，最近 September/October/November 改善 2/3。H2 V7 minus D1=+0.0003534315。D1 无发布资格，不因为对照成绩改变候选。

## 预算、预报与隔离

赛事新 CatBoost/其他基础回归模型 0；V7 时长 LAD 6 次尝试/6 次完成，D1 时长 LAD 6/6；铁量 LAD 0；最终 LAD 0；新 bundle/ZIP/challenger/平台上传/旧包覆盖 0。56 项新增无 CatBoost 合成测试与真实赛事预算分开；修复后完整锁定路径共 342 passed（Python 3.12.12、uv.lock）；修复前 340 passed 证据也保留。原有测试内微型模型不计入赛事 fit。

真实 H2 银行 April–October 模型预测 May–November；同日期 H1 使用 May–November 模型。每组 2165 个唯一样本。月窗口为 Asia/Shanghai 左闭右开日历月，bank 不含目标。八个月度 H1 OOF 与六个 outer 原 V1/R2/rate 全部由原模型复现。前两份 H2 是补做推理，后五份与原已存 H2 配对核验；没有改写 H1 cutoff 伪造 H2。

manifest/源码/配置/输入/旧证据身份先冻结，新 append-only 访问账本先登记；银行完成后逐 origin 从该 cutoff 的原认证历史取得合法校准标签。候选/对照同 ID、reference_time、spout、标签与 available_at；completed-month、reference<c、fold<c、label_available<=c 和最低 100 行全部核验。所有 outer V1/V7/D1 数组与摘要先落盘，随后才访问评分归档标签。滚动协议允许早期 outer 标签用于后期系数，不能描述成全局从未读取。

全部标签属于 RETROSPECTIVE_POST_HOLDOUT_CONSUMPTION；November 已消费，不是独立 holdout。未读取任何官方测试真值或测试分布来选系数/门槛。时间契约仍 ASSUMED，不新增官方时点确认。

## 六个 origin 的单标量

| Outer cutoff 月 | 合格行数 | V7 beta H2 | D1 beta H1 | 原 V1 alpha time |
| --- | ---: | ---: | ---: | ---: |
| 06 | 299 | 0.450446361735 | 0.709381755078 | 0.326735455296 |
| 07 | 591 | 0.841132696135 | 0.545084741581 | 0.335667627561 |
| 08 | 901 | 0.840950599569 | 0.507835204119 | 0.316445867170 |
| 09 | 1214 | 0.391256027937 | 0.732798553414 | 0.536127406085 |
| 10 | 1502 | 0.613139453330 | 0.591681256819 | 0.488714483446 |
| 11 | 1835 | 0.753852583897 | 0.493135833996 | 0.428274995666 |

每个系数只拟合时长，原 constrained LAD 的最小最优解/零方向规则不变；无额外时间权重、bias、网格或 shrinkage。全部 horizons 共享各自 origin 的同一系数。持久 intent 先记尝试，独立冷进程只验证已存系数最优性证书，不调用 LAD。

## 全网格分数

| Unit | V1 E | V7 E | D1 E | V7 delta E | V7 delta time WMAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEV_LONG | 0.1727030402 | 0.1742976906 | 0.1732490996 | +0.0015946504 | +0.0031893008 |
| DEV_SHORT | 0.1619176086 | 0.1621509590 | 0.1617483639 | +0.0002333503 | +0.0004667006 |
| O202406_H1 | 0.1265192822 | 0.1265316251 | 0.1266996762 | +0.0000123429 | +0.0000246858 |
| O202406_H2 | 0.1503342058 | 0.1501427750 | 0.1499706739 | -0.0001914307 | -0.0003828615 |
| O202406_H3 | 0.1926268251 | 0.1914673379 | 0.1895475084 | -0.0011594873 | -0.0023189745 |
| O202406_H4 | 0.1680071583 | 0.1678623463 | 0.1676799954 | -0.0001448120 | -0.0002896240 |
| O202407_H1 | 0.1525256076 | 0.1528735213 | 0.1525961920 | +0.0003479137 | +0.0006958275 |
| O202407_H2 | 0.1899830543 | 0.1921531251 | 0.1908196893 | +0.0021700707 | +0.0043401414 |
| O202407_H3 | 0.1684873684 | 0.1719244188 | 0.1696126451 | +0.0034370504 | +0.0068741008 |
| O202407_H4 | 0.1802105626 | 0.1808290197 | 0.1804251491 | +0.0006184570 | +0.0012369141 |
| O202408_H1 | 0.1849133816 | 0.1836458940 | 0.1843586915 | -0.0012674877 | -0.0025349753 |
| O202408_H2 | 0.1628540103 | 0.1614114821 | 0.1621582915 | -0.0014425282 | -0.0028850563 |
| O202408_H3 | 0.1787652319 | 0.1761903290 | 0.1776644705 | -0.0025749030 | -0.0051498059 |
| O202408_H4 | 0.1565277804 | 0.1545049944 | 0.1556458486 | -0.0020227860 | -0.0040455720 |
| O202409_H1 | 0.1514704074 | 0.1513991352 | 0.1516851896 | -0.0000712722 | -0.0001425444 |
| O202409_H2 | 0.1710486015 | 0.1715575882 | 0.1705318714 | +0.0005089867 | +0.0010179734 |
| O202409_H3 | 0.1514733366 | 0.1522470091 | 0.1507239180 | +0.0007736725 | +0.0015473449 |
| O202410_H1 | 0.1694303726 | 0.1696924145 | 0.1696446438 | +0.0002620420 | +0.0005240839 |
| O202410_H2 | 0.1517941671 | 0.1516251364 | 0.1516424231 | -0.0001690307 | -0.0003380614 |
| O202411_H1 | 0.1549848344 | 0.1532286532 | 0.1545467498 | -0.0017561812 | -0.0035123624 |

| Candidate | J | H1 mean E | H2 mean E | H3 mean E | H4 mean E |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 0.1657325366 | 0.1566406476 | 0.1652028078 | 0.1728381905 | 0.1682485004 |
| V7_H2_MATCHED_TIME | 0.1655739889 | 0.1562285405 | 0.1653780213 | 0.1729572737 | 0.1677321201 |
| D1_H1_SAME_CALENDAR | 0.1653543117 | 0.1565885238 | 0.1650245898 | 0.1718871355 | 0.1679169977 |

H2 为 5 个 origins，最近三个评价月对应 August/September/October 模型；November 模型只参加 H1。E 为两目标 WMAPE 等权，J 先各 horizon 内平均 origins、再四 horizon 等权；DEV 不进入 J。分目标原始分母、误差分子、铁口/月 signed mean/median 和预测差分位数保存在本地，不删高误差样本、不新增晋级切片。

## 冻结门槛

| 检查 | PASS |
| --- | --- |
| DEV_LONG_guardrail | False |
| DEV_SHORT_guardrail | True |
| H1_guardrail | True |
| H2_improved_origins | False |
| H2_mean_E | False |
| H2_same_calendar_control | False |
| H2_single_origin | False |
| H3_guardrail | True |
| H4_guardrail | True |
| J_guardrail | True |
| engineering_causal_identity | True |
| iron_exact | True |
| recent_H2_improved | True |

失败项：DEV_LONG_guardrail, H2_improved_origins, H2_mean_E, H2_same_calendar_control, H2_single_origin。阈值均来自评分前 experiment.yaml，未放宽。

## 独立冷审计与稳定性

独立进程重新加载原月度模型与历史，重放七份 H2 和同日期 H1，反序原特征/原预测一致；六个 outer 原组件重放、分块/子集/单样本一致，新序列化系数的 V7/D1 全量/反序/分块/子集/单样本逐值相等。所有评价单元铁量与 V1 exact equality，旧组件最大差 0（原协议 1e-10），新候选保存/恢复差 0；冷模型/校准 fit 尝试/完成全部 0。原 full-precision V1 E/J 与目标分母重建通过，metric tolerance 1e-12。

1000 次共享日历周 paired bootstrap，seed 2026，同样本跨 origin 共用周重数、重算分母、无效 draw 不补抽。以下是已消费回溯稳定性区间，不是独立确认、选择校正显著性或平台收益预测。

| Candidate | 有效 / 请求 | H2 delta E p025 / p975 | delta J p025 / p975 |
| --- | ---: | ---: | ---: |
| V7_H2_MATCHED_TIME | 975 / 1000 | -0.0003828581 / +0.0007503078 | -0.0005316091 / +0.0002137978 |
| D1_H1_SAME_CALENDAR | 975 / 1000 | -0.0005977982 / +0.0002708728 | -0.0006831030 / -0.0001199168 |

## 交付、正式数据与发布

产物集中 local/runs/optimization-v0.14-opt30-r1/：manifest、paired_bank、OOF_provenance、coefficients/intent/OOF/certificates、coefficient_provenance、outer_inputs、predictions_complete、metrics/summary/all_units/target_month_spout、bootstrap、acceptance、cold_validation、fit_counts、completion、原 failure、engineering_repair/ 与 engineering_scoring_repair/源码归档/修复 manifest。新的独立完成回执为 local/reports/optimization-v0.14-completion-r1.json；访问账本 local/ledgers/optimization-v0.14-calibration.jsonl。逐样本明细/模型/预测/账本/报告附件均不入 Git。

[工程修复记录](ENGINEERING_REPAIR.md) 与 [旧证据名称修复](SCORING_REPAIR.md)：序列化 JSON origins 键转字符串导致潜在 cold outer KeyError，在已完成前三份 H2 cold replay 后主动中止并保留 SIGTERM 失败。只修复 manifest 读取与无拟合恢复入口，登记原源码归档/新源码身份后独立冷重跑。后续旧 v8 U1 名称核对失败也单独保留，从已存全网格分数继续验收，不重跑评分单元。全部已保存系数、预测和分数摘要保持原值，额外赛事 fit 0；不放宽精度、不回写旧失败。

正式复赛数据仍未核验，旧 B 只 preview。historical_quality_passed 与 engineering_valid 分开；official_data_identity_verified、final_coefficient_fitted、ready_challenger、platform_verified 均 false。本阶段不拟合最终系数、不封包、不推送、不移动 cutoff、不覆盖 active_release/桌面/原 ZIP。关闭的 V2–V6 仍关闭，D1 不产生最终系数。

原 V1 ZIP SHA256 fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa，原 R2 SHA256 e42602d3045e43b4b49dd1e1c104aa8e5c1f29c639ff5f892b3b07434ed9bbdf；全部旧 artifact 身份与 v0.13 两份回执保持原哈希。

[本阶段维护观察](MAINTENANCE_20260912.md) 新增 v0.13 远端/CI 事实，不回写 v0.13 旧记录。公开历史处置与平台账号回执仍待独立处理，没有自动可见性修改、清史或上传。
