# optimization-v0.13 / OPT-27–29 执行结果

2026-09-12。**G0：零训练诊断与旧包 stage 工程预演通过；G1：未评估新模型质量。** 原 V1 与 R2 保留，recency60/V2–V6 保持关闭。新增赛事模型 fit、LAD/校准 fit、候选、challenger ZIP、上传和桌面覆盖全部为 0。

起点 `optimization-v0.12@62c62cc92efa660674877254c6ae8dc2bceaba83`；本地分支 optimization-v0.13，预注册实现提交 `fcbf896`。本轮仅本地提交，不推送公开仓库，不修改 visibility 或清史。正式复赛数据与平台回执未核验，分别保留缺失状态。

## OPT-27：全精度误差账本

主目录 `local/runs/optimization-v0.13-opt27-r1`。读取已消费标签前，冻结源码/配置/分箱/来源、旧完成回执、旧账本与合法过程输入；manifest SHA-256 为 `a4320bd4e6169870fbc238530b89d0676d2d204309d3b015249738411ca426f2`。新访问账本 `local/ledgers/optimization-v0.13-audit.jsonl` 追加保存，旧账本不改写。

原 18-cell 和 DEV_LONG/SHORT 的两目标分母、WMAPE、E 与 J 全精度重建通过；逐预测 J 贡献与候选配对 Delta J 恒等式通过。最大重建误差 `2.7755575615628914e-17`，低于固定 `1e-12`，没有采用打印值 `2e-10` 容差。DEV 不加入 J。

| 候选 | 重建 J | Delta J 相对 V1 |
| --- | ---: | ---: |
| R2 | 0.1669904445 | +0.0012579079 |
| V1 | 0.1657325366 | +0.0000000000 |
| V6B_RECENCY60_BOTH | 0.1653312357 | -0.0004013009 |
| V6I_RECENCY60_IRON | 0.1659601222 | +0.0002275856 |
| V6T_RECENCY60_TIME | 0.1651036501 | -0.0006288865 |

18-cell 每候选保留 **5651 次预测 exposure、1865 个唯一铁次**。同一铁次跨 origin 的不同预测全部保留，也按 sample_id 汇总唯一贡献，不能当独立重复案例。target_month_spout.csv 按目标、真实日历月、spout、horizon 展示原误差分子/分母、均值/中位数和 J 贡献；逐样本内容只存 local/。

| 原 V1 目标 | 最高 10% 唯一铁次数 | 对应预测次数 | 占该目标 J 误差 |
| --- | ---: | ---: | ---: |
| tap_iron | 187 | 715 | 32.3189% |
| tap_time_len | 187 | 725 | 33.7225% |

1%/5%/10% 的 unit 原始绝对误差排序与唯一样本累计 J 排序均保存。误差存在集中度，但不能从这些事后切片判断标签错误、删除样本或定义测试路由。H1 的 signed mean / median 正号数：铁量均 3/6，时长均 4/6；这只是偏差症状，没有拟合或应用常数校准。

逐样本 as-of 特征按原 schema 重新构建。六个 origins 的 role-exposure 中 operation/burden stale flags 均为 0；保存历史的最大年龄为 2,954.4 小时，涉及原长跨度冻结历史。缺失比例、history-age 与 stale 的固定分箱、分组误差和 provenance 已保存；旧 timestamp contract 仍为 ASSUMED。历史年龄与 horizon 相关，这些观察不足以因果区分漂移、条件偏差、报送时间不确定性或极端样本。

配对层变化（闭合的 v12 minus 原组件/原 V1；每 horizon 等权 origins）：

| 目标/horizon | Delta direct WMAPE | Delta base WMAPE | Delta corrected WMAPE |
| --- | ---: | ---: | ---: |
| tap_iron / H1 | -0.0020640196 | -0.0005636795 | -0.0009129976 |
| tap_time_len / H1 | -0.0007422423 | -0.0003360118 | -0.0004251739 |
| tap_iron / H2 | +0.0003904288 | +0.0017649179 | +0.0002700677 |
| tap_time_len / H2 | -0.0038548415 | -0.0025768101 | -0.0015791168 |
| tap_iron / H3 | +0.0040861404 | +0.0037185333 | +0.0021944408 |
| tap_time_len / H3 | -0.0052152776 | -0.0036407490 | -0.0022720394 |
| tap_iron / H4 | +0.0009053199 | +0.0019938181 | +0.0002691743 |
| tap_time_len / H4 | -0.0008328510 | -0.0001127427 | -0.0007547621 |

时长 H1 在结构修正后收益略扩大，H2/H3 则保留了较少的 base 收益。不能把它概括为结构修正始终有利或始终抵消；base、direction 和重新估计系数都在配对比较中变化，不能单独归因于 LAD。H3 铁量从 direct 到 corrected 均退化。raw/base 仍仅诊断，不作为发布候选；v0.12 FAIL 不变。

补充来源审计目录 `local/runs/optimization-v0.13-opt27-provenance-r1`：独立冻结的只读检查核验 April–November 原/新八个 OOF 的身份、可用性和训练 schema/label/weight 摘要，以及 June–November 原 V1/新 recency 两目标系数的先前可用来源和 LAD 最优性证书，0 fit。检查源已归档为该目录 provenance_check.py，原执行临时路径的同摘要副本保留；不重新估计系数。

failure_mode_evidence.md 与 supplement 区分数值事实、解释假设及缺少的证据。结论允许“暂不能区分”；没有登记下一模型干预。

## OPT-28：V1 旧包 stage 预演

目录 `local/runs/optimization-v0.13-opt28-preview-r1`。新增薄 wrapper 复用原 StructuralPredictor 和 v4 指定 stage comparator，旧 v8/v4 脚本、模型、alpha、历史与 source_contract 不修改。拒绝额外标签/外部历史、预测早于原 cutoff 和训练 ID 相交；源变化显式阻断，未授权迁移。每个 stage 在读取输入前独立冻结。

| 检查 | test_a | 旧 test_b |
| --- | --- | --- |
| 行数 | 335 | 322 |
| 实际参考时间范围 | 2024-12-01 01:44:00+08:00 至 2024-12-31 23:04:00+08:00 | 2025-01-01 01:34:00+08:00 至 2025-01-31 22:36:00+08:00 |
| 原模型 cutoff | 2024-12-01 01:44:00+08:00 | 同左，未移动 |
| 全量/反序/分块/子集/单样本 | exact equality | exact equality |
| 原 R2 stage 对照最大差 | 0 | 0 |
| 与旧 v8 冷进程输出 | 335 行 exact equality，最大差 0 | 旧 v8 硬编码 test_a，未冒称原入口支持 B |
| rate fallback | 0 | 0 |
| engineering_valid | true | true |
| official_data_identity_verified | false | false |
| quality_evaluated / platform_verified | false / false | false / false |

两个新入口均在独立进程冷加载，只从 bundle 恢复历史，模型/LAD/校准 fit 与尝试调用均 0。配置仅当前 stage 元数据及三个无标签源；全过程输入来源契约严格相同，完整模型/历史/ZIP 哈希前后核验通过。schema、缺失/stale/history age、源文件时间上界及 ID/文件摘要保存到各 stage 的 .evidence 和 .json；按每个 reference_time 截断，完整源文件上界仅作库存描述。

原 V1 ZIP 回读通过：UTF-8、只含 result.csv、三列、335 唯一 ID、有限非负、六位小数；与新 test_a 输出按原六位小数规则完全一致。原 SHA-256 `fdcbe03e8ea31577bfeed0c45cd0a0013bda42ab88d29db703fad9c2f7e557aa` 不变，result SHA-256 `cb530d3579e96efa7711b67f267bd68719b3e2fe70a31e9d500bdf27c9956a94`。原 R2 和桌面包未覆盖，没有新赛事 ZIP。

## OPT-29、维护观察与缺失项

旧 January preview 在原 December cutoff 下实际对应 H2；正式包还未开放，不宣称已覆盖正式复赛。后续 [接入清单与协议草案](SECOND_ROUND_PROTOCOL.md) 要求新 manifest 核验正式 ID、时间、字段和来源语义，再另行冻结唯一干预、主指标、双目标防退化、OOF、旧 J、预算及失败规则。本轮不设置 V6T 定制的新阈值，不 final fit/封包/晋级。

[2026-09-12 维护观察](MAINTENANCE_20260912.md) 核验 v12 远端 62c62cc 与 run 34685572032 completed/success，分别记录 commit time、仓库级 pushed_at、run 时间和查询时间；纠正 README/current_status 的过时摘要，原 v12 RESULTS 与完成回执原文不回写。当前仓库仍 public，数据历史处置未完成，额外私有 Git bundle 保存完整 28 refs；本轮未更改 visibility、force-push 或推送。

| 未完成项 | 精确状态/原因 |
| --- | --- |
| 正式复赛包身份及冷推理 | PENDING_OFFICIAL_RELEASE_20260921；尚未到正式开放时间，旧同名包只做 preview |
| 原 V1 平台有效提交回执/复赛资格 | BLOCKED_PLATFORM_RECEIPT_AUTHENTICATION_UNAVAILABLE；已自行查本地记录及公开页面，动态报名系统需账号会话，未取得 submission ID/上传时间/有效回执 |
| 正式复赛模型质量与平台成绩 | NOT_EVALUATED；未提供测试标签或独立平台回执，工程 PASS 不代替 |
| 公开数据历史处置 | PENDING_SEPARATE_REMEDIATION；不自动修改可见性或强制清史 |

用户回传 V1 83.0319、R2 83.0207 继续标记未独立核验，不宣布有效提交或复赛资格已确认。新官方通知的赛程、复赛每日 5 次取最高分、算分延迟、80 分基准线见维护观察和 [主办方通知](https://www.aicomp.cn/notice/notice-3/5248.html)。不能由此推断初赛取最高分。

## 测试、交付与不变边界

Python 3.12.12 / uv.lock：**286 passed**，新增 31 项无 fit 的合成测试。初始测试夹具未传完整冻结 constructor 参数导致的失败已留存（r1 JSON、r2/r3 JUnit），修复后 r4 合成测试和全套锁定测试通过；正式数据诊断/推理没有工程失败、额外训练或容差放宽。

```bash
uv sync --locked --extra dev --python 3.12 --offline
uv run --locked --offline --python 3.12 pytest -q --junitxml=local/reports/pytest-optimization-v0.13-locked-r1.xml
uv run --locked --offline --python 3.12 python scripts/optimization_v13_error_audit.py --output local/runs/optimization-v0.13-opt27-r1
```

上述目录是已执行记录，不能覆盖重跑。stage CLI 显式传 --stage/--bundle/--data-config/--output，实际 A/B 参数及配置分别保存在 preview 目录。原 v8 另开独立进程回归，不复写旧证据。所有产物、模型/源摘要、零 fit 预算、测试环境与验证结果汇总到 `local/reports/optimization-v0.13-completion-r1.json`；逐样本误差、feature states、预测、账本和附件只在忽略的 local/。

当前 README、EVIDENCE_STATUS.current_status、文档索引与维护入口更新。原 baseline、旧模型/配置/发布指针/失败报告/账本不变。后续不凭本诊断开启新训练；先核验正式数据与平台证据，再独立注册。
