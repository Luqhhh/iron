# optimization-v0.31 结果：两个目标隔离的 OOB 出现次数池化

执行日期：2026-09-18。分支：`optimization-v0.31-oob-occurrence-pooling`。授权本地运行：
`local/runs/optimization-v0.31-oob-occurrence-pooling-r3`，运行 manifest 绑定实现提交
`a66686b2cf441424b458764f481164dcf34e5352`。r1 在 P0 身份检查前保留，r2 在
回执视图字段修正前保留；二者均未进入评分，已按不覆盖原则保留。

## 结论

A/B 两个候选均按冻结定义完成，所有 dev 与最终预测在第一个本轮平台反馈前同时冻结。
G0 工程状态 **PASS**；新模型、树、预处理、CatBoost、铁量/时长森林、rate/q、LAD
或校准 fit 全部为 **0**。14 份已认证 v0.29 OOB 附件被复用，没有派生或保存新的
OOB 附件，没有读取 test target，没有递归回写测试预测。agent 自动上传、桌面覆盖和
公开推送均为 0。

本轮新增的是**叶响应质量分配**，不是选择叶成员：每棵树中被选叶成员的每一个出现
位置贡献一个质量单位；同一训练行跨树出现会重复计入，同一叶内不重复。若所有
`n_b` 相等，新规则逐项退化为旧 QRF 等树质量分布；本轮所有 `n_b` 并不全等，
故实际预测会发生小幅变化。OOB 不是独立验证集，`S` 不是新增样本数。

## 离线主口径

主口径为各 origin 等权的 `macro_origin_mean_wmape`；`exposure_pooled_wmape` 仅诊断。
变化均为候选减参照，负数表示改善：

| 候选 | 参照 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J ΔE | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V31I_OOB_OCCURRENCE_POOL_BLEND | V30A_OOB_BOTH_TARGETS | +0.00000149 | +0.00002955 | +0.00002086 | +0.00005612 | +0.00002701 | +0.00002372 | +0.00004773 |
| V31I_OOB_OCCURRENCE_POOL_BLEND | V29T_OOB_LEAF_QRF_TIME | +0.00021781 | -0.00001239 | -0.00027174 | -0.00019253 | -0.00006471 | -0.00043718 | +0.00048209 |
| V31I_OOB_OCCURRENCE_POOL_BLEND | V28I_PARENT | +0.00087898 | -0.00003231 | -0.00065697 | +0.00055214 | +0.00018546 | -0.00040905 | +0.00240025 |
| V31I_OOB_OCCURRENCE_POOL_BLEND | V1 | +0.00161582 | +0.00312354 | +0.00509061 | +0.00629917 | +0.00403228 | +0.00201939 | +0.00587894 |
| V31T_OOB_OCCURRENCE_POOL_TIME | V30A_OOB_BOTH_TARGETS | +0.00003565 | +0.00008169 | -0.00024978 | -0.00021126 | -0.00008593 | -0.00017209 | +0.00006632 |
| V31T_OOB_OCCURRENCE_POOL_TIME | V29T_OOB_LEAF_QRF_TIME | +0.00025197 | +0.00003974 | -0.00054238 | -0.00045992 | -0.00017765 | -0.00063299 | +0.00050068 |
| V31T_OOB_OCCURRENCE_POOL_TIME | V28I_PARENT | +0.00091314 | +0.00001983 | -0.00092761 | +0.00028476 | +0.00007253 | -0.00060486 | +0.00241884 |
| V31T_OOB_OCCURRENCE_POOL_TIME | V1 | +0.00164998 | +0.00317568 | +0.00481996 | +0.00603178 | +0.00391935 | +0.00182358 | +0.00589753 |

目标隔离恒等式由 canonical cells 与 macro 表格核对：A 相对 V30A 的
`delta_wmape_time` 全为 `0`，且 `ΔE = 0.5·ΔWMAPE_iron`；B 相对 V30A 的
`delta_wmape_iron` 全为 `0`，且 `ΔE = 0.5·ΔWMAPE_time`。40 个 cell 与 50 个
macro 摘要的最大绝对残差 `6.94e-17`，容差 `1e-12`。

两种候选都没有在当前已消费历史上超过 V1；A 相对 V30A 的 J 小幅退化
`+0.00002701`，B 相对 V30A 的 J 小幅改善 `-0.00008593`。这些量级接近噪声，
不支持把 B 的历史变化解释为新的泛化收益，也不改变 A→B 的预登记顺序。
1000 次 seed 2026 共享日历周 bootstrap 已按既有方法写入 `bootstrap.json`；
它描述已消费数据，不是独立显著性。

## OOB 池化诊断

诊断来自冻结的 worker 输出与 `pool_views/*.json`；完整逐查询数组保存在
`worker_predictions/{A,B}/{slot}.npz`，汇总见 `pool_diagnostics.json`。
`S = sum_b n_b` 为每个查询中树 × 被选成员的 occurrence 总数。

| 候选 | 开发变化行数范围 | S 范围（开发） | distinct selected rows 范围（开发） | 回退树比例均值最大 | 回退质量比例均值最大 | L1 权重变化最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 310–1123 | 1804–2342 | 385–1150 | 0.317% | 0.423% | 0.312 |
| B | 101–437 | 1831–2314 | 454–1224 | 0.421% | 0.550% | 0.320 |

最终 slot 12：A 预测相对 V30A 有 299/335 行铁量变化、0 行时长变化；
B 有 0 行铁量变化、107/335 行时长变化。A/B 的 lower/upper
boundary 命中均为 0，非有限或负数均为 0。旧等树质量切换回执在全部
7 个 cutoff、两个目标上均与认证 v0.29 端点逐字节一致；新规则的中位数使用
int64 累计 occurrence 质量首次达到 `ceil(S/2)`，偶数 `S` 返回较小中间值，
没有调用旧 `lower_median(y, weights, members)` 的 Fraction 路径来认证新规则。

## 工程、冷审计与预算

冷审计在独立进程中恢复铁量/时长源森林、源响应/预处理与旧 v0.15 gate，
重新从 bootstrap 派生 v0.29 附件并逐数组比对，核验旧树、旧 full/OOB 成员和
选择成员均未改变，只改变质量分配；全部 7 个 cutoff 的 A/B 均通过全量、反序、
分块、子集、单行一致性，CSV 与 ZIP 逐字节复核一致。所有 fit 拦截计数为 0。

| 项目 | 开发 | 最终 | 实际 |
| --- | ---: | ---: | ---: |
| 新模型/树 fit | 0 | 0 | 0 |
| 新预处理/LAD/beta/lambda/bias fit | 0 | 0 | 0 |
| A 池化读取协议清单 | 6 | 1 | 7 |
| B 池化读取协议清单 | 6 | 1 | 7 |
| 认证 OOB 附件复用 | 12 | 2 | 14 |
| 新候选 ZIP / 平台测试 | 2 / 2 | 2 / 2 | 2 / 0 |
| 自动上传/桌面覆盖/公开推送 | 0 | 0 | 0 |

测试：根锁定 Python 3.12.12 路径 **497 passed**；v0.31 worker 独立
`workers/qrf_v015/.venv` 下 **10 passed**。等树质量等价性由穷举小网格合成测试
覆盖，包含 `S` 偶数、跨树重复计数与叶内重复拒绝。

## 包身份与平台收口

两份 ZIP 只含 `result.csv`，覆盖 335 个唯一 ID，有限、非负、六位小数。它们已在
任何本轮反馈前冻结：

| 槽 | 候选 | result.csv SHA-256 | ZIP SHA-256 |
| --- | --- | --- | --- |
| A | V31I_OOB_OCCURRENCE_POOL_BLEND | `4684ba5aed30c05922da02e71d11b25909e174b8bf453000b31db74cb6f6e120` | `727a7f82132939d0785bee91e5f9e8d6b58f6c57e5cf94b32909945131e3aa73` |
| B | V31T_OOB_OCCURRENCE_POOL_TIME | `6991509caf751d5e9ec5c81b7e2da4146deea5a69083d04a6d279a5176d78e25` | `87d7671afee55a326eb4b54b518c649dc5d7a226af4ff37d45e54bf40310d5c1` |

当前父包 V30A 身份按要求核验为 result
`97b6c3e0c648a0c6454cc9b36f518625487c03d1efd7621746d049457c4aed9a`、ZIP
`fffcf23b04b3069cd71027682047eea764a2747c2dea48cca91320d67e113986`。
`platform_feedback.json` 目前为独立追加的 `AWAITING_USER_REPORTED_PLATFORM_FEEDBACK`
占位；在本轮用户反馈前不存在任何平台测试或自动上传。默认顺序 A→B，各一次；
第二项不根据第一项成绩改变。若预览包与父包在提交精度完全相同或出现真实工程
阻断，应明确报告并暂缓该无效/重复提交，不生成第三包、不按反馈追加 leaf-size
幂指数、最低回退计数、50/50 变化、测试子集或新树数。

更细的冻结源清单、池化汇总与后续平台反馈应分别读取同一运行目录下的
`source_inventory.json`、`pool_diagnostics.json` 与 `platform_feedback.json`；
后者只能由后续维护追加，不能覆盖。


## 推送与桌面交付（2026-09-18 追加）

用户在本轮冻结后显式要求提交推送并替换 C 盘桌面提交包。已执行：

- Git：`origin` 的 `optimization-v0.31-oob-occurrence-pooling` 分支首次推送，远端头
  `9b2b46fd2453e47dff2b198e55e534b905de4b9c` 与本地 HEAD 一致；仅推送代码、配置、
  文档和测试，不推送 `local/runs`、模型、预测、训练数据或提交 ZIP。
- 桌面旧包：递归检索 `C:\Users\lqh22\Desktop` 只发现 V30A/V30B 两份旧提交包；
  经 SHA-256 与 v0.30 冻结包一致后删除，对应两个旧桌面副本不再存在。
- 桌面新包：写入
  `Luqhhh_bf_tap_predict_prelim_V31I_OOB_OCCURRENCE_POOL_BLEND.zip`
  （SHA-256 `727a7f82132939d0785bee91e5f9e8d6b58f6c57e5cf94b32909945131e3aa73`）
  与
  `Luqhhh_bf_tap_predict_prelim_V31T_OOB_OCCURRENCE_POOL_TIME.zip`
  （SHA-256 `87d7671afee55a326eb4b54b518c649dc5d7a226af4ff37d45e54bf40310d5c1`）。
  两份 ZIP 均只含 `result.csv`，payload 与冻结 CSV 逐字节一致，335 行。
- 本地私有原包、旧运行目录和证据均保留；agent 平台上传仍为 0。回执：
  `local/runs/optimization-v0.31-oob-occurrence-pooling-r3/desktop_delivery_receipt.json`
  与 `publication_receipt.json`。
