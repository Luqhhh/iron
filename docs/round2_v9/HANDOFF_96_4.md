# 96.4 搜索交接（2026-09-26）

当前平台最好成绩仍为用户报告的 **A35 = 96.3366**，距目标 96.4 为
0.0634。V7–V9 的预注册离线任务全部结束；没有新平台成绩，不能宣布目标达成。
用户再次明确：**继续禁用外部预训练权重**。全部新模型只用复赛训练数据从零训练。

## G1：两条通过四切分验证的新方向

以下增益单位均为相对 A35 的完整提交评分点；每个切分包含全部五折。
权重在其他切分上选择，没有跨切分平均 OOF 预测。

| 优先级 | 冻结方向 | 四切分平均增益 | 配对 LCB95 | 正切分 | 正折（描述性） | 开发集包分 |
|---|---|---:|---:|---:|---:|---:|
| 1 | V7 时长，TabM PLR 0.01 | +0.015270 | +0.007168 | 4/4 | 14/20 | 96.229585 |
| 2 | V9 铁量，RealMLP TD-S | +0.004389 | +0.002721 | 4/4 | 15/20 | 96.214003 |

二者通过“至少四切分、每个切分为正、种子级配对置信下界为正”规则；
但开发集包分均低于冻结的 **96.25** 门槛，状态仅为 `candidate_pool`。
7777/12011 未参与本轮配方选择，但曾用于早期工作；这不是四份独立采集的数据。
本地增益不能乘固定系数外推平台，也不能保证 96.4。

对应证据：[V7](../round2_v7/RESULTS.md)、[V9](RESULTS.md)。
V7b 的旧 N-0048 全训练分区覆盖控制未通过双切分同向门槛；V8 特征注意力铁量
四切分只有 3/4 为正，LCB95 为 -0.000234，因此不进入本次建议。
失败记录保留在 [V7b](../round2_v7/COVERAGE_RESULTS.md) 和
[V8](../round2_v8/RESULTS.md)。

后续 [V10 损失对照](../round2_v10/RESULTS.md) 也已完成：Huber 在 V7 之上的
开发增益为+0.003069，但两个确认切分均为负；四切分均值+0.000551，
LCB95 为-0.001496，仅2/4 为正，故不晋级。没有转而追加 MAE 确认。
V7–V10 已注册的离线任务现已全部结束，上述两条建议的顺序不变。

## 独立发布方案：V7 已授权交付，V9 尚未授权

基准包是已提交的 A35，ZIP SHA-256：
`b4e1fc2d1287a213e9d88d1c30a7420ecc222235fcc6b7191e341dfd16924b06`。

1. 优先 V7 时长：新时长列为 `0.50*A35_time + 0.50*V7_tabm_plr001`；
   铁量保留 A35 原字符串。只新增该配方的全量训练模型。
2. 可选 V9 铁量：新铁量列为 `0.90*A35_iron + 0.10*V9_realmlp_td_s`；
   时长保留 A35 原字符串。只新增该配方的全量训练模型。

这两个方案各自相对同一个 A35 父包，不构成双目标组合。独立发布均需
冻结发布清单，完成折内重现、全量训练、冷进程推理、322 行顺序与唯一性、
未替换列字符串逐项一致、替换列混合算术和最终 ZIP 哈希检查。
任一步失败均保留证据并停止该包交付。用户自行上传并反馈成绩。

用户随后明确要求将 V7 包写入桌面 `submission`，已授权该候选的96.25 门槛例外、
全量训练和独立交付。V7 已完成，详见 [交付记录](../round2_v7/DELIVERY.md)。
这是单候选例外，未全局降低 [AGENTS.md](../../AGENTS.md) 的冻结阈值。
原 [V7 SPEC](../../configs/round2_v7/SPEC.yaml) 的搜索预算保持历史原样，新增发布使用
独立 `RELEASE_R3.yaml`。V9 仍只有候选池资格，尚未授权全量训练或打包。

## G0 与现有交付保护

V7 本次交付前，锁定 Python 3.12 路径的完整测试 **1011 passed，23 warnings**；
其中 20 条是新增合成测试触发的 Lightning 弃用警告。
候选与基准缓存身份、完整覆盖及独立评分复算均已通过检查。
模型、预测、审计和原始日志留在忽略的 `local/` 下。

追加的跨轮检查确认：V9 所复用的 V8 重建基准，与 V7 使用的 V5 缓存，
在7777/12011 两个切分共10 折上的 V36 时长列全部逐项一致（最大差值0），
留出行位置也完全相同。该检查不拟合模型、不评估双目标组合；私有记录为
`local/reports/v7-v9-reference-consistency-r1.json`。

桌面 `round2-V6-alpha-line-search-20260926` 的 A100/A85/A72/A60/A45 五个 ZIP
均已重新核对哈希，与原交付一致。A35 父包也通过哈希和 322 行结构复核。
截至初次交接，V7–V10 新全量模型0、新提交包0、桌面写入0。后续此次授权新增
V7 全量模型1、独立包1，并已写入 `submission/V7_TIME_PLR001_A50`；代理上传仍为0。
本次交付不替换原五包或其既定交接。

## Subsequent V11 result (2026-09-26)

After the user-authorized [V7 delivery](../round2_v7/DELIVERY.md), V11 completed80 quantile-representation development fits, one exact standard-coordinate control, and10 confirmation fits. Only periodic-uniform time qualified for confirmation; its four-seed increment over V7 failed (mean+0.000415, LCB95-0.003189,3/4 positive seeds). No V11 package was generated, no other finalist was substituted, and V7/V9 historical decisions are unchanged. See [V11 results](../round2_v11/RESULTS.md). The current platform best remains user-reported A35=96.3366;96.4 is still unverified.

## V12: subsequent qualified iron direction

[V12 joint periodic iron](../round2_v12/RESULTS.md) passed four-seed confirmation
against both A35 (mean+0.016088, LCB95+0.010939,4/4 positive) and the V9 iron
candidate (mean+0.011627, LCB95+0.008416,4/4 positive). It is the stronger measured
iron direction now available in the local candidate pool. Its frozen development
score96.226441 is still below96.25; no full-data model or submission package is
authorized or generated. This is not a platform rank or a joint-column release.
V7 remains the already delivered time direction with platform feedback pending.
V9 and all earlier decisions remain historical evidence. V13's separately frozen
PLE repair experiment retains its registered V7/V9 comparisons and zero-release
budget. Current reported platform best remains A35=96.3366;96.4 is unverified.

A concrete [V12 isolated iron release proposal](../round2_v12/RELEASE_PROPOSAL.md)
and tested adapter are prepared. They use the development-selected0.50 weight
against A35 and preserve A35 time strings. Authorization remains pending; no
full-data release fit, package or desktop write is implied by readiness.

The user subsequently authorized that V12 proposal, and the isolated iron ZIP
was delivered on2026-09-27. [V12 delivery](../round2_v12/DELIVERY.md) records the
exact SHA-256 and independent verification. V7 and V12 are delivered with no
reported platform scores; V9 remains an undelivered historical candidate-pool
direction. This explicit V12 exception does not change the global96.25 gate.

## Subsequent V13 closure (2026-09-27)

[V13 results](../round2_v13/RESULTS.md) record all 40 completed development fits and
a passed independent audit. The PLE repair activates learning and improves standalone
models, but no candidate adds a positive increment over the frozen V7/V9 references.
Both iron recipes also select zero weight on top of delivered V12 iron. No
confirmation fits or V13 packages were generated. V7 and V12 platform feedback
remains pending; A35=96.3366 remains the reported best.

## Subsequent V14 closure (2026-09-27)

[V14 results](../round2_v14/RESULTS.md) record 20 completed joint PLE fits and a
passed independent audit. Joint training improves PLE iron standalone quality,
but every recipe/target selects zero weight over V12 iron or V7 time at both
development splits. No confirmation candidate, full-data fit or package resulted.
Delivered V7/V12 feedback is pending; platform >96.4 remains unverified.

The separately recorded alpha feedback updates the current user-reported platform
best to A60=96.3465 (A45=96.3438, A72=96.3425). Historical A35 references above
remain frozen experiment comparisons, not the current platform incumbent. See
[alpha feedback](../round2_v6/RESULTS.md#13-α-线搜索反馈a60-成为当前最佳2026-09-27).
V12 then V7 remain the delivery priorities; their original packages are unchanged.

## Subsequent V15 confirmation (2026-09-27)

[V15 expert-routing results](../round2_v15/RESULTS.md) include a complete audited
20-fit development pool and 10-fit confirmation. The sole finalist, gated time,
fails the four-seed rule: relative to V7, mean +0.000727, LCB95 −0.001194,
3/4 positive seeds; relative to A60, mean +0.021567 and positive LCB95, but
still only 3/4 positive. No fallback or new package was generated. A60=96.3465
remains the reported platform best; V12 then V7 remain pending feedback.
