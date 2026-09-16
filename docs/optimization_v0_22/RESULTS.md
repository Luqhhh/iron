# optimization-v0.22 / V22_CAUSAL_H2_QRF_SHRINK

状态日期：2026-09-16

审阅基点：`optimization-v0.16-data-adaptive@9c6cf42d16e369d8ce632806670d1bb170482b91`

分支：`optimization-v0.22-causal-h2-qrf-shrink`
开发运行：`local/runs/optimization-v0.22-causal-h2-r8`

## 当前结论

- G0：**PASS**
- G1：**PASS**
- 状态：`DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY`
- 原开发阶段平台候选、上传、桌面写入、公开推送：`0 / 0 / 0 / 0`
- 后续用户授权 test_a 最终化：平台候选/ZIP/桌面写入/上传为 `1 / 1 / 1 / 0`
- 正式发布指针未改变；V21 仍是最高用户回传初赛包。

所有权威比较均使用六位小数序列化后的预测。V22 相对同精度 V10：H2 平均
ΔE **-0.0021665373**，5/5 origins 严格改善，最近三个月 3/3 改善，最差单个
H2 ΔE 仍为 **-0.0002691095**。V22 相对 V21_REPLAY 的 H2 平均 ΔE 为
**-0.0018899938**；18-cell J Δ 为 **-0.0011855502**。全部预注册门槛通过。

| 指标 | ΔE（V22−V10） |
| --- | ---: |
| H1 平均 | -0.0012258777 |
| H2 平均 | -0.0021665373 |
| H3 平均 | -0.0014566282 |
| H4 平均 | +0.0001068422 |
| DEV_LONG | -0.0011351721 |
| DEV_SHORT | -0.0007989413 |
| J | -0.0011855502 |

共享 calendar-week paired bootstrap 使用固定 1000 次、seed 2026；975 次分母有效，
无效 draw 未补抽。V22−V10 的 H2 ΔE 95% 描述区间为
[-0.00289238, -0.00148261]，J 为 [-0.00191339, -0.00045081]。这是已消费回溯
稳定性描述，不称独立显著性确认。

## P0 与 V21 waiver

P0 核验了 June–November 六个历史 QRF、最终 2,754 行 QRF、V6I 铁量模型/
系数、V10 原 ZIP 和其 `result.csv` 字节。V21 原 ZIP 摘要
`1a1d34ba96501da1391d2f97b237630f25661b718439efd070c7e52186d589a6`
仍未在当前执行端找到。

用户随后明确授权继续。本次只豁免“必须恢复 V21 原 ZIP 字节”这一项：P0 明确记录
`v21_original_zip_verified=false` 和 `EXPLICIT_USER_WAIVER`，没有把原包写成已核验。
原 V21 的 V10 六位输入、`500/0.25/spout=1`、查询日移动 60 日窗口和六位输出规则
仍完成回放；test_a 回放修改 32/335 行、无全历史回退。历史 V21 用户回传 83.2375
的证据等级不变。

## 实际训练与参数

- 新 QRF：April/May 各 1 forest + 1 preprocessor，共 512 棵树；原 June–November
  六个森林直接复用。
- 七份真正 H2 bank：April→May 至 October→November；bank 不含目标。
- 中位数：8 cutoff × 2 spout = 16 组，空窗回退 0 组。
- lambda：12 个 origin×spout 槽全部满足 100 行并执行既有 constrained LAD；不足
  样本回退 0。June/spout-2 的最优 lambda 为 0，但它是合法拟合结果，不是回退。
- 铁量模型、铁量 LAD、rate/q/E04 新 fit：0；V22 铁量与 V10 逐样本完全一致。

| outer | spout 1 | spout 2 | 合格行数（1/2） |
| --- | ---: | ---: | ---: |
| June | 1.000000 | 0.000000 | 149 / 150 |
| July | 0.977455 | 1.000000 | 295 / 296 |
| August | 1.000000 | 1.000000 | 444 / 457 |
| September | 1.000000 | 1.000000 | 605 / 609 |
| October | 1.000000 | 1.000000 | 744 / 758 |
| November | 1.000000 | 1.000000 | 914 / 921 |

## 冷审计与失败保留

冷审计 PASS：13 组 worker 推理、16 个中位数证书、12 个 lambda 最优性证书均核验；
全量/反序/分块/子集/单行一致，root 和 worker 的模型/校准 fit 尝试均为 0。

原始失败均保留在 r8：首次运行因 lambda glob 同时计数结果与 `.intent.json` 而提前
触发预算上限；第一次冷恢复的旧式失败身份缺 `bytes`；第二次冷恢复错误调用了被
零拟合守卫禁止的 LAD 求解；第三次发现 CSV 将整数型时长恢复为 int64，造成证书
标签摘要表示差异。修复均登记为零模型重训、零预测重算的工程恢复；最终通过时直接
核验已冻结 lambda、预测和独立 cold-worker 输出，没有改变算法、参数或预测值。

## 测试与证据

最新锁定 Python 3.12.12 根环境 **405 passed**；worker 独立锁环境 **34 passed**。
平台包生成后的 JUnit：`local/reports/pytest-optimization-v0.22-release-root-locked-r6.xml`
（SHA-256 `3e1657b8…a9d0`）和
`local/reports/pytest-optimization-v0.22-release-worker-locked-r6.xml`
（`329f781f…fc82`）。较早 P0 阻断、命令工作目录错误及 r8 工程/冷审计失败证据均
原样保留。

核心完成证据：`acceptance.json` SHA-256 `2f3438f4…c790e`；`completion.json`
`7745c02d…a9e8e`；`cold_validation.json` `56105687…dbaf3`。

## 用户授权的 test_a 平台包

开发验收后，用户明确要求把平台包生成到 C 盘桌面。独立运行
`local/runs/optimization-v0.22-test-a-platform-r1` 复用原最终 2,754 行 QRF、V10
铁量和七份 H2 bank；新增森林、预处理器、铁量模型及铁量 LAD 均为 0。只按冻结
规格拟合最终两个 lambda（spout 1/2 均为 1.0）并计算两个 cutoff 中位数
（116.0/113.0），未读取 test_a 目标或使用 test_a 分数拟合参数。

335 行 `result.csv` SHA-256 为
`be599de21a4b6063459e227cc939543b7403f2830d25dd3f5fab6f061b8fbb47`；ZIP 及桌面副本
SHA-256 均为 `7752863b3d88b0df071496c547c03d2a7f9a088685557dbc31531040a00fecee`。
独立冷进程重新加载最终模型、M 和 lambda，验证两个 M 证书、两个 lambda 证书、
QRF 全量/反序/分块/子集/单行一致性、结果字节及 ZIP payload，fit 尝试为 0。
封包状态为 `PASS_READY_FOR_USER_PLATFORM_UPLOAD`；代理上传为 0，封包完成时尚无
成绩回传。正式 V1 发布指针和最高已知用户回传 V21 均不改写。该包是现有 test_a 身份，不是
尚未开放并核验身份的正式复赛包。

## test_a 用户回传

用户随后回传该唯一新交付 ZIP 的平台显示分数 **83.1166**。证据等级为
`USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED`，未取得 submission ID、时间或账号回执。
相对 V21/V10/V11/V8/V1/R2 的显示分数差分别为
**-0.1209 / -0.0785 / -0.0640 / -0.0470 / +0.0847 / +0.0959**。

因此关闭 V22 的 test_a 候选，保留 V21 为最高用户回传、V10 为本阶段参照，正式
V1 发布指针不变。开发 G0/G1 与已冻结离线指标仍按原协议保留，不能由单次平台回传
改写；同样不根据该结果追加阈值、窗长、铁口子集或逐行修改。反馈记录：
`local/runs/optimization-v0.22-test-a-platform-r1/platform_feedback.json`，SHA-256
`824727d3b2eadc694d9bd9a007cd0cd055ca2115813c702f404a78f4ecd4cac7`。

## 状态口径

- `highest_user_reported_test_a`：V21 / 83.2375；用户回传、未独立平台核验。
- `historical_development_reference`：V1。
- `phase_v22_reference`：V10；另报 V21_REPLAY。
- V21 选择使用过平台反馈；V22 设计受既有平台观察启发，但参数拟合未使用 test
  分数、误差切片或 test target。
- November 全局已消费，本结果是已消费回溯开发，不称独立确认。
- v0.16 是 `V9_STATE_ANALOG_STRUCTURAL`，未与未实施的变料事件分段方案混用。
