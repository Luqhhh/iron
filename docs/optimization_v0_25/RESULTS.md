# optimization-v0.25 执行结果

执行日期为 2026-09-16，起点为 `d4342998176293262a030dcb2f6997cc39c871f3`，冻结实现提交为 `4fcf2b0ab88eb53a570da91c097e80f81c83bf39`。本阶段完成两个历史基准中心化候选。G0 为 **PASS**；两份 test_a 包已在任何本轮平台反馈前同时冻结。用户随后回传 A/I 为 **83.1516**、B/T 为 **83.0117**，两项均低于 V21 的 83.2375，固定候选关闭并保留 V21。运行本身没有自动上传；桌面替换与 Git 推送来自此前单独明确授权。

## P0 与训练身份

七个 cutoff 均恢复原 210 列 E09/R2 输入，未包含 v0.24 的 30 列变料特征。历史基准直接读取原 as-of 特征中的 last100 中位数与 count，选择次序为同铁口、全炉、零回退；基准不进入模型 X。每个训练 cutoff 只有首行使用零回退、第二行使用全炉回退，其余均使用同铁口基准；两个目标都保留大量正负偏差。

原 v0.15 QRF 预处理器按 cutoff 摘要恢复，新增 preprocessor fit 为 0。旧 V21 包、原 QRF gate、查询日 60 日中位数、V6I beta、E04、rate 与原 R2 跨目标输入均未重训或改写。

## 统一离线结果

误差变化为候选减 V21_REPLAY，负数表示改善。预测统一六位序列化，标签不舍入。

| 候选 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J Δ | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V25I_HISTORY_CENTERED_RECENCY_IRON | +0.00023041 | +0.00036825 | +0.00074596 | +0.00023048 | +0.00039378 | +0.00084274 | -0.00046073 |
| V25T_HISTORY_CENTERED_QRF_TIME | -0.00103613 | -0.00309714 | -0.00361144 | -0.00112251 | -0.00221680 | -0.00063292 | -0.00236964 |

A 只改变铁量；其 H1/H2/H3/H4 铁量 WMAPE 相对 V21 的变化为 +0.00046666、+0.00068485、+0.00142269、+0.00048443，未改时长差值严格为 0。B 只改变时长；对应四个 horizon 的时长 WMAPE 变化为 -0.00212737、-0.00627728、-0.00734820、-0.00198287，未改铁量严格为 0。

B 的 J 为 0.16677975，优于 V21_REPLAY 的 0.16899655，但仍高于 V1 的 0.16573254，ΔvsV1 为 +0.00104721。A 的 J 为 0.16939033，ΔvsV1 为 +0.00365779。共享 calendar-week bootstrap 相对 V21 的 J ΔE 95% 描述区间为 A `[+0.00012386,+0.00069334]`、B `[-0.00303783,-0.00137179]`；有效 draw 975/1000，无效 draw 未补抽。这仍是已消费回溯，不是独立确认。

仅诊断的 baseline-only J 为：铁量 0.16927451、时长 0.16580134。它们没有新 fit、不是第三候选，也未封包。

## 最终包

| 顺序 | 候选 | result.csv SHA-256 | ZIP SHA-256 | 相对 V21 改变 |
| --- | --- | --- | --- | --- |
| A | V25I_HISTORY_CENTERED_RECENCY_IRON | `5cabe3d0e77b565f52b562fdb706b0a00e52f4245f0169bd1a93af92fa13c03a` | `fbb2bed6a2f6653db22f1cfb37a33e6fec27d57488c01de28f2b230231e14423` | 铁量 335 行；时长逐字符串不变 |
| B | V25T_HISTORY_CENTERED_QRF_TIME | `436ff8d9109089a2d112794cbcc5db1741bd82aa77b01011077269499322cfab` | `eef2f7d1fe7faca533e04a7103a1ba0aa28502e53f1bf4dc3e734d4d4a8049d1` | 时长 307 行；铁量逐字符串不变 |

私有包路径为：

- `local/runs/optimization-v0.25-history-centered-targets-r1/submissions/V25I_HISTORY_CENTERED_RECENCY_IRON/Luqhhh_bf_tap_predict_prelim.zip`
- `local/runs/optimization-v0.25-history-centered-targets-r1/submissions/V25T_HISTORY_CENTERED_QRF_TIME/Luqhhh_bf_tap_predict_prelim.zip`

两份包均为 335 个唯一 ID，ZIP 内仅含 `result.csv`。平台顺序固定 A→B，各有一个预登记名额。两项用户回传已经按该冻结顺序登记，预算 2/2 已消费；没有 submission ID 或账号回执，因此仍是未独立核验的用户回传。原运行完成时尚未获得桌面写入或平台上传授权；后续桌面授权单独登记如下。

用户后续明确授权删除桌面旧提交包并替换。两份 v0.24 桌面副本已删除，其 private local 原包仍保留、可恢复；桌面当前只保留：

- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V25I_HISTORY_CENTERED_RECENCY_IRON.zip`
- `C:\Users\lqh22\Desktop\Luqhhh_bf_tap_predict_prelim_V25T_HISTORY_CENTERED_QRF_TIME.zip`

桌面摘要与上表冻结 ZIP 完全一致。该操作是文件交付，不是赛事平台上传；平台两次测试均由用户侧完成，agent 上传数为 0。

用户同时明确授权 Git 提交与推送。首个公开结果提交 `22545c0` 已推送至 `origin/optimization-v0.25-history-centered-targets`；后续发布核验提交 `0b21de4` 绑定的 locked-tests run 35110398739 为 `completed/success`。private local 模型、响应、预测、账本、反馈槽和 ZIP 均未进入 Git。

## 平台反馈与收口

分数均为用户回传，未取得平台账号原始回执。首个未附候选字母的 `83.1516` 按预先冻结的 A→B 顺序映射为 A/I；用户明确标注 `T:83.0117` 的分数映射为 B/T。

| 顺序 | 候选 | 用户回传 | Δ vs V21 | Δ vs V10 | 决策 |
| --- | --- | ---: | ---: | ---: | --- |
| A/I | V25I_HISTORY_CENTERED_RECENCY_IRON | 83.1516 | -0.0859 | -0.0435 | 关闭，保留 V21 |
| B/T | V25T_HISTORY_CENTERED_QRF_TIME | 83.0117 | -0.2258 | -0.1834 | 关闭，保留 V21 |

A 是本轮较高分，但仍低于 V21。按两个目标隔离和四位显示分数进行的纯算术组合为 `83.1516 + 83.0117 - 83.2375 = 82.9258`；这不是平台实测，也未生成或提交组合包。结果与离线排序不一致：B 的回溯指标较好，但平台回传更低；两种证据并列保留，不回写既有 G0/G1，也不据此追加基准、窗口、门控或参数搜索。

本轮平台预算按用户回传计为 2/2，剩余 0；agent 自动上传为 0。账号当前有效提交仍未知。如果初赛实际采用最后一次提交，最后的 B 可能是当前有效条目，但本报告不在无账号回执时作此断言。

## 预算、冷审计与测试

实际完成 7 个 centered CatBoost、7 个 signed-response QRF、1,792 棵树；新增 QRF preprocessor、LAD/beta/lambda/偏置均为 0。最终冷进程验证 A 全量/反序/分块/单行以及 B 全量/反序/分块/子集/单行完全一致，所有推理 fit 尝试为 0，只恢复摘要绑定的私有模型和原预处理器。

冻结实现的 Python 3.12.12 root 测试为 **441 passed**，私有产物守卫通过；worker 为 **35 passed**（原冻结 worker 34 项加 v0.25 adapter 1 项）。首次从仓库根调用 worker 测试因冻结测试要求 worker CWD 而在收集阶段失败，0 fit；失败回执已保留，随后在正确锁定目录通过，未隐藏为训练 retry。

权威运行目录为 `local/runs/optimization-v0.25-history-centered-targets-r1`。manifest SHA-256 为 `bd1b369b8d57ad301a296b7fa5a6ddd42c4b9aa12c8de2a09f9cc556e412a59c`，completion SHA-256 为 `94401c597bc10a080ab6c9c2d1ae0769debfde990609263c9298677953573176`。该 completion 是平台反馈前冻结的运行完成证据，保持原摘要不改写。当前阶段状态为 `COMPLETE_PLATFORM_FEEDBACK_BOTH_CLOSED_RETAIN_V21`。
