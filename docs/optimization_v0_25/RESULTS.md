# optimization-v0.25 执行结果

执行日期为 2026-09-16，起点为 `d4342998176293262a030dcb2f6997cc39c871f3`，冻结实现提交为 `4fcf2b0ab88eb53a570da91c097e80f81c83bf39`。本阶段完成两个历史基准中心化候选。G0 为 **PASS**；两份 test_a 包已在任何本轮平台反馈前同时冻结。当前没有自动上传、桌面写入或公开推送。

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

两份包均为 335 个唯一 ID，ZIP 内仅含 `result.csv`。平台顺序固定 A→B，各有一个预登记名额；当前平台上传 0/2、反馈为空。未获桌面写入或平台上传授权。

## 预算、冷审计与测试

实际完成 7 个 centered CatBoost、7 个 signed-response QRF、1,792 棵树；新增 QRF preprocessor、LAD/beta/lambda/偏置均为 0。最终冷进程验证 A 全量/反序/分块/单行以及 B 全量/反序/分块/子集/单行完全一致，所有推理 fit 尝试为 0，只恢复摘要绑定的私有模型和原预处理器。

冻结实现的 Python 3.12.12 root 测试为 **441 passed**，私有产物守卫通过；worker 为 **35 passed**（原冻结 worker 34 项加 v0.25 adapter 1 项）。首次从仓库根调用 worker 测试因冻结测试要求 worker CWD 而在收集阶段失败，0 fit；失败回执已保留，随后在正确锁定目录通过，未隐藏为训练 retry。

权威运行目录为 `local/runs/optimization-v0.25-history-centered-targets-r1`。manifest SHA-256 为 `bd1b369b8d57ad301a296b7fa5a6ddd42c4b9aa12c8de2a09f9cc556e412a59c`，completion SHA-256 为 `94401c597bc10a080ab6c9c2d1ae0769debfde990609263c9298677953573176`。当前状态为 `READY_FOR_TWO_EXPLICIT_PLATFORM_SUBMISSIONS`；这是本地候选准备完成，不代表平台提分或账号有效提交。

