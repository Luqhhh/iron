# optimization-v0.24 执行结果

执行日期为 2026-09-16，基点为 `a2e18e9adb30581666ac28b3ec954515ee315281`。本阶段按预注册方案完成两个目标隔离实验。G0 为 **PASS**；已消费历史上的 G1 证据显示两项均未超过 V21_REPLAY，但该证据不是本阶段两个预登记平台名额的提交门槛。两份 test_a 包已经在任何本轮平台反馈前同时冻结，尚未上传。

## 候选与离线结果

统一使用同一实际行、未舍入标签和六位序列化预测。误差变化为候选减 V21_REPLAY，正数表示退化。

| 候选 | 改动目标 | H1 ΔE | H2 ΔE | H3 ΔE | H4 ΔE | J Δ | DEV_LONG ΔE | DEV_SHORT ΔE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V24I_BURDEN_LAG_RECENCY_IRON | 铁量 | +0.00010983 | +0.00001992 | +0.00009460 | +0.00030941 | +0.00013344 | +0.00025761 | -0.00001968 |
| V24T_BURDEN_LAG_QRF_TIME | 时长 | +0.00027023 | +0.00004017 | -0.00005340 | +0.00025411 | +0.00012778 | +0.00063789 | +0.00030779 |

A 的未改时长各 horizon WMAPE 差值严格为 0；被改铁量的 H1/H2/H3/H4 WMAPE 差值为 +0.00021965、+0.00003984、+0.00018921、+0.00061882。B 的未改铁量严格为 0；被改时长差值为 +0.00054045、+0.00008034、-0.00010681、+0.00050823。

两项历史 J 也仍明显差于 V1：A/B 的 ΔJ 分别为 +0.00339746 / +0.00339179。共享日历周 bootstrap 相对 V21 的 J ΔE 95% 描述区间分别为 A `[-0.00014373, +0.00039439]`、B `[-0.00006729, +0.00029254]`；这是已消费回溯稳定性描述，不是独立确认或显著性选择。

## 最终包

两份包各有 335 个唯一 ID，ZIP 中仅含 `result.csv`：

| 顺序 | 候选 | result.csv SHA-256 | ZIP SHA-256 | 相对 V21 改变 |
| --- | --- | --- | --- | --- |
| A | V24I_BURDEN_LAG_RECENCY_IRON | `dd8b1a4d40648137f8f1b9f10636eb9042086587d56e28f0ad900847c038cb9e` | `c40592e9b1891098f4224d98c16de35034ac246d736df4e2e23f5fa751144cee` | 铁量 335 行；时长逐字符串完全不变 |
| B | V24T_BURDEN_LAG_QRF_TIME | `97d64f52c4dc0bfe9625513f447f4773daa8e52233f6790dda3e52c802f9cc3f` | `5b76ddbffe51bc72c59b09395822a57236bb3a7029401b40413830c73b4b6180` | 时长 212 行；铁量逐字符串完全不变 |

私有路径分别为：

- `local/runs/optimization-v0.24-dual-target-burden-lag-r1/submissions/V24I_BURDEN_LAG_RECENCY_IRON/Luqhhh_bf_tap_predict_prelim.zip`
- `local/runs/optimization-v0.24-dual-target-burden-lag-r1/submissions/V24T_BURDEN_LAG_QRF_TIME/Luqhhh_bf_tap_predict_prelim.zip`

平台顺序固定为 A 后 B，每项最多一次；当前上传数为 0、剩余新候选预算为 2。未写桌面，也未自动恢复、覆盖或上传 V21。账号当前有效提交仍未知。

## 预算、冷审计与修复

实际完成 7 个 A CatBoost、7 个 B QRF forest、7 个 B preprocessor，共 1,792 棵树；LAD/beta/lambda/偏置拟合为 0。P0 证明所有原 210 列值、dtype 和顺序保持不变，QRF 数值矩阵由 209 列增至 239 列。最终冷进程的全量、反序、分块、子集和单行结果完全一致，所有 fit 尝试为 0；只加载摘要绑定的私有模型。

运行中保留了两项工程失败：首次评分把 DEV 同时作为 CELL 和 DEVELOPMENT 建立非唯一索引；首次最终预测把扩展矩阵全部传给冻结旧 schema 的 recency 模型。两个修复都在原失败与产物摘要落盘后执行；前者只重算汇总，后者只显式选择旧模型登记列并复用已完成最终模型。两项修复的新增 fit 均为 0，候选定义和预测算法没有改变。

权威运行目录为 `local/runs/optimization-v0.24-dual-target-burden-lag-r1`。manifest SHA-256 为 `7c0ca6b770750e57ff6973ee1ca8ce94e2b2c7ed1f34995e8fdb641132e4c545`，completion SHA-256 为 `3be9d39b0477632e822c1f7ee195db2d692579981d4007771b940bcadd6943ff`。当前最高用户回传仍是 V21 的 83.2375；本阶段尚无平台结果，不能写成已提分。

最终锁定 Python 3.12.12 根测试为 **433 passed**；独立 QRF worker 为 **35 passed**（原冻结 worker 34 项加 v0.24 adapter 1 项），两者未混计。私有产物守卫通过；分支尚未推送，因此远端 CI 为 `NOT_RUN`，不能用本地结果代替远端状态。
