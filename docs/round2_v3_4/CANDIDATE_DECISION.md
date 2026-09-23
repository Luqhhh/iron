# Round2 V3.4 最终候选冻结与淘汰记录

日期：2026-09-23  
分支：`round2-v3.4-ebm-and-constrained-composition`

## 1. 冻结对象

本轮只冻结一个进入“候选讨论”状态的流程：

`V34_A_MECHANICAL_CONSTRAINED_16061`

开发 seeds 42/3407 学到的全精度部署权重：

| 目标 | 成员 | 权重 |
| --- | --- | ---: |
| tap_iron | L1 | 0.5000000000000001 |
| tap_iron | `v34-s1-ebm_boundary-0016` | 0.3397038169347897 |
| tap_iron | `v34-s1-ebm_boundary-0020` | 0.16029618306521032 |
| tap_time_len | L1 | 0.6192377408552066 |
| tap_time_len | `v34-s1-ebm_boundary-0089` | 0.28691289439355067 |
| tap_time_len | `v34-s1-global_spout_shrink-0136` | 0.09384936475124295 |

最终外层结果（seed 16061，inner seed 7771）：

- 本地完整包：`96.19973698527157`
- 同协议 L1：`96.16031215578445`
- `delta_L1`：`+0.03942482948711756`
- 五折包差：`+0.044103209, +0.064125554, +0.017002393, +0.007830375, +0.064093326`
- 目标 WMAPE：iron `0.037855457`，time `0.038149803`

状态：**FROZEN_FOR_DISCUSSION**。满足 `≥0.02` 候选讨论阈值，但非发布授权，也未证明平台增益。

## 2. 完整多样性精筛

`docs/round2_v3_4/RESULTS.md` 记录的 diversity selection 已完成：

- 每目标最多 10：5 个单模型强者、3 个 simple-mix 实际有效者、2 个结构不同候选；
- 结构键覆盖 EBM boundary、EBM residual 坐标/修正器、global+spout shrink 父配方/正则；
- 去重后铁量 10、时长 10。

多样性组合开发结果：

- 完整包 `delta_L1 = +0.04384483535657159`；
- seed 42 delta：`+0.048955302736645306`；
- seed 3407 delta：`+0.03873436797649753`。

该流程登记为：

`V34_B_DIVERSITY_CONSTRAINED_DEV_ONLY`

状态：**DEVELOPMENT_ALTERNATIVE_NOT_PROMOTED**。开发上略优于 A，但本轮只能使用一组新最终外层切分，16061 已用于 A；没有再使用 18041 开启第二组最终外层，也没有在已知 16061 结果后反向挑选 B。因此 B 不晋级为最终候选，只保留为下轮可预登记的替代流程。

## 3. 条件追加 16 项

触发规则已由开发 P0 包差满足。实际执行：

- 中心：`v34-s1-ebm_boundary-0020` 和 `v34-s1-global_spout_shrink-0136`；
- 追加槽位：16；
- 细筛：seed 42 folds [0,1]；
- 每目标精筛前 2 进入完整五折 seeds 42/3407。

精筛结果：

- 铁量最佳扩展 `v34-ext-00` 两 seed 均值 WMAPE `0.038058074743`，差于其中心 `0020` 的 `0.037993809`；
- 时长最佳扩展 `v34-ext-09` 两 seed 均值 WMAPE `0.039088149178`，相对其中心 `0136` 的 `0.039099198` 仅有 `1.1e-5` 量级改善；
- 把扩展项加入受约束组合后，组合器没有选中任何扩展项，目标成员和权重不变。

状态：**ELIMINATED_NO_MATERIAL_GAIN**。没有为了用满名额替扩展项强行入模。

## 4. 01_v31_s3_s1_full 冷审计

包目录：

`local/runs/round2-v3.1-directed-search/prepared-packages-r1/01_v31_s3_s1_full/`

回执：

`cold_audit_receipt.json`

结果级审计通过：

- ZIP SHA-256 与 manifest 一致；
- `result.csv` SHA-256 与 manifest 一致；
- ZIP 内 `result.csv` 与包内文件逐字节一致，ZIP 只含该文件且 `testzip` 通过；
- 322 个官方 test ID、顺序、列顺序、唯一性、有限非负值均通过；
- 分段重排/单行解析/行序恢复一致性通过；
- 审计未读取 `复赛_train` 下任何训练文件。

模型级冷推理：

**BLOCKED**。该包未保存五成员融合所需的序列化基础模型；若现场重新拟合基础模型，必须读取 `复赛_train`，不满足“禁止训练文件访问的冷推理”定义。因此不能声称完成模型级冷审计，也不能在审计失败/缺失后静默重训覆盖旧包。

## 5. 最终决定

- A 保持冻结讨论对象，不自动打包、不写桌面、不上传。
- B 仅保留为开发替代，不替代 A。
- 条件追加扩展全部淘汰。
- `01_v31_s3_s1_full` 字节保持原样；结果级冷审计通过，模型级冷审计 pending/blocked。
- 旧五包和 V3.1 原始 ZIP 字节未改变。
