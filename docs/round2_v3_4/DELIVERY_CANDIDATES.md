# Round2 V3.4 本地提交候选包

日期：2026-09-23  
分支：`round2-v3.4-ebm-and-constrained-composition`

## 1. 推荐顺序

| 排序 | 包 | 状态 | 推荐理由 |
| ---: | --- | --- | --- |
| 1 | `01_V34_A_MECHANICAL_CONSTRAINED` | 最终外层 16061 已验证 | 同协议 `delta_L1 +0.0394248`，五折均为正；本轮唯一完成独立外层确认的 V3.4 流程 |
| 2 | `02_V34_B_DIVERSITY_CONSTRAINED_DEV` | 仅开发 seeds 42/3407 验证 | 多样性完整候选，开发 `delta_L1 +0.0438448`，两个开发 seed 均为正；与 A 的实质差异在时长专家，作为不同结构备用 |

说明：

- 包只生成本地 ZIP，用户自行上传，本轮 `agent_uploads = 0`。
- A/B 都是完整双目标替换包，不是单列替换包。
- B 没有独立最终外层结果，不能声称与 A 同等确认强度。
- 当前平台已知最佳仍为 AJ3 铁量＋完整 B3 时长，用户回传 `96.1259`；它是平台已测参照，不是本轮新生成包。

## 2. 包位置与摘要

目录：

`local/runs/round2-v3.4-ebm-and-constrained-composition/release-candidates-r1/`

### A. `01_V34_A_MECHANICAL_CONSTRAINED`

- `result.csv` SHA-256：`1d163fa19b011d117fa1014e5d8f5fdeba9c3104aa5f86073e28b4a33344ba40`
- ZIP SHA-256：`0007230038f365126ae337c3e0af8b7bb040fe3d351c624ff2819ea1efe649e7`
- 铁量权重：L1 `0.5000000000000001`；`v34-s1-ebm_boundary-0016` `0.3397038169347897`；`v34-s1-ebm_boundary-0020` `0.16029618306521032`
- 时长权重：L1 `0.6192377408552066`；`v34-s1-ebm_boundary-0089` `0.28691289439355067`；`v34-s1-global_spout_shrink-0136` `0.09384936475124295`
- 最终外层 16061 本地包分：`96.19973698527157`
- 同协议 L1：`96.16031215578445`
- `delta_L1`：`+0.03942482948711756`

### B. `02_V34_B_DIVERSITY_CONSTRAINED_DEV`

- `result.csv` SHA-256：`3dcf83d9f9652d08b0231d204e5a549d5195d2616544a8773b17d61d5bdf31dd`
- ZIP SHA-256：`f844470202a3f93d96eca0bfa6a85674337367dc935e363bdfc81fe47e89bddb`
- 铁量权重与 A 相同。
- 时长权重：L1 `0.71574111835936`；`v34-s1-ebm_boundary-0089` `0.1937242967979882`；`v34-s1-ebm_residual-0114` `0.09053458484265177`
- 开发完整包 `delta_L1`：`+0.04384483535657159`
- seed 42 delta：`+0.048955302736645306`
- seed 3407 delta：`+0.03873436797649753`

## 3. 包验证

两个包均通过：

- 322 个官方 V2 test ID，顺序与 `test_samples.csv` 一致；
- 三列正确，无重复、无缺失；
- 预测有限且非负；
- ZIP 只含 `result.csv` 且逐字节回读一致；
- `result.csv` 与 ZIP SHA-256 已登记在各自 `manifest.json`。

## 4. 未执行

- 未上传平台；
- 未写桌面；
- 未做模型级冷审计（A/B 都是新拟合包，模型级审计需要单独保留并回读序列化模型）；
- 未生成单目标隔离包；若需要，可下一步从 A 的缓存预测生成“A 铁量 + AJ3 时长”或“AJ3 铁量 + A 时长”。

## 5. 桌面写入回执

桌面目录：

`/mnt/c/Users/lqh22/Desktop/submission/round2-v3.4_top2_20260924/`

内容：

- `01_V34_A_MECHANICAL_CONSTRAINED/Luqhhh_bf_tap_predict_round2.zip`
  - SHA-256：`0007230038f365126ae337c3e0af8b7bb040fe3d351c624ff2819ea1efe649e7`
- `02_V34_B_DIVERSITY_CONSTRAINED_DEV/Luqhhh_bf_tap_predict_round2.zip`
  - SHA-256：`f844470202a3f93d96eca0bfa6a85674337367dc935e363bdfc81fe47e89bddb`
- `TOP2_ORDER.txt`

直接挂载 `/mnt/c` 为只读，桌面文件通过 Windows PowerShell 互操作写入；写入后已通过 WSL 侧读取回环并核对 SHA-256。原桌面文件未覆盖，新目录为新唯一批次目录。
