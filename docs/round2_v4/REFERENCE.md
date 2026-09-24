# Round2 V4：A/B_star 同协议参照与环境指纹

日期：2026-09-24  
状态：**A 开发参照已恢复；完整 B_star 未恢复；已用 A 作允许的开发锚点。**

## 1. 参照身份

| 对象 | 状态 | 用途 |
|---|---|---|
| `V34_A` 开发重放 | 已从私有 V3.4 OOF/cache 恢复 | 本轮 coarse screen 的同协议 A 锚点 |
| `01_V34_NE3_CAP50` / `B_star` | 本 checkout 未找到完整产物、OOF 或可执行原始流程 | 未恢复，不用旧 outer 的 +0.0012 人工补分 |
| `V34_NE2_CAP65` / `V34_NE3_CAP65` | 本 checkout 未见平台回传或运行产物 | 不推断分数、排序或额度 |

任务书允许在完整 `B_star` 未恢复时先用可复现的 A 作开发锚点；本文所有 coarse 增量
均相对该 A 锚点报告，不表示“已经超过当前平台最佳”。

## 2. A 恢复配方

A 的测试包配方记录在：

`local/runs/round2-v3.4-ebm-and-constrained-composition/release-candidates-r1/01_V34_A_MECHANICAL_CONSTRAINED/manifest.json`

在开发 OOF 上按冻结顶层权重重放：

```text
iron = 0.5000000000000001 * L1_iron
     + 0.3397038169347897 * v34-s1-ebm_boundary-0016
     + 0.16029618306521032 * v34-s1-ebm_boundary-0020

time = 0.6192377408552066 * L1_time
     + 0.28691289439355067 * v34-s1-ebm_boundary-0089
     + 0.09384936475124295 * v34-s1-global_spout_shrink-0136
```

L1 OOF 来自：

- `local/runs/round2-v3.4-ebm-and-constrained-composition/l1-oof-r1/seed-{42,3407}/l1-oof-fold-*-{target}.csv`

专家 OOF 来自：

- `local/runs/round2-v3.4-ebm-and-constrained-composition/refine-r1/seed-{42,3407}/pred-v34-s1-*.npy`

## 3. A 恢复结果

### 3.1 完整五折

| seed | 铁量 WMAPE | 时长 WMAPE | 完整包分 |
|---|---:|---:|---:|
| 42 | 0.03764855932188847 | 0.03855355347511435 | 96.18989436014986 |
| 3407 | 0.037718435845472874 | 0.03829358110283327 | 96.19939915258469 |
| **两种子均值** | **0.03768349758368067** | **0.03842356728897381** | **96.19464675636728** |

这与 V3.6 登记的 A 开发重放分 `96.19464675636728` 在 double 精度上一致。

### 3.2 本轮 coarse screen 使用的 folds 0/1

| seed | 铁量 WMAPE | 时长 WMAPE | 完整包分 |
|---|---:|---:|---:|
| 42 | 0.03656327346412752 | 0.03771289042579546 | 96.28619180550385 |
| 3407 | 0.03884087827195198 | 0.03812281474464557 | 96.15181534917012 |

单侧候选完整包增量按“另一目标保持 A”计算：

```text
package_delta_single = 50 * (A_target_wmape - candidate_target_wmape)
```

## 4. 环境指纹

私有运行目录中的身份文件：

- `local/runs/round2-v4-mechanism-search/coarse-r2/environment.json`
- `local/runs/round2-v4-mechanism-search/coarse-r2/reference/a_dev_reference.json`
- `local/runs/round2-v4-mechanism-search/coarse-r2/reference/a_dev_oof.npz`

关键值：

| 项 | 值 |
|---|---|
| Python | 3.12.12 |
| platform | Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.35 |
| Git HEAD | `fccf219328177f02dbb51d4aa2c22b4b7a7adbfa` |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| interpret | 0.6.10 |
| lightgbm | 4.6.0 |
| catboost | 1.2.8 |
| train rows | 2754 |
| data hash | `1f517e21a449d34dfa631c1d19e259f76306236daf7d6ea121a3a846418ad550` |
| fold-42 hash | `11293ebc6e1e7ff633d1d8ccd1cc5a4a2c398da5bfa6f9b6f0bbbaca5d7469e6` |
| fold-3407 hash | `0cdb01e9f86d7095a97f77a67c49ca6c0f10635e5e8c3f0b49834c8a67061c86` |

## 5. 最终 outer seed 23003 状态修正

任务书撰写时记录 seed 23003 未消费；当前仓库后续的 V3.6 证据显示，用户显式指令
已在 V3.6 中执行 outer 23003：

- `docs/round2_v3_6/RESULTS.md` §4
- outer 23003 V3.6 包分 `96.19003538073855`，未达到 96.25 与 `Δ_A >= 0.02`
- 因此 V4 不得再消费该 seed，也不得把同一 outer 残差继续用于调模后重报
- 配置中已登记：`final_outer_status = CONSUMED_BY_V3_6_USER_INSTRUCTION_NOT_AVAILABLE_FOR_V4`

## 6. 边界

- 本文只恢复了 A 开发重放，没有恢复完整 B_star 流程，也没有生成平台包。
- A 锚点上的 delta 只能用于机制筛选，不能直接当作平台分差。
- `V34_A = 96.2684` 仍只是用户回传平台分，未独立核验。

## 7. 平台回传更新（2026-09-24）

`V36_USER_REQUESTED_OUTER_FAILED` 用户回传平台分 **96.2734**，超过 `V34_A = 96.2684`，
成为新的当前平台最佳用户报告；距严格大于 `96.3` 还差 **0.0266**。该包 outer 23003 仍失败，
且不是正式晋级包。A 开发锚点与 V4 coarse 结果不受影响；完整 `B_star` 仍未恢复。

ZIP SHA-256：`ef6e72f140e314c995ccb40a79439bf0fe3846c6089917936b9cba24399da7b1`  
result SHA-256：`b5ed9b51e9127aa0a7740be2100dd2b1d7464a357076aae9fe3068c664c35f20`。
