# Round2 V4.2 platform feedback: V42_IRON_N2_Q25

日期：2026-09-25

来源：**USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED**（用户回传，未经独立账号核验）

## 1. 回传

| 项 | 值 |
|---|---|
| 候选 | `V42_IRON_N2_Q25` |
| 配方 | 父包 V36 `tap_iron` 列 + 0.25 × V4.2 `N-N2-tap_iron` 全量重训；`tap_time_len` 不改动 |
| ZIP SHA-256 | `7fdfec5b3a2e3c58c01b526f1ee8f8b8e5d09625ec34ecbc581704b0c44fc2b5` |
| `result.csv` SHA-256 | `296e64c482f332191f7db5deaf26812c8ad057ced3e16bda1fc6ae67f7c9ee24` |
| 桌面路径 | `C:\Users\lqh22\Desktop\submission\round2-V42-IRON-N2-Q25-20260925` |
| 用户回传平台分 | **96.2533** |

私有回执：`local/runs/round2-v4.2-structure-search/release-r1/V42_IRON_N2_Q25/platform-feedback-r1/feedback.json`

## 2. 与当前最优的对比

| 候选 | 本地同协议分 | 平台（用户回传） | 平台−本地 |
|---|---:|---:|---:|
| `V36_USER_REQUESTED_OUTER_FAILED`（当前最优） | 96.2038 | **96.2734** | +0.0696 |
| `V42_IRON_N2_Q25` | 96.2186 | **96.2533** | +0.0347 |

- 相对当前最优：**−0.0201**；距下一阶段目标 96.3：**−0.0467**。
- **当前平台最优仍是 V36 `96.2734`，未被取代。**

## 3. 本地增量没有转移

同一个 N2 四分之一混合，在开发切分上是一致为正的：

| 证据 | 协议 | 平均增益 | 正折 |
|---|---|---:|---|
| `n2-seed-init-v2`（修复后，与出货模型代码一致） | 2 split seeds × 5 folds | **+0.014374** | 8/10 |
| `repl-t3407-r1`（manifest 所引用的未修复证据） | 2 split seeds × 5 folds | +0.013617 | 8/10 |

本地均值 `+0.0144`，平台相对 V36 却是 `−0.0201`，**缺口约 0.0345**。

结论：这个方向在本地的一致小增益**没有转移到平台**，反而相对在任包倒挂为一次平台损失。预登记的 `+0.02` 融合门槛本来就未通过（coarse `+0.0093`、完整覆盖 `+0.0144`），本次平台回传与该判断方向一致。本地 OOF 排名或 delta 不是平台分数预测。

## 4. 状态与后续

- 不构成 promotion，不重新分类任何已冻结结论；该包始终是探索包。
- 优先包保持 `V36_USER_REQUESTED_OUTER_FAILED`；`submission_priority` 仍为空。
- agent 上传数保持 0；桌面交付由用户执行。
- N2 这条「共享选择器的可微遗忘树在铁量上做固定 0.25 混合」的路线的平台证据到此为**负**；后续若要继续，应换结构假设（例如 `FOLLOWUP_SPEC.yaml` 中已声明但 `enabled: false` 的 `N2_DEPTHSELECT`），而不是继续在既有包上做小幅权重微调。

## 5. 待核验事项（G0，与本次回传无关）

出货包自述 `time_column_preserved_byte_for_byte: true` 与实测不符：父包 `pred_tap_time_len` 有 136/322 行字符串不同，其中 74 行是真正的 float64 差异（≤2 ULP，最大 2.84e-14，相对约 1.8e-16），其余 62 行只是 `.17g` 与最短 repr 的写法差异。根因在 `src/bf_tap_r2/v4_2_package.py:200-207`：父包经 `pd.read_csv`（默认 `float_precision='high'`，非精确往返）解析后又用 `str(v)` 重新格式化，而不是原样透传父包 CSV 字符串。数值影响可忽略，包身份与混合配方不受影响，但它不满足「未修改的列保留父包 CSV 字符串」的归因规则。该问题留待用户决定处理方式。
