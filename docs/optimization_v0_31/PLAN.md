# optimization-v0.31：两个目标隔离的 OOB 叶响应汇总实验

本阶段是独立优化阶段，审阅基点 `optimization-v0.30-oob-compose-and-time-growth@383610d`。
两个候选在任何本轮平台反馈前同时冻结，平台预算 2、各一次，默认顺序 A→B；A 的反馈不得改变 B。
不新增模型、树、预处理、校准或分类器拟合，不建立第三个候选，不自动上传或写桌面。

## 候选定义

- A `V31I_OOB_OCCURRENCE_POOL_BLEND`：铁量使用 v0.29 的 256 棵原铁量森林和原 OOB 叶附件；对每个查询把每棵树中被选叶成员的每个出现位置计一次，得到 `c_i` 与 `S = sum_b n_b`，新权重 `w_new[i] = c_i / S`。取原始铁量响应的整数质量较小中位数 `q_I`，按原精度来回六位得 `q6`；再与原始 V26A 完整铁量端点 `C6` 做 v0.28 整数微单位 ties-to-even 平均 `I_A = mean6(C6, q6)`。父 V30A 时长列逐字符串复制；禁止对 `0.5*V30A_iron + 0.5*new_QRF` 取平均。
- B `V31T_OOB_OCCURRENCE_POOL_TIME`：时长使用 v0.26 A 的原 256 棵绝对误差时长森林和原 OOB 叶附件；应用同一 occurrence pooling 规则得原始 `q_T`，六位来回得 `q6`。门控只使用原 v0.15 支持度的 `effective_neighbors < 500`、`spout_no == '1'` 与原始 60 日运动窗口中位数/空窗回退，最后做 25% 收缩；禁止用新池化的 effective_neighbors 或 V26A 支持度决定 gate。铁量逐字符串复制父 V30A。

## 预算

| 项目 | 开发 | 最终 | 总上限 |
| --- | ---: | ---: | ---: |
| 新模型/树 fit | 0 | 0 | 0 |
| 新预处理/LAD/beta/lambda/bias fit | 0 | 0 | 0 |
| A 池化读取协议清单 | 6 | 1 | 7 |
| B 池化读取协议清单 | 6 | 1 | 7 |
| 已认证 OOB 附件复用 | 12 | 2 | 14 |
| 新候选 ZIP / 平台测试 | — | 2 / 2 | 2 / 2 |
| 自动上传/桌面写入/公开推送 | 0 | 0 | 0 |

## 评价与诊断

主口径为各 origin 等权的 `macro_origin_mean_wmape`，`exposure_pooled_wmape` 仅独立诊断。
A 必须满足 `delta(time)=0, delta(E)=0.5*delta(iron)`；B 必须满足 `delta(iron)=0, delta(E)=0.5*delta(time)`。
诊断报告每查询树数、最小/最大 selected count、S 与 distinct selected rows、相对旧等树质量分布的 L1 权重变化、
新 effective_neighbors、最大单行权重、回退树比例与回退质量比例，并记录 1000 次 seed 2026 日历周 bootstrap。
OOB 不是独立验证集，S 不是新增样本数。冷审计恢复源森林、源附件和旧 gate，核验新算法只改变质量分配。
