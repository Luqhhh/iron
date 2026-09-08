# optimization-v0.3 / OPT-07–10

登记日期：2026-09-08。起点为 optimization-v0.2 的
`c88cdad01a52bce4270f20f7d2e82fa5f6fa42cf`。baseline tag、旧配置和旧证据不改写。
本文件记录用户方案的实施约束；进度与未完成项见 RESULTS_SUMMARY.md。

保留 E16 为已有平台 incumbent、E12-raw 为关键对照。旧 E14/E16 的全 J
仅作开发选择/历史复算证据；固定残差预登记不能恢复严格外层独立性。
不依据平台分数调整权重、残差或跨度对应关系，不承诺平台涨分。

1. OPT-07：按 cutoff 和真实预测时间输出阶段跨度身份；当前 November cutoff
   的 December 为 H2，March 为 H5。E12-CVcal 每个 outer cutoff C 使用
   C−28 天冻结的 E09/E04 800 轮底模产生 [C−28,C) 预测，只用 C 前已可用
   的标签，至少 100 个唯一样本才拟合时长 median(pred−actual)。铁量不校准。
   E12-raw/CVcal 全局按 J 选 C_ref，平分选 raw。
2. OPT-08：固定 E09 特征，9 个 CatBoost MAE、6 个 LightGBM L1。
   C−56 天训练，冻结内部历史，在 [C−56,C−28) 选择轮数，标签须在 C−28
   可用；patience=150。两旧 DEV 折的分目标 WMAPE 均值初筛，平分低容量：
   CatBoost 低 depth、较高 L2；LightGBM 少叶子、较高 min_data_in_leaf。
   每种模型每目标至多晋级一个配置，共最多四个单目标配置进入五 origin 网格。
   两目标分别记录轮数、训练身份、完整参数、编码器和模型摘要。
3. OPT-09：最多 F-A 去 burden、F-B 分段过程 mean/count、F-C 已知开铁索引、
   F-D burden 事件变化四项实验；最多前二名进行两次新模型交叉。F-B 窗口
   (t−b,t−a] 且 available_at≤t。F-C 输入来源和 F-D 字段语义未核验前 BLOCKED。
4. OPT-10：先合成测试 lifecycle 与完整组合离线恢复，再冻结至多一个
   challenger 和一个预先回退算法。保护评分与 final_training 分别授权，
   每次使用 protection policy、精确 frozen manifest digest、追加访问账本。
   当前用户方案未授权真实保护标签读取或平台上传。

新增开发门槛：相对全局 C_ref 的 J 至少降低 .001；至少 3/4 跨度改善，
各跨度退化≤.002，各目标等跨度 WMAPE 退化≤.003；DEV_LONG 胜同场景较好
B0/B1，DEV_SHORT 相对 C_ref 退化≤.002。阈值不回写旧 G1。
保护报告预设回退门槛：四场景等权损失≤C_ref，H1/H2 退化各≤.002，
每目标四场景平均 WMAPE 退化≤.003。四场景共享 November 标签，并非四个月。

真实日期周块配对分析须对重复出现在多个 origin 的同一样本同步重采样，
每次重算 WMAPE/J；不得堆叠 14 格作为独立 OOF。开发期被重复选择使用，
区间不覆盖模型选择偏差。

完成有限批次后停止追加微调。完整发布需统一恢复模型、编码器、特征、
组件拓扑、校准器、各阶段与来源身份，并分别报告算法/训练范围/历史更新影响。
实际未完成的步骤不得以配置登记或单元测试代替交付。
