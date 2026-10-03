## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / validate。
- Owner：本聊天 xjn；日期：2026-10-03；Verification Status：ANALYZED；Confidence：CAUTION。
- 完整科学结果、独立新进程冷审计与冻结零拟合分类完成；不宣称重新训练所有 greedy 分裂结构的权威复现。

## 结论

`RESIDUAL_L1_PARTITION_D3_A20` 未获确认或封包资格。固定20%仅改铁量，Q75时长不变；完整对照如下，单位是本地分数增量，**不是平台分数**。

| 外层 seed | 相对 Q75 | 相对原 LONG |
|---|---:|---:|
| 42 | −0.0243468231 | −0.0013897312 |
| 3407 | −0.0209725150 | −0.0012446119 |
| 均值 | −0.0226596691 | −0.0013171715 |

两个seed、十fold、各seed/spout相对Q75均负。冻结分类回读逐fold重组完整OOF并复算指标：`not_shortlisted`，formal/exploration/submission_priority 均为空，原科学确认决定仍为false。
分类器的 gain 字段为铁量 WMAPE 改善，不是上表两目标本地分数；`formal` 即使出现也仅是两seed开发shortlist，不能替代四seed正式门。

## 实际核验与计数

冻结科学源 be28245；当前SPEC、30个源码摘要、57个输出及4个输入摘要闭合。
G0另计4模型/1722树。科学实际20模型/169903树：10个完整12000轮选择器共120000树，10个fresh refit共49903树。
原LONG的20模型只复用比较，不重计为本次拟合。四阶段实际退出全0；科学20模型冷预测最大差异0。
节点残差/L1统计见证采用冻结1e−12容差，叶值、完整轨迹、选轮及预测精确核验；不是全部greedy结构重训。

零拟合分类实际退出0；新拟合、模型反序列化、新官方标签解析均0，允许原件字节SHA核验。
确认、全量、包、桌面写入和助手平台上传均0；没有更改队友队列或其他任务。

| 私有凭证 | SHA256 |
|---|---|
| development manifest | e58d4e99a57b66176ed1c1698c834758324841c0e5b2bfa2cf855a3be1efafbc |
| development result | 9d807b286d2c6782244a5f76f85f6d66aeffe77fb1dd1d2769d08ea813be9bd7 |
| development cold | ce47614916699b508c36448e834bddc53bee71b346501c0815517df484e7be3e |
| triage manifest | 927352f31c44f45cf4572e8dede0c7220e00fc3cef88c596962905bcf80e364c |
| triage metrics | 8a5f09343635d53fb7e4403ed5bac6703c0d09fb8faaf7f1f979368104262e4f |
| triage result | c484d9c8df7593113356ac1bc6f2ec6ecdb456f7ffcdf48bed4c51f0e72080d1 |

## 深度解释与下一步

选轮459–11621，0/10精确命中12000；仍有3/10位于末10%区间，最后1000轮校准3/10改善、7/10恶化。
不能认定全部训尽，也不因少数靠末端自动加epoch。更久训练、工程通过与独立选题身份都不构成提分依据。
构树目标与criterion同时改变，仅比较完整配方；旧sign是合法L1梯度，不是实现错误。
本结果不证明整家族或平台必负，也未给当前新OOF的所有正权重作数学否决。
旧LONG的凸性结论只属于旧固定向量，不能借用到本轮。

结合用户PDF经验，停止自动追逐这一负配方的轮数/权重近邻，转向明确不同机制与完整强配方。
新研究必须另冻范围、验证与失败条件；不追溯改变本轮失败，不为交付数量制造弱包。
两seed复用同2754行及多轮开发，限制选择后推断；Q75内层规则也与本候选不同，不声称纯机制因果隔离。

## 统计误读扫描：11/11

| 检查 | 核对与限制 |
|---|---|
| Simpson | 对Q75各seed/fold/spout均负；不外推未检查分组 |
| 生态谬误 | 不把seed均值当单样本或平台效果 |
| Berkson | 官方/平台筛选机制未知，CAUTION |
| Collider | 不事后筛错误样本或构造query误差路由 |
| 忽略基率 | 非分类筛查任务，不适用 |
| 均值回归 | 完整OOF，未挑极端样本 |
| 幸存者 | 唯一候选及全部二十拟合完整，历史负结果保留 |
| 多重搜索 | 多轮复用开发标签，不给选择后显著性结论 |
| 分叉路径 | 冻结选轮、20%和失败决定不变 |
| 相关当因果 | 不由耗时、训练下降或联合改动推平台因果 |
| 反向因果 | 长实验与高分无方向性保证 |

私有原件仅 `local/runs/laplace-leaf-partition-20261002/{development-r1,triage-r1,sequence-r1}`，不入Git。
