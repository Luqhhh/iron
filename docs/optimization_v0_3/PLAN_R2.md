# optimization-v0.3-r2：与已执行证据对齐的接续方案

后续用户已明确指示执行；OPT-10 的实际接续证据见
[OPT10_EXECUTION_R2.md](OPT10_EXECUTION_R2.md)。下文保留接续前的核验时点和待办登记。

核验日期：2026-09-08。用户提供的 r2 依据 c88cdad；本次实际核验
Git 工作树干净，origin/optimization-v0.3 与本地均为
`04a710d9c7bc137e765eb38ad79ab73ce138f7e2`，已包含后续实验与回退。
本地新分支 optimization-v0.3-r2 从该提交接续，其祖先包含 c88cdad。
没有重建旧实验、覆盖失败运行、推送远端或上传平台。

## 应保留的已执行事实

- OPT-07 阶段审计、每 origin 校准及全局 C_ref 选择已经完成。
  C_ref=E12-raw，J=0.17171342055719652；CVcal J=0.17258611678486493。
  旧 E16 全网格不具有新严格校准流水线的时间外身份。
- OPT-08 的 15 配置与目标级晋级、OPT-09 的 4 特征和 2 交叉已完成。
  F-B 对应 F-LAG；F-C 对应 F-RHYTHM；F-A/F-D 对应两个次级候选。
  顺序已成为历史事实，不能追溯改成 r2 的优先级。
- 原门槛下没有自动晋级候选，有限开发搜索已经关闭。
  最好新增算法 CB-FC-CVcal 的 J=0.17122511323998096，改善不足 .001。
- 该算法曾经用户例外选择生成开发发布包，回传 test_a=82.2871，随后回退。
  当前仍为 E16，test_a=82.9918；两者均为用户回传，非独立平台核验。
  E16 ZIP SHA256：`1200d4dd8dee6e797aeeba86db1ceb02293c78dd36796d2e8d75156ce77160aa`。
- 105 个锁定测试是此前提交的记录。本次测试结果另行记录，不能充作旧运行复现。
  保护标签仍未消费；完整 holdout_scoring / final_training 发布入口尚未完成。

旧定义、执行次数、模型参数、最佳轮数及发布历史以
[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) 和原本地运行记录为准，不回写。

## 本次补充：同月训练与历史更新对照

仅使用已有成功 reference run 的 E09/E12-raw 开发预测及误差文件。
按 sample_id 严格一一配对，并核验相同标签、参考时刻和登记训练 cutoff。
重新计算误差和、分母与两个目标 WMAPE，没有读取官方标签文件或新增模型 fit。
所有 8 对完整，缺失 0 对；不使用不同月份的均值代替配对。

|评估月|样本数|E09 H2−H1|E12-raw H2−H1|
|---|---:|---:|---:|
|2024-07|310|0.004151|0.001915|
|2024-08|313|0.009253|0.006757|
|2024-09|288|0.012687|0.011481|
|2024-10|333|0.002723|0.001926|
|均值||0.007204|0.005520|
|中位数||0.006702|0.004342|

四个月均为正值，但幅度不同。这是新增训练标签、训练跨度和历史快照更新的
联合效果，不能单独归因、线性外推到平台，也不用于决定读取哪些保护标签。
本次 G0 为配对分析通过，G1 未作新候选评价。

完整误差和、分母、WMAPE、E、cutoff、sample identity 摘要与输入哈希仅保存在
本地 `local/runs/optimization-v0.3-r2-refresh-r1/`。
`training_refresh_comparison.csv` SHA256：
`192187611c7ac06695d9b984cae45c846e9681f0b0c12b06ad14d32a5f0753fc`。

执行命令：

```bash
git ls-remote --heads origin optimization-v0.3 optimization-v0.2
git switch -c optimization-v0.3-r2
uv run --locked --python 3.12 python -m bf_tap.optimization.refresh --reference-run local/runs/optimization-v0.3-opt07-reference-r2 --output local/runs/optimization-v0.3-r2-refresh-r1
uv run --locked --python 3.12 pytest -q --junitxml=local/reports/pytest-optimization-v0.3-r2-r1.xml
```

辅助摘要命令首次调用系统 `python` 失败（命令不存在），改用锁定 uv Python 后成功。
没有因此重跑实验或删除证据。

本次锁定 Python 3.12 全套测试实际通过 **113 项**（27.66 秒），包括新增的
错配样本、标签不一致、越月、非有限预测、零分母与保护月份拒绝检查。
这证明当前工程测试通过，不等于重训复现旧模型或模型质量达标。

## 剩余执行顺序与身份边界

1. 不重跑已耗尽的 15/4/2 搜索预算，不用 83.9、82.2871 或新阈值微调旧候选。
   原 GENERAL 门槛及原 G1 结论保持。r2 的 STAGE_A 是现在才提出的规则，
   对已经看过结果的候选只能算事后分析，不能宣称预注册接受。
   本次不重新排名旧候选、不授予 STAGE_A_ACCEPTED。
2. 无原规则胜出 challenger，后续全量训练算法按 r2 回退为完整 C_ref=E12-raw：
   E09/E04 两组件固定 800 轮、80/20、非负下界、零校准。
   E16 当前可恢复发布包保持，直到正式包通过验收；它不替代严格 C_ref。
   这一步是后续方案身份，尚不是完成的冻结 manifest 或正式训练。
3. 先补齐 OPT-10 工程：支持 C_ref 的完整保存/加载、保护评分和正式训练入口，
   以及纯合成 lifecycle 测试。现有单目标 bundle 测试和授权函数
   不能代替完整两组件发布验收。测试须包含拒绝无授权、manifest/配置不一致、
   标签可用性边界、失败证据及追加账本保护、冷进程恢复与 predict 不调用 fit。
4. 实际测试索引最早参考时刻、完整输入摘要、policy digest、算法、源码和
   校准规则纳入冻结 manifest。候选冻结之前不读取真实 November 目标。
   保护报告与 final_training 分别使用与 manifest 绑定的授权记录和追加账本。
   用户 r2 已提出按流程执行二者的意图，不能因旧 PLAN 的“未授权”描述
   忽略新指令；但不能将文本意图冒充已通过的工程授权校验。
5. r2 STAGE_A 的保护回退门槛为 HOLDOUT_H1 E≤C_ref，两个目标增量≤.003。
   无新 challenger 时不得虚构二选一胜者；冻结记录须明确只有 C_ref，
   保护报告只作该参照的报告。四场景共享 November 标签，消费后不再 untouched。
6. final_training 使用 reference_time<fit cutoff 且 label_available_at≤fit cutoff
   的全部合法训练样本，不凑指定行数；历史快照随合法截止点更新，公共过程
   数据仍按每条预测时刻推进。保持 E12-raw 的零校准，不临时采用 E16 常数。
   验证冷进程恢复、同 bundle 原始预测一致、CSV/ZIP 回读、三列/sample_id、
   有限非负数及摘要后再形成可提交包。

上述第 3–6 项本次尚未执行，不声称 OPT-10 完成，也不声称已合法全量训练。
没有新的 release manifest、CSV/ZIP 或平台分数。后续反馈需分别说明相同数据下的
算法差异、训练和历史 cutoff 更新、实际平台回传；不能将 refresh gain 当分数承诺。
