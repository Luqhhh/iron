# v0.3 执行状态（2026-09-08）

起点 c88cdad01a52bce4270f20f7d2e82fa5f6fa42cf；本地 optimization-v0.3。
实验期间未修改远端，未自动提交平台，未读取 November 保护目标。用户现已
授权将本地 optimization-v0.3 提交推送；不合并或改写 main。E16 incumbent 身份保留。
发布历史：有限搜索收口后，用户明确批准 CB-FC-CVcal 例外晋级并替换当时的
发布候选；development test_a 包已生成并替换桌面包。原 G1_FAIL 不改写，
E16 作为历史平台 incumbent 和回退保留。详见 [CURRENT_RELEASE.md](CURRENT_RELEASE.md)。
后续用户回传新包 test_a 成绩 82.2871，比 E16 低 .7047；未独立核验。
未据总分调整模型、校准或开发阈值；用户随后确认回退，当前指针与桌面包已恢复 E16。
开发 J 的小幅改善未在此次
平台回传中体现，不能承诺加入 November 训练数据后一定弥补差距。

## 已完成：OPT-07A 元数据审计

`local/runs/optimization-v0.3-opt07-audit-r1/`：两个既有组件 E09/E04 的实际
cutoff 均为 2024-11-01。读取三个测试阶段的 sample_id/reference_time 后，
映射为 test_a H2、test_b H3、test_c H4/H5；现有网格未覆盖 H5。
test_b/c 仅映射审计，不声称生成这些阶段的新预测。
旧残差精确拟合样本仍 UNRESOLVED，详见 CALIBRATION_PROVENANCE_REVIEW.md。

## 已完成：OPT-08 两折 15 配置初筛

`local/runs/optimization-v0.3-opt08-screening-r1/`，G0_EXECUTION_PASS。
15 配置 × 2 目标 × 2 折 ×（内部选轮数 + 外层重训）= **120 次单目标 fit**。
模型使用 E09 同一特征列；两目标分别选择轮数。仅内部时间块作为 eval_set。
以下为两 DEV 折分目标 WMAPE 等权均值，越低越好；不是完整网格 J。

|配置|铁量|时长|
|---|---:|---:|
|CB01 depth3/L2=5|0.174339|0.166646|
|CB02 depth3/L2=20|0.173898|0.165606|
|CB03 depth3/L2=50|0.173907|0.165779|
|CB04 depth4/L2=5|0.173781|0.166141|
|CB05 depth4/L2=20|0.174272|0.166766|
|CB06 depth4/L2=50|0.173517|0.168062|
|CB07 depth5/L2=5|0.173459|0.166241|
|CB08 depth5/L2=20|0.172996|0.166907|
|CB09 depth5/L2=50|0.174069|0.165712|
|LG01 leaves7/min40|0.175393|0.165634|
|LG02 leaves7/min80|0.173544|0.166705|
|LG03 leaves15/min40|0.175983|0.168137|
|LG04 leaves15/min80|0.175334|0.166680|
|LG05 leaves31/min80|0.174533|0.167130|
|LG06 leaves31/min160|0.175533|0.168583|

按预登记分目标两折均值晋级：CB08 铁量、CB02 时长、LG02 铁量、LG01 时长。
不是每一种算法自动进入最终发布；初筛阶段仅为 G1_NOT_EVALUATED_FULL_GATE，
之后的完整门槛结论见下文。

## 已完成：OPT-07B 严格参照

`local/runs/optimization-v0.3-opt07-reference-r2/` 已完成两 DEV 折与 14 格网格，
70 次单目标 fit。每 origin 重新生成校准 OOF；所有校准块均达到 100 样本，
未触发零回退。

|完整算法|J|H1|H2|H3|H4|
|---|---:|---:|---:|---:|---:|
|E12-raw|0.171713|0.156806|0.170448|0.182095|0.177505|
|E12-CVcal|0.172586|0.158090|0.170033|0.183934|0.178288|

全局固定 **C_ref = E12-raw**，未逐格取最小值。新严格校准没有改善整体 J，
不能把旧 E16 的 J 改写为严格新算法的分数。新门槛需 J≤0.17071342055719652，
并满足其余跨度、目标与 DEV 条件。

## 已完成：晋级全网格与 F-A/F-B

`local/runs/optimization-v0.3-opt08-09-grid-r1/` 完成两个 DEV 折、五 origin 的
14 格网格，新增 68 次单目标 fit（初筛预测在旧折复用，未重复计作 fit）。
三个成功训练运行合计 258 次单目标 fit；失败 r1 的额外 fit 单独保留，不计入成功运行。

|候选|J|相对 C_ref 的 J 变化|新门槛|
|---|---:|---:|---|
|C_ref / E12-raw|0.171713|0|参照|
|E09|0.173861|+0.002148|冻结特征对照|
|CB-best-per-target|0.172670|+0.000957|FAIL|
|LG-best-per-target|0.174527|+0.002814|FAIL|
|F-A 去 burden|0.173876|+0.002162|FAIL|
|F-B 分段过程 mean/count|0.173197|+0.001484|FAIL|

CB 组合铁量使用 CB08、时长 CB02；LG 组合铁量 LG02、时长 LG01。
两组合四个跨度均未超过 C_ref。F-A 未显示相对 E09 的整体条件增益，
F-B 相对 E09 的 J 改善约 0.000664，但仍未超过 C_ref。
这些结果不支持替换发布候选；保留旧 E16 平台 incumbent，不追加微小权重/残差搜索。

四个候选各做 1,000 次同步 calendar-week 配对重采样，989 次具有所有格子的
正分母；丢弃的 11 次没有重抽或补造。J 差值（候选−C_ref）的 2.5%–97.5%
开发重采样区间：CB [0.000036, 0.002228]，LG [0.001454, 0.004567]，
F-A [0.001111, 0.003685]，F-B [0.000127, 0.003202]。
重复出现在不同 origin 的同一样本使用相同周权重；这些区间不覆盖模型选择偏差。

各候选两目标的最大五个误差预测实例已另行审计，保存在
`local/reports/optimization-v0.3-max-error-audit-r1.json`；保留 origin、horizon、
真实日期与样本身份。同一目标的前五个实例只涉及 2–4 个唯一样本，再次说明
不能将跨 origin 的重复预测当成独立误差样本。逐格完整误差和月/铁口 bias
分别位于各 run 的 `units/` 与 `candidate_metrics.json`。

60 个初筛模型 bundle 在另一个进程中恢复，按每折每个已见铁口抽取两个开发
样本核对，最大预测差异为 0.0。证据：
`local/runs/optimization-v0.3-opt08-bundle-check-r1/`。这是单目标恢复核验，
不是完整组合发布或逐样本全量恢复验收。

## 已完成：新模型时序校准与第一次特征交叉

`local/runs/optimization-v0.3-followup-r1/`，新增 49 次单目标 fit，
G0_EXECUTION_PASS / G1_FAIL_DEVELOPMENT。按 D009 登记，没有新增参数配置
或微调常数；复用轮数时校验源 bundle、内部样本、截止点、参数与 registry。
F-B 交叉重新内部选轮数；所有校准均在每个 outer origin 之前单独拟合。
源码及配置归档摘要：
`b535698b9e7542358afc71228e38407721cce61ad19380a7371fbe0a9e498920`。

|候选|J|候选−C_ref|新门槛|
|---|---:|---:|---|
|CB-CVcal|0.171389|−0.000325|FAIL：改善幅度不足 .001|
|LG-CVcal|0.172793|+0.001079|FAIL|
|CB-FB-raw|0.173054|+0.001340|FAIL|
|CB-FB-CVcal|0.172041|+0.000328|FAIL|

CB-CVcal 通过其余五项门槛，H2/H3/H4 改善，但不能据小幅改善放宽 J 门槛。
相对自身 raw，CB、LG、CB-FB 的时序校准分别改善 J 约 .001282、.001734、
.001013；这支持“该校准算法对这些新模型有用”，不等于超过强参照或平台提分。
CB-CVcal 相对 C_ref 的开发周块 J 差值区间为 [−.001310, +.001027]；
每项 1,000 次重采样有 989 次有效，仍不覆盖模型选择偏差。
完整来源、残差、分目标误差、最大误差实例、配对结果均保存在该 run，
旧实验文件未回写。保留 E16，不生成本批平台包。
另已实际审阅四个新候选各目标的最大五个误差实例：铁量均只涉及三个
唯一样本，时长涉及四至五个，不能把跨 origin 的重复实例当作新增独立样本。

## 已完成：F-C / F-D 剩余特征对照

`local/runs/optimization-v0.3-remaining-features-r1/` 新增 28 次单目标 fit，
完成两 DEV 折与 14 格网格。G0_EXECUTION_PASS / G1_FAIL_DEVELOPMENT。
源码及配置归档摘要：
`df90c7bf644bce23260231642d86ee3a8b791b4c6c6ef4e071f15a5e5d8b9b8c`。
前置核验见 REMAINING_FEATURE_REVIEW.md，不将旧 BLOCKED 历史登记回写。

|候选|J|候选−C_ref|候选−E09|新门槛|
|---|---:|---:|---:|---|
|F-C 已知开铁索引|0.173176|+0.001463|−0.000685|FAIL|
|F-D 可见 burden 事件变化|0.174270|+0.002556|+0.000408|FAIL|

F-C 比 E09 改善三个跨度，但相对 C_ref 四跨度均未改善，H3 退化约 .003264，
超过 .002 上限。F-D 未通过 DEV_LONG；不能仅凭 DEV_SHORT 的结果晋级。
F-C、F-D 相对 C_ref 的开发周块 J 差值区间分别为 [.000521,.002759]、
[.001561,.003735]，每项 1,000 次抽样有 989 次有效；不覆盖模型选择偏差。
最大五个误差实例审阅中，两个特征的铁量均只涉及三个唯一样本，
F-C/F-D 时长分别涉及三个、四个唯一样本；逐实例身份保留在本地，不公开。

两种 profile 的全部评价格子先进行了保存后完整批次恢复核验；另一个新进程
又恢复了 14 个实际 bundle，每个 origin 每个已见铁口抽取两个样本，最大差异 0。
后者只解析三列索引元数据和过程数据，不读取目标：
`local/runs/optimization-v0.3-profile-bundle-check-r1/`。
DEV_LONG 与 O202407 同截止点重复拟合，在全部 1,244 个共享样本上的预测
差异也为 0，证据为 `local/reports/optimization-v0.3-profile-repeat-check-r1.json`。
不据此声明跨硬件确定性或完整三阶段发布已完成。

四项特征中 F-C、F-B 为前两名。按 D011 运行最后一次 F-C×CatBoost 交叉，
raw/CVcal 配对，仍为原 CB08 铁量与 CB02 时长；结果见下一节。
两次交叉合计上限为 2，不因接近门槛增加实验。

## 已完成：最后一次交叉与有限搜索收口

`local/runs/optimization-v0.3-last-cross-r1/` 新增 35 次单目标 fit，
G0_EXECUTION_PASS / G1_FAIL_DEVELOPMENT。源码及配置归档摘要：
`08d21f0ec1385ce2f76c81082d729efa8e37f112bc8b324fb706be7ff4df26c8`。

|候选|J|候选−C_ref|新门槛|
|---|---:|---:|---|
|CB-FC-raw|0.172921|+0.001208|FAIL|
|CB-FC-CVcal|0.171225|−0.000488|FAIL：改善幅度不足 .001|

CB-FC-CVcal 是本轮最好新增完整算法，通过其余五项门槛，H2/H3/H4 改善。
但铁量等跨度 WMAPE 退化 .002874，时长改善 .003851，综合 J 改善仅 .000488，
不满足复杂新算法替换要求。没有据此增加目标组合、微调权重或放宽阈值。
它相对自身 raw 的 J 改善 .001696；相对 C_ref 的周块差值区间为
[−.001536,+.001277]，1,000 次中 989 次有效，不覆盖模型选择偏差。
七份校准来源均无外层重叠、无晚于 origin 的可用性违规，样本数 266–284，
未触发零回退。两候选最大五个误差实例分别只涉及三个铁量、四个时长唯一样本。

本次继续优化新增 112 次单目标 fit（49+28+35）；v0.3 成功训练运行累计
370 次。失败参照 r1 的额外拟合证据保留，不混入成功运行计数。
15 个参数配置、4 项独立特征、2 次特征×模型交叉已完成，开发搜索关闭。
没有新增候选通过全部门槛；不生成新平台包、不冻结新 challenger，保留已有 E16。
这不是认定整个 CatBoost/LightGBM 家族无效，而是本轮有限注册算法未达到替换要求。

## 运行入口与未完成项

```bash
uv sync --locked --extra dev --python 3.12
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_audit --output local/runs/<unique-audit-id>
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_run --suite reference --output local/runs/<unique-reference-id>
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_run --suite screening --output local/runs/<unique-screening-id>
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_grid --reference-run local/runs/<reference-id> --screening-run local/runs/<screening-id> --output local/runs/<unique-grid-id>
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_followup --output local/runs/<unique-followup-id>
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_remaining_features --output local/runs/<unique-feature-id> --sample-index <official-train-samples-path>
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_last_cross --output local/runs/<unique-last-cross-id> --sample-index <official-train-samples-path>
```

- OPT-07B：完成；r1 因 E09 简称未映射到旧完整候选 ID 失败，失败证据原样保留。
- OPT-08 晋级全网格及新模型自身校准比较：完成；已评价候选均未通过全部门槛。
  不将这个结论扩展为该模型家族所有算法都失败。
- OPT-09：四项独立特征、两次特征×CatBoost 交叉完成；有限开发搜索关闭。
  不新增参数、目标组合、特征或融合权重。
- OPT-10：独立授权、manifest/policy 校验、用途拒绝、一次授权消费和追加
  JSONL 账本的合成测试已实现。用户例外选择后完成了 CB-FC-CVcal development
  组合的原始输入恢复、test_a 发布和当前候选切换；**保护评分入口、final_training
  与三阶段全量发布验收仍未完成**，不能视作完整 protected lifecycle 交付。
- 没有自动 G1 晋级的 challenger；当前发布是用户明确批准的例外选择。
  保护评分/最终训练仍分别需要授权，未因替换开发发布而开放。

锁定 Python 3.12 路径已通过 105 项测试（`local/reports/pytest-optimization-v0.3-rollback-r1.xml`），
包括新进程单目标及特征 profile bundle 恢复、索引三列读取、同时间批次排除、
burden 发布延迟与重复事件，以及合成完整参照/初筛/晋级网格调度测试。
G0 测试状态与 G1 模型结论分开报告。开发周块区间只描述样本稳定性，
不取代保护报告，不覆盖多轮模型选择偏差。

这些运行来自本地开发工作树，分别记录源码摘要、配置及数据身份；后续补强的
配置一致性/晋级来源校验、缓存元数据键及 fit 记录字段没有用于回写旧产物。
正式冻结发布仍须固定最终源码并完成完整离线复现；不能把本批 G0 执行通过
当作 OPT-10 发布复现通过。
