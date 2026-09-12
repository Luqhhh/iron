# optimization-v0.13：复赛基线适配与零训练误差诊断

2026-09-12 注册。从 `optimization-v0.12@62c62cc92efa660674877254c6ae8dc2bceaba83` 建立独立本地分支，不从 main 开始。保持 V1 活动发布和 R2 回退，V2–V6 已关闭；不改历史 FAIL、模型、系数、cutoff、历史、源契约、旧脚本或旧证据。

预算：新模型 fit 0、新 LAD/校准 fit 0、新候选 0、新 challenger ZIP 0、平台上传 0、旧包覆盖 0。仅私有环境读取现成匹配哈希的产物；公开数据历史处置完成前不推送。本轮不自动更改仓库可见性、force-push 清史或销毁证据。

## OPT-27：冻结误差账本

使用原 v8 与 v12 已消费开发来源、完成回执、manifest、模型历史和原访问账本。读取逐样本标签前冻结 configs/optimization_v0_13/audit.yaml、access_scope.yaml、本轮脚本与全部依赖源码、输入/旧证据摘要、输出目录和 protection.yaml；追加新的 local 访问账本。缺少产物时列出路径及预期摘要，BLOCKED_MISSING_SOURCE，不补训。

逐 origin × horizon × target 重算 N、绝对误差和、真实目标和、WMAPE、signed mean 和 median residual，完整精度重建 E/J，误差容差 1e-12（不采用打印表 2e-10 容差）。18-cell 每条预测每目标 J 贡献为 abs_error/(8*n_h*unit_target_sum)，n_h 为该 horizon 原 origin 数。全部贡献重建 J，候选与 V1 配对差重建 Delta J；DEV 单独保存不加入 J。

每个 unit 内 sample_id 唯一；同一 sample 跨 origin 的预测全部保留，同时报告唯一铁次数和预测 exposure 数，不把重复当独立失败。分 target、真实日历月、spout、horizon 保存误差/分母及 J 贡献。固定 highest 1%/5%/10% 分析同时报告 unit 绝对误差排序和 unique sample 汇总 J 贡献排序，ceil 取样数，按 ID 确定 ties。

固定分箱：缺失比例 0/.01/.05/.20/1；历史年龄小时 0/1.5/6/24/72/168；stale flag 0/1。左闭右开，独立 MISSING 桶和无穷末端；不搜索边界。使用原逐样本 as-of builder 和原 schema 检查缺失/stale/历史年龄，不创建预测特征。沿用原 source_contract 和 cutoff。

实际读取原 component export 的 E09 OR direct、原 R2 base、原 V1 corrected，与已关闭 v12 direct/base/corrected 配对比较；无 direct 时明确缺失，不由最终分数猜 LAD/rate 原因。残差中位数只计算不应用校准；raw/base 只诊断不登记候选。不删除极端样本，不判断标签错误，不定义测试路由。

交付 error_budget.json、target_month_spout.csv、direct_structural_diagnostic.json、failure_mode_evidence.md、逐预测贡献/唯一样本贡献及合法来源证明，均 local/。failure_mode_evidence 区分观察事实、解释假设、缺失证据，允许不能区分。

## OPT-28：stage-aware V1 推理

新增薄 wrapper，显式 stage、bundle、data-config、output，支持 test_a/test_b；test_c 仅后续 smoke。配置只允许 schema_version=1、当前 stage 样本路径、operation_hourly/burden_change/data_dictionary，拒绝训练标签、外部历史与其他 stage。恢复历史只来自原 bundle。

预测参考时刻不得早于实际原 cutoff，sample_id 不得与任何保存历史相交。原 StructuralPredictor、原 v4 stage comparator 与旧 v8 test_a 入口不修改。逐样本 source_contract 校验不绕过；源变化标记 BLOCKED_SOURCE_CONTRACT_CHANGE，不自动授权迁移。

旧 test_a wrapper 与旧 v8 逐样本 exact equality；旧 test_b 全量、反序、127 行分块、子集及单样本按 ID exact equality；R2 comparator 维持旧 delta <1e-8。禁止所有模型 fit 与 LAD/校准 fit，计数记录。核验模型/历史/系数/原 ZIP 哈希，保存实际 schema、缺失/stale/fallback、源文件上界及逐样本时间截断说明。

仅输出内部 CSV，不创建 ZIP。engineering_valid、official_data_identity_verified、quality_evaluated、platform_verified 分开；现阶段为 PREVIEW_ENGINEERING_ONLY，后三项 false。正式复赛数据开放后新建 manifest 核验 ID、字段、时间范围、来源身份与时钟语义；不能把旧 test_b 同名文件当正式新包。

## OPT-29：复赛开发协议及维护观察

原 cutoff 2024-12-01 01:44+08 的 January 旧 preview 对应 H2，正式数据未核验则映射仅暂定。将来新实验需先冻结阶段主指标、双目标防退化、旧 J 可比记录、OOF、预算和失败处理；本轮不设为 V6T 定制的新阈值、不训练、不 final fit、不封包。

启动下一轮需：误差账本有数值支持的具体缺口；输入/报送时点足以实施干预；新版本有身份；候选/对照/主指标/门槛/预算先冻结；一次只引入一个主要变化。本轮不预定下一模型或隐藏训练额度。

新增 2026-09-12 维护观察，记录 62c62cc commit time、已观察远端时间、Actions run 创建/完成/查询时间；push 的精确时刻若无事件证据不造值。更新 README 和 EVIDENCE_STATUS.current_status，旧 v12 RESULTS、receipt 与已绑定文件原文不回写。

独立查找 V1 初赛有效提交回执，核对原包摘要和平台记录。用户回传 83.0319 不等于平台有效提交或复赛资格证明；若动态平台登录信息不可获得，具体记录搜索范围和限制。保留原 V1 包，禁止覆盖。

## 官方口径与来源

[2026-09-11 复赛通知](https://www.aicomp.cn/notice/notice-3/5248.html)：初赛只用于复赛资格，复赛每日最多 5 次取最高分、算分延迟，高炉复赛基准线 80。报名截止 09-20 19:00；初赛结果截止 09-20 20:00；复赛数据开放 09-21 11:00；结果提交 09-22 09:00 至 10-05 20:00；代码截止 10-07 23:59。按官网时间文字，报名系统为执行时间权威；通知未单独标时区，不推断初赛也取最高分。

[赛题规则](https://www.aicomp.cn/tracks/tracks-6/4177.html)：原 December/January 测试范围与 reference-time 限制，数据仅限赛事使用、禁止公开传播。stage 中的 sample_id/pred_tap_iron/pred_tap_time_len 是结果三列，正式文件 result.csv；本轮内部证据不用于上传。

固定代码证据来自 `62c62cc` 的 v12 RESULTS/config、v8/v4 cold scripts 与 StructuralPredictor。最新远端 Actions `34685572032` 和 repository metadata 查询单独留存；不把旧工程/CI PASS 变成新模型质量 PASS。
