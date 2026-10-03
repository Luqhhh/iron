# 项目文档索引

更新日期：2026-10-03。执行规则见 [AGENTS.md](../AGENTS.md)；机器状态见 [EVIDENCE_STATUS.json](../EVIDENCE_STATUS.json)。本文的分数和交付状态是该日期的登记快照，后续以状态文件为准。README 按用户要求保持不变，不作为最新队列入口。

## 当前复赛状态

| 项目 | 最新登记 |
| --- | --- |
| 平台回传最佳 | **EMA_MEAN3_FULL_Q75 = 96.3954**；用户回传，未独立平台核验 |
| 当前代表 | 固定训练seed42/1042/2042等权平均；原匹配BASE门失败与人工探索身份保持 |
| 目标 | **96.4 → 96.45 → 96.5**，尚差 0.0046 / 0.0546 / 0.1046 |
| 最新平台反馈 | **EMA_MEAN3=96.3954**（对旧Q75 +0.0034）、**ModernNCA=96.3707**（对旧Q75 −0.0213）；两项已测并移出待测队列，均为用户回传、未独立核验 |
| 优先信息问题 | 继续研究平台获益的EMA训练seed平均；固定中位数复核相对mean3两切分负，无确认或新包。**HARDTREE_GLOBAL_TIME_A20** 与 **GAUSS1_A20** 的旧Q75父包保留、暂不锁定下一名额。见[信息价值安排](platform_information_value/ALLOCATION.md) |
| 优化执行 | 96.45目标持续授权有效；EMA嵌套残差开发继续按冻结Q75参照运行，新阶段采用EMA_MEAN3参照。四名额来自用户目标说明，不根据日期、文件数或零散回传推算余额，不要求用满 |
| 替补 | **DE3_IRON_EMA_TIME_Q75_RESERVE 暂不平台测试**，等待信息量或收益更值得名额的候选 |
| 其他旧包 | 未回传不等于继续待测；Q25、旧 Q50 组合和更早包不因存在文件而自动恢复优先级 |
| 初赛历史最高 | V30A_OOB_BOTH_TARGETS = 83.3175；与复赛成绩分开，不是当前复赛参照 |

最新两项的包身份与反馈见[ModernNCA与EMA_MEAN3桌面交付](platform_information_value/DESKTOP_DELIVERY_20261003.md)。历史[N→V36探针](q75_combination_review/DELIVERY.md)、[SAM＋EMA](sam_ema_release/DELIVERY.md)及[PTaRL/EMA交付](ptarl_ema_exploration_release/DELIVERY.md)保留原范围及反馈，不生成第三个PTaRL双目标组合。分数均为用户回传，未独立核验平台凭证；本文不推断剩余额度。

新指定的[EMA_MEAN3桌面交付](ema_mean3_release/DELIVERY.md)复用原seed42，仅新增1042/2042两个全量程序；固定等权及.75替换，铁量保持Q75原字段。原配对确认门失败决定保持。

## 后续筛选与验证

**本地收益不能作为唯一指标。** 本地和平台的符号、排序、幅度、最佳权重可能不同；切分稳定性不等于独立数据泛化保证。SAM 时长本地 −0.01643、平台 +0.0031；EMA 时长本地 +0.00217、平台 +0.0168。Q75/Q100 相对 Q50 的本地两切分均负，平台均 +0.0025。

最新N→V36固定探针本地四seed均正、平均+0.006667、LCB95+0.005809，平台回传却相对Q75下降0.0024；SAM＋EMA固定人工探索平台下降0.0124；LAPLACE_FIXED_A20本地四seed均正、均值+0.011726、LCB95+0.007307，平台回传96.3891相对Q75下降0.0029；SEPARATE_MAE_IRON_A20换铁量列、本地对Q75与单目标MAE两项四seed均正，平台回传96.3867相对Q75下降0.0053。保留具体配方的反例与原门槛，不推广到整个模型家族，也不据此建立通用本地/平台转移规则。

保留完整同协议评估、当前目标参照、增量融合收益和冻结门槛；正式晋级仍要求至少四个完整 split seed、各 seed 正收益及 seed 层配对 LCB95 > 0。平台探索另行登记理由与授权，不追溯改写失败决定。不用固定偏移、放大倍数或条件列加法冒充平台预测。

原交付及其暂停记录保留；本线程按用户继续96.45目标指令完成新单高斯四seed确认及全量发布。优化不设时间预算，运行任务按 600 秒定时观察。数据保护、无泄漏、资源/数值门槛、追加式证据、独立冷推理与未修改列字符串检查仍有效。旧暂停、小时监控、时间拒绝和队列仅是历史记录。

## 当前维护入口

- [EMA三训练seed中位数复核](ema_median3/RESULTS.md)：最新mean3参照下，两个完整切分−0.000588/−0.001694；919项冻结身份、原状态及独立标量回读通过，0新拟合/确认/包，不安排本固定候选平台测试。

- [EMA组件嵌套残差开发](ema_nested_residual/PREREGISTRATION.md)：完整外层训练池内五折EMA OOF生成残差，应用到原完整T模型；固定RIDGE/GBM与25%组件修正，计划40新基础模型/80次optimizer，13项锁定工程检查通过，577项身份已冻结并[实际启动](ema_nested_residual/EXECUTION.md)。保留原残差失败结论的适用范围，不占平台名额。

- [本地筛选证据范围审计](q75_selection_scope_audit/RESULTS.md)：K32/EMA32历史增量多乘0.5、跨seed选权重复使用全部评价标签；混合OOF残差不满足整个outer隔离，不能证明整个家族无效。历史数字/决定保留，当前证据用途限制见新审计。

- [ModernNCA时长人工探索交付](modernnca_time_exploration/DELIVERY.md)：20%检索预测替换、铁量原字符串不变；45项锁定检查、两个状态独立冷审计及无标签包回读通过；用户回传96.3707，对旧Q75 −0.0213。

- [GLOBAL硬树人工探索交付](hard_tree_time_exploration/DELIVERY.md)：20%时长替换、原铁量字符串保持；选中15轮、32项锁定检查、独立冷推理与无标签封包通过。两完整开发seed小负，未正式晋级。

- [信息价值与队列核对](platform_information_value/ALLOCATION.md)：ModernNCA与EMA_MEAN3最新成绩已绑定原包并移出待测，mean3为新最佳；旧硬树与Gaussian保留但未分配。局部学习曲线/残差结果不构成家族无效或96.45不可达的证明。

- [记录库组合复核](q75_stack_review/RESULTS.md)：保留原结果；新增审计发现库归组遗漏trial身份、全标签筛列、层二与基模型验证依赖及增量单位问题，不能作为完整模型库或平台上界。
- [样本量学习曲线](q75_headroom_20261002/RESULTS.md)：控制臂与已记录 V12 预测逐位一致。铁量已饱和（训练行减半仅 +0.00067 WMAPE）；时长轻度数据受限（减半 +0.00233），但边际收益快速衰减。
- [训练日程与损失筛选](q75_schedule_screen/RESULTS.md)：22+20 次拟合、控制臂逐位一致。余弦日程一致改善铁量（`COS_MAE` 两 seed 均正、均值 +0.000289、7/10 折为正），时长在各日程/损失下均未改善。首次出现协议层（非加权层）正向信号。
- [余弦+L1 铁量四 seed 确认](q75_cosine_iron_confirmation/RESULTS.md)：单列替换四 seed 均负（均值 −0.00014），但作为增量融合成分四 seed 全正（均值 +0.0023、LCB95 +0.0008）→ 通过项目四 seed 门槛，幅度远小于目标差额。
- [选轮噪声与轨迹平均筛选](q75_selection_screen/RESULTS.md)：`INNER5`（铁量两折同向 −0.000634，时长变差）阶段 2 不成立已关闭；`SNAP5`（五个检查点预测平均）两 seed 两目标同向改善，但[四 seed 增量融合](q75_snap5_confirmation/RESULTS.md)不满足全正/LCB 门槛，配方关闭。harness 缺陷与两次预算更正均已登记。
- [残差可学习性判定](q75_residual_learnability/RESULTS.md)：原脚本所有gamma为0，但其基模型OOF依赖未完成outer隔离、基预测特征未加入、内层组数与计数有误；保留零修正观察，不能据此关闭残差或条件校准家族。
- [GRANDE 硬树试点](q75_hard_tree_pilot/RESULTS.md)：标准化/inner seed42的两折GLOBAL试点未过原门。更早V39已完成分位数正态/inner seed27001的两个完整切分；[Q75零拟合复核](q75_hard_tree_reuse/RESULTS.md)显示固定20% GLOBAL时长增量−0.001431/−0.000458，保留为另外冻结的人工信息探索，原未晋级决定不变。
- [特征增强与 mixup 筛选](q75_feature_augmentation/PREREGISTRATION.md)：显式对数/对数比/乘积特征与 mixup 在两折两目标上**全部变差**（时长 +0.0010/+0.0121）→ 筛选即关闭，未消耗确认预算。
- [TabM 超参筛选](q75_hpo_screen/RESULTS.md)：16 个单因素配置中**只有 `K32`（tabm_k 16→32）在两个目标、两个折上同向改善**；叠加余弦/L1/PLR 频率无复利，k=48/64/96 也不更好。
- [K32 四 seed 历史评估](q75_k32_confirmation/PREREGISTRATION.md)：原“时长向过门”决定保留；[新增审计](q75_selection_scope_audit/RESULTS.md)更正原权重描述性平均增量为+0.009114、LCB95+0.004422。跨seed选权复用全部评价标签，当前不作为独立确认资格。
- [K32 时长探针交付](q75_k32_release/DELIVERY.md)：`K32_TIME_A45`（时长 = 0.55×父包 + 0.45×K32，铁量列原字符串），ZIP `834adec1…`；独立回读审计通过（322 行、模板序、铁量差异 0、算术差 0）。**未写桌面、未上传**。
- [K32 邻域二次扫描](q75_hpo_neighborhood/RESULTS.md)：k=32 邻域 16 臂无一时长优于 K32；铁量 `K32_LR3`/`K32_DROP0` 两折同向但组合后效应消失 → 邻域关闭。
- [EMA×k32 四 seed 历史评估](q75_ema_k32/RESULTS.md)：原权重0.60下四seed描述性增量均正，算术更正后平均+0.012879、LCB95+0.009734；跨seed选权复用全部评价标签，不能解释为独立确认的叠加收益。原报告和决定保留。
- [EMA32 时长探针交付](q75_ema32_release/DELIVERY.md)：`EMA32_TIME_A60`（时长 = 0.40×父包 + 0.60×EMA32，铁量列原字符串），ZIP `97868fd3…`；独立回读审计与桌面回读均通过。**平台回传 96.3857，相对 Q75=96.3920 为 −0.0063**：原权重本地描述性平均+0.012879（旧报告+0.0064的算术已更正），选权验证范围有限。`K32_TIME_A45` 同机制、未上传，降级保留。

- [独立MAE铁量三训练seed均值结果](q75_separate_mae_iron_mean3/RESULTS.md)：40新optimizer/60新旧状态审计通过；两seed对Q75均正却均低于就绪单seed，不确认或新增包。
- [原V10 MAE零训练复核](q75_retained_v10_mae/RESULTS.md)：相对Q75两seed为正，但均弱于就绪Laplace和同骨干MSE；0新optimizer，不追加候选。仅保留预测来源/算术审计，原checkpoint缺失不冒称模型冷复现。

- [独立MAE铁量全量交付](q75_separate_mae_iron_release/DELIVERY.md)：2optimizer/2状态、117轮全量重训，独立冷/NumPy/无标签预测/322行包回读及五个真实exit0通过；只改铁量，时长原字符串保持；按用户指令写桌面`submission-SEPARATE-MAE-IRON-A20-20261002`，固定时长列差异0；平台回传96.3867（用户回传、未独立核验），相对Q75 −0.0053。
- [独立MAE铁量四seed确认](q75_separate_mae_iron/RESULTS.md)：对Q75均值+.006061/LCB+.003411，对单目标MAE均值+.004275/LCB+.001841，两项四seed均正；观察后控制线索的单独前瞻确认。
- [共享MAE完整开发](q75_joint_mae/RESULTS.md)：共享铁量一个负seed、共享时长弱于原Laplace，原准入门失败；不追溯更换候选。
- [两个名额暂定安排](platform_slot_review/ALLOCATION.md)：两名额均已消耗；LAPLACE_FIXED_A20回传96.3891（−0.0029）、SEPARATE_MAE_IRON_A20回传96.3867（−0.0053），两条不同目标列的本地正收益均未转移；Q75仍为最佳，不推断剩余额度。

- [Laplace三seed四切分确认](q75_laplace_mean3_confirmation/RESULTS.md)：对Q75均值+.012413/LCB+.008464，正式门过；对原单seed均值+.000687但LCB−.000347、一个负seed，新包门失败，0全量/包。40optimizer及60新旧状态、四真实exit0闭合。

- [Laplace三训练seed完整开发](q75_laplace_mean3/RESULTS.md)：40新optimizer/60新旧冷状态和四实际exit0通过；对Q75+.010263/+.013691，对原单seed+.001310/+.001520，取得单独确认准入，尚未正式晋级。

- [固定点损失完整结果](q75_fixed_point_losses/RESULTS.md)：30估计器/60科学optimizer/60冷状态和四真实exit0通过；时长固定Gaussian弱于就绪Laplace，铁量两臂各一负，没有新确认候选。
- [两名额无标签预测比较](platform_slot_review/RESULTS.md)：三包预测均有差别，不能据同架构视为重复或据差异大小推断收益；名额继续保留，原表头检查失败及独立恢复均记录。

- [固定尺度Laplace全量交付](q75_laplace_release/DELIVERY.md)：单一20%时长融合包，2正式optimizer/2状态、无标签新进程冷推理、322行回读与5个真实exit0通过；9项零optimizer发布检查通过，发布当时0桌面/上传；随后按用户指令复制到桌面`submission-LAPLACE-FIXED-A20-20261002`并独立回读通过；平台回传96.3891（用户回传、未独立核验），相对Q75 −0.0029，本地四seed正收益未转移。

- [固定尺度Laplace四seed确认](q75_laplace_confirmation/RESULTS.md)：相对Q75四seed均正、均值+.011726/LCB95+.007307；相对Gaussian均值+.003321但LCB95为负，不称统计独胜。20次新optimizer/20状态及真实终态闭合，后续独立全量包已完成。

- [Laplace时长网络完整开发](q75_laplace_time/RESULTS.md)：两臂共40正式optimizer/40状态闭合；FIXED相对Q75两seed+.008953/+.012171，平均略优于Gaussian，进入单独确认；SCALE两次弱于FIXED，未通过额外Gaussian比较门。平台队列不变。

- [单高斯时长全量交付](q75_gaussian_release/DELIVERY.md)：一个固定20%融合包，2正式optimizer/2状态、独立无标签冷推理、322行包回读与5个实际exit0通过；22项最终零optimizer检查通过。准备测试超额及首轮账本准入失败完整保留，未桌面复制或上传。
- [单高斯时长四seed确认](q75_gaussian_confirmation/RESULTS.md)：四seed均正，均值+.008405、LCB95+.005023，正式晋级；新增20optimizer/20状态冷审计和独立评分通过。平台未测。
- [单高斯时长对照相对Q75](q75_gaussian_time/RESULTS.md)：两个完整开发切分+.009442/+.010147，20个原状态冷回读与独立评分通过，0新拟合；这是原零拟合开发阶段；后续四seed确认与发布已在独立阶段完成。
- [SWA原开发交接复核](swa_original_intake/RESULTS.md)：1626份载荷、44个新进程冷状态和两个完整开发切分复算通过，0新拟合；队友已发布的四seed确认未过正式门槛，待确认原始包再做本机核验，不重复确认预算。后续四条SWA路线的已发布结果一并登记，避免重复研究。
- [联合EMA时长零拟合复查](q75_joint_ema_time/RESULTS.md)：原缓存时长列按冻结.20组件转移量对Q75两个完整切分均负，未进入确认；60个冷状态及独立标量评分通过，0新包。
- [EMA_MEAN3桌面交付](ema_mean3_release/DELIVERY.md)及[冻结协议](ema_mean3_release/PREREGISTRATION.md)：原42全量EMA复用，1042/2042两个新增全量程序、4次optimizer、4个新状态、一个ZIP及一份桌面复制均闭合；1548项锁定检查、独立冷/标量/包/桌面审计和实际exit0通过。G1平台未测人工探索，原确认门失败保持，交付后继续暂停。
- [SAM＋EMA固定人工探索交付](sam_ema_release/DELIVERY.md)：一个全量程序/两次optimizer/一个ZIP、原生与独立冷审计、包回读及实际exit0通过；用户回传96.3796，相对Q75−0.0124，未正式晋级。优化保持暂停。
- [96.45本轮实施顺序](q75_9645_execution/PLAN.md)与[完整结果](q75_9645_execution/RESULTS.md)：首批工作及初始化、更新步数、组件校准均已闭合；后续正式配对门未过；另行冻结的SAM＋EMA人工探索已交付，现按用户要求暂停。
- [含EMA零拟合组合结果](q75_combination_review/RESULTS.md)：六项、四个完整已有seed；开发首选N_TO_V36005均值+.006667、LCB95+.005809，[单一探索探针](q75_combination_review/DELIVERY.md)已通过独立回读，用户回传96.3896，相对Q75−0.0024。
- [按最终融合效果选轮协议](ema_fusion_selection/PREREGISTRATION.md)与[完整配对结果](ema_fusion_selection/RESULTS.md)：30optimizer/40状态及独立终态G0通过；选轮独立收益一正一负，不进入确认。
- [EMA组件同模型校准结果](ema_component_calibration/RESULTS.md)：80次校正估计、20个冷query见证及两个完整OOF已闭合；原报告错误exit 1保留，另目录零拟合恢复及独立计分exit 0通过。全局/压差相对Q75均两切分负、相对同F模型均一正一负，无确认或新包；前序初始化、更新步数匹配亦未过各自冻结配对门。
- [EMA双dropout一致性完整配对结果](ema_dropout_consistency/RESULTS.md)：20估计器/40optimizer、60个新旧冷状态与实际exit 0闭合；固定lambda=.5对Q75及双前向控制的两个完整切分均负，未过确认门。24项定向检查与合成工程拟合另计，原旁路监控失败证据保留；无新包或平台排程。
- [EMA时长width512完整结果](ema_width512/RESULTS.md)：10估计器/20optimizer、40个新旧冷状态、490份冻结文件与实际exit0闭合，G0通过；对Q75两完整切分−.009800118/−.011806141，固定配方未过确认门，0新平台包。20项定向检查与1次合成工程optimizer另计。
- [SAM＋EMA完整开发结果](sam_ema_time/RESULTS.md)：10估计器/20optimizer、60个新旧冷状态、545份冻结文件与实际exit0闭合。G0通过；对Q75两切分−.024537241/−.021519181，确认门失败；对匹配SAM均改善。另行冻结的人工探索已回传96.3796，原科学分类与失败决定保持。
- [实施报告](report.md)、[任务范围](task_contract.md)、[发布身份](release_identity.md)、[提交与反馈记录](submission_log.md)。
- [候选分类及正式晋级边界](candidate_tiers.md)、[数据契约范围](data_contract.md)、[待确认语义与平台口径](rule_questions.md)。
- [四项本地/平台诊断反馈](local_platform_diagnostic_release/DELIVERY.md)：四项均已回传，SAM 时长反转，EMA 时长获益。
- [EMA 稀疏权重反馈与替补安排](ema_time_followup/DELIVERY.md)：Q75/Q100 已回传，DE3＋Q75 仅作替补。
- [PTaRL 相对 Q75 的零拟合诊断](ema_evaluation_diagnostics/RESULTS.md)：两切分描述性证据，不预测平台分数。
- [PTaRL 时长 / EMA 铁量全量交付](ptarl_ema_exploration_release/DELIVERY.md)：PTaRL时长用户回传96.378，EMA铁量96.3816，均低于Q75=96.3920。
- [Q75误差地图](q75_error_relocation/RESULTS.md)及[压差时长校准完整开发](q75_error_relocation/CALIBRATION_RESULTS.md)：G0通过，压差两切分均负、全局一负一正，无确认候选及新平台包。
- [EMA时长平均跨度完整结果](ema_average_span/RESULTS.md)：20估计器/40optimizer和独立终态审计通过；短跨度两seed均正、平均+.001788，按冻结规则进入确认准备，自动分类仍exploration；长跨度两seed均负，没有新平台包。
- [EMA短跨度确认工程准备](ema_span_confirmation/PREPARATION.md)：控制器、原参照捕获、独立冷审计及预算冻结已完成；81项定向、1439项全库回归与实际启动准入通过，批次已完成，四切分及终态结果见下文。
- [SHORT_SPAN四切分确认协议](ema_span_confirmation/PREREGISTRATION.md)：控制器、原EMA状态桥、独立原参照组合与seed算术通过81项定向及1439项全库检查；静态协议已声明两个额外seed，实际运行准入及初始进程身份已通过；完整额外seed及独立终态已通过审计，无新提交包。
- [SHORT_SPAN四切分确认结果](ema_span_confirmation/RESULTS.md)：10单位/440状态及实际exit0通过；四个完整seed均正，平均+.001567、LCB95+.001047，原开发exploration保持；后续另获具体全量授权并已交付，平台反馈见下文。
- [SHORT_SPAN全量本地交付](ema_short_release/DELIVERY.md)：明确授权的1全量程序/2Optimizer/1本地ZIP及独立冷推理、包回读、实际exit0全部通过；用户回传96.3911，低于Q75；桌面写入/助手上传0。
- [SHORT_SPAN全量发布准备](ema_short_release/PREPARATION.md)：复用原训练/审计接口的单候选脚本通过39项定向及1464项全库检查，原EMA冷回放与Q75重建通过；原准备清单保持未准入；用户确认后的独立全量运行已完成，见上述交付，不继续扩展工程代码。
- [ModernNCA默认MSE完整结果](modernnca_q75_preparation/RESULTS.md)：20配对/80状态/40optimizer及终态审计通过，峰值823MiB；铁量两seed均负，时长平均+.000993、两seed均正但原门未过，无确认/新包。后续常规优化按用户持续授权执行。
- [ModernNCA MAE完整结果](modernnca_mae/RESULTS.md)：20配对/80状态/40optimizer及独立冷/标量/MSE对照闭合，峰值797MiB；铁量平均−.006586、时长−.005028，两目标两seed均负，无确认/新包，原开发门及MSE决定保持。
- [ModernNCA零拟合中位数头](modernnca_median/RESULTS.md)：40个原MSE refit状态复用，NumPy/Torch及标量闭合，0新拟合；两目标两seed均负，无确认/新包，保留该具体聚合方式失败。
- [Q75后续特征支持描述](q75_feature_support/RESULTS.md)：零拟合、无目标读取；未见大范围单列外推，不证明条件同分布或平台排序。
- [DE3 铁量历史交付与回传](de3_user_release/DELIVERY.md)：96.3749，相对 V32 +0.0022。

`current_status` 是项目摘要，`round2_current_platform_best` 是平台最佳，`round2_current_candidate_queue` 是当前候选安排。旧初赛摘要和旧队列完整保存在各自的 `history_before_documentation_refresh_20261001`；其他阶段条目保持原运行时含义。

## 历史文档阅读范围

`optimization_v*/`、编号 `round2_v*/`、`round2_next_phase/`、`round2_final_top5/`、`round2_slots_20260930/`、`review/` 及 `md/` 记录阶段当时的计划、参照、结果或交付，不是实时队列。预登记、实验配置、模型配方和失败门槛保持原样；不要照旧文恢复暂停、提交顺序、旧预算或旧“当前最佳”。原始历史内容和身份保留，不以文档更新重新授予拟合或发布资格。

- [Top5 历史反馈](round2_final_top5/FEEDBACK.md)：AJ3 的首选身份限于 2026-09-23 批次。
- [V3.4 历史得分转移与门槛](round2_v3_4/SCORE_TRANSFER_AND_NEXT_TARGET.md)：96.25 是原阶段门槛，不是所有探索的统一否决线。
- [V5 结果及判读更正](round2_v5/RESULTS.md)：0.0005 是单次扰动效应，不是平台分辨率；方向/排序保证已撤回。
- [V6 历史线搜索及更正](round2_v6/RESULTS.md)：固定放大推出的“96.35 不可达”已作废；旧五包顺序不再作为当前队列。
- [初赛 V1 发布入口](optimization_v0_8/CURRENT_RELEASE.md)、[R2 历史回退](optimization_v0_4/CURRENT_RELEASE.md)、[编号与身份修复](round2_round_numbering.md)。

## 初赛及基线阶段归档


| 阶段 | 冻结结果 | 阅读口径 |
| --- | --- | --- |
| v0.32 | [同铁口优先的 OOB 响应条件化](optimization_v0_32/RESULTS.md) | 0 fit；固定 v0.29 members 与每树等权，仅在原集合内同铁口优先；A/B 相对 V30A 的历史 ΔJ +0.00006051/-0.00018334；G0 PASS；平台回传 83.2764/83.0910，预算 2/2，两项关闭并保留 V30A，agent 上传 0 |
| v0.31 | [两个目标隔离的 OOB 叶响应汇总实验](optimization_v0_31/RESULTS.md) | 0 fit；14 份 v0.29 OOB 附件复用；整数 occurrence 质量池化；G0 PASS；用户回传 I=83.3123、T=83.3116，均低于 V30A=83.3175，两项关闭并保留当前最高 V30A；平台预算 2/2、agent 上传 0 |
| v0.30 | [双目标 OOB 收益组合与时长森林固定扩容](optimization_v0_30/RESULTS.md) | G0 PASS；7 次追加 fit/5,376 棵新树（2 个已完成拟合显式恢复、0 re-fit）；A 列组合恒等式残差 ≤5.55e-17；平台用户回传 A=83.3175（晋级，与加性推算一致）、B=83.2654（关闭），预算 2/2、agent 上传 0 |
| v0.29 | [冻结森林 OOB 叶响应双实验](optimization_v0_29/RESULTS.md) | 0 fit；14 份 OOB 附件；G0 PASS；平台回传 A/B=83.2970/83.3141，预算 2/2，B 晋级为当前最高；离线排序不回写 |
| v0.28 | [固定等权、双目标隔离集成](optimization_v0_28/RESULTS.md) | 0 fit；A 平均 V26A/V27I 铁量，B 平均 V26A/V21 时长；G0 PASS，平台回传 83.2936/83.2604，预算 2/2，A 晋级、B 关闭 |
| v0.27 | [铁量 QRF 与时长叶内 recency](optimization_v0_27/RESULTS.md) | G0 PASS；A/B 用户回传 83.2480/83.2710，均低于 V26A=83.2828，固定候选关闭 |
| v0.26 | [时长森林分区双实验](optimization_v0_26/RESULTS.md) | G0 PASS；V26A 用户回传 83.2828 并晋级为当前最高，V26B 83.0240 关闭 |
| v0.25 | [历史基准中心化双目标实验](optimization_v0_25/RESULTS.md) | G0 PASS；7 centered CatBoost + 7 signed QRF、0 新预处理器/校准；A 相对 V21 的 J 退化 +0.00039378，B 改善 -0.00221680 但仍不及 V1；平台用户回传 83.1516/83.0117，预算 2/2，两项关闭并保留 V21 |
| v0.23 | [恢复、统一参照与复赛完整算法预演](optimization_v0_23/RESULTS.md) | 0 fit；V21 原 ZIP/payload 恢复；六个历史 replay 逐字节复验；四算法统一六位 scorecard；M-only 数值对照优于 V22；旧 test_b 的 V1/V21/V22 双进程冷推理一致，正式复赛身份仍待核验 |
| v0.24 | [双目标隔离变料历史特征实验](optimization_v0_24/RESULTS.md) | 7 CatBoost + 7 QRF/preprocessor；A/B 相对 V21_REPLAY 的 ΔJ 均小幅退化；平台用户回传 83.1902 / 83.2288，均未超过 V21，固定候选关闭 |
| v0.22 | [V22 因果 H2 QRF 支持度收缩](optimization_v0_22/RESULTS.md) | 2 个 warmup QRF、七份真实 H2 bank、12 个开发 lambda；全部预注册门槛通过；后续 test_a 包冷验通过，用户回传 83.1166 后关闭该候选，V21/V10 保留 |
| v0.15 | [OPT-32–33](optimization_v0_15/RESULTS.md) | 固定QRF时长分支6 forest/6 preprocessor、1536树；根364/worker25测试、独立冷审计；十项质量失败、关闭V8，无新包；[工程恢复](optimization_v0_15/ENGINEERING_REPAIR.md)、[维护记录](optimization_v0_15/MAINTENANCE_20260912.md) |
| v0.14 | [OPT-30–31](optimization_v0_14/RESULTS.md) | 原基础模型 0 fit；V7/D1 6+6 时长 LAD；342 tests、独立冷审计；FAIL_CLOSE_V7_RETAIN_V1；[维护观察](optimization_v0_14/MAINTENANCE_20260912.md) |
| v0.13 | [OPT-27–29](optimization_v0_13/RESULTS.md) | 0 fit；E/J/贡献重建、旧 A/B stage 预演通过；286 tests；[维护观察](optimization_v0_13/MAINTENANCE_20260912.md)、[正式接入待办](optimization_v0_13/SECOND_ROUND_PROTOCOL.md)；随后按用户指令推送 2db6d5f，run 34687452551 成功 |
| v0.12 | [OPT-25–26](optimization_v0_12/RESULTS.md) | 三个 recency60 候选完整质量门槛失败；255 tests，G0 冷审计通过；16+12 fits，无新包；随后推送 62c62cc、CI 成功；[发布暂停](optimization_v0_12/DATA_PUBLICATION_REVIEW.md) |
| v0.11 | [OPT-24](optimization_v0_11/RESULTS.md) | V5 四项质量门槛失败；235 tests，G0 修正后冷审计通过；8+6 fits，无新包；已推送 |
| v0.10 | [OPT-23](optimization_v0_10/RESULTS.md) | V4 失败；221 tests，G0 冷审计通过；无新包 |
| v0.9 | [OPT-21/22](optimization_v0_9/RESULTS.md) | ratio 扩展失败；“v0.10 尚未训练”仅描述该轮收口时 |
| v0.8 | [OPT-20](optimization_v0_8/RESULTS.md) | 开发交付时 R2 仍 active；随后 V1 平台回传 83.0319，见当前发布页 |
| v0.7 | [OPT-19](optimization_v0_7/RESULTS.md) | pseudo-history 失败，当时恢复 R2 |
| v0.6 | [S1 / OPT-18](optimization_v0_6/RESULTS.md) | S1 平台失败；“桌面仍 S1”是回传记录当时，后来已替换 |
| v0.5 | [OPT-14/15](optimization_v0_5/RESULTS.md) | 旧组合研究结束，R2 保留；B/C 冷检查不等于 V1 的 B/C 验收 |
| v0.4 | [R2](optimization_v0_4/RESULTS.md) | 当时 R2 晋级，当前回退 |
| v0.3/r2 | [生命周期与回退](optimization_v0_3/OPT10_EXECUTION_R2.md) | November 从此已消费；旧 E16 发布页仅供历史回退 |
| v0.3 | [研究汇总](optimization_v0_3/RESULTS_SUMMARY.md) | 当时的候选、未完成项和平台状态 |
| v0.2 | [OPT-01–06](optimization_v0_2/RESULTS_SUMMARY.md) | E09/E12/E16 的“incumbent”按当时解释 |
| baseline | [冻结报告](review/FREEZE_REPORT.md) | 历史测试数、DEV_LONG 失败和当时未消费状态 |

- [强组件组合、选择与重训：闭合状态及后续配对协议](ema_retraining_validation/STATUS.md)
