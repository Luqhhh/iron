# EMA 短跨度四切分确认：工程准备（2026-10-01）

开发批次已完整通过 G0，SHORT_SPAN 相对 Q75 在 seed42/3407 为 +.002189932 / +.001385540，按[原预登记](../ema_average_span/PREREGISTRATION.md)获得下一阶段的确认准备资格。自动分类仍 exploration，四切分正式晋级尚未通过；详见[完整开发结果](../ema_average_span/RESULTS.md)。本文件是工程准备记录，**不是科学预算冻结或执行准入**。

独立工作树为 `local/worktrees/ema-span-confirmation`，分支 `codex/ema-span-confirmation`，基于已发布开发闭合提交3b6c280。使用已核验的锁定 Python3.12 CPU环境，不同步依赖。当前平台参照仍为用户回传 Q75=96.3920（未独立核验），本线程目标96.45；实际下一阶段冻结前仍须重新读取最新登记参照。

## 匹配参照范围

拟采用两个额外 seed271828/314159，各五个完整fold。它们尚未作为本阶段科学配置冻结，也未启动拟合。已登记工作树中的原强组件缓存只含开发seed42/3407，多个路径别名指向同一物理缓存，不能算额外seed或额外模型证据。

保持原 Q75 构造和 .75 权重：每个 seed 内 `Q75=B+.75*(old_EMA−V7)`，短跨度为 `Q75+.75*(SHORT−old_EMA)`；铁量使用匹配参照的同 seed 列。原 beta=.99、短跨度 beta=.9801；网络、损失、优化器、预处理、训练 seed42、独立内层选轮与全外层训练池 fresh-refit均沿用开发方案，不据条件铁口描述改配方。跨seed只汇总指标，不平均预测向量。

仅初始化工厂的目录核查使用禁止进入 fit/_train 的拦截器，实际拟合尝试0：

| 每个额外外层fold的参照工厂 | 固定管线实例数 |
| --- | ---: |
| 历史 L1 base | 21 |
| A专家（两目标共计） | 4 |
| B36专家（两目标共计） | 4 |
| N0048 / V12 joint / V7 periodic | 3 |
| 合计 | 32 |

因此拟定范围为10次匹配参照工厂调用、320条参照管线实例，另有10个原 beta 控制与10个短跨度估计器，后两者共40次 optimizer。**320是管线实例数，不是所有内部solver/optimizer调用数**；历史早停包装器还包含内层选择和新refit，下述新增账本、构造器预算草案及捕获接线已核查；真实运行身份、科学预算及控制器仍须在启动前闭合，不能用目录或合成检查代替。

目录收据仅存本工作树 `local/research/confirmation-catalog-r1/catalogue.json`，SHA为`c1da70f011c2dcd5b1815eeacf22f184e56bf36740551250692a5e67e09176ad`。包实际从本工作树src导入，torch=2.14.0+cpu、tabm=0.0.3、numpy=2.2.6、scikit-learn=1.8.0；OPENBLAS/OMP/MKL/NUMEXPR及torch线程各1。没有读取新的外层结果或生成新确认预测。

## 必须补齐的 G0

现有原参照工厂拟合后返回预测数组，丢弃模型对象。直接调用后只存预测，不能满足新参照的独立冷推理和模型身份复核。下一步在本隔离分支实现回收适配器，保留原训练方法和冻结配方：

- 给每条原管线、内部选轮/refit和数值优化过程登记开始/完成/失败及source directory、split seed、trial id身份；失败开始永久占用计数，不自动重试。
- 保存真实模型、训练内预处理/目标尺度及分区/选轮元数据，独立新进程仅凭保存状态和无目标query完成原参照冷推理；不能重拟合来冒充冷回读。
- 复核固定权重组合、全部训练/query ID、原回收脚本/私有清单/源码/运行环境及已有两个开发seed的缓存身份。原工厂已有的固定后处理保留，不为新外推临时加裁剪。
- 对原 beta 控制和短跨度分别执行训练内选轮与fresh-refit；审计新保存状态、完整同seed OOF和独立增量/LCB算术。四seed正式门仍为各seed正及seed层配对LCB95>0，fold仅描述，原阶段分类和失败记录不回写。
- 完成锁定Python3.12必要检查、真实运行预算/输入/源码冻结和单worker资源准入后，才进入科学执行。600秒观察、原内存/数值门和无优化时间预算约定保留。

当前 G0 已完成目录、环境、保存/冷回读基础设施、原选择器观察层、原生次数账本、预算草案及原工厂捕获和批次冷审计实现。新增接线的62项定向与1420项全库回归通过；真实32条异构管线运行审计、科学预算冻结和确认控制器尚未完成；G1没有新观测。新确认拟合、全2754行拟合、包、桌面写入、助手上传全部0。此准备不自动授权全量发布，也不将SHORT加入当前平台测试队列。EMA铁量反馈、DE3+Q75替补及ModernNCA工程范围保持各自原状态。

## 保存状态与独立冷回读基础设施

隔离分支新增 `bf_tap_r2.ema_reference_artifacts`：保存调用方已获得的预测和真实对象，不再次 warm predict、不拟合、不删除训练属性；每次保存使用独占新目录，失败记录保留。模型原拟合来源与新审计文件存放位置分别登记，复制旧模型不能算新拟合。外部持有的完整收据摘要绑定模型、无目标 query、预测、训练/query ID、实际模型类源码及冻结源码；独立进程拒绝 fit/_train 和 CSV 读取，并比较完整批次、反序、分块和单行输出。

锁定 Python3.12 CPU 路径的17项定向检查通过，覆盖独立进程、二维 joint 输出、文件/收据篡改、query ID 与训练分区、非有限容差、实际加载类源码、禁止冷拟合/CSV读取、分块依赖、旧拟合来源保留，以及失败与已完成目录禁止覆盖。定向收据为本工作树 `local/research/artifacts-engineering-r1/receipt.json`；这不是完整确认阶段准入。

真实对象零拟合检查使用原强组件 `development-r2/tap_time_len-EMA-s42-f0/refit.pt`，原 unit 收据及模型摘要在读取前核验。只读训练样本表 ID/spout 列并对齐原 fold，用其摘要核实原训练分区；query 是322行无目标官方测试特征。保存后由独立进程仅加载对象与 query 完成推理：完整批次差异0，反序最大差异0.000003248，分块0.000006496，单行0；冻结完整批次容差0、行序/分块容差0.0005。新拟合、新确认seed、全量拟合、提交包、桌面写入及助手上传均0。它验证一个既有 TabM 控制对象，不代替32条新参照管线的捕获与冷回读。

首轮 `native-artifact-smoke-r1` 在加载模型前因检查脚本使用 fold 文件行序而触发原训练 ID 摘要不一致，失败记录保留。第二轮按官方训练样本表对齐 ID，摘要与原记录一致，通过收据位于 `local/research/native-artifact-smoke-r2/receipt.json`，SHA为 `9935c895db4faad053fd6863aa162add4be98c895f8fb9f2ca28cf3d24f8b17d`；模型 witness 外部收据SHA为 `eb664763877db8e9bde42e03c462a08c37022b5e6ca4309470abc04d71e40889`。没有重跑科学训练或覆盖第一轮证据。

## 原生调用账本及预算草案

新增 `bf_tap_r2.ema_reference_ledger` 和 `ema_reference_plan`。账本给每条管线单独冻结预算和外层训练/query ID，内部 fit/calibration 分区必须局限外层训练池；source directory与split seed/fold/trial逐次登记。原生入口保持原返回值，开始记录先于数值调用，超额尝试不会进入原方法；失败或中断永久占用位置。原实现即使捕获某个内部异常，闭合账本仍拒绝将该管线判为通过。方法包装器退出时恢复原方法及原继承关系，各worker必须建立自己的账本，不能跨线程/PID共用。EBM还核查主效应/交互的阶段次数，不能用相同总数掩盖阶段错配。

实际原配方构造器在禁止进入fit/_train的守卫下核查32条管线、拟议seed271828/314159的10个无目标外层分区及铁口计数。各池每个铁口至少1094行，均满足原收缩专家200行门槛，拟议每池有两个local加一个global模型。没有读取训练目标来选择切分或预算。当前已安装InterpretML的六条RMSE配方各有4个主效应和4个交互boost入口；RMSE的最终截距校正是解析计算，不增设另一个boost入口。

| 跟踪入口 | 每个参照工厂的预算草案 | 10个拟议参照工厂 |
| --- | ---: | ---: |
| CatBoost fit | 29 | 290 |
| EBM fit API | 6 | 60 |
| EBM内部boost（主效应/交互） | 48（24/24） | 480（240/240） |
| sklearn joint MLP L-BFGS fit | 1 | 10 |
| torch optimizer初始化 | 6 | 60 |

原beta控制与短跨度另有40次torch optimizer初始化，整批拟议torch总数100。EBM API和其内部boost分层记录，不重复当作两组独立模型，也不将这些异质次数相加当作计算成本或时间预算。目录与配方草案不等于实际完成的拟合；任一真实调用/阶段不符将失败闭合，不自动调整预算或重试。

私有草案及两个无目标分区证据为本工作树 `local/research/native-budget-proposal-r1/`：总收据SHA `bf8a0db19757701675ed2de01b9c0790a40ff9cb46e9da837ab2017740c6aa5a`，32配方草案SHA `4469d7096768c69f3b8e60522f0febc832fa65991ed1a7288ed367f4bed257a1`。这是构造器及特征分区审计，仍未作为新科学协议冻结或执行准入。

## 原选择器状态观察及回归检查

新增 `bf_tap_r2.ema_reference_selector`，观察未改动的原V7/V12 `_train`返回处。原训练器返回最佳epoch供fresh-refit使用，当时网络却处在停止epoch；两者分别记录。观察层直接保存最后一次原验证张量及对应的终态网络，不新增warm predict，不将停止状态冒称最佳epoch状态，不改变原返回值、选轮逻辑或原训练源码。

42项定向检查通过，覆盖17项保存审计、18项原生账本和7项选择器观察。额外的原方法接线检查使用预设合成网络与不更新参数的mock optimizer：原PeriodicRegressor和JointRegressor `_train`各返回最佳epoch1、停止epoch3；两份状态在独立进程的完整批次、反序、分块和单行差异全为0。它验证原方法的观察与回读接口，未验证真实配方质量，实际torch optimizer初始化/参数更新0，新增科学拟合/确认拟合/全量拟合/提交包0。收据为 `local/research/native-selector-fixture-r1/receipt.json`，SHA `e9027f7b3719a82861923f86a6291c272194d24aa93ee0058e4332effd8ab02b`。

锁定Python3.12 CPU全库回归1400项通过、0跳过，保留23条原有警告。测试从拥有历史私有证据的根目录运行，新增包源码及两份新增测试取自隔离worktree；先核验两树540份既有Python文件逐字节相同，运行后复查其摘要及新增源码。最终收据在 `local/research/reference-engineering-checks-r2/receipt.json`，SHA `b9dc56560e3f35b412ee6dbe771d0201e128dc589cedda4e439af047b933ced8`。

首轮在无历史私有输入的隔离worktree执行全库回归，结果1389通过、2个缺参照输入失败、9跳过；该日志及收据原样保留于 `reference-engineering-checks-r1/`（收据SHA `15385a8a9cb977b6eb5a5149f5066e74835103364f8649f6cfb14b914338c64d`）。未修改历史科学代码、数据、冻结实验门槛或其失败决定，也未用skip掩盖失败。本节记录基础设施轮次，后续捕获接线验证见下节。

## 原工厂捕获与批次冷审计

新增 `ema_reference_capture` 以临时方法包装器接入原工厂，保留原全局模型类、构造器、配方、返回数组、原回收模块worker和单worker `ProcessPoolExecutor`。退出时恢复原方法及继承关系。每条管线在实际执行它的PID建立自己的原生账本，不把父进程账本跨fork复用。逐次核验训练特征值与顺序、外层ID、原V31内层seed及分区、反射的轮数和新refit配方；保存原最终预测及其对应对象，不新增最终warm predict。

原V31 helper未调用probe.predict。对此仅对probe的独立副本做无目标校准query推理并保存，明确标注为额外审计推理，原probe与其选轮反射不变。原V7/V12选择器仍保存停止epoch状态并分别记录返回的最佳epoch。每个真实参照工厂拟保存32个最终状态、6个内层树状态和2个网络选择器终态，共40个状态；这是预期范围，**尚非已完成的真实参照模型数**。

新增 `ema_reference_capture_audit` 由外部持有的warm收据SHA逐层绑定全配方、源目录/seed/fold/trial、32条管线、原生次数及每份状态。批次冷审计核验账本文件覆盖，拒绝额外调用、任何篡改及与warm拟合相同的PID，逐状态执行禁止拟合和CSV读取的完整批次、反序、分块、单行回读。只有全部状态和前后源码/产物摘要通过才生成独占的cold完成收据。

新增20项接线检查与原42项基础设施检查合计62项通过。合成数据及不更新参数的求解器fixture保留原TrialRegressor/V31Regressor的fit/predict和原V31选轮helper：原工厂单worker base-store路径闭合32条**合成角色**，worker PID一致且不同于父进程，独立新进程冷回读32份状态的全部差异为0；另核验V31选轮3及full-refit、原错误不被失败写入覆盖、错误seed/训练特征/query顺序拒绝，以及冷审计拒绝篡改和同warm进程。这些fixture没有优化器更新、官方目标读取或新增科学拟合，**没有验证实际21条历史base加A/B专家等32条异构管线的模型质量或完整运行G0**。

首轮 `reference-capture-checks-r1` 有55项通过。第二轮增加批次冷审计检查，59项通过、1项失败：原生账本篡改已正确拒绝，但测试的错误消息匹配遗漏了该分支；失败日志与收据保留，修正测试断言后第三轮61项定向通过。新源码及测试摘要在第三轮前后核查。第三轮全库1419项通过后，再补充调用方清单独立拷贝与Python3.12准入守卫；最终第四轮62项定向及1420项全库通过、0跳过，691份既有Python文件及新增源码前后逐字节核查一致。最终私有收据为 `local/research/reference-capture-checks-r4/receipt.json`，SHA `18e7bf4bcd7a91e9741db62ef53dcbae4531811af6773419ce5292361868168a`。

全库有24条警告：23条原有警告及1条Python3.12关于在已有多线程的全库测试进程中fork的警告；定向独立测试进程无该警告。保留完整日志，科学控制器的独立新进程、单worker/线程及资源准入仍须完成；不将测试通过直接视为科学执行准入。确认控制器、实际运行协议及科学预算冻结仍待补齐。

## 确认控制器与独立四seed算术

新增[确认协议](PREREGISTRATION.md)及配置，固定SHORT_SPAN、复用42/3407并声明271828/314159。控制器按每fold原参照、原EMA控制、SHORT的顺序执行，独立冷进程核验全部44个状态；十个单位全部闭合后才汇总四seed，独立审计用fsum和df=3的Student-t复核收益/LCB。新增模型桥保留原ComponentRegressor.fit/_train及选中的EMA状态。原开发池和失败决定保持原样，现行两seed分类API分别作开发回读和确认描述，不跨seed平均预测向量。

81项定向及1439项锁定Python3.12全库回归通过、0跳过，保留24条既有测试警告；751份源码/配置与694份两树公共Python文件前后摘要一致。最终收据为`local/research/confirmation-controller-full-r1/receipt.json`，SHA `03f45fcb8b4813c932c9ae81abe9acfc374d336d859dde175631cd17fe5807a5`。构造器还核查20份原参照输入身份和37个输出列的固定组合图，拟合与CSV读取均0；私有收据SHA `91fc21c42aad736810dd8ea552a20637e240996c85a50fd44a73970b63b7525a`。配置现绑定182份直接输入身份，包括原物理缓存和回收脚本。

本轮代码及静态协议已闭合，实际运行manifest、fresh单Python线程/单worker资源准入和activation尚待完成；新增科学拟合、确认seed启动、全量拟合和包仍0。真实320条异构参照管线及440个状态只有运行后实际审计通过才能计为完成。动态状态入口为`ema_span_confirmation_20261001`。

## 实际运行准入与启动

运行清单已冻结，manifest SHA为`7a81ac382bfd09fde2ff625a86f9a9e7fa5326adda6b9cdc23df62c628571120`，科学源码commit为`0f593434b5e8d5dda4e69c4e7ef7a94cdd6864f4`；751份源码/配置、1234份模型相关源码、182份输入和十个新增分区核验通过。启动前禁止fit/optimizer的独立审计记录尝试0，Python/torch/interop线程均1，原V31内层seed7771及四seed折向量一致。

确认controller/observer已启动，并以真实PID、create_time和命令核验初始活性；运行状态读取`ema_span_confirmation_20261001`。此处仅登记初始启动，不在600秒观察间隙轮询进度。科学完成次数、完整额外seed收益和440份状态审计尚未登记；平台最高仍为用户回传96.3920，没有新提交包。

## 第二次定时观察：首个真实单位闭合

1200秒观察记录1/10单位完成、0失败。已完成的seed271828/fold0绑定32条原参照管线、原EMA控制及SHORT；独立冷进程检查40份参照与4份EMA状态。只核查闭合单位，不计正在执行单位的部分调用，也不计算早期fold或完整seed收益。

独立收据链回读核验原运行目录/seed/fold/trial、manifest及分区、原生调用和外部保存状态摘要。闭合次数为CatBoost29、EBM fit6、EBM boost48、sklearn MLP1、torch Optimizer10；warm/cold最大RSS分别1266.35/626.75MiB，均低于1536MiB，固定组合差异2.27e-13低于1e-9。751份源/配置与1234份模型源码前后摘要一致。整批440份状态、原生.pt终态审计和四seed判定仍待完成。

追加式私有回读收据为`local/worktrees/ema-short-release-preparation/local/research/confirmation-observation-r1/receipt.json`，SHA `133bdbb3d8cf728abcf7507c9818b1395bbb6feca1874e01927372546ec1a87a`。此检查新增拟合/包0，没有中断、重启或改动确认训练及冻结协议。

4800秒定时观察为4/10单位完成、0失败。seed271828/fold0–3的闭合收据链及冻结源码回读通过，合计128条参照管线、8个EMA估计器、176份冷审计状态；闭合CatBoost116、EBM fit24/boost192、MLP4、torch Optimizer40。warm/cold峰值1283.45/641.17MiB，组合最大差仍2.27e-13。没有查看部分fold或seed质量，也不计未闭合单位的调用；整批G0及G1仍待完成。追加收据`local/research/confirmation-observation-r2/receipt.json`位于上述发布准备工作树，SHA `4144aee44059397a3e826b53a9c9243d1a133296cc8533527874bba4f8dd7918`。

5400秒定时观察为5/10单位完成、0失败；上述独立回读仍仅覆盖此前4个闭合单位。按冻结协议等待全部10个单位完成后再查看新seed收益，没有读取部分质量或新增拟合/封包。
