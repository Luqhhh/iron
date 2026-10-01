# SHORT_SPAN：四个完整切分确认协议（2026-10-01）

本线程目标96.45。新阶段参照为用户回传EMA_TIME_Q75=96.3920，未独立平台核验；包SHA为41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825。Q100同分事实保留，Q75只是外推较小的代表。本阶段不预测平台分数，不推断剩余测试额度，也不生成提交包。独立分支/worktree为codex/ema-span-confirmation，原基线和历史科学代码不修改。

## 资格、候选与参照

原[开发批次](../ema_average_span/RESULTS.md)已完整完成并通过G0及实际退出审计，SHORT_SPAN在seed42/3407相对Q75的增量为+.0021899322201757543 / +.0013855401783149184。按原预登记的两个完整seed均正门，选择SHORT_SPAN。LONG_SPAN的两seed失败决定保持，不再次拟合。原开发自动分类仍exploration，minimum_improved_folds失败记录不回写。

只增加seed271828/314159，各五个完整外层fold；复用42/3407的原物理缓存和原source directory/seed/trial身份。不同缓存别名不增加模型或seed数。四个seed仍重用同一批2754个训练样本，只检验切分稳定性，不保证平台泛化。

网络、损失、AdamW、学习率、初始化与inner seed42、batch256、max_epochs240、patience25、min_delta1e-5、训练内预处理和fresh-refit沿用原ComponentRegressor。每个额外外层池独立拟合原控制beta=.99与SHORT_SPAN beta=.9801，两者各自选择epoch，再新初始化于完整外层训练池refit；不借用控制的epoch，不调整轮数或上限，不扫描权重。

额外seed没有匹配的可复用参照。调用原fit_b0和V36FixedRecipeFactory，重建相同32条冻结管线及原权重：21条历史L1 base、4条A专家、4条B36专家及N0048/V12 joint/V7 periodic。保留原方法、配方和既有B36非负投影；不增加候选裁剪或其他后处理。原参照内部选轮、预处理和refit沿用各自历史冻结实现，不能将它们冒称统一的新训练协议。新增EMA控制和SHORT的预处理、尺度及选轮严格在自己的内层训练/校准分区内完成。

每seed独立构造`I=B+.75*(old_EMA−V7)`和`C=I+.75*(SHORT−old_EMA)`，铁量取该seed匹配参照的原V32列。禁止跨seed平均预测向量；只汇总指标。十个新增单位全部完成、冷审计通过后才查看完整额外seed收益，不用早期fold粗筛或跳过另一确认seed。

## 预算及模型状态

固定10次参照工厂、320条参照管线实例，另有10个原EMA控制和10个SHORT估计器。管线实例不等于内部solver次数。整批原生预算为：

| 入口 | 冻结次数 |
| --- | ---: |
| CatBoost fit | 290 |
| EBM fit API | 60 |
| EBM内部boost | 480，主效应240、交互240 |
| sklearn joint MLP L-BFGS fit | 10 |
| torch optimizer初始化 | 100，参照60、EMA控制和SHORT合计40 |

各管线的开始记录先于数值调用；失败/中断占用身份与预算，超额尝试不进入原求解器。原方法即使吞掉异常，账本闭合仍失败。不自动重试、不覆盖或恢复已消耗单位，不为失败修改预算。全2754行拟合、包、桌面写入、助手上传及平台队列新增均0。

每fold预计保存32个参照最终状态、6个内层树probe和2个原V7/V12选择器终态，共40个参照状态；另有原EMA和SHORT各两份选择器/refit状态，共44个。新增批次合计440个状态；这不是已完成数。原V7/V12终态对应停止epoch而非最佳epoch，最佳epoch返回值单独记录。ComponentRegressor选择器保存的是原选中的EMA状态，两者不得混称。原树probe和EMA选择器的额外审计推理仅在独立副本及无目标query上执行，原最终query预测不增加warm调用。

独立冷进程绑定外部warm收据、全部文件/源码、实际类、source directory/seed/fold/trial/fit、训练/query ID与原生预算，拒绝拟合及CSV读取。完整批次必须与warm观察逐项一致；反序、分块、单行容差5e-4。在32条最终模型的冷验证输出上独立重建固定组合，A的线性系数以原构造器的基向量传播冻结，B36原非负投影单独保留，组合误差门1e-9。原生.pt状态另核查训练内预处理、目标尺度、选轮轨迹及fresh-refit epoch。

## 判定与交付边界

四个完整seed各有2754行同协议OOF，增量为`50*(WMAPE_I−WMAPE_C)`。正式四切分证据门为四seed各自严格正、seed层配对单侧95% Student-t下界严格大于0；df=3。fold层仅描述，不作为独立样本计算LCB。不增加本地幅度、绝对包分或时间硬门槛。

现行candidate_tiers只接受两seed，保持原API：回读原开发两seed的SHORT分类，另对两个确认seed作描述性分类，各轮最多一个探索推荐。原开发候选池及分类保留，不以新的单候选回读追溯修改原开发池结果。四seed门、开发分类、确认描述及G0分别报告；通过四seed门不等于平台达标，也不自动授权全量发布或改变测试队列。

## 执行准入与保护

只读取已审计复赛V2训练表，不读初赛受保护月份或隐藏测试标签；sample_id仅用于身份和对齐。配置绑定数据、已有开发/控制模型、原回收脚本、原配方/权重/账本及静态输入摘要，实际运行manifest再绑定全部源/配置/环境、模型源码实际路径、四seed折向量、十个新增训练/query及内层分区、测试收据和源码commit。

锁定Python3.12 CPU、无依赖同步；OPENBLAS/OMP/MKL/NUMEXPR与torch数值/interop线程均1。原回收脚本有main/src路径插入，实际该源码位置也绑定，不能假定全部导入都来自工作树。原fork工厂前要求独立Python进程且没有其他Python线程；串行单worker，原RSS1536MiB门及可用内存门保持。无优化时间预算，600秒观察及实际完成事件触发审计；不重启或终止其他会话任务。

启动前要求本阶段完整锁定测试和精确源码清单通过、科学源码提交、前开发实际成功退出、最新平台参照仍与本配置一致、fresh私有目录和全部预算/分区闭合。prepare仅冻结manifest，worker必须有其绑定的activation；observer按原600秒规则运行。若参照在激活前改变，拒绝激活并保留证据；已激活阶段保留原冻结参照，不追溯变更。

本协议和实现写入时新增科学拟合0，测试和实际运行准入尚待完成。运行目标为local/runs/ema-span-confirmation-20261001/confirmation-r1。模型、预测、报告及账本仅存local/。EMA铁量待反馈、DE3＋Q75仅替补、ModernNCA仅工程准备的授权边界保持。
