# RealMLP保存状态前的原生接口核对

2026-10-04，本次为已保留时长方向的工程准备，未启动新拟合。原V9 runner及配方保持不变；本文件不是新科学训练准入或平台发布协议。

当前pytabkit1.7.3的`TabNNModule.on_validation_epoch_end`默认`use_last_best_epoch=True`，使用小于等于更新最佳轮。因此误差精确相同时保留最后一轮，不能套用EMA的首次严格改善规则。冻结合成MAE序列2、1、1后，直接调用作者原生回调：默认配置选第3轮；显式关闭该选项的对照选第2轮。单模型及均值最佳轮一致，六次原生验证回调全部通过。使用仅提供回调字段的轻量对象和合成输出，没有构造或训练神经网络。

原生状态生命周期也已按安装源码核对：Lightning先执行callback的`on_fit_end`，再执行module的同名hook；作者`ModelCheckpointCallback`先恢复选定参数，随后module删除数据加载器、索引和callback引用，接口在fit返回前移除trainer引用。原V9合成测试已有pickle后新进程预测证据，但这不替代新的全形状保存验证，也不能补回旧科学模型状态。

可采用的接入方式是调用原`RealMLPRegressor.fit`，在其外围临时捕获`make_estimator`及`InputEncoder`实际返回的两个对象，分别保存selector和fresh refit；捕获结束恢复原工厂。不能重写作者训练循环、选轮比较、256轮调度跨度、原生drop_last、预处理、损失或输出变换。后续工程必须检验捕获与保存不改变原流程的选轮、refit参数、预测或RNG，不能只检验pickle可以读取。

完整保存适配器仍待实现。其工程准入还需另行冻结合成拟合预算，并覆盖：两原生optimizer的真实构造/更新记录、分区和训练内预处理、精确选轮轨迹、模型/源码/配方身份、独立新进程禁止训练的selector/refit回读、完整/反序/分块查询一致性及内存门。旧缓存正收益不免除这些要求，不重复读取确认标签来代替工程问题。

本次固定Python3.12.12/torch2.14.0+cpu，导入前数值线程1，torch及interop1；probe实际exit0。保留一项torch.jit弃用警告，未据此同步依赖。私有范围、脚本和结果在`local/runs/realmlp-native-contract-20261004/contract-r1`。0神经网络构造、0拟合、0optimizer、0官方数据/历史预测/SiLU运行证据读取、0新确认seed、0包。G0仅为原生接口与选轮语义核对，G1无新增测量。
