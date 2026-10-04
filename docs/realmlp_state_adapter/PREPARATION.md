# RealMLP 原生状态适配器准备

本阶段只完成保存、读取及原训练对象捕获的基础接口，尚未取得真实模型保存后的冷推理准入。[范围清单](../../configs/realmlp_state_adapter/PREPARATION.json)限定0次神经网络构造、拟合、optimizer和官方数据读取；后续合成拟合及科学拟合均需另行冻结范围。

`realmlp_state_adapter.capture_fit`只调用一次原V9的`RealMLPRegressor.fit`，暂时截获其原工厂实际创建的selector/refit估计器及各自训练内编码器；不替换原训练循环。成功或异常退出均恢复工厂。另行计算的内层ID用于身份核对，真实训练下的RNG与数值等价性仍待验证。原V9文件摘要保持`705073d45b8fc9cc0ea6439aef4d153862363806d375aa484bebbefb23ddbbdc`。

保存接口保留两个阶段的原生估计器和编码器，绑定物理目录、split seed、fold、trial、训练ID与目标摘要、配方、依赖版本、原代码及适配器摘要。文件和旁侧清单不覆盖；加载前校验身份、版本与文件摘要，加载后检查原生配置、CPU有限状态及编码器。推理拒绝目标列、错误输出形状和非有限预测。上述保存正向路径尚未由真实拟合模型验证，不能据此宣称完整G0通过。

原生选轮使用最后一个精确最佳epoch，保留原256epoch调度跨度；已闭合的[原生回调验证](../realmlp_current_review/NATIVE_STATE_CONTRACT.md)为独立历史证据。

锁定Python3.12.12、CPU依赖及单数值/torch线程的检查在新目录`local/runs/realmlp-state-adapter-20261004/nofit-r2`完成，工具会话35259真实exit0，20项通过。这些检查使用记录型假估计器，覆盖原fit调用及工厂恢复、内外层数据身份、配置与角色、缓存身份、反序推理，以及反序列化前的错误拒绝；没有创建神经网络或optimizer。真实模型pickle成功回读、原生优化步数、完整验证轨迹、原版/捕获版数值等价和独立新进程冷推理尚待后续工程阶段。

首轮`nofit-r1`真实exit1，9项通过、11项因pytest临时目录父目录不存在而在setup失败，没有执行这些断言。原JUnit、源码副本及失败收据保留。第二轮提前创建独立父目录，同时把私有根路径改为可注入常量，并用测试隔离fixture提供pytest临时目录，使路径检查在普通pytest入口也能正确执行。未覆盖或清除失败证据。

本阶段G0仅为无拟合接口检查通过；G1无新质量评估。私有聚合收据见状态键`realmlp_state_adapter_preparation_20261004`。RealMLP时长A20的两split缓存正收益不因此成为四seed正式晋级，也不自动激活确认拟合、封包或平台队列。
