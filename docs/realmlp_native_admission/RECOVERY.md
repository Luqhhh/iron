# 原生冷加载的零拟合恢复

2026-10-04。原`engineering-r1`的control/capture真实exit0，均完成原生256epoch selector并选择118epoch refit；完整验证轨迹、实际优化步、预处理输入、元数据及refit参数逐项一致。cold真实exit1，parity阶段未执行。原目录、4次optimizer及失败log全部保留。

原因已定位于本项目保存适配器的清理检查：作者拟合结束删除`_trainer`，但已安装PyTorch Lightning的`LightningModule.__getstate__`序列化时显式写入`_trainer=None`。反序列化后的两个原模型均只有这个空槽，没有train_dl、val_dl或callbacks。旧`hasattr(_trainer)`误拒绝合法空槽；不是发现了保留训练器或新的模型质量失败。

本恢复在独立源码快照中仅把清理条件改为“`_trainer`存在非None对象则拒绝”，并继续拒绝残留训练/验证dataloader和callbacks。不改原V9、损失、选轮、任何状态字节或预测。新增guard测试同时覆盖缺失/None允许、活trainer和所有残留数据对象拒绝。

恢复目录`local/runs/realmlp-native-admission-20261004/recovery-r1`，仅一次新进程冷读原两状态并重算原控制/捕获配对，新增拟合/optimizer0。原header绑定旧写入器的路径和源码SHA；显式恢复读取器同时绑定旧写入器、当前读取器、当前验证器、原failure和所有输入，不假装新代码就是原写入器，不改写旁侧header。恢复只限这两个精确绑定的合成状态；以后新模型由修复后的适配器保存新源码身份。

完整batch精确相等、反序/分块/单行≤0.0005、校准MAE差≤0.00005等原门全部保留。新进程禁止目标fit、optimizer及预处理fit。验证旧完整原生轨迹、两训练分区、原生预处理行多重集、最后精确最佳轮、参数摘要、全部冻结文件及实际进程退出。没有重新训练、扩大预算或自动重试原阶段。内存/线程/600秒观察/无时间预算约定不变。
