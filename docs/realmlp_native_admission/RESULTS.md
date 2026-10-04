# 原生 RealMLP 工程准入结果

锁定 Python 3.12、原 pytabkit 1.7.3 CPU 环境中，原版与状态捕获版各完成一次合成数据程序，共四次原生 Adam、6456 次实际更新，均选择第118轮。选轮轨迹、参数状态、预测完全一致。未读取官方目标，未产生科学质量结论。

原冷读进程退出1，原因是 Lightning 序列化主动保留 `_trainer=None`，旧守卫将空槽误认为活跃训练器。原失败及模型保留在 `local/runs/realmlp-native-admission-20261004/engineering-r1`。修复只允许缺失或空 trainer，仍拒绝活跃 trainer、训练/验证 loader 和 callback。

另立 `recovery-r1`，绑定原写入器、原状态及新读取器，零拟合、零 optimizer 恢复成功，实际退出0。两状态冷读原 batch 精确一致；反序、分块和单行最大差7.62939453125e-06，低于预定0.0005。原生校准 MAE 与独立重算一致。修复测试26项通过；原始冷读失败没有被改写为成功。

独立终态核对重验983个冻结文件及全部所属进程已退出。终态收据 SHA256：`454ed683944b6f414ffb0ae9a0c5fd18e3bf8ad04be4d2cabd5a34a7f397be92`。

G0：合成原生等价及独立冷读闭合，允许另行冻结科学开发。G1：未测量质量，未晋级、未全量拟合、未封包。后续见[完整时长开发协议](../realmlp_time_development/PREREGISTRATION.md)。
