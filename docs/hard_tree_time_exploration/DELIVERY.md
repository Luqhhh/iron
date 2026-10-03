# GLOBAL硬树时长人工探索交付

2026-10-03。`HARDTREE_GLOBAL_TIME_A20` 已完成全量拟合、独立冷推理及无标签封包，G0通过。G1仍为**人工平台探索、未正式晋级、平台未测**：完整开发split42/3407相对Q75增量为−0.001431179053/−0.000458472100，两者均负。原V39控制角色及失败决定保留，不消耗新CV或确认seed。

配方与预算见[冻结协议](PREREGISTRATION.md)：时长为`0.8*Q75_time+0.2*GLOBAL_tree_time`，铁量直接保留Q75原CSV字段字符串。信息问题是硬路由树能否补充当前组合，不能由预测差异或旧SAM反例推断本包平台收益。

包位于私有目录：

`local/runs/hard-tree-time-exploration-20261003/release-r1/HARDTREE_GLOBAL_TIME_A20/Luqhhh_bf_tap_predict_round2.zip`

- ZIP SHA256：`d36d2af53b6367801fb7668c7de53cef6e53a682415d81b30d9d1f44a844b40c`
- CSV SHA256：`8a75edd4d14fd171b460a127a8c195c151c80da58f33a8e798ed0fb76e7183c6`
- 父包为Q75，SHA256：`41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825`。

实际执行1个全量程序、2个估计器/训练预处理、2次Adam构造、2保存状态、1ZIP。选轮拟合2203行、校准551行，选中15轮、停止65轮，实际585个optimizer step；fresh全2754行refit15轮、165个step。构造前账本、训练后状态与原生步数闭合。科学源码沿用冻结V39，发布脚本提交为`db67470dc0485ff6bb1ffd7854f1b623e413eca6`。

锁定Python3.12路径32项检查通过。新进程禁止训练和optimizer，重载两状态，独立重算训练分位数、类别词表和目标尺度，复核选轮、停止历史与全量轮数；冷预测、反序和分块最大差均为0。另一个禁止读取训练文件的进程核对公式、生成并回读ZIP：仅`result.csv`、322唯一ID按官方模板顺序、CRC正确、有限非负、铁量字符串差异0、算术差0。训练、冷审计、封包和supervisor实际exit均为0；峰值RSS1178.78MiB，小于冻结1536MiB门。1941项冻结依赖未变，自有进程全部退出；运行在首次600秒观察点之前实际完成，没有自动重试或时间门。

追加式最终对账在`local/runs/hard-tree-time-exploration-20261003/launch-r1/final-reconciliation.json`，模型/账本/报告均留在私有`local/`。本批0桌面写入、0助手上传；由用户自行平台测试并回传。

另行冻结的七包无标签比较见同目录`information-review-r1/`：本包相对Q75时长平均绝对改变量0.345668，与ModernNCA/EMA_MEAN3/已测EMA32改变量余弦为0.141149/0.136724/0.087270。七包均通过SHA/CRC/模板序/铁量字符串检查，独立`math.fsum`复核最大差5.55e-17。该结果仅说明预测方向不同，不证明误差独立或平台更优。当前顺序见[信息价值安排](../platform_information_value/ALLOCATION.md)；不因包就绪就占用名额。
