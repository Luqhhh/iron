# EMA-packed 执行记录

本阶段遵循[预登记](PREREGISTRATION.md)，参照用户回传DE3_IRON_EMA_MEAN3_Q100=96.3979，目标96.45。实时状态以根目录EVIDENCE_STATUS.json的ema_packed_development_20261004为准。

## 工程准入

锁定Python3.12.12、CPU依赖与导入前单数值线程。24项无拟合检查及1项合成检查全部通过，实际两个pytest子进程退出码均0。原生factory逐位输出一致，只有输出成员0的损失反传时，其他骨干/输出成员梯度逐位为0；共享数值嵌入存在非零梯度。原共享EMA训练循环未改变。mini终态的72个实际子退出及控制器/跟进/审计退出已闭合并绑定摘要。

工程合成seed964507、2204行、1个估计器、2次optimizer；selector/refit分别14/18次原生更新、各2epoch。冷进程PID68977与训练进程68974独立，selector/refit两状态审计通过；同完整batch逐位一致，反序/37行分块最大差2.765850126706937e-8，峰值763.609375MiB，低于1536MiB。无工程失败、无重试、无真实数据拟合。

检查收据：`local/runs/ema-packed-20261004/engineering-r3/checks.json`；SHA256 `48f4054c3dba4848e7208d126870249db2f3d861f2df3d6485e2fcc887d9025f`。原始JUnit、日志、合成状态与输入摘要保留在私有engineering-r1/r2。聚合检查只读取证据，无新拟合；engineering-r3实际工具退出0。

## 科学批次

待当前分支普通提交/推送后冻结独立worktree及输入清单。计划两split×五fold×三训练seed，共30估计器/60optimizer，60新状态和60原状态冷审计。600秒观察、单worker，无时间预算或自动重试；完整覆盖前不读取质量。当前科学拟合0，G1未测，未正式晋级；0确认seed、0全量拟合、0包、0桌面写入、0助手上传。
