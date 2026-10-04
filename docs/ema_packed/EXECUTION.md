# EMA-packed 执行记录

本阶段遵循[预登记](PREREGISTRATION.md)，参照用户回传DE3_IRON_EMA_MEAN3_Q100=96.3979，目标96.45。实时状态以根目录EVIDENCE_STATUS.json的ema_packed_development_20261004为准。

## 工程准入

锁定Python3.12.12、CPU依赖与导入前单数值线程。24项无拟合检查及1项合成检查全部通过，实际两个pytest子进程退出码均0。原生factory逐位输出一致，只有输出成员0的损失反传时，其他骨干/输出成员梯度逐位为0；共享数值嵌入存在非零梯度。原共享EMA训练循环未改变。mini终态的72个实际子退出及控制器/跟进/审计退出已闭合并绑定摘要。

工程合成seed964507、2204行、1个估计器、2次optimizer；selector/refit分别14/18次原生更新、各2epoch。冷进程PID68977与训练进程68974独立，selector/refit两状态审计通过；同完整batch逐位一致，反序/37行分块最大差2.765850126706937e-8，峰值763.609375MiB，低于1536MiB。无工程失败、无重试、无真实数据拟合。

检查收据：`local/runs/ema-packed-20261004/engineering-r3/checks.json`；SHA256 `48f4054c3dba4848e7208d126870249db2f3d861f2df3d6485e2fcc887d9025f`。原始JUnit、日志、合成状态与输入摘要保留在私有engineering-r1/r2。聚合检查只读取证据，无新拟合；engineering-r3实际工具退出0。

## 科学批次

源码提交8d996fd已普通推送当前upstream；独立detached worktree已冻结3576项文件身份。准备程序实际退出0。计划两split×五fold×三训练seed，共30估计器/60optimizer，60新状态和60原状态冷审计。600秒观察、单worker，无时间预算或自动重试；完整覆盖前不读取质量。科学批次已启动，实际消耗等待600秒观察或完成事件核对，G1未测，未正式晋级；0确认seed、0全量拟合、0包、0桌面写入、0助手上传。

控制器PID69348（工具会话57870），仅等待该进程内核退出事件的跟进器PID69383（工具会话53704）。控制器起始时间戳1791100474860903463；首次观察不早于1791101074860903463。跟进器不读取中途进度，在真实完成事件后执行冻结独立终态审计。清单SHA256 `9412bdf4e3257c2352fe512d427a675ef2cd4875f6da09c914277810f0e5f318`，启动收据 `local/runs/ema-packed-20261004/development-r1/execution/process-launch.json`。

## 首次定时观察

首次独立观察时间1791101088323888284ns，符合启动后至少600秒。已闭合12个子任务，全部exit0；完成5/30新估计器、5个新冷审计单元和2个原模型复用单元。已结束子任务峰值816.5MiB，当前worker在split42/fold1/init2042；控制器和跟进器均存活。没有读取局部质量，终态与G1均待完整覆盖。下一次独立观察不早于1791101688323888284ns；观察收据 `local/runs/ema-packed-20261004/development-r1/execution/observation-001-independent.json`。

第二次独立观察时间1791101699394812431ns，与上次间隔611.0709秒；已闭合26子任务且全部exit0，完成11/30新估计器、11个新冷审计单元、4个原复用单元。已结束子任务峰值817.73046875MiB，控制器和退出跟进器存活。未读局部质量；下一观察不早于1791102299394812431ns，原收据 `local/runs/ema-packed-20261004/development-r1/execution/observation-002-independent.json`。
