# Laplace时长：G0通过，完整开发已启动

2026-10-02。两臂FIXED/SCALE、两个完整开发seed42/3407、固定20%相对Q75融合，20估计器/40 optimizer/40新状态，细节见[预登记](PREREGISTRATION.md)。科学源码冻结24b82a4；对Q75及同seed GAUSS1_A20完整端点比较，现有Gaussian/EMA_MEAN3待测包不变。

锁定Python3.12零训练定向检查15项通过，optimizer构造尝试0。完整2754行合成G0两臂各selector+refit、合计4次optimizer/4状态通过；FIXED查询MAE .558412081498，SCALE .748516735670，训练中位数常量17.998104608608。两臂均可学；SCALE在该合成例较差不用于选择/删去某一臂。新进程冷回读最大差2.84217094304e-14，两个真实进程exit0，OS峰值RSS502.99609375MiB。

工程manifest `2b3d1651cbadd0290d3aedff78d8f127583d67e6d63abfef285f690d3870baa7`；工程terminal `94e0f9a4a325be97659f0a7e62db331ff39b4a9dba8049deee7310e4781756d7`。正式开发manifest `e8f7a9ea69ac7a497fa9f4352aa8492247d240981467b72894f2166bb3740863`，运行根`local/runs/q75-laplace-time-20261002/development-r1`。控制器和worker真实启动身份、命令、worktree、四项数值线程1均已核验，收据保留私有。

当前G0工程准入通过，G1官方完整开发未出结果。0参照重拟合、0确认、0全量、0包、0桌面写入、0助手上传。单worker/RSS1024MiB，600秒观察，无时间预算或自动重试。
