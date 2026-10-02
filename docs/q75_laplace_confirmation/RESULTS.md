# 固定尺度Laplace：四seed正式门通过

2026-10-02，固定`LAPLACE_FIXED_A20=.8*Q75_time+.2*LAPLACE_FIXED_time`。20个新selector/refit状态和完整四seed OOF已闭合，未重复开发、Q75或Gaussian拟合。

| split seed | 相对Q75 | 相对GAUSS1_A20 |
|---|---:|---:|
| 42 | +.008952557572 | −.000489536303 |
| 3407 | +.012171228685 | +.002024018546 |
| 271828 | +.008908561702 | +.004792443974 |
| 314159 | +.016872094481 | +.006957793057 |

相对Q75均值+.011726110610，单侧seed配对LCB95(df=3)+.007307128497，四seed严格为正，**G1相对当前平台参照的本地正式晋级**。相对已待测Gaussian均值+.003321179818，3/4 seed正，LCB95−.000497374980；不能宣称稳定优于Gaussian，更不能称平台已胜。四seed重用同批样本，平台最佳仍为用户回传、未独立核验的96.3920，96.45尚未实现。

**G0通过**：10个新outer估计器、20次原生optimizer、20状态；四真实进程exit0，OS峰值RSS507.5625MiB。训练分区/统计/词表、最佳checkpoint、fresh轮数及Laplace arm身份通过；冷/逆序/37行分块最大差2.84217094304e-14。独立math.fsum评分11124项最大差6.93889390391e-18，四seed门与相对Gaussian描述一致。6项锁定Python3.12准入/门槛检查通过，optimizer构造尝试0；原科学训练器的15项检查、4工程optimizer/4状态与40开发optimizer/40状态另阶段复用。

源码4994656；私有根`local/runs/q75-laplace-confirmation-20261002/confirmation-r1`，2473份冻结依赖。manifest `2cbc9cef460d7a51807ddecfcbe642c6ed1b49ba5efd134d886d4be08ad02f6d`；report `47ae9334f5bf8511dce06c89cbbdc1856648b062f5b4e55c21ec3aaa8166c627`；independent `5b842011857091f08344327e0cd8470bebebebd38ccaf070df8d5b18bccf04e8`；terminal `3688f9fb63ea93a41fb931f126b86f6f71b5d1a656c617abe8770c80d27dacbf`。

本阶段0全量、0包、0桌面写入、0助手上传。固定MAE位置网络比同骨干学习尺度Laplace在开发中更好，且当前Q75增量四seed稳定，构成独立全量发布依据；后续只准备一个固定包，不扫描权重。保留Gaussian作为有效待测参照，实际平台排序尚未知。
