# 固定尺度Laplace：单一全量发布

2026-10-02，96.45目标内持续授权。前置4994656科学确认已完成：四seed相对Q75均正，均值+.011726110610、LCB95+.007307128497，20个新状态冷回读和四个真实进程exit0。相对已待测GAUSS1_A20平均+.003321179818，但一seed负、LCB95−.000497374980；没有稳定优于Gaussian或平台提升的结论。

本阶段只生成`LAPLACE_FIXED_A20=.8*Q75_time+.2*LAPLACE_FIXED_time`。发布理由是同骨干固定MAE臂在完整两seed开发胜过学习尺度臂、相对现有Gaussian均值有正增量且相对真实平台参照四seed门通过；不是因本地微小差异而追加权重包。只一个权重、一个目标，Q75铁量原CSV字段字符串保留。父包SHA256 `41a046d5ce74e8a9c7c9acb124fa680cabf9e87e4a79edb50d81efb625bde825`。

固定尺度.5及 `laplace_time_model.LaplaceRegressor` 原码逐字节复用，训练配置取确认清单中的native配置，不回退Gaussian NLL。2754行训练数据按原group-safe inner27001/五折held0划F/C，F训练预处理和目标均值/总体标准差，C按标准化时长MAE选轮；fresh在全部2754行以选择轮数refit。原两层128 SiLU/周期嵌入、float32预处理转float64网络、初始化42、AdamW lr.001/wd.0001、batch256、max240、patience30、min_delta1e-5及梯度裁剪10均不变。322行test无标签query依官方模板顺序，不按ID假定时间，不做裁剪或新增后处理。

预算为一项全量程序、两次正式optimizer、两个新状态；0新CV、0参照拟合、0新增工程optimizer。已完成的Laplace两臂合成G0四次optimizer/四状态及后续40开发/20确认属于各自原阶段，直接复用其身份闭合证据。发布检查禁止optimizer构造。原生账本必须以整数split_seed=-1/fold=-1、独立source directory及新trial记录F/C和全量身份，恰好两次实际Optimizer构造。

在保护配置、授权、源码、环境、输入、确认和G0终态、父包及检查收据全部冻结后才追加访问账本并读授权复赛标签。独立worktree、独立私有运行目录，禁止初赛保护标签。单worker、数值与Torch线程1、RSS1024MiB、600秒观察、无时间预算、无自动重试，失败/中断证据不覆盖。

selector/refit两个状态必须新进程回读训练分区、统计、词表、Laplace标签、选轮轨迹、最佳checkpoint和fresh轮数，冷/反序/37行分块保持1e-8容差。另一个新进程禁止读取训练输入，只加载全量模型及query做冷推理。独立封包后再另进程核验ZIP仅含result.csv、322唯一ID按官方顺序、有限非负、CRC、固定融合的独立标量算术、ZIP/CSV字节一致及铁量字段原字符串差异0；五个实际进程exit0及预算全部闭合后才交付。

0桌面写入，0助手上传；用户自行测试。平台最佳仍是用户回传、未独立核验的Q75/Q100=96.3920，96.45目标尚未实现。本地切分稳定性与Gaussian对照不等于独立平台泛化，现有Gaussian/EMA_MEAN3包保留，DE3仅替补，不根据文件数量推断额度。
