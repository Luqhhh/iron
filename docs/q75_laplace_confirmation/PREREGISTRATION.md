# 固定尺度Laplace：两seed接续确认

2026-10-02，依照96.45目标内持续授权。前置完整开发源码24b82a4、公开结果9bd2596：FIXED相对Q75的42/3407增量+.008952557572/+.012171228685；相对GAUSS1_A20均值+.000767241121，符合原确认准入。SCALE两seed弱于FIXED且未过Gaussian比较门，保持失败决定。两开发seed分类不替代四seed正式晋级。

只确认`LAPLACE_FIXED_A20 = .8*Q75_time+.2*LAPLACE_FIXED_time`，不更改固定尺度.5、V33 128/128 SiLU骨干/周期嵌入、预处理、种子42、AdamW、batch256、max240、patience30、min_delta1e-5或梯度裁剪。原 `laplace_time_model.LaplaceRegressor` 源码和训练配置逐字节/逐字段绑定，不重跑开发或添加权重探针。271828/314159各完整五折，共10个outer估计器、20次selector/fresh-refit optimizer、20新状态，单worker/数值与Torch线程1/RSS1024MiB。

前置两臂4次完整合成工程optimizer和40次正式开发optimizer已有真实终态及独立冷证据，本阶段0新增工程optimizer。锁定Python3.12定向检查禁止optimizer构造。单元测试、原G0和开发状态、Gaussian四seed确认的Q75/GAUSS1_A20向量及源证据均冻结，再追加访问账本后读授权复赛标签。禁止初赛保护标签，禁止跨seed平均预测。原group-safe inner27001五折held0只在外层训练池选择轮数，fresh外层refit仅按选定轮数训练；query移除目标。

完整新状态必须在新进程验证训练/校准身份、均值/方差/词表、完整选轮轨迹与最佳checkpoint、fresh轮数、arm标签、参数有限、整批/逆序/37行分块预测，沿用1e-8容差。另进程math.fsum重算四seed完整端点、fold/铁口描述、相对Q75及GAUSS1_A20的配对收益及单侧Student-t LCB95(df=3)。原四seed参照只读复用已审计OOF，不重复母模型。

正式晋级条件为至少四完整seed相对Q75均严格正，且seed层配对LCB95严格正。对已待测Gaussian端点的差异单列描述，并用于之后安排平台测试优先级；不把小的开发均值差称为统计独胜，也不把其加入96.3920预测平台分。GAUSS1_A20与EMA_MEAN3仍为已交付待测包，平台最佳Q75/Q100=96.3920仍仅为用户回传。

本阶段0参照拟合、0重复开发、0全量、0新包、0桌面写入、0助手上传。600秒观察，实际完成事件立即审计，无时间预算、无自动重试，失败/中断证据保留。后续发布仍须独立冻结并考虑与现有候选的增量信息，不能因确认运行结束自动生成近似包。
