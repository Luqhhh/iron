# V2.4 本轮结果

正式推荐 DJ 铁量，探索推荐 BAY 铁量。时长无入围项。QRC 平滑残差方案两目标均退化，保留证据，不发布该路线。DJ=.5 C2+.25 D4+.25 J1。BAY 收益很小且只有6/10折改善，作为探索项，不能当成稳定提升。

|目标|路线|平均 WMAPE %|相对当前参照 Δ百分点|改善折|分类|
|---|---|---:|---:|---:|---|
|tap_iron|BAY|3.985036|-0.006882|6/10|exploration|
|tap_iron|R30|4.045547|+0.053628|0/10|not_shortlisted|
|tap_iron|QRC|4.371644|+0.379725|0/10|not_shortlisted|
|tap_iron|ABAY|3.971850|-0.020069|8/10|formal|
|tap_iron|AR30|4.004126|+0.012207|1/10|not_shortlisted|
|tap_iron|AQRC|3.999429|+0.007511|5/10|not_shortlisted|
|tap_iron|DJ|3.920512|-0.071406|10/10|formal|
|tap_time_len|BAY|4.084011|+0.035797|1/10|not_shortlisted|
|tap_time_len|R30|4.103373|+0.055159|2/10|not_shortlisted|
|tap_time_len|QRC|4.869347|+0.821133|0/10|not_shortlisted|
|tap_time_len|ABAY|4.052672|+0.004458|5/10|not_shortlisted|
|tap_time_len|AR30|4.062447|+0.014233|3/10|not_shortlisted|
|tap_time_len|AQRC|4.237564|+0.189350|0/10|not_shortlisted|

完整两组、逐折、逐铁口指标及相对 I2/AJ/T1 的差值见私有 selection/summary.json 与 comparison.csv。推荐项使用 2000 次配对重采样，未改变门槛。DJ 略胜已排队的 AJ；BAY 明显弱于 AJ，只适合低优先级探索。时长路线不足以交付第三个候选，下一阶段检验较弱 L2、更深树和 Ordered boosting，冻结新候选后独立登记，不修改本轮结果。

G0：664 项锁定测试通过，4 项新模型测试通过；60 个 CV 包装器模型独立回读 PASS，折身份、尺度、反序/分批预测、指标、分类和重采样一致。真实 CV 拟合为60 CatBoost+20 Ridge=80；初次定向测试4次估计器拟合，完整测试75次，均为工程合成拟合另计。G1 为复用开发折的本地证据，未获得新平台回传。

私有目录：local/runs/round2-v2.4/smooth-r1。现有两待测包和已移除的 T1 文件保持原样。本文件先记录 CV 阶段，最终交付见 DELIVERY.md。
