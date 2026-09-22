# V2.4–V2.6 优化交付：两个正式候选、一个探索候选

本次新增三个待测包均已写入桌面 submission。证据支持优先验证 DJ 铁量，其次 AORD 时长；BAY 铁量只是低优先级探索，不能将三个包都称为稳定提分方案。所有候选以当前用户回传96.0982的B包（C2铁量/完整B3时长）为父包；每包仅改一列，没有双目标组合包或平台上传。

最有效的路线是固定融合已有模型的互补误差：DJ使用不同结构及联合学习的铁量预测；AORD虽然独立Ordered模型较弱，但与B3的等权平均有收益。二阶平滑残差、更强/更弱L2和更深树的独立方案均未改善，不继续封装这些失败路线。

## 本次新包

| 新包优先级 | 包 | 改动 | 该目标平均 WMAPE | 当前参照 | 改善折数 | 资格 |
|---|---|---|---:|---:|---:|---|
|1|V24_FORMAL_IRON_DJ|铁量=.5 C2+.25 D4+.25 J1（seed42）|3.920512%|3.991918%|10/10|formal|
|2|V25_FORMAL_TIME_AORD|时长=.5 完整B3+.5 Ordered RMSE（seed42）|4.033918%|4.048214%|7/10|formal|
|3|V24_EXPLORATION_IRON_BAY|铁量=独立 Bayesian bootstrap RMSE（seed42）|3.985036%|3.991918%|6/10|exploration|

DJ还略优于旧待测AJ铁量（3.928656%）及I2（3.944227%）。AORD时长优于历史T1（4.040564%），但T1仍保持移出待测。BAY未达到至少7折改善，仅6折改善；两个完整OOF均小幅改善，因此仅取得 exploration 资格，明显弱于已有铁量融合。

配对重采样（2000次，固定OOF、铁口内按ID，两个切分共用索引）相对当前目标参照的平均WMAPE差值95%分位区间：DJ [-0.087125,-0.055509] 个百分点，AORD [-0.037615,+0.010318]，BAY [-0.029701,+0.016334]。AORD和BAY区间跨零；形式上的门槛通过不保证隐藏测试收益。这里没有把重采样当成额外淘汰门槛。

## 第三包的强度检查

为避免只凑数，本次又增加V2.6：固定J1三训练种子平均及其两种既定融合。三条路线相对C2均达到formal，但事先登记的附加交付要求是“所选路线两个切分均超过现有DJ”。最终J3平均3.921963%，略弱于DJ的3.920512%，未满足此条件；因此保留证据、不生成J3/AJ3/DJ3包、不事后放宽条件。第三个待测包仍只能是BAY探索项，建议额度紧张时先做前两个。

## 文件和队列

总待测顺序保留原先两个包在前：

1. V22_I_ONLY（旧包不变）
2. V23_AJ_I_ONLY（旧包不变）
3. V24_FORMAL_IRON_DJ
4. V25_FORMAL_TIME_AORD
5. V24_EXPLORATION_IRON_BAY（探索，最后）

已移出的V22_T_ONLY不恢复。没有新平台回传，用户自行上传；不会依据桌面文件数量推算剩余额度。

每个目录的提交文件均为 Luqhhh_bf_tap_predict_round2.zip，ZIP只含result.csv；官方322个V2 ID、三列、无重复缺失，新值使用.17g，未改列逐行复制B包字符串。铁量包改变322行铁量、0行时长；时长包改变0行铁量、322行时长。

- **V24_FORMAL_IRON_DJ**：`/mnt/c/Users/lqh22/Desktop/submission/round2-v2.4/V24_FORMAL_IRON_DJ/Luqhhh_bf_tap_predict_round2.zip`；SHA-256 `5f8017757139171f570641735146e42804c367fd95dd0e51ced9e4ec8a1a43f2`。
- **V25_FORMAL_TIME_AORD**：`/mnt/c/Users/lqh22/Desktop/submission/round2-v2.5/V25_FORMAL_TIME_AORD/Luqhhh_bf_tap_predict_round2.zip`；SHA-256 `26dccaf31c1f57c426d3944f666326e93b55d8deccaae8990212f67e40eb07da`。
- **V24_EXPLORATION_IRON_BAY**：`/mnt/c/Users/lqh22/Desktop/submission/round2-v2.4/V24_EXPLORATION_IRON_BAY/Luqhhh_bf_tap_predict_round2.zip`；SHA-256 `09eaaea110555fe66472bab9dc22e91adfb058804f6d14f26e8a567abd95f40b`。

## 工程与拟合

G0：最终锁定Python3.12完整测试 **670 passed**；三个新包的无训练数据读取冷推理CSV与正式CSV逐字节一致，反序/分批/单样本按ID一致，桌面SHA匹配。旧包、旧模型、原失败证据均保留。

真实新增CV：V2.4为60个CatBoost+20个Ridge=80，V2.5为30个CatBoost，V2.6为20个MultiRMSE模型，合计 **130个回归器拟合（110个模型流程）**。全量新增BAY铁量1次、ORD时长1次，共 **2次**；DJ复用预测，V2.6不做全量训练。不能把联合模型两列输出当作两次拟合。

工程合成拟合另计：三次完整测试进程观测75、78、78次估计器fit；相关定向测试额外4、3次，其余新增发布测试使用mock不拟合。本轮已知工程调用共238，统计范围为pytest父进程已插桩估计器，不声称覆盖子进程或未插桩类别。

G1：三个阶段均复用已开发多轮的两组切分，不是独立确认；所有收益均为本地开发证据，平台成绩未知。未承诺三包都能超96.0982。

私有证据：local/runs/round2-v2.4/smooth-r1、release-r1；round2-v2.5/time-r1、release-r1；round2-v2.6/joint-seeds-r1。模型、预测、原始指标、账本、ZIP及交付验证均不入Git。
