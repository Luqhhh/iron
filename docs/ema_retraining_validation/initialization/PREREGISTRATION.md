# 当前 EMA 时长的训练随机性配对验证（2026-10-02）

用户本轮要求按顺序“进行”；本目标持续授权有效。先闭合已有融合选轮真实终态与审计，再串行执行本阶段。单一 N_TO_V36_P05 包等待用户上传/回传，平台反馈不由助手制造；不扩展权重池。冻结平台参照为用户回传、未独立核验的 Q75/Q100=96.3920，目标96.45、差距0.0580；本地增量不能外加到该分数。

固定 outer split seed42/3407 各完整五折，固定 group-safe inner seed42 的 fold0 校准；训练 seeds 预先固定为42、1042、2042（42+1000*k），BASE/EMA 一一配对。不按结果选择训练 seed，不利用既有1042单折结果择优。原训练 random_seed 同时控制权重、dropout与小批量顺序，因此这里检验原训练程序的随机性，而不声称纯权重初始化的因果效应。每个训练 seed 的 selector/refit 都用同一训练 seed；内层划分、网络、MSE、预处理、原float32组件MAE选轮、patience25、min_delta1e-5、max240epoch、batch256、EMA beta0.99和按所选epoch fresh refit 完全沿用原设置。外层验证标签不进入任何训练或选择。原训练器、历史冻结协议、父包不改。

训练seed42的20个原估计器经原complete/manifest、fit/query ID、源码、选轮、预处理和新进程冷推理核验后复用，未通过即停止、不自动重训。1042/2042新增40估计器，每个selector/refit两次optimizer，合计80次；所有selector/refit共120状态保留并独立冷核验。单worker、数值和torch线程1、RSS1536MiB、冷分块/顺序容差0.0005、全批精确相等；600秒观察，真实完成事件立即审计，无时间预算、无自动科学重试。0新outer seed、0全量拟合、0包、0桌面写入、0助手上传。

所有组合沿用共同背景：`C(A)=Q75+0.75*(A−EMA42)`；铁量不改。同一outer seed、同一fold内按三训练seed等权平均BASE与EMA；禁止跨split seed平均向量。报告每训练seed的EMA−BASE组件与融合收益、meanEMA−meanBASE、meanEMA−每个单EMA，以及所有固定列相对原Q75完整OOF增量、折/铁口损害分布。均值和已观察单seed最大值仅作描述，不选择“最好seed”，不扫权重。

本阶段唯一可准备确认的科学候选是EMA_MEAN3：两个完整开发split对Q75均正，且对对应BASE_MEAN3均正才可准备确认。两开发split不构成正式晋级，训练seeds也不能凑成四个split；正式晋级仍须至少四完整split、各正与seed层配对LCB95>0，以及后续冻结的其他门。自动分类沿用candidate_tiers；本阶段不消耗确认seed挽救失败。G0与G1分别报告，不承诺平台提升。

先锁定Python3.12完整测试，再提交科学源码、冻结源/数据/旧缓存/分区/环境/授权摘要；每单位入口重验，拟合仅接收外层训练池，预测query无标签。每状态保存真实预测、模型和query见证，独立进程禁止拟合/CSV读取冷回放；结束后从闭合OOF由独立标量脚本重算。失败目录及原证据永不覆盖。
