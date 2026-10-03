# EMA_MEAN3：用户指定探索的桌面交付

2026-10-02。用户要求“EMA_MEAN3 写桌面”，固定三训练seed等权探索已完成全量拟合、冷推理、包审计和桌面复制。**G0通过；G1为平台未测的人工探索，未正式晋级。** 本次交付完成后继续暂停优化。

桌面包：`C:\Users\lqh22\Desktop\submission-EMA-MEAN3-20261002\Luqhhh_bf_tap_predict_round2.zip`。候选名 `EMA_MEAN3_FULL_Q75`；ZIP SHA256 `016e7e9cb3f750c74509dbb51961c61fea205d0296bb233056da66005f69a396`，result.csv SHA256 `5de7ed69d5843eed6ef8f3787b67587ff208c09652073a844ed9c650f74610f4`。

按[冻结协议](PREREGISTRATION.md)，训练seed42、1042、2042各自选轮和fresh全量refit，成员顺序固定、预测等权。时长为 `Q75_time+.75*(mean(EMA42,EMA1042,EMA2042)-EMA42)`；铁量直接复制Q75的CSV原字段字符串。复用原seed42全量EMA，仅新增1042/2042两个全量程序、4次真实AdamW初始化、4个selector/refit状态，重新核验旧2个状态。实际一个提交包、一份桌面复制；0新CV、0确认seed、0助手上传、0自动科学重试。

2754行训练、322行query身份与原EMA一致，group-safe inner为2203/551行；训练分区内拟合预处理及目标尺度。三个成员选中epoch依次91、87、86，新增成员分别112、111轮停止，fresh全量refit使用各自选中轮数。原width256/MSE/EMA beta=.99配方保持，未选择成员或追加权重扫描。

独立新进程核验原生训练分区、预处理、选轮、optimizer账本、四个新冷状态及旧状态。完整batch冷差0；新增状态逆序/分块/单行最大差 `1.2991926809036158e-5`，低于冻结容差 `5e-4`。独立新进程再加载三个refit，逐行标量重算等权均值及替换公式，算术差0、铁量原字段差0。ZIP仅result.csv、CRC正确、322个唯一ID按官方模板顺序、所有预测有限非负。桌面复制后另一个锁定Python3.12进程核对ZIP/CSV字节、模板顺序和固定字段，全部通过。

冻结396份源、573份输入及401份模型来源映射，终态重新哈希一致。单worker、单数值/torch线程，训练峰值RSS736.094MiB，低于1536MiB；controller和supervisor实际exit0，所登记进程均已退出。运行按600秒观察和真实完成事件审计，无时间预算。锁定Python3.12完整检查1548通过、0失败、0跳过；受测主工作区与独立执行worktree的公开源码在检查前后字节一致。工程阶段的初次检查及隔离目录缺少历史缓存的失败收据保留，没有消耗科学拟合。

终态追加归档曾误把桥接warm收据的固定状态要求为`passed`；原收据保持“warm闭合、cold待独立审计”的原始快照，cold通过在另份收据登记。归档检查的exit1保存在`post-terminal-failure-r1.json`，按原接口状态及独立cold结果复核后exit0通过，没有修改科学源码、清单、原生收据，也没有新增拟合、包或桌面复制。

原开发相对Q75的两个完整split收益为+0.003551371、+0.000572388；相对匹配BASE均值为+0.000046354、−0.001077053。原配对确认门失败和exploration分类保持；三个训练seed不等于三个独立验证切分。这次固定均值探索用于检验已获平台正收益的EMA方向能否从降低训练随机性中获益，不能把本地增益换算成平台分数。

当前平台最佳仍为用户回传、未独立核验的Q75/Q100=96.3920；本包平台分数尚未测得。前两项已回传探索继续保留具体失败结论，DE3＋Q75仍仅作替补。用户自行上传，不推断剩余额度，不启动其他优化。

公开状态键 `ema_mean3_manual_exploration_release_20261002`，源提交 `a7c7a0e813bbbbf9a28117accfd7984290591a01`；manifest SHA256 `662478cff95ce09e6f2e3c39765fddc2f71ccfe8bbb7e491697f179c0d325daa`。本地原包、模型、账本及审计保存在 `local/runs/ema-mean3-exploration-release-20261002/release-r1/`，工程检查及失败收据在同批 `engineering-r1/`，均不入Git。桌面交付与独立审计收据为 `desktop-delivery.json`、`desktop-independent-audit.json`。

2026-10-03后续：用户再次要求本包与ModernNCA写桌面，[两包交付](../platform_information_value/DESKTOP_DELIVERY_20261003.md)完成后明确回传 **EMAmean 96.3954**（未经独立平台核验）。绑定同一ZIP，相对Q75=96.3920提升 **0.0034**，成为新最佳，距当前目标96.45为0.0546。原匹配BASE平均门失败及人工探索身份保持，不追溯正式晋级。原暂停字段属于当时交付历史；当前按用户持续优化目标执行。此反馈支持继续研究EMA训练seed平均，不能作为所有平均或其他模型方向的平台收益保证。
