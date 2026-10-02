# SAM＋EMA时长固定Q75人工探索发布

2026-10-02。本阶段按用户持续优化授权“授权，以及之后不需要申请”，冻结一项人工平台交互探索。动态参照EMA_TIME_Q75=96.3920（用户回传、未独立核验），目标96.45；未测试的原N→V36探针与DE3替补安排保留，额度未知，不由文件数或日期推断。

[完整开发](../sam_ema_time/RESULTS.md)已通过G0，但对Q75两开发seed−.024537241/−.021519181，正式确认门失败、自动not_shortlisted，不改判或消费新增确认seed。对匹配原SAM两seed+.006092404/+.018952455；原SAM、EMA时长在同V32父包各有用户回传的平台正反馈。本探索检验二者训练机制交互能否改善当前EMA，不能由本地负收益排除，也不能将旧收益相加、映射或预报新平台分数。候选是人工探索，非四seed正式晋级。

只生成 `SAM_EMA_TIME_Q75_EXPLORATION` 一个包，沿用开发中已固定的.75权重：`parent_Q75_time+.75*(new_full_SAM_EMA_time-old_full_EMA_time)`。原SAM-EMA科学训练器与全部设置保持：periodic TabM width256/blocks2/k16/dropout.1、frequency.01、MSE、同dropout双前向SAM rho=.05/epsilon1e−12、一次AdamW lr=.001/wd=.0001后一次EMA beta=.99、inner seed42/fold0、训练内预处理/选轮和fresh refit。初始化42，max240/patience25/min_delta1e−5，验证使用EMA的标准化成员均值MAE。不扫描配方或权重、不同时改动其他参数，不加裁剪。

预算为1项新全量训练程序、2次真实AdamW初始化（selector/refit）、2个新原生状态、1个ZIP；0新增CV/确认seed、0其他模型或KMeans、0桌面写入、0助手上传。完整训练2754行，官方test/template322唯一ID。全量scope seed/fold均为−1，只是身份哨兵，不是新outer seed。复用旧完整EMA selector/refit和已测Q75父包，不重拟合基线；不同物理source directory/split/trial不混作同一拟合。

先通过锁定Python3.12的模型、真实两次optimizer桥接、冷见证、冻结范围、边界、字节保留和包回读定向检查；完整源、授权、开发终态、官方数据、旧完整EMA、包及环境先冻结并核验，之后才启动。新训练器与科学源字节一致，旧工程与确认失败证据保持，发布代码/范围在独立worktree。

训练分区、inner选轮与fresh refit的原生状态、完整EMA的同训练/同query身份均审计。新进程冷见证禁止训练CSV读取；另一个独立原生审计核验仅来自相应训练分区的预处理/目标尺度、选轮、轨迹及保存状态。包由Q75父CSV的铁量字段原字符串直接复制，时长按冻结公式；ZIP仅result.csv，官方模板顺序、322唯一ID、有限非负和CRC正确。独立包审计重放新/旧模型及算术，不依赖封包模块的计算结果。

单worker/单数值线程，torch/interop及OPENBLAS/OMP/MKL/NUMEXPR均1，uv --locked --no-sync --python3.12使用已核验CPU环境，不同步依赖。RSS≤1536MiB，full-batch冷预测差0，逆序/37行分块≤5e−4。600秒观察；实际完成立即执行后续审计，无时间预算/自动重试。G0和实际退出完全闭合后才交付并登记探索队列。用户自行上传回传，助手不上传，不要求再次逐项授权。

私有运行根 `local/runs/sam-ema-time-exploration-20261002/release-r1`，精确身份与输入摘要在 `configs/sam_ema_release/SPEC.json` 和追加式本地manifest/ledger。
