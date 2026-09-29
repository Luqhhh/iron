# RFM 数值核心准备，2026-09-30

本分支 `codex/rfm-kernel-preparation` 在 E-COMPOSE 运行期间推进独立准备。
三级目标仍为平台 **96.4 / 96.45 / 96.5**；当前最高记录仍为用户回传
V32_TIME_A60V7_50=96.3727，缺口分别为0.0273/0.0773/0.1273。
这些缺口不是本地筛选阈值，也不是预期收益。

## 本次实际进展

实现 [rfm_kernel.py](../../src/bf_tap_r2/rfm_kernel.py) 和
[数学测试](../../tests/test_rfm_kernel.py)：float64 Laplace核、数值坐标导数、
未中心化梯度外积均值（AGOP）、trace归一化与1%单位阵收缩、训练配对距离中位数、
固定0.01对角正则的单次Cholesky求解。

控制设计来自另一分支已提交的用户批准方案：
[DESIGN.md](https://github.com/Luqhhh/iron/blob/86ea6b3/docs/rfm_metric_learning/DESIGN.md)
及[实施计划](https://github.com/Luqhhh/iron/blob/86ea6b3/docs/rfm_metric_learning/IMPLEMENTATION_PLAN.md)。
该分支还登记 RFM→PTaRL→DNNR→DANet 的研究顺序；本次没有重新占用编号或改排序。
这里先完成可独立验证的数值模块，供后续集成；没有把远端另一环境的缓存路径、
运行记录或30分钟监控约定直接移植到本机。

作者的 [xRFM 算法说明](https://github.com/dmbeaglehole/xRFM/blob/0cea9ba107c0a26dc1376a4c61c0d0bfa3e0ee1e/ALGORITHM.md)
给出核回归、梯度外积和迭代度量学习。本项目方案只测试固定带宽的全局RFM，
没有实现作者的树分区或自适应带宽，不能称为完整xRFM论文复现。

## G0：已验证与待验证

锁定 Python3.12、现有环境、四个数值线程变量均为1，定向测试 **12 passed**。
数学见证覆盖：

- 独立逐元素公式与中心有限差分，而非调用同一导数实现作参照；
- K(x,x)=1、对称性、类别坐标只影响距离、不进入学习矩阵；
- 重合中心贡献为0，其他非重合中心仍贡献梯度；
- 259行跨越128查询块，反序/7行分块结果完全相同，输入数组没有被改写；
- 重复中心下 `(K+.01I)alpha=y` 残差小于1e-8，不乘样本数、不自适应加jitter；
- 常数非零梯度产生非零AGOP，21维metric的trace=21、最小特征值>=.01；
- 正配对距离的精确中位数，以及非有限、非对称、非PSD等输入拒绝。

距离在metric平方根坐标内直接计算，避免平方距离展开式在重复/接近样本处相消；
梯度每次处理至多128个查询，不分配 n×n×21 张量。当前只以小型人工数组检验数学，
没有测量2754行数据上的运行时或峰值内存。

完整锁定测试套件、原生缓存重放、分区隔离、模型保存/冷加载、完整资源准入、
预算账本与正式控制器 **尚未执行或实现**。12项定向检查不替代这些正式门。
G0整体尚未准入，G1没有新增观测，正式拟合/资源探针/标签读取/封包/上传均为0。

本次检查命令（在本分支工作树中）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
UV_CACHE_DIR=/tmp/iron-uv-cache UV_PROJECT_ENVIRONMENT=/home/lux1/iron/.venv \
PYTHONPATH=src uv run --locked --no-sync --python 3.12 pytest tests/test_rfm_kernel.py -q
```

## 机制去重带来的决策

拉取远端后，对96个origin引用的公开src/configs/docs进行了关键词去重，
并核读相关方案/结果。有限关键词检索不能排除异名或未推送实现。

| 方向 | 已有证据 | 本次处理 |
|---|---|---|
| 非对称辅助梯度投影 | V17 J2的两个确认划分均负，四划分LCB<0 | 不重跑 |
| 独立种子平均 | 已审计DE3铁量均值+.003480，时长+.001324且一划分负 | 不扩大种子网格 |
| 遮蔽重建 | `codex/round2-v20-masked-recon` 已有同步重建实现；编号登记称负面关闭 | 保留历史，不因本分支缺少源码而误认未试；本次未独立重算其结果 |
| SCARF对比预训练 | 此次有限检索未发现登记；原论文主要是分类证据 | 不占为优先队列，不因机制不同就承诺回归增量 |
| RFM度量学习 | 远端已有用户批准的有界设计，尚无公开实现 | 准备兼容该设计的纯数值核心 |

梯度投影的证据见 [V17结果](../round2_v17/RESULTS.md)；DE3来源为
[已完成结果](https://github.com/Luqhhh/iron/blob/8b30a16/docs/independent_ensemble_checkpoints/DE3_RESULTS.md)。
遮蔽重建来源为
[原始方案](https://github.com/Luqhhh/iron/blob/a6076af/docs/round2_v20/PREREGISTRATION.md)
和[分支编号登记](../round2_round_numbering.md)。SCARF的证据范围来自
[原论文](https://arxiv.org/abs/2106.15147)，不能据其分类结果预测本赛题平台收益。

## E-COMPOSE 隔离与后续

开始时真实 systemd 服务 `iron-independent-ensemble-checkpoints-20260929.service`
为active/running。其工作树为
`/home/lux1/iron/local/worktrees/independent-ensemble-checkpoints`，公开HEAD为8b30a16。
本次没有轮询其训练指标、重启服务、修改环境或修改其工作树。既有600秒观察约定保留。

只读源清单保存633个运行中工作树的源码/配置/脚本/锁文件摘要。
本分支产物单独存储；private source-inventory在本工作树
`local/research/rfm-kernel-preparation-r1/`，不入Git。
清单SHA256：`cc794fd8da77c06063ce8422c34b2ea7884914695391e93a15ffccc7a75077ce`。

下一步在本隔离分支实现RFM训练分区、保存/回读与有界账本，先做小型人工测试。
在E-COMPOSE及其审计、已获准确认流程终止前，不启动任何全尺寸探针或第二条正式训练队列。
之后核查本机真实引用缓存和成本，再冻结完整SPEC；不能仅凭数值核心测试通过就启动G1。
原有完整两开发划分、.01机制门、96.25本地工作门及四划分确认门均保留。
确认正面也不等于平台目标达成，平台结果仍由用户自行上传后回传。
