# SAM＋EMA时长工程准入与开发启动

2026-10-02，科学源worktree `local/worktrees/sam-ema-time-development`，提交 `90350c9404d0dc33166802e4b88af97a824efcc3`。[冻结协议](PREREGISTRATION.md)检验SAM更新后一次EMA参数平均，原width256、损失、数据分区及其他训练设置保持。

锁定Python3.12定向检查26通过、0跳过，未运行全套测试。rho=0对原EMA、beta=0对原SAM的预测、参数、训练轨迹、选轮与RNG完全等价；独立冷状态、更新顺序、训练分区、完整OOF和真实进程reap检查通过。首次适配器25通过/1失败因SAM比较列未初始化，在任何拟合前修正；失败原日志/源码摘要与实际exit1保留。

完整规模合成工程程序及独立冷进程、supervisor实际exit0通过；只进行1次工程optimizer，0官方CSV读取、0科学拟合。

| 工程检查 | 实测 |
| --- | ---: |
| 240epoch/2204训练行/550 query行 MAE | 0.361078252 |
| 训练中位数常量query MAE | 5.418177179 |
| AdamW更新/梯度计算/EMA更新 | 2160/4320/2160 |
| 实际峰值RSS | 673.492 MiB |
| full-batch冷差 | 0 |
| 逆序/37行分块最大差 | 1.626300e−6 |

397份工程冻结文件重新哈希一致，实际主进程已退出；原合成配方与receipt不改写。工程manifest SHA256 `61101d70be2dd5a14567d252adbeae80cf78bfde767eb7c314efb9f9d5bde465`，实际退出收据SHA256 `5235971b4caf6868862932cbcbd1a00724cc617912d7156529742cdfc8d5e869`。

随后独立冻结正式开发：split42/3407各五折，10估计器/20次科学selector-refit optimizer/20新状态，复用40个旧EMA/SAM状态。每split独立完整OOF；两个完整切分对Q75及匹配SAM都为正后才准备确认。正式晋级仍须四完整split、各seed正及seed层配对LCB95>0。工程可学习不代表科学质量或平台增益，G1待完整开发和独立审计。

私有运行根 `local/runs/sam-ema-time-20261002/development-r1`，manifest SHA256 `1229e9be9cf51332a5db8e5629a6c561fbf4be564b0e7702d2d460721983acb7`。单worker/单数值线程、600秒观察、实际完成自动审计，无时间预算/自动重试。0新增确认seed、0全量拟合、0平台包、0桌面写入、0助手上传；既有未测组合探针和DE3替补安排保持。
