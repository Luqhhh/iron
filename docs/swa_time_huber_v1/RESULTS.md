# SWA_TIME_HUBER_V1 完整开发及确认结果

2026-10-03。开发和确认各10外层模型/20新保存状态/20次优化器流程完整完成，独立冷审计及终态零拟合复核通过。四seed对Q75增量均正，平均+0.003683563400696，seed单侧t-LCB95(df3)+0.001206941572002，相对原SWA平均机制增量+0.001163007530851。通过冻结本地正式门，成为优先候选；尚无平台成绩、全量拟合或封包授权。

## G0 工程通过

唯一改变为time标准化Huber2损失：|r|<=1时r²，超过1时2|r|−1，每head .5ironMSE+.5timeHuber2后取mean。中心区域与原MSE值及梯度一致，尾部梯度受限；delta1单候选预登记，不扫描。原V12jointperiodicTabM/random42/inner42/frequency.01、AdamW恒LR.001/max240/patience25/min_delta1e-5、原jointstdMAE选轮、freshrefit与last10完整epoch raw参数等权窗口全部不变，梯度raw，不含初始化。

7 targeted回归先RED后神经7pass、locked2pass5skip（无Torch），0科学optimizer。必要完整全测神经1461 passed/23 warnings；锁定Python3.12为1042 passed/96 skipped/3 warnings。没有重复旧批次全测或拟合。新合成G0消耗2states/2optimizer，新旧4模型cold通过，最大预测差2.52961386593142e-06，窗口差4.76837158203125e-07，RSS444.05859375MiB。

开发40个新旧模型cold：最大预测差1.29526275429726e-05，窗口差4.76837158203125e-07，RSS443.59765625MiB；确认40个新旧模型cold：最大预测差1.93230351044349e-05，窗口差4.76837158203125e-07，RSS445.4375MiB。均低于1024MiB。独立新进程禁fit/optimizer，核对损失身份/LR、10epoch窗口float64重算、真实选轮、训练校准query分区、scaler/词表、完整逆序chunk预测及预算。

原SWA_WINDOW10控制工程2/dev20/confirm20状态真实只读复用，保留原source directory/phase/split/unit身份及模型/input/audit摘要，0新控制/参照fit。新候选总G02+dev20+confirm20保存状态与相同优化器次数；20个正式outer fits，不将42states写为42outer模型。

## G1 四seed通过冻结本地门

固定时长端点0.8Q75+0.2member，无权重搜索、校准或裁剪。

| split seed | Huber A20对Q75增量 | 原SWA A20对Q75增量 | 相对原SWA |
|---|---:|---:|---:|
| 42 | +0.002186735112343 | +0.004339763110890 | -0.002153027998547 |
| 3407 | +0.006781264411785 | +0.004547392986334 | +0.002233871425451 |
| 271828 | +0.002589626586258 | -0.000758008738967 | +0.003347635325224 |
| 314159 | +0.003176627492396 | +0.001953076121121 | +0.001223551371275 |
| 四seed均值 | +0.003683563400696 | +0.002520555869845 | +0.001163007530851 |

两开发seed平均+0.004483999762064，相对原SWA机制均值+0.000040421713452，两seed各正且机制mean正，才自动执行已冻结271828/314159确认。最终四seed各正、LCB95正和机制mean正，failure_reasons为空，formal_promoted=true。fold仅描述；相对原SWA并非每seed都改善（42为负），不能声称全面优胜或证明具体离群机制。

这里“正式”仅指本地冻结门，四split重用同一官方样本且此前已用于历史实验，不是四新独立数据集，不能保证平台泛化。Q75平台96.3920来自用户回传、未经独立核验；本轮无平台分，不将局部增量加到平台分预测成绩，不报告完整当前包本地总分。原UPDATE_WINDOW70仍是用户指定探索候选，原未晋级/确认0历史不改。

## 终态和发布

独立零fit/零optimizer/零模型推理复核每seed2754唯一ID及五折完整OOF、Q75绑定、端点算术、原控制字段逐值、source/runtime/reference哈希、原账本和新追加账本、各阶段计数和四seed决定通过；fit账本前后逐字节一致，没有覆盖原审计或重复cold。标签读取前登记protection与freeze摘要，仅使用官方R2缓存，不访问初赛2024年11月目标。

controller-finished为completed，绑定PID不活动；不声称已观测科学controller的OS waitpid退出码0。preflight SHA256：c208366ae4e42de1945b5d02de2388ebf725915c52efc6a28018c3f9ee7bd7b6。运行local/runs/swa-time-huber-v1，辅助local/swa-time-huber-20261003，最终复核final-verification-r1.json。模型/窗口/预测/标签/ledger全部私有local。fullfit/packages/uploads/desktop均0。完整实现及结果推送当前upstream后删除本轮定时检查，不擅自启动下一配方。
