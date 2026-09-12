# v0.15 P0 工程失败与恢复

首次登记 `5de6523`；P0 在读旧 `predictions/{month}_inputs.csv` 的 META 字段时失败：该文件实际只含 sample_id 与 R2/rate 预测分量，没有 spout_no/reference_time。此时 forest/preprocessor/旧模型/LAD 的比赛 fit attempt 均为0。原 `failure.json`、manifest、三份源码归档及已生成的 June train NPZ/JSON 均保持原摘要，不删除或覆盖。

修复登记 `b515046`：从认证原月度 OOF 的 META 读取样本，按真实日历窗口选择，按原预测 ID 顺序恢复，并检查唯一 ID 集合一致。原训练矩阵/标签/参数/预算/数据来源不变；既有 June 交接只读验证后复用。修复 manifest 绑定原/新源码、原失败、既有交接与新测试证据。只修改本阶段 qrf_time_run、cold wrapper 与测试，未修改冻结 builder、旧训练/推理或 worker。

根 r3 合成锁定测试为363 passed / 1 failed：测试期间新增源码摘要保护使冻结的合成生命周期夹具检测到 qrf_time_run 源码改变，原保护拒绝了执行；没有放宽检查。停止源码修改后 r4 364 passed。worker 初始错误 cwd 的收集失败，以及 r2 的24 passed / 1 failed（多余权重归一化引入一 ulp）也留存；按原每树权重公式修正实现，原容差保持不变，worker r4 25 passed。全部是合成/工程证据，不改写模型质量结果。

恢复后六个 P0 与原 V1/R2 包的旧推理最大差0、原矩阵摘要与 E/J/分母重建通过，随后才解锁6 forest + 6 preprocessor。六份模型一次成功保存，独立冷审计没有重 fit。历史质量十项门槛均失败，固定 V8 关闭；不因工程通过而晋级。

私有路径：`local/runs/optimization-v0.15-opt32-r1/engineering_repair/`。数字完成后的既定诊断仅读取已冻结 errors；原 ledger 数字完成快照 SHA 与 core completion 一致，追加的诊断访问保留前缀证明，最终回执分别绑定两者。旧阶段账本始终不变。
