# SHORT_SPAN 全量发布准备（2026-10-01）

当前平台参照为用户回传 Q75/Q100=96.3920，未独立核验；本线程目标96.45。SHORT_SPAN 正在[冻结的四切分确认阶段](../ema_span_confirmation/PREREGISTRATION.md)，尚无新增完整确认结果。本准备核查旧全量模型的可复用性，不启动全量拟合，不形成发布准入或新增平台测试包。

独立工作树为 `local/worktrees/ema-short-release-preparation`，分支 `codex/ema-short-release-preparation`，基于cc40016。原科学训练器和正在运行的确认工作树均未修改。

## 已核查的旧输入

私有探针在锁定Python3.12.12、torch2.14.0+cpu环境执行，四项数值线程和torch数值/interop线程各1。绑定22份输入及源码，校验原EMA全量时长selection/refit、原fit/audit/cold/独立封包收据链、Q75父包及原冷预测。五份核心模型源码与原诊断manifest及本工作树逐字节一致。

独立冷进程禁止fit/_train/fit_predict、Optimizer初始化和CSV读取，实际尝试均0。加载两个旧checkpoint核查原EMA beta=.99、完整训练设置、所选epoch91及原训练身份：selection2203行，refit2754行。322行query无目标、ID唯一，顺序与两个旧包一致。

- 原EMA refit完整批次冷预测与已有cold.npy差异0；反序、37行分块、单行最大差分别为0.0000008065、0.0000064523、0，均在原5e-4容差内。
- 用原V32包、旧EMA冷预测及原V7冷预测重建Q75，时长浮点值逐项相同；Q75铁量字段与原父包字符串逐项相同。
- 没有生成新预测文件或CSV/ZIP，没有读取官方目标或新增拟合。检查只证明旧输入的这一项复用路径，不证明新SHORT全量模型或平台收益。

私有证据位于 `local/worktrees/ema-short-release-preparation/local/research/release-input-probe-r1/`：manifest SHA为 `bf11e8a1296336277ff6461bcb50b77e4a0b109ceaa607bb4bc5ee6f8a6a0d31`；独立cold收据SHA为 `941406f9604e39755beb69adf17fb83cfd2226d99229ccbbc6422cce6db8ed92`；父收据SHA为 `c5e9dd6711012536b54aa6be6444b268733240d8acc430c4273c4e20e7632770`。

## 条件发布范围

拟保留确认阶段固定配方：`time=Q75_time+.75*(SHORT_full_time−old_EMA_full_time)`，SHORT beta=.9801，铁量直接复制Q75原CSV字段字符串。若后续获得具体全量发布授权，只需一个新时长训练程序、训练内选轮及fresh-refit，共2次Optimizer、2份新状态，0新CV/确认seed；旧EMA模型不重新拟合。单包范围，不组合EMA铁量或PTaRL。

实际执行前仍需完整确认终态、G0/G1判定、具体任务授权、发布控制器实现与必要检查，并冻结最新参照、源码/环境、官方模板顺序、数据和全部模型身份。新模型须保存状态、独立冷推理及原生.pt审计；最终包再由独立进程回读，核验322个官方模板顺序唯一ID、有限非负值、ZIP仅含result.csv且CRC正确、铁量字符串差异0。原数值/RSS门、600秒观察、无时间预算及不自动重试保持。

目前新增全量拟合、Optimizer、CV、包、桌面写入、助手上传与平台队列新增均0。G0仅为本节旧输入检查通过，G1没有新质量证据；发布控制器尚未实现。四切分通过亦不自动授权全量发布。
