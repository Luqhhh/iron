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

目前新增全量拟合、Optimizer、CV、包、桌面写入、助手上传与平台队列新增均0。G0为旧输入及下述控制器工程检查通过，G1没有新全量或平台质量证据。四切分通过亦不自动授权全量发布。

## 发布准备已完成，执行尚未准入

[单候选脚本](../../scripts/ema_short_release.py)复用现有fit_component、audit_component、verify_saved及原CSV/ZIP工具，仅接入固定SHORT配方。全量账本使用预声明−1/−1作用域标记，与CV身份分开；内层及初始化seed仍42。脚本保留一个训练程序、两个Optimizer、训练内选轮与fresh-refit、独立冷审计和独立包回读。没有扩展候选、权重或平台排程。

39项定向与1464项锁定Python3.12全库检查通过，源码及702份两树公共Python文件前后摘要一致。全库保留24条既有测试警告。新增合成接线调用原ComponentRegressor.fit/_train，使用不更新参数的mock优化器；验证专用全量身份、两次初始化和两份原选轮/refit状态，不证明正式全量模型质量。

运行准备入口在禁止fit/_train/fit_predict下通过，绑定377份源文件、382份模型相关源码、27份输入和2754/322行训练/query分区。私有输出`local/runs/ema-short-release-preparation-20261001/preparation-r1`的manifest SHA为`471f07c25b8228c1bdc23f53dead614b3901f1f4c31b5f84d670ca0c0b4e082d`，scientific_execution_admitted=false、authorization=null，没有activation、模型或包。

全库最终收据为本工作树`local/research/release-controller-full-r2/receipt.json`，SHA `91565fe7197a7333e8167be2ae6072673aa67632b20898ff5fede2b069211a80`；运行准备收据为`local/research/release-runtime-preparation-r1/receipt.json`，SHA `094edd44e3f3d6a9c3092d4ca25276f94f2af90d06cc4ecf8502274ce5885285`。中间失败原样保留：合成query重复特征、mock优化器源码未绑定，均在进入优化器前拒绝；首轮全库临时目录放在local内导致两项历史边界测试失败，修正测试调用后全库通过，没有改动原训练器或放宽门槛。

按用户“不要过分工程化”的要求，本轮到此停止扩展发布实现。后续工作以正在运行的确认结果及平台测试价值为主；只有具体全量发布授权与完整确认终态满足后，才创建新的正式运行清单。当前准备清单不能激活训练或封包。
