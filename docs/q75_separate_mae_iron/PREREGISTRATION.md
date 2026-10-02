# 独立MAE网络铁量：控制线索的前瞻确认

2026-10-02。用户96.45目标持续授权，两个平台名额仍保留；参照为用户回传、未经独立核验的EMA_TIME_Q75=96.3920。

## 选择来源与不变的历史决定

SHARED_MAE开发未过原门槛，确认finalist仍为null。其SEPARATE_MAE机制控制的铁量端点相对Q75在split42/3407为+0.009192873/+0.004181399；相对已完成单目标固定MAE为+0.005489664/+0.006169938。这是观察结果后识别的线索，不能追溯声称原控制是预注册发布候选。当前单独预登记唯一候选SEPARATE_MAE_IRON_A20，再使用原先未对这套配方拟合的271828/314159完整五折做确认。开发样本/切分已使用，四split均来自同一官方样本池，不是四个新独立数据集。

## 科学范围、控制与预算

候选完整保留`q75_joint_mae`的SEPARATE_MAE科学实现：两个独立、相同初始函数的V33周期主干，两目标各自train-only标准化，损失为两项固定b=.5 Laplace NLL之和；等权两目标stdMAE选轮，fresh外层重训。铁量输出固定.8*Q75铁量+.2*member铁量，时长保持Q75，不发布模型自己的时长。

配对SINGLE_MAE控制完整保留`q75_fixed_point_losses`的IRON_LAPLACE_FIXED科学实现：同V33位置函数、固定b=.5 MAE、单目标stdMAE选轮。两套已验证训练器均逐字节复用；各自native training配置完整沿用。两者训练参数均初始化42、周期频率.01、两层128 SiLU、float64网络/float32预处理、AdamW .001/wd.0001、batch256、clip10、max240/patience30/min_delta1e-5、inner27001分组held0。两目标联合选择与全参数裁剪都是候选完整程序的一部分，不称纯选轮因果效应。

新确认两seed × 五fold × 候选/单目标控制，共20外层估计器、40optimizer、40selector/refit状态。原开发候选及单目标控制两个完整seed的物理source/seed/trial身份、OOF、warm/cold链只读复用，不补训或改名为新拟合。两套全形状合成G0已在本会话通过，源码/环境/配置核对后复用，0新增工程optimizer。锁定Python3.12测试阻止Optimizer构造，预算0。

Q75铁量两个确认seed从已完整原生审计的`ema-span-confirmation-20261001/confirmation-r1`读取，逐fold核对原分区、query ID、warm/cold链、铁量=.5*V36+.5*V12。开发与两份旧OOF逐项核对标签、ID、fold和Q75列。参照新fit为0，不重跑队友SWA确认。禁止跨split平均向量、调权重、选训练seed、改损失或增加候选。

单worker、四数值环境变量/Torch/interop均1、锁定CPU依赖不同步；RSS≤1024MiB。候选独立冷回读/反序/chunk37/NumPy前向≤1e-8，单目标控制明确tag读回并核对原位置reader；全部预处理/目标统计、inner选择与fresh refit轮数独立审计。独立math.fsum重算收益、fold/spout及seed单侧t LCB95，误差≤1e-10。所有进程真实退出；600秒运行观察、无时间预算/自动重试。新目录和追加保护访问账本，数据/源码/缓存身份冻结，不触碰初赛2024年11月目标。

## 事先门槛

正式质量门：相对Q75四个完整seed收益各严格正、seed层配对单侧t LCB95(df3)>0。额外全量发布资格还要求四seed相对SINGLE_MAE各严格正且LCB95>0。候选因观察控制结果而被选中的来源在最终报告保留，不改原共享阶段分类，不把未舍入本地收益当平台预测。

不对控制自动另开发布，不把任一单seed负收益藏进平均值。若不满足门槛保留具体失败结论，0新重试。当前阶段full fit/包/桌面/上传均0；合格后另冻全量与包审计。现有Laplace首测只是暂定，两个名额等充分证据及首项平台反馈后决定，不要求用完。
