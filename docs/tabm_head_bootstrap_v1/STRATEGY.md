# TABM_HEAD_BOOTSTRAP_V1

用户已批准后备机制。原V12随机联合periodicTabM K16、两目标标准化、预处理、AdamW恒LR.001、max240/patience25/min_delta1e-5、raw joint stdMAE选轮和fresh refit全部保持；只在实际fit训练batch为每row/head独立Poisson(1)权重，两目标共用。同头权重不跨batch持久，属于在线bootstrap扰动，不称固定重采样独立模型。固定B×16×2分母，不按抽样权重和归一；全零batch不重抽不跳步，数据梯度零仍执行原AdamW与weightdecay。

专用NumPy RNG为random_seed+1000003，selection/refit各重置，不消耗模型初始化或原minibatch RNG；仅训练batch抽样，校准/查询/测试预测均原无权重head均值。每epoch保存累计权重/样本次序摘要和数量；独立新进程从fit IDs和冻结seed反向审计权重，不调用训练或优化器。原BASE真实缓存复用，不用协方差MSE作控制，不叠加SWA/EMA/Huber/协方差或扫描参数。

原Q75平台96.3920为用户回传；刚完成JOINT_COV时长四seed本地门83f44d4尚无平台分，不替换incumbent。工程2states/2optimizer、开发42/3407全五折10outerfits20states20optim，条件确认271828/314159同预算。两个目标各固定A20=.8Q75+.2member，完整两seed各正且机制mean vs raw BASE正者中均值最高唯一目标进入确认，平局时长优先。最终所选目标四seed各正/seed单侧tLCB95df3正/机制mean正才本地正式；另一目标描述，最多1探索推荐，轻微负机制保留探索与正式门分开。

开发铁量沿已审native Q75锚点与同ID/折；确认原铁量=.5V36+.5V12逐值核对；四seed时长沿原包源审计绑定。保护文件、全部源/运行环境/数据/参照/控制/原账本预冻结，官方标签读取前追加访问账本，禁止2024年11月保护标签访问。已有切分同样本不是四新独立数据，本地不预测平台。singleworker1024MiB、四数值线程及Torch1，无时间预算。只优化训练，0fullfit/封包/上传/桌面；仅占位及完整结果push，不重复历史fits。
