# RealMLP历史缓存接入范围核对

2026-10-04，目标96.45，当前参照为用户回传、未独立核验的DE3_IRON_EMA_MEAN3_Q100=96.3979。本次仅检查已闭合V9开发缓存、源码和依赖身份，不读取SiLU局部结果，不进行新质量评估或训练。

[V9原结果](../round2_v9/RESULTS.md)包含RealMLP的旧参照增量：铁量TD-S在原A35参照下通过四seed门，但未过该阶段96.25发布门；时长TD有两个完整开发split的正增量。原参照、权重选择与决定全部保留。这些结果不是相对当前DE3＋EMA三seed的收益，也不赋予新阶段资格。[旧模型库复核的范围限制](../q75_selection_scope_audit/RESULTS.md)同样意味着不能凭其“饱和”描述否决整个RealMLP家族。

接入核对先冻结私有scope，随后在锁定Python3.12中只用标准库读取闭合元数据、计算原始字节摘要及读取安装包元数据，实际exit0：

- 开发目录`local/runs/round2-v9-realmlp/development-r1`共44个文件：40份预测、manifest、fit ledger、summary和原审计。40个预定target/recipe/split/fold键各出现一次，账本均为complete；预测摘要同时匹配账本与旧审计。
- 两目标、TD/TD-S、split42/3407各五fold齐备；旧账本共记录80次optimizer。此处只核对旧计数和元数据一致性，不重新执行或独立重建当年的训练。
- 原runner、原spec、89个作者Python源码文件、128个项目依赖源码文件与原manifest逐字节一致；7项登记运行依赖版本一致。没有导入科学包、同步依赖或读取官方CSV。
- 该限定目录没有pt/pth/ckpt/pickle/joblib/safetensors模型文件；原runner源码也只保存预测和metadata。原合成冷推理通过不能替代这些科学估计器的保存状态与独立冷重放。本检查不推断其他目录不存在副本。

G0达到缓存字节身份与保存范围核对；训练分区digest沿用旧证据，本次未读官方数据重算。G1没有新增测量。没有解码预测向量，没有读取旧确认预测，也未选择候选、权重或新增平台排程。

后续若研究这一方向，先另行冻结完整同split的当前参照、候选池及稀疏权重、身份重建和评分协议。缓存可以作为来源明确的历史OOF材料；若要求新的权威复现或发布，还需定位原科学状态，或在独立阶段先完善保存/冷重放并冻结必要的新拟合预算。不能把新拟合冒充旧缓存，不能以缺少状态为由追溯删除旧质量证据。SiLU原阶段继续执行，不因本接入记录自动启动第二个训练批次。

私有证据：`local/runs/realmlp-cache-intake-20261004/intake-r1/{scope.json,intake.py,intake.json}`。新模型构造、拟合、optimizer、质量评估、官方数据读取、确认预测读取、全量拟合、包、桌面写入及助手上传均为0。
