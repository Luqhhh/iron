# EMA-mini 的条件确认缓存核对

2026-10-04。当前开发批次仍按原冻结协议运行；本次没有读取其局部结果。只核对历史JSON收据与原始文件哈希，不加载预测数组、pickle或torch状态，不读新确认标签，不启动确认。

历史缓存为local/runs/ema-span-confirmation-20261001/confirmation-r1，split271828/314159各五fold。复用旧五成员缓存清点的身份清单并重新核对1,680个见证文件、1,234份源码和20个OLD_EMA原生selector/refit状态；原manifest、成功终态和逐单位warm/cold收据匹配。原seed42的训练设置、EMA机制与本阶段旧控制一致，五个核心实现文件也逐字节相同。

这只证明当前文件身份与元数据条件。激活确认后仍需独立新进程冷推理、真实分区/选轮和原生步数核验，不能以旧cold收据替代本次冷重放。

对主local/runs目录执行训练seed1042/2042及split271828/314159的文件路径检索，命中180个路径，属于20个旧LAPLACE_INIT1042/2042单位；原warm收据绑定的metadata确认它们是另一配方，不能当作EMA控制复用。没有发现符合此命名检索的EMA附加成员；该结论只覆盖所声明目录和带seed的路径，不证明全机所有缓存不存在。确认激活前还须按完整模型身份排查可复用缓存。

若当前两个完整开发split均正、综合证据选出一个确认候选，并另行冻结确认协议，且旧seed42冷重放通过，则条件预算为：

| 组成 | 新估计器 | 新optimizer构造 |
| --- | ---: | ---: |
| mini三训练seed × 两确认split × 五fold | 30 | 60 |
| 旧完整TabM EMA附加训练seed1042/2042控制 | 20 | 40 |
| 合计 | 50 | 100 |

旧seed42是10个估计器/20个原生状态的复用，不再计入新拟合。表格尚未激活；若额外缓存可用或冷重放失败，须在正式启动前冻结实际预算，不自动重试或追加拟合。旧失败决定与当前开发四seed晋级要求保持。

本批仅改时长，当前DE3铁量在同样本的配对分数增量中相消。因此确认时可只报告`50*(WMAPE旧时长−WMAPE新时长)`；不需要为了该增量额外重训DE3铁量，也不据此报告未重建的完整包绝对分数。

私有证据位于local/runs/ema-mini-20261004/conditional-confirmation-cache-r1，SPEC先于哈希清点冻结。collect、独立元数据/预算核对及匹配配方补充核对均实际exit0；report、independent-audit、matched-recipe-audit与terminal-reconciliation闭合。独立核对重验2,990项文件身份。新增拟合、预测调用、数组加载、确认seed、包、桌面写入、助手上传均0。

## 已登记 worktree 的补充检索

另行冻结 `local/runs/ema-mini-20261004/registered-cache-search-r1/SPEC.json` 后，检索扩展到58个已登记worktree对应的5个实际local/runs目录；只读取start.json、metadata.json、warm-meta.json、warm-metadata.json，提前排除整个当前mini目录。共清点4,540份历史元数据，第二个锁定Python3.12进程逐一重验文件哈希。按路径及嵌套身份字段同时匹配确认split271828/314159、训练seed1042/2042，命中80份元数据；可见配方标签只有LAPLACE_INIT1042/2042以及嵌套的GAUSS1，没有EMA匹配。这些文件不能计为新的EMA控制缓存。

本次范围仍受四类元数据文件名、已登记目录、未遍历嵌套目录符号链接等限制，不是全机不存在的证明，也不据此激活确认或修改50估计器/100optimizer的条件预算。没有读取当前mini产物、预测数组、模型或新确认标签。两个核验程序实际exit0；report SHA256为`246ff0572821408fd435c9074c6c1eda6bc06ec0d97393eb37ee2192cd39f8ae`。
