# v0.15 / OPT-32–33：固定 QRF 时长分支

2026-09-12 登记；起点 `optimization-v0.14-h2-time@f940f9f4ed09382b7948f84145ceccd3f542cdd7`；本地分支 `optimization-v0.15-qrf-time`。本文件为执行前规格，不能当作完成记录。V1 活动发布、R2 回退及 V2–V7/D1 的关闭结论不变。

唯一候选 `V8_QRF_TIME`：铁量逐值复制原完整 V1，时长直接输出原 E09/R2 as-of 特征下的 QRF 0.5 分位数。无 E04 混合、rate 修正、LAD、残差、偏置、缩放、融合或分组路由。`D2_QRF_WEIGHTED_MEAN_TIME` 只作同森林加权均值诊断，不晋级。该完整分支替换同时涉及学习器、必要编码和输出方式，不能分别作因果归因。

固定森林完整参数见 [experiment.yaml](../../configs/optimization_v0_15/experiment.yaml)：256 树、squared_error 构树、min_samples_leaf=10、max_features=0.7、bootstrap=true、seed=2026、n_jobs=8，其余显式默认参数固定。协议 `QRF_FULLTRAIN_LEAF_v1`：构树使用训练块 bootstrap；随后每棵树投射全部原始训练行，各行每树恰好一次，分布不计 bootstrap 重数。每树总质量 1/256，叶子内平均分配，最终在训练响应支持上取 CDF 首次达到 0.5 的较小中位数，不插值、不平均逐树中位数、不无权拼叶子。接近半质量的浮点边界以原整数成员计数的有理数质量判定，不调整预测。

原特征 schema、列序/dtype/值、训练身份/排序、history、cutoff 和 source_contract 保持不变。使用原 Context.X / ComponentFeatures.X；拒绝 V5 轨迹列与 recency 权重。预处理仅训练数值列中位数填充，全缺失列保留填 0，训练 spout 词表 one-hot 加固定 missing/unknown 列。Inf、非法 dtype、ID 错位、float32 溢出阻断；不标准化、target encoding 或特征选择。raw/transformed schema/hash 分开保存。

独立 `workers/qrf_v015` 项目锁定 Python 3.12.12、sklearn 1.8.0、NumPy 2.2.6；scipy/joblib/threadpoolctl 解析版本在 manifest 中冻结。根 pyproject/uv.lock 不改。原环境恢复 V1/构造特征/处理账本/评分，worker 只接收带摘要和明确 ID 的训练与无标签评价 NPZ。仅加载本轮自生成且摘要验证的私有 joblib；森林响应及叶子成员只在被忽略 local/。

P0 先检查精确起点、干净注册树、原发布/旧模型/历史/OOF/回执摘要与等价登记；零 fit 重现 V1/R2 旧包及六个 origin，重建原 E/J 与分母，核验原训练矩阵。缺产物为 BLOCKED_MISSING_SOURCE，不能补训；存在等价关闭实验则停止重复。冻结源码/配置/锁/数据/旧证据/protection 后追加新账本。训练标签只从认证原 cutoff history 恢复，不直接读官方目标列。

预算：June–November 月初训练行 888/1180/1490/1803/2091/2424（身份核验，不凑行数）。每个 origin 一次预处理与一次森林 fit，开发共 6+6 fit、1536 内部树。DEV_LONG 复用 July 模型，DEV_SHORT 复用 September；无 April/May seed OOF、新 CatBoost/LightGBM/rate/q/LAD/后校准。每次 fit 前持久 intent、成功后存产物；失败尝试保留，不换目录重置预算，不以修复名义重新 fit。

六个 origin 的 V1/V8/D2 输出和全精度身份摘要全部落盘后，才读取已消费 outer 评分归档；November 标记 RETROSPECTIVE_POST_HOLDOUT_CONSUMPTION。P0 仅允许已有 V1 误差重建。测试标签与 test_a/b/c 分布选择禁止。铁量复制原全精度数组，不能六位量化后称 exact equality。日历月份 horizon，18 cells，J 按各 horizon 内等权 origins、再四 horizon 等权；DEV 不混进 J。

评分前固定门槛（delta=V8−V1）：工程/因果/身份/预算全通过、铁量 exact；H2 mean E≤−0.0005、至少4/5严格改善、目标月 September/October/November 至少2/3改善、单 H2 origin≤+0.0010；J≤+0.0002；H1/H3/H4 mean E各≤+0.0005；DEV_LONG/SHORT各≤+0.0007。H2 最近三项对应 August/September/October 模型。D2 无选择资格。1000 次共享 calendar-week 配对 bootstrap、seed=2026、重算分母、无效 draw 不补抽；只是已消费回溯稳定性描述。

独立 root+worker 冷审计验证可信摘要、保存/恢复、全量/反序/分块/子集/单行 exact equality、原 V1 铁量、原组件 tolerance=1e-10、E/J=1e-12；推理禁止全部 forest/preprocessor/CatBoost/LAD fit。原锁定全套测试和新 worker 合成测试分别报告，附件19项与旧342项不合并。诊断记录目标/月/铁口误差、signed mean/median、预测差分位数、唯一 ID 极端误差与 exposure、邻居 effective N/max weight/月质量、支持范围/边界比例、填充/未知 spout、模型字节/加载/预测时间/峰值内存。不能据诊断追加变体。

失败 `FAIL_CLOSE_V8_RETAIN_V1`，final/新 ZIP=0，不改变叶子/树数/分位数/融合/校准/种子重开。通过仅 `DEV_ACCEPTED_PENDING_OFFICIAL_IDENTITY`。正式包身份、来源与语义/训练范围/H2 兼容后才允许最多一个最终 forest+preprocessor，原2754行、cutoff 2024-12-01 01:44+08:00，不移动 cutoff 或补造 December 标签。旧包保持其原 README“正式发布包”身份；与09-21开放版本的关系独立核验。现在无正式新版核验，因此不 final fit/封包，不上传、不改 active/桌面包。

实际执行入口（实现后登记）：

```bash
uv run --locked --python 3.12 --extra dev pytest --junitxml=local/reports/pytest-v015-root-locked-r2.xml
cd workers/qrf_v015
uv run --locked --python 3.12.12 pytest --junitxml=/home/lux1/iron/local/reports/pytest-qrf-v015-worker-synthetic-r4.xml
cd ../..
uv run --locked --python 3.12 -m bf_tap.optimization.qrf_time_run --output local/runs/optimization-v0.15-opt32-r1
```

运行目录不可覆盖；独立冷入口由 runner 调用 `scripts/optimization_v15_cold_check.py --run …`。合成测试 fit 与赛事预算分别登记。公开数据历史尚待独立处置，本轮只本地/私有执行，不自动改变 visibility、清史或推送敏感数据。维护当前摘要以新增记录纠正 v14 已推送/成功 CI，旧报告不改写。

来源：[QRF 原论文](https://jmlr.org/papers/v7/meinshausen06a.html)、[sklearn 1.8 RF 官方参数](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestRegressor.html)、[复赛通知](https://www.aicomp.cn/notice/notice-3/5248.html)。这些支持方法/规则，不能保证本候选提分。
