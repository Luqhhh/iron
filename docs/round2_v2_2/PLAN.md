# V2.2：时长前缀与浅树铁量

用户授权从 `9f880141` 新建 `round2-v2.2-checkpoint-and-iron-blend`。基准是已获用户回传96.0982的V2.1 B完整包：铁量C2 seed42/1500树，时长B3种子42/2026/2027各1500树。ZIP SHA `ec12c3663edf77fdc0d7e8e9fa7bb1aeb1d1c46db4107a8c6724f25c8039e1e2`。不能沿用离线selected双B3，也不以96.0537为本轮平台参照。

复核旧输入/源码/模型摘要、原折分、B包及CSV身份；不一致即停止，不补训旧模型。新增 `v2_checkpoint.py` 和目标独立选择函数，旧 choose_c2()/历史配置/报告不改写。

固定候选：铁量I1=D4（原3000树深4）和I2=C2/D4等权；时长T1=三个B3成员全部使用[0,1000)树前缀。只复用相同验证折的成员。先恢复完整B3并与既有OOF对齐，再计算前缀；不增加其他切点、权重或种子。前缀使用原包装器输入转换和目标scale还原，不修改或shrink源模型。

相对各目标现参照，两种切分pooled WMAPE均改善、至少7/10折改善、每切分每铁口退化不超过0.001。平局容差1e-6，铁量优先单成员I1。最后选中的候选仅做2000次铁口内ID配对重采样，两切分共用索引；只描述冻结OOF的样本组成敏感性。原I1资格应复现，不能称为新增独立确认。

真实拟合预算：新增CV为0，全量D4铁量最多1。发布时若铁量被安排交付才训练一次D4，不重训C2。时间T1直接引用原全量三个模型，逐成员绑定模型SHA、训练种子、源树数和ntree_start/end，清单绑定成员顺序与等权平均。未改目标列直接复制当前96.0982 B包的CSV字符串，新预测采用.17g。冷推理禁读训练标签，核验反序、分批和完整CSV字节。

用户随后确认：本轮剩余1次；此后固定用户自行上传回传，每日上限5次，不再重复询问上传方式或额度。若两目标均合格，按两切分中较小的WMAPE改善排序；差距<=1e-6时优先无新增全量训练的时长。只安排一个单目标包，其余候选保留离线记录。原C2、V2.1 A/B、所有模型保留，组合包需两个隔离包均高于96.0982且有额度后另行生成，当前不生成。

定向测试含1次小型合成拟合，另运行完整锁定Python3.12测试集。可选pytest_fit_audit插件分别记录本pytest进程观测到的估计器fit入口和顶层调用；子进程或未纳入监测的类不计入，不能把此统计解释为所有环境的全局拟合数。它们与真实比赛拟合账本分开。

```bash
PYTHONPATH=scripts BF_TAP_TEST_FIT_REPORT=local/reports/v22-targeted-fits-r1.json UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra dev --extra round2 pytest -p pytest_fit_audit tests/test_round2_v2_checkpoint.py -q
PYTHONPATH=scripts BF_TAP_TEST_FIT_REPORT=local/reports/v22-full-fits-r1.json UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra dev --extra round2 pytest -p pytest_fit_audit -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra round2 python -u -m bf_tap_r2.v2_checkpoint evaluate --output local/runs/round2-v2.2/checkpoint-r1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/iron-uv-cache uv run --locked --python 3.12 --extra round2 python -u -m bf_tap_r2.v2_checkpoint release --output local/runs/round2-v2.2/checkpoint-r1
```
