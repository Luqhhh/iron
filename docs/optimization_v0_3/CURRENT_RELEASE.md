# 当前发布候选：E16（r2 平台回传后再次回退）

后续 r2 已完成 E12-raw 的正式全量训练，回传 82.5430 后经用户确认再次恢复 E16。
当前指针为 `configs/optimization_v0_3_r2/active_release.yaml`。r2 归档包及证据见
[OPT10_EXECUTION_R2.md](OPT10_EXECUTION_R2.md)。以下保留当时 E16 回退记录；
原 v0.3 导出命令仍只恢复 E16 归档，不导出 r2 新包。

用户确认回退后，active_release.yaml 与桌面同名 ZIP 均已恢复原 E16，
其平台回传成绩为 82.9918。新包 CB-FC-CVcal 的 82.2871 及全部实验产物
保留，不删除失败证据、不继续围绕平台总分调参。
桌面另存 `Luqhhh_bf_tap_predict_prelim.CB-FC-CVcal-backup.zip`；旧 E16
备份及原 local/submissions 包也保留。当前 E16 ZIP SHA-256：
`1200d4dd8dee6e797aeeba86db1ceb02293c78dd36796d2e8d75156ce77160aa`。

导出当前 E16 原包（精确归档导出，不是新训练或原始输入重新推理）：

```bash
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_active \
  --export-submission --stage test_a --output local/submissions/<unique-id>
```

## 以下为已归档的 CB-FC-CVcal 发布记录

2026-09-08，经用户明确批准替换。它是本轮最好新增完整算法，开发 J 为
0.1712251132，相对 C_ref 改善 .0004883073，未通过原 .0010 晋级幅度。
本次是 **USER_OVERRIDE_G1_FAIL**，不改写原门槛或宣称平台已超过 E16。
用户回传平台成绩 **82.2871**，比 E16 的 82.9918 低 **0.7047**。
证据状态为 USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED；未自动上传。
该总分不能单独定位底模、特征或
校准的影响，也不支持承诺后续全量训练一定弥补差距。

## 已交付

- 归档选择：`configs/optimization_v0_3/cb_fc_release.yaml`。
- 训练及一体化 bundle：`local/runs/optimization-v0.3-cb-fc-cvcal-release-r1/`。
- 新 ZIP：上述 run 的 `submission/Luqhhh_bf_tap_predict_prelim.zip`。
- 桌面同名 ZIP 已替换；桌面 `Luqhhh_bf_tap_predict_prelim.E16-backup.zip`
  保留旧 E16，原 `local/submissions/optimization-v0.2-opt06-e16-test-a-r1/` 也未改动。

新 ZIP SHA-256：
`7a18681495803b22f3a8a0156606f0297864be2cfced2fe0fe16339c65d3d36c`。
result.csv SHA-256：
`7d62409a4c433b1cab73de9593fb31cbb1042110a7ccd7b959a1b7fc6af73d91`。
bundle.json SHA-256：
`061714cd6e0bcf5476d5b39404492d48036e00e025fcd2718f85d91765270a74`。
旧 E16 ZIP SHA-256：
`1200d4dd8dee6e797aeeba86db1ceb02293c78dd36796d2e8d75156ce77160aa`。

## 训练与证据边界

Development 截止点仍为 2024-11-01，合法训练样本 2,424 条，没有读取
November 保护目标。这不是 final_training，也没有运行保护评分。
铁量 CB08 选定 115 轮、时长 CB02 选定 220 轮；新增 5 次单目标 fit。
两目标均使用 E09+F-C。早停、校准两块各自冻结历史；校准标签截至
2024-10-31 22:38 可用，299 个唯一样本，来源无重叠或晚于 origin 的违规。
时长残差重估为 10.710930574878446 分钟，铁量不校准；没有沿用 E16 常数。

test_a 为 335 行，对应 H2，全部从原始授权输入和显式 train/test_a 三列
样本索引生成。训练对象预测与新进程一体化恢复的最大差异为 0；交错分批
推理差异为 0。当前默认发布入口再次运行，result.csv 摘要也完全相同。
ZIP 已核验根目录仅含 result.csv，列名、ID、顺序、有限非负值均通过。
104 个锁定 Python 3.12 测试通过；G0 与原 G1_FAIL 分开记录。

## 显式复现归档 CB-FC-CVcal（非当前候选）

从仓库根目录执行，输出目录必须是新的、未存在的路径：

```bash
uv run --locked --python 3.12 python -m bf_tap.optimization.v3_active \
  --release-config configs/optimization_v0_3/cb_fc_release.yaml \
  --stage test_a \
  --sample-index 初赛数据集/train/train_samples.csv 初赛数据集/test/test_a_samples.csv \
  --output local/predictions/<unique-id>
```

必须显式指定归档配置并校验 bundle 摘要；默认活动配置已恢复 E16，
未指定 --export-submission 时会拒绝将归档 E16 冒充原始输入重新推理。
该 development 发布协议锁定声明的输入路径及内容身份，不自动扫描本地文件。
bundle 同时声明 test_b/test_c 的阶段输入角色，入口支持相应阶段，但本次只生成
并验收 test_a 平台包；未宣称已验收三阶段全量发布或完整 protected lifecycle。
旧实验配置中的 incumbent 保留历史平台含义，不代表当前活动发布指针。
