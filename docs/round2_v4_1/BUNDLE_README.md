# iron V4.1 delivery

这里包含更新任务书、V4旧OOF的零模型训练诊断工具，以及合成测试。

**本次已完成：** GitHub源码/结果审查，诊断代码实现，30项合成数学与模拟IO测试。
**本次未完成：** 在真实赛事OOF上运行本工具、V4.1新模型训练、完整嵌套验收、封包、上传。运行环境的测试分数不是比赛分数。

## 在原 iron 环境运行

解压后，将此目录放在任意位置。先进入原仓库根目录（其中有`复赛_train/`和`local/runs/`）。

```bash
uv run --locked --extra round2 python /绝对路径/iron_v41_bundle/diagnose_v4_oof.py \
  --root . \
  --source local/runs/round2-v4-mechanism-search/coarse-r2 \
  --output local/runs/round2-v4.1-strong-increment/diagnostic-r1
```

上面的脚本和CLI已存在；任务书列出的未来训练器尚未实现。替换目录占位符即可，不能用不存在的历史OOF文件伪造输入。

输入缺失、ID/group/数据哈希/fold哈希/历史指标重放不一致都会停止。既有输出目录存在也停止；重复运行使用新的输出目录，不覆盖证据。工具仅访问训练文件和本地OOF，无网络访问、没有`.fit()`调用、没有测试加载、没有平台上传。

输出：`direction_diagnostics.csv`、`diagnostic_shortlist_NOT_PROMOTION.csv`、`manifest.json`。所有输出都必须留在本地`local/runs/`，不要提交赛事逐样本数据。

## 如何解释

`oracle_alpha_same_labels`和`oracle_package_delta_same_labels`使用正在评价的同一份OOF标签求解，属于描述性乐观诊断。不能用作独立验证成绩、测试权重或平台提分预测；只有完整嵌套重测才能决定保留。

工具不把跨seed调权称为独立验证，亦不把权重优化称为0次统计拟合：仅基础模型训练为0，标量优化次数单独记录。

## 本地测试

```bash
python -m pytest /绝对路径/iron_v41_bundle/test_increment_diagnostic.py -q
```

开发测试环境：Python 3.13.5、NumPy 2.3.5、pandas 2.2.3、pytest 9.0.2；不同于原项目锁定环境，未宣称原环境端到端集成已经通过。30项测试含数学精确解、50/50误杀反例、零残差单边导数、随机折点校验、模拟76个seed-unit IO、哈希/指标/重复组/缺文件拒绝、旧文件不覆盖。全部数据为代码内构造，不是额外赛事训练数据。

本工具继承V4的数组位置身份，通过原数据/fold哈希和原指标重放检查；原`.npy`本身没有内嵌ID，因此不声称有比源产物更强的行级来源认证。
