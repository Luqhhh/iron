# DE3＋EMA三成员Q100桌面交付

2026-10-04。用户询问DE3能否稳定保留增益，并在名称核对后明确选择“DE3＋三成员Q100（现成探针）”。已按[冻结配方与预算](../../configs/de3_ema_mean3_q100_release/SPEC.json)完成0拟合列组合、独立包审计及桌面交付；五成员的原开发失败和未发布决定保持。

提交文件：`C:\Users\lqh22\Desktop\submission-DE3-IRON-EMA-MEAN3-Q100-20261004\Luqhhh_bf_tap_predict_round2.zip`。

- 候选：`DE3_IRON_EMA_MEAN3_Q100`。
- ZIP SHA256：`528b8bf91102bea7ce120a71c560f6b021382132fb425a8f95cbfe79f9e3712c`。
- CSV SHA256：`28ca0771ff8456867ffbfe53506d4ebd26708947ca30e068998f83044529acd5`。
- 私有包：`local/runs/de3-ema-mean3-q100-release-20261004/release-r1/DE3_IRON_EMA_MEAN3_Q100/Luqhhh_bf_tap_predict_round2.zip`。

铁量复制原DE3字段，和当前最佳96.3977包完全一致；时长复制已审计的EMA_MEAN3_Q100原字段，即 `V32_time+1.00*(mean(EMA42,EMA1042,EMA2042)-V7_time)`。仅改时长强度，三个成员固定为42/1042/2042；322行时长均与当前q=.75包不同。原最佳桌面包、原Q100探针、所有源码和冻结决定保留，没有重算预测或增加后处理。

G0通过：Python3.12.12生成与独立审计新进程均核对两个父包及既有独立冷审计、当前最佳包、模板、配方、源码和依赖锁文件，共11项冻结身份。新CSV/ZIP和桌面回读均仅result.csv、322个唯一ID官方顺序、预测有限非负、CRC正确；铁量对DE3与当前最佳、时长对原Q100字段字符串差异全部0，桌面与私有包字节一致。生成、独立包审计、桌面复制、独立桌面审计4阶段实际exit0。原父包冷推理证据复用，本次未执行新的模型冷推理或拟合。

G1为用户指定、平台未测的强度探索，未正式晋级。原Q100时长探针的完整两split增量−0.003689/−0.008504及not_shortlisted决定保持。当前最佳已经含DE3，因此比较本包与96.3977时不能再次加上DE3收益；能否提分取决于q=.75到1.00的时长变化，不提供平台点预测。

DE3的“稳定”仅限固定原铁量列、相同平台测试样本与可加评分。原DE3相对V32回传增量0.0022，新DE3＋mean3相对mean3为0.0023，两者与四位小数报分舍入相容。相同铁量列的得分贡献不随时长列变化；这不是独立测试集重复验证，也不保证新铁量配方或五成员时长提升。历史平台分数均为用户回传、未经独立核验。

本批0新拟合/optimizer/CV/确认seed、1新包、1桌面复制、0助手上传。用户自行测试并回传，不推断剩余额度。当前测试顺序用这个含DE3的新包承接三成员Q100问题，原不含DE3的Q100包保留但不重复占用优先位置。

私有证据根为 `local/runs/de3-ema-mean3-q100-release-20261004/release-r1/`，含manifest、release、package-audit、desktop-copy、desktop-audit和terminal-reconciliation收据；机器状态键 `de3_ema_mean3_q100_release_20261004`。
