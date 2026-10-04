# TabM-packed 零拟合结构核对

2026-10-04，当前平台参照为用户回传、未独立核验的 DE3_IRON_EMA_MEAN3_Q100=96.3979，目标96.45。EMA-mini科学批次仍按冻结协议执行；本次只作后续结构问题的准备，没有读取其局部结果或激活第二个训练批次。

[作者论文3.3节](https://arxiv.org/html/2410.24210v3)的packed变体使用独立MLP骨干，训练时按整体集成表现选轮；作者总体结果仍支持共享权重的TabM，不据此预设packed更好。它与改变小批量顺序、增加外层训练seed或仅增加内部成员数有区别。本地已安装[作者实现](https://github.com/yandex-research/tabm)tabm0.0.3提供原生`arch_type='tabm-packed'`；代码明确数值嵌入仍共享，因此带PLR的本地结构不能称为所有参数完全独立的深度集成。

先冻结私有零拟合范围，再在锁定Python3.12/torch2.14.0+cpu及单数值线程环境中构造三个原生模型：21数值列、3类别、单输出、width256、两层、k16、dropout.1、PLR频率.01、embedding/frequencies16、lite。每个只对4行合成零输入执行一次eval forward，输出均为有限`(4,16,1)`。参数量如下；另一个无torch进程用层尺寸公式及参数shape乘积独立核对。

| 结构 | 共享嵌入 | 骨干 | 输出head | 总参数 |
| --- | ---: | ---: | ---: | ---: |
| TabM | 864 | 178,224 | 4,112 | 183,200 |
| TabM-mini | 864 | 158,256 | 4,112 | 163,232 |
| TabM-packed | 864 | 2,445,312 | 4,112 | 2,450,288 |

三次未训练forward的进程峰值353.375MiB不代表训练峰值；尚未核验完整batch反传、AdamW/EMA状态、选轮、fresh refit或冷重放。G0只达到结构/参数清点，未取得科学训练准入；G1无测量。尚未登记科学候选权重、确认预算或平台名额，不能以此跳过后续完整工程和质量门。

当前worktree在e3a0ff1时的src/configs/docs未检出`tabm-packed`命名路线；该限定检索不证明所有历史实现都没有等价结构。若后续选择研究，仍需结合mini完整结果、当前参照、旧证据和信息价值另行冻结精确阶段。

私有证据位于`local/runs/tabm-packed-research-20261004/architecture-r1`及`architecture-r2`。r1脚本名inspect.py遮蔽Python标准库，导入阶段actual exit1，0模型构造；原文件与失败收据保留。r2只改入口文件名，3次构造/3次合成forward成功，独立参数核对成功，两程序actual exit0。report SHA256为`f5bdccfc29fac88682fef6fb4cd464a634b93d2f67645b37906dd056cb05bd0f`。全部过程0拟合、0optimizer、0官方数据读取、0确认标签、0包、0桌面写入、0助手上传。
