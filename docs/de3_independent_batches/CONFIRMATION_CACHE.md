# DE3联合独立小批量的条件确认缓存清点

2026-10-04。当前两split开发仍在运行，局部质量未读取。本次仅做预登记的历史模型/收据身份与文件哈希清点，0拟合、0预测、0数组或pickle/torch对象加载、0新增确认seed及包。确认没有启动，也没有因本清点获得准入。

既有可定位缓存为local/runs/ema-span-confirmation-20261001/confirmation-r1的split271828/314159，各五fold。它们与更早7777/12011不是同一切分，不混用预测向量。前次缓存报告及原清单哈希匹配，400个原reference序列化状态重新核对原哈希；十个外层单位的V12_joint final和selector-terminal见证共20个状态，原source directory/split seed/fold/trial、训练/查询行身份及原冷完成收据保持。

这不等于20个原生BASE的selected-selector/refit状态对已经可直接复用。V12原类保存的是final及终止selector见证；其与当前DE3 seed42控制的源码、选择轨迹和精确模型复用兼容性仍须在独立G0衔接中证明。此次未反序/分块冷推理，也不把原冷通过收据当作本次新进程回放。

在两个明确的历史缓存搜索根（主local/runs、原独立集成worktree的local/runs）中，未找到同时对应271828/314159与训练seed104729/130363的候选文件名；这不是整个机器或其他会话都不存在模型的证明。正式预算冻结前仍须按完整模型身份排除额外可复用缓存。

若且仅若当前候选两个完整开发split均正，且另行冻结一个确认候选和具体协议，当前信息对应：新独立批次三训练seed×两split×五fold为30估计器；旧DE3两个附加训练seed控制为20估计器；seed42原V12桥接若不能满足G0则还需10个控制估计器。因此条件范围为50至60估计器、100至120次optimizer构造。该范围不是启动预算，也不缩减四完整split、全部正收益及seed层LCB95>0的正式门。

本批只改铁量。对同一split、相同样本和不变时长列，成对总分增量严格等于：

`50 * [sum(abs(y_iron - old_iron)) - sum(abs(y_iron - new_iron))] / sum(abs(y_iron))`。

时长项相消，因此这一铁量确认不需要为了成对增量而重训时长EMA的附加成员。铁量参照仍须精确重建当前DE3组件，不能用旧单seed V12替代。未重建时长列时不报告完整包绝对分数；上述恒等式也不预测平台收益。

私有证据位于local/runs/de3-independent-batches-20261004/conditional-confirmation-cache-r1：SPEC在哈希清点前冻结；collect.py和独立收据复核均实际exit0，只导入标准库；report.json及独立复核绑定来源、输入和预算算术。本次没有修改运行中的源码、清单、候选池或平台队列。
