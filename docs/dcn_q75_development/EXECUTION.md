# DCN新阶段工程准入与执行

科学源冻结提交29cdd4a；原dcn-cross-preparation网络/预处理/训练器、独立NumPy验证器
及原准备SPEC逐字节复制并对照原工作树摘要。新增controller只记录原生初始化调用和
训练产物，未改变优化器、网络或选轮/fresh-refit。针对测试还独立比对带记录训练与原
训练函数：同初始随机数下预测逐元素完全相同。没有修改旧准备分支或其他运行任务。

锁定Python3.12相关检查32通过、0跳过，精确来源收据在
local/runs/dcn-q75-development-20261002/tests-r1。Python3.12.12、numpy2.2.6、
pandas2.3.3、scikit-learn1.8.0、scipy1.18.1、torch2.14.0+cpu；不进行依赖同步。
四数值线程变量、torch计算/interop线程均1。科学阶段使用已核验环境的同一精确
/home/lux1/iron/.venv/bin/python3路径，显式PYTHONPATH指向本工作树src。

完整尺寸人工探针只使用预登记seed63002及1762/441/2203/551分区，0官方标签读取。
2过程、4次optimizer、4状态闭合并通过新进程冷审计；两臂参数数7864、初始化摘要相同。
最大独立完整/逆序/分块预测差2.842170943040401e−14，冷峰值487.48MiB。
ADDITIVE/CROSS的人工query MAE为0.775869/0.110528，常数为3.953854；这些数字仅是
可学习性和完整尺寸工程见证，不支持真实数据排序或平台分外推。selector选235/239，
两臂都到240epoch上限，不把此记录称作充分收敛，也不据此改变配方。

探针actual controller PID297750 exit0，独立观察器PID297744；26.5563秒只描述耗时。
冷manifest/保存状态/轨迹/参数摘要及实际终态绑定保存在engineering-r1，不重复原512行
准备预算。开发读取标签前绑定protection.yaml和冻结manifest并追加本地访问账本；
官方外层/内层分区身份与同seedQ75铁量/时长均在首次科学拟合前冻结。

开发为42/3407完整五折、铁量/时长各两臂，单worker按seed/fold/target顺序执行。
80次optimizer和80保存状态的预约/实际初始化分别追加，不返还失败预算；每600秒
观察，真实完成即触发独立80状态冷审计、全OOF评分和终态回读。无时间预算、自动
重试或中间fold质量筛选。最终结论登记在RESULTS.md及状态文件，本工程准入不代表
开发完成或四seed晋级。
