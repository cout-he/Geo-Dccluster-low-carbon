# 控制变量及执行路径

| 项目 | 原 SAC 正式实验 | 本次 RBC |
|---|---|---|
| 年月日/小时 | 2023-07-01 00:00 UTC | 相同 |
| 结束/步长/步数 | 07-08 00:00，不含；15分钟；672 | 相同 |
| DC 顺序 | 1 US-CAL-CISO；2 DE-LU；3 CL-SIC；4 SG；5 AU-NSW | 相同 |
| DC、reward、物理模型、网络数据、workload文件 | 已冻结文件 | 同一文件，运行前后 SHA256 检查 |
| HVAC | SAC快照显式false/none | 原DC文件缺省false/none，实际断言关闭RL HVAC |
| 外部输入/归一化 | UTC对齐；30天参考窗口含右端2881点 | 相同；逐run验证 |
| 种子 | 0、10、20、30、40 | 相同；Seed42排除 |
| simulation.strategy | manual_rl | local_only/lowest_carbon/lowest_price/most_available/round_robin |
| 模型 | 冻结3620k SAC | 不加载，不推理，不训练 |
| 任务流验证 | 已登记 realized_workload_sha256 | 相同JSON序列化，在实际loader返回时观察；CSV可重新构造hash |
| 外部输入验证 | 已登记 external_inputs_sha256 | 相同逐步逐DC序列化；CSV可重新构造hash |
| 运行入口 | make_env → TaskSchedulingEnv → 手动动作 | 同一make_env → env.step([]) → cluster.step内置规则 |
| 新输出 | 原15次结果只读 | outputs/table3_7d_rule_baselines_20261112/ |

唯一更改的仿真配置字段是 simulation.strategy。RBC包装器只观察真实loader返回值，不重复调用、不重排任务、不拦截随机数。每次环境独立创建并reset(seed)，Round Robin状态随cluster.reset清零。

## 真实规则与局限

- Local Only：来源ID；本任务所有来源有效，未触发缺失来源的随机fallback。
- Lowest Carbon：先can_schedule检查，再比较当前原始碳强度。
- Lowest Price：先can_schedule检查，再比较当前原始USD/MWh；未做额外转换。
- Most Available：先can_schedule检查，再选available_cores绝对数最大者。论文G.2写的是最高可用CPU比例；保留仓库实现，不称为该公式的严格复现。
- Round Robin：按数值DC ID排序循环；不使用容量过滤，进入DC待执行队列。
- 容量过滤发生在各DC实际step调度前，同一时间步的规则分配之间没有额外资源预留。未改变此机制。
- 内置RBC直接加入目标DC.pending_tasks；SAC在TaskSchedulingEnv经in_transit_tasks投递，连本地动作也经过该路径。两条路径的入队时机不同，因此SLA差距不能全部归因于Actor决策质量。网络计费/能耗计算代码未改。
- SAC的deadline强制回源在手动动作分支；RBC没有策略defer和待决全局任务，走原DC队列完成时的SLA统计。

九指标按SAC公式：DC+TX费用/能耗/碳；水仅DC；SLA分母为完成任务；CPU/GPU对DC和时间等权；defer为有效决策次数。未完成和已逾期未完成任务单列。

论文G.2规则定义已从本地论文文本核实。该实验是七天冻结条件的仓库基准对照，不等同于论文30天/10种子的数值复现。
