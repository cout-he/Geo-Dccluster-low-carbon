# 七天规则算法正式对照实验报告

**验收通过：五种RBC × 五种子，共25/25次；每次672步，总16800步。**
既有15次SAC仅离线读取，未重新运行。所有RBC的同seed任务流及外部输入SHA256均与SAC一致，且可从保存CSV独立重算。

## 配置与控制变量

2023-07-01 00:00 UTC至2023-07-08 00:00 UTC右开；首步07-01 00:00，末步07-07 23:45。五个DC原顺序、原资源、原物理模型、固定HVAC、Alibaba 2020原缩放、30天归一化、奖励和外部源文件均未改。
种子0、10、20、30、40；Seed42未计入。CPU，torch线程2。唯一仿真配置字段变化为simulation.strategy。RBC未加载checkpoint、未调用Actor、未训练。
详见control_variables.md与run_manifest.json；受保护源文件、配置、数据、checkpoint、原SAC核心结果指纹在运行前后保持一致。

外部输入流SHA256：`3c13373be09d6964b54713428f84caf59ecb2d6319871379162a978d56448a38`；模型仅作比较引用，SHA256：`c589fe58450f33f76a4d07a344997ab8a2f14ccba7f78f17fd8258616c69adca`。每seed任务流和源文件指纹详见run_hashes.csv。

## 指标及完整结果

Mean ± sample standard deviation，n=5，ddof=1。费用/能耗/碳含TX；水仅DC；SLA为违约完成任务/全部完成任务；CPU/GPU按DC和时间等权；Defer为有效决策次数。

| Controller | Cost USD | CO2 t | Energy MWh | Water m3 | SLA % | CPU % | GPU % | TX USD | Defer |
|---|---|---|---|---|---|---|---|---|---|
| RBC Local Only | 111899.980 ± 258.874 | 316.128 ± 0.728 | 1110.680 ± 1.692 | 7490.121 ± 4.133 | 0.020 ± 0.025 | 2.612 ± 0.007 | 14.888 ± 0.128 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| RBC Lowest Carbon | 104603.454 ± 70.350 | 292.814 ± 0.008 | 1108.309 ± 0.032 | 7325.784 ± 0.000 | 0.977 ± 0.000 | 2.615 ± 0.000 | 12.861 ± 0.000 | 4219.449 ± 70.350 | 0.000 ± 0.000 |
| RBC Lowest Price | 102695.098 ± 55.370 | 294.486 ± 0.009 | 1088.909 ± 0.019 | 7319.810 ± 0.000 | 0.996 ± 0.000 | 2.974 ± 0.000 | 11.851 ± 0.000 | 4253.825 ± 55.370 | 0.000 ± 0.000 |
| RBC Most Available | 122786.674 ± 28.803 | 321.680 ± 0.010 | 1204.819 ± 0.027 | 7473.994 ± 0.000 | 2.702 ± 0.000 | 1.802 ± 0.000 | 19.959 ± 0.000 | 3471.827 ± 28.803 | 0.000 ± 0.000 |
| RBC Round Robin | 114818.444 ± 26.041 | 320.672 ± 0.010 | 1114.764 ± 0.024 | 7527.226 ± 0.000 | 0.045 ± 0.000 | 2.685 ± 0.000 | 15.394 ± 0.000 | 3957.723 ± 26.041 | 0.000 ± 0.000 |
| SAC Geo Only | 111067.135 ± 81.802 | 304.846 ± 0.261 | 1110.866 ± 1.040 | 7374.355 ± 1.816 | 36.245 ± 0.306 | 2.367 ± 0.005 | 13.380 ± 0.065 | 4086.340 ± 51.489 | 0.000 ± 0.000 |
| SAC Time Only | 112039.832 ± 254.884 | 316.336 ± 0.736 | 1110.439 ± 1.699 | 7489.019 ± 4.233 | 34.793 ± 0.020 | 2.607 ± 0.007 | 14.872 ± 0.129 | 0.000 ± 0.000 | 820.000 ± 41.018 |
| SAC Geo+Time | 111076.893 ± 106.852 | 304.881 ± 0.240 | 1110.847 ± 0.742 | 7374.664 ± 0.458 | 36.546 ± 0.182 | 2.372 ± 0.005 | 13.375 ± 0.048 | 4080.000 ± 63.998 | 475.600 ± 10.015 |

## 主要观察及解释边界

规则组费用最低：RBC Lowest Price；碳排最低：RBC Lowest Carbon；能耗最低：RBC Lowest Price；水耗最低：RBC Lowest Price；SLA违约率最低：RBC Local Only。
不存在对九项指标都可直接宣布的统一最佳算法。CPU/GPU较高或Defer较少不能单独当作更优。

**重要局限**：Most Available实际选available_cores绝对数最大者，论文G.2描述为最高可用CPU比例，未擅自修改。
**执行路径差异**：RBC直接进入目标DC.pending_tasks；SAC执行动作进入in_transit_tasks后再入队，因此SLA差异包含投递时机影响，不能全部归因于调度策略质量。网络模型及TX统计代码均保留。
容量候选在DC实际调度前判断，同步分配之间没有增加资源预留；保留原队列和完成时SLA规则。七天/五种子/当前模型和归一化不等同于论文原始30天/十种子实验。

## 配对比较

paired_differences_vs_sac.csv共75行：5种RBC × 3种SAC × 5个相同seed。先逐seed计算RBC−SAC，再计算五差值均值和样本标准差；汇总见paired_differences_summary.csv。SLA及利用率差值为百分点。仅作描述性分析，不声明统计显著。

| RBC | SAC | Cost USD | CO2 t | SLA pp |
|---|---|---|---|---|
| RBC Local Only | SAC Geo Only | 832.846 ± 306.402 | 11.281 ± 0.908 | -36.225 ± 0.308 |
| RBC Local Only | SAC Time Only | -139.852 ± 23.233 | -0.209 ± 0.035 | -34.774 ± 0.030 |
| RBC Local Only | SAC Geo+Time | 823.087 ± 307.216 | 11.247 ± 0.852 | -36.527 ± 0.206 |
| RBC Lowest Carbon | SAC Geo Only | -6463.681 ± 115.513 | -12.033 ± 0.262 | -35.267 ± 0.306 |
| RBC Lowest Carbon | SAC Time Only | -7436.379 ± 268.776 | -23.523 ± 0.733 | -33.816 ± 0.020 |
| RBC Lowest Carbon | SAC Geo+Time | -6473.439 ± 85.405 | -12.067 ± 0.235 | -35.569 ± 0.182 |
| RBC Lowest Price | SAC Geo Only | -8372.037 ± 102.805 | -10.361 ± 0.262 | -35.248 ± 0.306 |
| RBC Lowest Price | SAC Time Only | -9344.734 ± 260.101 | -21.851 ± 0.733 | -33.797 ± 0.020 |
| RBC Lowest Price | SAC Geo+Time | -8381.795 ± 69.870 | -10.395 ± 0.235 | -35.550 ± 0.182 |
| RBC Most Available | SAC Geo Only | 11719.539 ± 85.141 | 16.834 ± 0.261 | -33.543 ± 0.306 |
| RBC Most Available | SAC Time Only | 10746.842 ± 234.567 | 5.344 ± 0.731 | -32.091 ± 0.020 |
| RBC Most Available | SAC Geo+Time | 11709.781 ± 104.841 | 16.800 ± 0.241 | -33.844 ± 0.182 |
| RBC Round Robin | SAC Geo Only | 3751.309 ± 86.965 | 15.826 ± 0.259 | -36.199 ± 0.306 |
| RBC Round Robin | SAC Time Only | 2778.611 ± 241.664 | 4.335 ± 0.732 | -34.748 ± 0.020 |
| RBC Round Robin | SAC Geo+Time | 3741.550 ± 101.949 | 15.791 ± 0.241 | -36.501 ± 0.182 |

## 任务完成与远程路由诊断

| Controller | new_arrivals | tasks_finished | unfinished_tasks | late_unfinished_tasks | remote_route_decisions |
|---|---|---|---|---|---|
| RBC Local Only | 26809.00 ± 0.00 | 26499.00 ± 0.00 | 310.00 ± 0.00 | 2.00 ± 0.00 | 0.00 ± 0.00 |
| RBC Lowest Carbon | 26809.00 ± 0.00 | 26499.00 ± 0.00 | 310.00 ± 0.00 | 2.00 ± 0.00 | 20353.60 ± 50.25 |
| RBC Lowest Price | 26809.00 ± 0.00 | 26499.00 ± 0.00 | 310.00 ± 0.00 | 2.00 ± 0.00 | 20596.80 ± 63.69 |
| RBC Most Available | 26809.00 ± 0.00 | 26499.00 ± 0.00 | 310.00 ± 0.00 | 2.00 ± 0.00 | 21759.20 ± 52.27 |
| RBC Round Robin | 26809.00 ± 0.00 | 26499.00 ± 0.00 | 310.00 ± 0.00 | 2.00 ± 0.00 | 21429.80 ± 71.35 |

未完成任务不进入完成任务SLA分母；已过期未完成任务另列。每DC收到任务数和实际DC顺序在每run的summary.json中；TX能耗/碳的均值和样本标准差在diagnostics_mean_std.csv中。

## 验收证据

- sac_offline_regression.json：原15次SAC九指标离线回归通过。
- seed0_gate_validation.json：五次首批完整运行及CSV重建指纹通过后才运行其余20次。
- validation.json：25次步数、窗口、任务守恒、有效Defer=0、Local Only零远程/零TX、哈希及九指标重算通过。
- validated_index_25_runs.json、run_hashes.csv：每run记录及完整输入/配置/原始结果指纹。
- 每run保存run.log、summary.json、steps.csv、inputs.csv、decisions.csv、runtime_config.json、sim_eval.yaml、run_record.json。
- figures/：九指标概览、逐seed配对差值及Table 3风格结果表；PNG/PDF/SVG，含绘图源数据和自动QA。

## 失败及异常

25次正式运行无失败、无提前终止、无重跑；全部正常完成。无任务流不一致或外部输入不一致。Most Available及入队路径差异为已知保留语义，未伪装为论文精确复现。

## 文件

本目录包含metrics_per_seed.csv、metrics_mean_std.csv、table3_combined.csv、table3_combined_numeric.csv、paired_differences_vs_sac.csv、paired_differences_summary.csv、diagnostics_mean_std.csv、run_hashes.csv、run_manifest.json及上述验收文件。
图表为报告用途，不声称满足指定期刊的投稿排版要求。

## 最终图表及交付复核

三份图表已完成预览检查，PDF碰撞和字体检查通过；两个多面板图对齐检查通过。配对图首次标题重叠已修复，仅重新导出图表，未重跑仿真。完整图表检查说明见figures/qa_notes.md。

普通SAC执行动作是在本步入站队列处理之后才进入in_transit_tasks，最早下一步才能投递；内置RBC可以本步入队和调度。这是代码调用顺序的已知差异，并未通过额外消融量化其对SLA差距的贡献。
