# DCcluster-Opt Table 3 SAC 复现

分支 `experiment/table3-sac-reproduction` 从基线 `c79e8f78a9229b904f123276168a9ef7746177fd` 创建。正式训练从随机初始化开始，运行 10,000,000 个环境步。正式评估用同一份 best checkpoint，分别运行 Geo Only、Time Only 和 Geo+Time，每种策略跑 10 个配对种子和每种子 2,880 步。默认输出目录为 `outputs/table3_sac_repro/`。

论文参数以 [NeurIPS 2025 正式版](https://proceedings.neurips.cc/paper_files/paper/2025/file/075661dddd5afc49e905b24e12e6a968-Paper-Datasets_and_Benchmarks_Track.pdf) 为准。奖励公式、仿真时钟和计量时点见 [奖励与仿真审计](reward_audit.md)。

## 配置与出处

| 配置 | 训练 | 评估 | 出处 |
| --- | --- | --- | --- |
| 基准时间 | 2023-07-01 00:00 UTC | 2023-07-01 00:00 UTC | 评估 G.1.1。训练时间由本任务指定 |
| 仿真步长与时长 | 15 分钟，30 天 | 15 分钟，30 天 | 步长及评估时长 G.1.1。训练时长由本任务指定 |
| Episode 起点 | 基准日 ±7 天，整点随机 | 固定起点 | 基线训练随机化，统一对齐全部数据流。评估固定采用 G.1.1 指定日期 |
| 工作负载 | Alibaba 2020 `result_df_full_year_2020.pkl` | 同左 | G.1.1。按 UTC 月、日、时、分查 2020 年轨迹 |
| 云网络与调度器 | AWS，`manual_rl` | 同左 | G.1.1 |
| DC 顺序 | 每次 reset 洗牌 | 固定为 Table 8 顺序 | 本任务采用 G.1.1 示例用于训练、正文建议用于评估 |
| 动作 | 每任务 6 个动作，含 0=Defer | 相同环境，按第 G.3.5 节调整采样 | G.3.1、G.3.5；基线环境为目的 DC 顺序索引 |
| 网络 | `34 → 64 → 64 → 6`，ReLU，双 Q＋目标网络 | 同一 Actor | F.2、G.3.3、Table 9；关闭 Attention 和 LayerNorm |
| SAC 学习参数 | γ=0.99，固定 α=0.01，Actor LR=1e-4，Critic LR=3e-4，Adam，τ=0.005 | checkpoint 记录 | Table 9 |
| SAC 步数与批次 | 10,000,000 步，warmup=1,000，replay=100,000，batch=512，update=1，policy update=2，max_tasks=400 | 30 天 × 96 步 × 10 种子／策略 | Table 9、G.1.3 |
| 硬件与日志 | device=auto，TensorBoard 开，save=10,000，CSV log=100 | device=auto，TensorBoard 关 | Table 9 的设备配置。日志由本任务指定 |
| 随机种子 | 42，记录 NumPy、Python、Torch、CUDA 配置 | 0–9，每个种子重置环境和策略采样 | 个数来自 G.1.3，具体种子由本任务指定 |
| SLA 与负载 | `task_scale=5, scale=1, group_size=1, sla_multiplier=1.5` | 同左 | 基线源码。CPU、GPU、内存按比例缩放，时长和带宽使用原值 |
| 任务来源 | `population_weight × activity_factor`，本地 08:00–20:00 左闭右开取 1.0，其余取 0.3 | 同左 | F.5 与基线 `utils/workload_utils.py` |
| IT、HVAC 与队列 | `configs/dcs/dc_config.json`，RL HVAC 关，HRU 关，FIFO | 同左 | Table 8 与基线源码 |

| DC ID | 站点 | 时区偏移 | 来源权重 | CPU 核 | GPU | 内存 GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | US-CAL-CISO | −7 | 0.18 | 50000 | 1000 | 80000 |
| 2 | DE-LU | +1 | 0.22 | 85000 | 600 | 80000 |
| 3 | CL-SIC | −5 | 0.20 | 110000 | 300 | 60000 |
| 4 | SG | +8 | 0.25 | 75000 | 700 | 50000 |
| 5 | AU-NSW | +11 | 0.15 | 65000 | 300 | 60000 |

| G.3.2 奖励分量 | 权重 | 分量参数 |
| --- | ---: | --- |
| Electricity price | 0.5 | 本步 DC 电费美元除以 100000 |
| Carbon emissions | 0.3 | 本步 DC 和传输碳排放 kg 除以 10 |
| Transmission cost | 0.2 | 本步传输费用美元除以 1 |
| SLA penalty | 0.1 | 本步完成且违约的任务数乘 5.0 |
| Transmission emissions | 0.1 | 本步传输碳排放 kg 除以 1 |

每项取负。Composite reward 关闭归一化，SAC 写入经验池前仍按基线运行统计预处理标量奖励。经验池保留完整环境步与基线的位置掩码目标。超过 400 条任务时记录溢出并保留任务，避免基线预分配数组的广播错误。Best checkpoint 按最近 10 个已完成 Episode 的平均奖励选出，正式跑完另存 final。

## 仿真与对照口径

- 训练旧入口未传入完整 `sim_config`。新入口显式传入六维、逐任务、允许 Defer 的环境配置。日期从真实日历换算为零基 day-of-year，reset 记录 UTC 起点、负载轨迹起点及站点输入索引。
- Geo+Time 从全动作空间分类采样。Geo Only 保留原始抽样结果，原始动作是 Defer 时从 1–5 的条件分布重采样。Time Only 的正动作映射到该任务的实际来源 DC，ID 已从 1 开始。
- 同区分配直接进入本地 FIFO。已超期任务的环境强制回源单独记录。保存原始动作、传入环境动作和最终动作，校验 Geo Only 的延迟次数及 Time Only 的跨 DC 分配次数都为零。
- Table 3 总费用为电费加 TX 费用，TX 同时作为独立列报告。碳排放与能耗包含站点和 TX 贡献。用 kg/1000 得 t，kWh/1000 得 MWh，L/1000 得 m³。SLA 分母为当期已完成任务，分子为其中违约任务。CPU 和 GPU 利用率在同一站点和时间步网格上等权平均。延迟动作次数和曾延迟的唯一任务数分别保存。
- Table 3 的 Geo Only SLA 为 25.42 ± 0.28%，与 §6.3 的零违约叙述冲突。Table 3 的 Geo+Time 碳排放与能耗列为 308 t、1037 MWh，Table 4 对同一行给出 308.7 t、1037.9 MWh。比对 CSV 使用 Table 3 的印刷值。相对差异在论文均值为零时留空。评估的 10 种子使用样本标准差。
- Figure 3 使用五次训练并只展示前 100 万步。本实验曲线是单次训练的实测全局步数和滚动均值。冒烟试验使用缩短的时间窗，不参与正式差异比较。

## 输入指纹

预检验证时间覆盖、数值、所选 AWS 费用矩阵和物理配置。训练与评估均记录 18 项输入的 SHA256。首次运行从 Git 跟踪的 ZIP 仅提取指定 pickle。基线 pickle 的 SHA256 为 `3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e`。

对本实验实际读取的五份 2023 碳强度 CSV，基线与公开官方 HEAD `3f6ea95cb835b89ba50b0ef76d66d14b8037643e` 的文件字节哈希均一致。该核对范围为这五个站点。论文原始运行的数据快照 SHA 未公开。 [官方代码版本](https://github.com/HewlettPackard/sustain-cluster/tree/3f6ea95cb835b89ba50b0ef76d66d14b8037643e)

| 站点 | 2023 碳强度 CSV SHA256 |
| --- | --- |
| US-CAL-CISO | `44bafab775c6b0d4762b1dc971eb293cfea154f80799a5230f9268d32e1ad55b` |
| DE-LU | `b6040328169313ca91b28806f3d69191cb6c035f45a7bcdd338a7c090d914dcb` |
| CL-SIC | `1f5287c0143ec2638f67ed8f367a13cddfc25f3ba3307c49dd4df6e49d8fc44c` |
| SG | `d668136330b50887e9736d194586b984b71f32c3192382b7df65739a37839de3` |
| AU-NSW | `cc6cea81513c83442219f9b1891d0d514e7fe7a6d0b15fc7447b2fad4b359809` |

## Ubuntu 安装与运行

从仓库根目录运行。使用带 `venv` 的 Python 3.11 或 3.12。下例使用 3.12 及 PyTorch CUDA 12.8 wheel。CPU 主机将索引路径中的 `cu128` 改为 `cpu`。训练和评估自动选择可用设备。可从 PyTorch 的 wheel 列表中选与实验机驱动匹配的索引。 [PyTorch wheel 版本表](https://pytorch.org/get-started/previous-versions/)

```bash
git fetch origin experiment/table3-sac-reproduction
git switch experiment/table3-sac-reproduction
python3.12 -m venv .venv-table3
source .venv-table3/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-repro-table3.txt
python -m pip check

python -m pytest -q tests/test_repro_table3_*.py
for script in scripts/repro_table3/*.sh; do bash -n "$script" || exit; done

config=configs/repro_table3
out=outputs/table3_sac_repro
smoke=outputs/table3_sac_smoke/ubuntu
python -m scripts.repro_table3.preflight --stage train --config-dir "$config" \
  --output-dir "$smoke" --smoke-steps 8 --env-smoke
bash scripts/repro_table3/run_train.sh --config-dir "$config" --output-dir "$smoke" --smoke-steps 8
bash scripts/repro_table3/run_eval.sh --config-dir "$config" --output-dir "$smoke" \
  --checkpoint "$smoke/checkpoints/final.pt" --smoke-steps 8 --seeds 0 1
bash scripts/repro_table3/run_plot.sh --output-dir "$smoke"
```

正式训练和评估分别启动。确认训练元数据的运行结束状态和 `final.pt` 的步数为 10,000,000 后再执行正式评估。

```bash
bash scripts/repro_table3/run_train.sh --config-dir "$config" --output-dir "$out"

bash scripts/repro_table3/run_eval.sh --config-dir "$config" --output-dir "$out" \
  --checkpoint "$out/checkpoints/best.pt" --seeds 0 1 2 3 4 5 6 7 8 9

bash scripts/repro_table3/run_plot.sh --output-dir "$out"
```

中断后在原输出目录恢复。恢复状态包括模拟器、经验池、优化器、目标网络、计数、日志位置和随机数状态。保留训练代码、配置、输入、依赖和设备。硬中断最多回退到最新完整恢复点。使用自己生成的恢复文件。

```bash
bash scripts/repro_table3/run_train.sh --config-dir "$config" --output-dir "$out" \
  --resume "$out/checkpoints/resume_state.pkl"
```

`--help` 可查看各命令。三个 Bash 入口先运行预检，并通过 `tee` 实时写入 `run_train.log`、`run_eval.log` 和 `run_plot.log`。输出目录可指定绝对路径。若显示 GPU 未启用，先核对预检记录的设备及驱动。

## 产物和同步

| 阶段 | 产物 |
| --- | --- |
| 训练 | `train_episodes.csv`、`train_steps.csv`、`train_reward_components.csv`、`checkpoints_manifest.csv`、`training_metadata.json`、`training_config_snapshot/`、`checkpoints/` |
| 评估 | `metrics_per_seed.csv`、`metrics_mean_std.csv`、`step_metrics.csv`、`dc_step_metrics.csv`、`action_statistics.csv`、`action_decisions.csv.gz`、`paired_differences.csv`、`paper_table3_comparison.csv`、`eval_manifest.json` |
| 绘图 | `figures/` 下的 300 DPI PNG 和矢量 PDF |

默认 Git 忽略正式输出与冒烟输出。先用 `rsync --dry-run` 从本地预览，再同步。保留服务器端训练恢复状态，按需单独下载。

```bash
mkdir -p table3_sac_repro
rsync -avh --partial --dry-run --exclude='resume_state.pkl' \
  labuser@lab-server:/absolute/repo/outputs/table3_sac_repro/ ./table3_sac_repro/
rsync -avh --partial --exclude='resume_state.pkl' \
  labuser@lab-server:/absolute/repo/outputs/table3_sac_repro/ ./table3_sac_repro/
```

## 启动前核对

- 分支和基线 SHA 正确，记录当前提交，工作树无预期外的修改。
- 锁定依赖安装成功，训练设备可用，训练输出磁盘至少预留 25 GiB，确认可承载经验池和恢复文件写入峰值。大规模训练的吞吐和资源占用需要在实验机上实测。
- 数据预检通过，18 项输入指纹可追溯。训练和评估指纹一致。
- 单元测试和短冒烟训练、保存、评估、绘图通过。冒烟输出路径与正式输出分开。
- 正式选用这一轮训练的 best checkpoint，保留 final 及哈希清单。正式评估检查每组每种子 2,880 行、5 个 DC 每步齐全、动作约束和完整的三组配对。

正式数值仍依赖服务器运行。复现采用基线输入、修正后的奖励、时钟和本地入队行为。论文未提供训练 checkpoint 的确定 SHA 或原始输入清单。历史表格和本实验的数值差异需要结合这些记录解释。
