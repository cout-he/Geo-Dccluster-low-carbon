# Paper SAC v1

本工程复用原 SAC 网络、奖励及五 DC 物理仿真，独立实现论文配置、预检查、顺序训练、事务式断点恢复、三种随机策略评估及分层结果汇总。原始配置、核心源码、工作负载、旧权重和旧结果均不修改。开发只在 `feature/paper-sac-v1`，基线 `master` 保持不变。

本次开发仅运行小规模验证，没有运行 1000 万步训练或 27 次正式评估。正式命令在下文供服务器使用，不会自动执行。

## 配置与论文差异

全部配置位于 `configs/experiments/paper_sac_v1/`，不能直接编辑旧 `configs/env/`。

| 项目 | 值 |
|---|---|
| 训练日历 | 2023-08-01 05:00 UTC 起 7 天，每 Episode 672 步 |
| 测试日历 | 2023-07-01 00:00 UTC 起 30 天，每次 2880 步 |
| 步长/数据 | 15 分钟；2020 workload 月日映射至 2023 外部数据 |
| 训练种子 | 42、43、44，每个全新初始化，依次执行，每个 10,000,000 步 |
| 网络 | 34 维任务观察、6 动作；MLP 64→64 ReLU；Twin Q；无 Attention/LayerNorm |
| SAC | gamma=0.99，固定 alpha=0.01，tau=0.005 |
| 学习率 | Actor=1e-4，Critic=3e-4，Adam |
| Buffer/批次 | 100000 / 512；max_tasks=400；warmup=1000 |
| 更新/保存/日志 | update=1，policy=2，save=50000，log=1000 |
| 训练内评估 | eval_frequency=0、eval_episodes=0；不创建额外评估环境、不对 0 取模 |
| DC | 固定 DC1～DC5；资源、人口权重、时区及 dc_config.json 沿用原值；关闭 RL HVAC |

奖励沿用原负向计算公式，权重/组件因子如下：

| 组件 | 权重 | 参数 |
|---|---:|---|
| energy_price | 0.5 | normalize_factor=100000 |
| carbon_emissions | 0.3 | normalize_factor=10 |
| transmission_cost | 0.2 | normalize_factor=1 |
| sla_penalty | 0.1 | penalty_per_violation=5.0 |
| transmission_emissions | 0.1 | normalize_factor=1 |

三层处理必须分开理解：组件自身除以固定因子；`CompositeReward(normalize=false)` 加权组合，不再对组件做在线归一化；训练对最终标量继续使用与 `train_SAC_agent.py` 一致的 RunningStats 更新并归一化，才写入 Replay。评估的物理指标不使用归一化奖励替代。

论文 Appendix G.3/Table 9 与本次算法参数一致；正文 Figure 3 描述 1M/5 runs，Table 3 报告 10 seeds，而本次用户明确要求 10M×3 训练与 3 测试种子，因此本工程是指定的新实验协议，不宣称已复现表中数值。附录 Time Only 的 `k+1` 与仓库的 1-based origin ID 不一致，代码按实际 origin ID 1～5 回源。

## 时间与数据适配

原脚本估算 `month*30.5+day`，集群 reset 又随机化 DC 日期/小时，而任务时钟保持固定；CI、天气、电价的插值/时区索引也不一致。原 DC 先执行物理步再推进管理器，仅核验管理器当前值不足以证明真实物理输入正确。

独立适配层使用真实 day-of-year、UTC 时间索引和按时间插值；在物理步前设置对应 UTC 温度、湿球温度及电池 CI，检查 `agent_dc.dc_exterior_ambient_temp` 与 `agent_bat.bat_avg_CI`。原 IT/HVAC/Battery 公式不变，湿球温度仍是原来的 `0.9*temperature` 近似。原 DC 先执行物理步、再释放/调度任务、更新负载的顺序保留：该间隔的功率使用进入本步前的负载，利用率报告本步调度后状态。

默认 `reset_mode: fixed` 严格使用配置日期；`randomized_aligned` 保留原有 ±7 天、随机小时的种子机制，并让任务、外部数据同时切换到实际日期。默认 `weather_augmentation: false` 使用真实时间天气；显式开启可保留旧天气扰动，两种协议必须分别报告。预检查按所选模式检查所需时间范围，测试覆盖两种模式。

配置的 workload 字面路径不存在时，只读查找 `result_df_full_year_2020/result_df_full_year_2020.pkl`，再查 ZIP 中同名成员；不会解压或修改源数据。manifest 记录实际源路径和哈希。原 loader 的 `scale=1, task_scale=5, group_size=1` 保留；task_scale 改变资源需求而非任务个数。缺失 workload 时间槽沿用原语义视为零到达，并在报告中给出数量；重复或非 UTC 对齐时间槽直接报错。

来源 DC 的人口/本地活动概率不变，随机流按 episode seed 和 UTC 时间槽派生，避免策略采样或 SAC Buffer 采样改变后续任务来源。评估还比较任务来源/日历指纹，保证同一测试种子下三个策略配对。

## 本地小规模验证

在仓库根目录使用独立虚拟环境。Windows 可将下文 `python` 换成 `outputs/paper_sac_v1/.venv/Scripts/python.exe`。脚本从任意工作目录启动时，配置中的相对路径仍按仓库根解析。

```bash
python -m venv outputs/paper_sac_v1/.venv
# Linux: source outputs/paper_sac_v1/.venv/bin/activate
# Windows PowerShell: outputs/paper_sac_v1/.venv/Scripts/Activate.ps1
python -m pip install -r scripts/paper_sac/requirements.txt
python scripts/paper_sac/preflight.py
python -m pytest tests -q --basetemp outputs/paper_sac_v1/pytest-new-unique-id
python scripts/paper_sac/train.py --smoke --seeds 42 --stop-after 32
```

每次 pytest 使用新的 `--basetemp` 名称，避免 pytest 清理已有验证目录。`--smoke` 只在内存中覆盖 total_steps=64、Buffer=128、batch=8、warmup=8、save=32、log=8、CPU；观察任务上限仍为 400。Smoke Episode 为 16 步，便于在 32 步完整停止并恢复，正式 YAML 保持原样。

从控制台记录的 RUN_ID 中找到 `checkpoints/seed_42/RUN_ID/latest.json`，用其中的完整检查点恢复：

```bash
python scripts/paper_sac/train.py --smoke --seeds 42 \
  --resume outputs/paper_sac_v1/checkpoints/seed_42/RUN_ID/step_000000032.pt
```

若此前发布被打断，检查点可能有 `_retry_...` 后缀，始终以 `latest.json` 或 Buffer `commit.json` 的已提交路径为准。

完成 64 步后，只对一个模型、一个测试种子做三策略各 8 步验证：

```bash
python scripts/paper_sac/evaluate.py --smoke --steps 8 \
  --checkpoints outputs/paper_sac_v1/checkpoints/seed_42/RUN_ID/final_step_000000064.pt
python scripts/paper_sac/summarize.py --evaluation-run EVAL_RUN_ID \
  --allow-incomplete --include-smoke
```

输入检查点必须来自当前源码/数据指纹。源文件变化后旧快照被拒绝；这不会删除旧文件，应为新版本重新执行 smoke。两种正式入口都要求显式 `--production`，未给运行模式不会开始训练或评估。

## memmap 与完整恢复

| 数组 | dtype | 正式字节数 |
|---|---|---:|
| obs | float32 | 5,440,000,000 |
| next_obs | float32 | 5,440,000,000 |
| actions | int64 | 320,000,000 |
| mask_obs | float32 | 160,000,000 |
| mask_next | float32 | 160,000,000 |
| rewards | float32 | 400,000 |
| dones | float32 | 400,000 |
| 合计/seed | | **11,520,800,000（10.73 GiB）** |

Buffer 保存相对路径、布局、指针、有效长度、事务代号及最近检查点。抽样只复制所需批次。检查点先 flush/fsync Buffer，再原子保存网络状态，最后原子发布 Buffer `commit.json`。更新已提交槽位前，先把旧行写入 undo 日志并 fsync（POSIX 还 fsync 目录）；两个日志交替使用，避免每次全量复制 11.52 GB。打开时使用 OS 排他锁，恢复提交后的未提交覆盖，忽略尚未完成的日志尾部。掉电/进程退出会自动释放锁，不依赖删除锁文件。

正式三个 Buffer 约 34.56 GB；WAL 上限按 `min(capacity, save_interval+episode_steps)` 的保护槽位计算，每个 seed 最多保留两份相邻事务日志，预检查再加模型/检查点磁盘余量。恢复按实际分配块抵扣已有文件，保守计算稀疏文件剩余增长。RAM 检查针对抽样/激活工作集，memmap 页驻留由操作系统管理。

完整检查点保存 Actor、Critic、Target Critic、两组 Adam、global_step、episode、best Avg10、最近 10 Episode 奖励、RunningStats、Python/NumPy/Torch CPU/CUDA RNG、网络/动作元数据、完整配置及哈希、源码/数据指纹、Replay 提交元数据。恢复后重置下一 Episode 的确定性环境，再保持学习器随机流位置，不假装保存了中途环境对象。相同 CPU 环境的参数/下一次更新完全一致性已有自动化验证；跨硬件/CUDA bitwise 一致性未验证。

每 50000 步产生保存请求，在随后首个 Episode 边界发布完整状态；best Avg10 每 10 Episode 比较，保存独立历史推理快照。初始 step 0 和结束恰逢边界也保存完整状态。**只允许最近已提交完整检查点恢复训练**；旧检查点对应的 Buffer 行可能已被后续事务改写，旧/最佳/最终快照仍可评估，但不能宣称可完整恢复。

10M 不能整除 672，最后中途停止严格保持 10M 总步数，保存 `final_step_010000000.pt` 推理模型；最近完整边界仍可恢复并重放剩余步数。最终快照文件存在时不会覆盖它。所有中断保留 Buffer；没有自动 Replay 清理功能，也不删除历史结果。

## 评估与统计口径

- Geo Only：softmax 前屏蔽 Defer logit，从其余五个动作的条件分布采样。
- Time Only：从完整六动作采样；非零动作映射为实际 origin DC ID，保留 Defer。
- Geo+Time：完整六动作采样。每次评估使用独立的 CPU `torch.Generator`，不是 Argmax。

超期任务沿用原强制回源逻辑，优先于策略动作。`defer_decisions/tasks_deferred` 统计实际有效的 Defer 次数；`unique_tasks_deferred` 另报去重任务数；`policy_local_execution_count`、`forced_origin_count`、`remote_migration_count` 分开统计。迁移次数指有效远程路由决策，不等同于当步成功开始执行数。

总成本=DC 能源成本+传输成本；总 CO2=DC kg+传输 kg，再除 1000；总能耗=DC kWh+传输 kWh，再除 1000；水耗原模型为 L/15min，除 1000 后累加。所有物理字段都是每步增量，不再把环境中仅指传输的 `total_energy_kwh` 当成 DC 总能耗。CPU/GPU 为 DC 和时间点的等权平均百分比。

SLA 分母为已完成任务；没有完成任务时输出 null。截止时刻已完成但尚未被下一步 release 的任务计入完成数，且不增加新的物理步。未完成、未完成且逾期任务单独报告；不加载截止时刻的新到达任务。检查 `new_arrival_tasks == completed_tasks + unfinished_tasks`。

汇总保留 train_seed/eval_seed/strategy 三维：先在每个训练模型内汇总三个测试种子，再对三个模型的均值计算样本标准差 `ddof=1`。只有一个样本时 SD 为 null。拒绝重复单元格和不完整正式矩阵；smoke 必须显式允许，不能混入论文结果。

## Linux 正式训练命令（本次未执行）

先在服务器 checkout 本开发分支，确认没有修改原始数据；以下命令仅供之后手动运行。

```bash
git switch feature/paper-sac-v1
git status --short
python3 -m venv outputs/paper_sac_v1/.venv
source outputs/paper_sac_v1/.venv/bin/activate
python -m pip install -r scripts/paper_sac/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/paper_sac/preflight.py
python -m pytest tests -q --basetemp outputs/paper_sac_v1/server-check-unique-id
# 建议先重复上文 --smoke 流程，再由你手动启动正式任务。
python scripts/paper_sac/train.py \
  --config configs/experiments/paper_sac_v1/algorithm.yaml \
  --production --seeds 42 43 44
```

三个 seed 在同一进程依次训练，每个 seed 创建全新的网络、优化器、RunningStats 和 Buffer；不读 `colab_checkpoints`。单个 seed 的完整断点续跑：

```bash
python scripts/paper_sac/train.py --production --seeds 42 \
  --resume outputs/paper_sac_v1/checkpoints/seed_42/TRAIN_RUN_ID/COMMITTED_STEP.pt
tensorboard --logdir outputs/paper_sac_v1/tensorboard --port 6006
```

多种子任务中断后，按相应 seed 单独恢复；其他已完成 seed 不重新训练。未开始的 seed 可另建实验批次，例如 `python scripts/paper_sac/train.py --production --seeds 44`。正式模式只接受 42/43/44 的有序子集，默认训练全部三个。最终评估可传入不同批次的三个最终模型，保留其独立来源。

三个正式模型完成后，手动执行完整评估与汇总：

```bash
python scripts/paper_sac/evaluate.py --production \
  --checkpoints \
  outputs/paper_sac_v1/checkpoints/seed_42/TRAIN_RUN_ID/final_step_010000000.pt \
  outputs/paper_sac_v1/checkpoints/seed_43/TRAIN_RUN_ID/final_step_010000000.pt \
  outputs/paper_sac_v1/checkpoints/seed_44/TRAIN_RUN_ID/final_step_010000000.pt
python scripts/paper_sac/summarize.py --evaluation-run EVAL_RUN_ID
```

## 输出

`outputs/paper_sac_v1/` 内按 `checkpoints/logs/tensorboard/replay_buffer/evaluations/seed_N/RUN_ID/` 保存，各命令生成唯一 RUN_ID。恢复会追加原 run 的日志和 manifest，不覆盖历史检查点。`manifests/RUN_ID/experiment_manifest.json` 记录配置、配置哈希、每个数据文件 SHA256、源码文件指纹与 Git 版本、软件版本、训练/测试种子、开始/结束时间、实际步数、检查点及运行状态。

每次评估同时保存 CSV/JSON。`final_results/EVAL_RUN_ID/SUMMARY_ID/` 包含 `metrics_per_run.csv`、`metrics_per_model.csv`、`metrics_mean_std.csv` 和 `summary.json`。预检查单独保存 `manifests/preflight_.../preflight.json`。运行产物由本实验 `.gitignore` 排除，源码和配置仍正常跟踪；没有忽略或清理任何旧输出。

## 当前限制

新到达任务最大值在训练/测试分别为 279/320，但持续 Defer 队列上界可达 687/946。因此 400 尚不能保证任意策略下都足够，实际观察/Replay 入口会立即停止而不会截断；不能在未运行正式模型前宣称整段 30 天不会溢出。

原 SAC 的 next-task 目标按数组位置对应而非任务身份，本工程保留此算法以满足复用要求；原奖励中的 CPU×duration 代理成本/排放与评估的物理模型能耗不同，不修改其公式。湿球近似、资源负载的间隔顺序等原物理集成限制也保留。没有进行 Linux/CUDA、100000 容量 Buffer 的长期 I/O、正式三种子训练或正式 27 次评估验收。
