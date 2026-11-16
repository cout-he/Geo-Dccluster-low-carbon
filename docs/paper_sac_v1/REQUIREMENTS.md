# 任务：为 Geo-Dccluster-low-carbon 开发独立的论文配置 SAC 训练与评估工程

我的仓库：

https://github.com/cout-he/Geo-Dccluster-low-carbon

本次目标是以论文 DCcluster-Opt 的 Table 3、Appendix G.3 为主要参考，重新训练三个独立的 SAC 模型，并为后续 Geo Only、Time Only、Geo+Time 三种策略评估做好代码准备。

**本次只负责代码开发、配置创建、测试和预检查。严禁启动正式的 1000 万步训练。**

---

## 一、开发前的要求

1. 首先检查当前 Git 分支、工作区状态和仓库目录结构。
2. 不覆盖任何已有实验配置、模型权重和结果。
3. 不修改原始工作负载文件 `result_df_full_year_2020.pkl`。
4. 尽可能复用现有仿真环境、SAC 网络、奖励函数和数据加载代码。
5. 新建独立实验目录，不要直接修改 `configs/env/` 下的原始配置。
6. 如果发现论文参数与现有代码实现不一致，先分析并在报告中说明，不能擅自修改论文参数。
7. 所有路径使用相对路径，兼容 Windows 和 Linux。

主要参考：

- `train_SAC_agent.py`
- `envs/task_scheduling_env.py`
- `simulation/cluster_manager.py`
- `rl_components/agent_net.py`
- `rl_components/replay_buffer.py`
- `rewards/predefined/composite_reward.py`
- `utils/checkpoint_manager.py`

优先以 `train_SAC_agent.py` 作为 SAC 训练实现参考。不要直接采用较早的 `train_rl_agent.py`。

## 二、创建独立实验目录

创建：

```text
configs/experiments/paper_sac_v1/
    train_sim.yaml
    eval_sim.yaml
    algorithm.yaml
    reward.yaml
    datacenters.yaml

scripts/paper_sac/
    train.py
    evaluate.py
    preflight.py
    summarize.py

outputs/paper_sac_v1/
    checkpoints/
    logs/
    tensorboard/
    replay_buffer/
    evaluations/
    final_results/
    manifests/
```

可以根据实际代码结构调整文件组织，但必须保持训练、评估、预检查和结果汇总相互独立。

## 三、仿真环境配置

### 1. 训练配置 train_sim.yaml

```yaml
simulation:
  year: 2023
  month: 8
  init_day: 1
  init_hour: 5
  duration_days: 7
  timestep_minutes: 15

  workload_path: data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl

  cloud_provider: aws
  shuffle_datacenters: false
  strategy: manual_rl

  use_tensorboard: true

  single_action_mode: false
  disable_defer_action: false

  aggregation_method: average
```

训练每个 Episode 为7天，即672个15分钟时间步。

### 2. 测试配置 eval_sim.yaml

```yaml
simulation:
  year: 2023
  month: 7
  init_day: 1
  init_hour: 0
  duration_days: 30
  timestep_minutes: 15

  workload_path: data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl

  cloud_provider: aws
  shuffle_datacenters: false
  strategy: manual_rl

  use_tensorboard: false

  single_action_mode: false
  disable_defer_action: false

  aggregation_method: average
```

测试30天，共2880步。

注意：

- 工作负载来自2020年数据。
- 外部电价、碳强度和气象数据使用2023年数据。
- 检查现有代码中的 `align_external_data` 等时间对齐机制，确认输入时间戳正确。
- 不要因为配置文件增加了某个参数，就假定现有代码已经支持它。
- 检查 `DatacenterClusterManager.reset()` 的随机日期和小时逻辑，以及 DC 内部时间更新逻辑，确认任务时间与外部数据时间的实际对应关系。不要未经验证就删除原有随机化机制。

## 四、五个数据中心

创建独立的 `datacenters.yaml`，复制现有五个数据中心的配置，不改变任何资源参数。

| DC | Location | CPU | GPU | Memory |
|---|---|---:|---:|---:|
| DC1 | US-CAL-CISO | 50000 | 1000 | 80000 |
| DC2 | DE-LU | 85000 | 600 | 80000 |
| DC3 | CL-SIC | 110000 | 300 | 60000 |
| DC4 | SG | 75000 | 700 | 50000 |
| DC5 | AU-NSW | 65000 | 300 | 60000 |

其他要求：

- 使用原始 `configs/dcs/dc_config.json`。
- 所有数据中心明确设置 `use_rl_hvac: false`。
- `hvac_controller_type: none`。
- 不修改 IT、HVAC、Battery 的物理模型。
- 保持数据中心顺序固定，确保动作1～5始终映射到 DC1～DC5。
- 不允许随机打乱 DC 顺序。

## 五、SAC 算法配置

创建独立的 `algorithm.yaml`。

```yaml
algorithm:
  name: SAC

  gamma: 0.99
  alpha: 0.01

  actor_learning_rate: 1e-4
  critic_learning_rate: 3e-4

  batch_size: 512
  tau: 0.005

  replay_buffer_size: 100000
  warmup_steps: 1000

  total_steps: 10000000

  update_frequency: 1
  policy_update_frequency: 2

  save_interval: 50000
  log_interval: 1000

  hidden_dim: 64
  use_layer_norm: false

  max_tasks: 400

  device: auto

  eval_frequency: 0
  eval_episodes: 0

  use_attention: false
```

网络结构：

- Actor：MLP，两个隐藏层，每层64个神经元，ReLU。
- Critic：Twin Q Networks。
- 离散动作 SAC。
- 不使用 Attention。
- 不使用 LayerNorm。
- 固定温度系数 alpha=0.01，不进行自动温度调整。

观测和动作必须检查：

- 每个任务观测维度：34。
- 动作维度：6。
- 动作0：Defer。
- 动作1～5：DC1～DC5。

必须在环境初始化后进行断言，发现维度不匹配立即停止。

**注意：** 训练期间不进行频繁的完整评估。`eval_frequency=0` 必须在代码中得到正确处理，不能直接对0取模。

## 六、Reward 配置

创建独立的 `reward.yaml`。

```yaml
reward:
  normalize: false

  components:

    energy_price:
      weight: 0.5
      args:
        normalize_factor: 100000

    carbon_emissions:
      weight: 0.3
      args:
        normalize_factor: 10

    transmission_cost:
      weight: 0.2
      args:
        normalize_factor: 1

    sla_penalty:
      weight: 0.1
      args:
        penalty_per_violation: 5.0

    transmission_emissions:
      weight: 0.1
      args:
        normalize_factor: 1
```

检查现有奖励注册机制，确认五个奖励组件均能正常实例化。

所有奖励组件沿用现有的负惩罚方向，不要改变其物理量计算公式。

另外：

现有 `train_SAC_agent.py` 使用 `RunningStats` 对最终组合奖励进行在线归一化，再存入 Replay Buffer。

保留这一训练行为，并在代码和文档中区分：

1. 各奖励组件的归一化。
2. CompositeReward 的组合。
3. 最终标量奖励的 RunningStats 归一化。

不要将这三者混为一谈。

## 七、三个独立模型的训练

正式训练设置：

```python
TRAIN_SEEDS = [42, 43, 44]

TOTAL_STEPS_PER_SEED = 10_000_000
```

要求：

1. 三个种子分别从随机初始化的 Actor、Critic 开始训练。
2. 每个种子独立初始化优化器、Replay Buffer、RunningStats。
3. 三个模型依次训练，不并行训练，避免同时占用过多内存。
4. 不加载之前的 `colab_checkpoints` 或其他历史模型。
5. 每个模型训练1000万步，总计3000万步。
6. 训练完成后分别保存三个模型。

训练过程中记录：

- Episode Reward。
- Avg10 Reward。
- Actor Loss。
- Critic Loss。
- Policy Entropy。
- Reward Components。
- SLA Violations。
- Tasks Deferred。
- Energy Cost。
- Carbon Emissions。
- CPU/GPU Utilization。

使用 TensorBoard。

## 八、Checkpoint 与断点恢复

这是本次开发的重点。

现有 `utils/checkpoint_manager.py` 主要保存 Actor、Critic 和优化器，不能满足完整的训练恢复要求。

请扩展独立实验使用的 Checkpoint 管理功能。

至少保存：

- Actor state_dict。
- Critic state_dict。
- Target Critic state_dict。
- Actor optimizer。
- Critic optimizer。
- global_step。
- 当前 episode 编号。
- best Avg10 Reward。
- RunningStats。
- Python random 状态。
- NumPy random 状态。
- PyTorch CPU RNG 状态。
- CUDA RNG 状态。
- 网络结构和动作空间元信息。
- 训练配置及其哈希。

还需要考虑 Replay Buffer 的恢复。

目前 `FastReplayBuffer` 使用预分配数组。按照100000容量、400个任务、34维观测，两个观测数组就约10.88 GB，加上动作和Mask后约11.52 GB。

因此：

1. 先计算并报告准确的内存需求。
2. 优先设计基于 `numpy.memmap` 的持久化 Replay Buffer，避免每次保存Checkpoint时复制十几GB数据。
3. 定期 flush Buffer，并保存其当前指针、有效长度和元信息。
4. 断点恢复时必须能重新打开对应Buffer。
5. 如果完整恢复需要额外磁盘空间，提前进行容量检查。
6. 不允许通过只恢复网络权重，却声称完整恢复了训练。

Checkpoint 保存策略：

- 每50000步保存最新训练状态。
- 每10个Episode计算Avg10 Reward。
- 保存历史最佳Avg10模型。
- 训练结束时保存最终模型。
- 训练只在Episode边界执行完整Checkpoint，避免恢复时丢失环境内部状态。
- 保证中断后能够从最近一个完整Checkpoint继续。

如果采用内存映射方案，训练结束后可以提供可选的Replay Buffer清理参数，但不能在训练未完成时自动删除恢复所需文件。

## 九、评估代码

创建独立的 `evaluate.py`。

```python
EVAL_SEEDS = [0, 1, 2]

STRATEGIES = [
    "geo_only",
    "time_only",
    "geo_time"
]
```

每个训练模型分别评估三个测试种子和三种策略。

总评估次数：

3个训练模型 × 3个测试种子 × 3种策略 = 27次。

每次30天、2880步。

### 三种策略的动作约束

必须正确实现以下协议。

**1. Geo Only**

- 不允许 Defer。
- 对 Actor 输出的 Defer 动作Logit进行屏蔽。
- 对剩余五个数据中心的概率重新归一化。
- 从条件分布中采样。
- 任务可以被路由到任意一个数据中心。

**2. Time Only**

- 保留 Defer 动作。
- 使用完整六动作策略进行采样。
- 如果动作是Defer，保持推迟。
- 如果动作是执行，则强制映射到该任务的 origin DC。
- 不允许远程迁移。

**3. Geo+Time**

- 使用完整六动作策略。
- 允许任务推迟。
- 允许任务跨数据中心调度。

以上评估采用随机策略采样，不要擅自改成 Argmax。

每次评估使用独立的 `torch.Generator`，保证随机过程可复现。

三个策略之间必须保持：

- 相同测试种子。
- 相同工作负载。
- 相同外部电价数据。
- 相同碳强度数据。
- 相同气象数据。
- 相同数据中心物理配置。

注意保留现有环境的 SLA 超期强制回源行为，并区分强制回源与策略主动选择本地执行。

### 评估指标

至少输出：

1. Total Cost (USD)
2. Total CO2 (t)
3. Total Energy (MWh)
4. Water Consumption (m³)
5. SLA Violation Rate (%)
6. Average CPU Utilization (%)
7. Average GPU Utilization (%)
8. Transmission Cost (USD)
9. Tasks Deferred

同时记录：

- 新到达任务数量。
- 已完成任务数量。
- 未完成任务数量。
- 未完成且逾期任务数量。
- 实际 Defer 决策次数。
- 强制回源次数。
- 远程迁移次数。

严格检查指标是否存在重复累计或单位错误。

SLA 分母按照已完成任务计算，未完成任务单独报告。

三种策略的结果分别保存，不能覆盖历史实验。

## 十、预检查与数据检查

创建 `preflight.py`。

在正式训练之前，必须完成以下检查：

1. 工作负载文件是否存在。
2. 五个数据中心是否全部加载。
3. 数据中心ID是否为1～5。
4. 外部电价、碳强度和天气数据是否存在。
5. 外部数据是否覆盖所需的训练和测试时间。
6. UTC时间戳是否正确对齐。
7. 仿真步长是否为15分钟。
8. 是否为Multi-action + Defer。
9. 观测维度是否为34。
10. 动作维度是否为6。
11. DC顺序是否稳定。
12. Replay Buffer容量是否足够。
13. 内存及磁盘空间是否满足要求。

特别检查：

`max_tasks=400` 是否足够。

利用现有工作负载加载和 `task_scale=5` 处理逻辑，检查训练和测试时间窗口内每个时间步的实际任务数，输出：

- Maximum Tasks。
- P95。
- P99。
- Mean。

如果任一时间步任务数超过400，必须停止预检查并报告，不允许通过截断任务来绕过问题。

不得修改原始数据集和 `task_scale=5`。

## 十一、测试要求

本次不进行正式训练。

只运行小规模测试：

1. 配置文件能否正常读取。
2. 环境能否正常初始化。
3. Actor、Critic 能否正常构建。
4. 是否得到34维观测和6维动作。
5. 是否能完成几个环境步。
6. 是否能正常写入Replay Buffer。
7. SAC更新能否正常执行。
8. Checkpoint能否保存。
9. Checkpoint能否恢复。
10. Geo Only、Time Only、Geo+Time三种动作约束是否正确。
11. 评估结果是否能正确保存为CSV和JSON。

增加必要的单元测试。

使用小规模测试配置，例如：

```yaml
total_steps: 64
replay_buffer_size: 128
batch_size: 8
warmup_steps: 8
max_tasks: 400
```

测试时不得修改正式配置。

执行相关 pytest 测试。

**严禁启动1000万步训练，严禁自动启动27次正式评估。**

## 十二、结果和实验记录

每个训练种子独立保存：

```text
outputs/paper_sac_v1/
    checkpoints/
        seed_42/
        seed_43/
        seed_44/

    logs/
        seed_42/
        seed_43/
        seed_44/

    tensorboard/
        seed_42/
        seed_43/
        seed_44/

    evaluations/
        seed_42/
        seed_43/
        seed_44/

    final_results/
        metrics_per_run.csv
        metrics_mean_std.csv
        summary.json

    manifests/
        experiment_manifest.json
```

每次实验记录：

- 训练种子。
- 测试种子。
- 使用的Checkpoint。
- 配置文件。
- 源码版本。
- 数据文件哈希。
- 软件版本。
- 实际开始时间。
- 实际结束时间。
- 实际运行步数。
- 实验状态。

统计时使用样本标准差（ddof=1）。

保留训练种子和测试种子两个维度，不要未经说明将27次运行简单当成27个独立训练模型。

## 十三、文档

创建：

`docs/paper_sac_v1/README.md`

写明：

1. 实验目的。
2. 配置参数。
3. 训练流程。
4. 测试流程。
5. Checkpoint机制。
6. 断点恢复方式。
7. Linux服务器运行方法。
8. TensorBoard查看方法。
9. 输出文件说明。
10. 正式训练命令。

正式训练命令只写在文档中，不要执行。

例如：

```bash
python scripts/paper_sac/train.py \
    --config configs/experiments/paper_sac_v1/algorithm.yaml \
    --seeds 42 43 44
```

具体命令根据最终代码调整。

---

## 十四、最终交付要求

完成后请给我一份开发验收报告，包含：

**A. 修改了哪些文件**

逐一说明新增和修改文件的用途。

**B. 最终配置**

列出最终使用的训练参数、测试参数和Reward参数。

**C. 代码审计**

说明发现了哪些原有代码问题，以及如何处理。

特别说明：

- Multi-action + Defer 是否正确。
- 观测和动作维度是否正确。
- DC动作映射是否稳定。
- Replay Buffer 是否存在任务数溢出。
- Checkpoint 是否支持完整恢复。
- 数据时间戳是否对齐。

**D. 测试结果**

列出实际执行的测试、通过情况和失败情况。

**E. 正式训练方法**

给出实验室 Linux 服务器上从环境检查到启动三个种子训练的完整命令。

**F. 未解决的问题**

如果有任何尚未验证的配置、数据或物理模型问题，明确列出来，不要隐瞒。

### 最终验收标准

本次任务完成的标准是：

- 独立实验配置全部建立。
- 独立训练代码可运行。
- 独立评估代码可运行。
- 三种策略动作约束通过测试。
- 断点恢复机制通过测试。
- 数据预检查通过。
- 小规模训练测试通过。
- 不影响历史实验。
- 没有启动正式训练。

**请先检查，再实施，最后测试。不要只给我方案，要实际完成代码开发。**