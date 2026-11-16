# Paper SAC v1 开发验收报告

验收日期：2026-11-16。当前分支：`feature/paper-sac-v1`。基线分支：`master`，提交 `c79e8f78a9229b904f123276168a9ef7746177fd`。仓库不存在 `main`。开发验收时新增源码、配置及论文 PDF 已暂存供审阅；用户随后明确授权将本次工作提交并推送到 GitHub 同名开发分支。不会合并或删除分支。

**交付结论：独立工程、小规模真实仿真、断点恢复和三策略 smoke 验证已完成。项目 `tests/`：25 passed。不能将此结论表述为“全部验收条件无风险通过”：持续 Defer 时 400 槽位可能不足，全仓库 pytest 另有原有下载脚本收集错误；Linux/CUDA 和正式规模尚未运行。没有启动任何 1000 万步训练，也没有自动运行 27 次正式评估。**

## A. 新增和修改文件

没有修改任何原有核心源码、原有配置、物理模型、旧模型或旧结果。以下为新增文件；论文 PDF 原本未跟踪，按用户授权加入跟踪。

| 文件 | 用途 |
|---|---|
| `configs/experiments/paper_sac_v1/train_sim.yaml` | 独立 7 天训练配置 |
| `configs/experiments/paper_sac_v1/eval_sim.yaml` | 独立 30 天测试配置 |
| `configs/experiments/paper_sac_v1/algorithm.yaml` | 正式 SAC 参数，不被 smoke 覆盖写回 |
| `configs/experiments/paper_sac_v1/reward.yaml` | 五项论文奖励权重和组件参数 |
| `configs/experiments/paper_sac_v1/datacenters.yaml` | 五 DC 原资源参数，固定顺序，显式关闭 RL HVAC |
| `scripts/paper_sac/__init__.py` | 独立实验包，无导入即训练行为 |
| `scripts/paper_sac/config.py` | 配置读取、路径解析、维度/资源约束、内存 smoke 副本 |
| `scripts/paper_sac/data.py` | 只读 workload 路径回退、UTC 数据插值、时间覆盖和任务统计 |
| `scripts/paper_sac/environment.py` | 继承环境/集群，时间适配、任务计数、物理输入和来源随机流隔离 |
| `scripts/paper_sac/policy.py` | 三种策略随机采样及动作约束 |
| `scripts/paper_sac/learner.py` | 复用 ActorNet/CriticNet 的离散 Twin-Q SAC 和标量 RunningStats |
| `scripts/paper_sac/replay.py` | 原 dtype memmap、WAL 回滚、原子提交、OS 排他锁 |
| `scripts/paper_sac/checkpoint.py` | 完整状态保存/恢复、RNG、指纹验证、拒绝推理快照恢复 |
| `scripts/paper_sac/metrics.py` | 单位换算、SLA 分母、CSV/JSON、分层样本统计 |
| `scripts/paper_sac/provenance.py` | 唯一运行目录、数据/源码 SHA256、软件和 Git 元信息 |
| `scripts/paper_sac/preflight.py` | 数据、时钟、维度、任务上限、内存/磁盘预检查 |
| `scripts/paper_sac/train.py` | 三种子依次训练、边界检查点、恢复、TensorBoard、smoke 保护 |
| `scripts/paper_sac/evaluate.py` | 独立配对随机评估，完整矩阵要求显式 production |
| `scripts/paper_sac/summarize.py` | 单个评估批次的逐模型和跨模型汇总，拒绝误混 smoke |
| `scripts/paper_sac/requirements.txt` | 独立环境依赖清单，不修改原 requirements |
| `tests/test_paper_sac.py` | 配置、策略、Buffer、恢复、RNG、指标、统计和入口保护测试 |
| `tests/test_paper_sac_environment.py` | 真实 DC、固定/随机 UTC、任务守恒、发布中断故障重放 |
| `outputs/paper_sac_v1/.gitignore` | 仅排除本实验运行产物/虚拟环境，不影响旧输出 |
| `docs/paper_sac_v1/REQUIREMENTS.md` | 保存用户完整开发规格 |
| `docs/paper_sac_v1/IMPLEMENTATION_PLAN.md` | 实施任务与验证清单 |
| `docs/paper_sac_v1/AUDIT_LEDGER.md` | 实现决定、代价、审查及修复记录 |
| `docs/paper_sac_v1/README.md` | 配置、运行、恢复、指标与 Linux 命令 |
| `docs/paper_sac_v1/ACCEPTANCE_REPORT.md` | 本验收报告 |
| 仓库根目录 `Guillen-Perez 等 - DCcluster-Opt … Wor.pdf` | 用户授权跟踪的已有论文参考文件，内容未修改 |

运行产物保存在 `outputs/paper_sac_v1/`，按 seed 和 RUN_ID 隔离，另有 checks/logs/tensorboard/replay/evaluations/manifests/final_results，不纳入 Git。没有清理开发过程中此前的 smoke 产物。

**原有测试的数据副作用：** `tests/test_env_setup.py` 调用旧环境加载器，执行了原有 ZIP 自动解压逻辑，在 `data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl` 新增一份数据副本。原嵌套目录文件和 ZIP 均保留，没有覆盖或删除。新旧 `.pkl` 都是 142,336,499 字节，SHA256 均为 `3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e`，与第一次预检查记录的原文件哈希相同。该副本受原 `.gitignore` 排除。新工程只读 loader 自身不解压、不修改数据。不能宣称数据目录中没有新增副本。

## B. 最终配置

| 参数 | 正式值 |
|---|---|
| 训练时间 | 2023-08-01 05:00 UTC，7 天，672 步/Episode |
| 测试时间 | 2023-07-01 00:00 UTC，30 天，2880 步/次 |
| 时间步长 | 15 分钟 |
| Workload | 2020 数据，同月日映射；scale=1、task_scale=5、group_size=1 |
| 外部数据 | 2023 年电价、CI、气象；UTC 对齐 |
| 时间模式 | fixed 默认；randomized_aligned 可选；weather_augmentation=false 默认 |
| 训练/测试种子 | 42/43/44；0/1/2 |
| 每模型步数 | 10,000,000；三模型顺序执行，共 30,000,000（仅配置，未执行） |
| 算法 | 离散 SAC，gamma=0.99，固定 alpha=0.01，tau=0.005 |
| 优化器/学习率 | Adam；Actor 1e-4；Critic 3e-4 |
| Replay/batch/warmup | 100000 / 512 / 1000 |
| 更新/策略更新频率 | 1 / 2 |
| 保存/日志频率 | 50000 / 1000，完整保存延迟到随后 Episode 边界 |
| 网络 | 输入 34、动作 6、两层 64 ReLU、Twin Q；无 Attention、无 LayerNorm |
| 观察任务上限 | 400；超限报错，不截断 |
| 训练内评估 | 0 / 0，明确禁用 |
| 硬件 | device=auto；本次只用 CPU 小规模测试 |

DC1～DC5 的位置/CPU/GPU/Memory 分别为 US-CAL-CISO/50000/1000/80000、DE-LU/85000/600/80000、CL-SIC/110000/300/60000、SG/75000/700/50000、AU-NSW/65000/300/60000。固定动作 1～5，动作 0=Defer。沿用原 `configs/dcs/dc_config.json`；`use_rl_hvac=false`、`hvac_controller_type=none`。

Reward：energy_price 0.5÷100000；carbon_emissions 0.3÷10；transmission_cost 0.2÷1；sla_penalty 权重 0.1、每违反 5.0；transmission_emissions 0.1÷1。组件固定缩放、Composite 加权（normalize=false）、最终标量 RunningStats 是三层独立处理。没有更改原奖励的代理物理量公式或固定温度策略。

Smoke 仅在内存中设 total_steps=64、capacity=128、batch=8、warmup=8、max_tasks=400、CPU、16 步 Episode，不写回正式 YAML。

## C. 代码审计与处理

| 审计点 | 结论与处理 |
|---|---|
| Multi-action + Defer | 已验证；不使用 Single Action，不禁用 Defer，不做 Argmax |
| 34/6 维度 | 环境初始化后断言，任务观察逐条检查；Actor/Critic 与元信息一致 |
| DC 映射 | 固定 1～5 顺序；每次生成观察再次校验；Time Only 非零直接映射 origin ID |
| 奖励注册 | 显式导入并注册五项奖励；真实环境可实例化并计算 |
| 时间戳 | 用真实 day-of-year 与 UTC 按时插值，修正 local hour/随机日期错位，保留独立随机模式 |
| 实际物理输入 | DC.step 先执行物理步，单校验 manager 不够；现已在步前注入准确 UTC 温度/湿球/CI，并校验实际消耗字段 |
| 任务数上限 | 新到达最大 279/320；最坏 Defer 上界 687/946；不能保证 400 总足够，真实队列超限立即停止 |
| Replay 内存 | 每 seed 精确 11,520,800,000 bytes = 10.72958 GiB，使用 memmap，不在 RAM 分配完整数组 |
| 恢复一致性 | 最近完整提交支持网络/优化器/统计/RNG/Replay 全恢复；实际 64 步两路径完全相同 |
| 覆盖提交数据风险 | 两个 WAL 写前保存旧行并 fsync；目录持久化；残缺尾部回滚；排他写锁 |
| 发布中断 | 保留孤立 checkpoint，重放保存为 retry 文件名，避免覆盖或卡死 |
| 恢复容量检查 | 以实际磁盘块抵扣已有 memmap，计入尚需增长和 WAL，避免重复要求完整容量 |
| 历史/推理快照 | 保留用于评估，明确拒绝完整恢复；旧 full 文件如果不再对应当前 Replay 提交也拒绝 |
| SLA/任务累计 | 按完成任务算 SLA，未完成另报；端点完成无额外物理步；不纳入下一观察中新到达任务 |
| 物理单位 | DC+传输分别每步累计一次；kWh→MWh、kg→t、L→m³；没有重复把传输别名相加 |
| 动作事件 | Defer 决策、去重 Defer 任务、主动本地执行、超期强制回源、远程路由分别计数 |
| 随机配对 | 每次评估独立 torch.Generator；任务来源 RNG 与策略分离；比较实际 workload/origin/calendar 指纹 |
| 统计 | 每训练模型内汇总测试种子，再汇总模型均值；ddof=1；单样本 SD=null，保留两种 seed |
| 正式运行保护 | 必须显式 --production；smoke 限制单模型单测试 seed；导入模块不运行实验 |

默认正式三个 Buffer+双 WAL+2 GiB 预留的磁盘预算为 **71,739,234,608 字节（约 71.74 GB）**；RAM 工作集估算为 2,017,525,760 字节。预算是保守峰值，非实际已分配正式 Buffer；本次未创建 100000 容量 Buffer。正式预检查输出即时可用容量。

论文冲突与保留限制：Table 9 的 10M 与正文 Figure 3 的 1M/5 runs、Table 3 的 10 seeds 不一致，按用户明确的 10M×3 协议实施。附录 origin `k+1` 按仓库 1-based ID 修正。原 SAC 的下一任务槽位目标、代理型成本/排放奖励、湿球近似、物理步/任务占用更新顺序保留，不擅自改变算法与物理公式。

## D. 实际测试结果

| 检查/执行 | 结果 |
|---|---|
| `python -m pytest tests -q ...` | **25 passed in 26.75s**；22 项本实验测试/参数化实例，加 3 项既有测试 |
| 真实固定/随机环境 | DC1～5、34/6 维度、外部/物理时钟、超期强制回源、任务守恒通过 |
| 三策略分布 | 六动作随机采样、Defer 条件屏蔽、origin 映射、独立 Generator、极端 logits 通过 |
| Buffer | 精确容量、拒绝 401、不同任务数、空 next_obs、排他锁、覆盖回滚、残缺 WAL 通过 |
| Checkpoint | Target、优化器、RunningStats、RNG、Replay、指纹拒绝、推理快照拒绝通过 |
| 故障注入 | 发布前中断遗留文件后重放成功且不覆盖；日志目录同步失败前阻止数据覆盖 |
| 完整数据预检查 | passed，伴随 400 槽位最坏队列风险警告；路径/覆盖/UTC/真实物理消耗校验通过 |
| 最终受控训练 | seed 42，32 步停止，完整恢复至 64；无旧权重加载 |
| 连续对照训练 | 同 seed 独立连续 64 步；Actor/Critic/Target/两优化器/统计/RNG/七数组与恢复路径**完全一致** |
| 最终评估 | 一个 smoke 模型 × eval seed 0 × 三策略，各 8 步，共 24 步；配对指纹一致 |
| CSV/JSON/汇总 | 实际保存，读取及分层统计通过；smoke 与不完整矩阵显式标记 |
| 语法 | `python -m compileall -q scripts/paper_sac` 通过 |
| 原文件保护 | `git diff --exit-code master -- train_SAC_agent.py envs simulation rl_components rewards utils configs/env configs/dcs data` 通过 |
| 全仓库 `python -m pytest -q` | **未通过收集**：原 `data/electricity_prices/scripts/test_prices.py` 缺少 `openelectricity`；未安装并运行该联网下载脚本来绕过 |

原 `tests/test_training_loop.py` 的旧 smoke 使用已不存在的脚本名且不检查 os.system 退出码，因此其通过不能作为 SAC 可运行证据；本次使用新增真实训练、更新及恢复测试作为证据。初期测试遇到系统临时目录权限问题，后续使用全新的实验内部 `--basetemp`；Windows fsync 写句柄等实现问题均已复现并修复，不计为仍未解决的问题。

最终可核验证据（仓库相对路径，运行产物不进 Git）：

- 预检查：`outputs/paper_sac_v1/manifests/preflight_20261116T040334_02901c60/preflight.json`。
- 项目测试：`outputs/paper_sac_v1/project-tests-final.log`。
- 全仓库失败：`outputs/paper_sac_v1/full-suite-final.log`。
- 恢复训练：`outputs/paper_sac_v1/manifests/smoke_20261116T035937_e5c7c992/experiment_manifest.json`。
- 连续对照：`outputs/paper_sac_v1/manifests/smoke_20261116T040118_cc8c014a/experiment_manifest.json`。
- 完全一致性：`outputs/paper_sac_v1/resume-equivalence-final.json`。
- 三策略评估：`outputs/paper_sac_v1/manifests/eval_smoke_20261116T040211_d254f666/experiment_manifest.json`。
- 分层结果：`outputs/paper_sac_v1/final_results/eval_smoke_20261116T040211_d254f666/summary_20261116T040316_a8e9f328/`。

实际测试软件：Windows 11、Python 3.13.5、NumPy 2.4.0、pandas 2.3.3、PyTorch 2.10.0 CPU、Gymnasium 1.4.0、TensorBoard 2.21.0、PyYAML 6.0.2、PsychroLib 2.5.0。依赖安装在本实验虚拟环境；系统 PyTorch 通过 system-site-packages 复用。pip 提示共享系统环境中的 Streamlit 依赖冲突，本实验不使用 Streamlit，系统包未卸载或升级。

最终源码文件指纹：`97fb91c68c706609a61bebea4258f364e7ee567cf43f0ef10c155e13568e115f`。具体各文件哈希、数据哈希、软件和 Git dirty 状态均记录在 manifest。该指纹不包含文档；验证时 Git 提交仍为基线、实现位于工作树，因此原 manifest 保留当时版本信息。后续按用户授权提交和推送不改写历史实验记录。

## E. Linux 正式运行方法（只记录，未执行）

服务器安装、预检查和 smoke：

```bash
git switch feature/paper-sac-v1
git status --short
python3 -m venv outputs/paper_sac_v1/.venv
source outputs/paper_sac_v1/.venv/bin/activate
python -m pip install -r scripts/paper_sac/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/paper_sac/preflight.py
python -m pytest tests -q --basetemp outputs/paper_sac_v1/server-check-unique-id
python scripts/paper_sac/train.py --smoke --seeds 42
```

检查并接受 F 节限制后，再由用户手动运行正式三个模型：

```bash
python scripts/paper_sac/train.py \
  --config configs/experiments/paper_sac_v1/algorithm.yaml \
  --production --seeds 42 43 44
tensorboard --logdir outputs/paper_sac_v1/tensorboard --port 6006
```

完整恢复时，根据相应 seed 的 `latest.json`/Replay `commit.json` 选择已提交完整文件，不要用 best/final 推理快照：

```bash
python scripts/paper_sac/train.py --production --seeds 43 \
  --resume outputs/paper_sac_v1/checkpoints/seed_43/RUN_ID/COMMITTED_STEP.pt
# 如 seed 44 尚未开始，另建新批次继续，不重训已完成的 42。
python scripts/paper_sac/train.py --production --seeds 44
```

正式 27 次评估和汇总命令见 README。它们需要三个 10M 最终模型及显式 `--production`，本次没有执行。

## F. 未解决/未验证事项与验收边界

1. **400 槽位不是无条件充足。** 完整新到达统计通过，但持续 Defer 的理论队列上界超过 400。正式模型可能触发 fail-fast；不能通过截断、修改 task_scale 或默默增大 max_tasks 绕过。需要后续正式策略轨迹验证，或由用户明确批准另一个实验配置。
2. **全仓库 pytest 未全绿。** 原下载脚本缺少 openelectricity，收集期失败。项目 tests 25 项通过不等于全仓库通过，不能隐去该失败。
3. **Linux/CUDA 未实机验证。** POSIX 文件锁/目录 fsync 有实现，CUDA RNG 有保存/恢复，但本次只在 Windows CPU 验证。CUDA 恢复要求相同设备数量，跨硬件 bitwise 一致性不作保证。
4. **正式容量与性能未实测。** 只验证小容量 memmap 和准确大小计算，未做 11.52 GB/seed 的长期 I/O、3×10M 训练或 27×2880 评估。
5. **原算法/物理集成限制保留。** next_obs 的槽位不代表相同任务；奖励为原代理量，不等于物理 KPI；湿球近似及先物理步后任务负载更新未改。新默认时间/天气协议不同于旧脚本错位轨迹，不宣称论文数值复现。
6. **完整恢复只支持最近提交。** 更旧完整 checkpoint 文件及 best/final 快照可推理，不能保证其 Replay 历史仍在。10M 末尾非边界的最终模型是推理快照，需要从最近完整边界重放才能恢复训练过程。
7. **原测试产生一份等哈希 workload 副本。** 已明确记录并保留，原文件字节内容没有改变；没有自动删除清理。

分支保持独立；没有合并到 master/main，没有删除原有模型、配置、结果或数据。
