# Table 3 reward audit

The reproduction takes the component equations from Appendix F.4.2 and the SAC
weights and scales from Appendix G.3.2. The component terms apply to one finished
simulation interval, and the composite is their weighted sum. [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/075661dddd5afc49e905b24e12e6a968-Paper-Datasets_and_Benchmarks_Track.pdf)

| Component | Weight | Formula used for Table 3 | Baseline audit and resolution |
|---|---:|---|---|
| Energy price | 0.5 | `-sum(DC electricity cost in USD) / 100000` | Baseline estimated task energy from cores and duration, multiplied by destination price. The experiment opts into the metered DC electricity cost, including intervals without new assignments. |
| Carbon emissions | 0.3 | `-(sum(DC operational emissions in kg) + transmission emissions in kg) / 10` | Baseline estimated task emissions from cores, duration and destination intensity. The experiment opts into the DC and network totals. |
| Transmission cost | 0.2 | `-transmission cost in USD / 1` | Baseline formula matches. The simulator books costs when tasks start at the destination. |
| SLA penalty | 0.1 | `-5.0 * sum(DC tasks finished after the SLA deadline)` | Baseline formula matches. Each DC reports completed-task SLA events once. |
| Transmission emissions | 0.1 | `-transmission emissions in kg / 1` | Baseline formula matches. The composite penalizes transmission emissions through this term and the carbon term, following F.4.2 and G.3.2 together. |

`configs/repro_table3/reward.yaml` selects the metered energy and carbon
formulas for the full run with `use_metered_totals: true`. The existing
configurations keep the proxy defaults. The source cost conversion multiplies
DC electricity in kWh by the price in USD/MWh and divides by 1000.
The carbon conversion multiplies DC electricity in kWh by intensity in
gCO₂eq/kWh and divides by 1000. These values feed the metered rewards directly.
Evaluation KPIs use their own physical totals and units.

The composite running mean and standard deviation are disabled. The SAC
trainer keeps the baseline's separate reward preprocessing before insertion
into replay. `RunningStats` starts with scalar mean 0, variance 1 and count
`1e-5`. It incorporates each step's raw composite reward and then standardizes
that value with the updated mean and a variance floor of `1e-6`. The trainer
uses the raw composite reward for episode logging and best-checkpoint selection.


# Simulation inputs to the reward

The reproduction uses one episode clock. Training draws the start day uniformly
from the configured day ±7 days and the start hour from 00 through 23 UTC,
following the baseline's seeded reset. That sampled time drives the environment's
time features, workload, HVAC weather, intensity and price. Evaluation starts at
2023-07-01 00:00 UTC for each seed. An episode spans 30 days from its actual start.

- Workload lookup keeps month, day and UTC time and replaces the year with 2020.
  For the sorted trace the experiment uses binary search, retaining the first
  row for an interval. The existing behavior for an absent interval is an empty
  workload. The baseline scale, grouping and resource units are retained.
  CPU and GPU resource values are multiplied by 5 and divided by 100. Memory
  is multiplied by 5. Duration and bandwidth pass through unchanged.
  The SLA deadline is arrival plus 1.5 times the original task duration.
- Origin selection follows Appendix F.5 using population weight times 1.0 for
  local hours in `[08:00, 20:00)` and times 0.3 otherwise. The configured
  Table 8 offset determines that local clock and the original 1-based DC ID
  is stored when the task is created.
- The five carbon and standardized price files each have 8,760 hourly UTC
  rows in 2023. Both use the UTC episode index and exact 15-minute linear
  interpolation. The baseline shifted the carbon array and could start
  UTC price data on a local index. Its interpolation endpoints also made
  samples drift relative to exact quarter hours.
- Each weather file has 8,760 nominal local-hour labels. Weather and the
  DC local clock use the paper's fixed offsets, with correct previous-day
  rollover for a negative shift. The export's offsets pair with the Table 8
  offsets as follows.

| Location | JSON timezone | Export offset | Table 8 offset used |
|---|---|---:|---:|
| US-CAL-CISO | America/Los_Angeles | -7 | -7 |
| DE-LU | Europe/Berlin | +1 | +1 |
| CL-SIC | America/Santiago | -3 | -5 |
| SG | Asia/Singapore | +8 | +8 |
| AU-NSW | Australia/Sydney | +11 | +11 |

The reproduction indexes the weather wall-clock labels with the Table 8 offset
and keeps the baseline's seeded coherent temperature noise and 0–13-day
roll. The position of the recorded weather values is
`(weather_local_15m - 96 * weather_roll_days) % 35040`. Each reset exposes
the real UTC start and end, 2020 workload start, per-DC carbon and price
indices, and weather nominal index, roll and underlying source index.

In the aligned experiment, cost and operational emissions use the environmental
inputs for the interval being metered. Transmission emissions use origin CI
from that same interval. The source physics sequence still meters the previously
prepared resource loads before preparing the new loads. Queue and SLA metrics
reflect scheduling at the current interval, while the resulting power effect
appears at the next interval. This preserves the baseline's physical stepping
sequence.

Local assignments enter the destination pending queue immediately, as described
in F.3.2. Baseline routing put local assignments into the transit queue after
checking the current arrivals and delayed them by at least one interval.
Remote tasks use the original transmission delay and queue. The FIFO order,
task execution rules and pre-action overdue-to-origin override are retained.

The aligned experiment propagates physical sub-environment and missing network
metering errors so the run stops before recording stale measurements.
