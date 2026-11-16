# Paper SAC v1 Implementation Plan

**Goal:** Implement the supplied paper experiment specification without changing the original simulator, datasets, results, or starting production training.

**Architecture:** New Python modules under `scripts/paper_sac` wrap the existing environment, networks, reward classes and RunningStats. Separate configuration, preflight, training, evaluation and summary entry points. Only `feature/paper-sac-v1` receives changes; original `master` remains at c79e8f78a9229b904f123276168a9ef7746177fd.

**Tech Stack:** Python, PyTorch, NumPy memmap, pandas, YAML, pytest, TensorBoard.

**Spec:** User supplied SAC specification, preserved as REQUIREMENTS.md.

## Global constraints

- Never run 10,000,000-step training or the 27-run production evaluation during development.
- Never modify existing core files or original data; use subclasses and independent modules.
- Keep exact paper hyperparameters; smoke overrides are in memory and recorded separately.
- Fail on any observation task count above 400, including deferred queues.
- New experiment run IDs prevent collisions; no automatic merging or deletion of historical artifacts.

## Design decisions

- Reuse ActorNet/CriticNet and the multi-action SAC target in train_SAC_agent.py (positional next-task bootstrapping retained and documented as an existing limitation).
- Use exact configured dates by default (`fixed`); preserve seeded date/hour randomization in an explicit `randomized_aligned` mode. Both align tasks and UTC external data; tests cover both.
- Reuse physical models and reward formulas unchanged. Explicitly register five rewards. Keep component factors, CompositeReward and scalar RunningStats separate.
- Memmap uses exact original dtypes and bounded write-ahead undo journals. Only the latest committed episode-boundary checkpoint is a full resume point; older/best snapshots remain inference artifacts.
- Persist configuration/data/source fingerprints, RNGs and run status. Resume validates identities and restores RNG after deterministic next-episode reset.
- Evaluation samples with a per-run CPU torch.Generator; routes implement the specified conditional and origin-mapping protocols. Accounting uses per-step physical fields, unique task identities, and excludes arrivals at the terminal boundary.
- SLA denominator is completed tasks; finish timestamps at the horizon count as completed even if the simulator has not released them yet. Pending/in-transit/deferred tasks remain separately reportable.

## Review focus

- Crash between memmap writes and checkpoint publication: journal rollback must restore committed samples.
- Changed config/data or stale historical checkpoint: reject full resume.
- Empty observations, unequal task counts and overflow: never truncate; masked updates must remain finite.
- Timezone offsets and manager advance-before-consume: test actual UTC values at reset and after step.
- No completed tasks or missing result cells: report null SLA/sample SD, never fabricate zeros or pool seeds.

## Implementation tasks

- [x] Add behavioral tests for configuration, strategy sampling, memory size, buffer durability, update, checkpoint/RNG restoration, metric units and summary hierarchy. Run red.
- [x] Add five isolated configs plus config, provenance and data/time utilities. Validate original resource parameters.
- [x] Add read-only workload lookup and cluster/environment adapters; preflight real data and task-count distribution.
- [x] Implement persistent replay, transactional checkpoint, SAC updates and sequential seeded training; run resume equivalence tests.
- [x] Implement stochastic evaluation, CSV/JSON results and hierarchical sample statistics.
- [x] Run focused pytest, real environment smoke, interrupted/resumed smoke, limited evaluation and existing suite. Record all failures.
- [x] Write README and acceptance report with commands, limitations and actual test evidence. Verify master hash and diff scope.

Final project suite: 25 passed. Bare repository pytest has an existing collection error in `data/electricity_prices/scripts/test_prices.py` (`openelectricity` absent); do not install/run that network download script as a workaround. Final CPU uninterrupted-vs-resume states and all seven replay arrays are exactly equal at step 64. Independent review's four important findings were reproduced, fixed and rechecked. See AUDIT_LEDGER.md and ACCEPTANCE_REPORT.md.
