# 收集器v2修复、Geo Only复验及Time Only执行报告

**通过：Geo Only已有五次正式验收完成；Time Only新运行五次并验收完成。共10/15次，Geo+Time未启动。**

## 1. 代码差异和版本登记

原collect_formal_results.py未覆盖，原字节备份保留在本目录original_collect_formal_results.py。frozen_manifest、freeze_integrity、run_commands及所有原实验源码均未改动。
- 原版SHA256：2b93c34eae77e9122497811c96a59fe77e3967a4c3071281b49bed3e598ccf62
- v2 SHA256：ce447b991376a5987bdd54c5a4c6cf4991db5598938b1bab87c06468d160cf76

```diff
--- collect_formal_results.py (original)
+++ collect_formal_results_v2.py
@@ -82,7 +82,7 @@
                 raise AssertionError('Geo Only deferred.')
             if variant=='time_only' and (remote.any() or any(summary[k]!=0 for k in ['transmission_cost_usd','tx_energy_kwh','tx_carbon_kg'])):
                 raise AssertionError('Time Only migrated or incurred TX.')
-            expired=pd.to_datetime(decisions.time_utc,utc=True)>pd.to_datetime(decisions.sla_deadline_utc,utc=True)
+            expired=pd.to_datetime(decisions.time_utc,utc=True)>pd.to_datetime(decisions.sla_deadline_utc,utc=True,format="ISO8601",errors="raise")
             if not np.array_equal(expired.values,(kind=='forced_origin_after_deadline').values):
                 raise AssertionError('Deadline override classification differs.')
             if not np.isfinite(steps.select_dtypes('number').to_numpy()).all():
```

审查了三处pd.to_datetime调用：steps.time_utc与decisions.time_utc均为统一整秒UTC；只修复混合精度sla_deadline_utc。format=ISO8601、utc=True、errors=raise，不丢弃或coerce非法值。其余代码AST与原版完全相同。

## 2. 实际格式测试

| 小数秒精度 | 实际CSV示例 | 原Geo Only出现次数 | 结果 |
|---|---|---:|---|
| 0位 | 2023-07-01T04:36:12+00:00 | 43485 | UTC datetime，通过 |
| 6位 | 2023-07-01T04:00:28.500000+00:00 | 50895 | UTC datetime，精度保留 |
| 9位 | 2023-07-01T05:07:14.999999999+00:00 | 39665 | UTC datetime，纳秒保留 |

6项测试全部通过：唯一修改AST检查、三种实际精度、全部134045条原deadline无缺失解析、非法字符串仍抛错。输出dtype为datetime64[ns, UTC]。测试未启动仿真。

## 3. 正式收集和索引

v2首次调用退出0，生成Geo Only五份run_record.json及validated_index_05_runs.json；九项均值和ddof=1标准差均与原aggregate.csv相同。只读验收未重跑Geo Only。
随后按原run_commands第2节相同参数运行Time Only五种子，CPU/torch threads=2，每次672步；运行前后verify_frozen均通过。按本次授权用v2替代旧版做后处理，第二次v2调用退出0，生成Time Only五份run_record.json及validated_index_10_runs.json。
索引中incomplete_not_for_final_statistics表示15次计划尚缺Geo+Time五次，不是这十次验收失败；每run状态均为complete_verified。原05索引和五份Geo Only记录未被改写。

## 4. 每次实验九项指标与状态

| 策略 | Seed | 状态/步数 | Cost USD | CO₂ t | Energy MWh | Water m³ | SLA % | CPU % | GPU % | TX USD | Defer |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo_only | 0 | 通过/672 | 111000.85 | 304.6509 | 1110.5277 | 7371.7174 | 36.4636 | 2.3662 | 13.3516 | 4077.28 | 0 |
| geo_only | 10 | 通过/672 | 111050.64 | 304.8540 | 1111.3983 | 7375.0650 | 36.1863 | 2.3710 | 13.3972 | 4027.09 | 0 |
| geo_only | 20 | 通过/672 | 111067.92 | 304.8167 | 1110.1183 | 7376.0992 | 35.8601 | 2.3584 | 13.3481 | 4168.42 | 0 |
| geo_only | 30 | 通过/672 | 111011.34 | 304.6310 | 1109.8758 | 7373.2921 | 36.0829 | 2.3710 | 13.3179 | 4069.96 | 0 |
| geo_only | 40 | 通过/672 | 111204.92 | 305.2785 | 1112.4116 | 7375.6027 | 36.6300 | 2.3687 | 13.4856 | 4088.96 | 0 |
| time_only | 0 | 通过/672 | 112238.53 | 316.8598 | 1112.9830 | 7494.7851 | 34.7874 | 2.6047 | 15.0401 | 0.00 | 828 |
| time_only | 10 | 通过/672 | 111735.29 | 315.3913 | 1110.4513 | 7485.5475 | 34.7949 | 2.6187 | 14.8082 | 0.00 | 815 |
| time_only | 20 | 通过/672 | 111902.36 | 316.2308 | 1110.2916 | 7488.6803 | 34.7685 | 2.6031 | 14.8683 | 0.00 | 769 |
| time_only | 30 | 通过/672 | 112360.41 | 317.2522 | 1110.2690 | 7491.5283 | 34.7912 | 2.6013 | 14.9429 | 0.00 | 882 |
| time_only | 40 | 通过/672 | 111962.58 | 315.9483 | 1108.1979 | 7484.5515 | 34.8252 | 2.6047 | 14.7025 | 0.00 | 806 |

十次时间均为2023-07-01 00:00 UTC开始，最后步2023-07-07 23:45 UTC，右端2023-07-08 00:00。种子严格为0、10、20、30、40，Seed42未纳入。

## 5. 五种子均值 ± 样本标准差

| 指标 | Geo Only | Time Only |
|---|---:|---:|
| total_cost_usd | 111067.134858 ± 81.802019 | 112039.832448 ± 254.884340 |
| total_co2_t | 304.846217 ± 0.260877 | 316.336483 ± 0.736145 |
| total_energy_mwh | 1110.866345 ± 1.039983 | 1110.438556 ± 1.698905 |
| total_water_m3 | 7374.355275 ± 1.815894 | 7489.018537 ± 4.232881 |
| sla_violation_pct_finished | 36.244584 ± 0.305737 | 34.793423 ± 0.020461 |
| avg_cpu_util_pct | 2.367065 ± 0.005237 | 2.606505 ± 0.006982 |
| avg_gpu_util_pct | 13.380053 ± 0.065449 | 14.872386 ± 0.128553 |
| transmission_cost_usd | 4086.339670 ± 51.489377 | 0.000000 ± 0.000000 |
| tasks_deferred | 0.000000 ± 0.000000 | 820.000000 ± 41.018289 |

n=5、ddof=1。费用/能耗/碳包含TX；Time Only TX为零；水耗仅DC；SLA分母为完成任务，未完成及逾期未完成单列；defer是有效决策次数。

## 6. 动作约束与一致性

| 策略 | Seed | 远程执行决策 | raw0 | 有效defer | 强制回源 | TX USD | TX kWh | TX kgCO₂ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| geo_only | 0 | 21023 | 0 | 0 | 0 | 4077.275304 | 3168.515452 | 961.987164 |
| geo_only | 10 | 21000 | 0 | 0 | 0 | 4027.088607 | 3178.407358 | 962.636647 |
| geo_only | 20 | 21104 | 0 | 0 | 0 | 4168.416872 | 3200.415762 | 975.764681 |
| geo_only | 30 | 21116 | 0 | 0 | 0 | 4069.961091 | 3178.136018 | 978.047556 |
| geo_only | 40 | 21112 | 0 | 0 | 0 | 4088.956477 | 3192.707581 | 973.765940 |
| time_only | 0 | 0 | 909 | 828 | 105 | 0.000000 | 0.000000 | 0.000000 |
| time_only | 10 | 0 | 892 | 815 | 97 | 0.000000 | 0.000000 | 0.000000 |
| time_only | 20 | 0 | 838 | 769 | 88 | 0.000000 | 0.000000 | 0.000000 |
| time_only | 30 | 0 | 972 | 882 | 108 | 0.000000 | 0.000000 | 0.000000 |
| time_only | 40 | 0 | 878 | 806 | 91 | 0.000000 | 0.000000 | 0.000000 |

Geo Only raw0/effective defer/tasks_deferred均0。Time Only所有非零动作和实际执行目的均为来源DC，五次每个step的TX费用、能耗和碳都为0。有效defer与SLA过期强制回源分开统计。保留p0的依据为冻结的完整六动作采样再映射实现，不用观测defer次数代替概率证明。

全部十次checkpoint SHA、source config SHA与实际external_inputs_sha256一致；同一seed两种策略的realized_workload_sha256匹配。原Geo Only CSV/JSON、原收集器、原冻结文件、checkpoint、数据和Seed42全部保持原hash。

## 7. 当前边界与后续

未重跑Geo Only，新增仿真只有Time Only五次；没有重训、参数更新或修改模型/reward/物理/外部数据。仍未启动Geo+Time五种子。原Geo Only批报告中“收集阻塞”为历史状态，本报告和geo_only_v2_acceptance.json记录解除阻塞后的状态，未覆盖历史报告。
以后后处理使用collect_formal_results_v2.py，原run_commands.md作为冻结历史保留不改。待第三批按授权完成后，再使用v2 --require-complete验收全部15次；当前不伪造完整15次汇总。
