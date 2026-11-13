"""Report saved runs after collector failure; NOT a replacement acceptance tool.

No simulator imports, no rerun, no correction of frozen code or result files.
The final acceptance status remains blocked_collector_error.
"""
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
FROZEN=ROOT/'outputs/table3_7d_preflight'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    if (OUT/'geo_only_batch_report.md').exists():raise FileExistsError('Refuse to overwrite report.')
    frozen=json.loads((FROZEN/'frozen_manifest.json').read_text(encoding='utf-8'))
    batch=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert batch['status']=='complete' and batch['completed_runs']==5
    assert batch['seeds']==[0,10,20,30,40] and batch['variants']==['geo_only']
    assert batch['checkpoint']['sha256']==frozen['checkpoint']['sha256']
    assert batch['runtime_simulation']==frozen['simulation']
    integrity=json.loads((FROZEN/'freeze_integrity.json').read_text(encoding='utf-8'))
    frozen_changes=[name for name,h in integrity['bundle_sha256'].items() if digest(FROZEN/name)!=h]
    assert not frozen_changes
    current_drift=[r['path'] for r in frozen['frozen_files'] if digest(ROOT/r['path'])!=r['sha256']]
    assert not current_drift
    previous=json.loads((FROZEN/'preservation_before.json').read_text(encoding='utf-8'))
    prior_drift=[p for p,h in previous.items() if digest(ROOT/p)!=h]
    assert not prior_drift
    offline=pd.read_csv(ROOT/frozen['offline_csv']['path'])
    offline['time_utc']=pd.to_datetime(offline.time_utc,utc=True)
    offline=offline.set_index(['dc_id','time_utc'])
    window=pd.date_range('2023-07-01T00:00Z','2023-07-08T00:00Z',freq='15min',inclusive='left')
    metric_names=['total_cost_usd','total_co2_t','total_energy_mwh','total_water_m3','sla_violation_pct_finished',
                  'avg_cpu_util_pct','avg_gpu_util_pct','transmission_cost_usd','tasks_deferred']
    rows=[];diagnostic_rows=[];hash_rows=[];max_errors=[]
    for seed in batch['seeds']:
        folder=OUT/'geo_only'/f'seed_{seed}'
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        steps=pd.read_csv(folder/'steps.csv');decisions=pd.read_csv(folder/'decisions.csv');inputs=pd.read_csv(folder/'inputs.csv')
        assert len(steps)==672 and np.array_equal(steps.step,np.arange(672))
        assert pd.DatetimeIndex(pd.to_datetime(steps.time_utc,utc=True)).equals(window)
        assert summary['external_inputs_sha256']==frozen['expected_external_inputs_sha256']
        assert summary['new_arrivals']==summary['tasks_finished']+summary['unfinished_tasks']
        assert (decisions.action==0).sum()==0 and (decisions.effective_decision=='defer').sum()==0 and summary['tasks_deferred']==0
        assert len(inputs)==3360
        inputs['time_utc']=pd.to_datetime(inputs.time_utc,utc=True)
        errors={}
        for dc,g in inputs.groupby('datacenter'):
            assert pd.DatetimeIndex(g.time_utc).equals(window)
            expected=offline.loc[dc].loc[window]
            for key in ['electricity_price_usd_mwh','carbon_intensity_g_kwh','ambient_temperature_c']:
                err=float(np.max(np.abs(g[key].to_numpy()-expected[key].to_numpy())))
                assert err<1e-10
                errors[key]=max(errors.get(key,0),err)
        total=steps.sum(numeric_only=True)
        recalc={'total_cost_usd':float(total.dc_energy_cost_usd+total.transmission_cost_usd),
                'total_co2_t':float(total.dc_carbon_kg+total.tx_carbon_kg)/1000,
                'total_energy_mwh':float(total.dc_energy_kwh+total.tx_energy_kwh)/1000,
                'total_water_m3':float(total.water_l)/1000,'sla_violation_pct_finished':100*float(total.sla_violated)/float(total.tasks_finished),
                'avg_cpu_util_pct':float(steps.avg_cpu_util_pct.mean()),'avg_gpu_util_pct':float(steps.avg_gpu_util_pct.mean()),
                'transmission_cost_usd':float(total.transmission_cost_usd),'tasks_deferred':int(total.tasks_deferred)}
        assert all(np.isclose(v,summary[k],rtol=1e-12,atol=1e-7) for k,v in recalc.items())
        forced=int((decisions.effective_decision=='forced_origin_after_deadline').sum())
        assert int(total.forced_origin_decisions)==forced
        routed=decisions.loc[decisions.effective_decision=='route']
        remote=int((routed.origin_dc_id!=routed.destination_dc_id).sum())
        row={'seed':seed,'simulation_status':'complete','official_collection_status':'blocked_collector_error',
             'steps':len(steps),'first_utc':steps.time_utc.iloc[0],'last_step_utc':steps.time_utc.iloc[-1],
             **{k:summary[k] for k in metric_names}}
        rows.append(row)
        diagnostic_rows.append({'seed':seed,'raw_action_0':int((decisions.action==0).sum()),
            'effective_defer':int((decisions.effective_decision=='defer').sum()),'tasks_deferred':summary['tasks_deferred'],
            'forced_origin_after_deadline':forced,'remote_route_decisions':remote,
            'tx_energy_kwh':summary['tx_energy_kwh'],'tx_carbon_kg':summary['tx_carbon_kg'],
            'new_arrivals':summary['new_arrivals'],'tasks_finished':summary['tasks_finished'],
            'sla_violated':summary['sla_violated'],'unfinished_tasks':summary['unfinished_tasks'],
            'late_unfinished_tasks':summary['late_unfinished_tasks']})
        hash_rows.append({'seed':seed,'checkpoint_sha256':batch['checkpoint']['sha256'],
            'source_config_sha256':next(r['sha256'] for r in frozen['frozen_files'] if r['role']=='simulation_config'),
            'runtime_sim_eval_sha256':digest(folder/'sim_eval.yaml'),'workload_file_sha256':frozen['workload_file_sha256'],
            'realized_workload_sha256':summary['workload_sha256'],'external_inputs_sha256':summary['external_inputs_sha256']})
        max_errors.append({'seed':seed,**errors})
    data=pd.DataFrame(rows)
    aggregates=[{'metric':key,'n':5,'mean':float(data[key].mean()),'sample_std_ddof1':float(data[key].std(ddof=1))} for key in metric_names]
    producer=pd.read_csv(OUT/'aggregate.csv').iloc[0]
    for row in aggregates:
        assert np.isclose(row['mean'],producer[row['metric']+'_mean'],rtol=1e-12,atol=1e-8)
        assert np.isclose(row['sample_std_ddof1'],producer[row['metric']+'_std'],rtol=1e-10,atol=1e-8)
    for name,records in [('geo_only_metrics_per_seed.csv',rows),('geo_only_metrics_aggregate.csv',aggregates),('geo_only_constraints.csv',diagnostic_rows),('geo_only_run_hashes.csv',hash_rows)]:
        if (OUT/name).exists():
            buffer=io.StringIO(newline='')
            writer=csv.DictWriter(buffer,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            if (OUT/name).read_bytes()!=buffer.getvalue().encode('utf-8-sig'):
                raise FileExistsError(f'Existing report data differs: {name}')
            continue
        with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    first=pd.read_csv(OUT/'geo_only/seed_0/decisions.csv')
    example_labels=first.sla_deadline_utc.iloc[:8].tolist()
    # Diagnose ISO string validity only; do not rerun the frozen collector.
    # Scalar Timestamp preserves legitimate nine-digit fractional seconds too;
    # Python 3.10 datetime.fromisoformat supports fewer fractional digits.
    examples_valid=all(pd.Timestamp(s).tzinfo is not None for s in example_labels)
    assert examples_valid
    failure={'command':'python -B outputs/table3_7d_preflight/collect_formal_results.py','exit_code':1,
        'stage':'post-run collection','error_type':'ValueError','file':'outputs/table3_7d_preflight/collect_formal_results.py',
        'line':85,'statement':'pd.to_datetime(decisions.sla_deadline_utc, utc=True)',
        'error_excerpt':'time data "2023-07-01T04:36:12+00:00" does not match inferred format "%Y-%m-%dT%H:%M:%S.%f%z", at position 7',
        'diagnosis':'Valid ISO8601 timestamps with and without fractional seconds coexist. Pandas default inferred one fixed fractional format.',
        'first_eight_deadline_strings':example_labels,'diagnostic_examples_all_valid_iso8601':examples_valid,
        'frozen_collector_modified':False,'collector_rerun':False,'alternative_collector_used':False,
        'formal_acceptance_passed':False,'time_only_allowed_to_start':False}
    save('collection_failure.json',failure)
    check={'status':'simulation_complete_but_official_acceptance_blocked','created_at_shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'simulation_runs_completed':5,'actual_steps_per_seed':672,'total_steps':3360,
        'reported_by':'Read-only reporting checks; not a successful run of the frozen collector.',
        'raw_action_0_all_zero':True,'effective_defer_all_zero':True,'tasks_deferred_all_zero':True,
        'external_inputs_sha256_all_match':True,'input_rows_read':16800,'max_offline_input_errors':max_errors,
        'checkpoint_matches_freeze':True,'changed_frozen_files':frozen_changes+current_drift,
        'preexisting_files_rehashed':len(previous),'changed_preexisting_files':prior_drift,
        'seed42_untouched_and_excluded':True,'nine_metrics_independently_recomputed':True,
        'sample_std_ddof':1,'collector_exit_code':1,'official_collection_passed':False,
        'next_batch_allowed':False,'other_strategy_directories_exist':{v:(ROOT/p).exists() for v,p in frozen['output_directories'].items() if v!='geo_only'}}
    assert not any(check['other_strategy_directories_exist'].values())
    save('batch_reporting_checks.json',check)
    lines=['# Geo Only × 5 Seeds：第一批正式实验报告','',
        '**状态：五次仿真全部完成；运行后冻结校验通过；冻结结果收集器失败，正式验收暂时阻塞。不得进入第二批。**','',
        '本次严格使用run_commands.md第1节参数，只运行Geo Only，seed为0、10、20、30、40。未重训、未换种子、未跳过种子，未启动Time Only或Geo+Time。', '',
        '## 1. 执行和失败记录','',
        '运行前verify_frozen通过73份文件核查，确认输出目录不存在。原评估命令退出0，batch manifest记录complete、completed_runs=5；随后verify_frozen再次通过。',
        '按冻结命令执行collect_formal_results.py后退出1，错误位于第85行：deadline列既有带小数秒也有不带小数秒的合法ISO8601字符串，Pandas默认推断固定小数秒格式，遇到2023-07-01T04:36:12+00:00时报错。详见collection_failure.json。',
        '这是此前生成的结果收集器的解析缺陷，并非已观察到的仿真失败或时间偏移。按边界立即停止：没有修改任何冻结文件、没有重跑收集器、没有用替代收集器继续登记，也没有启动下一批。',
        '下列指标和检查仅为出具本报告对既有CSV/JSON的只读复核，不代表冻结collect已经通过。run_record.json和validated_index_05_runs.json没有成功生成。原始summary/steps/inputs/decisions/aggregate全部保留。', '',
        '## 2. 五个种子的实际状态','',
        '| Seed | 仿真 | 实际步数 | 首步UTC | 最后一步UTC | 冻结收集验收 |','|---:|---|---:|---|---|---|']
    for row in rows:lines.append(f'| {row["seed"]} | 完成 | {row["steps"]} | 2023-07-01 00:00 | 2023-07-07 23:45 | 阻塞：collector解析失败 |')
    lines+=['','五次均为完整七天，最后区间结束于2023-07-08 00:00 UTC。saved steps为0…671，无缺步、重复UTC或多执行一步的保存记录。没有新增环境reset计数hook；结论依据连续时间序列及冻结评估器一次reset的实现。','',
        '## 3. 各seed九项指标','',
        'Cost含TX，CO₂/能耗含DC+TX；SLA分母为已完成任务；水仅DC；CPU/GPU为DC和时间等权平均；defer为有效决策次数。','',
        '| Seed | Cost USD | CO₂ t | Energy MWh | Water m³ | SLA % | CPU % | GPU % | TX USD | Defer |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["seed"]} | {r["total_cost_usd"]:.2f} | {r["total_co2_t"]:.4f} | {r["total_energy_mwh"]:.4f} | {r["total_water_m3"]:.4f} | {r["sla_violation_pct_finished"]:.4f} | {r["avg_cpu_util_pct"]:.4f} | {r["avg_gpu_util_pct"]:.4f} | {r["transmission_cost_usd"]:.2f} | {r["tasks_deferred"]} |')
    lines+=['','## 4. 五种子均值与样本标准差','',
        'n=5，样本标准差ddof=1；与原评估器aggregate.csv一致。它们是完整保存数据的算术结果，不能被描述成冻结collector已验收的正式索引。','',
        '| 指标 | Mean | Sample Std |','|---|---:|---:|']
    for r in aggregates:lines.append(f'| {r["metric"]} | {r["mean"]:.6f} | {r["sample_std_ddof1"]:.6f} |')
    lines+=['','## 5. Geo Only动作与任务情况（只读复核）','',
        '| Seed | raw0 | effective defer | tasks_deferred | 强制回源 | 远程route决策 | TX kWh | TX kgCO₂ | 完成 | 违约完成 | 未完成 | 逾期未完成 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in diagnostic_rows:
        lines.append(f'| {r["seed"]} | {r["raw_action_0"]} | {r["effective_defer"]} | {r["tasks_deferred"]} | {r["forced_origin_after_deadline"]} | {r["remote_route_decisions"]} | {r["tx_energy_kwh"]:.6f} | {r["tx_carbon_kg"]:.6f} | {r["tasks_finished"]} | {r["sla_violated"]} | {r["unfinished_tasks"]} | {r["late_unfinished_tasks"]} |')
    lines+=['','以上每seed新到任务均26809，满足arrivals=finished+unfinished。强制回源与defer分别计数；资源排队不属于policy defer。远程路由允许并真实出现，TX费用、能耗和碳均按原规则统计。', '',
        '## 6. 数据、模型、配置与源码SHA256','',
        '- checkpoint：'+frozen['checkpoint']['sha256'],
        '- 七天源YAML：'+next(r['sha256'] for r in frozen['frozen_files'] if r['role']=='simulation_config'),
        '- workload文件：'+frozen['workload_file_sha256'],
        '- 外部源集合：'+frozen['external_source_set_sha256'],
        '- 五run共同实际外部输入流：'+frozen['expected_external_inputs_sha256'],
        '- runtime源码集合：'+frozen['source_version']['source_tree_sha256'],
        '- frozen_manifest：'+digest(FROZEN/'frozen_manifest.json'),
        '', '每run的source config、runtime sim_eval、checkpoint、workload文件/实际任务流和外部输入流hash见geo_only_run_hashes.csv。全部16800行保存的运行时输入与离线CSV逐点比较，绝对差均小于1e-10。',
        f'冻结的73个源/模型/配置/数据文件，以及审计包已登记文件均未变；另对原有{len(previous)}个文件复核，包含Seed42和旧30天结果，均保持原hash。',
        '评估沿用CPU、torch threads=2、Actor.eval和torch.inference_mode，无optimizer/backward路径。checkpoint文件前后SHA一致；本批没有额外安装逐run内存Actor张量hook，不虚构额外观测。', '',
        '## 7. 是否进入Time Only','',
        '**暂不允许。**仿真数据完整且只读检查未发现Geo Only约束或时间错误，但冻结collect未通过。须先经明确授权修复/复核收集器的合法ISO时间解析并重新冻结相关审计工具，再对当前保存结果重新进行收集验收；无需因本次解析错误自动重跑五次仿真。',
        '本次没有执行该修复、没有更新freeze、没有重跑或更换种子。此报告明确保留“仿真完成”和“正式验收未通过”的区别。']
    with (OUT/'geo_only_batch_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':check['status'],'runs_completed':5,'collector_exit_code':1,'next_batch_allowed':False,
        'mean_std':aggregates,'constraints':diagnostic_rows,'report':str(OUT/'geo_only_batch_report.md')},ensure_ascii=True))


if __name__=='__main__':main()
