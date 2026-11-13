"""Finalize validated 15-run tables and genuinely paired seed differences.

Reads completed results only. Does not import/construct/step a simulator.
Never changes raw results, prior accepted records or frozen configuration.
"""
from datetime import datetime,timezone,timedelta
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
COHORT=OUT.parent
ROOT=COHORT.parents[1]
ORDER=['geo_only','time_only','geo_time']
LABEL={'geo_only':'Geo Only','time_only':'Time Only','geo_time':'Geo+Time'}
SEEDS=[0,10,20,30,40]
METRICS=['total_cost_usd','total_co2_t','total_energy_mwh','total_water_m3','sla_violation_pct_finished',
         'avg_cpu_util_pct','avg_gpu_util_pct','transmission_cost_usd','tasks_deferred']
UNITS=['USD','t','MWh','m3','percentage points','percentage points','percentage points','USD','decisions']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save_json(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def save_csv(name,rows):
    with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    frozen_path=ROOT/'outputs/table3_7d_preflight/frozen_manifest.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    index_path=COHORT/'validated_index_15_runs.json'
    index=json.loads(index_path.read_text(encoding='utf-8'))
    verified=json.loads((COHORT/'verified_aggregate.json').read_text(encoding='utf-8'))
    assert index['status']=='complete' and index['formal_runs']==15 and index['seed42_excluded']
    assert index['cross_strategy_pairs_checked']
    assert {(r['strategy'],r['seed']) for r in index['runs']}=={(v,s) for v in ORDER for s in SEEDS}
    for field in ['checkpoint_sha256','config_sha256','workload_file_sha256','external_inputs_sha256']:
        assert len({r[field] for r in index['runs']})==1,field
    assert index['runs'][0]['external_inputs_sha256']==frozen['expected_external_inputs_sha256']
    assert sha(ROOT/frozen['checkpoint']['path'])==frozen['checkpoint']['sha256']
    registry_path=ROOT/'outputs/table3_7d_preflight/collector_v2_revision/collector_v2_registration.json'
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    assert sha(ROOT/registry['revision']['path'])==registry['revision']['sha256']
    assert sha(ROOT/registry['original']['path'])==registry['original']['sha256']
    offline=pd.read_csv(ROOT/frozen['offline_csv']['path'])
    assert sha(ROOT/frozen['offline_csv']['path'])==frozen['offline_csv']['sha256']
    offline['time_utc']=pd.to_datetime(offline.time_utc,utc=True)
    offline=offline.set_index(['dc_id','time_utc'])
    expected=pd.date_range(frozen['start_utc'],frozen['end_utc_exclusive'],freq='15min',inclusive='left')
    records=sorted(index['runs'],key=lambda r:(ORDER.index(r['strategy']),r['seed']))
    rows=[];constraints=[];hash_rows=[];input_errors=[]
    for record in records:
        variant,seed=record['strategy'],record['seed']
        folder=ROOT/record['output_directory']
        assert record['status']=='complete_verified' and record['actual_steps']==672
        assert record['actual_start_utc']==expected[0].isoformat() and record['actual_last_step_utc']==expected[-1].isoformat()
        assert record['actual_end_utc_exclusive']==frozen['end_utc_exclusive']
        for name,digest in record['result_files_sha256'].items():assert sha(folder/name)==digest
        summary=record['metrics'];steps=pd.read_csv(folder/'steps.csv');decisions=pd.read_csv(folder/'decisions.csv')
        inputs=pd.read_csv(folder/'inputs.csv')
        assert len(steps)==672 and np.array_equal(steps.step,np.arange(672))
        assert pd.DatetimeIndex(pd.to_datetime(steps.time_utc,utc=True)).equals(expected)
        assert len(inputs)==3360
        inputs['time_utc']=pd.to_datetime(inputs.time_utc,utc=True)
        assert not inputs.duplicated(['datacenter','time_utc']).any()
        max_error={key:0.0 for key in ['electricity_price_usd_mwh','carbon_intensity_g_kwh','ambient_temperature_c']}
        for dc,g in inputs.groupby('datacenter'):
            assert pd.DatetimeIndex(g.time_utc).equals(expected)
            target=offline.loc[dc].loc[expected]
            for key in max_error:
                error=float(np.abs(g[key].to_numpy()-target[key].to_numpy()).max())
                assert error<=1e-10
                max_error[key]=max(max_error[key],error)
        input_errors.append({'strategy':variant,'seed':seed,**max_error})
        kind=decisions.effective_decision
        executing=decisions.loc[kind.isin(['route','forced_origin_after_deadline'])]
        remote=int((executing.origin_dc_id!=executing.destination_dc_id).sum())
        raw0=int((decisions.action==0).sum());defer=int((kind=='defer').sum())
        forced=int((kind=='forced_origin_after_deadline').sum())
        expired=pd.to_datetime(decisions.time_utc,utc=True)>pd.to_datetime(decisions.sla_deadline_utc,format='ISO8601',utc=True,errors='raise')
        assert np.array_equal(expired,(kind=='forced_origin_after_deadline'))
        assert defer==summary['tasks_deferred'] and forced==int(steps.forced_origin_decisions.sum())
        assert decisions.action.between(0,5).all()
        if variant=='geo_only':assert raw0==0 and defer==0
        if variant=='time_only':
            assert remote==0
            assert (decisions.loc[decisions.action!=0,'action']==decisions.loc[decisions.action!=0,'origin_dc_id']).all()
            assert (steps[['transmission_cost_usd','tx_energy_kwh','tx_carbon_kg']]==0).all().all()
        if variant=='geo_time':
            batch=json.loads((COHORT/'geo_time/manifest.json').read_text(encoding='utf-8'))
            assert batch['variants']==['geo_time'] and not batch['runtime_simulation']['disable_defer_action']
            assert batch['checkpoint']['extra_info']['act_dim']==6
        assert summary['new_arrivals']==summary['tasks_finished']+summary['unfinished_tasks']
        rows.append({'strategy':variant,'seed':seed,'status':record['status'],'steps':672,
            'first_step_utc':record['actual_start_utc'],'last_step_utc':record['actual_last_step_utc'],
            'end_utc_exclusive':record['actual_end_utc_exclusive'],**{k:summary[k] for k in METRICS},
            'new_arrivals':summary['new_arrivals'],'tasks_finished':summary['tasks_finished'],
            'sla_violated':summary['sla_violated'],'unfinished_tasks':summary['unfinished_tasks'],
            'late_unfinished_tasks':summary['late_unfinished_tasks']})
        constraints.append({'strategy':variant,'seed':seed,'raw_action_0':raw0,'effective_defer':defer,
            'forced_origin_after_deadline':forced,'remote_execution_decisions':remote,
            'tx_cost_usd':summary['transmission_cost_usd'],'tx_energy_kwh':summary['tx_energy_kwh'],'tx_carbon_kg':summary['tx_carbon_kg'],
            'runtime_config_sha256':record['runtime_sim_config_sha256'],'passed':True})
        hash_rows.append({'strategy':variant,'seed':seed,**{k:record[k] for k in ['checkpoint_sha256','config_sha256',
            'runtime_sim_config_sha256','workload_file_sha256','realized_workload_sha256','external_inputs_sha256']}})
    frame=pd.DataFrame(rows)
    averages=[];formatted=[]
    for variant in ORDER:
        group=frame.loc[frame.strategy==variant]
        assert len(group)==5 and set(group.seed)==set(SEEDS)
        display={'strategy':variant,'n':5}
        for metric,unit in zip(METRICS,UNITS):
            mean=float(group[metric].mean());std=float(group[metric].std(ddof=1))
            assert np.isclose(mean,verified[variant][metric]['mean'],rtol=1e-12,atol=1e-7)
            assert np.isclose(std,verified[variant][metric]['sample_std_ddof1'],rtol=1e-10,atol=1e-8)
            averages.append({'strategy':variant,'metric':metric,'unit':'%' if unit=='percentage points' else unit,'mean':mean,'sample_std_ddof1':std,'n':5})
            display[metric]=f'{mean:.6f} +/- {std:.6f}'
        formatted.append(display)
    lookup={(r['strategy'],r['seed']):r for r in rows}
    pairs=[('geo_only','geo_time'),('time_only','geo_time'),('geo_only','time_only')]
    paired=[];paired_summary=[]
    for a,b in pairs:
        contrast=f'{a}_minus_{b}'
        for seed in SEEDS:
            ar=next(r for r in records if r['strategy']==a and r['seed']==seed)
            br=next(r for r in records if r['strategy']==b and r['seed']==seed)
            assert ar['realized_workload_sha256']==br['realized_workload_sha256']
            assert ar['external_inputs_sha256']==br['external_inputs_sha256']
            delta={'contrast':contrast,'strategy_a':a,'strategy_b':b,'seed':seed}
            for metric in METRICS:delta[metric]=float(lookup[(a,seed)][metric])-float(lookup[(b,seed)][metric])
            paired.append(delta)
        deltas=pd.DataFrame([r for r in paired if r['contrast']==contrast])
        for metric,unit in zip(METRICS,UNITS):
            mean=float(deltas[metric].mean());std=float(deltas[metric].std(ddof=1))
            ma=next(r['mean'] for r in averages if r['strategy']==a and r['metric']==metric)
            mb=next(r['mean'] for r in averages if r['strategy']==b and r['metric']==metric)
            assert np.isclose(mean,ma-mb,rtol=0,atol=1e-7)
            paired_summary.append({'contrast':contrast,'metric':metric,'unit':unit,'mean_paired_difference':mean,
                'sample_std_of_paired_differences_ddof1':std,'n_pairs':5,
                'a_greater_count':int((deltas[metric]>0).sum()),'a_lower_count':int((deltas[metric]<0).sum()),
                'equal_count':int((deltas[metric]==0).sum())})
    save_csv('metrics_per_seed.csv',rows)
    save_csv('metrics_mean_std.csv',averages)
    save_csv('strategy_mean_std_comparison.csv',formatted)
    save_csv('paired_differences_per_seed.csv',paired)
    save_csv('paired_differences_summary.csv',paired_summary)
    save_csv('action_constraint_checks.csv',constraints)
    save_csv('run_hashes.csv',hash_rows)
    save_csv('runtime_input_comparison.csv',input_errors)
    save_csv('geo_time_metrics_per_seed.csv',[r for r in rows if r['strategy']=='geo_time'])
    qa={'status':'passed','table3_frozen_7day_experiment':'complete','accepted_runs':15,
        'seeds_per_strategy':SEEDS,'steps_per_run':672,'total_environment_steps':10080,'runtime_dc_input_rows_compared':50400,
        'seed42_excluded':True,'checkpoint_sha256':frozen['checkpoint']['sha256'],
        'config_sha256':records[0]['config_sha256'],'external_inputs_sha256':frozen['expected_external_inputs_sha256'],
        'same_seed_workload_and_exogenous_hashes_matched':True,'geo_only_zero_defer':True,'time_only_zero_remote_and_tx':True,
        'geo_time_full_six_action_policy':True,
        'geo_time_observed_defer_and_remote_in_each_seed':all(r['effective_defer']>0 and r['remote_execution_decisions']>0 for r in constraints if r['strategy']=='geo_time'),
        'new_simulations_in_this_stage':5,'geo_only_or_time_only_rerun_in_this_stage':False,
        'paired_contrasts':3,'paired_seed_rows':15,'paired_metric_summaries':27,'std_ddof':1,
        'original_collector_sha256':registry['original']['sha256'],'registered_v2_sha256':registry['revision']['sha256'],
        'official_index_sha256':sha(index_path),'official_aggregate_sha256':sha(COHORT/'verified_aggregate.json'),
        'no_training_or_model_updates':'Frozen inference-only evaluator; pre/post frozen guard passed. No extra live parameter hook is claimed.',
        'checks_scope':'Current 15 formal result files, frozen identity and offline target input comparison; no 812-file historical rescan.',
        'completed_at_shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat()}
    save_json('final_verification.json',qa)
    save_json('paired_differences.json',{'definition':'For each identical seed, delta=A-B. Std is over five within-seed deltas, not independent group standard errors.',
        'percentage_units':'SLA/CPU/GPU differences are percentage points, not relative percentages.',
        'per_seed':paired,'summary':paired_summary,'inference':'Descriptive paired differences only; no p-value or significance claim with five seeds.'})
    lines=['# Table 3 七天冻结协议正式实验：已完成（15/15）','',
        '**Geo Only、Time Only、Geo+Time各五个种子全部完成并通过v2联合验收。**','',
        f'完成时间：{qa["completed_at_shanghai"]}。本阶段只新增Geo+Time五次，没有重跑前两组。', '',
        '## 1. 执行与联合验收','',
        '- 正式种子：0、10、20、30、40；Seed42只为预实验，不参与统计。',
        '- 每次672步，首步2023-07-01 00:00 UTC，末步2023-07-07 23:45 UTC，结束时钟2023-07-08 00:00 UTC。总计10080个环境步。',
        '- 原3620k checkpoint，obs34、act6、hidden256、LayerNorm开启；CPU、torch线程2；无训练或参数更新流程。',
        '- 运行前后verify_frozen通过；v2文件SHA与独立登记一致。collect_formal_results_v2.py --require-complete退出0、status=complete、runs_verified=15。',
        '- 原运行命令的策略标识为geo_time，未施加Geo Only屏蔽0或Time Only强制本地限制；SLA过期回源规则照常保留。',
        '- 全部15份run_record完整。三组模型/config/外部输入流相同；同一seed跨策略的实际任务流hash一致。',
        '- 额外只读核对50400行实际DC输入与离线3360点，对齐误差均小于1e-10；没有重新扫描812份历史文件。',
        '- 消息中的Python路径缺少目录分隔符；实际按冻结run_commands使用E:/a_myprofile/nature-AI data centres/sustain-cluster/.conda-env/python.exe，未换环境。','',
        '官方产物：`../validated_index_15_runs.json`、`../verified_aggregate.json`。已有05/10索引及原冻结收集器、原始结果均保留，不覆盖。','',
        '## 2. Geo+Time逐seed九项指标','',
        '| Seed | 状态/步数 | Cost USD | CO₂ t | Energy MWh | Water m³ | SLA % | CPU % | GPU % | TX USD | Defer |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['strategy']!='geo_time':continue
        lines.append(f'| {r["seed"]} | 通过/672 | {r["total_cost_usd"]:.2f} | {r["total_co2_t"]:.4f} | {r["total_energy_mwh"]:.4f} | {r["total_water_m3"]:.4f} | {r["sla_violation_pct_finished"]:.4f} | {r["avg_cpu_util_pct"]:.4f} | {r["avg_gpu_util_pct"]:.4f} | {r["transmission_cost_usd"]:.2f} | {r["tasks_deferred"]} |')
    lines+=['','三种策略全部15行指标、状态及任务完成/违约/未完成情况见metrics_per_seed.csv；Geo+Time单表见geo_time_metrics_per_seed.csv。','',
        '## 3. 最终三策略均值 ± 样本标准差','',
        '| 指标 | Geo Only | Time Only | Geo+Time |','|---|---:|---:|---:|']
    for metric in METRICS:
        cells=[]
        for variant in ORDER:
            a=next(r for r in averages if r['strategy']==variant and r['metric']==metric)
            cells.append(f'{a["mean"]:.6f} ± {a["sample_std_ddof1"]:.6f}')
        lines.append('| '+metric+' | '+' | '.join(cells)+' |')
    lines+=['','每策略n=5，样本std采用ddof=1。费用、能耗和碳包含TX，水仅DC；SLA分母为已完成任务；CPU/GPU为DC和时间等权平均；defer为有效决策次数。三组归一化继续使用原30天min/span窗口，没有改为7天参考。','',
        '## 4. 同seed配对差值','',
        '对每个seed分别计算A(seed)-B(seed)，再对五个差值取mean与sample std。正数表示A较大；百分率类使用百分点。未用三个独立均值替代配对数据，也未用两组独立方差相加替代配对差值方差。','',
        '| 指标 | Geo Only − Geo+Time | Time Only − Geo+Time | Geo Only − Time Only |','|---|---:|---:|---:|']
    for metric in METRICS:
        cells=[]
        for a,b in pairs:
            q=next(r for r in paired_summary if r['contrast']==f'{a}_minus_{b}' and r['metric']==metric)
            cells.append(f'{q["mean_paired_difference"]:.6f} ± {q["sample_std_of_paired_differences_ddof1"]:.6f}')
        lines.append('| '+metric+' | '+' | '.join(cells)+' |')
    lines+=['','paired_differences_per_seed.csv包含3个对比×5seed的15行九指标差值；paired_differences_summary.csv包含27行指标汇总，另记录每对比正/负/零差值次数。结果是描述性比较，不宣称统计显著性或超出五seed的普遍优势。','',
        '## 5. 动作约束和SLA过期回源','',
        '| 策略 | Seed | raw0 | 有效defer | 强制回源 | 远程执行决策 | TX USD | TX kWh | TX kgCO₂ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in constraints:
        lines.append(f'| {LABEL[r["strategy"]]} | {r["seed"]} | {r["raw_action_0"]} | {r["effective_defer"]} | {r["forced_origin_after_deadline"]} | {r["remote_execution_decisions"]} | {r["tx_cost_usd"]:.6f} | {r["tx_energy_kwh"]:.6f} | {r["tx_carbon_kg"]:.6f} |')
    lines+=['','Geo Only零raw0、零有效defer；Time Only所有执行回来源且每步TX费用/能耗/碳为0；Geo+Time保留完整六动作，五个seed实际均发生defer和远程执行。过期动作被强制回源时不计有效defer，因此raw0与有效defer可以不同。本地投递缓冲不是远程迁移。','',
        '## 6. 身份指纹与验收边界','',
        '- checkpoint SHA256：'+frozen['checkpoint']['sha256'],
        '- source config SHA256：'+records[0]['config_sha256'],
        '- 实际外部输入流SHA256：'+frozen['expected_external_inputs_sha256'],
        '- collector v2 SHA256：'+registry['revision']['sha256'],
        '- validated_index_15 SHA256：'+qa['official_index_sha256'],
        '- verified_aggregate SHA256：'+qa['official_aggregate_sha256'],
        '', '逐run文件/实际任务流hash见run_hashes.csv和官方run_record；最终报告自己的产物校验见delivery_manifest.json。',
        '当前冻结七天、五seed、256+LN模型的Table3三策略实验已经完成。该结论不等于作者原始Table3数值严格复现：论文披露30天/10seed/hidden64，而原始checkpoint与实际窗口的来源链缺口仍如先前溯源报告所述。',
        '本次没有修改源码、奖励、配置、外部数据或checkpoint；没有重跑Geo Only/Time Only或纳入Seed42。没有额外逐run内存Actor张量hook，模型不更新依据为冻结推理代码、原checkpoint一致以及运行前后冻结校验。']
    with (OUT/'final_table3_experiment_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    delivery={'status':'complete_15_of_15','files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(OUT.iterdir()) if p.is_file()],
        'official_outputs':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in [index_path,COHORT/'verified_aggregate.json']]}
    save_json('delivery_manifest.json',delivery)
    print(json.dumps({'status':'complete','accepted_runs':15,'paired_contrasts':3,'paired_seed_rows':15,
        'geo_time_mean_std':[r for r in averages if r['strategy']=='geo_time'],
        'report':str(OUT/'final_table3_experiment_report.md')},ensure_ascii=True))


if __name__=='__main__':main()
