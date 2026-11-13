"""Validate RBC artifacts, recompute metrics, and compare matched seeds with saved SAC."""
from rbc_common import *
import argparse
import yaml

LABELS = {'local_only': 'RBC Local Only', 'lowest_carbon': 'RBC Lowest Carbon',
          'lowest_price': 'RBC Lowest Price', 'most_available': 'RBC Most Available',
          'round_robin': 'RBC Round Robin', 'geo_only': 'SAC Geo Only',
          'time_only': 'SAC Time Only', 'geo_time': 'SAC Geo+Time'}
UNITS = ['USD', 't', 'MWh', 'm3', 'percentage points', 'percentage points', 'percentage points', 'USD', 'decisions']

def aggregate(rows):
    frame = pd.DataFrame(rows)
    agg = frame.groupby('strategy', sort=False)[METRICS].agg(['mean', 'std'])
    agg.columns = ['_'.join(x) for x in agg.columns]
    agg['n'] = frame.groupby('strategy').size()
    return agg.reset_index()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed0-gate', action='store_true')
    args = parser.parse_args()
    regression()
    manifest = read_json(OUT / 'run_manifest.json')
    allowed = ['seed0_gate_passed'] if args.seed0_gate else ['runs_complete_pending_collection', 'complete_verified']
    assert manifest['status'] in allowed
    assert len(manifest['completed_runs']) == (5 if args.seed0_gate else 25)
    for path, digest in manifest['protected_files_sha256'].items():
        assert file_hash(ROOT/path) == digest, f'Protected file changed: {path}'
    records, rows, hash_rows = [], [], []
    expected = pd.date_range('2023-07-01', '2023-07-08', tz='UTC', inclusive='left', freq='15min')
    base = yaml.safe_load((ROOT/'configs/env/sim_config_table3_7d_aligned.yaml').read_text(encoding='utf-8'))
    refs = sac_records()
    for strategy in STRATEGIES:
        for seed in ([0] if args.seed0_gate else SEEDS):
            folder = OUT / strategy / f'seed_{seed}'
            record = read_json(folder/'run_record.json'); summary = read_json(folder/'summary.json')
            assert record['status'] == summary['status'] == 'complete_verified'
            assert summary['strategy'] == strategy and summary['seed'] == seed
            for name, digest in record['result_files_sha256'].items():
                assert file_hash(folder/name) == digest, f'RBC artifact changed: {folder/name}'
            config = yaml.safe_load((folder/'sim_eval.yaml').read_text(encoding='utf-8'))
            assert config['simulation']['strategy'] == strategy
            config['simulation']['strategy'] = 'manual_rl'; assert config == base
            runtime = read_json(folder/'runtime_config.json')
            assert not runtime['checkpoint_loaded'] and not runtime['actor_inference']
            assert runtime['normalization'] == read_json(PREFLIGHT/'normalization_frozen.json')
            assert runtime['dc_order'] == [1,2,3,4,5]
            steps = pd.read_csv(folder/'steps.csv', float_precision='round_trip')
            inputs = pd.read_csv(folder/'inputs.csv', float_precision='round_trip')
            decisions = pd.read_csv(folder/'decisions.csv', float_precision='round_trip', dtype={'task': str})
            assert len(steps) == 672 and np.array_equal(steps.step, np.arange(672))
            assert pd.DatetimeIndex(pd.to_datetime(steps.time_utc, utc=True)).equals(expected)
            assert len(inputs) == 3360 and not inputs.duplicated(['step','datacenter']).any()
            assert decisions.effective_decision.eq('route_rule').all()
            assert decisions.destination_dc_id.isin([1,2,3,4,5]).all()
            assert summary['tasks_deferred'] == 0 and steps.tasks_deferred.sum() == 0
            assert len(decisions) == summary['new_arrivals']
            assert summary['new_arrivals'] == summary['tasks_finished'] + summary['unfinished_tasks']
            assert summary['tasks_finished'] == int(steps.tasks_finished.sum()) == int(steps.sla_met.sum()+steps.sla_violated.sum())
            remote = int((decisions.origin_dc_id != decisions.destination_dc_id).sum())
            assert remote == summary['remote_route_decisions']
            counts = {str(k): int(v) for k,v in decisions.groupby('destination_dc_id').size().items()}
            assert all(counts.get(k,0) == v for k,v in summary['tasks_received_by_dc'].items())
            if strategy == 'local_only':
                assert remote == 0 and all(summary[k] == 0 for k in ['transmission_cost_usd','tx_energy_kwh','tx_carbon_kg'])
            wh, ih = hashlib.sha256(), hashlib.sha256()
            grouped = dict(tuple(decisions.groupby('step', sort=False)))
            for step, timestamp in enumerate(expected):
                group = grouped.get(step, decisions.iloc[:0])
                task_rows = [[str(t.task), int(t.origin_dc_id), float(t.cores_req), float(t.gpu_req),
                              float(t.mem_req), float(t.duration), float(t.bandwidth_gb), str(t.sla_deadline_utc)]
                             for t in group.itertuples()]
                assert group.time_utc.eq(timestamp.isoformat()).all()
                update_hash(wh, {'time': timestamp.isoformat(), 'tasks': task_rows})
                frame = inputs.loc[inputs.step == step]
                assert frame.datacenter.tolist() == [1,2,3,4,5]
                assert frame.time_utc.eq(timestamp.isoformat()).all()
                for row in frame.to_dict('records'): update_hash(ih, row)
            assert wh.hexdigest() == summary['realized_workload_sha256'], 'Saved decisions cannot reproduce task hash.'
            assert ih.hexdigest() == summary['external_inputs_sha256'], 'Saved inputs cannot reproduce input hash.'
            for ref in [r for r in refs if r['seed'] == seed]:
                assert wh.hexdigest() == ref['realized_workload_sha256']
                assert ih.hexdigest() == ref['external_inputs_sha256']
            calculated = metrics(steps)
            for key, value in calculated.items():
                assert np.isclose(value, summary[key], atol=1e-7, rtol=1e-12), key
            row = {'strategy': strategy, 'seed': seed, **calculated}
            for key in ['steps','new_arrivals','tasks_finished','unfinished_tasks','late_unfinished_tasks',
                        'sla_violated','remote_route_decisions','tx_energy_kwh','tx_carbon_kg']:
                row[key] = summary[key]
            rows.append(row); records.append(record)
            hash_rows.append({k: record[k] for k in ['strategy','seed','config_sha256','runtime_sim_config_sha256',
                'workload_file_sha256','realized_workload_sha256','external_source_set_sha256','external_inputs_sha256',
                'comparison_checkpoint_sha256']} | {'summary_sha256': file_hash(folder/'summary.json'),
                                                   'run_record_sha256': file_hash(folder/'run_record.json')})
    if args.seed0_gate:
        write_json(OUT/'seed0_gate_validation.json', {'status':'passed','runs':5,'steps_per_run':672,
            'hashes_reproduced_from_csv':True,'actual_workload_and_inputs_match_sac':True,
            'task_accounting_balanced':True,'metrics_recomputed':True})
        print('SEED0 GATE PASSED: all five full runs, CSV-reconstructed workload/input hashes, metrics, constraints.')
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'metrics_per_seed.csv', index=False, encoding='utf-8-sig')
    agg = aggregate(rows); agg.to_csv(OUT/'metrics_mean_std.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(hash_rows).to_csv(OUT/'run_hashes.csv', index=False, encoding='utf-8-sig')
    paired = []
    for row in rows:
        for ref in [r for r in refs if r['seed'] == row['seed']]:
            paired.append({'rbc_strategy': row['strategy'], 'sac_strategy': ref['strategy'], 'seed': row['seed'],
                           **{k: row[k]-ref['metrics'][k] for k in METRICS}})
    pairs = pd.DataFrame(paired)
    pairs.to_csv(OUT/'paired_differences_vs_sac.csv', index=False, encoding='utf-8-sig')
    ps = pairs.groupby(['rbc_strategy','sac_strategy'],sort=False)[METRICS].agg(['mean','std'])
    ps.columns = ['_'.join(x) for x in ps.columns]
    ps.reset_index().to_csv(OUT/'paired_differences_summary.csv',index=False,encoding='utf-8-sig')
    all_rows = rows + [{'strategy': r['strategy'], 'seed': r['seed'], **{k:r['metrics'][k] for k in METRICS}} for r in refs]
    combined = aggregate(all_rows)
    combined.to_csv(OUT/'table3_combined_numeric.csv',index=False,encoding='utf-8-sig')
    formatted = [{'controller': LABELS[r.strategy], **{k:f'{getattr(r,k+"_mean"):.3f} ± {getattr(r,k+"_std"):.3f}' for k in METRICS}}
                 for r in combined.itertuples()]
    pd.DataFrame(formatted).to_csv(OUT/'table3_combined.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'validated_index_25_runs.json', {'status':'complete_verified','runs':records,'formal_runs':25,
        'seed42_excluded':True,'sac_reference_runs':15,'actual_workload_and_external_hashes_match_sac':True})
    write_json(OUT/'verified_aggregate.json', {'n':5,'std_ddof':1,'strategies':agg.to_dict('records')})
    validation = {'status':'passed','runs':25,'steps_per_run':672,'total_steps':16800,
        'external_input_rows':84000,'paired_difference_rows':len(paired),'sac_offline_regression_runs':15,
        'runtime_hashes_reproduced_from_csv':True,'zero_effective_defer':True,'local_only_zero_remote_and_tx':True,
        'all_task_accounting_balanced':True,'protected_files_unchanged':True}
    write_json(OUT/'validation.json',validation)
    manifest.update(status='complete_verified', validation=validation,
                    collector_sha256=file_hash(Path(__file__)), common_metrics_sha256=file_hash(OUT/'rbc_common.py'))
    write_json(OUT/'run_manifest.json',manifest)
    print(json.dumps(validation))
    print(combined[['strategy','total_cost_usd_mean','total_co2_t_mean','sla_violation_pct_finished_mean']].to_string(index=False))

if __name__ == '__main__': main()
