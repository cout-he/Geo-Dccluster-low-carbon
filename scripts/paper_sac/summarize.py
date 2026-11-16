"""Summarize one selected evaluation experiment without pooling trained models."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'scripts.paper_sac'

import argparse
import json
import pandas as pd
from .config import resolve, STRATEGIES
from .metrics import summarize_records
from .provenance import relative, utc_now, new_id
from .replay import atomic_json


def summarize(output, evaluation_run, allow_incomplete=False, include_smoke=False):
    base = resolve(output)
    manifest_path = base / 'manifests' / evaluation_run / 'experiment_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete' and not allow_incomplete:
        raise ValueError('Evaluation experiment is incomplete')
    rows = []
    for item in manifest['runs']:
        if item['status'] != 'complete':
            continue
        row = json.loads(resolve(item['result_file']).read_text())
        if row.get('run_kind') == 'smoke' and not include_smoke:
            raise ValueError('Smoke results require --include-smoke and are not paper results')
        rows.append(row)
    cells = {(r['train_seed'], r['eval_seed'], r['strategy']) for r in rows}
    expected = {(t, e, s) for t in (42, 43, 44) for e in (0, 1, 2) for s in STRATEGIES}
    if not allow_incomplete and cells != expected:
        raise ValueError('Expected all 27 paired evaluation cells')
    if not allow_incomplete and any(r['actual_steps'] != 2880 or r['run_kind'] != 'production' for r in rows):
        raise ValueError('Paper summary requires full 2880-step production evaluations')
    result = summarize_records(rows)
    result.update(evaluation_run_id=evaluation_run, generated_at=utc_now(),
                  complete_matrix=cells == expected, missing_cells=[list(v) for v in sorted(expected-cells)],
                  run_kind='smoke' if any(r.get('run_kind') == 'smoke' for r in rows) else 'production')
    destination = base / 'final_results' / evaluation_run / new_id('summary')
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(destination / 'metrics_per_run.csv', index=False)
    pd.DataFrame(result['per_model']).to_csv(destination / 'metrics_per_model.csv', index=False)
    pd.DataFrame(result['across_models']).to_csv(destination / 'metrics_mean_std.csv', index=False)
    atomic_json(destination / 'summary.json', result)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/paper_sac_v1')
    parser.add_argument('--evaluation-run', required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--include-smoke', action='store_true')
    args = parser.parse_args()
    path = summarize(args.output, args.evaluation_run, args.allow_incomplete, args.include_smoke)
    print(f'Summary: {relative(path)}')


if __name__ == '__main__':
    main()
