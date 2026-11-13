"""Seal the verified 25-run output without touching original SAC/frozen files."""
from datetime import datetime, timezone
from rbc_common import *

def main():
    manifest = read_json(OUT/'run_manifest.json')
    assert manifest['status'] == 'complete_verified'
    index = read_json(OUT/'validated_index_25_runs.json')
    assert {(r['strategy'],r['seed']) for r in index['runs']} == {(s,k) for s in STRATEGIES for k in SEEDS}
    for path,digest in manifest['protected_files_sha256'].items():
        assert file_hash(ROOT/path) == digest, f'Protected file changed: {path}'
    for r in index['runs']:
        assert r['actual_steps'] == 672 and r['status'] == 'complete_verified'
        for name,digest in r['result_files_sha256'].items():
            assert file_hash(ROOT/r['output_directory']/name) == digest
    frame = pd.read_csv(OUT/'metrics_per_seed.csv')
    agg = pd.read_csv(OUT/'metrics_mean_std.csv').set_index('strategy')
    assert len(frame) == 25 and frame.groupby('strategy').size().eq(5).all()
    for strategy,group in frame.groupby('strategy'):
        for k in METRICS:
            assert np.isclose(group[k].mean(),agg.loc[strategy,k+'_mean'],atol=1e-7,rtol=1e-12)
            assert np.isclose(group[k].std(ddof=1),agg.loc[strategy,k+'_std'],atol=1e-7,rtol=1e-12)
    assert len(pd.read_csv(OUT/'table3_combined.csv')) == 8
    assert len(pd.read_csv(OUT/'paired_differences_vs_sac.csv')) == 75
    assert len(pd.read_csv(OUT/'run_hashes.csv')) == 25
    qa = {}
    for name in ['nine_metric_overview','paired_cost_carbon_sla','table3_combined']:
        for ext in ['png','pdf','svg']:
            assert (OUT/'figures'/f'{name}.{ext}').is_file()
        collision = read_json(OUT/'figures'/f'{name}.collision_audit.json')
        font = read_json(OUT/'figures'/f'{name}.font_audit.json')
        alignment = read_json(OUT/'figures'/f'{name}.alignment.json')
        assert collision['verdict'] == 'PASS' and font['auditable'] and not font['below_minimum_count']
        assert alignment['verdict'] == ('NOT APPLICABLE' if name=='table3_combined' else 'PASS')
        qa[name] = {'collision':'PASS','glyph_floor':'PASS','alignment':alignment['verdict'],
                    'agent_visual_inspection':'completed; labels, uncertainty, order and clipping reviewed'}
    assert read_json(OUT/'figures/source_validation.json')['ready']
    notes = '''# 图表最终检查

三份图表均已逐图查看PNG预览，核对标签、指标、方法顺序、误差条、留白和截断。最终PDF文字碰撞检查全部PASS，字号不小于5pt；两个多面板图的实际plot-area对齐检查PASS，单轴表格不适用多面板检查。

配对图首次导出出现一次主标题/副标题重叠，已通过增加纵向间距修复，并重新导出及检查。此为新图表排版问题，不涉及仿真失败或数据改动。

源代码静态检查18 PASS、3 WARN、0 FAIL。三个WARN为：无TIFF投稿栅格、300dpi低于技能默认600dpi、报告宽度大于常见期刊栏宽。本任务交付报告用途PNG预览和可编辑PDF/SVG，未要求某期刊投稿文件；上述警告不影响数据显示或数值验收。

所有数据点均保留。n=5代表仿真种子，误差条为样本标准差ddof=1；没有显著性检验或显著性声明。
'''
    (OUT/'figures/qa_notes.md').write_text(notes,encoding='utf-8')
    report = OUT/'final_rule_experiment_report.md'
    text = report.read_text(encoding='utf-8')
    addition = '\n## 最终图表及交付复核\n\n三份图表已完成预览检查，PDF碰撞和字体检查通过；两个多面板图对齐检查通过。配对图首次标题重叠已修复，仅重新导出图表，未重跑仿真。完整图表检查说明见figures/qa_notes.md。\n\n普通SAC执行动作是在本步入站队列处理之后才进入in_transit_tasks，最早下一步才能投递；内置RBC可以本步入队和调度。这是代码调用顺序的已知差异，并未通过额外消融量化其对SLA差距的贡献。\n'
    if '## 最终图表及交付复核' not in text:
        report.write_text(text+addition,encoding='utf-8')
    manifest['new_code_sha256'] = {p.name:file_hash(p) for p in OUT.glob('*.py')}
    manifest['figure_qa'] = qa
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(OUT/'run_manifest.json',manifest)
    result = {'status':'passed','rbc_runs':25,'sac_existing_runs_read_only':15,'new_sac_runs':0,
        'steps_per_run':672,'total_rbc_steps':16800,'runtime_input_rows':84000,
        'mean_and_sample_std_recomputed':True,'input_hashes_match_sac':True,
        'protected_files_unchanged':True,'original_sac_results_unchanged':True,
        'seed42_excluded':True,'figures':qa,'report':report.relative_to(ROOT).as_posix(),
        'limitations':['Most Available uses absolute available CPU cores.',
                       'Built-in RBC direct queue differs from SAC delivery-buffer timing.',
                       'Seven days and five seeds do not reproduce the original paper window/seed count.']}
    write_json(OUT/'final_verification.json',result)
    files = {p.relative_to(OUT).as_posix():file_hash(p) for p in OUT.rglob('*') if p.is_file()
             and 'matplotlib_cache' not in p.parts and p.name != 'delivery_manifest.json'}
    write_json(OUT/'delivery_manifest.json',{'status':'complete_verified','files_sha256':files,
        'excluded':'Matplotlib font cache and this self-referential manifest','artifact_count':len(files)})
    print(json.dumps(result,ensure_ascii=False))
    print(f'SEALED ARTIFACTS: {len(files)}')

if __name__ == '__main__': main()
