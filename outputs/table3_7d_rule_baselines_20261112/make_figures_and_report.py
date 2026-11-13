"""Reproducible descriptive plots and report; consumes only verified saved results."""
import os
import sys
from pathlib import Path
from rbc_common import *
from collect_rule_results import LABELS

FIG = OUT/'figures'
FIG.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT/'matplotlib_cache')
sys.path.insert(0, str(ROOT/'.figure_deps'))
SKILL = Path('C:/Users/lll/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from audit_figure_collisions import audit_pdf
from audit_pdf_text import audit_pdf as audit_text
from validate_figure import validate_source, summarize

TITLES = ['Total cost (USD)', 'Total CO2 (t)', 'Total energy (MWh)', 'Total water (m3)',
          'SLA violations (%)', 'Average CPU utilization (%)', 'Average GPU utilization (%)',
          'Transmission cost (USD)', 'Effective defer decisions']
ORDER = STRATEGIES + ['geo_only','time_only','geo_time']
SHORT = {s: LABELS[s].replace('RBC ', '').replace('SAC ', '') for s in ORDER}

def save(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=FIG/f'{name}.alignment.json',
                                      tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(FIG/f'{name}.pdf', facecolor='white')
    fig.savefig(FIG/f'{name}.svg', facecolor='white')
    fig.savefig(FIG/f'{name}.png', dpi=300, facecolor='white')
    plt.close(fig)
    pdf = FIG/f'{name}.pdf'
    font = audit_text(pdf.read_bytes(), minimum_pt=5.0)
    write_json(FIG/f'{name}.font_audit.json',font)
    assert font['auditable'] and not font['below_minimum_count'], 'Small PDF glyphs.'
    collisions = audit_pdf(pdf)
    write_json(FIG/f'{name}.collision_audit.json',collisions)
    assert collisions['verdict'] == 'PASS', f'PDF collisions: {name}: {collisions["verdict"]}'

def markdown_table(frame):
    return '| '+' | '.join(frame.columns)+' |\n|'+ '|'.join(['---']*len(frame.columns))+'|\n'+ '\n'.join(
        '| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    validation = read_json(OUT/'validation.json'); assert validation['status']=='passed' and validation['runs']==25
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'legend.frameon':False})
    rbc = pd.read_csv(OUT/'metrics_per_seed.csv')
    sac = pd.DataFrame([{'strategy':r['strategy'],'seed':r['seed'],**r['metrics']} for r in sac_records()])
    runs = pd.concat([rbc,sac],ignore_index=True)
    runs[['strategy','seed']+METRICS].to_csv(FIG/'source_data_all_40_runs.csv',index=False,encoding='utf-8-sig')
    colors = ['#587F95']*5 + ['#B16E58']*3
    fig, axes = plt.subplots(3,3,figsize=(11.5,8.2))
    fig.subplots_adjust(left=.14,right=.985,bottom=.09,top=.91,wspace=.58,hspace=.56)
    for ax, metric, title in zip(axes.flat,METRICS,TITLES):
        for i,s in enumerate(ORDER):
            values = runs.loc[runs.strategy==s].sort_values('seed')[metric].to_numpy()
            ax.scatter(values, i+np.linspace(-.10,.10,5), s=10,color=colors[i],alpha=.5,linewidths=0)
            ax.errorbar(values.mean(),i,xerr=values.std(ddof=1),fmt='o' if i<5 else 's',
                        color=colors[i],markersize=4,elinewidth=1,capsize=2)
        ax.set_yticks(range(8),[LABELS[s] for s in ORDER]); ax.invert_yaxis()
        ax.set_title(title,loc='left',pad=9)
        ax.tick_params(axis='y',length=0,pad=5)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.locator_params(axis='x',nbins=3)
        ax.margins(x=.08,y=.10)
    fig.suptitle('Seven-day comparison under matched UTC inputs',x=.14,y=.975,ha='left',fontsize=13)
    fig.text(.14,.035,'Dots: all five seeds. Larger markers and whiskers: mean ± sample SD (ddof=1). Blue: RBC; brown: SAC.',fontsize=8)
    save(fig,'nine_metric_overview')
    pairs = pd.read_csv(OUT/'paired_differences_vs_sac.csv')
    fig, axes = plt.subplots(1,3,figsize=(11.5,7))
    fig.subplots_adjust(left=.25,right=.98,bottom=.12,top=.82,wspace=.35)
    contrasts = [(a,b) for a in STRATEGIES for b in ['geo_only','time_only','geo_time']]
    for index,(ax,metric,title) in enumerate(zip(axes,['total_cost_usd','total_co2_t','sla_violation_pct_finished'],
                                              ['Cost difference (USD)','CO2 difference (t)','SLA difference (pp)'])):
        ax.axvline(0,color='#999999',linewidth=.7,zorder=0)
        for i,(a,b) in enumerate(contrasts):
            values = pairs.loc[(pairs.rbc_strategy==a)&(pairs.sac_strategy==b)].sort_values('seed')[metric].to_numpy()
            ax.scatter(values,i+np.linspace(-.09,.09,5),s=10,color='#587F95',alpha=.5,linewidths=0)
            ax.errorbar(values.mean(),i,xerr=values.std(ddof=1),fmt='o',color='#244D65',markersize=4,capsize=2)
        ax.set_yticks(range(15),[SHORT[a]+' - '+SHORT[b] for a,b in contrasts] if index==0 else ['']*15)
        ax.invert_yaxis(); ax.set_title(title,loc='left',pad=10)
        ax.tick_params(axis='y',length=0,pad=7); ax.locator_params(axis='x',nbins=3)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    fig.suptitle('Matched-seed differences: RBC minus SAC',x=.25,y=.975,ha='left',fontsize=13)
    fig.text(.25,.91,'Negative values favor RBC for these three metrics.',fontsize=8)
    fig.text(.25,.04,'Each contrast uses the same five seeds; points show differences, whiskers show sample SD. No significance test.',fontsize=8)
    save(fig,'paired_cost_carbon_sla')
    numeric = pd.read_csv(OUT/'table3_combined_numeric.csv').set_index('strategy').loc[ORDER]
    fig,ax = plt.subplots(figsize=(15,4.8)); ax.axis('off')
    fig.subplots_adjust(left=.025,right=.995,bottom=.07,top=.88)
    positions = [.0,.235,.325,.415,.505,.595,.685,.775,.865,.955]
    headers = ['Controller','Cost\nUSD','CO2\nt','Energy\nMWh','Water\nm3','SLA\n%','CPU\n%','GPU\n%','TX cost\nUSD','Defer\ndecisions']
    for x,h in zip(positions,headers): ax.text(x,.98,h,ha='left' if x==0 else 'center',va='top',fontsize=9,weight='bold',transform=ax.transAxes)
    for i,s in enumerate(ORDER):
        y=.78-i*.085
        ax.text(0,y,LABELS[s],ha='left',va='center',fontsize=9,transform=ax.transAxes)
        for j,k in enumerate(METRICS):
            r=numeric.loc[s]; decimals = 0 if k in ['total_cost_usd','total_water_m3','transmission_cost_usd'] else 2
            if k=='tasks_deferred': decimals=1
            txt=f'{r[k+"_mean"]:.{decimals}f} ± {r[k+"_std"]:.{decimals}f}'
            ax.text(positions[j+1],y,txt,ha='center',va='center',fontsize=8,transform=ax.transAxes)
    fig.text(.025,.95,'Table 3-style comparison — seven days, five matched seeds',fontsize=13,weight='bold')
    fig.text(.025,.055,'Mean ± sample SD (ddof=1). Includes TX in cost, CO2 and energy. SLA denominator: completed tasks. Water: DC only.',fontsize=8)
    fig.text(.025,.025,'Existing built-in routing paths retained; Most Available uses absolute available CPU cores. This is not the paper’s original 30-day/10-seed table.',fontsize=8)
    save(fig,'table3_combined')
    source = summarize(validate_source(Path(__file__).read_text(encoding='utf-8'),'python'),False)
    write_json(FIG/'source_validation.json',source)
    formatted = pd.read_csv(OUT/'table3_combined.csv')
    short = pd.DataFrame({'Controller':[LABELS[s] for s in ORDER],
        **{title:[f'{numeric.loc[s,k+"_mean"]:.3f} ± {numeric.loc[s,k+"_std"]:.3f}' for s in ORDER]
           for title,k in zip(['Cost USD','CO2 t','Energy MWh','Water m3','SLA %','CPU %','GPU %','TX USD','Defer'],METRICS)}})
    best = {k: min(STRATEGIES,key=lambda s:numeric.loc[s,k+'_mean']) for k in [METRICS[0],METRICS[1],METRICS[2],METRICS[3],METRICS[4]]}
    diagnostics = rbc.groupby('strategy')[['new_arrivals','tasks_finished','unfinished_tasks','late_unfinished_tasks',
        'sla_violated','remote_route_decisions','tx_energy_kwh','tx_carbon_kg']].agg(['mean','std'])
    diagnostics.columns=['_'.join(c) for c in diagnostics.columns]
    diagnostics.reset_index().to_csv(OUT/'diagnostics_mean_std.csv',index=False,encoding='utf-8-sig')
    diag = diagnostics.reset_index()
    diagnostic_table = pd.DataFrame({'Controller': [LABELS[s] for s in diag.strategy],
        **{k: [f'{r[k+"_mean"]:.2f} ± {r[k+"_std"]:.2f}' for _,r in diag.iterrows()]
           for k in ['new_arrivals','tasks_finished','unfinished_tasks','late_unfinished_tasks','remote_route_decisions']}})
    paired_table = []
    for (a,b), group in pairs.groupby(['rbc_strategy','sac_strategy'],sort=False):
        paired_table.append({'RBC':LABELS[a], 'SAC':LABELS[b],
            **{title:f'{group[k].mean():.3f} ± {group[k].std(ddof=1):.3f}'
               for title,k in [('Cost USD','total_cost_usd'),('CO2 t','total_co2_t'),('SLA pp','sla_violation_pct_finished')]}})
    lines = ['# 七天规则算法正式对照实验报告','',
        '**验收通过：五种RBC × 五种子，共25/25次；每次672步，总16800步。**',
        '既有15次SAC仅离线读取，未重新运行。所有RBC的同seed任务流及外部输入SHA256均与SAC一致，且可从保存CSV独立重算。','',
        '## 配置与控制变量','',
        '2023-07-01 00:00 UTC至2023-07-08 00:00 UTC右开；首步07-01 00:00，末步07-07 23:45。五个DC原顺序、原资源、原物理模型、固定HVAC、Alibaba 2020原缩放、30天归一化、奖励和外部源文件均未改。',
        '种子0、10、20、30、40；Seed42未计入。CPU，torch线程2。唯一仿真配置字段变化为simulation.strategy。RBC未加载checkpoint、未调用Actor、未训练。',
        '详见control_variables.md与run_manifest.json；受保护源文件、配置、数据、checkpoint、原SAC核心结果指纹在运行前后保持一致。','',
        f'外部输入流SHA256：`{sac_records()[0]["external_inputs_sha256"]}`；模型仅作比较引用，SHA256：`{sac_records()[0]["checkpoint_sha256"]}`。每seed任务流和源文件指纹详见run_hashes.csv。','',
        '## 指标及完整结果','',
        'Mean ± sample standard deviation，n=5，ddof=1。费用/能耗/碳含TX；水仅DC；SLA为违约完成任务/全部完成任务；CPU/GPU按DC和时间等权；Defer为有效决策次数。','',markdown_table(short),'',
        '## 主要观察及解释边界','',
        f'规则组费用最低：{LABELS[best[METRICS[0]]] }；碳排最低：{LABELS[best[METRICS[1]]] }；能耗最低：{LABELS[best[METRICS[2]]] }；水耗最低：{LABELS[best[METRICS[3]]] }；SLA违约率最低：{LABELS[best[METRICS[4]]] }。',
        '不存在对九项指标都可直接宣布的统一最佳算法。CPU/GPU较高或Defer较少不能单独当作更优。','',
        '**重要局限**：Most Available实际选available_cores绝对数最大者，论文G.2描述为最高可用CPU比例，未擅自修改。',
        '**执行路径差异**：RBC直接进入目标DC.pending_tasks；SAC执行动作进入in_transit_tasks后再入队，因此SLA差异包含投递时机影响，不能全部归因于调度策略质量。网络模型及TX统计代码均保留。',
        '容量候选在DC实际调度前判断，同步分配之间没有增加资源预留；保留原队列和完成时SLA规则。七天/五种子/当前模型和归一化不等同于论文原始30天/十种子实验。','',
        '## 配对比较','',
        'paired_differences_vs_sac.csv共75行：5种RBC × 3种SAC × 5个相同seed。先逐seed计算RBC−SAC，再计算五差值均值和样本标准差；汇总见paired_differences_summary.csv。SLA及利用率差值为百分点。仅作描述性分析，不声明统计显著。','',
        markdown_table(pd.DataFrame(paired_table)),'',
        '## 任务完成与远程路由诊断','',markdown_table(diagnostic_table),'',
        '未完成任务不进入完成任务SLA分母；已过期未完成任务另列。每DC收到任务数和实际DC顺序在每run的summary.json中；TX能耗/碳的均值和样本标准差在diagnostics_mean_std.csv中。','',
        '## 验收证据','',
        '- sac_offline_regression.json：原15次SAC九指标离线回归通过。',
        '- seed0_gate_validation.json：五次首批完整运行及CSV重建指纹通过后才运行其余20次。',
        '- validation.json：25次步数、窗口、任务守恒、有效Defer=0、Local Only零远程/零TX、哈希及九指标重算通过。',
        '- validated_index_25_runs.json、run_hashes.csv：每run记录及完整输入/配置/原始结果指纹。',
        '- 每run保存run.log、summary.json、steps.csv、inputs.csv、decisions.csv、runtime_config.json、sim_eval.yaml、run_record.json。',
        '- figures/：九指标概览、逐seed配对差值及Table 3风格结果表；PNG/PDF/SVG，含绘图源数据和自动QA。','',
        '## 失败及异常','',
        '25次正式运行无失败、无提前终止、无重跑；全部正常完成。无任务流不一致或外部输入不一致。Most Available及入队路径差异为已知保留语义，未伪装为论文精确复现。','',
        '## 文件','',
        '本目录包含metrics_per_seed.csv、metrics_mean_std.csv、table3_combined.csv、table3_combined_numeric.csv、paired_differences_vs_sac.csv、paired_differences_summary.csv、diagnostics_mean_std.csv、run_hashes.csv、run_manifest.json及上述验收文件。',
        '图表为报告用途，不声称满足指定期刊的投稿排版要求。']
    (OUT/'final_rule_experiment_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FIGURES AND REPORT GENERATED')

if __name__ == '__main__': main()
