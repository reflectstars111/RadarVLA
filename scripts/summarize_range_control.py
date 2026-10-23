"""Report predeclared paired epoch results, including shared critical matches."""
import argparse
import json
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    history = json.loads((output / 'metrics.json').read_text())
    finished = (output / 'results.json').exists()
    epoch = history[-1]['epoch']
    rows = {a: [json.loads(s) for s in (output / f'val_{a}_epoch_{epoch:03d}.jsonl').read_text().splitlines()]
            for a in ['full118', 'crop50']}
    baseline = output.parent / 'risk_layer_diagnosis_20261019/early_input_evidence.json'
    groups = {r['sample_id']: r['group'] for r in json.loads(baseline.read_text())['evidence']}
    common = {}
    maps = {a: {(r['sample_id'], c['id']): c for r in rs for c in r['critical'] if c['matched']}
            for a, rs in rows.items()}
    ids = set(maps['full118']) & set(maps['crop50'])
    for a in maps:
        common[a] = {}
        for key in ['velocity_error_mps', 'radial_error_mps', 'future_ade_m']:
            shared = [i for i in ids if all(maps[b][i][key] is not None for b in maps)]
            common[a][key] = dict(count=len(shared), mean=mean([maps[a][i][key] for i in shared]))
    subgroup = {}
    for a, rs in rows.items():
        miss = [r for r in rs if groups.get(r['sample_id']) == 'miss']
        critical = [c for r in miss for c in r['critical']]
        subgroup[a] = dict(original_risk_miss_samples=len(miss),
            pcol3_at_05_positive=sum(r['risk'][2] >= .5 for r in miss),
            pcol3_mean=mean([r['risk'][2] for r in miss]),
            critical_agents=len(critical), critical_detected=sum(c['matched'] for c in critical))
    comparison = dict(epoch=epoch, training_complete=finished, common_matched_critical=common,
                      original_risk_miss_subgroup=subgroup, history=history)
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    lines = ['# 50m雷达输入裁剪对照实验', '',
        f"状态：{'两组均完成5个epoch' if finished else '训练进行中，当前结果为中间结果'}。当前报告epoch：{epoch}。", '',
        '完整6847帧Grounding训练集、1472帧验证集；标签不变，不使用测试集评估。两组相同随机初始化、批次顺序、AdamW、学习率0.0003、有效batch=8，各训练5个epoch；仅输入裁剪不同。', '',
        '| Epoch | 输入 | 检测召回 | 危险车辆召回 | Pcol3 AP | Pcol3 Brier | dmin MAE(m) | areq MAE(m/s²) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for h in history:
        for a, v in h['validation'].items():
            lines.append(f"| {h['epoch']} | {a} | {fmt(v['detection']['recall'])} | {fmt(v['critical_agents']['recall'])} | {fmt(v['risk']['pcol3']['average_precision'])} | {fmt(v['risk']['pcol3']['brier'])} | {fmt(v['risk']['dmin']['mae'])} | {fmt(v['risk']['areq']['mae'])} |")
    lines += ['', '## 原36个Risk漏检样本', '',
              '| 输入 | 样本数 | Pcol3≥0.5 | 危险车辆检出/总数 | 平均Pcol3 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for a, s in subgroup.items():
        lines.append(f"| {a} | {s['original_risk_miss_samples']} | {s['pcol3_at_05_positive']} | {s['critical_detected']}/{s['critical_agents']} | {fmt(s['pcol3_mean'])} |")
    lines += ['', 'Pcol3阈值统计是风险事件识别，不是LLM的SHORT召回率。', '',
              '## 两组共同检出的危险车辆', '',
              '| 误差 | 共同有效数量 | full118 | crop50 |', '| --- | ---: | ---: | ---: |']
    for key in common['full118']:
        f, c = common['full118'][key], common['crop50'][key]
        lines.append(f"| {key} | {f['count']} | {fmt(f['mean'])} | {fmt(c['mean'])} |")
    lines += ['', '## 解释边界', '',
        '- 固定第5个epoch为主要对照；完整展示各epoch，不能挑选最有利epoch作为结论。',
        '- 单个配对随机种子只提供初步证据；不等于跨种子显著性结论。',
        '- 裁剪由256个原生距离格变为109格，未做resize；同步裁剪Power、Doppler、有效掩码、XY和LOS。',
        '- 50m外背景监督在两组中均被屏蔽；边界正例和Risk标签保持一致。',
        '- 裁剪同时改变GroupNorm统计、8×8池化布局和检测候选数量（3456→1512），结果不能单独归因于远处杂波。',
        '- 两组重新训练Grounder，均未初始化旧权重；没有训练LLM，也没有模式召回或rollout改善的结论。',
        '- 与历史两卡训练不同，本轮两组在同一张卡依次计算，共享原始批次读取；有效batch一致，但采样/归约细节不同。',
        '- 非共同匹配集合上的运动误差受检出集合变化影响；上表补充了共同检出车辆的比较。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(epoch=epoch, training_complete=finished, original_miss=subgroup,
                         primary=history[-1]['validation']), allow_nan=False))


if __name__ == '__main__':
    main()
