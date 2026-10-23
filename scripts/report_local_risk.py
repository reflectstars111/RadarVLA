"""Render the accepted Local Risk migration and its traceable comparison report."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from radar_vla.agent50_migration import write_json,sha256
from radar_vla.local_risk_migration import state
from collections import Counter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--test-log',type=Path,required=True);a=p.parse_args();out=a.directory
    migration=json.loads((out/'migration_report.json').read_text());acceptance=json.loads((out/'acceptance.json').read_text())
    excluded=json.loads((out/'exclusion_audit_report.json').read_text());sources=json.loads((out/'source_audit_report.json').read_text())
    assert all(x['status']=='PASS' for x in (migration,acceptance,excluded,sources))
    log=a.test_log.read_text();assert 'failed' not in log.lower() and 'passed' in log
    total=migration['totals'];splits=migration['groups']['split'];ec=excluded['counts'];sc=sources['counts']
    exclusion_comparison=Counter()
    for row in map(json.loads,(out/'excluded_samples.jsonl').open()):
        if row['status']!='reconstructed':continue
        for i in range(3):
            exclusion_comparison[f'{i+1}s/global_{state(row["global_risk"],i)}__local_{state(row["local_risk"],i)}']+=1
        exclusion_comparison['sample_with_hazard_source_center_beyond_50m']+=any(s.get('center_beyond_50m',False) for s in row['sources'])
        exclusion_comparison['local_sft_teacher_available']+=row['local_planner_teacher_mode'] is not None
    write_json(out/'exclusion_comparison.json',dict(counts=exclusion_comparison,
        note='Per-sample diagnostic counts; center range is never used to filter GT'))
    lines=['# Task 2：Local Agent Collision Risk 验收报告','',
        '本轮完成标签迁移、原始证据核对、Global/Local 对照及教师迁移。没有启动真实数据训练，没有优化器更新，旧数据与权重保留。','',
        '## 发布数据','',f'- 样本 {migration["samples"]}，episode {migration["episodes"]}，原 split、检测 GT、Radar 引用及 Agent/Ego Future 不变。',
        '- Local 只使用 Task 1 保留的 Agent；行人/cyclist 仅保留原 Global 诊断语义。',
        '- 输入 Task 1 manifest SHA256：`'+migration['source_manifest_sha256']+'`。',
        '- 输出 manifest SHA256：`'+migration['manifest_sha256']+'`。',
        '- 新 schema：`local_agent_collision_v1`，配置 `configs/l40s_qwen25_3b_localrisk_v1.json`。','',
        '## Global / Local 对照','',
        '下面危险/安全/未知仅指该窗口累计 pcol；Global 危险 / Local 安全要求 Local 独立计算得到有效负例，不由“危险来自域外”推断。连续风险与 SFT 复合危险另计。','',
        '| split | 帧数 | 窗口 | Global危险/Local危险 | Global危险/Local安全 | Global危险/Local未知 | Local正例 | Local负例 | Local未知 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for split in ('train','val','test'):
        c=splits[split]
        for h in (1,2,3):
            prefix=f'{h}s/'
            values=[c.get(prefix+k,0) for k in ('global_danger__local_danger','global_danger__local_safe','global_danger__local_unknown','local_danger','local_safe','local_unknown')]
            lines.append(f'| {split} | {c["samples"]} | {h}s | '+' | '.join(map(str,values))+' |')
    lines += ['','| split | dmin有效 | areq有效 | areq有效零 | areq有效正值 | SFT可用 | 新增可用 | 新增不可用 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('train','val','test'):
        c=splits[split];keys=('dmin/valid','areq/valid','areq/valid_zero','areq/valid_positive','sft_available','sft_became_available','sft_became_unavailable')
        values=[]
        for k in keys:
            count=c.get(k,0)
            values.append(f'{count} ({count/c["samples"]:.1%})' if k in ('dmin/valid','areq/valid') else str(count))
        lines.append('| '+split+' | '+' | '.join(values)+' |')
    lines += ['','Local pcol 有效覆盖率：']
    for split in ('train','val','test'):
        c=splits[split]
        rates=[(c.get(f'{h}s/local_danger',0)+c.get(f'{h}s/local_safe',0))/c['samples'] for h in (1,2,3)]
        lines.append('- '+split+'：1/2/3 s = '+' / '.join(f'{v:.1%}' for v in rates)+'。')
    lines += ['','每个 split/episode/场景的完整计数见 migration_report.json。当前 Local 空集合按旧约定，dmin 和 areq 不适用；无效 areq 存 0 不代表有效零刹车。','',
        '### 当前样本的危险来源','',
        f'- 危险来源观测 {sc.get("hazard_source_observations",0)} 个；中心超过 50 m 的观测 {sc.get("center_beyond_50m",0)} 个。中心距离只用于诊断，不用于 GT 纳入。',
        f'- 原 BBox/FOV 判定为 50 m 外 {sc.get("spatial/outside_50m",0)}，FOV 外 {sc.get("spatial/outside_fov",0)}。',
        f'- Local 保留来源 {sc.get("membership/retained_local",0)}，忽略类别来源 {sc.get("membership/ignored_class",0)}，空间域外来源 {sc.get("membership/outside_spatial_domain",0)}。',
        f'- 未进入 Planner selected 的来源观测 {sc.get("not_planner_selected",0)}；逐帧、逐 actor 见 risk_sources.jsonl。','',
        '### 旧排除帧复核','',
        f'- 混合风险排除记录 {ec["samples"]} 条，成功重建 {ec.get("reconstructed",0)}，无法恢复 {ec.get("unrecoverable",0)}。',
        f'- 含空间域外危险来源 {ec.get("domain_excluded_hazard",0)} 条，其中通过原 FOV 但在 50 m 外的危险来源 {ec.get("beyond_50m_hazard",0)} 条。',
        f'- 含忽略类别危险来源 {ec.get("ignored_class_hazard",0)}，含 Planner 未选入的危险来源 {ec.get("unselected_hazard",0)}，Local 未来不完整 {ec.get("future_missing",0)}。',
        f'- 危险来源中心距离超过 50 m 的排除帧 {exclusion_comparison.get("sample_with_hazard_source_center_beyond_50m",0)} 条；中心越界与 BBox/FOV 不相交分别记录。',
        '- 这些标志可重叠。旧原因将风险未知与不可见危险混合记录，重建结果不能证明当时排除器的唯一原因。未将任何排除帧补入训练集。',
        '- 诊断重建的 Global 使用原六点协议、全局覆盖和原风险目标界限，已与原标量实现对照测试；现有 8,835 帧的 Global 直接逐值复制。','',
        '现有 cohort 的 Global/Local 差异不能代表原始全部采集帧：旧转换曾排除无法归因的危险或未知样本。必须同时查看上述排除帧审计。','',
        '## 有效性与教师','',
        '- Local 覆盖依赖固定当前 Local 集合和各目标未来，域外 actor 出现/消失不使 Local 变未知。',
        '- 保留六个实际未来轨迹点；原始 30 帧有效性用于防止跨缺失证据插值，不改成 30 点位置轨迹。',
        '- 每个窗口独立确认 pcol：观测危险可确认正例，负例要求完整覆盖。dmin 缺未来时无效，已观测零间距下界除外。',
        '- 全部 17 个恒定刹车候选均已检查并存状态，不假设制动风险单调；当前危险接近、无解、未来不足时 areq 无效。',
        '- SFT 用 Local 风险重判模式，保留原 Planner 目标与顺序；危险无法归因于所选目标或证据不足时只标记 SFT 不可用，Grounding 帧仍保留。',
        '- 旧 risk_label/agent_risk_labels/risk_agents 保持原 Global 含义；新版 Dataset、训练/恢复/评估及可视化检查显式风险协议。','',
        '## 验收与追溯','',
        f'- 全量验收 PASS：{acceptance["samples"]} 帧、{acceptance["episodes"]} episode、{acceptance["sensor_files_existing"]} 个传感器文件路径存在。',
        '- 真实样本覆盖：'+', '.join(r['case'] for r in acceptance['representatives'])+'。',
        f'- 真实样本前向/loss/反向检查通过，参数最大变化 {acceptance["parameter_max_change"]}；Radar、mask、GT 和未来张量一致。',
        '- 专项回归：`'+log.strip().splitlines()[-1]+'`。',
        '- test 仅验收标签和完整性，没有用于参数选择。',
        '- samples.jsonl：逐样本五维对照、mask、来源与教师变化；local_agent_risk_labels：逐 Agent 标签。',
        '- local_risk.braking_candidates：全部候选状态与危险来源；coverage：有效观测区间与时间网格。',
        '- raw_evidence.json、provenance.json：原始帧/metadata/选中 anchor 摘要及执行阶段源码；implementation_provenance.json：最终实现和验收摘要。',
        '- excluded_samples.jsonl：旧排除帧逐条诊断；risk_sources.jsonl：原始空间域归因。','',
        '下一步训练必须配套新 manifest 与新配置：先重新训练 Grounding，再从相同 Local Risk 协议的 Grounding 初始化 SFT。本任务没有启动训练。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    paths=[*Path('src').glob('*.py'),Path('configs/l40s_qwen25_3b_localrisk_v1.json'),
        Path('docs/LOCAL_RISK_MIGRATION.md'),Path('tests/test_local_risk.py'),Path('tests/test_agent50.py'),
        *[Path('scripts')/name for name in ('migrate_local_risk.py','audit_local_risk.py','verify_local_risk.py','report_local_risk.py','diagnose_sft_trajectory.py')]]
    outputs=('manifest.jsonl','samples.jsonl','raw_evidence.json','migration_report.json','provenance.json','acceptance.json',
        'excluded_samples.jsonl','exclusion_audit_report.json','exclusion_comparison.json','risk_sources.jsonl','source_audit_report.json','REPORT.md')
    write_json(out/'implementation_provenance.json',dict(schema='local_risk_delivery_v1',
        source_files={str(p.resolve()):sha256(p) for p in paths},artifacts={n:sha256(out/n) for n in outputs},
        tests=dict(log=str(a.test_log.resolve()),sha256=sha256(a.test_log),summary=log.strip().splitlines()[-1]),
        verification_note='Final implementation adds a conservative nonempty-Local/empty-Planner teacher guard; all 8835 published modes were independently rechecked by verify_local_risk.py. Execution-stage hashes remain in provenance.json.',
        training_started=False,optimizer_steps=0,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print('REPORT PASS',flush=True)

if __name__=='__main__':main()
