"""Render the Task-3 comparison and bind final artifacts to execution provenance."""
import argparse,json,shutil,subprocess
from collections import Counter,defaultdict
from pathlib import Path
from radar_vla.agent50_migration import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--test-log',type=Path,required=True);a=p.parse_args();out=a.directory
    migration=json.loads((out/'migration_report.json').read_text());accept=json.loads((out/'acceptance.json').read_text());prov=json.loads((out/'provenance.json').read_text())
    mode_audit=json.loads((out/'mode_attribution_report.json').read_text())
    if mode_audit['status']!='PASS':raise ValueError('Mode attribution audit required')
    if migration['status']!='PASS' or accept['status']!='PASS':raise ValueError('Acceptance required before final report')
    log=a.test_log.read_text()
    if 'failed' in log.lower() or 'passed' not in log:raise ValueError('Passing test log required')
    if a.test_log.resolve()!=(out/'test_results.log').resolve():shutil.copy2(a.test_log,out/'test_results.log')
    origin=defaultdict(Counter);unknown=Counter();attribution=Counter();recoveries=Counter();rows=0;changed=[];coverage=defaultdict(lambda:defaultdict(set));coverage_counts=defaultdict(Counter)
    for line in (out/'samples.jsonl').open():
        r=json.loads(line);coverage[r['scenario']][r['split']].add(r['episode_id']);coverage_counts[r['scenario']][r['split']]+=1;c=origin[r['origin']];c['samples']+=1;c['mode/'+r['mode_label']]+=1
        for key in ('grounding_available','sft_available','detection_valid','moving','turning'):c[key]+=r[key]
        for i,name in enumerate(('pcol_1s','pcol_2s','pcol_3s','dmin','areq')):c['risk/'+name]+=r['risk_mask'][i]
        c['ego_future_any']+=any(r['ego_future_valid']);c['ego_future_points']+=sum(r['ego_future_valid'])
        c['agent_future_points']+=sum(sum(v) for v in r['agent_future_valid'].values())
        c['road_geometry_any']+=any(any(s['geometry']) for s in r['road']);c['road_width_any']+=any(s['width'] for s in r['road'])
        if r['mode_label']=='UNKNOWN':
            unknown['samples']+=1;unknown['detection_valid']+=r['detection_valid'];unknown['grounding_available']+=r['grounding_available'];unknown['sft_available']+=r['sft_available']
            for i,name in enumerate(('pcol_1s','pcol_2s','pcol_3s','dmin','areq')):unknown['risk/'+name]+=r['risk_mask'][i]
        for reason in r['mode_reasons']:attribution[r['origin']+'/'+reason]+=1
        if r['origin']=='recovered':recoveries[r['old_exclusion_reason']]+=1
        elif r['previous_mode'] != r['mode_label'].lower():changed.append(dict(sample_id=r['sample_id'],old=r['previous_mode'],new=r['mode_label'],reasons=r['mode_reasons']))
        rows+=1
    if rows!=migration['samples']:raise ValueError('Audit row count mismatch')
    write_json(out/'scenario_coverage.json',{scenario:{split:dict(samples=coverage_counts[scenario][split],episodes=sorted(coverage[scenario][split])) for split in ('train','val','test')} for scenario in sorted(coverage)})
    write_json(out/'comparison_report.json',dict(by_origin=dict(origin),unknown_tasks=unknown,mode_attribution=attribution,recovered_old_reasons=recoveries,baseline_mode_changes=changed))
    before_cfg=json.loads(Path(prov['source']).with_name('provenance.json').read_text())['config']
    old_cfg=json.loads(Path(before_cfg).read_text());new_cfg=json.loads(Path(prov['config']).read_text())
    diffs=[]
    def compare(a,b,prefix=''):
        for k in sorted(set(a)|set(b)):
            key=prefix+'.'+k if prefix else k
            if isinstance(a.get(k),dict) and isinstance(b.get(k),dict):compare(a[k],b[k],key)
            elif a.get(k)!=b.get(k):diffs.append(dict(field=key,old=a.get(k),new=b.get(k)))
    compare(old_cfg,new_cfg);write_json(out/'config_diff.json',diffs)
    totals=migration['totals'];baseline=origin['baseline'];recovered=origin['recovered']
    lines=['# Task 3：Local Mode 与按任务保留协议验收报告','',
        '已完成独立数据发布、候选回收、Mode/任务 mask 迁移及端到端数据加载验收。未训练 Grounding/SFT，未修改模型结构、自由生成语法或损失权重，未覆盖旧数据/权重。','',
        '## 发布结果','',
        f'- 原始候选 {migration["input_candidates"]}；基线 {migration["baseline_samples"]}；核查新增候选 {migration["additional_candidates"]}。',
        f'- 保留输入完整记录 **{migration["samples"]}**；回收 **{migration["recovered"]}**；硬性排除 **{migration["hard_excluded"]}**。',
        f'- **{migration["episodes"]} 个 episode**，split 完全沿用 Task 2 的 episode 映射。',
        f'- Grounding 可用 **{totals["grounding_available"]}**；SFT 可用 **{totals["sft_available"]}**。输入完整与阶段可用分别计数。',
        '- Manifest：`manifest.jsonl`；配置：`configs/l40s_qwen25_3b_taskwise_local_v1.json`；协议：`taskwise_local_v1`。',
        '- Agent/Risk schema 保持 `agent50_v1` / `local_agent_collision_v1`。','',
        '## 修改前后与 Mode','',
        '| 来源 | 帧数 | Detection有效 | SHORT | LONG | UNKNOWN | Grounding可用 | SFT可用 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for title,key in (('原 Task2 基线（迁移后）','baseline'),('新回收帧','recovered')):
        c=origin[key];fields=('samples','detection_valid','mode/SHORT','mode/LONG','mode/UNKNOWN','grounding_available','sft_available')
        lines.append('| '+title+' | '+' | '.join(str(c[k]) for k in fields)+' |')
    lines += ['','基线 Mode 发生变化的帧数：'+str(len(changed))+'。逐帧变化在 comparison_report.json 和 samples.jsonl 中列出。','',
        'Mode 规则：有效 pcol_1s/2s 为正、dmin≤0.5m 或 areq≥3m/s² 任一项即可确认 SHORT；非空集合须四项条件均有效且未触发才为 LONG。按用户确认保留已知空集合 LONG 例外。其余为 UNKNOWN。','',
        f'UNKNOWN 共 {unknown["samples"]} 帧，其中 Detection 有效 {unknown["detection_valid"]}，Grounding 可用 {unknown["grounding_available"]}，SFT 可用 {unknown["sft_available"]}。UNKNOWN 不改为 LONG，也不新增推理模式；整帧仅从 SFT 排除。','',
        '### 逐 split 分布','',
        '| split | 帧数 | SHORT | LONG | UNKNOWN | Grounding | SFT | Ego速度≥1m/s | 转弯≥0.05rad/s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sp in ('train','val','test'):
        c=migration['groups']['split'][sp];fields=('samples','mode/SHORT','mode/LONG','mode/UNKNOWN','grounding_available','sft_available','moving','turning')
        lines.append('| '+sp+' | '+' | '.join(str(c.get(k,0)) for k in fields)+' |')
    lines += ['','### 场景覆盖','', '| scenario | train帧/episode | val帧/episode | test帧/episode |','|---|---:|---:|---:|']
    for scenario in sorted(coverage):
        lines.append('| '+scenario+' | '+' | '.join(f'{coverage_counts[scenario][sp]}/{len(coverage[scenario][sp])}' for sp in ('train','val','test'))+' |')
    lines += ['','各 split 的零 episode 即覆盖缺口；回收没有增加独立 episode，也未改变其 split。详见 scenario_coverage.json。','','### 归因','']
    for reason,count in sorted(recoveries.items()):lines.append(f'- 回收 {count} 帧，旧原因：{reason}。')
    for reason,count in migration['hard_exclusion_reasons'].items():lines.append(f'- 硬性排除 {count} 帧：{reason}。')
    for origin_name,c in mode_audit['by_origin'].items():
        lines.append(f'- {origin_name}：LONG 且 Global 3s 危险 {c.get("LONG/global_3s_danger",0)} 帧，LONG 且 Global 未知 {c.get("LONG/global_3s_unknown",0)} 帧；前者含空间域外来源 {c.get("LONG/global_danger_outside_spatial_domain",0)}，含忽略类别来源 {c.get("LONG/global_danger_ignored_class",0)}。')
    lines += ['','LONG 的增加按新增样本、Local 风险证据及已确认空集合单列；不能解释成总体驾驶风险降低。旧转换因 Global 风险/教师归因排除的帧，可能在 Local 定义下可确认为 LONG，也可能证据不足而为 UNKNOWN。这里没有把域外危险直接当作局部安全。','',
        '## 任务级监督覆盖','',
        '| 来源 | pcol1有效 | pcol2有效 | pcol3有效 | dmin有效 | areq有效 | Ego Future任一点有效 | Agent Future有效点 | Road几何有效帧 | Road宽度有效帧 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for title,key in (('基线','baseline'),('回收','recovered')):
        c=origin[key];fields=('risk/pcol_1s','risk/pcol_2s','risk/pcol_3s','risk/dmin','risk/areq','ego_future_any','agent_future_points','road_geometry_any','road_width_any')
        lines.append('| '+title+' | '+' | '.join(str(c[k]) for k in fields)+' |')
    lines += ['','每个 split/scenario/episode 的有效数量、正负分布及静止/运动 Agent 数见 migration_report.json；未知帧各 Risk 分量的有效计数见 comparison_report.json。速度 1m/s、绝对转率 0.05rad/s 仅用于统计，没有用于筛选。','',
        '### Agent 集合与上下文','',
        f'- 合法 Local Agent 观测数 {totals["local_agent_ids"]}；Risk 使用 {totals["risk_agent_ids"]}；Planner selected {totals["planner_selected_ids"]}；SFT 描述 {totals["sft_described_ids"]}。这些是跨帧观测数量。',
        f'- 有危险来源未选入 Planner 的帧 {totals["unselected_hazard_sources"]}；其中 SFT 因上下文不一致排除 {totals.get("sft_excluded/local_hazard_outside_planner_selected",0)} 帧。',
        '- 某个来源未选入与完全无法归因不同：只要仍有符合原规则的 selected 关键目标，继续使用原文本选择；完全无法归因时 Mode 事实仍是 SHORT，SFT 可用性另行置否。',
        '- 四套集合及顺序逐帧记录。Risk 使用全部合法 Agent；Planner 最多 8，SHORT 继续最多描述 2 个关键目标。','',
        '### 损失与缺失字段','',
        '- Detection/Risk/Agent Future 独立使用有效 mask；无效场景不作为空场景背景。运动派生项仅用共同有效的相邻时间点。',
        '- 已知 Mode 的 SFT 保留有效任务。道路节未知时屏蔽该节数值与结构标签，不监督为没有道路；未知宽度不清空有效几何。',
        '- 轨迹不能合法拟合完整控制点时，控制点 CE 不用外推 GT；metric loss 只用真实有效时间点。',
        '- 超词表 scalar 仅屏蔽其 CE，不裁剪真实未来；保留合法 metric loss。全量 metadata 教师审计的非径向超界字段数：'+str(accept['teacher_audit']['masked_out_of_range_nonradial_scalars'])+'。测量相关径向字段另由真实样本验证。',
        '- 平滑与一致性按任务/观测区间门控；全 mask 时有限零 loss，可反传且无无效梯度。','',
        '## 数据不变量、兼容性及验证','',
        '- 8,835 个基线样本的检测标注、Local/Global Risk、传感器引用、道路结构及 Agent/Ego Future 逐字段保持；split 不变。',
        '- 新增缓存只提取保存产品，沿用现有坐标约定；逐份核对数组数值及 SHA256，Power/Doppler/mask 不裁剪，不重算雷达。',
        '- 原 Actor 来源、静止/运动处理、BBox/FOV 规则及行人/cyclist 忽略策略保持。采集未记录的目标不补造。',
        '- 配置差异：'+', '.join(d['field'] for d in diffs)+'。旧协议继续显式分支；混合数据/权重协议拒绝加载，不修改旧 checkpoint protocol。',
        '- 专项回归：`'+log.strip().splitlines()[-1]+'`。',
        f'- 全量验收：{accept["samples"]} 帧、{accept["episodes"]} episode；{accept["sensor_files_existing"]} 个传感器输入路径存在，{accept["new_cache_hashes_verified"]} 份新增缓存哈希通过。',
        '- 每帧教师语法、Mode 和描述 ID/顺序均核对；metadata 审计不伪造径向测量。',
        '- 真实代表帧：'+', '.join(r['case'] for r in accept['representatives'])+'。',
        '- 未在当前真实数据出现、由合成专项测试覆盖的边界：'+(', '.join(accept['missing_real_cases']) or '无')+'。',
        f'- 代表帧通过 canonical Dataset 和共享语法/loss 的 tiny Planner 做端到端前向、反向和混合 batch 检查；参数最大变化 {accept["parameter_max_change"]}，optimizer steps=0。未加载 Qwen 训练。',
        '- test 仅核验数据和监督定义，未用于调参。','',
        '## 文件与版本追溯','',
        '- manifest.jsonl：发布数据；samples.jsonl：逐样本任务/Mode/集合/运动统计；hard_exclusions.jsonl：硬排除清单。',
        '- migration_report.json、comparison_report.json、teacher_validation.jsonl、acceptance.json：对照与验收。',
        '- raw_evidence.json、radar_cache_hashes.json、provenance.json：输入、原始证据、缓存和执行源码摘要。',
        '- mode_attribution.jsonl / mode_attribution_report.json：Global/Local、mask、场景与原始危险来源归因。',
        '- source_snapshot/、source_snapshot_hashes.json：执行实现与测试源码快照，可在既有项目中还原；不含数据或模型权重。',
        '- implementation_provenance.json、test_results.log、config_diff.json：最终交付实现、测试和配置差异。',
        '- 协议说明：docs/TASKWISE_LOCAL_PROTOCOL.md。','',
        '后续 Grounding/SFT 必须使用本目录 manifest、configs/l40s_qwen25_3b_taskwise_local_v1.json 和 taskwise_local_v1 协议；先重训 Grounding，再由匹配的新版 Grounding 初始化 SFT。本轮没有启动这两项训练。','']
    changed_files=['src/collection_adapter.py','src/data.py','src/planner.py','src/pipeline.py','src/visual_diagnosis.py','src/local_risk.py','src/task_supervision.py','src/taskwise_migration.py',
        'configs/l40s_qwen25_3b_taskwise_local_v1.json','scripts/migrate_taskwise.py','scripts/verify_taskwise.py','scripts/report_taskwise.py','scripts/audit_taskwise_modes.py','tests/test_taskwise_supervision.py','docs/TASKWISE_LOCAL_PROTOCOL.md']
    lines += ['## 修改文件','']+['- '+p for p in changed_files]+['','Grounding losses.py 的既有逐维/逐时刻 mask 机制经专项测试验证后保留；Planner 的任务门控与共享 Mode 判定由本次更新接入。','']
    (out/'REPORT.md').write_text('\n'.join(lines));write_json(out/'modified_files.json',changed_files)
    paths=[Path(p) for p in changed_files];artifacts=('manifest.jsonl','samples.jsonl','hard_exclusions.jsonl','migration_report.json','comparison_report.json','teacher_validation.jsonl','acceptance.json','raw_evidence.json','radar_cache_hashes.json','provenance.json','test_results.log','config_diff.json','REPORT.md','mode_attribution.jsonl','mode_attribution_report.json','scenario_coverage.json','source_snapshot_hashes.json')
    project=Path.cwd().resolve()
    snapshot_files={Path(path) for path in prov['source_files']}
    snapshot_files.update(paths)
    snapshot_files.update(Path('collection/collector').glob('*.py'))
    snapshot_files.update(Path('tests').glob('*.py'))
    snapshot_files.update(Path(name) for name in ('pyproject.toml','requirements.txt','environment.yml','AGENTS.md'))
    snapshot_hashes={}
    for path in snapshot_files:
        path=path.resolve();relative=path.relative_to(project);destination=out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and sha256(destination)!=sha256(path):raise ValueError('Snapshot source changed')
        if not destination.exists():shutil.copy2(path,destination)
        snapshot_hashes[str(relative)]=sha256(destination)
    write_json(out/'source_snapshot_hashes.json',snapshot_hashes)
    write_json(out/'implementation_provenance.json',dict(schema='taskwise_local_delivery_v1',source_files={str(p.resolve()):sha256(p) for p in paths},artifacts={n:sha256(out/n) for n in artifacts},
        training_started=False,optimizer_steps=0,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print('REPORT PASS',flush=True)

if __name__=='__main__':main()
