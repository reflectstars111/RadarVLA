"""Aggregate auditable artifacts into a Chinese diagnosis and next-run specification."""
from __future__ import annotations
import argparse
import math
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from radar_vla.pipeline import file_digest
from radar_vla import collection_adapter as adapter
from radar_vla.annotation_domain import box_intersects_annotation_domain


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mean(v):return float(np.mean(v)) if v else None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();out=args.output_dir
    inventory=read(out/'train_val_inventory.json');by_id={r['sample_id']:r for r in inventory}
    integrity=read(out/'integrity.json');raw=read(out/'sample_audit.json');prov=read(out/'provenance.json')
    if {r['sample_id'] for r in raw}!={r['sample_id'] for r in read(out/'audit_selection.json')}:
        raise ValueError('raw audit incomplete or cohort differs')
    flags=['frame_id_match','visible_ids_match','roads_match','ego_future_masks_match','history_frame_ids_match','risk_label_match','dataset_domain_known','raw_doppler_mask_match','radar_product_frame_match']
    scalar=['time_delta_s','ego_future_error_m','pose_error','history_time_error_s','corrected_sensor_velocity_error_mps','dataset_power_error','dataset_unmasked_doppler_error','dataset_velocity_error','dataset_future_error','raw_power_error','raw_doppler_error']
    raw_summary=dict(samples=len(raw),booleans={k:dict(checked=sum(k in r for r in raw),failed=sum(not r[k] for r in raw if k in r)) for k in flags},
        maximum_errors={k:max(r[k] for r in raw if k in r) for k in scalar},
        agent_maximum_errors={k:max(r['agent_errors'].get(k,0) for r in raw) for k in ['position','velocity','size','heading','future_xy','future_valid','future30_xy','future30_valid']},
        measured_radial_residual_mps=dict(samples=sum(len(r['measured_radial_residual_mps']) for r in raw),mean=mean([v for r in raw for v in r['measured_radial_residual_mps']])))
    axis_checks=[]
    with Path(integrity['manifest']).open('rb') as manifest_stream:
        for r in raw:
            if 'raw_radar_product' not in r:continue
            manifest_stream.seek(by_id[r['sample_id']]['offset']);record=json.loads(manifest_stream.readline())
            axes_path=Path(r['raw_radar_product'])/'radar_axes.npz'
            with np.load(axes_path) as axes:
                axis_checks.append(dict(sample_id=r['sample_id'],source=str(axes_path),
                    range_max_error_m=float(np.max(np.abs(axes['range_m']-record['sensors']['radar']['range_m']))),
                    azimuth_max_error_rad=float(np.max(np.abs(np.deg2rad(-axes['azimuth_deg'][::-1])-record['sensors']['radar']['azimuth_rad'])))))
    write(out/'radar_axis_checks.json',axis_checks)
    raw_summary['axis_samples']=len(axis_checks)
    raw_summary['range_axis_max_error_m']=max(r['range_max_error_m'] for r in axis_checks)
    raw_summary['azimuth_axis_max_error_rad']=max(r['azimuth_max_error_rad'] for r in axis_checks)
    mapping=read(out/'raw_type_mapping.json')
    bicycles=[r for r in mapping if any(s in r['type_id'] for s in ('bh.crossbike','gazelle.omafiets','diamondback.century'))]
    exclusions=read(out/'excluded_hazards.json')
    episode_paths={r['scene_id']:Path(r['route_archive']).parent for r in inventory}
    source=Path(read('analysis/rebalanced_v4/split_report.json')['source'])
    expected_excluded=[json.loads(line) for line in (source.parent/'exclusions.jsonl').open()]
    expected_excluded=[r for r in expected_excluded if 'cannot be attributed' in r['reason'] and r['job_id'] in episode_paths]
    if {(r['job_id'],r['carla_frame']) for r in exclusions}!={(r['job_id'],r['carla_frame']) for r in expected_excluded}:
        raise ValueError('excluded hazard audit incomplete or cohort differs')
    # Backfill geometry for a run already in progress when the audit was extended.
    for ex in exclusions:
        if not any('intersects_radar_bev' not in h for h in ex['hazards']):continue
        episode=episode_paths[ex['job_id']]
        frame=read(episode/'frames'/f"{ex['source_index']:06d}"/'frame.json')
        metadata=read(episode/'metadata.json');origin=frame['ego']['pose']['position_m'][:2];yaw=frame['ego']['pose']['rotation_deg'][2]
        actors={str(a['actor_id']):a for a in frame['actors']}
        pose=adapter._radar_pose(frame,origin,yaw,metadata['sensors']['semantic_lidar']['x'])
        for h in ex['hazards']:
            a=adapter._agent(actors[h['actor_id']],frame,origin,yaw,None,[4,9,14,19,24,29])
            h['intersects_radar_bev']=box_intersects_annotation_domain(a['position'],a['size'],a['heading'],
                [metadata['radar_range_m'],-math.radians(53),math.radians(53)],pose)
    write(out/'excluded_hazards.json',exclusions)
    ex_summary=dict(audited=len(exclusions),
        with_outside_visible_hazard=sum(any(not a['visible'] for a in r['hazards']) for r in exclusions),
        with_visible_but_not_selected_hazard=sum(any(a['visible'] and not a['planner_selected'] for a in r['hazards']) for r in exclusions),
        with_selected_hazard=sum(any(a['planner_selected'] for a in r['hazards']) for r in exclusions),
        with_hazard_outside_radar_bev=sum(any(not a['intersects_radar_bev'] for a in r['hazards']) for r in exclusions),
        with_hazard_outside_labels_but_in_radar_bev=sum(any(not a['visible'] and a['intersects_radar_bev'] for a in r['hazards']) for r in exclusions),
        no_individual_hazard_reproduced=sum(not r['hazards'] for r in exclusions),
        with_incomplete_future=sum(r['actors_with_incomplete_future']>0 for r in exclusions),
        note='Rows can have multiple causes; counts overlap. Test excluded from cause analysis.')
    # Existing oracle experiment is a deliberately enriched diagnostic cohort, not full validation.
    rp=out/'risk_modes_fresh.json';risk_run=read(rp);risk_rows=risk_run['rows']
    risk_summary={}
    for source in ('predicted','oracle'):
        groups=defaultdict(Counter)
        for r in risk_rows:
            if r['sample_id'] not in by_id or by_id[r['sample_id']]['split']!='val':raise ValueError('risk cohort is not current validation')
            truth=r['teacher_mode'];pred=r['mode'][source]['mode'];meta=by_id[r['sample_id']]
            keys=['overall','episode:'+r['episode_id'],'scenario:'+meta['category'],
                'risk_masks:'+('complete' if all(r['risk_mask']) else 'partial'),
                'speed:'+('moving' if meta['speed_mps']>=1 else 'slow'),'turn:'+('turning' if abs(meta['yaw_rate_radps'])>=.05 else 'straight')]
            for key in keys:groups[key][truth+'->'+pred]+=1
        risk_summary[source]={}
        for key,c in groups.items():
            tp=c['short->short'];fp=c['long->short'];fn=c['short->long'];tn=c['long->long']
            risk_summary[source][key]=dict(tp_short=tp,fp_short=fp,fn_short=fn,tn_long=tn,short_recall=tp/(tp+fn) if tp+fn else None,short_precision=tp/(tp+fp) if tp+fp else None)
    write(out/'risk_oracle_review.json',dict(source=str(rp),source_sha256=file_digest(rp),rows=risk_rows,groups=risk_summary,
        partial_label_rows=sum(not all(r['risk_mask']) for r in risk_rows),
        checkpoint_sha256=risk_run['checkpoint_sha256'],manifest_sha256=risk_run['manifest_sha256'],
        note='Re-run on historical 73-frame cohort, with fresh checkpoint/source hashes; oracle substitutes known dimensions only. Partial risk labels separately stratified; oracle not available at deployment.'))
    traj=read(out/'trajectory_decomposition.json');old=read('analysis/rebalanced_validation/sft_trajectory_decomposition_epoch_003.json')
    if {r['sample_id'] for r in traj['rows']}!={r['sample_id'] for r in old['rows']}:raise ValueError('trajectory cohort changed')
    methods=['constant_velocity','codec','teacher_forced','free_generation'];paired=defaultdict(lambda:defaultdict(list));coverage=defaultdict(lambda:Counter())
    denominators={m:dict(samples=len(traj['rows']),
        scored_points=sum(len(r['scores'][m].get('point_errors_m',[])) for r in traj['rows']),
        fde3_samples=sum(r['scores'][m].get('fde_m') is not None for r in traj['rows']),
        velocity_samples=sum(r['scores'][m].get('initial_velocity_error_mps') is not None for r in traj['rows'])) for m in methods}
    for r in traj['rows']:
        maps={m:dict(zip(r['scores'][m].get('scored_times_s',[]),r['scores'][m].get('point_errors_m',[]))) for m in methods}
        common=set.intersection(*(set(x) for x in maps.values()))
        meta=by_id[r['sample_id']];keys=['overall','episode:'+r['episode_id'],'scenario:'+meta['category'],
            'speed:'+('moving' if meta['speed_mps']>=1 else 'slow'),'turn:'+('turning' if abs(meta['yaw_rate_radps'])>=.05 else 'straight')]
        for key in keys:
            coverage[key]['samples']+=1;coverage[key]['common_points']+=len(common);coverage[key]['full_3s_samples']+=3. in common
            for m in methods:
                paired[key][m+'_common_errors'].extend(maps[m][t] for t in sorted(common))
                if 3. in common:paired[key][m+'_paired_fde3'].append(maps[m][3.])
    paired_summary={key:dict(**dict(coverage[key]),metrics={m:dict(paired_ade_m=mean(v[m+'_common_errors']),paired_fde3_m=mean(v[m+'_paired_fde3'])) for m in methods}) for key,v in paired.items()}
    write(out/'trajectory_paired_summary.json',dict(groups=paired_summary,all_available=traj['summary'],
        all_available_denominators=denominators,
        note='All four methods share identical sample/time points within each group. FDE3 uses only samples with 3s output for all four. Teacher SHORT plans also limit codec/teacher-forced coverage.'))
    detections=[]
    for p in sorted((out/'detection').glob('*sweep.json')):
        d=read(p)
        for threshold,v in d['thresholds'].items():
            a=v['overall'];n=a['samples']
            detections.append(dict(model=p.stem,threshold=float(threshold),samples=n,
                precision=a['precision'],recall=a['recall'],f1=a['f1'],count_mae=a['count_mae'],
                raw_per_frame=a['raw_predictions']/n,evaluated_per_frame=a['evaluated_predictions']/n,
                ignored_per_frame=a['ignored_outside_annotation_domain']/n,fn=a['false_negatives']))
    if len(detections)!=20:raise ValueError('expected four checkpoints times five thresholds')
    write(out/'detection_summary.json',detections)
    epochs=[]
    for stage in ('grounding','sft'):
        for line in Path(f'analysis/retrained_{stage}_v2/metrics.jsonl').open():
            d=json.loads(line);v=d['validation'];epochs.append(dict(stage=stage,epoch=d['epoch'],train_loss=d['train_loss'],validation_loss=v['loss'],ap3=v['collision']['3s']['average_precision'],brier3=v['collision']['3s']['brier']))
    write(out/'training_curves.json',epochs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for stage in ('grounding','sft'):
        values=[r for r in epochs if r['stage']==stage]
        axes[0,0].plot([r['epoch'] for r in values],[r['validation_loss'] for r in values],marker='o',label=stage)
        axes[0,1].plot([r['epoch'] for r in values],[r['ap3'] for r in values],marker='o',label=stage)
    axes[0,0].set(title='Historical validation total loss (stage-specific scale)',xlabel='Epoch',ylabel='Loss')
    axes[0,1].set(title='Historical 3s risk average precision',xlabel='Epoch',ylabel='AP')
    axes[0,1].axhline(56/1337,color='gray',linestyle=':',label='val positive fraction')
    for model in sorted({r['model'] for r in detections}):
        values=sorted([r for r in detections if r['model']==model and r['precision'] is not None and r['recall'] is not None],key=lambda r:r['threshold'])
        axes[1,0].plot([r['recall'] for r in values],[r['precision'] for r in values],marker='o',label=model.replace('retrained_','').replace('_sweep',''))
    axes[1,0].set(title='96-frame domain-aware detection threshold scan',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
    scores=paired_summary['overall']['metrics']
    axes[1,1].bar(['CV','Codec','Teacher forced','Free'],[scores[m]['paired_ade_m'] for m in methods],color=['gray','#4daf4a','#377eb8','#e41a1c'])
    axes[1,1].set(title=f"Trajectory ADE: {paired_summary['overall']['common_points']} identical sample/time points",ylabel='ADE (m)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in axes.flat[:3]:ax.legend(fontsize=8)
    fig.savefig(out/'diagnosis_overview.png',dpi=150)
    plt.close(fig)
    experiment_defaults=dict(seed=42,epochs=5,lr=.0003,effective_batch_size=12,optimizer='AdamW',weight_decay=.0001,
        split='preserve current episode assignment',threshold_selection='validation only; 0.5/0.7/0.9/0.95/0.99',
        checkpoints='save every epoch plus task-specific best; no test during selection',
        generation_validation='fixed 96 validation frames; teacher-required coverage and same-time CV; full val only for final candidates')
    experiments=[
        dict(id='G0',parent=None,change='current corrected detection focal normalization + explicit domain + raw-corrected sensor velocity; old planner architecture fields disabled',stage='grounding',planner_ego_context=False),
        dict(id='G1',parent='G0',change='only collision focal alpha: positive .75 / negative .25 instead of .25/.75; do not combine with resampling',stage='grounding',requires_implementation='expose collision alpha in config and provenance'),
        dict(id='S0',parent='G0/G1 validation Pareto candidate',change='freeze entire Grounding in eval mode, train planner/LoRA; ego context off; initial velocity and mode weights zero',stage='sft',requires_implementation='explicit freeze option, excluded from optimizer and provenance checked'),
        dict(id='S1',parent='same Grounding as S0',change='joint training; all other settings identical to S0',stage='sft'),
        dict(id='S2',parent='S0',change='only enable planner_ego_context; reuse identical Grounding state tensors',stage='sft',
             requires_implementation='initializer currently requires identical ModelConfig; add explicit context-only transfer option with source checkpoint hash/config diff recorded, never silently rewrite checkpoint protocol',
             note='Grounding ego encoder is already risk-supervised; flag adds only its token to planner; no extra Grounding training budget'),
        dict(id='S3',parent='S2',change='only ego_initial_velocity_weight=1.0',stage='sft'),
        dict(id='S4',parent='best preceding controlled candidate',change='only mode_loss_weight=1.0 if SHORT errors persist',stage='sft',conditional=True)]
    write(out/'next_experiments.json',dict(defaults=experiment_defaults,experiments=experiments,status='SPECIFICATION ONLY; no training started',
        data_gate='First fix category mapping and rebuild state fields from raw with unchanged split. Do not simply toggle flags on old checkpoints.',
        selection='G1 only preferred if AP/calibration improve without detection F1 regression; retain separate detection/risk Pareto checkpoints. SFT compare required coverage, mode confusion, moving ADE/CV, paired FDE, initial velocity, and risk/detection regression.'))
    result=dict(integrity=integrity['summary'],raw=raw_summary,bicycle_mapping=bicycles,excluded=ex_summary,
        risk={k:v['overall'] for k,v in risk_summary.items()},trajectory=paired_summary['overall'],
        source_provenance=str(out/'provenance.json'),test_usage='integrity only; no test metrics, thresholds or weights selected')
    write(out/'summary.json',result)
    lines=['# RadarVLA 当前 pipeline 诊断（2026-10-16）','',
        '## 优先修改的三项','',
        '1. **先修数据字段与类别映射，再重训 Grounding。** 当前 50 m manifest 仍保留旧传感器速度；转弯帧需要从原始 ego 运动和安装位姿重新生成速度字段。原始自行车蓝图被统一标成 vehicle（见下方实测数量）。保留 split 和 Power/Doppler 数组。',
        '2. **Grounding 使用已修复的检测损失与显式监督域，风险权重另做单项对照。** 旧权重未训练这些修复；不要继续增加旧实验 epoch。风险 AP 下降已证实，其成因尚不能单独归为梯度冲突。',
        '3. **随后重训 SFT，先比较冻结与联合训练，再分别增加 ego 状态和初速约束。** 编解码误差较小，自由生成差距较大；任务级选权重替代单看总 loss，SHORT 必须同时看召回、误判和覆盖。','',
        '本轮只执行源码/原始数据审计和旧权重推理，没有训练、覆盖权重或按 test 选择参数。','',
        '**这轮最重要的新判断：** 抽查未发现普遍的时间、坐标或轨迹转换错误；自行车类别与旧传感器速度是已确认的数据字段问题。被排除危险帧的归因结果没有支持“最近8个目标截断是主要原因”的怀疑。检测损失、模式/风险条件和模型选择仍需通过分开的新训练验证。','',
        '![训练退化、检测阈值与轨迹分解](diagnosis_overview.png)','',
        '## 证据版本与检查范围','',
        f"- Git HEAD：`{prov['git_head']}`；工作区源码摘要见 `provenance.json`，不能把 HEAD 当成全部实际源码。",
        f"- Manifest SHA256：`{integrity['manifest_sha256']}`；{integrity['samples']} 帧/{len(integrity['episodes'])} episodes。",
        f"- 所有 {len(integrity['episodes'])} episode metadata 确认 Agent/道路 50 m，雷达约 118.04 m。引用文件 {integrity['unique_referenced_files']} 个，缺失 {len(integrity['missing'])} 个，隔离/ID 错误 {len(integrity['errors'])} 个。",
        f"- 原始数值检查 {len(raw)} 帧，检测推理 {integrity['inference_samples']} 帧（10 个 val episode，5 个分散帧＋全部碰撞正例），轨迹复用固定 {len(traj['rows'])} 帧。危险富集子集不代表总体分布。",
        '- v2 protocol 中 loss/model/data/planner 等源码摘要与当前源码不同；新增修复效果需要新训练验证。','',
        '- 原始审计执行快照和归因简化的执行记录见 `execution_notes.json` / `audit_executed_snapshot.py`；脚本运行期间的后续补充没有冒充为原始执行版本。','',
        '## 当前监督链路怎样影响诊断','',
        '采集器保存的是路线连接的 ego 道路走廊（16个采样点），不是视域内所有道路；adapter再拆为road segments。Agent标注为50m视域内目标，风险则可引用完整world snapshot中的risk_agents，因此两者覆盖并不相同。',
        'Grounding监督检测存在性、类别、框/位置/速度/目标未来与五维风险；它没有独立的道路输出头。SFT从压缩雷达token、风险和指令生成模式、道路、目标描述及ego轨迹；当前新增的ego token不在旧权重配置中。',
        'SFT阶段仍把Grounding loss与token/物理loss相加，优化器包含两部分可训练参数，风险条件也没有自动detach。检测框列表并未直接成为规划器逐目标条件。因此“检测不错”不能自动推出“SFT生成目标/道路也正确”，冻结对照需要明确冻结整个上游并保持eval模式。',
        '这些定位分别对应 `src/collection_adapter.py`、`src/model.py`、`src/planner.py` 与 `src/pipeline.py::_loss`。','',
        '## 数据与转换审计','',
        '| Split | 帧 / episode | 低于1m/s | SHORT / 危险episode | 类别实例 | 速度差>0.1m/s |',
        '|---|---:|---:|---:|---|---:|']
    for split,s in integrity['summary'].items():lines.append(f"| {split} | {s['frames']} / {s['episodes']} | {s['slow_lt1']} | {s['modes'].get('short',0)} / {s['positive_episodes']} | {s['classes']} | {s['velocity_difference_gt_01']} |")
    failed={k:v for k,v in raw_summary['booleans'].items() if v['failed']}
    lines += ['',f"{len(raw)}帧的帧ID、可见目标集合、道路、未来掩码、风险重算与域字段检查，失败项：`{failed}`。Agent位置、速度、框及未来轨迹与原始转换的最大差异均为 {max(raw_summary['agent_maximum_errors'].values()):.3g}。",'',
        '| 数值检查 | 最大绝对差 |', '|---|---:|',
        *[f"| {k} | {raw_summary['maximum_errors'][k]:.6g} |" for k in ('time_delta_s','ego_future_error_m','pose_error','corrected_sensor_velocity_error_mps','dataset_future_error','raw_power_error','raw_doppler_error')], '',
        f"另外核对 {len(axis_checks)} 帧原始range/azimuth坐标轴，最大差分别为 {raw_summary['range_axis_max_error_m']:.3g}m / {raw_summary['azimuth_axis_max_error_rad']:.3g}rad；细节见 `radar_axis_checks.json`。",'',
        f"抽查发现自行车蓝图仍标 vehicle：`{bicycles}`。这是当前采集分类规则的确定问题；数量为抽查的目标观测次数，不是独立自行车数。应修 collector 和原始 type_id→类别转换，并在原数据上重建分类标签。",
        '','传感器速度检查区分当前帧全量统计与四帧历史抽查。它影响几何编码及基于状态的径向速度监督；不能据此认定雷达文件本身有错。超过0.1m/s的只有358/7503个train/val帧，不能用它单独解释全部误报。类别映射修正影响类别监督，但总检测P/R使用类别无关中心匹配，修类别也不会直接解决位置误报。',
        f"抽查实测 Doppler 与状态导出径向速度绝对残差均值 `{raw_summary['measured_radial_residual_mps']}`；该差值还包含关联/噪声，不能全归因于杆臂。",'',
        '## 被筛掉的危险样本','',
        '| 原始预处理排除原因（全池既有记录） | 帧数 |','|---|---:|',
        *[f"| {k} | {v} |" for k,v in read(out/'exclusion_counts.json').items()], '',
        '| train/val 危险归因检查 | 帧数 |','|---|---:|',
        *[f"| {k} | {v} |" for k,v in ex_summary.items() if isinstance(v,int)], '',
        f"实测 {ex_summary['audited']} 帧危险均有标签集合外来源，{ex_summary['with_hazard_outside_radar_bev']} 帧涉及118m雷达BEV外来源，{ex_summary['with_hazard_outside_labels_but_in_radar_bev']} 帧涉及雷达BEV内但标签集合外来源；可见但未进入最近8个的危险帧为 {ex_summary['with_visible_but_not_selected_hazard']}。这批证据不支持优先扩大max_agents。部分筛除有可观测性理由，不能把排除危险帧本身一概称为pipeline错误。",'',
        '若要回收这些帧，应先拆开全局风险与当前传感器可监督风险，保留可观测性/未知掩码，再分别决定Grounding和SFT是否可用；不可把域外危险改成安全负例，也不可把单向雷达不可观测事件无条件强加为预测目标。','',
        '逐 actor 重新计算反事实危险，保留其 visible/planner_selected 状态，并区分标注域外但仍在118m雷达BEV内、以及雷达BEV外的危险。BEV相交不保证真实可见，仍有垂向视域和遮挡限制。多个原因可同时发生；此分析不将缺失未来标签改为安全。对可见但不在最近8个中的危险，应改选取策略或解除其对 Grounding 样本保留的限制。当前 nearest-8 策略会让准备阶段和规划器容量共同决定数据分布。','',
        '## 检测：修正监督域后的旧权重诊断','',
        '缓存新增域与位姿字段，绑定 manifest/权重/评分源码摘要；采用旧配置 NMS 与 2m 匹配。域外仅忽略未匹配预测，与 GT 匹配的边界目标仍计 TP。局部域重新计分不能称为模型能力提升。','',
        '| 权重 | 阈值 | P | R | F1 | 原始/帧 | 域内计分/帧 | 忽略/帧 | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in detections:
        if d['threshold'] not in (.5,.9):continue
        lines.append(f"| {d['model']} | {d['threshold']} | {d['precision'] or 0:.3f} | {d['recall'] or 0:.3f} | {d['f1'] or 0:.3f} | {d['raw_per_frame']:.2f} | {d['evaluated_per_frame']:.2f} | {d['ignored_per_frame']:.2f} | {d['fn']} |")
    lines += ['', '完整五阈值、逐样本、逐 episode、场景、速度和转弯分组在 `detection/*_sweep.json`。这里的0.9是预先固定展示点，尚未据子集确定部署阈值。','',
        'Grounding在0.9阈值仅有9/321个原始预测被域外规则忽略；重新定义评分区域没有消除域内误报和漏检。需要修训练目标，不能将更换计分口径当成模型改善。','',
        '## 轨迹与风险','',
        f"共同时间点 ADE 使用 {paired_summary['overall']['samples']} 帧、{paired_summary['overall']['common_points']} 个时间点；3秒 FDE 使用 {paired_summary['overall']['full_3s_samples']} 帧。FDE不能外推至其余SHORT帧。",'',
        '| 方法 | 配对 ADE (m) | 配对3秒 FDE (m) |','|---|---:|---:|',
        *[f"| {m} | {paired_summary['overall']['metrics'][m]['paired_ade_m']:.3f} | {paired_summary['overall']['metrics'][m]['paired_fde3_m']:.3f} |" for m in methods], '',
        '教师SHORT同样会缩短编解码/正确前序输出，不能只检查自由生成的长度。各方法未配对均值和分母见 `trajectory_paired_summary.json`，计划坐标见 `trajectory_decomposition.json`。','',
        '本轮轨迹推理沿用权重 protocol 的 bfloat16 autocast；历史分解脚本未启用该 autocast，数值不应要求逐位相同。编解码、正确前序和自由生成的误差量级结论应以同一轮的配对结果为准。','',
        '风险替换重新推理使用73帧旧诊断子集（56 SHORT、17 LONG）：','',
        '| 风险条件 | SHORT选对 | SHORT漏判 | LONG误判成SHORT | SHORT召回 | SHORT精度 |','|---|---:|---:|---:|---:|---:|',
        *[f"| {source} | {risk_summary[source]['overall']['tp_short']} | {risk_summary[source]['overall']['fn_short']} | {risk_summary[source]['overall']['fp_short']} | {risk_summary[source]['overall']['short_recall']:.3f} | {risk_summary[source]['overall']['short_precision']:.3f} |" for source in ('predicted','oracle')], '',
        '真实风险替换只证明该条件对当前模式预测有影响，不能证明风险头是全部轨迹误差的原因，也不是可部署方案。逐 episode、运动和转弯结果见 `risk_oracle_review.json`。','',
        f"历史文件没有保存权重与执行代码摘要，所以本轮按同一73帧重新运行并补齐摘要。其中 {sum(not all(r['risk_mask']) for r in risk_rows)} 帧五维风险标签不完整，已单独分组；未知维度保留模型预测值，不能当成oracle真值。",'',
        '风险标签来自“自车维持当前速度/转率”的反事实几何 rollout；轨迹教师是实际驾驶未来轨迹。两者不同不必然是标签错误，例如教师已经刹车但反事实仍危险。当前 pcol 为几何事件标签，模型输出概率不等于已校准的真实事故概率。','',
        '## 已修正的诊断工具','',
        '- RA 标记从 current_ego 转到 current_radar；扇区与雷达原点同位姿，BEV 坐标保持 ego。非零平移/旋转用专项测试验证。',
        '- 完整显示保存的 unfolded Doppler，无可视化遮罩；模型 forward 仍按 radar_doppler_valid 将无效 Doppler 置零。视觉看到的无效区域并不是模型实际 Doppler 数值输入。',
        '- 单独缓存域内/域外计数、样本级 TP/FP/FN；旧不含域的扫描文件保留历史语义。','',
        '已保存真实转弯帧的完整叠加图：[turning_overlay.png](turning_overlay.png)；对应sample、安装位姿和权重摘要见 `turning_overlay.json`。该图验证坐标路径并展示漏检，不能代替定量匹配。','',
        '## 下一轮对照与训练选择','',
        '| 问题 | 修改位置 | 数据/训练影响 | 验收 |',
        '|---|---|---|---|',
        '| 旧传感器速度 | `src/collection_adapter.py::_radar_sensor_velocity`；重新生成 manifest 状态字段 | 当前函数已修，旧数据仍需更新；随后重训 Grounding 和 SFT | 与原始刚体速度一致，转弯组径向残差 |',
        '| 自行车类别统一为 vehicle | `collection/collector/recorder.py::_actor` 的 actor_type 映射；`src/collection_adapter.py::_agent` 的 type_id 映射 | 从原始 type_id 重建类别；两阶段重训 | 自行车映射与未知类掩码专项检查、类别覆盖统计 |',
        '| 检测负例归一化/域 | `src/losses.py::loss_grounding`、`src/annotation_domain.py` | 当前代码已修；Grounding 必须重训，SFT 从新版初始化 | 域内P/R/F1及原始输出数 |',
        '| 域外危险样本排除/可观测性 | `src/collection_adapter.py::_mode_from_risk` / `_attach_supervision` | 本轮未发现最近8截断导致的排除；回收数据前先明确风险监督域与阶段可用性，改变样本池另立实验 | 域内/域外危险来源及未知掩码 |',
        '| 风险退化 | `src/losses.py` collision focal alpha；`src/pipeline.py` 记录分量与任务选择 | G1单因素对照；不同时改采样 | AP/Brier及检测回归 |',
        '| SFT上游退化 | `src/pipeline.py::_loss` / optimizer / train-eval模式 | 新增冻结配置后做S0/S1，保持同一初始化 | 风险/检测保持、模式和覆盖 |',
        '| ego条件/初速监督 | `src/model.py` planner_ego_context；`src/planner.py::planner_physics_loss` | 当前实现已有；分开开关并重训SFT | 运动/转弯轨迹、初速及CV对照 |',
        '| 统一best选择失真 | `src/training_quality.py`、`src/pipeline.py` | 当前已有检测/规划选择；风险任务选择需补 | 不以缩短输出换低ADE；val选权重与阈值 |','',
        '具体参数见 `next_experiments.json`：种子42、5 epochs、学习率3e-4、有效batch12、AdamW、相同数据与更新预算。',
        '顺序为 G0 修正基线 → G1 仅风险 focal alpha 调整 → 同一个 Grounding 的 S0冻结/S1联合 → S2仅ego状态 → S3仅初速约束；模式CE另作条件实验。当前默认配置同时开启若干修复，直接整套训练可建新基线，却不能归因每项收益。',
        '冻结开关和风险alpha目前需要新增配置支持；S2仅追加ego token、不改变Grounding权重张量，但现有初始化检查要求ModelConfig完全一致，需要增加显式的context-only迁移选项并记录配置差异与源权重摘要。不能静默篡改旧checkpoint protocol。JSON是实验规格，不是假装现有CLI已经支持这些参数。新实验不得 strict resume 旧权重；固定的数据预处理修正也必须进入 provenance。',
        '检测按域内F1/数量误差选，风险按AP与Brier保留Pareto候选；SFT同时检查教师要求覆盖、模式混淆、运动样本相对CV误差、初速以及上游风险/检测退化。阈值先在完整validation确认，最后才评估test一次。','',
        '## 覆盖缺口与未决问题','',
        '- scenario4 的唯一实质 episode 留在 train，val/test 缺少该场景，不能据现有验证宣称覆盖。验证无行人/骑行者类别证据；先修类别再统计缺口。',
        '- 补采优先增加独立的运动、转弯、危险交互与行人/骑行者 episode；避免仅加静止或重复帧。',
        '- 风险退化可能涉及不均衡、过拟合、损失竞争和不可观测标签；本轮不宣称已识别梯度冲突，不宣称初速约束已改善自由生成。',
        '- 数据标签数值对齐和小样本推理不能替代完整验证及新训练。所有性能结论限定于报告给出的样本和分母。','',
        '## 复运行','',
        '仓库根目录运行 `bash scripts/run_current_diagnosis.sh analysis/current_pipeline_audit_new`。环境按 AGENTS.md；默认GPU0，可设 RADAR_VLA_AUDIT_DEVICE=cpu；可用 screen 启动。原始数据路径从现有manifest推导。输出目录必须独立。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    write(out/'summary_execution.json',dict(script_sha256=file_digest(Path(__file__)),
        final_source_sha256={str(p):file_digest(p) for folder in ('src','scripts') for p in Path(folder).glob('*.py')},
        note='Aggregation execution snapshot. Raw-audit runtime differences are documented separately in execution_notes.json.'))
    write(out/'artifact_digests.json',{str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_digests.json'})
    print('WROTE',out/'REPORT.md')

if __name__=='__main__':main()
