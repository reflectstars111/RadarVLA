"""Supplement native reports without loading radar arrays or changing models."""
from pathlib import Path
import datetime
import json
import time
import traceback
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from radar_vla.metrics import planning_metrics
from radar_vla.task_supervision import stage_available

ROOT=Path(__file__).resolve().parent


def write(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def supplemental(rows,records):
    samples=[];details=[];counts={str(h):dict(targets=0,detected=0,positive_frames=0,risk_below_0p5=0,missed_target_and_risk_below_0p5=0) for h in [.5,1.,2.]}
    overall_targets=overall_detected=0
    for row in rows:
        r=records[row['sample_id']];ego=r['ego'];times=np.asarray(r['future_times_s'],float)
        samples.append(dict(sample_id=row['sample_id'],future_times_s=times,ego_future=np.asarray(ego['future_xy'],np.float32),ego_future_mask=np.asarray(ego.get('future_valid',[True]*len(times)),bool)))
        agents=sorted(r['agents'],key=lambda a:(np.linalg.norm(a['position']),str(a['id'])))
        predictions=row.get('detections') or [];matched=set()
        if agents and predictions:
            xy=np.asarray([d['position_m'] for d in predictions]);gt=np.asarray([a['position'] for a in agents])
            distance=np.linalg.norm(xy[:,None]-gt[None],axis=-1)
            pi,gi=linear_sum_assignment(np.where(distance<=2.,distance,(len(agents)+len(predictions)+1)*3.))
            matched={str(agents[g]['id']) for p,g in zip(pi,gi) if distance[p,g]<=2.}
        overall_targets+=len(agents);overall_detected+=len(matched)
        labels=r.get('local_agent_risk_labels',{});frame=dict(sample_id=row['sample_id'],risk=row['risk'],horizons={})
        for i,h in enumerate([.5,1.,2.]):
            dangerous={str(a['id']) for a in agents if (lambda l:len(l.get('pcol',[]))==3 and l['pcol_mask'][i] and l['pcol'][i]>=.5)(labels.get(str(a['id']),{}))}
            detected=dangerous&matched;c=counts[str(h)];c['targets']+=len(dangerous);c['detected']+=len(detected)
            if dangerous:
                low=row['risk'][i]<.5;c['positive_frames']+=1;c['risk_below_0p5']+=int(low);c['missed_target_and_risk_below_0p5']+=int(bool(dangerous-detected) and low)
                frame['horizons'][str(h)]=dict(dangerous_ids=sorted(dangerous),detected_ids=sorted(detected),risk_below_0p5=low)
        if frame['horizons']:details.append(frame)
    for c in counts.values():c['recall']=c['detected']/c['targets'] if c['targets'] else None
    horizons={}
    for h in [.5,1.,2.,2.5,3.]:
        truth=[]
        for sample in samples:
            mask=sample['future_times_s']<=h+1e-6
            truth.append(dict(sample,sample_id=sample['sample_id'],future_times_s=sample['future_times_s'][mask],ego_future=sample['ego_future'][mask],ego_future_mask=sample['ego_future_mask'][mask]))
        horizons[str(h)]=planning_metrics(rows,truth)
    failures=Counter(str(row.get('plan',{}).get('failure_reason') or row.get('plan',{}).get('errors') or 'unspecified') for row in rows if not row.get('plan',{}).get('valid'))
    return dict(trajectory_horizons_s=horizons,hazard_detection=counts,total_targets=overall_targets,total_detected=overall_detected,invalid_plan_reasons=dict(failures),
                hazard_semantics='Retained agents with existing known per-agent cumulative conflict labels at0.5/1/2s; native2m maximum-cardinality Hungarian matching; unknown labels not safe; frame risk threshold0.5 is diagnostic only'),details


def paired_comparison(plan,truth):
    result={}
    for split in ['val','test']:
        matrices={};identifiers=None
        for job in [j for j in plan['jobs'] if j['split']==split]:
            rows=[json.loads(line) for line in (Path(job['output'])/'predictions.jsonl').read_text().splitlines()]
            by_id={row['sample_id']:row for row in rows}
            if identifiers is None:identifiers=sorted(by_id)
            assert set(by_id)==set(identifiers)
            errors=np.full((len(identifiers),6),np.nan)
            for i,identifier in enumerate(identifiers):
                row=by_id[identifier];r=truth[identifier];p=row.get('plan') or {}
                if not p.get('valid'):continue
                times=np.asarray(r['future_times_s'],float);gt=np.asarray(r['ego']['future_xy'],np.float32)
                assert len(times)==6
                valid=np.asarray(r['ego'].get('future_valid',[True]*6),bool)
                for point,t in zip(p.get('ego_trajectory',[]),p.get('ego_times_s',[])):
                    if point is None:continue
                    index=np.flatnonzero(np.isclose(times,t,atol=1e-5,rtol=0));point=np.asarray(point,float)
                    if len(index)==1 and point.shape==(2,) and np.isfinite(point).all() and valid[index[0]]:
                        errors[i,index[0]]=np.linalg.norm(point-gt[index[0]])
            matrices[job['arm']]=errors
        common=np.logical_and.reduce([np.isfinite(x) for x in matrices.values()]);danger=np.array([any(l.get('pcol_mask',[False]*3)[2] and l.get('pcol',[0]*3)[2]>=.5 for l in truth[i]['local_agent_risk_labels'].values()) for i in identifiers])
        stats={}
        for arm,errors in matrices.items():
            hazardous=common & danger[:,None]
            stats[arm]=dict(ade_m=float(errors[common].mean()) if common.any() else None,
                fde_3s_m=float(errors[common[:,-1],-1].mean()) if common[:,-1].any() else None,
                dangerous_scene_ade_m=float(errors[hazardous].mean()) if hazardous.any() else None,
                dangerous_scene_fde_3s_m=float(errors[common[:,-1]&danger,-1].mean()) if (common[:,-1]&danger).any() else None)
        result[split]=dict(samples=len(identifiers),common_points=int(common.sum()),common_final_samples=int(common[:,-1].sum()),dangerous_frames=int(danger.sum()),common_dangerous_points=int((common&danger[:,None]).sum()),
            semantics='Identical GT point intersection covered by valid plans of all six frozen models; excluded points remain excluded for every model; not a replacement for all-sample coverage',arms=stats)
    return result


def main():
    plan=json.loads((ROOT/'plan.json').read_text());truth={};reports={};processed={}
    while True:
        state=json.loads((ROOT/'STATUS.json').read_text())
        ready=[j for j in plan['jobs'] if state['jobs'][j['name']]['state']=='complete' and j['name'] not in processed]
        if ready and not truth:
            with Path(plan['training']['manifest']).open() as f:
                for line in f:
                    record=json.loads(line)
                    if record['split'] in ['val','test'] and stage_available(record,'sft'):
                        truth[record['sample_id']]={k:record[k] for k in ['sample_id','split','ego','agents','future_times_s','local_agent_risk_labels']}
        for job in ready:
            out=Path(job['output']);report=json.loads((out/'report.json').read_text())
            rows=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
            assert len(rows)==plan['expected_splits'][job['split']]
            assert len({r['sample_id'] for r in rows})==len(rows)
            assert all(truth[r['sample_id']]['split']==job['split'] for r in rows)
            extra,details=supplemental(rows,truth)
            assert extra['total_detected']==report['detection']['true_positives'],'Supplement matching differs from native metrics'
            assert extra['total_targets']==report['detection']['true_positives']+report['detection']['false_negatives']
            write(out/'supplemental.json',extra);write(out/'dangerous_frames.json',details)
            processed[job['name']]=dict(arm=job['arm'],split=job['split'],epoch=job['epoch'],report=report,supplemental=extra)
            write(ROOT/'FULL_RESULTS.json',processed)
        write(ROOT/'FINALIZER_STATUS.json',dict(state='running',processed=list(processed),time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
        if state['state'] in ['complete','ended_with_failures']:
            if state['state']!='complete' or len(processed)!=len(plan['jobs']):raise RuntimeError('Evaluation campaign incomplete')
            break
        time.sleep(15)
    lines=['# 完整规划验证与test评估','',f"完成时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}",'',
           '六组 checkpoint 在运行前固定为原训练 best_planning；完整验证不重选，test 不参与选模型。验证集1451帧、test1522帧均逐帧生成并保存预测。','',
           'ADE/FDE只统计有效且覆盖的点，必须同时阅读有效率与覆盖率。补充1/2/2.5/3秒误差使用同一预测和原生指标函数，不外推短轨迹。碰撞为离线几何指标，不代表闭环安全。共享GPU时延不用于宣称性能优势。','',
           '|组别|划分|Epoch|有效率|ADE(m)|FDE(m)|教师要求点覆盖率|教师要求点ADE(m)|SHORT召回|2s危险目标召回|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    for job in plan['jobs']:
        x=processed[job['name']];g=x['report']['free_generation'];d=x['supplemental']['hazard_detection']['2.0']
        lines.append('| '+' | '.join([job['arm'],job['split'],str(job['epoch']),fmt(g['schema_valid_rate']),fmt(g['ego_ade_m']),fmt(g['ego_final_horizon_fde_m']),fmt(g['required_point_coverage']),fmt(g['required_ego_ade_m']),fmt(g['short_recall']),fmt(d['recall'])])+' |')
    lines.extend(['','逐组report.json包含Risk AP/校准、检测/运动误差、free_generation、physical_planning、instruction_following；supplemental.json包含分时间窗轨迹误差与危险目标召回；dangerous_frames.json保存逐例漏检与Risk低估诊断。','',
                  '原始全量数值：FULL_RESULTS.json。逐样本预测：results/<arm>/<val或test>/predictions.jsonl。'])
    paired=paired_comparison(plan,truth);write(ROOT/'PAIRED_COMPARISON.json',paired)
    lines.extend(['','## 所有模型共同覆盖点上的配对比较','','以下只比较六组模型都有效且覆盖的完全相同GT点；不能代替上表全样本覆盖率。','',
                  '|组别|划分|配对ADE(m)|配对3s FDE(m)|危险场景配对ADE(m)|危险场景配对3s FDE(m)|','|---|---|---:|---:|---:|---:|'])
    for split,group in paired.items():
        for arm,value in group['arms'].items():lines.append('| '+' | '.join([arm,split,fmt(value['ade_m']),fmt(value['fde_3s_m']),fmt(value['dangerous_scene_ade_m']),fmt(value['dangerous_scene_fde_3s_m'])])+' |')
    lines.extend(['','无效原因见supplemental.json的invalid_plan_reasons。例如initial_acceleration_exceeds_limit表示观测初始加速度超过当前8m/s²动作约束，不应解释为文本格式解析失败。'])
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    write(ROOT/'FINALIZER_STATUS.json',dict(state='complete',processed=list(processed),time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(ROOT/'FINALIZER_STATUS.json',dict(state='failed',error=str(error),traceback=traceback.format_exc()));raise
