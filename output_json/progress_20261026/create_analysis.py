"""Render the frozen full evaluation results, without selecting a new model."""
from pathlib import Path
import datetime
import hashlib
import json
import time

R=Path(__file__).resolve().parent
ARMS=['baseline','action','assignment','risk','combined','baseline_readout']


def num(x):return '—' if x is None else f'{x:.3f}'
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def rate(x,count):return f'{pct(x)} ({round(x*count)}/{count})' if x is not None and count else '—'
def table(lines,headers,rows):
    lines.extend(['','| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
    lines.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows)


def main():
    while True:
        f=R/'FINALIZER_STATUS.json'
        if f.exists():
            state=json.loads(f.read_text())
            if state['state']=='failed':raise RuntimeError(state)
            if state['state']=='complete':break
        time.sleep(15)
    data=json.loads((R/'FULL_RESULTS.json').read_text());pair=json.loads((R/'PAIRED_COMPARISON.json').read_text());plan=json.loads((R/'plan.json').read_text())
    assert len(data)==12 and all(json.loads((R/'STATUS.json').read_text())['jobs'][j]['state']=='complete' for j in data)
    lines=['# RadarVLA 完整规划验证与 test 评估报告','',
           '本报告覆盖六组固定 checkpoint 的完整 validation（1451帧）和 test（1522帧），共17838次逐帧规划评估。所有结果均已完成并通过样本数、唯一ID和原生检测匹配一致性检查。','',
           '模型在运行前固定为训练期间选出的 best_planning.pt：baseline 为第3轮，其余为第5轮。没有使用完整validation或test重新选模型、调整阈值、修改损失或继续训练。下列末端轨迹FDE为3秒GT终点误差；SHORT缺失的3秒终点不计为成功预测，覆盖率另列。',
           '', '## 1. 规划结果与覆盖率']
    for split in ['val','test']:
        lines.extend(['',f'### {split}'])
        table(lines,['组别','有效输出','ADE(m)','3s FDE(m)','全时域点覆盖','教师要求点覆盖','教师要求点ADE(m)','SHORT召回'],[
            [arm,pct((g:=data[arm+'_'+split]['report']['free_generation'])['schema_valid_rate']),num(g['ego_ade_m']),num(g['ego_final_horizon_fde_m']),pct(g['ego_point_coverage']),pct(g['required_point_coverage']),num(g['required_ego_ade_m']),pct(g['short_recall'])] for arm in ARMS])
        counts=data['baseline_'+split]['report']['free_generation']['teacher_mode_counts'];lines.extend(['',f'真实模式数量：{counts}。模式准确率容易受LONG占比影响，SHORT召回必须单独阅读。'])
    lines.extend(['','## 2. 相同覆盖点的配对比较','',
                  '仅比较六组有效输出共同覆盖的完全相同GT点，防止通过少预测、输出无效或选择较短时域降低误差。这个子集的结果不能替代全样本有效率和覆盖率。危险场景按既有、已知的2秒per-agent冲突标签识别。'])
    for split in ['val','test']:
        group=pair[split];lines.extend(['',f"{split}：共同轨迹点{group['common_points']}，共同3秒终点{group['common_final_samples']}，原始危险帧{group['dangerous_frames']}，共同危险轨迹点{group['common_dangerous_points']}。"])
        table(lines,['组别','共同ADE(m)','共同3s FDE(m)','危险场景共同ADE(m)','危险场景共同3s FDE(m)'],[
            [arm,*[num(group['arms'][arm][k]) for k in ['ade_m','fde_3s_m','dangerous_scene_ade_m','dangerous_scene_fde_3s_m']]] for arm in ARMS])
    lines.extend(['','## 3. 检测、危险目标与Risk','',
                  '检测为固定confidence/NMS设置、2m中心匹配；不是检测AP。危险目标分母仅包含原标签已知且对应时窗有冲突的保留目标，不将未知标注当作安全。Risk AP使用原生全量评估，三个时窗为0.5/1/2秒。'])
    for split in ['val','test']:
        report=data['baseline_'+split]['report'];positives={h:v['positives'] for h,v in report['collision'].items()};lines.extend(['',f'{split}冲突正例数量：{positives}。'])
        table(lines,['组别','检测P/R/F1','0.5s危险目标检出','1s危险目标检出','2s危险目标检出','0.5/1/2s Risk AP','d_min MAE(m)','a_req MAE(m/s²)'],[
            [arm,' / '.join(num((rep:=data[arm+'_'+split]['report'])['detection'][k]) for k in ['precision','recall','f1']),
             *[f"{(d:=data[arm+'_'+split]['supplemental']['hazard_detection'][h])['detected']}/{d['targets']}" for h in ['0.5','1.0','2.0']],
             ' / '.join(num(v['average_precision']) for v in rep['collision'].values()),num(rep['dmin_mae']),num(rep['areq_mae'])] for arm in ARMS])
        table(lines,['组别','0.5s危险帧Risk<0.5','1s危险帧Risk<0.5','2s危险帧Risk<0.5'],[
            [arm,*[f"{(d:=data[arm+'_'+split]['supplemental']['hazard_detection'][h])['risk_below_0p5']}/{d['positive_frames']}" for h in ['0.5','1.0','2.0']]] for arm in ARMS])
    lines.extend(['','Risk<0.5仅为固定诊断阈值，不是本报告重新优化的工作阈值。较高AP表示排序较好，不保证在0.5阈值下有足够危险召回；少量正例也会使AP波动较大。','',
                  '## 4. 离线物理指标与无效原因','',
                  '碰撞基于原生0.05秒采样的朝向框接触，在实际生成轨迹时域内计算；不同SHORT比例、未知目标未来与无效输出会改变可评分母。必须读取分母与安全覆盖率，不能将此率视作全样本闭环碰撞率。局部恒速TTC、实际轨迹碰撞与KRS标签是不同量。'])
    for split in ['val','test']:
        lines.extend(['',f'### {split}'])
        table(lines,['组别','碰撞率(例/可评分母)','安全违规率(例/可评分母)','安全覆盖率','初始速度误差(m/s)','jerk RMS(m/s³)','无效原因'],[
            [arm,rate((p:=data[arm+'_'+split]['report']['physical_planning']).get('collision_rate'),p.get('collision_rate_count')),
             rate(p.get('safety_violation_rate'),p.get('safety_violation_rate_count')),pct(p.get('safety_coverage')),num(p.get('initial_velocity_error_mps')),num(p.get('jerk_rms_mps3')),
             json.dumps(data[arm+'_'+split]['supplemental'].get('invalid_plan_reasons',{}),ensure_ascii=False)] for arm in ARMS])
    lines.extend(['','initial_acceleration_exceeds_limit表示观测初始加速度已超过当前8m/s²动作约束，不是文本解析失败；本次未放宽约束或修改输入来提高有效率。','',
                  '## 5. 分时窗轨迹误差','',
                  '以下按原生planning_metrics对同一输出在1/2/2.5/3秒GT点上统计，不重采样、不外推。每个单元格为ADE/FDE，后附点覆盖率；不同时间窗、不同覆盖率的误差不可直接等价。'])
    for split in ['val','test']:
        table(lines,['组别',split+'：1s',split+'：2s',split+'：2.5s',split+'：3s'],[
            [arm,*[f"{num((h:=data[arm+'_'+split]['supplemental']['trajectory_horizons_s'][t])['ego_ade_m'])}/{num(h['ego_final_horizon_fde_m'])} ({pct(h['ego_point_coverage'])})" for t in ['1.0','2.0','2.5','3.0']]] for arm in ARMS])
    lines.extend(['','## 6. 证据范围与复核入口','',
                  '- 单seed实验，不能据此宣称统计显著；按用户要求未新增三seed训练。','- 初始化审计发现Risk SFT与等预算baseline readout对照的planner初始可训练权重SHA256不同。相同seed不等于相同初始权重，本轮SFT差异不能完全归因于Risk输入。原始记录见INITIALIZATION_AUDIT.json；这不影响已完成指标的真实性，但限制单变量因果解释。','- 常规误差、共同点误差、危险场景误差、覆盖率和碰撞可评分母必须联合判断，不能凭单个ADE或AP挑选骨干/方案。','- 轨迹范围仍为3秒，KRS为0.5/1/2秒、几何窗口2秒；二者没有混同。','- 当前仍使用既有离线双向预处理数据，不宣称严格在线因果性或闭环安全。','- 共享GPU时的时延受节流及其他任务影响，本报告不据其宣称独占性能。','',
                  f"训练源码：`{plan['training']['source_commit']}`；manifest SHA256：`{plan['training']['manifest_sha256']}`。",
                  '', '各组完整数据：results/<arm>/<val或test>/report.json、predictions.jsonl、supplemental.json、dangerous_frames.json。原生报告保留检测/运动误差与instruction_following等全部字段；本报告没有省略或改写其数值。',
                  '', '机器可读汇总：FULL_RESULTS.json、PAIRED_COMPARISON.json。checkpoint及评估代码SHA256见plan.json；资源调度变更与迁移另有审计记录。'])
    (R/'EVALUATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    record=dict(state='complete',time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),jobs=len(data),report_sha256=hashlib.sha256((R/'EVALUATION_REPORT.md').read_bytes()).hexdigest())
    (R/'DELIVERY_STATUS.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
