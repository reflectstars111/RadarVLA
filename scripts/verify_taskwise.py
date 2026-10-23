"""Task-3 full metadata/teacher audit and canonical real-batch gradient checks."""
from __future__ import annotations
import argparse,copy,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from radar_vla.agent50_migration import sha256,write_json
from radar_vla.taskwise_migration import assert_baseline
from radar_vla.task_supervision import validate_tasks,stage_available,MODE_INDEX
from radar_vla.data import _validate_record,collate_batch
from radar_vla.pipeline import resolve_config,validate_input_shapes,_loss
from radar_vla.planner import PlannerConfig,RiskConditionedPlanner,create_supervision
from radar_vla.tokenizer import PhysicalTokenizer
from radar_vla.visual_diagnosis import _single_sample
from radar_vla.model import RadarVLAGrounder,ModelConfig


def teacher_metadata_batch(r):
    """No radar synthesis: inspect label grammar with radial values explicitly masked.

    Real sensor-dependent radial teachers and end-to-end model tensors are checked
    separately through the canonical Dataset on representative real samples.
    """
    agents=sorted(r['agents'],key=lambda a:(np.linalg.norm(a['position']),str(a['id'])))
    n=len(agents);selected=set(map(str,r['planner_agent_ids']));known=bool(r['agent_supervision_available'])
    state=torch.zeros(1,n,5);future=torch.zeros(1,n,6,2);future_mask=torch.zeros(1,n,6,dtype=torch.bool)
    risks=torch.zeros(1,n,5);riskmask=torch.zeros(1,n,5,dtype=torch.bool)
    for i,a in enumerate(agents):
        state[0,i,[0,1]]=torch.tensor(a['position']);state[0,i,[3,4]]=torch.tensor(a['velocity'])
        future[0,i]=torch.tensor(a.get('future_xy',[[0.,0.]]*6));future_mask[0,i]=torch.tensor(a.get('future_valid',[False]*6))
        q=r['local_agent_risk_labels'][str(a['id'])];risks[0,i]=torch.tensor([*q['pcol'],q['dmin'],q['areq']]);riskmask[0,i]=torch.tensor([*q['pcol_mask'],q['dmin_mask'],q['areq_mask']])
    def order(i):
        q=r['local_agent_risk_labels'][str(agents[i]['id'])];hits=[h for h,(p,m) in enumerate(zip(q['pcol'],q['pcol_mask'])) if p>.5 and m]
        return (min(hits,default=4),q['dmin'] if q['dmin_mask'] else float('inf'),str(agents[i]['id']))
    q=r['local_risk'];state_mask=torch.ones_like(state,dtype=torch.bool);state_mask[:,:,2]=False
    if not known:state_mask[:]=False;future_mask[:]=False
    batch=dict(agent_state=state,agent_mask=torch.full((1,n),known,dtype=torch.bool),agent_state_mask=state_mask,
        agent_planner_mask=torch.tensor([[str(a['id']) in selected for a in agents]],dtype=torch.bool),
        agent_risk=risks,agent_risk_mask=riskmask,critical_agent_order=torch.tensor([sorted(range(n),key=order)],dtype=torch.long),
        agent_future=future,agent_future_mask=future_mask,ego_future=torch.tensor([r['ego'].get('future_xy',[[0.,0.]]*6)]),
        ego_future_mask=torch.tensor([r['ego'].get('future_valid',[False]*6)]),future_times_s=torch.tensor([r['future_times_s']]),
        risk_target=torch.tensor([[*q['pcol'],q['dmin'],q['areq']]]),risk_mask=torch.tensor([[*q['pcol_mask'],q['dmin_mask'],q['areq_mask']]]),
        taskwise_protocol_id=torch.tensor([1]),mode_target=torch.tensor([MODE_INDEX[r['mode_label']]]),mode_supervision_mask=torch.tensor([r['mode_mask']]),
        local_set_known=torch.tensor([known]),planner_supervision_available=torch.tensor([r['sft_available']]),
        road_segments=[r.get('road_segments',[])],road_task_masks=[r['task_masks']['road']])
    return batch,agents


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();out=a.directory
    prov=json.loads((out/'provenance.json').read_text());source=Path(prov['source'])
    if sha256(source)!=prov['source_sha256'] or sha256(out/'manifest.jsonl')!=prov['manifest_sha256']:raise ValueError('Manifest hash mismatch')
    for path,digest in prov['source_files'].items():
        if sha256(path)!=digest:raise ValueError('Execution source changed: '+path)
    cfg=resolve_config(json.loads(Path(prov['config']).read_text()));pc=PlannerConfig(**cfg['planner']);tok=PhysicalTokenizer(pc.bins)
    counts=Counter();choices={};ids=set();episodes={};sensors=set();teacher_audit=Counter()
    with source.open() as baseline,(out/'manifest.jsonl').open() as incoming,(out/'teacher_validation.jsonl').open('w') as audit:
        for i,line in enumerate(incoming,1):
            r=json.loads(line);_validate_record(r,i);validate_tasks(r)
            if i<=8835:
                old=json.loads(next(baseline));assert_baseline(old,r)
            else:old=None
            if r['sample_id'] in ids:raise ValueError('Duplicate sample ID')
            ids.add(r['sample_id']);ep=r['episode_id'];sp=r['split']
            if ep in episodes and episodes[ep]!=sp or sp!=prov['episode_split'][ep]:raise ValueError('Episode split leakage')
            episodes[ep]=sp;counts[sp]+=1
            for field in ('power','unfolded_doppler','doppler_valid'):
                paths=r['sensors']['radar'][field];sensors.update(paths if isinstance(paths,list) else [paths])
            batch,ordered_agents=teacher_metadata_batch(r);sup=create_supervision(batch,tok,pc);row=sup['rows'][0]
            expected=r['teacher_mode'] if r['sft_available'] else None
            if row['mode']!=expected:raise ValueError('Teacher/eligibility mismatch: '+r['sample_id'])
            described=[str(ordered_agents[q['source_index']]['id']) for q in row['agents']]
            if described!=r['agent_sets']['sft_described_ids']:raise ValueError('SFT description set/order mismatch: '+r['sample_id'])
            if r['mode_label']=='UNKNOWN' and bool((sup['target_ids']!=tok.pad_id).any()):raise ValueError('UNKNOWN has SFT targets')
            teacher_audit['samples']+=1;teacher_audit['mode/'+r['mode_label']]+=1
            teacher_audit['masked_out_of_range_nonradial_scalars']+=row.get('masked_out_of_range_fields',0)
            audit.write(json.dumps(dict(sample_id=r['sample_id'],mode_label=r['mode_label'],sft_available=r['sft_available'],
                sft_described_ids=described,teacher_mode=row['mode'],masked_out_of_range_fields=row.get('masked_out_of_range_fields',0)),separators=(',',':'))+'\n')
            flags=dict(baseline=old is not None,recovered=old is None,unknown=r['mode_label']=='UNKNOWN',
                short=r['sft_available'] and r['mode_label']=='SHORT',long=r['sft_available'] and r['mode_label']=='LONG',
                context_inconsistent=not r['planner_context_consistent'],missing_road=r['sft_available'] and r['mode_label']=='LONG' and not any(any(s['geometry']) or s['width'] for s in r['task_masks']['road']),
                partial_agent_future=any(any(v) and not all(v) for v in r['task_masks']['agent_future'].values()),
                partial_ego_future=any(r['task_masks']['ego_future']) and not all(r['task_masks']['ego_future']),
                many_agents=len(r['agents'])>8)
            for name,yes in flags.items():
                if yes and name not in choices:choices[name]=(old,r)
            if i%1000==0:print('VERIFIED LABELS',i,flush=True)
        if baseline.readline():raise ValueError('Baseline cohort truncated')
    missing=[p for p in sensors if not Path(p).is_file()]
    if missing:raise FileNotFoundError(f'Missing sensor inputs {missing[:5]}')
    cache=json.loads((out/'radar_cache_hashes.json').read_text())
    for path,digest in cache.items():
        if sha256(path)!=digest:raise ValueError('New radar cache hash mismatch')
    torch.manual_seed(42);grounder=RadarVLAGrounder(ModelConfig(**cfg['model'])).eval();planner=RiskConditionedPlanner(pc).eval()
    before_g={k:v.clone() for k,v in grounder.state_dict().items()};before_p={k:v.clone() for k,v in planner.state_dict().items()};representatives=[];real_samples=[]
    for name,(old,r) in choices.items():
        sample=_single_sample(r);batch=collate_batch([sample]);real_samples.append(sample)
        if old is not None:
            prev=_single_sample(old)
            for key in ('radar','radar_doppler_valid','radar_pose','radar_sensor_velocity','range_m','azimuth_rad','agent_state','agent_future','agent_future_mask','ego_future','ego_future_mask','risk_target','risk_mask','agent_class'):
                if not torch.equal(prev[key],sample[key]):raise ValueError('Baseline tensor changed '+key)
        # The production loss path, with existing numerical weights and no optimizer.
        grounder.zero_grad(set_to_none=True);planner.zero_grad(set_to_none=True)
        total,losses,_=_loss(grounder,planner,batch)
        if not torch.isfinite(total):raise ValueError('Nonfinite real-sample loss')
        total.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in [*grounder.parameters(),*planner.parameters()]):raise ValueError('Nonfinite real gradient')
        if not r['sft_available']:
            if losses['token']!=0 or any(value!=0 for key,value in losses.items() if key.startswith('planner_')):raise ValueError('Excluded SFT sample has Planner loss')
            if any(p.grad is not None and p.grad.count_nonzero() for p in planner.parameters()):raise ValueError('Excluded sample has Planner gradient')
        actual=create_supervision(batch,tok,pc)['rows'][0]
        described=[sample['agent_ids'][q['source_index']] for q in actual['agents']]
        if described!=r['agent_sets']['sft_described_ids']:raise ValueError('Canonical real-batch described set mismatch')
        representatives.append(dict(case=name,sample_id=r['sample_id'],episode_id=r['episode_id'],mode_label=r['mode_label'],
            sft_available=r['sft_available'],total_loss=float(total.detach()),described_ids=described,baseline_tensors_identical=old is not None))
    if len(real_samples)>1:
        mixed=collate_batch(real_samples[:min(4,len(real_samples))]);loss,_,_=_loss(grounder,planner,mixed)
        if not torch.isfinite(loss):raise ValueError('Nonfinite mixed real batch')
    change=max([float((v-before_g[k]).abs().max()) for k,v in grounder.state_dict().items()]+[float((v-before_p[k]).abs().max()) for k,v in planner.state_dict().items()])
    if change:raise ValueError('Parameters changed')
    class Metadata:pass
    ds=Metadata();ds.records=[choices['baseline'][1]];ds.class_schema='agent_v1';ds.risk_schema='local_agent_collision_v1';ds.supervision_protocol='taskwise_local_v1';ds.supervision=True
    validate_input_shapes(ds,cfg)
    wrong=copy.deepcopy(cfg);wrong['data'].pop('supervision_protocol')
    try:validate_input_shapes(ds,wrong)
    except ValueError:cross_protocol_rejected=True
    else:raise ValueError('Old protocol accepted new data')
    report=dict(status='PASS',samples=sum(counts.values()),splits=counts,episodes=len(episodes),
        baseline_preserved=8835,teacher_audit=teacher_audit,representatives=representatives,sensor_files_existing=len(sensors),
        new_cache_hashes_verified=len(cache),parameter_max_change=change,optimizer_steps=0,training_started=False,cross_protocol_rejected=cross_protocol_rejected,
        missing_real_cases=[name for name in ('unknown','context_inconsistent','partial_agent_future','partial_ego_future','missing_road','many_agents') if name not in choices])
    write_json(out/'acceptance.json',report);print('ACCEPTANCE PASS',report['samples'],list(choices),flush=True)

if __name__=='__main__':main()
