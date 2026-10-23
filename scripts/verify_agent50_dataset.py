"""Check the full migration and real-frame tensors without updating any weights."""
import argparse
from itertools import zip_longest
from collections import Counter,defaultdict
import json
from pathlib import Path
import torch
from radar_vla.agent50_filter import excluded_agent_kind
from radar_vla.agent50_migration import sha256,write_json
from radar_vla.data import _validate_record,collate_batch
from radar_vla.visual_diagnosis import _single_sample
from radar_vla.model import RadarVLAGrounder,ModelConfig
from radar_vla.losses import loss_grounding
from radar_vla.detection import decode_detections
from radar_vla.detection_metrics import DetectionMetrics
from radar_vla.pipeline import load_models
from radar_vla.planner import PlannerConfig,create_supervision
from radar_vla.tokenizer import PhysicalTokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();out=a.directory
    prov=json.loads((out/'provenance.json').read_text());source=Path(prov['source_manifest']);manifest=out/'manifest.jsonl'
    report=dict(status='PASS',source_manifest_sha256=sha256(source),manifest_sha256=sha256(manifest),samples=0,splits={},representatives=[],training_started=False)
    totals=defaultdict(Counter);choices={};eps={}
    with source.open() as orig,manifest.open() as final:
        for i,(l1,l2) in enumerate(zip_longest(orig,final),1):
            if l1 is None or l2 is None:raise ValueError('Manifest row counts differ')
            old=json.loads(l1);new=json.loads(l2);_validate_record(new,i)
            for key in ('sample_id','episode_id','split','timestamp_s','ego','road_segments','risk_label','risk_agents','agent_risk_labels','teacher_mode'):
                if old.get(key)!=new.get(key):raise ValueError(f'Unrelated field differs: {key}')
            oldagents={str(x['id']):x for x in old['agents']};kept={str(x['id']):x for x in new['agents']};ignored={str(x['id']):x for x in new['ignored_agents']}
            if set(kept)&set(ignored):raise ValueError('Positive/ignored overlap')
            if set(oldagents)!=set(kept)|set(ignored):raise ValueError('Instance IDs not conserved across positive/ignored sets')
            for identifier,agent in kept.items():
                if agent['category']!='agent' or excluded_agent_kind(agent) is not None:raise ValueError('Forbidden retained category')
                for k,v in oldagents[identifier].items():
                    if k!='category' and agent.get(k)!=v:raise ValueError(f'Agent physical field changed: {k}')
            for agent in ignored.values():
                if agent['ignore_reason'] not in ('pedestrian','cyclist'):raise ValueError('Unknown ignore reason')
            for key,value in old['sensors']['radar'].items():
                if key not in ('sensor_velocity_current_ego','poses_current_ego') and new['sensors']['radar'][key]!=value:
                    raise ValueError(f'Radar inputs changed: {key}')
            ep=new['episode_id'];split=new['split']
            if ep in eps and eps[ep]!=split:raise ValueError('Split leakage')
            eps[ep]=split;c=totals[split];c['samples']+=1;c['old_agents']+=len(oldagents);c['positive_agents']+=len(kept);c['ignored_agents']+=len(ignored)
            for agent in ignored.values():c[agent['ignore_reason']]+=1
            c['sft_teacher_unavailable']+=not new['planner_supervision_available']
            flags=dict(positive=bool(kept),empty_with_ignore=not kept and bool(ignored),
                short_available=new['teacher_mode']=='short' and new['planner_supervision_available'],short_unavailable=not new['planner_supervision_available'])
            for name,selected in flags.items():
                if selected and name not in choices:choices[name]=(old,new)
            report['samples']+=1
            if i%1000==0:print('validated records',i,flush=True)
        if orig.readline() or final.readline():raise ValueError('Manifest row counts differ')
    report['splits']=dict(totals);report['episodes']=len(eps)
    cfg=json.loads(Path(prov['config']).read_text());torch.manual_seed(42)
    model=RadarVLAGrounder(ModelConfig(**cfg['model'])).eval();state={k:v.clone() for k,v in model.state_dict().items()}
    pcfg=PlannerConfig(**cfg['planner']);tokenizer=PhysicalTokenizer(bins=pcfg.bins)
    for name,(old,new) in choices.items():
        old_sample=_single_sample(old);sample=_single_sample(new)
        for key in ('radar','radar_doppler_valid','range_m','azimuth_rad','time_offsets_s','ego_future','ego_future_mask','ego_velocity','risk_target','risk_mask'):
            if not torch.equal(old_sample[key],sample[key]):raise ValueError(f'Real tensor changed: {key}')
        indices={key:i for i,key in enumerate(old_sample['agent_ids'])}
        for j,key in enumerate(sample['agent_ids']):
            i=indices[key]
            for field in ('agent_future','agent_future_mask','agent_size','agent_heading'):
                if not torch.equal(old_sample[field][i],sample[field][j]):raise ValueError(f'Real Agent tensor changed: {field}')
            assert torch.equal(old_sample['agent_state'][i,[0,1,3,4]],sample['agent_state'][j,[0,1,3,4]])
        batch=collate_batch([sample]);prediction=model(batch);losses=loss_grounding(prediction,batch)
        if not torch.isfinite(losses['total']) or losses['detection_classification']!=0:raise ValueError('Invalid single-class loss')
        model.zero_grad(set_to_none=True);losses['total'].backward()
        decoded=decode_detections(prediction,model.config,batch['future_times_s'])
        meter=DetectionMetrics(2.,'agent_v1');meter.update(decoded,batch)
        teacher=create_supervision(batch,tokenizer,pcfg)['rows'][0]['mode']
        if new['planner_supervision_available'] and teacher!=new['teacher_mode']:raise ValueError('Teacher changed')
        if not new['planner_supervision_available'] and teacher is not None:raise ValueError('Unavailable teacher was not masked')
        report['representatives'].append(dict(case=name,sample_id=new['sample_id'],episode_id=new['episode_id'],positive_agents=len(new['agents']),
            ignored_agents=len(new['ignored_agents']),loss=float(losses['total'].detach()),teacher_label=new['teacher_mode'],effective_teacher=teacher,
            radar_arrays_and_masks_identical=True,retained_physical_targets_identical=True,metrics=meter.compute()))
    report['parameter_max_change']=max(float((v-state[k]).abs().max()) for k,v in model.state_dict().items())
    legacy,_,_=load_models('analysis/retrained_grounding_v2/best.pt','cpu')
    report['legacy_checkpoint_loaded']=legacy.config.detection_class_schema=='legacy4_v1'
    try:model.load_state_dict(legacy.state_dict())
    except ValueError:report['cross_schema_rejected']=True
    else:raise ValueError('Legacy checkpoint unexpectedly loaded in single-class model')
    write_json(out/'acceptance.json',report);print('ACCEPTANCE PASS',report['samples'],list(choices),flush=True)

if __name__=='__main__':main()
