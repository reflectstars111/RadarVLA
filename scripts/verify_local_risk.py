"""Full invariant audit plus real-sample tensor/gradient checks; no optimization."""
from collections import Counter
import argparse,json
from pathlib import Path
import torch
from radar_vla.agent50_migration import sha256,write_json
from radar_vla.local_risk_migration import assert_invariants
from radar_vla.local_risk import LOCAL_RISK_SCHEMA
from radar_vla.data import _validate_record,collate_batch
from radar_vla.visual_diagnosis import _single_sample
from radar_vla.model import RadarVLAGrounder,ModelConfig
from radar_vla.pipeline import resolve_config,validate_input_shapes
from radar_vla.losses import loss_grounding
from radar_vla.planner import PlannerConfig,create_supervision
from radar_vla.tokenizer import PhysicalTokenizer


def main():
    from itertools import zip_longest
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();out=a.directory
    prov=json.loads((out/'provenance.json').read_text());source=Path(prov['source']);manifest=out/'manifest.jsonl'
    if sha256(source)!=prov['source_sha256'] or sha256(manifest)!=prov['manifest_sha256']:raise ValueError('Provenance hash mismatch')
    counts=Counter();choices={};eps={};sensors=set()
    with source.open() as orig,manifest.open() as final:
        for i,(la,lb) in enumerate(zip_longest(orig,final),1):
            if la is None or lb is None:raise ValueError('Cohort size changed')
            old,new=json.loads(la),json.loads(lb);assert_invariants(old,new);_validate_record(new,i)
            from radar_vla.collection_adapter import _mode_from_risk
            selected={str(identifier):new['local_agent_risk_labels'][str(identifier)] for identifier in new['planner_agent_ids']}
            expected_mode=_mode_from_risk(new['local_risk'],selected,bool(new['agents']))
            if expected_mode=='long' and any(not all(a['future_valid']) for a in new['agents'] if str(a['id']) in selected):
                expected_mode=None
            if expected_mode!=new['teacher_mode']:raise ValueError('All-frame SFT mode consistency failed')
            ep=new['episode_id'];sp=new['split']
            if ep in eps and eps[ep]!=sp:raise ValueError('Episode leakage')
            eps[ep]=sp;counts[sp]+=1
            radar=new['sensors']['radar']
            for field in ('power','unfolded_doppler','doppler_valid'):
                values=radar[field] if isinstance(radar[field],list) else [radar[field]]
                sensors.update(values)
            local=new['local_risk']
            flags=dict(agents=bool(new['agents']),empty=not new['agents'],
                short=new['teacher_mode']=='short' and new['planner_supervision_available'],
                long=new['teacher_mode']=='long' and new['planner_supervision_available'],
                teacher_changed=old.get('teacher_mode')!=new.get('teacher_mode'),
                teacher_unavailable=not new['planner_supervision_available'],
                future_unknown=not all(local['pcol_mask']),areq_invalid=not local['areq_mask'])
            for name,yes in flags.items():
                if yes and name not in choices:choices[name]=(old,new)
    missing=[p for p in sorted(sensors) if not Path(p).is_file()]
    if missing:raise FileNotFoundError(f'Missing sensor files: {missing[:5]}')
    cfg=resolve_config(json.loads(Path(prov['config']).read_text()));torch.manual_seed(42)
    model=RadarVLAGrounder(ModelConfig(**cfg['model'])).eval();before={k:v.clone() for k,v in model.state_dict().items()}
    pc=PlannerConfig(**cfg['planner']);tokenizer=PhysicalTokenizer(bins=pc.bins);rows=[]
    for case,(old,new) in choices.items():
        a,b=_single_sample(old),_single_sample(new)
        for key in ('radar','radar_doppler_valid','range_m','azimuth_rad','radar_pose','radar_sensor_velocity',
                    'agent_state','agent_mask','agent_class','agent_future','agent_future_mask','agent_size','agent_heading',
                    'ego_future','ego_future_mask','ego_velocity','agent_planner_mask','detection_ignore_boxes'):
            if not torch.equal(a[key],b[key]):raise ValueError(f'Unrelated tensor changed: {key}')
        batch=collate_batch([b]);pred=model(batch);loss=loss_grounding(pred,batch)
        if not torch.isfinite(loss['total']):raise ValueError('Nonfinite loss')
        model.zero_grad(set_to_none=True);loss['total'].backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
        teacher=create_supervision(batch,tokenizer,pc)['rows'][0]['mode']
        if teacher!=new['teacher_mode']:raise ValueError(f'Teacher mismatch {case}: {teacher} vs {new["teacher_mode"]}')
        expected=torch.tensor([*new['local_risk']['pcol'],new['local_risk']['dmin'],new['local_risk']['areq']])
        assert torch.equal(b['risk_target'],expected)
        rows.append(dict(case=case,sample_id=new['sample_id'],episode_id=new['episode_id'],teacher_mode=teacher,
            risk_mask=b['risk_mask'].tolist(),loss=float(loss['total'].detach()),unrelated_tensors_identical=True))
    change=max(float((v-before[k]).abs().max()) for k,v in model.state_dict().items())
    if change!=0:raise ValueError('Parameters changed')
    # A metadata-only mock exercises input/schema gates without loading an old model.
    class MetadataDataset:pass
    ds=MetadataDataset();ds.records=[next(iter(choices.values()))[1]];ds.risk_schema=LOCAL_RISK_SCHEMA;ds.class_schema='agent_v1';ds.supervision=True
    validate_input_shapes(ds,cfg)
    wrong={**cfg,'data':{k:v for k,v in cfg['data'].items() if k!='risk_supervision_schema'}}
    try:validate_input_shapes(ds,wrong)
    except ValueError:cross_schema_rejected=True
    else:raise ValueError('Legacy risk configuration accepted Local data')
    report=dict(status='PASS',samples=sum(counts.values()),episodes=len(eps),splits=counts,
        sensor_files_existing=len(sensors),representatives=rows,source_sha256=prov['source_sha256'],manifest_sha256=prov['manifest_sha256'],
        parameter_max_change=change,optimizer_steps=0,training_started=False,cross_risk_schema_rejected=cross_schema_rejected)
    write_json(out/'acceptance.json',report);print('ACCEPTANCE PASS',report['samples'],list(choices),flush=True)

if __name__=='__main__':main()
