"""Observe native forwards/losses; never changes predictions, losses, RNG or updates."""
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from radar_vla.losses import DEFAULT_WEIGHTS
from radar_vla.metrics import binary_metrics
from radar_vla.detection import decode_detections


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def risk_parts(prediction,batch,components):
    """Split actual canonical focal loss using its GLOBAL valid-element denominator."""
    truth=batch['risk_target'][:,:3]
    mask=batch['risk_mask'][:,:3].bool() & torch.isfinite(truth)
    truth=torch.where(mask,truth,torch.zeros_like(truth))
    logits=prediction['risk_logits'];prob=logits.sigmoid()
    bce=F.binary_cross_entropy_with_logits(logits,truth,reduction='none')
    pt=truth*prob+(1-truth)*(1-prob)
    focal=(.25*truth+.75*(1-truth))*(1-pt).square()*bce
    denom=mask.sum().clamp_min(1)
    parts=[(focal[:,j]*mask[:,j]).sum()/denom for j in range(3)]
    torch.testing.assert_close(sum(parts),components['collision'],rtol=2e-5,atol=2e-6)
    return parts+[components['distance'],components['brake']],focal,mask,denom


def inspect_gradients(parts,components,captures,model):
    weights={**DEFAULT_WEIGHTS,**getattr(model,'loss_weights',{})}
    multipliers=[weights['collision']]*3+[weights['distance'],weights['brake']]
    weighted=[part*multiplier for part,multiplier in zip(parts,multipliers)]
    risk_total=sum(weighted)+weights['monotonicity']*components['monotonicity']
    losses=weighted+[risk_total,components['total']-risk_total]
    names=['pcol0.5','pcol1','pcol2','dmin','areq','risk_total','detection_total']
    tensors=[captures['spatial'],captures['tokens']]
    report={}
    for name,loss in zip(names,losses):
        gradients=torch.autograd.grad(loss,tensors,retain_graph=True,allow_unused=True)
        report[name]={block:float(g.detach().float().norm()) if g is not None else 0.
                      for block,g in zip(['encoder_feature','shared_radar_tokens'],gradients)}
    return report


class Observer:
    def __init__(self,pipeline,output,audit,matched_objects,attempt=1):
        self.pipeline=pipeline;self.output=Path(output);self.audit=Path(audit);self.audit.mkdir(exist_ok=True)
        self.matched_objects=matched_objects;self.attempt=attempt
        self.epoch=0;self.step=0;self.probed=set();self.probe_count=0;self.captures={};self.model=None
        self.original_loss=pipeline._loss;self.original_validate=pipeline.validate;self.original_loader=pipeline.make_loader

    def install(self):
        self.pipeline._loss=self.loss;self.pipeline.validate=self.validate;self.pipeline.make_loader=self.loader

    def loader(self,*args,**kwargs):
        if kwargs.get('shuffle',False):
            epoch=json.loads((self.output/'status.json').read_text())['epoch']
            self.epoch=epoch;self.step=0;self.probed=set();self.probe_count=0
        return self.original_loader(*args,**kwargs)

    def capture_model(self,model):
        if self.model is model:return
        assert self.model is None
        self.model=model
        def spatial(m,i,o):self.captures['spatial']=o
        def tokens(m,i,o):self.captures['tokens']=o
        model.encoder.register_forward_hook(spatial);model.output_norm.register_forward_hook(tokens)

    def loss(self,grounder,planner,batch,*args,**kwargs):
        self.capture_model(grounder)
        total,components,prediction=self.original_loss(grounder,planner,batch,*args,**kwargs)
        if not grounder.training:return total,components,prediction
        self.step+=1
        parts,focal,mask,denom=risk_parts(prediction,batch,components)
        weights={**DEFAULT_WEIGHTS,**getattr(grounder,'loss_weights',{})}
        multipliers=[weights['collision']]*3+[weights['distance'],weights['brake']]
        p=batch['risk_target'];valid=batch['risk_mask'].bool()
        positives=[int(((p[:,j]>.5)&valid[:,j]).sum()) for j in range(3)]
        brake_positive=int(((p[:,4]>0)&valid[:,4]).sum())
        triggers=[]
        if self.step==1:triggers.append('first_batch')
        for i,n in enumerate(positives):
            if n and 'positive_'+str(i) not in self.probed:triggers.append('positive_'+str(i))
        if brake_positive and 'brake_positive' not in self.probed:triggers.append('brake_positive')
        if self.step%400==0:triggers.append('periodic_'+str(self.step))
        record=dict(epoch=self.epoch,batch=self.step,attempt=self.attempt,samples=len(p),sample_ids=batch['sample_id'],
            risk_horizons_s=[.5,1.,2.],losses={k:float(v.detach()) for k,v in components.items()},
            risk_dimension_loss=[float(x.detach()) for x in parts],risk_dimension_weights=multipliers,
            weighted_risk_dimension_loss=[float(x.detach())*m for x,m in zip(parts,multipliers)],
            collision_valid_element_denominator=int(denom),physical_scales=dict(dmin=10.,areq=5.),
            positive_counts=positives,known_counts=valid.sum(0).tolist(),brake_positive=brake_positive,
            positive_focal_contribution=[float((focal[:,j]*mask[:,j]*(p[:,j]>.5)).sum().detach()/denom) for j in range(3)],
            negative_focal_contribution=[float((focal[:,j]*mask[:,j]*(p[:,j]<=.5)).sum().detach()/denom) for j in range(3)])
        # Avoid treating the unweighted raw component values as a comparable objective.
        record['weighted_losses']={k:float(v.detach())*weights.get(k.removeprefix('detection_'),1.) for k,v in components.items() if k!='total'}
        torch.testing.assert_close(sum(record['weighted_losses'].values()),float(total.detach()),rtol=2e-5,atol=2e-5)
        record['regression_contributions']={}
        for index,name,scale,component in [(3,'dmin',10.,'distance'),(4,'areq',5.,'brake')]:
            known=valid[:,index]&torch.isfinite(p[:,index]);count=known.sum().clamp_min(1)
            target=torch.where(known,p[:,index],torch.zeros_like(p[:,index]))
            each=F.smooth_l1_loss(prediction['risk'][:,index]/scale,target/scale,reduction='none')
            positive=(p[:,4]>0) if index==4 else (p[:,2]>.5)
            values=[(each*known*positive).sum()/count,(each*known*~positive).sum()/count]
            torch.testing.assert_close(sum(values),components[component],rtol=2e-5,atol=2e-6)
            record['regression_contributions'][name]=dict(positive=float(values[0].detach()),other=float(values[1].detach()),
                weighted_positive=float(values[0].detach())*weights[component],weighted_other=float(values[1].detach())*weights[component],
                positive_definition='areq>0' if index==4 else 'Pcol2s positive')
        if triggers and self.probe_count<8:
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state() if p.is_cuda else None
            record['gradient_probes']=inspect_gradients(parts,components,self.captures,grounder)
            assert torch.equal(cpu_rng,torch.get_rng_state()),'Observer changed CPU RNG'
            if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state()),'Observer changed CUDA RNG'
            record['gradient_triggers']=triggers;self.probed.update(triggers);self.probe_count+=1
            record['gradient_semantics']='Per-batch objective feature gradients before accumulation scaling and clipping; no parameter .grad writes.'
        with (self.audit/f'epoch_{self.epoch:03d}_train_attempt_{self.attempt:02d}.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        if self.step==1 or self.step%50==0:
            atomic(self.audit/'progress.json',dict(state='training',epoch=self.epoch,batch=self.step,epochs=20,
                samples_seen_in_epoch=self.step*4,gradient_probe_batches=self.probe_count,time=time.time(),last_loss=float(total.detach())))
        self.captures.clear()
        return total,components,prediction

    def validate(self,grounder,planner,dataset,batch_size,device,precision='float32',progress_desc=None):
        rows=[];records={r['sample_id']:r for r in dataset.records}
        def observe(model,inputs,prediction):
            batch=inputs[0];decoded=decode_detections(prediction,model.config,batch['future_times_s'])
            keys=['agent_state','agent_state_mask','agent_radial_mask','agent_radial_observed','agent_future','agent_future_mask']
            for i,sid in enumerate(batch['sample_id']):
                single={k:batch[k][i:i+1] for k in keys}
                objects=self.matched_objects(records[sid],single,decoded[i]['detections'])
                rows.append(dict(sample_id=sid,episode_id=records[sid]['episode_id'],risk=prediction['risk'][i].float().cpu().tolist(),
                    target=batch['risk_target'][i].cpu().tolist(),mask=batch['risk_mask'][i].cpu().tolist(),objects=objects))
        hook=grounder.register_forward_hook(observe)
        try:result=self.original_validate(grounder,planner,dataset,batch_size,device,precision,progress_desc)
        finally:hook.remove();self.captures.clear()
        assert len(rows)==len(dataset) and len({r['sample_id'] for r in rows})==len(rows)
        assert sum(o['matched'] for r in rows for o in r['objects'])==result['detection']['true_positives']
        summary={}
        for i,h in enumerate(['0.5s','1s','2s']):
            objects=[]
            for row in rows:
                sources=set(map(str,records[row['sample_id']]['local_risk']['pcol_source_agent_ids'][i]))
                objects.extend(o for o in row['objects'] if o['id'] in sources)
            eligible=[r for r in rows if r['mask'][i]];positive=[r for r in eligible if r['target'][i]>.5];negative=[r for r in eligible if r['target'][i]<=.5]
            summary[h]=dict(critical_objects=len(objects),critical_detected=sum(o['matched'] for o in objects),
                positives=len(positive),positive_episodes=len({r['episode_id'] for r in positive}),
                risk_tp_at_0p5=sum(r['risk'][i]>=.5 for r in positive),risk_fp_at_0p5=sum(r['risk'][i]>=.5 for r in negative))
        conditional={}
        for name,index,select in [('danger_dmin',3,lambda r:r['target'][2]>.5),('positive_brake',4,lambda r:r['target'][4]>0),('zero_brake',4,lambda r:r['target'][4]==0)]:
            selected=[r for r in rows if r['mask'][index] and select(r)]
            conditional[name]=dict(samples=len(selected),mae=sum(abs(r['risk'][index]-r['target'][index]) for r in selected)/len(selected) if selected else None,
                near_zero_predictions=sum(r['risk'][index]<.01 for r in selected))
        episode={}
        for ep in sorted({r['episode_id'] for r in rows}):
            selected=[r for r in rows if r['episode_id']==ep]
            episode[ep]={h:binary_metrics([r['risk'][j] for r in selected if r['mask'][j]],[r['target'][j] for r in selected if r['mask'][j]]) for j,h in enumerate(['0.5s','1s','2s'])}
        result['critical_and_risk']=summary;result['risk_conditional']=conditional
        atomic(self.audit/f'epoch_{self.epoch:03d}_validation.json',dict(horizons=summary,conditional=conditional,by_episode=episode))
        path=self.audit/f'epoch_{self.epoch:03d}_val_samples.jsonl';tmp=path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));tmp.replace(path)
        return result
