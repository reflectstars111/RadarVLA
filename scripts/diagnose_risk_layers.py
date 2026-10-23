"""Read-only, episode-separated Risk input/representation/gradient diagnosis."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F
from radar_vla.data import RadarDataset,collate_batch
from radar_vla.geometry import nominal_rollout
from radar_vla.local_risk import box_distances
from radar_vla.losses import loss_grounding
from radar_vla.metrics import binary_metrics
from radar_vla.model import ModelConfig,RadarVLAGrounder
from radar_vla.pipeline import autocast_context,file_digest,source_fingerprint,to_device,write_json
from radar_vla.risk_distillation import is_risk_parameter
from radar_vla.task_supervision import stage_available

LOSS_NAMES=['pcol1','pcol2','pcol3','dmin','areq','monotonicity']

def component_losses(risk,raw,target,mask):
    """Per-record decomposition of canonical B=1 Risk loss, including masks/scales."""
    mask=mask.bool()&torch.isfinite(target)
    target=torch.where(mask,target,torch.zeros_like(target))
    logits=raw[:,:3].float();prob=logits.sigmoid();truth=target[:,:3]
    bce=F.binary_cross_entropy_with_logits(logits,truth,reduction='none')
    pt=truth*prob+(1-truth)*(1-prob)
    focal=(.25*truth+.75*(1-truth))*(1-pt).square()*bce
    collision=focal*mask[:,:3]/mask[:,:3].sum(-1,keepdim=True).clamp_min(1)
    distance=F.smooth_l1_loss(risk[:,3].float()/10,target[:,3]/10,reduction='none')*mask[:,3]
    brake=F.smooth_l1_loss(risk[:,4].float()/5,target[:,4]/5,reduction='none')*mask[:,4]
    valid=mask[:,:2]&mask[:,1:3]
    mono=.1*(F.relu(risk[:,:2].float()-risk[:,1:3].float())*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return torch.cat([collision,distance[:,None],brake[:,None],mono[:,None]],-1)

def flatten_grad(values,parameters):
    return torch.cat([(torch.zeros_like(p) if v is None else v).float().reshape(-1) for v,p in zip(values,parameters)])

def cosine_matrix(vectors):
    values=torch.stack(vectors).double();norm=values.norm(dim=-1)
    result=values@values.T/(norm[:,None]*norm[None,:]).clamp_min(1e-30)
    return [[float(result[i,j]) if norm[i]>0 and norm[j]>0 else None for j in range(len(norm))] for i in range(len(norm))]

def inside_box(xy,agent,margin=0.):
    delta=xy-np.asarray(agent['position']);c=np.cos(agent['heading']);s=np.sin(agent['heading'])
    local=delta@np.array([[c,-s],[s,c]])
    return (np.abs(local)<=np.asarray(agent['size'])/2+margin+1e-6).all(-1)

def describe(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None)

def paired_train_indices(cache,negative_limit=1000):
    positive=torch.nonzero(cache['risk_mask'][:,2]&(cache['risk_target'][:,2]>.5)).flatten()
    negative=torch.nonzero(cache['risk_mask'][:,2]&(cache['risk_target'][:,2]<=.5)).flatten()
    order=torch.randperm(len(negative),generator=torch.Generator().manual_seed(42))
    return sorted(torch.cat([positive,negative[order[:negative_limit]]]).tolist())

def input_evidence(record,item):
    """Uses GT boxes only to audit attribution, never as a deployed model input."""
    item={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in item.items()}
    agents=sorted(record['agents'],key=lambda a:(np.linalg.norm(a['position']),str(a['id'])))
    sources=set(record['local_risk']['pcol_source_agent_ids'][2]);current=item['radar'][-1]
    xy=item['radar_cartesian'][-1];valid=item['radar_doppler_valid'][-1]
    power=current[0];doppler=current[1];los=item['radar_los'][-1]
    output=[];source_mask=np.zeros(power.shape,bool)
    for j,agent in enumerate(agents):
        inside=inside_box(xy,agent);supported=inside&valid&(power>0)
        ring=inside_box(xy,agent,2.)&~inside
        selected=supported
        expected=((np.asarray(agent['velocity'])-item['radar_sensor_velocity'][-1])*los).sum(-1)
        weights=power[selected];n=int(selected.sum());synchronous=abs(float(item['time_offsets_s'][-1]))<=1e-6
        residual=float(np.average(np.abs(doppler[selected]-expected[selected]),weights=weights)) if n and synchronous else None
        observed=float(item['agent_radial_observed'][j]) if item['agent_radial_mask'][j] else None
        radial_gt=float(item['agent_state'][j,2]) if synchronous else None
        t=np.asarray(record['future_times_s']);positions=np.asarray(agent['future_xy']);fm=np.asarray(agent['future_valid'],bool)
        cv=np.asarray(agent['position'])+t[:,None]*np.asarray(agent['velocity'])
        deviation=float(np.linalg.norm(positions[-1]-cv[-1])) if fm[-1] else None
        if str(agent['id']) in sources:source_mask|=inside
        output.append(dict(id=str(agent['id']),critical=str(agent['id']) in sources,
            position=agent['position'],velocity=agent['velocity'],range_m=float(np.linalg.norm(agent['position'])),
            cells=int(inside.sum()),valid_positive_power_cells=n,
            valid_fraction=float(supported.sum()/inside.sum()) if inside.any() else None,
            mean_log_power=float(np.log1p(power[inside]).mean()) if inside.any() else None,
            box_vs_2m_ring_power_db=float(10*np.log10(max(float(power[inside].mean()),1e-30)/max(float(power[ring].mean()),1e-30))) if inside.any() and ring.any() else None,
            observed_radial_mps=observed,gt_projected_radial_mps=radial_gt,
            association_abs_error_mps=abs(observed-radial_gt) if observed is not None and radial_gt is not None else None,
            cell_weighted_radial_abs_error_mps=residual,future3_deviation_from_current_cv_m=deviation,
            future3_valid=bool(fm[-1])))
    # Privileged current-state CV diagnostic: no future GT is used for this forecast.
    times=np.arange(0,3.0001,.05);ex,ey=nominal_rollout(record['ego']['velocity'],record['ego']['yaw_rate'],times)
    cv_hit=False
    for agent in agents:
        ax=np.asarray(agent['position'])+times[:,None]*np.asarray(agent['velocity'])
        distance=box_distances(ex,ey,np.asarray(record['ego']['box_size']),ax,
            np.full_like(times,agent['heading']),np.asarray(agent['size']))
        cv_hit=cv_hit or bool((distance<.5).any())
    return dict(sample_id=record['sample_id'],episode_id=record['episode_id'],
        synchronous_latest=abs(float(item['time_offsets_s'][-1]))<=1e-6,
        time_offsets_s=item['time_offsets_s'].tolist(),critical_agent_ids=sorted(sources),agents=output,
        critical_box_area_fraction=float(source_mask.mean()),cv_gt_current_state_pcol3=int(cv_hit)),source_mask

def fit_probe(train_features,val_features,train_y,val_y,seed=42,nonlinear=False):
    """One fixed diagnostic fit; all scaling and threshold selection use train only."""
    x=train_features.float();v=val_features.float();mean=x.mean(0);std=x.std(0).clamp_min(.05)
    x=((x-mean)/std).clamp(-20,20);v=((v-mean)/std).clamp(-20,20)
    torch.manual_seed(seed)
    network=(torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.GELU(),torch.nn.Linear(64,1))
        if nonlinear else torch.nn.Linear(x.shape[1],1))
    optimizer=torch.optim.Adam(network.parameters(),lr=.01 if nonlinear else .03)
    y=train_y.float().view(-1,1);positive_weight=(1-y.mean())/y.mean().clamp_min(1e-6)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(network(x),y,pos_weight=positive_weight)
        loss=loss+.001*sum(p.square().mean() for p in network.parameters())
        loss.backward();optimizer.step()
    with torch.no_grad():pt=network(x).flatten().sigmoid();pv=network(v).flatten().sigmoid()
    # Matched train false-positive operating point; not a validation-tuned cutoff.
    threshold=float(torch.quantile(pt[train_y==0],.99))
    def measured(pred,truth):
        out=binary_metrics(pred.tolist(),truth.tolist());chosen=pred>=threshold;positive=truth.bool()
        tp=int((chosen&positive).sum());fp=int((chosen&~positive).sum())
        out.update(tp=tp,fp=fp,recall=tp/int(positive.sum()),precision=tp/max(tp+fp,1))
        return out
    return dict(train=measured(pt,train_y),val=measured(pv,val_y),threshold_from_train_negative_q99=threshold,
        dimensions=x.shape[1],seed=seed,steps=120,positive_weight=float(positive_weight)),pv

def fit_motion_probe(train_features,val_features,train_motion,val_motion):
    """Read current nearest-agent vx/vy/relative radial velocity; no future target."""
    x=train_features.float();v=val_features.float();mean=x.mean(0);std=x.std(0).clamp_min(.05)
    x=((x-mean)/std).clamp(-20,20);v=((v-mean)/std).clamp(-20,20)
    torch.manual_seed(42);network=torch.nn.Linear(x.shape[1],3);optimizer=torch.optim.Adam(network.parameters(),lr=.03)
    target=train_motion['target'];mask=train_motion['mask'];vt=val_motion['target'];vm=val_motion['mask']
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True);prediction=network(x)
        loss=F.smooth_l1_loss(prediction[mask],(target/10)[mask])+.001*sum(p.square().mean() for p in network.parameters())
        loss.backward();optimizer.step()
    with torch.no_grad():pt=network(x)*10;pv=network(v)*10
    result={}
    for j,name in enumerate(['vx','vy','relative_radial']):
        baseline=target[mask[:,j],j].mean();known=vm[:,j]
        result[name]=dict(train_samples=int(mask[:,j].sum()),val_samples=int(known.sum()),
            train_mae_mps=float((pt[mask[:,j],j]-target[mask[:,j],j]).abs().mean()),
            val_mae_mps=float((pv[known,j]-vt[known,j]).abs().mean()),
            train_mean_baseline_val_mae_mps=float((vt[known,j]-baseline).abs().mean()))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['base-experiment','output','assets']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True);args.assets.mkdir(parents=True,exist_ok=True)
    assert not (root/'results.json').exists(),'Do not overwrite completed diagnosis'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
    device='cuda:0';torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.12,0)
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    done=threading.Event()
    def status(phase,**kw):
        value=dict(state='running',phase=phase,pid=os.getpid(),**kw);write_json(root/'status.json',value);print(json.dumps(value),flush=True)
    def guard():
        while not done.wait(10):
            if torch.cuda.mem_get_info()[0]<8*1024**3:
                write_json(root/'resource_guard.json',dict(pid=os.getpid(),reason='Self exit below 8GiB free VRAM'))
                os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=guard,daemon=True).start()
    try:
        status('loading_grounder_only')
        old=json.loads((args.base_experiment/'comparison.json').read_text());protocol=old['protocol']
        assert source_fingerprint()==protocol['source_files']
        assert file_digest(protocol['checkpoint'])==protocol['checkpoint_sha256']
        assert file_digest(protocol['manifest'])==protocol['manifest_sha256']
        checkpoint=torch.load(protocol['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        grounder=RadarVLAGrounder(ModelConfig(**checkpoint['protocol']['config']['model'])).to(device)
        grounder.load_state_dict(checkpoint['grounder']);grounder.eval();del checkpoint
        initial={k:v.detach().cpu().clone() for k,v in grounder.state_dict().items()}
        before_source=source_fingerprint()
        run_protocol=dict(base=protocol,script_sha256=file_digest(__file__),smoke=args.smoke,
            probes='fixed train-only scaling; 120 steps; train q99 negative threshold; linear all stages, MLP tokens; seeds42/43/44',
            motion_probe='nearest current agent vx/vy/relative radial velocity; train-only standardization; linear120steps seed42',
            model_updated=False,planner_loaded=False,test_evaluated=False,gt_box_evidence_is_privileged=True,
            gradient_precision='FP32 loss arithmetic on original BF16 forward; component sum/derivatives checked against canonical FP32 loss',
            paired_raw_training='all Pcol3 positives + seed42 sample of1000 negatives; same subset across stages; full val',
            full_cached_training='all6728 train samples for cached-feature probes and loss-gradient decomposition',
            resources=dict(cpu_threads=1,memory_fraction=.12,start_free_gib=32,runtime_self_exit_free_gib=8))
        write_json(root/'protocol.json',run_protocol)
        feature_cache=Path(old['variants']['physical_only']['checkpoint']).parent.parent/'cache'
        caches={}
        for split in ['train','val']:
            chunks=[torch.load(p,map_location='cpu',weights_only=True) for p in sorted(feature_cache.glob(split+'_*.pt'))]
            assert all(c['source_files']==before_source and c['checkpoint_sha256']==protocol['checkpoint_sha256']
                and c['manifest_sha256']==protocol['manifest_sha256'] for c in chunks)
            caches[split]=dict(rows=[r for c in chunks for r in c['rows']])
            for key in ['radar_tokens','risk_target','risk_mask','risk','risk_logits','mode_logits','oracle_mode_logits']:
                caches[split][key]=torch.cat([c[key] for c in chunks])
        val=caches['val'];complete=val['risk_mask'].all(-1);short=torch.tensor([r['teacher_mode']=='short' for r in val['rows']])
        predshort=val['mode_logits'].argmax(-1)==0
        groups={r['sample_id']:('miss' if short[i] and not predshort[i] else 'hit' if short[i] else 'false_alarm' if predshort[i] else 'safe')
            for i,r in enumerate(val['rows']) if complete[i]}
        misses=[r for r in val['rows'] if groups.get(r['sample_id'])=='miss']
        assert len(misses)==36
        chosen={r['sample_id'] for r in val['rows'] if groups.get(r['sample_id']) in ['miss','hit','false_alarm']}
        # Safe controls nearest in ego speed / minimum current agent range, same episode.
        status('loading_validated_train_val_datasets')
        datasets={s:RadarDataset(protocol['manifest'],s,max_agents=None) for s in ['train','val']}
        for ds in datasets.values():ds.records=[r for r in ds.records if stage_available(r,'sft')]
        assert not {r['episode_id'] for r in datasets['train'].records}&{r['episode_id'] for r in datasets['val'].records}
        records={r['sample_id']:r for r in datasets['val'].records};safe_ids=set()
        for miss in misses:
            record=records[miss['sample_id']]
            candidates=[r for r in datasets['val'].records if r['episode_id']==record['episode_id'] and groups.get(r['sample_id'])=='safe' and r['sample_id'] not in safe_ids]
            speed=np.linalg.norm(record['ego']['velocity']);distance=min([np.linalg.norm(a['position']) for a in record['agents']] or [80])
            if candidates:
                def cost(r):return abs(np.linalg.norm(r['ego']['velocity'])-speed)/5+abs(min([np.linalg.norm(a['position']) for a in r['agents']] or [80])-distance)/20
                safe_ids.add(min(candidates,key=cost)['sample_id'])
        chosen|=safe_ids
        write_json(root/'cohort.json',dict(groups={k:sum(v==k for v in groups.values()) for k in ['miss','hit','false_alarm','safe']},
            chosen_ids=sorted(chosen),missed_ids=[r['sample_id'] for r in misses],safe_control_ids=sorted(safe_ids),
            oracle_recoverable=sum(int(val['oracle_mode_logits'][i].argmax())==0 for i,r in enumerate(val['rows']) if groups.get(r['sample_id'])=='miss'),
            missed_episodes=sorted({r['episode_id'] for r in misses})))
        captures={}
        def encoder_hook(module,inputs,output):captures['encoder']=output
        def kv_hook(module,inputs):captures['kv']=inputs[1]
        def raw_hook(module,inputs,output):captures['raw']=output
        def risk_tokens_hook(module,inputs):captures['risk_tokens']=inputs[1]
        hooks=[grounder.encoder.register_forward_hook(encoder_hook),grounder.temporal_attention.register_forward_pre_hook(kv_hook),
            grounder.risk_head.register_forward_hook(raw_hook),grounder.risk_attention.register_forward_pre_hook(risk_tokens_hook)]
        features={};motion={};evidence=[];grad_rows=[];ablation_rows=[];exact=0
        train_indices=paired_train_indices(caches['train'])
        write_json(root/'paired_train_sampling.json',dict(seed=42,negative_limit=1000,
            sample_ids=[caches['train']['rows'][i]['sample_id'] for i in train_indices],
            positive_samples=sum(float(caches['train']['risk_target'][i,2])>.5 for i in train_indices)))
        for split,ds in datasets.items():
            cache=caches[split];assert [r['sample_id'] for r in ds.records]==[r['sample_id'] for r in cache['rows']]
            indices=train_indices if split=='train' else list(range(len(ds)))
            if args.smoke:indices=indices[:2] if split=='train' else [i for i,r in enumerate(ds.records) if r['sample_id'] in chosen][:2]
            packets={k:[] for k in ['ego','raw_grid','cnn_before_pool','positioned_kv','tokens','risk_feature','risk_values']}
            motion_values=[];motion_masks=[]
            for count,i in enumerate(indices):
                item=ds[i];batch=to_device(collate_batch([item]),device)
                if bool(item['agent_mask'].any()):
                    motion_values.append(item['agent_state'][0,[3,4,2]].float().cpu())
                    motion_masks.append(item['agent_state_mask'][0,[3,4,2]].bool().cpu())
                else:motion_values.append(torch.zeros(3));motion_masks.append(torch.zeros(3,dtype=torch.bool))
                with torch.no_grad(),autocast_context(device,'bfloat16'):
                    prediction=grounder(batch)
                    assert torch.equal(prediction['risk'].float().cpu(),cache['risk'][i:i+1]),f'Risk cache mismatch {split} {i}'
                    assert torch.equal(prediction['radar_tokens'].cpu(),cache['radar_tokens'][i:i+1]),f'Token cache mismatch {split} {i}'
                    exact+=1;b,t,_,r,a=batch['radar'].shape
                    normalized=grounder.normalize_radar(batch['radar']);valid=batch['radar_doppler_valid']
                    normalized[:,:,1]=torch.where(valid,normalized[:,:,1],0.)
                    rawgrid=F.adaptive_avg_pool2d(torch.cat([normalized,valid[:,:,None].to(normalized.dtype)],2).reshape(t,3,r,a),(8,8)).flatten()
                    ego=torch.cat([batch['ego_velocity'],batch['ego_acceleration'],batch['ego'][:,2:3]],-1)/batch['ego'].new_tensor([15,15,5,5,1])
                    cnn=F.adaptive_avg_pool2d(captures['encoder'],(2,2)).flatten()
                    kv=captures['kv'].reshape(1,t,8,8,64).permute(0,1,4,2,3).reshape(t,64,8,8)
                    kv=F.adaptive_avg_pool2d(kv,(2,2)).flatten()
                    tokens=prediction['radar_tokens'][:,:-1]
                    query=grounder.risk_query[None]
                    attended=grounder.risk_attention(query,tokens,tokens,need_weights=False)[0][:,0]
                    risk_feature=torch.cat([attended,prediction['radar_tokens'][:,-1]],-1).flatten()
                    values=dict(ego=ego.flatten(),raw_grid=torch.cat([rawgrid,ego.flatten()]),
                        cnn_before_pool=torch.cat([cnn,ego.flatten()]),positioned_kv=torch.cat([kv,ego.flatten()]),
                        tokens=prediction['radar_tokens'].flatten(),risk_feature=risk_feature,risk_values=prediction['risk'].flatten())
                    for key,value in values.items():packets[key].append(value.float().cpu())
                if split=='val' and ds.records[i]['sample_id'] in chosen:
                    record=ds.records[i];audit,source_mask=input_evidence(record,item);audit['group']=groups[record['sample_id']]
                    audit['risk_target']=cache['risk_target'][i].tolist();audit['baseline_risk']=cache['risk'][i].tolist();evidence.append(audit)
                    # Interventions change a single input component; GT fields never enter Grounder forward.
                    with torch.no_grad(),autocast_context(device,'bfloat16'):
                        ablations={}
                        for name in ['zero_doppler','current_frame_only']:
                            changed=dict(batch)
                            if name=='zero_doppler':
                                changed['radar']=batch['radar'].clone();changed['radar'][:,:,1]=0.
                            else:
                                for key in ['radar','time_offsets_s','radar_cartesian','radar_los','radar_pose','radar_sensor_velocity','radar_doppler_valid']:
                                    changed[key]=batch[key][:,-1:]
                            ablations[name]=grounder(changed)['risk'][0].float().cpu().tolist()
                    ablation_rows.append(dict(sample_id=record['sample_id'],group=audit['group'],baseline=cache['risk'][i].tolist(),**ablations))
                    # Original weights; gradient inspection only, no optimizer.
                    batch['radar']=batch['radar'].detach().requires_grad_(True)
                    with autocast_context(device,'bfloat16'):
                        prediction=grounder(batch);raw=captures['raw'];components=component_losses(prediction['risk'],raw,batch['risk_target'],batch['risk_mask'])
                    token=captures['risk_tokens'];parameters=[p for n,p in grounder.named_parameters() if is_risk_parameter(n)]
                    named=list(grounder.named_parameters());all_params=[p for _,p in named]
                    vectors=[];per=[]
                    for j,name in enumerate(LOSS_NAMES):
                        gradients=torch.autograd.grad(components[0,j],all_params+[batch['radar'],token,raw],retain_graph=True,allow_unused=True)
                        block={k:0. for k in ['encoder','temporal_attention','risk','ego_encoder']};head=[]
                        for (pname,param),grad in zip(named,gradients):
                            if grad is not None:
                                blockname='risk' if is_risk_parameter(pname) else pname.split('.')[0]
                                if blockname in block:block[blockname]+=float(grad.float().square().sum())
                            if is_risk_parameter(pname):head.append(torch.zeros_like(param) if grad is None else grad)
                        headvector=torch.cat([g.float().flatten() for g in head]);vectors.append(headvector.detach().cpu())
                        sensor_grad,token_grad,raw_grad=gradients[-3:]
                        latest=sensor_grad[0,-1].float().abs().detach().cpu().numpy() if sensor_grad is not None else np.zeros((2,*source_mask.shape))
                        sums=latest.sum(axis=(1,2));inside=latest[:,source_mask].sum(axis=1)
                        per.append(dict(loss=name,value=float(components[0,j]),parameter_gradient_norm={k:v**.5 for k,v in block.items()},
                            raw_output_gradient=raw_grad[0].float().detach().cpu().tolist() if raw_grad is not None else [0.]*5,
                            token_gradient_norm=float(token_grad.float().norm()) if token_grad is not None else 0.,
                            input_gradient_l1_per_channel=sums.tolist(),critical_box_input_gradient_fraction=(inside/np.maximum(sums,1e-30)).tolist()))
                    # Sensitivity of Pcol3 itself separates network sensitivity from loss scaling.
                    sensor_grad=torch.autograd.grad(prediction['risk'][0,2],batch['radar'],retain_graph=False)[0]
                    latest=sensor_grad[0,-1].float().abs().cpu().numpy();sums=latest.sum(axis=(1,2));inside=latest[:,source_mask].sum(axis=1)
                    grad_rows.append(dict(sample_id=record['sample_id'],group=audit['group'],losses=per,
                        head_gradient_cosine=cosine_matrix(vectors),pcol3_input_gradient_l1=sums.tolist(),
                        pcol3_critical_box_gradient_fraction=(inside/np.maximum(sums,1e-30)).tolist(),
                        critical_box_area_fraction=float(source_mask.mean())))
                    if len(evidence)<=2 or (audit['group']=='miss' and audit['episode_id'] not in {e['episode_id'] for e in evidence[:-1] if e['group']=='miss'}):
                        render_case(root,record,item,audit)
                    grounder.zero_grad(set_to_none=True)
                if (count+1)%100==0:status('extracting_and_auditing',split=split,done=count+1,total=len(indices))
            features[split]={k:torch.stack(v) for k,v in packets.items()}
            motion[split]=dict(target=torch.stack(motion_values),mask=torch.stack(motion_masks))
            torch.save(dict(features=features[split],motion=motion[split],sample_ids=[ds.records[i]['sample_id'] for i in indices],source_files=before_source),args.assets/f'{split}_features.pt')
        for hook in hooks:hook.remove()
        write_json(root/'input_evidence.json',evidence);write_json(root/'case_gradients.json',grad_rows);write_json(root/'input_ablations.json',ablation_rows)
        if args.smoke:
            assert all(torch.equal(v.cpu(),initial[k]) for k,v in grounder.state_dict().items())
            write_json(root/'status.json',dict(state='complete',scope='smoke',exact_matches=exact));return
        status('full_train_loss_gradient_decomposition')
        gradient_summary=training_gradients(grounder,caches['train'],device)
        write_json(root/'training_gradients.json',gradient_summary)
        full_features=dict(tokens=caches['train']['radar_tokens'].float().flatten(1),risk_values=caches['train']['risk'].float())
        attended=[]
        with torch.no_grad(),autocast_context(device,'bfloat16'):
            for start in range(len(caches['train']['rows'])):
                tokens=caches['train']['radar_tokens'][start:start+1].to(device)
                query=grounder.risk_query[None].expand(len(tokens),-1,-1)
                read=grounder.risk_attention(query,tokens[:,:-1],tokens[:,:-1],need_weights=False)[0][:,0]
                attended.append(torch.cat([read,tokens[:,-1]],-1).float().cpu())
        full_features['risk_feature']=torch.cat(attended)
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in grounder.state_dict().items())
        del grounder;torch.cuda.empty_cache()
        status('train_only_diagnostic_probes_on_cpu')
        probes={};predictions={};train_y=caches['train']['risk_target'][train_indices,2];val_y=val['risk_target'][:,2]
        for stage in features['train']:
            probes[stage]=[]
            for seed in [42,43,44]:
                measured,pv=fit_probe(features['train'][stage],features['val'][stage],train_y,val_y,seed)
                selected=[i for i,r in enumerate(val['rows']) if groups.get(r['sample_id'])=='miss']
                measured['missed_36_recalled']=int((pv[selected]>=measured['threshold_from_train_negative_q99']).sum())
                probes[stage].append(measured)
                if seed==42:predictions[stage]=pv.tolist()
            status('probe_finished',stage=stage)
            write_json(root/'probe_progress.json',probes)
        probes['tokens_mlp']=[]
        for seed in [42,43,44]:
            measured,pv=fit_probe(features['train']['tokens'],features['val']['tokens'],train_y,val_y,seed,True)
            selected=[i for i,r in enumerate(val['rows']) if groups.get(r['sample_id'])=='miss']
            measured['missed_36_recalled']=int((pv[selected]>=measured['threshold_from_train_negative_q99']).sum())
            probes['tokens_mlp'].append(measured)
            if seed==42:predictions['tokens_mlp']=pv.tolist()
        for stage in ['tokens','risk_feature','risk_values','tokens_mlp']:
            name='full_train_'+stage;probes[name]=[]
            for seed in [42,43,44]:
                feature='tokens' if stage=='tokens_mlp' else stage
                measured,pv=fit_probe(full_features[feature],features['val'][feature],caches['train']['risk_target'][:,2],val_y,seed,stage=='tokens_mlp')
                selected=[i for i,r in enumerate(val['rows']) if groups.get(r['sample_id'])=='miss']
                measured['missed_36_recalled']=int((pv[selected]>=measured['threshold_from_train_negative_q99']).sum())
                probes[name].append(measured)
                if seed==42:predictions[name]=pv.tolist()
            status('probe_finished',stage=name);write_json(root/'probe_progress.json',probes)
        write_json(root/'probes.json',probes)
        motion_probes={}
        for stage in features['train']:
            motion_probes[stage]=fit_motion_probe(features['train'][stage],features['val'][stage],motion['train'],motion['val'])
        write_json(root/'motion_probes.json',motion_probes)
        with (root/'probe_predictions.jsonl').open('w') as stream:
            for i,row in enumerate(val['rows']):stream.write(json.dumps(dict(sample_id=row['sample_id'],episode_id=row['episode_id'],
                pcol3_target=float(val_y[i]),baseline_pcol3=float(val['risk'][i,2]),group=groups.get(row['sample_id'],'incomplete'),
                probes={k:v[i] for k,v in predictions.items()}))+'\n')
        assert source_fingerprint()==before_source
        assert file_digest(protocol['checkpoint'])==protocol['checkpoint_sha256']
        assert file_digest(protocol['manifest'])==protocol['manifest_sha256']
        write_json(root/'results.json',dict(exact_cache_matches=exact,samples={s:len(c['rows']) for s,c in caches.items()},
            raw_extracted_samples={s:len(v['tokens']) for s,v in features.items()},paired_train_samples=len(train_indices),
            model_unchanged=True,source_unchanged=True,checkpoint_unchanged=True,manifest_unchanged=True,
            inspected_cases=len(evidence),probe_results=probes,source_files=before_source,
            test_evaluated=False,planner_loaded=False,full_trajectory_evaluated=False))
        write_json(root/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as error:
        write_json(root/'status.json',dict(state='failed',pid=os.getpid(),error=str(error)));raise
    finally:done.set()

def training_gradients(grounder,cache,device):
    parameters=[p for n,p in grounder.named_parameters() if is_risk_parameter(n)]
    sizes=sum(p.numel() for p in parameters);sums={g:torch.zeros(6,sizes) for g in ['positive','negative']}
    losses={g:torch.zeros(6) for g in sums};counts={g:0 for g in sums}
    max_replay_error=0.
    for start in range(0,len(cache['rows']),128):
        sl=slice(start,start+128);tokens=cache['radar_tokens'][sl].to(device);target=cache['risk_target'][sl].to(device);mask=cache['risk_mask'][sl].to(device)
        saved=[]
        hook=grounder.risk_head.register_forward_hook(lambda m,a,o:saved.append(o))
        with autocast_context(device,'bfloat16'):
            output=grounder.risk_from_context(tokens[:,:-1],tokens[:,-1]);components=component_losses(output['risk'],saved[0],target,mask)
        hook.remove()
        max_replay_error=max(max_replay_error,float((output['risk'].float().cpu()-cache['risk'][sl]).abs().max()))
        positive=target[:,2]>.5
        for name,selected in [('positive',positive),('negative',~positive)]:
            counts[name]+=int(selected.sum());losses[name]+=components[selected].detach().float().sum(0).cpu()
            for j in range(6):
                gradients=torch.autograd.grad(components[selected,j].sum(),parameters,retain_graph=True,allow_unused=True)
                sums[name][j]+=flatten_grad(gradients,parameters).detach().cpu()
    total=counts['positive']+counts['negative'];aggregate=(sums['positive']+sums['negative'])/total
    output=dict(loss_names=LOSS_NAMES,samples=total,normalization='mean canonical batch-one risk loss; full natural train distribution; no planner loss',
        cached_head_gradient_batch_size=128,max_replayed_risk_abs_error_vs_batch_one_cache=max_replay_error,
        global_gradient_norm=aggregate.norm(dim=-1).tolist(),global_mean_loss=((losses['positive']+losses['negative'])/total).tolist(),
        global_cosine=cosine_matrix(list(aggregate)),groups={})
    for group in sums:
        means=sums[group]/counts[group]
        output['groups'][group]=dict(samples=counts[group],mean_loss=(losses[group]/counts[group]).tolist(),
            conditional_mean_gradient_norm=means.norm(dim=-1).tolist(),
            contribution_to_global_gradient_norm=(sums[group]/total).norm(dim=-1).tolist(),cosine=cosine_matrix(list(means)))
    output['positive_negative_same_loss_cosine']=[cosine_matrix([sums['positive'][i],sums['negative'][i]])[0][1] for i in range(6)]
    return output

def render_case(root,record,item,audit):
    item={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in item.items()}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from radar_vla.geometry import _corners
    folder=root/'case_figures';folder.mkdir(exist_ok=True)
    xy=item['radar_cartesian'][-1];power=np.log1p(item['radar'][-1,0]);valid=item['radar_doppler_valid'][-1]
    region=(np.abs(xy[...,0])<60)&(np.abs(xy[...,1])<35)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,values,title in zip(axes,[power,np.where(valid,item['radar'][-1,1],np.nan)],['Current log(1+power)','Signed radial Doppler (valid cells)']):
        image=ax.scatter(xy[...,0][region],xy[...,1][region],c=values[region],s=3,cmap='viridis' if ax is axes[0] else 'coolwarm',
            vmin=None if ax is axes[0] else -15,vmax=None if ax is axes[0] else 15)
        fig.colorbar(image,ax=ax)
        for agent in record['agents']:
            critical=str(agent['id']) in audit['critical_agent_ids'];color='red' if critical else 'gray'
            ax.add_patch(Polygon(_corners(agent['position'],agent['heading'],agent['size']),fill=False,edgecolor=color,linewidth=1.2))
            if critical:
                future=np.asarray(agent['future_xy']);mask=np.asarray(agent['future_valid'],bool)
                ax.plot(np.r_[agent['position'][0],future[mask,0]],np.r_[agent['position'][1],future[mask,1]],color=color,label='GT critical future (audit only)')
        ax.set(xlim=(-10,60),ylim=(-30,30),xlabel='Current ego x / m',ylabel='Current ego y / m',title=title);ax.set_aspect('equal')
    fig.suptitle(record['sample_id']+' / '+audit['group']+' / GT boxes for attribution only',fontsize=9)
    fig.savefig(folder/(record['sample_id']+'.png'),dpi=130);plt.close(fig)

if __name__=='__main__':main()
