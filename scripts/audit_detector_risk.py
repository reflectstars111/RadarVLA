"""Trace critical-agent detector failures and shared-encoder gradients, no updates."""
import json,os,signal,threading
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from radar_vla.data import RadarDataset,collate_batch
from radar_vla.detection import decode_detections
from radar_vla.detection_metrics import DetectionMetrics
from radar_vla.losses import loss_grounding
from radar_vla.model import RadarVLAGrounder,ModelConfig
from radar_vla.pipeline import autocast_context,to_device,write_json,source_fingerprint,file_digest

def cosine(vectors):
    v=torch.stack(vectors).double();norm=v.norm(dim=-1);result=v@v.T/(norm[:,None]*norm[None,:]).clamp_min(1e-30)
    return [[float(result[i,j]) if norm[i]>0 and norm[j]>0 else None for j in range(len(norm))] for i in range(len(norm))]

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path('/home/luxinlan/data/lxl_data/RadarVLA/analysis/risk_layer_diagnosis_20261019')
    out=root/'detector_audit';out.mkdir(exist_ok=True);assert not (out/'results.json').exists()
    protocol=json.loads((root/'protocol.json').read_text());base=protocol['base'];device='cuda:0'
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.08,0)
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    finish=threading.Event()
    def status(phase,**kw):
        value=dict(state='running',phase=phase,pid=os.getpid(),**kw);write_json(out/'status.json',value);print(json.dumps(value),flush=True)
    def guard():
        while not finish.wait(10):
            if torch.cuda.mem_get_info()[0]<8*1024**3:os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=guard,daemon=True).start()
    try:
        status('loading_same_grounder')
        assert source_fingerprint()==base['source_files']
        state=torch.load(base['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        grounder=RadarVLAGrounder(ModelConfig(**state['protocol']['config']['model'])).to(device)
        grounder.load_state_dict(state['grounder']);grounder.eval();del state
        initial={k:v.detach().cpu().clone() for k,v in grounder.state_dict().items()}
        assets=root.parents[1]/'model_runs/risk_layer_diagnosis_20261019'
        dataset=RadarDataset(assets/'fixed_cohort_metadata.jsonl','val',max_agents=None)
        reference={r['sample_id']:r for r in map(json.loads,(root/'reference_grounding_samples.jsonl').open())}
        evidence={r['sample_id']:r for r in json.loads((root/'early_input_evidence.json').read_text())['evidence']}
        named=[(n,p) for n,p in grounder.named_parameters() if n.startswith(('encoder.','temporal_attention.','radar_queries','position_encoder.','token_norm.','token_mlp.','output_norm.'))]
        parameters=[p for _,p in named];encoder_mask=torch.cat([torch.full((p.numel(),),n.startswith('encoder.'),dtype=torch.bool) for n,p in named])
        task_names=['risk_physical','detection_total','detection_objectness','detection_position','detection_velocity','detection_future']
        sums={};counts={};rows=[];metrics={g:DetectionMetrics(2.,'agent_v1') for g in ['miss','hit','false_alarm','safe']}
        for i,record in enumerate(dataset.records):
            item=dataset[i];batch=to_device(collate_batch([item]),device);audit=evidence[record['sample_id']];group=audit['group']
            with autocast_context(device,'bfloat16'):
                output=grounder(batch)
                risk=loss_grounding(dict(risk=output['risk'],risk_logits=output['risk_logits']),dict(risk_target=batch['risk_target'],risk_mask=batch['risk_mask']))
                without_risk={k:v for k,v in batch.items() if k not in ['risk_target','risk_mask']}
                detection=loss_grounding(output,without_risk)
            torch.testing.assert_close(output['risk'][0].float().cpu(),torch.tensor(reference[record['sample_id']]['risk']),rtol=0,atol=0)
            decoded=decode_detections(output,grounder.config,batch['future_times_s']);pred=decoded[0]['detections']
            metrics[group].update(decoded,batch)
            agents=sorted(record['agents'],key=lambda a:(np.linalg.norm(a['position']),str(a['id'])))
            gt=item['agent_state'].numpy();pairs=[]
            if pred and agents:
                distances=np.linalg.norm(np.asarray([d['position_m'] for d in pred])[:,None]-gt[None,:,:2],axis=-1)
                cost=np.where(distances<=2.,distances,(len(pred)+len(agents)+1)*3.)
                pi,gi=linear_sum_assignment(cost);pairs=[(int(p),int(g)) for p,g in zip(pi,gi) if distances[p,g]<=2.]
            matched={g:p for p,g in pairs}
            assert {str(a['id']) for j,a in enumerate(agents) if j in matched}=={str(r['agent_id']) for r in reference[record['sample_id']]['gt_objects'] if r['matched']}
            targets=torch.tensor(sorted(range(len(agents)),key=lambda j:tuple(gt[j].tolist()+[float(item['agent_class'][j]),float(item['agent_heading'][j])]+item['agent_size'][j].tolist())),device=device)
            assigned={}
            if len(targets):
                cost=torch.cdist(output['detection_anchors'][0].float(),batch['agent_state'][0,targets,:2].float())
                pi,gi=linear_sum_assignment(cost.detach().cpu().numpy());assigned={int(targets[int(g)]):int(p) for p,g in zip(pi,gi)}
            positions=output['detection_state'][0,:,:2].detach().cpu().numpy();scores=output['detection_logits'][0].float().sigmoid().detach().cpu().numpy()
            actors=[]
            for j,agent in enumerate(agents):
                distance=np.linalg.norm(positions-gt[j,:2],axis=-1);near=distance<=2.;best=float(scores[near].max()) if near.any() else None
                assigned_index=assigned[j];d=pred[matched[j]] if j in matched else None
                fm=item['agent_future_mask'][j].numpy().astype(bool);future=item['agent_future'][j].numpy()
                errors=np.linalg.norm(np.asarray(d['future_xy'])-future,axis=-1) if d else None
                is_critical=str(agent['id']) in audit['critical_agent_ids']
                actors.append(dict(id=str(agent['id']),critical=is_critical,matched=d is not None,
                    raw_best_score_within2m=best,raw_nearest_center_error_m=float(distance.min()),
                    assigned_score=float(scores[assigned_index]),assigned_center_error_m=float(distance[assigned_index]),
                    miss_stage=None if d else 'no_raw_center_within2m' if best is None else 'low_objectness' if best<.5 else 'NMS_or_one_to_one_competition',
                    center_error_m=float(np.linalg.norm(np.asarray(d['position_m'])-gt[j,:2])) if d else None,
                    velocity_error_mps=float(np.linalg.norm(np.asarray(d['velocity_mps'])-gt[j,3:5])) if d else None,
                    radial_error_mps=abs(d['radial_velocity_mps']-float(gt[j,2])) if d else None,
                    future_ade_m=float(errors[fm].mean()) if d and fm.any() else None,
                    future_fde_m=float(errors[-1]) if d and fm[-1] else None,
                    observed_radial_abs_error_mps=next(a['association_abs_error_mps'] for a in audit['agents'] if a['id']==str(agent['id']))))
            tasks=[risk['total'],detection['total'],detection['detection_objectness'],detection['detection_position'],detection['detection_velocity'],detection['detection_agent_future']]
            vectors=[]
            for loss in tasks:
                gs=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).float().flatten() for g,p in zip(gs,parameters)]).detach().cpu())
            enc=[v[encoder_mask] for v in vectors];temporal=[v[~encoder_mask] for v in vectors]
            if group not in sums:sums[group]=torch.zeros(len(tasks),len(vectors[0]));counts[group]=0
            sums[group]+=torch.stack(vectors);counts[group]+=1
            rows.append(dict(sample_id=record['sample_id'],episode_id=record['episode_id'],group=group,actors=actors,
                task_losses=[float(v.detach()) for v in tasks],encoder_gradient_norm=[float(v.norm()) for v in enc],
                temporal_gradient_norm=[float(v.norm()) for v in temporal],encoder_gradient_cosine=cosine(enc),temporal_gradient_cosine=cosine(temporal)))
            grounder.zero_grad(set_to_none=True)
            if (i+1)%10==0:status('critical_detector_and_gradient_replay',done=i+1,total=len(dataset))
        aggregate={}
        for group,total in sums.items():
            mean=total/counts[group]
            aggregate[group]=dict(samples=counts[group],encoder_mean_gradient_norm=mean[:,encoder_mask].norm(dim=-1).tolist(),
                encoder_mean_gradient_cosine=cosine(list(mean[:,encoder_mask])),
                temporal_mean_gradient_norm=mean[:,~encoder_mask].norm(dim=-1).tolist(),temporal_mean_gradient_cosine=cosine(list(mean[:,~encoder_mask])))
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in grounder.state_dict().items())
        assert source_fingerprint()==base['source_files']
        write_json(out/'samples.json',rows)
        write_json(out/'results.json',dict(task_names=task_names,groups=aggregate,detection_metrics={g:m.compute() for g,m in metrics.items()},
            exact_reference_risk_and_matched_flags=True,samples=len(rows),model_unchanged=True,source_unchanged=True,
            script_sha256=file_digest(__file__),source_files=base['source_files'],checkpoint_sha256=base['checkpoint_sha256'],
            architecture='shared3conv encoder -> dense detector(current stride4 spatial + mean tokens/Ego context); Risk reads8 radar tokens, not decoded boxes',
            no_optimizer=True,test_evaluated=False))
        write_json(out/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as error:
        write_json(out/'status.json',dict(state='failed',pid=os.getpid(),error=str(error)));raise
    finally:finish.set()

if __name__=='__main__':main()
