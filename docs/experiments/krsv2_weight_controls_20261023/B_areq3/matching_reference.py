"""Canonical 2m one-to-one matching reused from prior accepted validation audits."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def matched_objects(record, batch, detections):
    agents=sorted(record['agents'],key=lambda a:(np.linalg.norm(a['position']),str(a['id'])))
    gt=batch['agent_state'][0].cpu().numpy()[:len(agents)]
    source=set(map(str,record['local_risk']['pcol_source_agent_ids'][2]))
    pairs={}
    if detections and agents:
        distance=np.linalg.norm(np.asarray([d['position_m'] for d in detections])[:,None]-gt[None,:,:2],axis=-1)
        cost=np.where(distance<=2.,distance,(len(detections)+len(agents)+1)*3.)
        pi,gi=linear_sum_assignment(cost)
        pairs={int(g):detections[int(p)] for p,g in zip(pi,gi) if distance[p,g]<=2.}
    rows=[]
    for i,a in enumerate(agents):
        row=dict(id=str(a['id']),critical=str(a['id']) in source,matched=i in pairs,
            velocity_error_mps=None,radial_gt_error_mps=None,radial_observed_error_mps=None,future_ade_m=None)
        if i in pairs:
            d=pairs[i];sm=batch['agent_state_mask'][0,i].bool().cpu().numpy()
            if sm[3:5].all():row['velocity_error_mps']=float(np.linalg.norm(np.asarray(d['velocity_mps'])-gt[i,3:5]))
            if sm[2]:row['radial_gt_error_mps']=abs(float(d['radial_velocity_mps'])-float(gt[i,2]))
            if bool(batch['agent_radial_mask'][0,i]):
                row['radial_observed_error_mps']=abs(float(d['radial_velocity_mps'])-float(batch['agent_radial_observed'][0,i]))
            mask=batch['agent_future_mask'][0,i].bool().cpu().numpy()
            if mask.any():
                error=np.linalg.norm(np.asarray(d['future_xy'])-batch['agent_future'][0,i].cpu().numpy(),axis=-1)
                row['future_ade_m']=float(error[mask].mean())
        rows.append(row)
    return rows
