"""Whole-episode stratification with a frozen existing test set."""
import math
import re
import warnings
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def allocate_episode_splits(records, labels, tolerance=.005):
    groups={}
    for r in records:
        g=groups.setdefault(r['episode_id'],dict(records=[],split=r['split']))
        if g['split']!=r['split']:raise ValueError('Source episode already crosses splits')
        g['records'].append(r)
    keys=sorted(groups);n=len(keys);splits=['train','val','test'];ratios=np.array([.7,.15,.15])
    categories=sorted({r.get('category','unknown') for r in records})
    towns=sorted({(re.search(r'Town\d+',r['scene_id']) or ['unknown'])[0] for r in records})
    feature_names=['samples','positive_0.5s','positive_1s','positive_2s']+['category/'+x for x in categories]+['town/'+x for x in towns]
    matrix=np.zeros((n,len(feature_names)))
    for i,key in enumerate(keys):
        for r in groups[key]['records']:
            row=labels[r['sample_id']];matrix[i,0]+=1
            matrix[i,1:4]+=np.asarray(row['pcol'])*np.asarray(row['pcol_mask'],bool)
            matrix[i,4+categories.index(r.get('category','unknown'))]+=1
            town=(re.search(r'Town\d+',r['scene_id']) or ['unknown'])[0]
            matrix[i,4+len(categories)+towns.index(town)]+=1
    totals=matrix.sum(0);d=len(feature_names);size=3*n+3*d
    objective=np.zeros(size);lower=np.zeros(size);upper=np.full(size,np.inf);upper[:3*n]=1
    # Soft L1 balance; strong sample balance and explicit rare-event coverage.
    weights=np.r_[100.,3.,3.,3.,np.full(d-4,.25)]/np.maximum(totals,1)
    for s in range(3):objective[3*n+s*d:3*n+(s+1)*d]=weights
    for i,key in enumerate(keys):
        old=splits.index(groups[key]['split'])
        for s in range(3):objective[i*3+s]=0.0001*(s!=old)+1e-9*(i*3+s)
        if old==2:lower[i*3+2]=upper[i*3+2]=1
        else:upper[i*3+2]=0
    rows=[];lo=[];hi=[]
    def constraint(values,a,b):rows.append(values);lo.append(a);hi.append(b)
    for i in range(n):
        row=np.zeros(size);row[i*3:i*3+3]=1;constraint(row,1,1)
    for s in range(3):
        count=np.zeros(size);count[s:3*n:3]=matrix[:,0]
        constraint(count,ratios[s]*totals[0]-max(1,math.ceil(tolerance*totals[0])),ratios[s]*totals[0]+max(1,math.ceil(tolerance*totals[0])))
        for f in range(d):
            row=np.zeros(size);row[s:3*n:3]=matrix[:,f];row[3*n+s*d+f]=-1
            constraint(row,-np.inf,ratios[s]*totals[f])
            row=np.zeros(size);row[s:3*n:3]=-matrix[:,f];row[3*n+s*d+f]=-1
            constraint(row,-np.inf,-ratios[s]*totals[f])
        for f in (1,2,3):
            row=np.zeros(size);row[s:3*n:3]=matrix[:,f]
            mutable=[i for i,key in enumerate(keys) if groups[key]['split']!='test']
            minimum=max(1,int(matrix[mutable,f].max())) if s==0 else 1
            constraint(row,minimum,np.inf)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected')
        result=milp(objective,integrality=np.r_[np.ones(3*n),np.zeros(3*d)],bounds=Bounds(lower,upper),
            constraints=LinearConstraint(np.asarray(rows),np.asarray(lo),np.asarray(hi)),
            options=dict(time_limit=30.,mip_rel_gap=.001,threads=1,random_seed=42))
    if result.x is None:raise ValueError(f'No feasible episode split: {result.message}')
    allocation={key:splits[int(np.argmax(result.x[i*3:i*3+3]))] for i,key in enumerate(keys)}
    stats={s:dict(samples=0,episodes=0,positive_counts=[0,0,0],positive_episodes=[0,0,0]) for s in splits}
    for i,key in enumerate(keys):
        r=stats[allocation[key]];r['samples']+=int(matrix[i,0]);r['episodes']+=1
        for f in range(3):r['positive_counts'][f]+=int(matrix[i,1+f]);r['positive_episodes'][f]+=int(matrix[i,1+f]>0)
    for r in stats.values():r['fraction']=r['samples']/len(records)
    return allocation,dict(target_ratios=dict(zip(splits,ratios.tolist())),actual=stats,
        source_episodes=len(keys),frozen_original_test=True,solver_status=int(result.status),solver_message=result.message,
        label_horizons_s=[.5,1.,2.],note='Frame counts balanced within0.5pp for real cohort; entire episodes retained; rare events remain few episodes')
