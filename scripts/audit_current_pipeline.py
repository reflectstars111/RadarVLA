"""Read-only raw/prepared audit; outputs are isolated under --output-dir."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from radar_vla import collection_adapter as adapter
from radar_vla.geometry import build_risk_labels
from radar_vla.road_segments import road_segments
from radar_vla.visual_diagnosis import _single_sample
from radar_vla.annotation_domain import box_intersects_annotation_domain


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def compact(r, offset):
    radar = r['sensors']['radar']; v = np.asarray(r['ego']['velocity']); w = r['ego']['yaw_rate']
    pose = np.asarray(radar['poses_current_ego'][-1]); sensor = np.asarray(radar['sensor_velocity_current_ego'][-1])
    corrected = v + w*np.array([-pose[1,3], pose[0,3]])
    return dict(sample_id=r['sample_id'], episode_id=r['episode_id'], scene_id=r['scene_id'],
        split=r['split'], offset=offset, carla_frame=r['carla_frame'], category=r['category'], weather=r['weather'],
        speed_mps=float(np.linalg.norm(v)), yaw_rate_radps=w, teacher_mode=r['teacher_mode'],
        collision_positive=any(a>.5 and m for a,m in zip(r['risk_label']['pcol'],r['risk_label']['pcol_mask'])),
        sensor_velocity_error_mps=float(np.linalg.norm(corrected-sensor)),
        categories=dict(Counter(a['category'] for a in r['agents'])), route_archive=r['route_archive'])


def read_at(path, offset):
    with path.open('rb') as f: f.seek(offset); return json.loads(f.readline())


def hazard_risk_labels(record):
    # areq>=3 requires complete nominal violation, already caught by known
    # dmin<=0.5. Braking-grid search cannot add a new hazardous actor here.
    # Do not use these abbreviated labels as training targets.
    return build_risk_labels(record, max_deceleration=0.)


def audit_exclusions(args, rows):
    out=args.output_dir
    # Audit excluded danger rows using the original full-world actors, preserving unknown futures.
    index_rows=[json.loads(l) for l in (args.raw_root/'dataset_index.jsonl').open()]
    lookup={(r['job_id'],r['carla_frame']):r for r in index_rows}
    source=Path(json.load(Path('analysis/rebalanced_v4/split_report.json').open())['source'])
    exclusions=[json.loads(l) for l in (source.parent/'exclusions.jsonl').open()]
    write(out/'exclusion_counts.json',dict(Counter(r['reason'] for r in exclusions)))
    dangerous=defaultdict(list)
    split_by_job={r['scene_id']:r['split'] for r in rows}
    for ex in exclusions:
        if 'cannot be attributed' in ex['reason'] and split_by_job.get(ex['job_id']) in ('train','val'):dangerous[ex['job_id']].append(ex)
    excluded=[]
    for job,group in dangerous.items():
        episode=args.raw_root/'jobs'/job/'episode';anchors=adapter._selected_anchors(episode/'anchors.jsonl',{e['carla_frame'] for e in group})
        for ex in group:
            source_row=lookup[(job,ex['carla_frame'])];frame=adapter._load_frame(episode,source_row['source_index']);ego=frame['ego'];origin=ego['pose']['position_m'][:2];yaw=ego['pose']['rotation_deg'][2]
            anchor=anchors[ex['carla_frame']];future=anchor['futures_by_actor_id'];visible={str(a['actor_id']) for a in json.load((episode/'frames'/f"{source_row['source_index']:06d}"/'agents.json').open())['agents']}
            selected={a['id'] for a in sorted([adapter._agent(a,frame,origin,yaw,future.get(str(a['actor_id'])),[4,9,14,19,24,29]) for a in frame['actors'] if str(a['actor_id']) in visible],key=lambda a:math.hypot(*a['position']))[:8]}
            base=dict(ego=dict(velocity=adapter._vector_to_ego(ego['velocity_world_mps'],yaw),yaw_rate=-math.radians(ego['angular_velocity_world_degps'][2]),box_size=[2*x for x in ego['bbox']['extent_m'][:2]]),future_times_s=[.5,1,1.5,2,2.5,3],tracking_coverage_valid=[True]*6,agent_supervision_available=True)
            hazards=[];unknown=0
            for actor in frame['actors']:
                if not adapter._could_affect_risk(actor,future.get(str(actor['actor_id'])),origin,math.hypot(*ego['velocity_world_mps'][:2]),ego['bbox']['extent_m']):continue
                a=adapter._agent(actor,frame,origin,yaw,future.get(str(actor['actor_id'])),[4,9,14,19,24,29]);label=hazard_risk_labels(dict(base,agents=[a],risk_agents=[a]))
                if not all(a['future_valid']):unknown+=1
                if any(v>=.5 and m for v,m in zip(label['pcol'][:2],label['pcol_mask'][:2])) or label['dmin_mask'] and label['dmin']<=.5 or label['areq_mask'] and label['areq']>=3:
                    metadata=json.load((episode/'metadata.json').open())
                    pose=adapter._radar_pose(frame,origin,yaw,metadata['sensors']['semantic_lidar']['x'])
                    hazards.append(dict(actor_id=a['id'],visible=a['id'] in visible,planner_selected=a['id'] in selected,type_id=actor['type_id'],
                        intersects_radar_bev=box_intersects_annotation_domain(a['position'],a['size'],a['heading'],
                            [metadata['radar_range_m'],-math.radians(53),math.radians(53)],pose)))
            excluded.append(dict(**ex,split=split_by_job[job],source_index=source_row['source_index'],hazards=hazards,actors_with_incomplete_future=unknown,
                note='per-agent counterfactual audit; full tracking coverage not re-inferred; no unknown is changed to safe'))
        write(out/'excluded_hazards.json',excluded);print('excluded audit',job,len(group),flush=True)
    print('EXCLUSION AUDIT COMPLETE',len(excluded),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--exclusions-only',action='store_true')
    args=parser.parse_args(); out=args.output_dir
    if args.exclusions_only:
        saved=json.load((out/'integrity.json').open())
        if saved['manifest_sha256']!=digest(args.manifest):raise ValueError('raw audit manifest digest changed')
        audit_exclusions(args,json.load((out/'train_val_inventory.json').open()))
        return
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; episodes={}; paths=set(); ids=set(); errors=[]
    with args.manifest.open('rb') as f:
        while True:
            offset=f.tell(); line=f.readline()
            if not line:break
            r=json.loads(line); ep=r['episode_id']; split=r['split']
            if r['sample_id'] in ids:errors.append('duplicate sample '+r['sample_id'])
            ids.add(r['sample_id'])
            if ep in episodes and episodes[ep]['split']!=split:errors.append('split leak '+ep)
            if ep not in episodes:
                mpath=Path(r['route_archive']).parent/'metadata.json'; m=json.load(mpath.open())
                episodes[ep]=dict(split=split,metadata=str(mpath),metadata_sha256=digest(mpath),
                    schema=m['schema'],agent_range_m=m['agent_label_range_m'],map_range_m=m['map_label_range_m'],radar_range_m=m['radar_range_m'])
                if m['episode_id']!=ep or m['agent_label_range_m']!=50 or m['map_label_range_m']!=50:errors.append('metadata mismatch '+ep)
            radar=r['sensors']['radar']
            for key in ('power','unfolded_doppler'):
                paths.update((args.manifest.parent/Path(v)).resolve() for v in radar[key])
            paths.add((args.manifest.parent/Path(radar['doppler_valid'])).resolve())
            paths.add(Path(r['route_archive']))
            if split!='test':rows.append(compact(r,offset))
    missing=[str(p) for p in paths if not p.is_file()]
    summary={}
    for split in ('train','val'):
        group=[r for r in rows if r['split']==split]; c=Counter()
        for r in group:c.update(r['categories'])
        summary[split]=dict(frames=len(group),episodes=len({r['episode_id'] for r in group}),
            slow_lt1=sum(r['speed_mps']<1 for r in group),turning_abs_yaw_ge_005=sum(abs(r['yaw_rate_radps'])>=.05 for r in group),
            modes=dict(Counter(r['teacher_mode'] for r in group)),classes=dict(c),
            positive_frames=sum(r['collision_positive'] for r in group),positive_episodes=len({r['episode_id'] for r in group if r['collision_positive']}),
            velocity_difference_gt_01=sum(r['sensor_velocity_error_mps']>.1 for r in group),
            max_velocity_difference_mps=max(r['sensor_velocity_error_mps'] for r in group))
    audit_ids=set(); infer_ids=set(); by_episode=defaultdict(list)
    for r in rows:by_episode[r['episode_id']].append(r)
    for group in by_episode.values():
        group.sort(key=lambda r:r['carla_frame'])
        audit_ids.update(group[round(f*(len(group)-1))]['sample_id'] for f in (0,.5,1))
        if group[0]['split']=='val':infer_ids.update(group[round(f*(len(group)-1))]['sample_id'] for f in (0,.25,.5,.75,1))
    for r in rows:
        if r['split']=='val' and r['collision_positive']:audit_ids.add(r['sample_id']);infer_ids.add(r['sample_id'])
        if r['categories'].get('pedestrian',0):audit_ids.add(r['sample_id'])
    audit_ids.update(r['sample_id'] for r in sorted(rows,key=lambda r:r['sensor_velocity_error_mps'],reverse=True)[:20])
    chosen=[r for r in rows if r['sample_id'] in audit_ids]
    with (out/'detection_manifest.jsonl').open('w') as f:
        for r in rows:
            if r['sample_id'] in infer_ids:f.write(json.dumps(read_at(args.manifest,r['offset']),ensure_ascii=False)+'\n')
    write(out/'audit_selection.json',chosen)
    write(out/'train_val_inventory.json',rows)
    write(out/'integrity.json',dict(manifest=str(args.manifest.resolve()),manifest_sha256=digest(args.manifest),samples=len(ids),
        episodes=episodes,unique_referenced_files=len(paths),missing=missing,errors=errors,summary=summary,
        inference_samples=len(infer_ids),audit_samples=len(chosen)))
    print('inventory complete',summary,flush=True)
    # Group anchor reads by episode, so multi-GB JSONL archives are scanned only once.
    samples=[]; types=Counter(); checks=Counter()
    for ep,group in by_episode.items():
        selected=[r for r in group if r['sample_id'] in audit_ids]
        if not selected:continue
        episode=Path(selected[0]['route_archive']).parent
        anchors=adapter._selected_anchors(episode/'anchors.jsonl',{r['carla_frame'] for r in selected})
        products=None
        for item in selected:
            r=read_at(args.manifest,item['offset']); radar=r['sensors']['radar']
            index=int(Path(radar['power'][-1]).name.split('_')[0]); frame=adapter._load_frame(episode,index)
            origin=frame['ego']['pose']['position_m'][:2];yaw=frame['ego']['pose']['rotation_deg'][2]
            raw_agents=json.load((episode/'frames'/f'{index:06d}'/'agents.json').open())['agents']
            anchor=anchors[r['carla_frame']]; amap={str(a['actor_id']):a for a in raw_agents}
            def delta(a,b):return float(np.max(np.abs(np.asarray(a,float)-np.asarray(b,float)))) if np.size(a) else 0.
            row=dict(sample_id=r['sample_id'],episode_id=ep,split=r['split'],raw_frame=str(episode/'frames'/f'{index:06d}'/'frame.json'),
                time_delta_s=abs(frame['timestamp_s']-r['timestamp_s']),frame_id_match=frame['carla_frame']==r['carla_frame'],
                visible_ids_match=set(amap)=={str(a['id']) for a in r['agents']},agent_errors={},type_ids=[])
            for a in r['agents']:
                raw=amap[str(a['id'])]; fresh=adapter._agent(raw,frame,origin,yaw,anchor['futures_by_actor_id'].get(str(a['id'])),[4,9,14,19,24,29])
                for k in ('position','velocity','size','heading','future_xy','future_valid','future30_xy','future30_valid'):
                    row['agent_errors'][k]=max(row['agent_errors'].get(k,0.),delta(a[k],fresh[k]))
                types[(raw['type_id'],a['category'])]+=1
                row['type_ids'].append(dict(actor_id=a['id'],type_id=raw['type_id'],category=a['category'],
                    wheels=raw.get('attributes',{}).get('number_of_wheels')))
            raw_map=json.load((episode/'frames'/f'{index:06d}'/'map.json').open()); roads=road_segments(raw_map)
            for road in roads:road['centerline_ego_m']=[[p[0],-p[1]] for p in road['centerline_ego_m']]
            row['roads_match']=roads==r['road_segments']
            ef,_,ev=adapter._future(anchor['futures_by_actor_id'][str(frame['ego']['actor_id'])],origin,yaw,[4,9,14,19,24,29])
            row['ego_future_error_m']=delta(ef,r['ego']['future_xy']);row['ego_future_masks_match']=ev==r['ego']['future_valid']
            sensor_x=json.load((episode/'metadata.json').open())['sensors']['semantic_lidar']['x']
            history=[adapter._load_frame(episode,i) for i in range(index-3,index+1)]
            row['history_frame_ids_match']=[x['carla_frame'] for x in history]==list(range(r['carla_frame']-3,r['carla_frame']+1))
            row['pose_error']=delta([adapter._radar_pose(x,origin,yaw,sensor_x) for x in history],radar['poses_current_ego'])
            row['corrected_sensor_velocity_error_mps']=delta([adapter._radar_sensor_velocity(x,yaw,sensor_x) for x in history],radar['sensor_velocity_current_ego'])
            row['history_time_error_s']=delta([x['timestamp_s']-r['timestamp_s'] for x in history],radar['time_offsets_s'])
            risk=build_risk_labels(r);row['risk_label_match']=risk==r['risk_label']
            sample=_single_sample(r)
            row['dataset_power_error']=delta(sample['radar'][-1,0].numpy(),np.load(radar['power'][-1]))
            row['dataset_unmasked_doppler_error']=delta(sample['radar'][-1,1].numpy(),np.load(radar['unfolded_doppler'][-1]))
            row['dataset_velocity_error']=delta(sample['ego_velocity'].numpy(),r['ego']['velocity'])
            row['dataset_future_error']=delta(sample['ego_future'].numpy(),r['ego']['future_xy'])
            row['dataset_domain_known']=bool(sample['detection_annotation_domain_mask'])
            row['valid_doppler_fraction']=float(sample['radar_doppler_valid'][-1].float().mean())
            observed=sample['agent_radial_mask'].numpy().astype(bool)
            residual=np.abs(sample['agent_state'].numpy()[:,2]-sample['agent_radial_observed'].numpy())
            row['measured_radial_residual_mps']=residual[observed].tolist()
            # Compare actual product arrays and metadata for the middle selected frame per episode.
            if item==selected[len(selected)//2]:
                if products is None:products=adapter._product_frames(episode.parent)
                _,folder,product,meta=products[index]
                with np.load(folder/'folded_ra.npz') as x:row['raw_power_error']=delta(x['power_peak_ra'][:,::-1],np.load(radar['power'][-1]))
                with np.load(folder/'unfolded_ra.npz') as x:
                    row['raw_doppler_error']=delta(x['doppler_unfolded_dense_ra_mps'][:,::-1],np.load(radar['unfolded_doppler'][-1]))
                    row['raw_doppler_mask_match']=np.array_equal(x['reliable_mask'][:,::-1],sample['radar_doppler_valid'][-1].numpy())
                row['radar_product_frame_match']=meta['carla_frame']==r['carla_frame']
                row['raw_radar_product']=str(product)
            samples.append(row)
        print('raw audit',ep,len(selected),flush=True)
        write(out/'sample_audit.json',samples)
    write(out/'raw_type_mapping.json',[dict(type_id=k[0],category=k[1],observations=v) for k,v in types.most_common()])
    # Source provenance is separate from performance: latest source is not what v2 trained.
    protocols={}
    for p in Path('analysis').glob('retrained*_v2/protocol.json'):
        d=json.load(p.open());protocols[str(p)]=dict(protocol_sha256=digest(p),config=d['config'],
            manifest_sha256=d['data']['manifest_sha256'],epochs=d['epochs'],seed=d['seed'],
            changed_sources=[k for k,v in d['source_files'].items() if not Path('src',k).exists() or digest(Path('src',k))!=v])
    write(out/'provenance.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):digest(p) for folder in ('src','scripts','collection/collector') for p in Path(folder).glob('*.py')},
        manifest_sha256=digest(args.manifest),protocols=protocols,seed=42))
    audit_exclusions(args, rows)
    print('AUDIT COMPLETE',len(samples),flush=True)


if __name__=='__main__':main()
