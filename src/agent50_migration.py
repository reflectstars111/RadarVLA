"""Versioned, state-only migration of admitted agents; never creates sensor arrays."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from . import collection_adapter as adapter
from .categories import AGENT_LABEL_SCHEMA, category_name
from .annotation_domain import detection_domain_values


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _hazard(label):
    return (any(v >= .5 and m for v,m in zip(label['pcol'][:2],label['pcol_mask'][:2]))
            or label['dmin_mask'] and label['dmin'] <= .5
            or label['areq_mask'] and label['areq'] >= 3.)


def migrate_record(old, episode, frame_loader, metadata):
    """Return a copied record and audit detail. Only spatial corrections remove IDs."""
    if old.get('grounding_label_schema') is not None:
        raise ValueError('Source already has a grounding label schema; use the legacy input manifest')
    if metadata['episode_id'] != old['episode_id']:
        raise ValueError('Raw episode ID differs from manifest')
    if metadata.get('agent_label_range_m') != 50 or metadata.get('map_label_range_m') != 50:
        raise ValueError('Original metadata must certify 50m labels')
    radar = old['sensors']['radar']
    if not isinstance(radar['power'], list) or len(radar['power']) != 4:
        raise ValueError('Migration requires the existing four-frame collection products')
    indices = [int(Path(p).name.split('_')[0]) for p in radar['power']]
    if indices != list(range(indices[-1]-3, indices[-1]+1)):
        raise ValueError('History is not four consecutive source frames')
    frames = [frame_loader(episode/'frames'/f'{i:06d}'/'frame.json') for i in indices]
    current = frames[-1]
    if current['carla_frame'] != old['carla_frame'] or current['episode_id'] != old['episode_id']:
        raise ValueError('Current raw frame differs from manifest')
    if [f['carla_frame'] for f in frames] != list(range(old['carla_frame']-3, old['carla_frame']+1)):
        raise ValueError('CARLA history is not consecutive')
    if not np.allclose([f['timestamp_s']-current['timestamp_s'] for f in frames],radar['time_offsets_s'],atol=1e-8,rtol=0):
        raise ValueError('History timestamp differs')
    if abs(current['timestamp_s']-old['timestamp_s'])>1e-8:
        raise ValueError('Current timestamp differs')
    raw_agents = frame_loader(episode/'frames'/f'{indices[-1]:06d}'/'agents.json')
    if raw_agents['carla_frame'] != old['carla_frame']:
        raise ValueError('Raw agent frame differs')
    admitted = {str(a['actor_id']): a for a in raw_agents['agents']}
    if len(admitted) != len(raw_agents['agents']):
        raise ValueError('Duplicate raw admitted actor ID')
    world = {str(a['actor_id']): a for a in current['actors']}
    mount = metadata['sensors']['semantic_lidar']
    geometry = adapter.collector_geometry()
    sensor_position, sensor_rotation = geometry.radar_pose(current['ego']['pose'],mount)
    # This is precisely the historical collector membership test, including its
    # sampled bbox boundary convention; no new actor types or centre cutoff.
    def inside(actor):
        return geometry.actor_intersects_fov(actor,sensor_position,sensor_rotation,50.)
    expected = {key for key,a in world.items()
                if (a['type_id'].startswith('vehicle.') or a['type_id'].startswith('walker.pedestrian.')) and inside(a)}
    original_ids = [str(a['id']) for a in old['agents']]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError('Duplicate prepared Agent ID')
    missing = set(original_ids)-admitted.keys()
    if missing:
        raise ValueError(f'Prepared IDs absent from original admitted agents: {sorted(missing)}')
    new = copy.deepcopy(old)
    new['agents'] = []
    removed = []; mapping = Counter(); stationary = moving = parked = 0; missing_meta=[]
    for agent in old['agents']:
        identifier = str(agent['id']); raw = admitted[identifier]
        if not inside(raw):
            removed.append(identifier)
            continue
        category = category_name(agent)
        mapping[category] += 1
        result = adapter.agent50_label(agent,raw)
        missing_fields = [key for key in ('type_id','actor_type','role_name') if key not in raw]
        if missing_fields:
            missing_meta.append(dict(id=identifier,fields=missing_fields))
        new['agents'].append(result)
        speed = np.linalg.norm(agent['velocity'])
        stationary += int(speed < .1); moving += int(speed >= .1)
        parked += int('park' in str(raw.get('role_name','')).lower())
    if removed and 'planner_agent_ids' in new:
        new['planner_agent_ids']=[i for i in new['planner_agent_ids'] if str(i) not in set(removed)]
        if new['teacher_mode']=='short' and not any(_hazard(new['agent_risk_labels'][str(i)]) for i in new['planner_agent_ids']):
            raise ValueError('Spatial correction would invalidate the existing SHORT teacher')
    new['grounding_label_schema'] = AGENT_LABEL_SCHEMA
    new['preparation_contract'] = adapter.PREPARATION_CONTRACT
    origin=current['ego']['pose']['position_m'][:2];yaw=current['ego']['pose']['rotation_deg'][2]
    velocities=[adapter._radar_sensor_velocity(f,yaw,mount) for f in frames]
    # An actual mount projection is part of the spatial-domain correction.
    # This changes extrinsic state only, not RA arrays, axes, or GT coordinates.
    poses=[adapter.radar_bev_pose(f,origin,yaw,mount) for f in frames]
    velocity_delta=np.asarray(velocities)-np.asarray(radar['sensor_velocity_current_ego'])
    pose_delta=np.asarray(poses)-np.asarray(radar['poses_current_ego'])
    new['sensors']['radar']['sensor_velocity_current_ego']=velocities
    new['sensors']['radar']['poses_current_ego']=poses
    if not np.allclose([radar['azimuth_rad'][0],radar['azimuth_rad'][-1]],np.radians([-53.,53.]),atol=1e-7,rtol=0):
        raise ValueError('Saved radar azimuth differs from original collector FOV; cannot guess')
    new['detection_annotation_domain']=dict(type='radar_bev_sector',range_m=50.,
        azimuth_min_rad=float(radar['azimuth_rad'][0]),azimuth_max_rad=float(radar['azimuth_rad'][-1]),
        source='raw 50m collector contract; saved Radar azimuth axis; actual mount projected to BEV')
    detection_domain_values(new['detection_annotation_domain'])
    # Compare everything outside the explicitly permitted edits, including the
    # entire risk_agents collection, risk labels and teacher mode.
    before=copy.deepcopy(old);after=copy.deepcopy(new)
    for r in (before,after):
        for key in ('grounding_label_schema','preparation_contract','detection_annotation_domain','agents','planner_agent_ids'):
            r.pop(key,None)
        for key in ('sensor_velocity_current_ego','poses_current_ego'):
            r['sensors']['radar'].pop(key,None)
    if before != after:
        raise ValueError('Unrelated record fields changed')
    originals={str(a['id']):a for a in old['agents']}
    for agent in new['agents']:
        previous=originals[str(agent['id'])]
        for key,value in previous.items():
            if key!='category' and agent.get(key)!=value:
                raise ValueError(f'Original Agent field changed: {key}')
    old_pose=np.asarray(radar['poses_current_ego'][-1]);new_pose=np.asarray(poses[-1])
    radial_deltas=[]
    for a in new['agents']:
        center=np.asarray(a['position']);v=np.asarray(a['velocity'])
        direction_old=center-old_pose[:2,3];direction_old/=max(np.linalg.norm(direction_old),1e-12)
        direction_new=center-new_pose[:2,3];direction_new/=max(np.linalg.norm(direction_new),1e-12)
        radial_deltas.append(float(np.dot(v-np.asarray(velocities[-1]),direction_new)
            -np.dot(v-np.asarray(radar['sensor_velocity_current_ego'][-1]),direction_old)))
    detail=dict(sample_id=old['sample_id'],episode_id=old['episode_id'],split=old['split'],
        source_index=indices[-1],old_ids=original_ids,new_ids=[str(a['id']) for a in new['agents']],removed_outside_original_fov=removed,
        raw_only_admitted_ids=sorted(admitted.keys()-set(original_ids)),
        original_collection_missing_ids=sorted(expected-admitted.keys()),
        original_collection_extra_ids=sorted(admitted.keys()-expected),
        class_mapping=dict(mapping),stationary=stationary,moving=moving,park_role=parked,
        missing_metadata=missing_meta,velocity_delta_mps=velocity_delta.tolist(),
        velocity_error_norms_mps=np.linalg.norm(velocity_delta,axis=1).tolist(),
        mount_pose_max_abs_change=float(np.abs(pose_delta).max()),
        derived_radial_delta_mps=radial_deltas,
        raw_frame=str(episode/'frames'/f'{indices[-1]:06d}'/'frame.json'))
    return new,detail


def migrate(source, output, config, retry_empty_failure=False, audit_missing_sensors=False):
    source=Path(source).resolve();output=Path(output).resolve();config=Path(config).resolve()
    if output.exists():
        failure=output/'failure.json'
        permitted={'failure.json','manifest.partial.jsonl','samples.jsonl','sensor_preflight.json'}
        if (not retry_empty_failure or not failure.is_file() or json.loads(failure.read_text())['processed']!=0
                or any(p.name not in permitted for p in output.iterdir())
                or any((output/name).exists() and (output/name).stat().st_size for name in ('manifest.partial.jsonl','samples.jsonl'))):
            raise FileExistsError(f'Refusing to overwrite {output}')
        failure.rename(output/'initial_failure.json')
    output.mkdir(parents=True,exist_ok=True)
    raw_sources={}; metadata_cache={}; source_states={}; sensor_files=set(); counters=defaultdict(Counter)
    classes=Counter(); samples=set(); episode_splits={}; radial=[]; max_velocity=0.; max_pose=0.; count=0
    source_hash=sha256(source)
    missing_sensors=set()
    @lru_cache(maxsize=64)
    def read_raw(path):
        raw_sources[str(path)]=sha256(path)
        return json.loads(Path(path).read_text())
    partial=output/'manifest.partial.jsonl';details=output/'samples.jsonl'
    try:
        with source.open() as incoming,partial.open('w') as outgoing,details.open('w') as audit:
            for number,line in enumerate(incoming,1):
                if not line.strip():continue
                old=json.loads(line);episode=Path(old['route_archive']).resolve().parent
                if episode not in metadata_cache:
                    metadata_cache[episode]=read_raw(episode/'metadata.json')
                new,detail=migrate_record(old,episode,read_raw,metadata_cache[episode])
                identifier=old['sample_id'];ep=old['episode_id'];split=old['split']
                if identifier in samples:raise ValueError('Duplicate sample ID')
                if ep in episode_splits and episode_splits[ep]!=split:raise ValueError('Episode split leakage')
                samples.add(identifier);episode_splits[ep]=split
                if 'planner_agent_ids' in old and not detail['removed_outside_original_fov']:
                    if new['planner_agent_ids']!=old['planner_agent_ids']:raise ValueError('Planner IDs changed')
                for branch in ('power','unfolded_doppler','doppler_valid'):
                    values=old['sensors']['radar'][branch];values=values if isinstance(values,list) else [values]
                    for path in values:
                        p=(source.parent/path).resolve()
                        if p!=(output/path).resolve():raise ValueError('Relative sensor path would resolve differently; no silent rebase')
                        if p not in sensor_files:
                            if not p.is_file() and audit_missing_sensors:
                                missing_sensors.add(str(p))
                            else:
                                stat=p.stat();source_states[str(p)]=[stat.st_size,stat.st_mtime_ns]
                            sensor_files.add(p)
                if (source.parent/old['route_archive']).resolve()!=(output/new['route_archive']).resolve():raise ValueError('Route reference changed')
                c=counters[split];c['samples']+=1;c['agents_old']+=len(old['agents']);c['agents_new']+=len(new['agents'])
                c['id_set_differences']+=int(set(detail['old_ids'])!=set(detail['new_ids']))
                for key in ('stationary','moving','park_role'):c[key]+=detail[key]
                norms=np.array(detail['velocity_error_norms_mps']);c['samples_velocity_changed_gt_1e_9']+=int(norms.max()>1e-9)
                c['samples_latest_velocity_changed_gt_01']+=int(norms[-1]>.1)
                c['history_vectors_velocity_changed_gt_01']+=int((norms>.1).sum())
                c['samples_mount_pose_changed_gt_1e_9']+=int(detail['mount_pose_max_abs_change']>1e-9)
                c['metadata_missing_instances']+=len(detail['missing_metadata'])
                c['collection_missing_observations']+=len(detail['original_collection_missing_ids'])
                classes.update(detail['class_mapping']);radial.extend(detail['derived_radial_delta_mps'])
                max_velocity=max(max_velocity,float(norms.max()));max_pose=max(max_pose,detail['mount_pose_max_abs_change'])
                outgoing.write(json.dumps(new,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                audit.write(json.dumps(detail,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                count+=1
                if count%250==0:print(f'migrated {count}',flush=True)
        if not count:raise ValueError('Empty source')
        if source_hash!=sha256(source):raise ValueError('Source manifest changed during migration')
        for path,state in source_states.items():
            stat=Path(path).stat()
            if [stat.st_size,stat.st_mtime_ns]!=state:raise ValueError(f'Sensor file modified during migration: {path}')
        manifest=output/'manifest.jsonl'
        missing_sensors={p for p in missing_sensors if not Path(p).is_file()}
        report=dict(status='BLOCKED_MISSING_SENSORS' if missing_sensors else 'PASS',samples=count,episodes=len(episode_splits),splits=dict(counters),
            class_mapping={k:dict(target='agent',observations=v) for k,v in classes.items()},
            sensor_files=len(sensor_files),sensor_arrays_written=0,sensor_paths_unchanged=True,
            missing_sensor_files=sorted(missing_sensors),
            max_sensor_velocity_change_mps=max_velocity,max_mount_pose_change=max_pose,
            derived_radial_change_mps=dict(observations=len(radial),mean_abs=float(np.mean(np.abs(radial))) if radial else None,
                max_abs=float(np.max(np.abs(radial))) if radial else None),
            invariants='Risk, teachers, risk_agents, Ego Future, road, all retained Agent physical/future fields unchanged',
            parked_definition='original role_name contains park; no parked-state filter added; stationary speed<0.1m/s is reporting only')
        files=['src/agent50_migration.py','src/collection_adapter.py','src/categories.py','src/annotation_domain.py','collection/collector/road_map.py','collection/collector/recorder.py']
        root=Path(__file__).resolve().parents[1]
        provenance=dict(schema='agent50_migration_v1',source_manifest=str(source),source_manifest_sha256=source_hash,
            manifest=str(manifest),manifest_sha256=sha256(partial),config=str(config),config_sha256=sha256(config),
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            source_sha256={name:sha256(root/name) for name in files},raw_sources=raw_sources,
            radar_mount_source='semantic_lidar mount shared by the synthesized radar; BEV projection only; original collector FOV membership retained',
            file_identity_proof='unchanged absolute paths, no sensor writes, identical pre/post size and mtime; original arrays loaded unchanged by Dataset',
            sensor_file_states=source_states)
        write_json(output/'provenance.json',provenance);write_json(output/'migration_report.json',report)
        # Publish only after invariants and file checks succeed.
        if not missing_sensors:
            partial.replace(manifest)
        write_json(output/'validation.json',dict(status=report['status'],checks=['sample/episode/split','raw membership and metadata','all unrelated fields identical',
            'four-frame raw rigid velocity','actual mount BEV domain','sensor paths and file states unchanged'],training_started=False))
        print(json.dumps({k:v for k,v in report.items() if k!='missing_sensor_files'},ensure_ascii=False,indent=2),flush=True)
        print('missing sensor files',len(missing_sensors),flush=True)
        return report
    except BaseException as exc:
        write_json(output/'failure.json',dict(status='FAILED',processed=count,error=str(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--retry-empty-failure',action='store_true')
    parser.add_argument('--audit-missing-sensors',action='store_true');args=parser.parse_args()
    report=migrate(args.source,args.output_dir,args.config,args.retry_empty_failure,args.audit_missing_sensors)
    if report['status']=='PASS':
        from .agent50_filter import filter_manifest
        filter_manifest(args.output_dir)

if __name__=='__main__':main()
