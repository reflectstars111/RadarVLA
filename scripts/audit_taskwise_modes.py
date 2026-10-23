"""Attribute Mode distribution by Global/Local sources, masks and scenario."""
import argparse,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
from radar_vla.agent50_migration import sha256,write_json
from radar_vla.local_risk import legacy_hazard_sources
from radar_vla.local_risk_audit import classify_sources


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();out=a.directory
    counters=Counter();groups=defaultdict(Counter);origins=defaultdict(Counter);metadata={};evidence={}
    def read(path):
        path=Path(path);data=path.read_bytes();evidence[str(path.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
    with (out/'manifest.jsonl').open() as source,(out/'mode_attribution.jsonl').open('w') as dest:
        for i,line in enumerate(source,1):
            r=json.loads(line);origin='recovered' if 'taskwise_recovery_provenance' in r else 'baseline'
            global_risk=r['global_risk'];local=r['local_risk'];mode=r['mode_label']
            def status(q,j):return ('danger' if q['pcol'][j] else 'safe') if q['pcol_mask'][j] else 'unknown'
            flags=Counter({'samples':1,'mode/'+mode:1})
            for j in range(3):flags[f'{j+1}s/global_{status(global_risk,j)}__local_{status(local,j)}']+=1
            flags['mask/'+''.join('1' if m else '0' for m in r['task_masks']['risk'])]+=1
            for reason in r['mode_reasons']:flags['reason/'+reason]+=1
            entries=[];sources=legacy_hazard_sources(r);ids=set().union(*map(set,sources))
            if ids:
                ep=Path(r['route_archive']).resolve().parent
                if str(ep) not in metadata:metadata[str(ep)]=read(ep/'metadata.json')
                frame=r['risk_provenance'].get('raw_evidence',{}).get('frame')
                if frame is None:
                    index=r.get('taskwise_recovery_provenance',{}).get('source_index',int(Path(r['sensors']['radar']['power'][-1]).name.split('_')[0]))
                    frame=ep/'frames'/f'{index:06d}'/'frame.json'
                current=read(frame)
                if current['carla_frame']!=r['carla_frame']:raise ValueError('Raw attribution frame mismatch')
                sets=r['agent_sets'];entries=classify_sources(current,metadata[str(ep)],ids,set(sets['local_agent_ids']),
                    {str(q['id']) for q in r.get('ignored_agents',[])},set(sets['planner_selected_ids']))
                flags['global_hazard_source_outside_spatial_domain']+=any(q.get('membership_reason')=='outside_spatial_domain' for q in entries)
                flags['global_hazard_source_ignored_class']+=any(q.get('membership_reason')=='ignored_class' for q in entries)
                flags['global_hazard_source_not_selected']+=any(not q.get('planner_selected',False) for q in entries)
                flags['source_unrecoverable']+=any(q.get('status')=='raw_actor_unrecoverable' for q in entries)
            if mode=='LONG':
                flags['LONG/global_3s_'+status(global_risk,2)]+=1
                if status(global_risk,2)=='danger':
                    flags['LONG/global_danger_outside_spatial_domain']+=any(q.get('membership_reason')=='outside_spatial_domain' for q in entries)
                    flags['LONG/global_danger_ignored_class']+=any(q.get('membership_reason')=='ignored_class' for q in entries)
            counters.update(flags);groups[r.get('category','unknown')].update(flags);origins[origin].update(flags)
            dest.write(json.dumps(dict(sample_id=r['sample_id'],episode_id=r['episode_id'],split=r['split'],scenario=r.get('category'),
                origin=origin,mode_label=mode,mode_reasons=r['mode_reasons'],risk_mask=r['task_masks']['risk'],
                global_status=[status(global_risk,j) for j in range(3)],local_status=[status(local,j) for j in range(3)],
                global_source_agent_ids=sources,source_attribution=entries,sft_available=r['sft_available']),separators=(',',':'))+'\n')
            if i%2000==0:print('ATTRIBUTION',i,flush=True)
    if counters['source_unrecoverable']:raise ValueError('Hazard source attribution incomplete')
    write_json(out/'mode_attribution_report.json',dict(status='PASS',totals=counters,by_scenario=dict(groups),by_origin=dict(origins),raw_evidence=evidence,
        source_sha256=sha256(Path(__file__)),note='Labels/masks unchanged. Counts are per-sample flags, may overlap; Global unknown is not Global safe.'))
    print('ATTRIBUTION PASS',counters['samples'],flush=True)

if __name__=='__main__':main()
