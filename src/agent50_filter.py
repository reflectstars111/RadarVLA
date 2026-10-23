"""Remove pedestrian/cyclist detection labels without teaching their pixels as background."""
from __future__ import annotations
import copy

IGNORE_POLICY = 'pedestrian_cyclist_regions_v1'
BICYCLE_BLUEPRINTS = frozenset(('vehicle.bh.crossbike','vehicle.diamondback.century','vehicle.gazelle.omafiets'))


def excluded_agent_kind(agent):
    source = str(agent.get('source_category', agent.get('category','unknown'))).strip().lower()
    kind = str(agent.get('type_id',''))
    if source in ('pedestrian','person') or agent.get('actor_type') == 'pedestrian' or kind.startswith('walker.pedestrian.'):
        return 'pedestrian'
    if source in ('cyclist','bicycle','bicyclist') or kind in BICYCLE_BLUEPRINTS:
        return 'cyclist'
    return None


def _hazard(label):
    return (any(v>=.5 and m for v,m in zip(label['pcol'][:2],label['pcol_mask'][:2]))
        or label['dmin_mask'] and label['dmin']<=.5 or label['areq_mask'] and label['areq']>=3.)


def filter_record(record):
    if record.get('grounding_label_schema')!='agent50_v1':
        raise ValueError('Ignored-agent filtering requires agent50_v1')
    new=copy.deepcopy(record);kept=[];ignored=list(new.get('ignored_agents',[]));removed={}
    for agent in new['agents']:
        kind=excluded_agent_kind(agent)
        if kind is None:
            kept.append(agent)
        else:
            removed[str(agent['id'])]=kind
            ignored.append({**{k:copy.deepcopy(agent[k]) for k in ('id','position','size','heading','type_id','actor_type','role_name','source_category') if k in agent},
                'ignore_reason':kind})
    new['agents']=kept;new['ignored_agents']=ignored;new['agent_ignore_policy']=IGNORE_POLICY
    # The same 2m tolerance used for detection matching defines ambiguous centres.
    new['detection_ignore_margin_m']=2.
    prior_ids=list(new.get('planner_agent_ids',[a['id'] for a in record['agents']]))
    new['planner_agent_ids']=[i for i in prior_ids if str(i) not in removed]
    available=bool(new.get('planner_supervision_available',True))
    if new.get('teacher_mode')=='short' and any(str(i) in removed for i in prior_ids):
        labels=new.get('agent_risk_labels',{})
        if not any(_hazard(labels[str(i)]) for i in new['planner_agent_ids']):
            available=False
    new['planner_supervision_available']=available
    # Risk/world actors, ego/agent futures of retained agents, and mode labels stay unchanged.
    return new,dict(sample_id=record['sample_id'],episode_id=record.get('episode_id'),split=record['split'],
        removed=removed,kept_ids=[str(a['id']) for a in kept],
        teacher_mode_unchanged=new.get('teacher_mode')==record.get('teacher_mode'),
        planner_supervision_available=available)


def filter_manifest(directory):
    """Publish the filtered manifest only after an invariant-preserving state migration."""
    from collections import Counter, defaultdict
    import json
    from pathlib import Path
    from .agent50_migration import sha256, write_json
    from .collection_adapter import PREPARATION_CONTRACT
    out=Path(directory)
    manifest=out/'manifest.jsonl'
    if (out/'filter_report.json').exists():raise FileExistsError('Already filtered; refusing repeat publication')
    source_hash=sha256(manifest);stats=defaultdict(Counter);rows=0
    partial=out/'manifest.filtered.partial.jsonl'
    with manifest.open() as incoming,partial.open('w') as output,(out/'filter_samples.jsonl').open('w') as audit:
        for line in incoming:
            old=json.loads(line);new,detail=filter_record(old);new['preparation_contract']=PREPARATION_CONTRACT
            for key in ('risk_label','risk_agents','agent_risk_labels','ego','road_segments','teacher_mode','sensors','split','sample_id','episode_id'):
                if new.get(key)!=old.get(key):raise ValueError(f'Forbidden field changed: {key}')
            original={str(x['id']):x for x in old['agents']}
            assert set(original)==set(detail['kept_ids'])|set(detail['removed'])
            assert not set(detail['kept_ids'])&set(detail['removed'])
            for agent in new['agents']:assert agent==original[str(agent['id'])]
            c=stats[new['split']];c['samples']+=1;c['agents_before_filter']+=len(old['agents']);c['agents_after_filter']+=len(new['agents']);c.update(detail['removed'].values())
            c['samples_with_ignored_agents']+=bool(detail['removed']);c['sft_teacher_unavailable']+=not detail['planner_supervision_available']
            audit.write(json.dumps(detail,separators=(',',':'))+'\n');output.write(json.dumps(new,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');rows+=1
    report=dict(status='PASS',policy=IGNORE_POLICY,samples=rows,splits=dict(stats),source_manifest_sha256=source_hash,
        manifest_sha256=sha256(partial),ignore_rule='GT bbox interior OR distance<=2m to removed centre; nearest anchor cell also ignored during training; matched retained positives override ignore',
        teacher_rule='Original mode/risk labels unchanged. SFT supervision unavailable only if removal leaves SHORT without a retained selected critical target; Grounding keeps every frame.',
        raw_data_deleted=False,frames_deleted=0,script_sha256=sha256(Path(__file__)),filter_sha256=sha256(Path(__file__)))
    manifest.rename(out/'manifest.before_filter.jsonl');partial.replace(manifest)
    for name in ('provenance','migration_report','validation'):
        (out/f'{name}.json').rename(out/f'{name}.before_filter.json')
    parent=json.loads((out/'provenance.before_filter.json').read_text())
    write_json(out/'provenance.json',dict(schema='agent50_filtered_migration_v1',source_manifest=parent['source_manifest'],
        source_manifest_sha256=parent['source_manifest_sha256'],manifest=str(manifest.resolve()),manifest_sha256=report['manifest_sha256'],
        state_migration_provenance='provenance.before_filter.json',state_migration_provenance_sha256=sha256(out/'provenance.before_filter.json'),
        filter_report=report,config=parent['config'],config_sha256=sha256(parent['config']),
        sensor_restoration='sensor_restoration.json' if (out/'sensor_restoration.json').exists() else None,
        sensor_restoration_sha256=sha256(out/'sensor_restoration.json') if (out/'sensor_restoration.json').exists() else None))
    parent_report=json.loads((out/'migration_report.before_filter.json').read_text())
    write_json(out/'migration_report.json',dict(status='PASS',samples=rows,episodes=parent_report['episodes'],
        state_migration='migration_report.before_filter.json',state_migration_sha256=sha256(out/'migration_report.before_filter.json'),
        class_mapping='All retained positives -> agent; pedestrians/cyclists -> ignored regions, not background',filter=report))
    write_json(out/'filter_report.json',report)
    write_json(out/'validation.json',dict(status='PASS',state_checks='validation.before_filter.json',frames_preserved=rows,
        ignored_policy=IGNORE_POLICY,risk_and_mode_labels_unchanged=True,training_started=False))
    print(json.dumps(report,indent=2))
