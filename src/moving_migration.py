"""Resumable motion-label migration and independent raw-evidence acceptance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import time

from . import collection_adapter as adapter
from .agent50_migration import sha256
from .moving_agents import POLICY, rebuild_agents, attach_supervision, validate_motion, _path


def write_json(path, value):
    path = Path(path)
    staged = path.with_suffix(path.suffix + '.tmp')
    with staged.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    staged.replace(path)


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class Evidence:
    def __init__(self, source):
        self.targets = defaultdict(set)
        self.hashes = {}
        for r in rows(source):
            self.targets[str(Path(r['route_archive']).resolve().parent)].add(r['carla_frame'])
        self.anchors = lru_cache(maxsize=2)(self._anchors)

    def read(self, path):
        path = Path(path)
        data = path.read_bytes()
        import hashlib
        self.hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def _anchors(self, episode):
        path = Path(episode) / 'anchors.jsonl'
        return adapter._selected_anchors(path, self.targets[episode])

    def rebuild(self, r):
        episode = str(Path(r['route_archive']).resolve().parent)
        # These names are written by the canonical adapter's radar cache.
        index = int(Path(r['sensors']['radar']['power'][-1]).name.split('_')[0])
        current = self.read(Path(episode)/'frames'/f'{index:06d}'/'frame.json')
        metadata = self.read(Path(episode)/'metadata.json')
        original = self.read(Path(episode)/'frames'/f'{index:06d}'/'agents.json')
        if original['carla_frame'] != r['carla_frame']:
            raise ValueError('Raw agents frame mismatch')
        anchor = self.anchors(episode).get(r['carla_frame'])
        if anchor is None:
            raise ValueError('Missing raw anchor')
        # Require the unfiltered, captured in-domain set, not only saved positives.
        geom = adapter.collector_geometry()
        position, rotation = geom.radar_pose(current['ego']['pose'], metadata['sensors']['semantic_lidar'])
        expected = {a['actor_id'] for a in current['actors'] if geom.actor_intersects_fov(a, position, rotation, 50.)}
        observed = original.get('observed_agents', original['agents'])
        if len(observed) != len(expected) or {a['actor_id'] for a in observed} != expected:
            # Some taskwise recovery rows explicitly marked the old spatial labels
            # unknown; do not turn those into known negatives by guessing.
            if r.get('agent_supervision_available', False):
                raise ValueError('Raw in-domain Agent archive incomplete')
        import hashlib
        anchor_key = f'{Path(episode)/"anchors.jsonl"}#carla_frame={r["carla_frame"]}'
        self.hashes[anchor_key] = hashlib.sha256(json.dumps(anchor, sort_keys=True).encode()).hexdigest()
        return rebuild_agents(r, current, metadata, anchor)


def organize_existing(data_root):
    """Move data to prepared; retain every historical path through compatibility links."""
    root = Path(data_root)
    old, dest = root/'analysis/taskwise_local_v1', root/'prepared/taskwise_local_v1'
    if not dest.exists():
        if not old.is_dir() or old.is_symlink():
            raise ValueError('Expected existing taskwise dataset directory')
        # Absolute references remain valid via links. Refuse unverified relative
        # references before moving anything.
        for r in rows(old/'manifest.jsonl'):
            if not Path(r['route_archive']).is_absolute():
                raise ValueError('Relative route path requires an explicit migration')
            for key in ('power', 'unfolded_doppler', 'doppler_valid'):
                values = r['sensors']['radar'][key]
                if any(not Path(p).is_absolute() for p in (values if isinstance(values, list) else [values])):
                    raise ValueError('Relative sensor path requires an explicit migration')
        dest.parent.mkdir(parents=True, exist_ok=True)
        old.rename(dest)
    old.mkdir(parents=True, exist_ok=True)
    # Metadata needed to reproduce/use the dataset travels with it; analysis
    # products live in analysis, with backwards links from the prepared directory.
    data_names = {'manifest.jsonl', 'radar', 'radar_masks', 'provenance.json',
                  'validation.json', 'acceptance.json', 'raw_evidence.json',
                  'radar_cache_hashes.json', 'source_snapshot', 'source_snapshot_hashes.json',
                  'implementation_provenance.json'}
    for p in list(dest.iterdir()):
        if p.is_symlink():
            continue
        historical = old/p.name
        if p.name in data_names:
            if not historical.exists():
                historical.symlink_to(p)
        else:
            if historical.exists():
                raise FileExistsError(historical)
            p.rename(historical)
            p.symlink_to(historical)
    if sha256(old/'manifest.jsonl') != sha256(dest/'manifest.jsonl'):
        raise ValueError('Compatibility manifest hash mismatch')
    return dest/'manifest.jsonl'


def migrate(source, output, reports, config):
    from .data import _validate_record
    from .pipeline import resolve_config
    source, output, reports, config = map(lambda p: Path(p).resolve(), (source, output, reports, config))
    cfg = resolve_config(json.loads(config.read_text()))
    if cfg['data'].get('agent_selection_policy') != POLICY:
        raise ValueError('Moving configuration required')
    signature = dict(source=str(source), source_sha256=sha256(source), config=str(config),
        config_sha256=sha256(config), policy=POLICY,
        code={str(p):sha256(p) for p in [*Path(__file__).parent.glob('*.py'), _path,
             _path.with_name('road_map.py')]})
    output.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    provenance = output/'provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text()) != signature:
            raise ValueError('Source/code/config changed; use a new output version')
    else:
        if any(output.iterdir()):
            raise ValueError('Unrecognized nonempty migration output')
        write_json(provenance, signature)
    evidence = Evidence(source)
    partial, audit_path, progress = output/'manifest.partial.jsonl', reports/'samples.jsonl', output/'progress.json'
    if not progress.exists():
        partial.touch(); audit_path.touch()
        write_json(progress, dict(next_index=0, manifest_bytes=0, audit_bytes=0))
    journal = json.loads(progress.read_text())
    final = output/'manifest.jsonl'
    if final.exists():
        return
    for p, key in ((partial, 'manifest_bytes'), (audit_path, 'audit_bytes')):
        with p.open('r+b') as stream:
            stream.truncate(journal[key])
    started = time.monotonic()
    with partial.open('ab') as manifest, audit_path.open('ab') as audit:
        for index, old in enumerate(rows(source)):
            if index < journal['next_index']:
                continue
            rebuilt = evidence.rebuild(old)
            new = attach_supervision(rebuilt)
            _validate_record(new, index+1)
            for key in ('sensors', 'ego', 'road_segments', 'split', 'sample_id', 'episode_id', 'route_archive'):
                if old.get(key) != new.get(key):
                    raise ValueError('Forbidden migration change: ' + key)
            previous = {str(a['id']) for a in old['agents']}
            detail = dict(sample_id=new['sample_id'], split=new['split'], agents_before=len(old['agents']),
                agents_after=len(new['agents']), recovered_ids=[str(a['id']) for a in new['agents'] if str(a['id']) not in previous],
                background=Counter(a['reason'] for a in new['background_agents']),
                mode_before=old['mode_label'], mode_after=new['mode_label'], sft_available=new['sft_available'])
            manifest.write((json.dumps(new, separators=(',', ':'), allow_nan=False)+'\n').encode())
            audit.write((json.dumps(detail, separators=(',', ':'))+'\n').encode())
            manifest.flush(); audit.flush()
            os.fsync(manifest.fileno()); os.fsync(audit.fileno())
            write_json(progress, dict(next_index=index+1, manifest_bytes=manifest.tell(), audit_bytes=audit.tell()))
            if (index+1)%100 == 0:
                print(f'MIGRATED {index+1} elapsed={time.monotonic()-started:.1f}s', flush=True)
    # Recompute the complete evidence hashes also on resume; never publish only
    # the subset processed since the last interruption.
    for r in rows(source):
        evidence.rebuild(r)
    write_json(output/'raw_evidence.json', evidence.hashes)
    write_json(output/'migration.json', dict(status='MIGRATED', manifest_sha256=sha256(partial),
        source_sha256=signature['source_sha256']))
    partial.replace(final)


def verify(source, output, reports, config):
    from .data import _validate_record, collate_batch
    from .pipeline import resolve_config, validate_input_shapes, _loss
    from .local_risk import compute_local_risk
    from .model import RadarVLAGrounder, ModelConfig
    from .planner import PlannerConfig, RiskConditionedPlanner
    from .visual_diagnosis import _single_sample
    import torch
    source, output, reports, config = map(Path, (source, output, reports, config))
    evidence = Evidence(source); counts = Counter(); splits = Counter(); episodes = {}; choices = {}; seen = set()
    incoming = iter(rows(output/'manifest.jsonl'))
    for index, old in enumerate(rows(source), 1):
        r = next(incoming)
        _validate_record(r, index); validate_motion(r)
        rebuilt = evidence.rebuild(old)
        for key in ('agents', 'risk_agents', 'background_agents', 'planner_agent_ids', 'sensors', 'ego', 'road_segments', 'split', 'sample_id', 'episode_id'):
            if r.get(key) != rebuilt.get(key):
                raise ValueError('Raw reconstruction mismatch: ' + key)
        risk, individual = compute_local_risk(r)
        if risk != r['local_risk'] or individual != r['local_agent_risk_labels'] or risk != r['risk_label']:
            raise ValueError('Motion risk recomputation mismatch')
        if r['sample_id'] in seen or (r['episode_id'] in episodes and episodes[r['episode_id']] != r['split']):
            raise ValueError('Duplicate sample or episode split leakage')
        seen.add(r['sample_id']); episodes[r['episode_id']] = r['split']; splits[r['split']] += 1
        counts['samples'] += 1; counts['agents'] += len(r['agents'])
        counts['frames_over_8_agents'] += len(r['agents']) > 8
        counts['recovered_positive_instances'] += len({str(a['id']) for a in r['agents']} - {str(a['id']) for a in old['agents']})
        counts['mode/'+r['mode_label']] += 1; counts['sft_available'] += r['sft_available']
        counts.update({'background/'+k:v for k,v in Counter(a['reason'] for a in r['background_agents']).items()})
        for key, yes in dict(empty=not r['agents'], many=len(r['agents'])>8, short=r['mode_label']=='SHORT',
                             long=r['mode_label']=='LONG', unknown=r['mode_label']=='UNKNOWN', background=bool(r['background_agents'])).items():
            if yes and key not in choices:
                choices[key] = r
        if index%100 == 0:
            print('VERIFIED', index, flush=True)
    if next(incoming, None) is not None:
        raise ValueError('Unexpected extra output records')
    provenance = json.loads((output/'provenance.json').read_text())
    if sha256(source) != provenance['source_sha256'] or sha256(config) != provenance['config_sha256']:
        raise ValueError('Source or configuration changed during migration')
    for path, digest in provenance['code'].items():
        if sha256(path) != digest:
            raise ValueError('Implementation changed during migration: ' + path)
    if evidence.hashes != json.loads((output/'raw_evidence.json').read_text()):
        raise ValueError('Raw evidence changed during migration')
    if sha256(output/'manifest.jsonl') != json.loads((output/'migration.json').read_text())['manifest_sha256']:
        raise ValueError('Published manifest changed during migration')
    cfg = resolve_config(json.loads(config.read_text()))
    class Metadata: pass
    ds = Metadata(); ds.records = list(choices.values()); ds.class_schema='agent_v1'
    ds.risk_schema='local_agent_collision_v1'; ds.supervision_protocol='taskwise_local_v1'
    validate_input_shapes(ds, cfg)
    torch.set_num_threads(1); torch.manual_seed(42)
    model = RadarVLAGrounder(ModelConfig(**cfg['model'])).eval()
    planner = RiskConditionedPlanner(PlannerConfig(**cfg['planner'])).eval()
    gradients = []
    for case, r in choices.items():
        sample = _single_sample(r); batch = collate_batch([sample])
        if batch['detection_ignore_mask'].any():
            raise ValueError('Background still ignored')
        model.zero_grad(set_to_none=True); planner.zero_grad(set_to_none=True)
        loss, _, _ = _loss(model, planner, batch)
        if not torch.isfinite(loss): raise ValueError('Nonfinite representative loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in [*model.parameters(), *planner.parameters()]):
            raise ValueError('Nonfinite representative gradient')
        gradients.append(dict(case=case, sample_id=r['sample_id'], loss=float(loss.detach())))
    report = dict(status='PASS', policy=POLICY, counts=dict(counts), splits=dict(splits), episodes=len(episodes),
        manifest_sha256=sha256(output/'manifest.jsonl'), source_sha256=sha256(source),
        raw_evidence_checked=True, representative_gradients=gradients, optimizer_steps=0, training_started=False)
    write_json(output/'acceptance.json', report)
    write_json(reports/'summary.json', report)
    (reports/'REPORT.md').write_text('# Motion dataset acceptance\n\n```json\n'+json.dumps(report, indent=2)+'\n```\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--organize-existing', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.data_root.resolve()
    source = root/'prepared/taskwise_local_v1/manifest.jsonl'
    output = root/'prepared/taskwise_moving_v1'; reports = root/'analysis/taskwise_moving_v1'
    reports.mkdir(parents=True, exist_ok=True)
    # A process lock prevents concurrent resumes from truncating each other's journals.
    import fcntl
    with (reports/'migration.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(reports/'status.json', dict(status='RUNNING', pid=os.getpid()))
            if args.organize_existing:
                source = organize_existing(root)
            if not args.verify_only:
                migrate(source, output, reports, args.config)
            write_json(reports/'status.json', dict(status='VERIFYING', pid=os.getpid()))
            report = verify(source, output, reports, args.config)
            # Existing canonical validation reads every sensor array. Its matching
            # manifest hash is required by the production launch wrapper.
            import subprocess
            project = Path(__file__).resolve().parents[1]
            subprocess.run(['bash', str(project/'scripts/radarvla.sh'), 'validate', '--full',
                            '--manifest', str(output/'manifest.jsonl')], check=True, cwd=project)
            write_json(reports/'status.json', dict(status='COMPLETE', pid=os.getpid(),
                       manifest=str(output/'manifest.jsonl'), manifest_sha256=report['manifest_sha256']))
        except BaseException as exc:
            write_json(reports/'status.json', dict(status='FAILED', pid=os.getpid(), error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__ == '__main__':
    main()
