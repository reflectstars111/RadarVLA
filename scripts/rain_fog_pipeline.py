"""Explicit, locked rain/fog batch orchestration; never launches training."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'collection/collector'))
import batch
from agent_policy import POLICY, label_exclusion_reason
from road_map import actor_intersects_fov, radar_pose


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(manifest):
    files = [Path(__file__).resolve(), PROJECT/'scripts/collect_rain_fog_10k.sh']
    files += sorted((PROJECT/'src').glob('*.py'))
    files += sorted((PROJECT/'collection/collector').glob('*.py'))
    return hashlib.sha256((batch.signature(manifest) + batch.digest(files)).encode()).hexdigest()


def run(command):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=str(PROJECT), check=True)


def resources(config):
    runtime = config['runtime']
    if runtime.get('xvfb') and (not os.access(runtime['xvfb'], os.X_OK) or not Path(runtime['vulkan_icd']).is_file()):
        raise RuntimeError('Batch-local virtual display / Vulkan runtime missing')
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    state = {int(p[0]): (int(p[1]), int(p[2])) for p in (line.split(',') for line in out.splitlines())}
    selected = [runtime['carla_device']] + runtime['radar_devices']
    if len(set(runtime['radar_devices'])) != len(runtime['radar_devices']):
        raise ValueError('Radar worker GPUs must be distinct')
    # CARLA closes before radar processing; these stages may share a free GPU.
    busy = {i: state.get(i) for i in selected if i not in state or state[i][0] > 2048 or state[i][1] > 15}
    if busy:
        raise RuntimeError('Selected GPUs are busy; stop without disturbing jobs: '+str(busy))
    if batch.port_open(runtime['host'], runtime['port']):
        raise RuntimeError('CARLA port occupied; refusing to control another server')
    if batch.port_open(runtime['host'], runtime['tm_port']):
        raise RuntimeError('TrafficManager port occupied')
    return {'gpus': {i: state[i] for i in selected}, 'ports_available': True}


def audit_raw(raw, manifest):
    """Independently compare every retained label ID against immutable snapshots."""
    counts, groups, splits = Counter(), Counter(), Counter()
    episodes, seen, boundary = {}, set(), 0
    for record in rows(manifest):
        key = (record['episode_id'], record['carla_frame'])
        if key in seen:
            raise ValueError('Duplicate prepared anchor')
        seen.add(key)
        ep = raw/'jobs'/record['scene_id']/'episode'
        metadata = read(ep/'metadata.json')
        idx = int(Path(record['sensors']['radar']['power'][-1]).name.split('_')[0])
        current = read(ep/'frames'/('%06d'%idx)/'frame.json')
        if current['episode_id'] != key[0] or current['carla_frame'] != key[1]:
            raise ValueError('Raw/prepared frame mismatch')
        pos, rotation = radar_pose(current['ego']['pose'], metadata['sensors']['semantic_lidar'])
        expected, excluded = set(), set()
        for actor in current['actors']:
            if not actor_intersects_fov(actor, pos, rotation, 50.):
                continue
            reason = label_exclusion_reason(actor, pos)
            if reason is None:
                expected.add(str(actor['actor_id']))
            elif reason == 'range_boundary':
                boundary += 1
                excluded.add(str(actor['actor_id']))
        if record.get('agent_selection_policy') != POLICY or {str(a['id']) for a in record['agents']} != expected:
            raise ValueError('Raw moving/full-BBox membership mismatch')
        if not excluded.issubset({str(a['id']) for a in record.get('background_agents', []) if a['reason']=='range_boundary'}):
            raise ValueError('Missing boundary background audit')
        if episodes.setdefault(key[0], record['split']) != record['split']:
            raise ValueError('Episode crosses split')
        counts['samples'] += 1
        counts['agents'] += len(expected)
        counts['grounding_available'] += bool(record['grounding_available'])
        counts['sft_available'] += bool(record['sft_available'])
        groups[record['category']+'/'+record['weather']] += 1
        splits[record['split']] += 1
    if not counts['samples']:
        raise ValueError('No prepared records')
    return {'status':'PASS', 'counts':dict(counts), 'category_weather':dict(groups),
            'splits':dict(splits), 'range_boundary_background_instances':boundary,
            'raw_evidence_checked':True, 'manifest_sha256':sha(manifest)}


def prepare_validate(config, raw, output, reports, *, sample=None):
    python = config['pipeline']['python']
    cmd = [python, '-m', 'radar_vla.collection_adapter', '--data-root', config['runtime']['data_root'],
           '--raw-root', raw, '--output-dir', output, '--moving-agents']
    if sample is not None:
        cmd += ['--sample', str(sample)]
    run(cmd)
    manifest = output/'manifest.jsonl'
    run([python, '-m', 'radar_vla', 'validate', '--manifest', manifest])
    report = audit_raw(raw, manifest)
    batch.atomic_json(output/'validation.json', report)
    batch.atomic_json(reports/'acceptance.json', report)
    return report


def verify_release(config, manifest, reports):
    stamp = signature(manifest)
    release = read(reports/'smoke_release.json')
    if release.get('status') != 'PASS' or release.get('signature') != stamp:
        raise ValueError('Matching PASS smoke is required; run smoke first')
    for path, digest in release['evidence_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Smoke evidence changed: '+path)
    return release


def smoke(config, manifest, reports):
    stamp = signature(manifest)
    release_root = reports
    reports = reports/'smoke_attempts'/stamp[:12]
    reports.mkdir(parents=True, exist_ok=True)
    journal = reports/'smoke_input.json'
    if journal.exists() and read(journal)['signature'] != stamp:
        raise ValueError('Smoke source/config changed; preserve previous attempt and use a new smoke attempt')
    batch.atomic_json(journal, {'signature':stamp})
    output = reports/'smoke'
    output.mkdir(parents=True, exist_ok=True)
    pilot_path = output/'scenario2_pilot/episode/metadata.json'
    if not (reports/'qualification/scenario2_pilot.json').exists():
        with batch.CarlaServer(config, output):
            batch.prepare_scenario2_for_smoke(config, output, reports)
    else:
        batch.prepare_scenario2_for_smoke(config, output, reports)
    accepted, failures = batch.collect(config, 'smoke', output)
    if failures or len(accepted) < config['smoke_min_anchors']:
        raise RuntimeError('Smoke collection failed or insufficient anchors: '+str(failures))
    products, failures = batch.process_radar(config, output, accepted)
    if failures:
        raise RuntimeError('Smoke radar failed: '+str(failures))
    # Audit needs the radar interpreter for the C-Shenron dependencies.
    audit_cmd = [config['runtime']['radar_python'], PROJECT/'scripts/rain_fog_pipeline.py',
                 '_radar-audit', '--manifest', manifest]
    run(audit_cmd)
    radar_report = read(reports/'smoke_radar.json')
    if radar_report['status'] != 'PASS':
        raise RuntimeError('Smoke radar audit did not pass')
    acceptance = prepare_validate(config, output, output/'prepared', reports, sample=config['smoke_min_anchors'])
    preview(config, output, reports)
    evidence = [output/'rendering.json', reports/'smoke_radar.json', reports/'acceptance.json', output/'prepared/manifest.jsonl',
                output/'prepared/validation.json', reports/'qualification/scenario2_pilot.json']
    batch.atomic_json(release_root/'smoke_release.json', {'status':'PASS', 'signature':stamp,
        'reports':str(reports),
        'accepted_raw':len(accepted), 'prepared':acceptance['counts']['samples'],
        'evidence_sha256':{str(p):sha(p) for p in evidence}})
    print('SMOKE PASS; full 10k collection has NOT been started.', flush=True)


def preview(config, raw, reports):
    # Existing joint viewer renders synchronized radar/map/true trajectories.
    chosen = {}
    for row in rows(raw/'dataset_index.jsonl'):
        chosen.setdefault(row['weather'], row)
    for weather, row in chosen.items():
        ep = raw/'jobs'/row['job_id']/'episode'
        import matplotlib
        matplotlib.use('Agg')
        from joint_viewer import JointPreview, render
        import matplotlib.pyplot as plt
        view = JointPreview(ep, Path(row['radar_product']))
        fig = render(view, row['radar_frame_index'])
        fig.savefig(str(reports/('preview_'+weather+'.png')), dpi=120, bbox_inches='tight')
        plt.close(fig)
        import shutil
        shutil.copyfile(ep/'frames'/('%06d'%row['source_index'])/'front_rgb.png', reports/('front_'+weather+'.png'))


def full(config, manifest, reports, resume):
    release = verify_release(config, manifest, reports)
    stamp = signature(manifest)
    raw = Path(config['pipeline']['raw_root'])
    prepared = Path(config['pipeline']['prepared_root'])
    journal = raw/'input_signature.json'
    if raw.exists():
        if not resume:
            raise FileExistsError('Batch exists; use resume')
        if not journal.exists() or read(journal)['signature'] != stamp:
            raise ValueError('Batch signature changed; refusing resume')
    else:
        raw.mkdir(parents=True)
        batch.atomic_json(journal, {'signature':stamp})
    # Validate the stronger pipeline release first, then the real S2 certificate.
    batch.preflight(config, manifest, Path(release['reports']), 'full', skip_smoke=True)
    # The stronger pipeline release above binds sensor validation and all code.
    while True:
        accepted, failures = batch.collect(config, 'full', raw)
        deficit = batch.deficits(accepted, config)
        batch.atomic_json(raw/'collection_progress.json', {'accepted':len(accepted), 'deficits':deficit,
            'historical_failed_jobs':len(list((raw/'jobs').glob('*/failure.json')))})
        if not deficit['total']:
            break
        config['runtime']['repetitions'] += config['pipeline']['seed_round_size']
        if config['runtime']['repetitions'] > config['pipeline']['maximum_seed_rounds']:
            raise RuntimeError('Candidate seed budget exhausted; inspect quota deficits')
        print('Extending deterministic candidate seeds to '+str(config['runtime']['repetitions']), flush=True)
    joint = Counter(r['category']+'/'+r['weather'] for r in accepted)
    if any(joint[k] != n for k, n in config['quotas']['category_weather'].items()) or any(any(v.values()) for k,v in deficit.items() if isinstance(v,dict)):
        raise ValueError('Final quota mismatch')
    products, radar_failures = batch.process_radar(config, raw, accepted)
    if radar_failures or len(products) != 10000:
        raise RuntimeError('Radar index incomplete; resume after inspecting radar failures')
    batch.atomic_json(raw/'status.json', {'status':'RAW_RADAR_COMPLETE','accepted':len(accepted),
        'radar_products':len(products), 'deficits':deficit, 'signature':stamp})
    prepare_validate(config, raw, prepared, reports)
    batch.atomic_json(reports/'status.json', {'status':'COMPLETE','manifest':str(prepared/'manifest.jsonl'),
                                           'training_started':False})


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Pipeline received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check','status','smoke','run','resume','_radar-audit'])
    parser.add_argument('--manifest', type=Path, default=PROJECT/'collection/configs/collection_rain_fog_10k_v1.yaml')
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    config = batch.load_manifest(manifest)
    reports = Path(config['pipeline']['reports_root'])
    if args.action == 'status':
        for path in (reports/'status.json', reports/'smoke_release.json', reports/'smoke_radar.json',
                     Path(config['pipeline']['raw_root'])/'collection_progress.json',
                     Path(config['pipeline']['prepared_root'])/'report.json'):
            print(str(path)+': '+(path.read_text() if path.exists() else 'not started'))
        return
    if args.action == '_radar-audit':
        reports = reports/'smoke_attempts'/signature(manifest)[:12]
        raw = reports/'smoke'
        batch.prepare_scenario2_for_smoke(config, raw, reports)
        accepted = list(rows(raw/'accepted_anchors.jsonl'))
        products = list(rows(raw/'dataset_index.jsonl'))
        report = batch.smoke_report(config, raw, accepted, products)
        batch.atomic_json(reports/'smoke_radar.json', report)
        preview(config, raw, reports)
        return
    check = batch.preflight(config, manifest, Path(config['runtime']['data_root']), 'smoke')
    batch.frame_quota(config)
    if args.action == 'check':
        readiness = resources(config)
        try:
            verify_release(config, manifest, reports)
            readiness['smoke'] = 'PASS'
        except (OSError, ValueError) as exc:
            readiness['smoke'] = str(exc)
        print(json.dumps(readiness, indent=2))
        return
    reports.mkdir(parents=True, exist_ok=True)
    with (reports/'pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another rain/fog pipeline is active')
        resources(config)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['runtime']['carla_device'])
        batch.atomic_json(reports/'status.json', {'status':'RUNNING', 'phase':args.action,'pid':os.getpid()})
        try:
            if args.action == 'smoke':
                smoke(config, manifest, reports)
                batch.atomic_json(reports/'status.json', {'status':'SMOKE_PASS','full_collection_started':False})
            else:
                full(config, manifest, reports, args.action=='resume')
        except BaseException as exc:
            batch.atomic_json(reports/'status.json', {'status':'FAILED','phase':args.action,
                'error':type(exc).__name__+': '+str(exc), 'pid':os.getpid()})
            raise


if __name__ == '__main__':
    main()
