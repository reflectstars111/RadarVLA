"""Single operational entry point for collection, preparation and model stages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def _resolve(root, value):
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _json(path):
    return json.loads(path.read_text())


def _latest(data_root, mode):
    choices = [p for p in (data_root/'prepared').glob(f'{mode}_v3_*/manifest.jsonl') if p.is_file()]
    return max(choices, key=lambda p: p.stat().st_mtime) if choices else None


def _lines(path):
    if not path.is_file():
        return None
    with path.open() as stream:
        return sum(1 for line in stream if line.strip())


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    from bottleneck_campaign import add_arguments, run_campaign
    add_arguments(commands.add_parser('bottleneck', help='Reproducible eight-GPU bottleneck comparisons'))
    for name in ('doctor','status','collect-smoke','collect-full','prepare','validate','train-grounding','train-sft'):
        sub = commands.add_parser(name)
        if name == 'collect-full':
            sub.add_argument('--resume', action='store_true')
        if name == 'prepare':
            choice = sub.add_mutually_exclusive_group(required=True)
            choice.add_argument('--sample', type=int)
            choice.add_argument('--full', action='store_true')
            sub.add_argument('--reuse-previous', action='store_true')
            sub.add_argument('--legacy-labels', action='store_true', help='Reproduce historical labels rather than the current motion policy')
        if name == 'validate':
            choice = sub.add_mutually_exclusive_group(required=True)
            choice.add_argument('--sample', action='store_true')
            choice.add_argument('--full', action='store_true')
            choice.add_argument('--manifest', type=Path)
        if name.startswith('train-'):
            sub.add_argument('--dry-run', action='store_true')
            sub.add_argument('--resume', action='store_true')
            sub.add_argument('--gpus', help='Comma-separated physical GPU indices, for example 1,3,6')
            sub.add_argument('--manifest', type=Path)
            sub.add_argument('--config', type=Path, help='Model configuration override (default: local_paths.json model_config)')
            sub.add_argument('--output', type=Path)
            sub.add_argument('--init-grounding', type=Path)
            if name == 'train-grounding':
                sub.add_argument('--risk-readout-only', action='store_true', default=None)
            if name == 'train-sft':
                sub.add_argument('--freeze-grounder', action='store_true', default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    paths = _json(args.paths)
    python = _resolve(root, paths['python'])
    data_root = _resolve(root, paths['data_root'])
    collection = _resolve(root, paths['collection_manifest'])
    model_config = _resolve(root, paths['model_config'])
    if getattr(args, 'config', None) is not None:
        model_config = args.config.resolve()
    qwen = _resolve(root, paths['qwen_model'])
    run_root = _resolve(root, paths['run_root'])
    if args.command == 'bottleneck':
        return run_campaign(args, root, data_root, run_root)
    full = data_root/'full'
    if args.command in ('doctor','status'):
        if args.command == 'doctor':
            checks = {'python': python.is_file(), 'model_config': model_config.is_file(),
                      'collection_manifest': collection.is_file(), 'qwen': (qwen/'config.json').is_file(),
                      'data_root': data_root.is_dir(), 'editable_import': False}
            missing_collection_inputs = []
            if collection.is_file():
                manifest = _json(collection)
                runtime = manifest['runtime']
                required = [runtime['carla_python'], runtime['radar_python'], runtime['carla_server'],
                            *[row['xml'] for row in manifest['routes']],
                            *manifest['scenarios'].values(), manifest['scenario2']['route_xml']]
                for value in required:
                    path = Path(value)
                    path = path if path.is_absolute() else root.parent/path
                    if not path.is_file():
                        missing_collection_inputs.append(str(path))
                towns = {row['town'] for row in manifest['routes']} | {manifest['scenario2']['town']}
                maps = root.parent/'Carla/CarlaUE4/Content/Carla/Maps'
                for town in towns:
                    path = maps/f'{town}.umap'
                    if not path.is_file():
                        missing_collection_inputs.append(str(path))
            checks['collection_inputs'] = not missing_collection_inputs
            probe = subprocess.run([str(python),'-I','-c','import radar_vla; print(radar_vla.__file__)'],
                                   capture_output=True, text=True, cwd='/tmp')
            checks['editable_import'] = probe.returncode == 0 and probe.stdout.strip() == str(root/'src/__init__.py')
            print(json.dumps({'checks': checks, 'missing_collection_inputs': missing_collection_inputs,
                              'scenario2_certificate': (full/'qualification/scenario2_pilot.json').is_file(),
                              'import_path': probe.stdout.strip(),
                              'environment': str(python), 'data_root': str(data_root)}, indent=2))
            return 0 if all(checks.values()) else 1
        status = _json(full/'status.json') if (full/'status.json').is_file() else {'status':'IN_PROGRESS_OR_UNKNOWN'}
        collection_report = _json(full/'collection_remaining_status.json') if (full/'collection_remaining_status.json').is_file() else None
        radar_report = _json(full/'radar_existing_status.json') if (full/'radar_existing_status.json').is_file() else None
        phase = ('complete' if status.get('status') == 'RAW_RADAR_COMPLETE' else
                 'radar_or_final_index_pending' if collection_report and collection_report.get('status') == 'QUOTAS_FILLED' else
                 'collection_or_preflight_pending')
        print(json.dumps({'phase':phase, 'full_status':status, 'accepted_anchors':_lines(full/'accepted_anchors.jsonl'),
                          'collection_status':collection_report, 'radar_status':radar_report,
                          'dataset_index':_lines(full/'dataset_index.jsonl'),
                          'sample_manifest':str(_latest(data_root,'sample')) if _latest(data_root,'sample') else None,
                          'full_manifest':str(_latest(data_root,'full')) if _latest(data_root,'full') else None}, indent=2))
        return 0
    if args.command.startswith('collect-'):
        if not collection.is_file() or not data_root.is_dir():
            raise FileNotFoundError('collection manifest or data root is missing')
        if args.command == 'collect-full' and not args.resume:
            raise ValueError('collect-full requires --resume so an existing batch cannot be restarted accidentally')
        filled_path = full/'collection_remaining_status.json'
        if args.command == 'collect-full' and filled_path.is_file() and _json(filled_path).get('status') == 'QUOTAS_FILLED':
            completed_path = full/'status.json'
            if completed_path.is_file() and _json(completed_path).get('status') == 'RAW_RADAR_COMPLETE':
                print('full raw/radar batch is already complete')
                return 0
            radar_command = [str(python),str(root/'collection/collector/radar_existing.py'),
                             '--manifest',str(collection),'--output',str(full)]
            code = subprocess.call(radar_command,cwd=root)
            if code:
                return code
            return subprocess.call([str(python),str(root/'collection/collector/finish_full.py'),
                                    '--manifest',str(collection)],cwd=root)
        mode = 'smoke' if args.command == 'collect-smoke' else 'full'
        cmd = [str(python),str(root/'collection/collector/batch.py'),'--manifest',str(collection),'--mode',mode]
        if mode == 'full':
            cmd.append('--resume')
        return subprocess.call(cmd, cwd=root)
    if args.command == 'prepare':
        cmd = [str(python),'-m','radar_vla.collection_adapter','--data-root',str(data_root)]
        if not args.legacy_labels:
            if args.reuse_previous:
                raise ValueError('Motion labels must be rebuilt; --reuse-previous requires explicit --legacy-labels')
            cmd.append('--moving-agents')
        if args.sample is not None:
            if args.reuse_previous:
                raise ValueError('--reuse-previous applies only to --full')
            cmd += ['--sample',str(args.sample)]
        if args.reuse_previous:
            candidates = [p.parent for p in (data_root/'prepared').glob('full_v3_*/report.json')
                          if _json(p).get('status') == 'PREPARED' and
                          not _json(p).get('reuse_source') and
                          (p.parent/'manifest.jsonl').is_file()]
            if not candidates:
                raise RuntimeError('no completed prior full prepared manifest; wait for the current run to finish')
            previous = max(candidates, key=lambda p: (p/'report.json').stat().st_mtime)
            print(f'reusing compatible successful records from {previous}', flush=True)
            cmd += ['--reuse-prepared',str(previous)]
        return subprocess.call(cmd,cwd=root)
    if args.command == 'validate':
        mode = 'sample' if args.sample else 'full'
        manifest = args.manifest.resolve() if args.manifest else _latest(data_root,mode)
        if manifest is None:
            raise FileNotFoundError(f'no prepared {mode} manifest')
        result = subprocess.run([str(python),'-m','radar_vla','validate','--manifest',str(manifest)],
                                cwd=root,capture_output=True,text=True)
        if result.stdout:
            print(result.stdout,end='')
        if result.stderr:
            print(result.stderr,end='',file=sys.stderr)
        if result.returncode:
            return result.returncode
        report = {'status':'PASS','manifest':str(manifest),'manifest_sha256':_digest(manifest),
                  'validated_at_utc':datetime.now(timezone.utc).isoformat(),
                  'result':json.loads(result.stdout)}
        target = manifest.parent/'validation.json'
        staged = target.with_suffix('.json.inprogress')
        staged.write_text(json.dumps(report,indent=2)+'\n')
        staged.replace(target)
        return 0
    stage = 'grounding' if args.command == 'train-grounding' else 'sft'
    manifest = args.manifest.resolve() if args.manifest else _latest(data_root,'full')
    output = args.output.resolve() if args.output else run_root/stage
    risk_readout = (getattr(args, 'risk_readout_only', None) or os.environ.get('RISK_READOUT_ONLY') == '1')
    init = args.init_grounding.resolve() if args.init_grounding else (
        run_root/'grounding/best.pt' if stage == 'sft' else None)
    if risk_readout and stage != 'grounding':
        raise ValueError('--risk-readout-only requires the grounding stage')
    if risk_readout and not args.resume and init is None:
        raise ValueError('--risk-readout-only requires --init-grounding')
    if not args.dry_run:
        if manifest is None or not manifest.is_file():
            raise FileNotFoundError('prepared full manifest is missing; run prepare --full')
        validation_path = manifest.parent/'validation.json'
        validation = _json(validation_path) if validation_path.is_file() else {}
        if validation.get('status') != 'PASS' or validation.get('manifest_sha256') != _digest(manifest):
            raise RuntimeError('full manifest has no matching PASS validation report; run validate --full')
        if not args.resume and (stage == 'sft' or risk_readout or init is not None) and (init is None or not init.is_file()):
            raise FileNotFoundError(f'grounding checkpoint is missing: {init}')
        if stage == 'sft':
            has_known_mode = False
            with manifest.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if any(record.get('tracking_coverage_valid', [])) or record.get('risk_label'):
                        has_known_mode = True
                        break
            if not has_known_mode:
                raise RuntimeError('SFT has no certified risk coverage/labels; adaptive SHORT/LONG teachers would all be masked')
    env = os.environ.copy()
    if args.gpus is not None:
        selected_gpus = [gpu.strip() for gpu in args.gpus.split(',')]
        if (not selected_gpus or len(selected_gpus) > 8 or
                any(not gpu.isascii() or not gpu.isdigit() or str(int(gpu)) != gpu
                    for gpu in selected_gpus) or
                len(set(selected_gpus)) != len(selected_gpus)):
            raise ValueError('--gpus requires 1–8 distinct nonnegative GPU indices, e.g. 1,3,6')
        env['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
        env['NPROC_PER_NODE'] = str(len(selected_gpus))
    env.update(MANIFEST=str(manifest or data_root/'prepared/full_v3_PENDING/manifest.jsonl'),
               OUTPUT=str(output), STAGE=stage, RADAR_VLA_PYTHON=str(python),
               RADAR_VLA_CONFIG=str(model_config), QWEN_MODEL_PATH=str(qwen),
               INIT_GROUNDING=str(init) if init else '', RESUME='1' if args.resume else '0',
               RISK_READOUT_ONLY='1' if risk_readout else '0')
    if getattr(args, 'freeze_grounder', None):
        env['FREEZE_GROUNDER'] = '1'
    cmd = ['bash',str(root/'launch_l40s.sh')]
    if args.dry_run:
        cmd.append('--dry-run')
    return subprocess.call(cmd,cwd=root,env=env)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(2)
