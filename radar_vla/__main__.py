"""Run with python -m radar_vla --help; no dataset/model downloads or GPU default."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent RadarVLA research pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    synthetic = sub.add_parser('synthetic', help='Write scene-disjoint synthetic RA and annotations')
    synthetic.add_argument('--output', type=Path, required=True)
    synthetic.add_argument('--scenes-per-split', type=int, default=2)
    synthetic.add_argument('--frames-per-scene', type=int, default=2)
    synthetic.add_argument('--seed', type=int, default=42)
    synthetic.add_argument('--frames', type=int, default=4)
    synthetic.add_argument('--range-bins', type=int, default=256)
    synthetic.add_argument('--azimuth-bins', type=int, default=107)
    synthetic.add_argument('--horizon-steps', type=int, default=6)
    raw = sub.add_parser('synthetic-frames', help='Write a complete frame_t format fixture, including actual image/point-cloud assets')
    raw.add_argument('--output', type=Path, required=True)
    raw.add_argument('--scenes-per-split', type=int, default=1)
    raw.add_argument('--frames-per-scene', type=int, default=6)
    raw.add_argument('--seed', type=int, default=42)
    raw.add_argument('--range-bins', type=int, default=256)
    raw.add_argument('--azimuth-bins', type=int, default=107)
    check = sub.add_parser('validate', help='Validate metadata, arrays, masks and split isolation')
    check.add_argument('--manifest', type=Path, required=True)
    check.add_argument('--max-agents', type=int, default=8)
    prepare = sub.add_parser('prepare', help='Cache counterfactual risk labels in a new manifest')
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--safety-margin-m', type=float, default=0.5)
    frames = sub.add_parser('prepare-frames', help='Load calibrated frame_t records and construct causal history windows')
    frames.add_argument('--input-jsonl', type=Path, required=True)
    frames.add_argument('--output-dir', type=Path, required=True)
    frames.add_argument('--history-frames', type=int, default=4)
    frames.add_argument('--future-times-s', type=float, nargs='+', default=[.5, 1., 1.5, 2., 2.5, 3.])
    frames.add_argument('--max-gap-s', type=float, default=.5)
    frames.add_argument('--max-sensor-skew-s', type=float, default=.05)
    frames.add_argument('--max-future-gap-s', type=float, default=1.1)
    frames.add_argument('--observations-only', action='store_true', help='No future/agent supervision required or synthesized')
    cohort = sub.add_parser('build-cohort', help='Explicitly select a common annotated cohort; save all exclusion reasons')
    cohort.add_argument('--manifest', type=Path, required=True)
    cohort.add_argument('--output', type=Path, required=True)
    cohort.add_argument('--config', type=Path)
    cohort.add_argument('--require-oracle', action='store_true')
    train = sub.add_parser('train', help='Train physical grounding or risk-conditioned Qwen SFT')
    train.add_argument('--manifest', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--stage', choices=['grounding', 'sft'], default='grounding')
    train.add_argument('--config', type=Path)
    train.add_argument('--epochs', type=int, default=5)
    train.add_argument('--batch-size', type=int, default=2)
    train.add_argument('--lr', type=float, default=3e-4)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--device', default='cpu')
    train.add_argument('--init-grounding', type=Path)
    train.add_argument('--resume', action='store_true')
    train.add_argument('--stop-after-epoch', type=int)
    train.add_argument('--gt-risk-mix', type=float, default=0.)
    train.add_argument('--precision', choices=['float32', 'bfloat16'], default='float32')
    train.add_argument('--accumulation-steps', type=int, default=1)
    train.add_argument('--workers', type=int, default=0)
    train.add_argument('--language-model-path', type=Path)
    for name in ('evaluate', 'predict'):
        command = sub.add_parser(name)
        command.add_argument('--checkpoint', type=Path, required=True)
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--split', choices=['train', 'val', 'test'], default='test')
        command.add_argument('--batch-size', type=int, default=2)
        command.add_argument('--device', default='cpu')
        if name == 'predict':
            command.add_argument('--max-new-tokens', type=int, default=1024)
        else:
            command.add_argument('--risk-source', choices=['predicted', 'none', 'oracle'])
    matrix = sub.add_parser('plan-experiments', help='Write paired Q1/Q2/Q3/Q5 protocol; does not launch training')
    matrix.add_argument('--config', type=Path, required=True)
    matrix.add_argument('--manifest', type=Path, required=True)
    matrix.add_argument('--output', type=Path, required=True, help='Future experiment output root')
    matrix.add_argument('--plan-file', type=Path, required=True)
    matrix.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    matrix.add_argument('--epochs', type=int, default=5)
    matrix.add_argument('--batch-size', type=int, default=1)
    matrix.add_argument('--accumulation-steps', type=int, default=4)
    matrix.add_argument('--world-size', type=int, default=8)
    matrix.add_argument('--lr', type=float, default=3e-4)
    execute = sub.add_parser('run-experiments', help='Explicitly execute the frozen paired training plan')
    execute.add_argument('--plan-path', type=Path, required=True)
    execute.add_argument('--device', default='cuda')
    execute.add_argument('--resume', action='store_true')
    sweep = sub.add_parser('doppler-sweep', help='Q4: change observed Doppler while preserving other inputs')
    for name in ('checkpoint', 'manifest', 'output'):
        sweep.add_argument('--' + name, type=Path, required=True)
    sweep.add_argument('--sample-id', required=True)
    sweep.add_argument('--range-interval', type=float, nargs=2, required=True)
    sweep.add_argument('--azimuth-interval', type=float, nargs=2, required=True)
    sweep.add_argument('--velocities', type=float, nargs='+', default=[-2., -5., -8.])
    sweep.add_argument('--history', choices=['current', 'all'], default='current')
    sweep.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    sweep.add_argument('--device', default='cpu')
    smoke = sub.add_parser('smoke', help='Synthetic CPU end-to-end check; not a quality experiment')
    smoke.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    options = vars(args).copy()
    command = options.pop('command')
    if command == 'synthetic':
        from .synthetic import write_synthetic_dataset
        result = {'manifest': str(write_synthetic_dataset(**options))}
    elif command == 'synthetic-frames':
        from .synthetic import write_synthetic_frames
        result = {'manifest': str(write_synthetic_frames(**options))}
    elif command == 'validate':
        from .data import RadarDataset, load_manifest
        records = load_manifest(args.manifest)
        counts = {}
        for split in sorted({r['split'] for r in records}):
            dataset = RadarDataset(args.manifest, split, max_agents=args.max_agents)
            for i in range(len(dataset)):
                dataset[i]
            counts[split] = len(dataset)
        result = {'status': 'valid', 'samples': counts}
    elif command == 'prepare':
        from .data import prepare_labels
        if args.output.exists():
            raise ValueError('Prepared manifest output already exists')
        result = {'manifest': str(prepare_labels(**options))}
    elif command == 'prepare-frames':
        from .records import prepare_frames
        options['supervision'] = not options.pop('observations_only')
        result = {'manifest': str(prepare_frames(**options))}
    elif command == 'plan-experiments':
        from .experiments import build_experiment_plan
        from .pipeline import write_json
        plan_file = options.pop('plan_file')
        options['config'] = json.loads(args.config.read_text())
        result = build_experiment_plan(**options)
        if plan_file.exists():
            raise ValueError('Plan file already exists; use a new file')
        write_json(plan_file, result)
        result = {'plan_file': str(plan_file), 'tasks': len(result['tasks']), 'launched': False}
    elif command == 'build-cohort':
        from .cohort import build_cohort
        options['config'] = json.loads(args.config.read_text()) if args.config else None
        result = build_cohort(**options)
    elif command == 'run-experiments':
        from .experiments import run_experiment_plan
        result = run_experiment_plan(**options)
    elif command == 'doppler-sweep':
        from .experiments import doppler_sweep
        result = doppler_sweep(**options)
    elif command == 'train':
        from .pipeline import run_training
        options['config'] = json.loads(args.config.read_text()) if args.config else None
        result = run_training(**options)
    elif command == 'evaluate':
        from .pipeline import evaluate_checkpoint
        result = evaluate_checkpoint(**options)
    elif command == 'predict':
        from .pipeline import predict_checkpoint
        rows = predict_checkpoint(**options)
        result = {'samples': len(rows), 'output': str(args.output)}
    else:
        from .pipeline import run_smoke
        result = run_smoke(**options)
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
