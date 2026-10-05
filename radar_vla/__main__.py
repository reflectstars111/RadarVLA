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
    check = sub.add_parser('validate', help='Validate metadata, arrays, masks and split isolation')
    check.add_argument('--manifest', type=Path, required=True)
    check.add_argument('--max-agents', type=int, default=8)
    prepare = sub.add_parser('prepare', help='Cache counterfactual risk labels in a new manifest')
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--safety-margin-m', type=float, default=0.5)
    train = sub.add_parser('train', help='Train grounding or prototype risk-conditioned SFT')
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
            command.add_argument('--max-new-tokens', type=int, default=512)
    smoke = sub.add_parser('smoke', help='Synthetic CPU end-to-end check; not a quality experiment')
    smoke.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    options = vars(args).copy()
    command = options.pop('command')
    if command == 'synthetic':
        from .synthetic import write_synthetic_dataset
        result = {'manifest': str(write_synthetic_dataset(**options))}
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
