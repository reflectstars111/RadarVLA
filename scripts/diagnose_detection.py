"""Cache validation detector candidates and sweep objectness without retraining."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import torch

from radar_vla.data import RadarDataset, collate_batch
from radar_vla.detection import decode_detections
from radar_vla.detection_metrics import DetectionMetrics
from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.pipeline import file_digest, to_device


KEYS = ('detection_logits', 'detection_class_logits', 'detection_state',
        'detection_size', 'detection_heading', 'detection_future',
        'detection_future_velocity')
TRUTH = ('agent_mask', 'agent_state', 'agent_supervision_mask', 'agent_class',
         'agent_class_mask', 'agent_future', 'agent_future_mask', 'future_times_s',
         'detection_annotation_domain', 'detection_annotation_domain_mask', 'radar_pose',
         'detection_class_schema_id', 'detection_ignore_boxes', 'detection_ignore_mask',
         'detection_ignore_margin_m')
THRESHOLDS = (.5, .7, .9, .95, .99)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RadarDataset(args.manifest, 'val')
    provenance = dict(manifest_sha256=file_digest(args.manifest),
                      scoring_sha256={name: file_digest(Path(__file__).resolve().parents[1] / name)
                          for name in ('scripts/diagnose_detection.py', 'src/detection_metrics.py',
                                       'src/annotation_domain.py', 'src/detection.py', 'src/data.py')},
                      schema='domain_detection_audit_v2')
    for checkpoint in args.checkpoints:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = ModelConfig(**state['protocol']['config']['model'])
        if dataset.class_schema != config.detection_class_schema:
            raise ValueError('Checkpoint and diagnostic dataset class schema differ')
        tag = f'{checkpoint.parent.name}_{checkpoint.stem}'
        cache_path = args.output_dir / f'{tag}_raw.pt'
        digest = file_digest(checkpoint)
        if cache_path.exists():
            saved = torch.load(cache_path, map_location='cpu', weights_only=True)
            if (saved['checkpoint_sha256'] != digest or saved['manifest'] != str(args.manifest.resolve())
                    or saved.get('provenance') != provenance):
                raise ValueError('candidate cache provenance differs from input')
            cache = saved['rows']
        else:
            model = RadarVLAGrounder(config).to(args.device).eval()
            model.load_state_dict(state['grounder'])
            cache = []
            for i, record in enumerate(dataset.records):
                batch = collate_batch([dataset[i]])
                with torch.no_grad():
                    output = model(to_device(batch, args.device))
                eligible = output['detection_logits'][0].sigmoid() >= min(THRESHOLDS)
                predicted = {key: output[key][0, eligible].detach().cpu() for key in KEYS}
                truth = {key: batch[key].detach().cpu() for key in TRUTH if key in batch}
                cache.append(dict(sample_id=record['sample_id'], episode_id=record['episode_id'],
                                  category=record['category'],
                                  speed_group=('moving' if float(torch.linalg.vector_norm(batch['ego_velocity'][0])) >= 1 else 'slow'),
                                  turn_group=('turning' if abs(float(record['ego']['yaw_rate'])) >= .05 else 'straight'),
                                  prediction=predicted, truth=truth))
                if (i + 1) % 100 == 0:
                    print(f'{tag}: {i+1}/{len(dataset)}', flush=True)
            torch.save(dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
                            manifest=str(args.manifest.resolve()), config=state['protocol']['config']['model'],
                            provenance=provenance, rows=cache), cache_path)
        result = dict(checkpoint=str(checkpoint.resolve()), cache=str(cache_path),
                      checkpoint_sha256=digest, provenance=provenance,
                      sample_ids=[row['sample_id'] for row in cache], thresholds={})
        metrics = {threshold: DetectionMetrics(config.detection_match_distance_m, config.detection_class_schema)
                   for threshold in THRESHOLDS}
        episode_metrics = {threshold: defaultdict(lambda: DetectionMetrics(config.detection_match_distance_m, config.detection_class_schema))
                           for threshold in THRESHOLDS}
        group_metrics = {threshold: defaultdict(lambda: DetectionMetrics(config.detection_match_distance_m, config.detection_class_schema))
                         for threshold in THRESHOLDS}
        sample_metrics = {threshold: [] for threshold in THRESHOLDS}
        base_config = replace(config, detection_confidence=min(THRESHOLDS))
        for i, row in enumerate(cache):
            prediction = {key: value.unsqueeze(0) for key, value in row['prediction'].items()}
            base = decode_detections(prediction, base_config, row['truth']['future_times_s'])[0]
            for threshold in THRESHOLDS:
                detections = [item for item in base['detections'] if item['confidence'] >= threshold]
                counts = {name: sum(item['category'] == name for item in detections)
                          for name in (*base['class_counts'],)}
                decoded = [dict(base, detections=detections, num_agents=len(detections),
                                num_vehicles=counts.get('vehicle'), class_counts=counts)]
                metrics[threshold].update(decoded, row['truth'])
                episode_metrics[threshold][row['episode_id']].update(decoded, row['truth'])
                for key in ('category', 'speed_group', 'turn_group'):
                    group_metrics[threshold][f'{key}:{row[key]}'].update(decoded, row['truth'])
                one = DetectionMetrics(config.detection_match_distance_m, config.detection_class_schema)
                one.update(decoded, row['truth'])
                sample_metrics[threshold].append(dict(sample_id=row['sample_id'],
                    episode_id=row['episode_id'], **one.compute()))
            if (i + 1) % 100 == 0:
                print(f'{tag} NMS: {i+1}/{len(cache)}', flush=True)
        for threshold in THRESHOLDS:
            result['thresholds'][str(threshold)] = dict(overall=metrics[threshold].compute(),
                episodes={episode: value.compute() for episode, value in
                          sorted(episode_metrics[threshold].items())},
                groups={key: value.compute() for key, value in sorted(group_metrics[threshold].items())},
                samples=sample_metrics[threshold])
        (args.output_dir / f'{tag}_sweep.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(tag, {threshold: {k: round(value['overall'][k], 4) if value['overall'][k] is not None else None
                                 for k in ('precision', 'recall', 'f1', 'count_mae')}
                    for threshold, value in result['thresholds'].items()}, flush=True)


if __name__ == '__main__':
    main()
