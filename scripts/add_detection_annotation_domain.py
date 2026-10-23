"""Add a verified finite detection annotation domain to a prepared manifest.

The command preserves every existing record and sensor path.  It reads the raw
episode metadata beside ``route_archive`` and refuses to infer collection
coverage when that metadata is absent or inconsistent.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


AZIMUTH_DEG = 53.0


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def metadata_path(record: dict) -> Path:
    archive = record.get('route_archive')
    if not isinstance(archive, str) or not archive:
        raise ValueError(f"{record.get('sample_id', '<unknown>')} has no route_archive")
    return Path(archive).resolve().with_name('metadata.json')


def verified_domain(record: dict, cache: dict[Path, tuple[str, dict]]) -> dict:
    path = metadata_path(record)
    if path not in cache:
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = json.loads(path.read_text())
        episode = metadata.get('episode_id')
        if not isinstance(episode, str) or not episode:
            raise ValueError(f'{path} has no episode_id')
        for field in ('agent_label_range_m', 'map_label_range_m'):
            value = metadata.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != 50.0:
                raise ValueError(f'{path} requires {field}=50, got {value!r}')
        cache[path] = episode, metadata
    episode, metadata = cache[path]
    if record.get('episode_id') != episode:
        raise ValueError(f"{record.get('sample_id')} episode_id differs from {path}")
    if record.get('detection_annotation_domain') is not None:
        raise ValueError(f"{record.get('sample_id')} already has a detection annotation domain")
    return {
        'type': 'radar_bev_sector',
        'range_m': float(metadata['agent_label_range_m']),
        'azimuth_min_rad': -math.radians(AZIMUTH_DEG),
        'azimuth_max_rad': math.radians(AZIMUTH_DEG),
        'source': 'raw episode metadata agent_label_range_m and collector bbox/FOV contract',
    }


def add_domains(source: Path, output_dir: Path) -> dict:
    source, output_dir = source.resolve(), output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise ValueError(f'output directory already exists: {output_dir}')
    output_dir.mkdir(parents=True)
    staged = output_dir / 'manifest.partial.jsonl'
    output = output_dir / 'manifest.jsonl'
    episodes: dict[Path, tuple[str, dict]] = {}
    samples: set[str] = set()
    splits, episode_splits = Counter(), {}
    count = 0
    try:
        with source.open() as incoming, staged.open('w') as outgoing:
            for line_number, line in enumerate(incoming, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample = record.get('sample_id')
                if not isinstance(sample, str) or not sample or sample in samples:
                    raise ValueError(f'invalid or duplicate sample_id at line {line_number}: {sample!r}')
                samples.add(sample)
                split, episode = record.get('split'), record.get('episode_id')
                if split not in ('train', 'val', 'test'):
                    raise ValueError(f'{sample} has invalid split {split!r}')
                if episode in episode_splits and episode_splits[episode] != split:
                    raise ValueError(f'episode {episode} crosses splits')
                episode_splits[episode] = split
                record['detection_annotation_domain'] = verified_domain(record, episodes)
                outgoing.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                          separators=(',', ':')) + '\n')
                splits[split] += 1
                count += 1
        if not count:
            raise ValueError('source manifest is empty')
        staged.replace(output)
        report = {
            'schema': 'radar_vla_detection_annotation_domain_migration_v1',
            'source_manifest': str(source),
            'source_sha256': digest(source),
            'manifest': str(output),
            'manifest_sha256': digest(output),
            'samples': count,
            'episodes': len(episode_splits),
            'splits': dict(sorted(splits.items())),
            'domain': {'type': 'radar_bev_sector', 'range_m': 50.0,
                       'azimuth_min_rad': -math.radians(AZIMUTH_DEG),
                       'azimuth_max_rad': math.radians(AZIMUTH_DEG)},
            'sensor_arrays': 'unchanged; all existing paths preserved',
        }
        (output_dir / 'migration_report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    except BaseException:
        if staged.exists():
            staged.unlink()
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(add_domains(args.source, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
