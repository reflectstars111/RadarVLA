import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'add_detection_annotation_domain.py'
SPEC = importlib.util.spec_from_file_location('add_detection_annotation_domain', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(tmp_path, *, label_range=50.0):
    episode = tmp_path / 'job' / 'episode'
    episode.mkdir(parents=True)
    (episode / 'route.json').write_text('{}')
    (episode / 'metadata.json').write_text(json.dumps({
        'episode_id': 'episode-a', 'agent_label_range_m': label_range,
        'map_label_range_m': label_range,
    }))
    source = tmp_path / 'manifest.jsonl'
    records = [
        {'sample_id': 'a', 'episode_id': 'episode-a', 'scene_id': 'job', 'split': 'val',
         'route_archive': str(episode / 'route.json'),
         'sensors': {'radar': {'power': ['/data/a.npy'], 'unfolded_doppler': ['/data/d.npy']}}},
        {'sample_id': 'b', 'episode_id': 'episode-a', 'scene_id': 'job', 'split': 'val',
         'route_archive': str(episode / 'route.json'),
         'sensors': {'radar': {'power': ['/data/b.npy'], 'unfolded_doppler': ['/data/e.npy']}}},
    ]
    source.write_text(''.join(json.dumps(row) + '\n' for row in records))
    return source, records


def test_add_domains_preserves_records_paths_and_splits(tmp_path):
    source, before = fixture(tmp_path)
    report = MODULE.add_domains(source, tmp_path / 'output')
    after = [json.loads(line) for line in Path(report['manifest']).read_text().splitlines()]
    assert report['samples'] == 2 and report['episodes'] == 1 and report['splits'] == {'val': 2}
    for old, new in zip(before, after):
        domain = new.pop('detection_annotation_domain')
        assert new == old
        assert domain['range_m'] == 50.0
        assert domain['azimuth_min_rad'] == pytest.approx(-math.radians(53))
        assert domain['azimuth_max_rad'] == pytest.approx(math.radians(53))


def test_add_domains_rejects_unverified_range(tmp_path):
    source, _ = fixture(tmp_path, label_range=100.0)
    with pytest.raises(ValueError, match='agent_label_range_m=50'):
        MODULE.add_domains(source, tmp_path / 'output')
