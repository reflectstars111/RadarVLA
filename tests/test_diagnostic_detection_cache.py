"""Diagnostic scans must use the same explicit domain as production scoring."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import sys
import pytest
import torch


def test_scan_preserves_domain_and_rejects_manifest_content_changes(tmp_path, monkeypatch):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    manifest = write_synthetic_dataset(tmp_path/'data',scenes_per_split=1,frames_per_scene=1)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row['episode_id'] = row['scene_id']
        row['category'] = 'synthetic'
        row['detection_annotation_domain'] = dict(type='radar_bev_sector',range_m=50.,
            azimuth_min_rad=-.925,azimuth_max_rad=.925)
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config = ModelConfig(use_detection=True)
    model = RadarVLAGrounder(config)
    checkpoint = tmp_path/'model.pt'
    torch.save(dict(grounder=model.state_dict(),protocol=dict(config=dict(model=asdict(config)))),checkpoint)
    path = Path(__file__).resolve().parents[1]/'scripts/diagnose_detection.py'
    spec = importlib.util.spec_from_file_location('scan',path)
    scan = importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)
    out=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',[str(path),'--manifest',str(manifest),'--checkpoints',str(checkpoint),'--output-dir',str(out)])
    scan.main()
    cache=torch.load(next(out.glob('*raw.pt')),weights_only=True)
    assert {'radar_pose','detection_annotation_domain','detection_annotation_domain_mask'} <= cache['rows'][0]['truth'].keys()
    result=json.loads(next(out.glob('*sweep.json')).read_text())
    assert result['thresholds']['0.9']['overall']['annotation_domain_samples']==1
    assert result['thresholds']['0.9']['samples'][0]['sample_id'] in result['sample_ids']
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='provenance'):scan.main()
