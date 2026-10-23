"""Danger attribution need not recompute irrelevant braking-grid targets."""
import importlib.util
import json
from pathlib import Path


def hazard(label):
    return (any(v>=.5 and m for v,m in zip(label['pcol'][:2],label['pcol_mask'][:2]))
        or label['dmin_mask'] and label['dmin']<=.5
        or label['areq_mask'] and label['areq']>=3)


def test_abbreviated_counterfactual_preserves_attribution_and_unknowns(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.geometry import build_risk_labels
    path=Path(__file__).resolve().parents[1]/'scripts/audit_current_pipeline.py'
    spec=importlib.util.spec_from_file_location('audit',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    manifest=write_synthetic_dataset(tmp_path/'data',scenes_per_split=3,frames_per_scene=1)
    for line in manifest.open():
        r=json.loads(line)
        for coverage in ([True]*6,[True,True,False,False,False,False]):
            r['tracking_coverage_valid']=coverage
            assert hazard(audit.hazard_risk_labels(r))==hazard(build_risk_labels(r))
