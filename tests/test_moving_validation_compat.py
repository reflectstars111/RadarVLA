"""Regression for the Metadata.supervision failure after motion-label audit."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('moving_validation_compat', PROJECT/'scripts/moving_validation_compat.py')
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


def metadata():
    from radar_vla.pipeline import resolve_config
    config = resolve_config(json.loads((PROJECT/'configs/l40s_qwen25_3b_taskwise_moving_v1.json').read_text()))
    Metadata = type('Metadata', (), {'__module__': 'radar_vla.moving_migration'})
    ds = Metadata()
    ds.class_schema = 'agent_v1'
    ds.risk_schema = 'local_agent_collision_v1'
    ds.supervision_protocol = 'taskwise_local_v1'
    ds.records = [dict(sample_id='example', agent_selection_policy='moving_vehicle_01_background_v1',
        agent_ignore_policy='none', sensors=dict(radar=dict(range_m=list(range(256)),
        azimuth_rad=list(range(107)), time_offsets_s=[-.3,-.2,-.1,0.])),
        future_times_s=[.5,1.,1.5,2.,2.5,3.], agents=[])]
    return ds, config


def test_supplies_missing_supervision_to_real_validator():
    from radar_vla.pipeline import validate_input_shapes
    ds, config = metadata()
    with pytest.raises(AttributeError, match='supervision'):
        validate_input_shapes(ds, config)
    compat.validate_metadata(ds, config, validate_input_shapes)
    assert ds.supervision is True


def test_preserves_policy_checks_and_restores_validator():
    import radar_vla.pipeline as pipeline
    ds, config = metadata()
    ds.records[0]['agent_selection_policy'] = 'wrong_policy'
    original = pipeline.validate_input_shapes
    with compat.supervised_metadata_adapter():
        with pytest.raises(ValueError, match='selection policy'):
            pipeline.validate_input_shapes(ds, config)
    assert pipeline.validate_input_shapes is original


def test_does_not_override_real_dataset_or_explicit_supervision():
    ds = SimpleNamespace(supervision=False)
    compat.validate_metadata(ds, {}, lambda dataset, config: None)
    assert ds.supervision is False
    unknown = SimpleNamespace()
    compat.validate_metadata(unknown, {}, lambda dataset, config: None)
    assert not hasattr(unknown, 'supervision')


def test_normalizes_only_old_motion_validation_invocation():
    old=['bash','/project/scripts/radarvla.sh','validate','--full','--manifest','/data/manifest.jsonl']
    assert compat.validation_command(old) == ['bash','/project/scripts/radarvla.sh','validate','--manifest','/data/manifest.jsonl']
    for other in (['bash','other.sh','validate','--full','--manifest','file'],
                  ['bash','radarvla.sh','validate','--full'], 'echo --full --manifest'):
        assert compat.validation_command(other) is other


def test_validation_subprocess_preserves_exit_checks_and_restores(monkeypatch):
    import subprocess
    calls=[]
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return 'result'
    monkeypatch.setattr(subprocess, 'run', run)
    with compat.supervised_metadata_adapter():
        result=subprocess.run(['bash','radarvla.sh','validate','--full','--manifest','file'], check=True)
    assert result=='result' and calls==[(['bash','radarvla.sh','validate','--manifest','file'], {'check':True})]
    assert subprocess.run is run


def test_finalize_requires_unchanged_full_acceptance(tmp_path):
    from radar_vla.agent50_migration import sha256
    from radar_vla.moving_agents import POLICY
    spec=importlib.util.spec_from_file_location('finalize_moving_agents', PROJECT/'scripts/finalize_moving_agents.py')
    finalize=importlib.util.module_from_spec(spec);spec.loader.exec_module(finalize)
    out=tmp_path/'output';out.mkdir()
    manifest=out/'manifest.jsonl';manifest.write_text('manifest')
    source=tmp_path/'source.jsonl';source.write_text('source')
    config=tmp_path/'config.json';config.write_text('{}')
    code=tmp_path/'code.py';code.write_text('original implementation')
    acceptance=dict(status='PASS',policy=POLICY,raw_evidence_checked=True,
        representative_gradients=[dict(loss=1.)],manifest_sha256=sha256(manifest),source_sha256=sha256(source))
    (out/'acceptance.json').write_text(json.dumps(acceptance))
    (out/'raw_evidence.json').write_text('{}')
    (out/'migration.json').write_text(json.dumps(dict(manifest_sha256=sha256(manifest))))
    (out/'provenance.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha256(source),
        config_sha256=sha256(config),code={str(code):sha256(code)})))
    assert finalize.check_acceptance(out,config)==acceptance
    manifest.write_text('changed')
    with pytest.raises(ValueError,match='Manifest changed'):
        finalize.check_acceptance(out,config)
    manifest.write_text('manifest');code.write_text('changed')
    with pytest.raises(ValueError,match='implementation changed'):
        finalize.check_acceptance(out,config)
