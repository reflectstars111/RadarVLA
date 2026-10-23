"""Prevent Oracle-cohort leakage and apparent gains from dropping time coverage."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('risk_rollout_audit', Path(__file__).parents[1] / 'scripts' / 'diagnose_risk_rollout.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('target,mask,expected', [
    ([0., 0., 0., 10., 0.], [True]*5, True),
    ([0., 0., 0., 10., 0.], [True]*4+[False], False),
    ([0., 0., 0., 10., float('nan')], [True]*5, False),
    ([0., 0., 0., 10., float('inf')], [True]*5, False),
])
def test_only_complete_finite_oracle_is_admitted(target, mask, expected):
    assert audit.oracle_eligible(dict(risk_target=target, risk_mask=mask)) == expected


def test_changed_mode_cannot_hide_missing_long_horizon():
    truth = dict(future_times_s=[.5, 1., 1.5, 2., 2.5, 3.],
                 ego_future=[[0., 0.]]*6, ego_future_mask=[True]*6, ego_velocity=[0., 0.])
    original = dict(valid=True, ego_times_s=truth['future_times_s'],
                    ego_trajectory=[[1., 0.], [1., 0.]]+[[20., 0.]]*4)
    shorter = dict(valid=True, ego_times_s=[.5, 1.], ego_trajectory=[[0., 0.]]*2)
    score = audit.paired_scores(original, shorter, truth, 'long')
    assert score['oracle']['required_covered'] == 2
    assert score['oracle']['required_points'] == 6
    assert score['predicted_common_errors_m'] == [1., 1.]
    assert score['oracle_common_errors_m'] == [0., 0.]
    assert score['common_required_indices'] == ['0', '1']


def test_invalid_oracle_does_not_score_as_zero_error():
    truth = dict(future_times_s=[.5, 1.], ego_future=[[0., 0.]]*2,
                 ego_future_mask=[True]*2, ego_velocity=[0., 0.])
    plan = dict(valid=False, ego_times_s=[.5, 1.], ego_trajectory=[[0., 0.]]*2)
    score = audit.score_trajectory(plan, truth, 'short')
    assert score['required_covered'] == 0
    assert score['required_points'] == 2
    assert score['required_errors_m'] == []


def test_duplicate_sample_ids_reject_unsafe_resume(tmp_path):
    path = tmp_path/'samples.jsonl'
    path.write_text((json.dumps(dict(sample_id='same'))+'\n')*2)
    with pytest.raises(ValueError, match='Duplicate'):
        audit.read_rows(path)
