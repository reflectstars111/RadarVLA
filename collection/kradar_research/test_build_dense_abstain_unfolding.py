import numpy as np

import build_dense_abstain_unfolding as dense


def test_invalid_and_tied_candidates_never_choose_minimum_order():
    orders = np.array([-10, 0, 3], np.int16)
    none = np.full((2, 3), np.inf, np.float32)
    x = dense.choose_direct(none, none, orders)
    assert np.array_equal(x['source'], [0, 0])
    assert np.array_equal(x['k'], [0, 0])
    tied = np.array([[1, 1, 3], [3, 1, 3]], np.float32)
    y = dense.choose_direct(tied, tied, orders)
    assert y['source'][0] == 0 and y['joint_tie'][0]
    assert y['source'][1] == dense.SOURCE['direct_two_sided'] and y['k'][1] == 0


def test_one_sided_winner_and_zero_fallback():
    orders = np.array([-2, 0, 2], np.int16)
    after = np.array([[3, 1, 5], [np.inf, np.inf, np.inf]], np.float32)
    d = dense.choose_direct(None, after, orders)
    assert d['source'][0] == dense.SOURCE['direct_one_sided'] and d['k'][0] == 0
    assert d['source'][1] == 0 and d['k'][1] == 0
    grid = {key: value.reshape(1, 2) for key, value in d.items()}
    vf = np.zeros((1, 2), np.float32)
    filled = dense.fill_orders(grid, vf, np.zeros((1, 2), np.uint8), None,
                               {}, np.eye(4), None, 3.86518)
    assert filled['source'][0, 1] == dense.SOURCE['zero_order']
    assert filled['k'][0, 1] == 0


def test_spatial_fallback_requires_three_agreeing_direct_neighbors():
    k = np.zeros((5, 5), np.int16)
    source = np.zeros((5, 5), np.uint8)
    agreed = np.zeros((5, 5), bool)
    for r, a in ((1, 2), (2, 1), (2, 3)):
        k[r, a] = -3
        source[r, a] = dense.SOURCE['direct_two_sided']
        agreed[r, a] = True
    used = dense.spatial_fill(k, source, np.zeros((5, 5), np.float32), 3.86518, agreed)
    assert used[2, 2] and k[2, 2] == -3
    assert source[2, 2] == dense.SOURCE['spatial_neighbor']
    assert agreed.sum() == 3


def test_temporal_fallback_uses_previous_direct_only():
    axes = {'range_m': np.arange(256, dtype=float),
            'azimuth_deg': np.arange(-53, 54, dtype=float),
            'elevation_deg': np.arange(-18, 19, dtype=float)}
    now_k = np.zeros((256, 107), np.int16)
    now_source = np.zeros((256, 107), np.uint8)
    prev = dict(direct_k=np.zeros((256, 107), np.int16),
                direct_source=np.zeros((256, 107), np.uint8),
                vf=np.zeros((256, 107), np.float32))
    prev['direct_k'][10, 53] = -4
    prev['direct_source'][10, 53] = dense.SOURCE['direct_two_sided']
    old = dense.base.OFFSET_LIDAR_TO_RADAR.copy()
    try:
        dense.base.OFFSET_LIDAR_TO_RADAR[:] = 0
        used = dense.temporal_fill(now_k, now_source, np.zeros((256, 107), np.float32),
                                   np.full((256, 107), 18, np.uint8), prev, axes,
                                   np.eye(4), np.eye(4), 3.86518)
    finally:
        dense.base.OFFSET_LIDAR_TO_RADAR[:] = old
    assert used[10, 53] and now_k[10, 53] == -4
    assert now_source[10, 53] == dense.SOURCE['temporal_previous']


def test_clip_match_power_and_foldback():
    power = np.arange(1000, dtype=float).reshape(10, 10, 10)
    clipped, bounds = dense.clip_match_log_power(power)
    assert clipped.shape == power.shape and clipped.dtype == np.float32
    assert clipped.min() >= bounds[0] - 1e-5 and clipped.max() <= bounds[1] + 1e-5
    axis = -1.932591 + np.arange(64) * .0603935
    period = np.median(np.diff(axis)) * 64
    for raw in (0, 1, 31, 63):
        vf = axis[dense.corrected_bins(np.array(raw), 64)]
        for k in (-4, 0, 3):
            assert abs(((vf + k * period - axis[0]) % period) - ((vf - axis[0]) % period)) < 1e-12


def test_parameter_selection_uses_training_half_only(monkeypatch, tmp_path):
    import select_dense_abstain_params as selector

    def candidate(lag, dense, coverage, holdout):
        rows = [dict(threshold=t, stable_n=600,
                     stable_pseudo_agreement=.91,
                     strong_reliable_coverage=coverage,
                     strong_reliable=100, strong_n=500) for t in selector.MARGINS]
        return dict(lag=lag,
                    train=dict(dense_pseudo_agreement=dense, margins=rows),
                    report=dict(dense_pseudo_agreement=holdout, margins=rows))

    data = {1: candidate(1, .59, .20, .10),
            2: candidate(2, .60, .10, .99),
            3: candidate(3, .57, .90, 1.00)}
    monkeypatch.setattr(selector, 'score_lag', lambda path: data[int(path.name[-1])])
    assert selector.select(tmp_path)['selected'] == dict(lag=1, margin_threshold=min(selector.MARGINS))


def test_edge_frames_use_nearest_available_match():
    assert [dense.effective_match_lag(i, 50, 2) for i in (0, 1, 2, 47, 48, 49)] == [1, 1, 2, 2, 1, 1]
    assert [dense.effective_match_lag(i, 50, 3) for i in (0, 1, 2, 3, 46, 47, 48, 49)] == [1, 1, 2, 3, 3, 2, 1, 1]
