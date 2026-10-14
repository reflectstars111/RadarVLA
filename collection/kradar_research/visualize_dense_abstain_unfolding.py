#!/usr/bin/env python3
"""Render native [range, azimuth] RA diagnostics and a local frame switcher."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

import dense_abstain_support as base
from build_dense_abstain_unfolding import OUTPUT, SOURCE


def render(output: Path = OUTPUT) -> None:
    result = json.loads((output / 'results.json').read_text())
    frames = json.loads((output / 'frames.json').read_text())
    if len(frames) != 50 or result['shape'] != [50, 2, 256, 107]:
        raise ValueError('unexpected frame count or dense shape')
    with np.load(base.CACHE / 'radar_axes.npz') as z:
        ranges = z['range_m']
        azimuths = z['azimuth_deg']
    dr = np.median(np.diff(ranges)); da = np.median(np.diff(azimuths))
    extent = [azimuths[0] - da/2, azimuths[-1] + da/2,
              ranges[0] - dr/2, ranges[-1] + dr/2]
    dense = np.load(output / 'version_a_50x2x256x107.npy', mmap_mode='r')
    global_peak = float(np.max(dense[:, 0]))
    if global_peak <= 0:
        raise ValueError('invalid global power maximum')
    view_dir = output / 'ra_frames'
    view_dir.mkdir(exist_ok=True)
    source_cmap = ListedColormap(['#3087c4', '#31a8a3', '#e0b33b', '#c77345', '#777777'])
    source_norm = BoundaryNorm(np.arange(.5, 6.5, 1), source_cmap.N)
    vel_cmap = plt.get_cmap('coolwarm').copy(); vel_cmap.set_bad('#aaaaaa')
    width, height = 15, 11
    for i, frame in enumerate(frames):
        stem = frame['radar_stem']
        power = dense[i, 0]
        doppler = dense[i, 1]
        with np.load(output / 'version_b_qc' / f'{stem}.npz') as qc:
            reliable = qc['doppler_reliable_ra_mps']
            mask = qc['reliable_mask']
            source = qc['source_code']
        if not np.array_equal(reliable[mask], doppler[mask]):
            raise AssertionError(f'B differs from A at valid cells: {stem}')
        if not np.isnan(reliable[~mask]).all():
            raise AssertionError(f'B invalid cells must be NaN: {stem}')
        power_db = 10 * np.log10(np.maximum(power, 1) / global_peak)
        fig, ax = plt.subplots(2, 2, figsize=(width, height), constrained_layout=True)
        p = ax[0, 0].imshow(power_db, origin='lower', aspect='auto', extent=extent,
                            cmap='magma', vmin=-60, vmax=0, interpolation='nearest')
        d = ax[0, 1].imshow(doppler, origin='lower', aspect='auto', extent=extent,
                            cmap=vel_cmap, vmin=-35, vmax=35, interpolation='nearest')
        b = ax[1, 0].imshow(reliable, origin='lower', aspect='auto', extent=extent,
                            cmap=vel_cmap, vmin=-35, vmax=35, interpolation='nearest')
        s = ax[1, 1].imshow(source, origin='lower', aspect='auto', extent=extent,
                            cmap=source_cmap, norm=source_norm, interpolation='nearest')
        for a, title in zip(ax.flat, ('Power (native RA)', 'Dense unfolded Doppler hypothesis',
                                      'Reliable Doppler (gray = abstain)', 'Decision source')):
            a.set_title(title)
            a.set_xlabel('Azimuth (deg)')
            a.set_ylabel('Range (m)')
        fig.colorbar(p, ax=ax[0, 0], label='dB relative to 50-frame maximum')
        fig.colorbar(d, ax=ax[0, 1], label='m/s (+ receding)')
        fig.colorbar(b, ax=ax[1, 0], label='m/s (+ receding)')
        cb = fig.colorbar(s, ax=ax[1, 1], ticks=range(1, 6))
        cb.ax.set_yticklabels(['two-sided', 'one-sided', 'neighbor', 'previous', 'k=0'])
        fig.suptitle(f'K-Radar seq1 frame {stem} ({i+1}/50) | RA [256 range, 107 azimuth]', fontsize=14)
        fig.savefig(view_dir / f'{stem}.png', dpi=130)
        plt.close(fig)
        print(f'RA {i+1}/50 {stem}', flush=True)
    stems = [f['radar_stem'] for f in frames]
    js_stems = json.dumps(stems)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>K-Radar dense and abstain RA</title>
<style>body{font:16px system-ui,sans-serif;background:#1b1c20;color:#eee;margin:20px auto;max-width:1250px}
img{display:block;width:100%;height:auto;background:#fff}input{width:80%}a{color:#aad5ff}</style>
<h1>K-Radar sequence 1: dense and abstain unfolded Doppler</h1>
<p>Native range–azimuth matrix [256,107]. All 50 frames use fixed Power (−60 to 0 dB) and Doppler (−35 to 35 m/s) color scales. Gray means abstention. Dense values are hypotheses, not ground truth.</p>
<p><button onclick="step(-1)">Previous</button> <input id="frame" type="range" min="0" max="49" value="0" oninput="show()"> <button onclick="step(1)">Next</button> <span id="label"></span></p>
<img id="plot" alt="Four native RA plots for selected frame">
<script>const stems=STEMS;function show(){const i=Number(document.getElementById('frame').value);document.getElementById('plot').src='ra_frames/'+stems[i]+'.png';document.getElementById('label').textContent=stems[i]+' ('+(i+1)+'/50)';}function step(n){const x=document.getElementById('frame');x.value=Math.max(0,Math.min(49,Number(x.value)+n));show();}show();</script></html>'''.replace('STEMS', js_stems)
    (output / 'index.html').write_text(html)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args=parser.parse_args()
    render(args.output)
