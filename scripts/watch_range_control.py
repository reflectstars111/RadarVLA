"""Low-frequency, CPU-only report/export watcher for one paired experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

project = Path('/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA')
data = Path('/home/luxinlan/data/lxl_data/RadarVLA')
root = data / 'analysis/range_control_20261019'
seen = None


def read_status():
    try:
        return json.loads((root / 'status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def report():
    subprocess.run([sys.executable, str(project / 'scripts/summarize_range_control.py'),
                    '--output', str(root)], cwd=project, check=True,
                   stdout=subprocess.DEVNULL)


while True:
    status = read_status()
    metrics = root / 'metrics.json'
    stamp = metrics.stat().st_mtime_ns if metrics.exists() else None
    if stamp is not None and stamp != seen:
        report()
        seen = stamp
    if status.get('state') == 'complete':
        report()
        destination = root.parent / 'RadarVLA_range_control_20261019.zip'
        paths = [p for p in root.rglob('*') if p.is_file() and p.suffix in ['.json', '.jsonl', '.md', '.py']]
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None
        (root / 'export_status.json').write_text(json.dumps(dict(state='complete', archive=str(destination),
            bytes=destination.stat().st_size, crc_verified=True)) + '\n')
        break
    if status.get('state') == 'failed':
        (root / 'export_status.json').write_text(json.dumps(dict(state='training_failed',
            error=status.get('error'), partial_report=(root / 'REPORT.md').exists())) + '\n')
        break
    try:
        os.kill(json.loads((root / 'launch.json').read_text())['pid'], 0)
    except ProcessLookupError:
        # Allow final writes to finish before declaring an unexpected exit.
        time.sleep(5)
        if read_status().get('state') == 'complete':
            continue
        (root / 'export_status.json').write_text(json.dumps(dict(state='worker_exited', last_status=read_status())) + '\n')
        break
    time.sleep(20)
