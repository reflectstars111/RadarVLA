"""Launch one paired worker only on a sufficiently spare GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

project = Path('/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA')
data = Path('/home/luxinlan/data/lxl_data/RadarVLA')
output = data / 'analysis/range_control_20261019'
assets = data / 'model_runs/range_control_20261019'
output.mkdir(parents=True, exist_ok=True)
assert not (output / 'launch.json').exists(), 'Do not launch a duplicate worker'
query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'], text=True)
candidates = []
for row in query.splitlines():
    index, uuid, free, utilization = [s.strip() for s in row.split(',')]
    if int(free) >= 32768 and int(utilization) <= 10:
        candidates.append((int(free), int(index), uuid))
assert candidates, 'No sufficiently spare GPU; nothing launched'
free, index, uuid = max(candidates)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
command = ['nice', '-n', '19', 'ionice', '-c', '3', sys.executable,
           str(project / 'scripts/train_range_control.py'),
           '--manifest', str(data / 'analysis/taskwise_local_v1/manifest.jsonl'),
           '--config', str(project / 'configs/l40s_qwen25_3b_taskwise_local_v1.json'),
           '--output', str(output), '--assets', str(assets)]
with (output / 'training.log').open('a') as log:
    proc = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=proc.pid, gpu=index, gpu_uuid=uuid, initial_free_mib=free,
              command=command, gpu_snapshot=query, concurrent_workers=1)
(output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
