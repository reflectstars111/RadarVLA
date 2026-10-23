import argparse,json,os,subprocess,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
data=Path('/home/luxinlan/data/lxl_data/RadarVLA');project=Path('/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA')
root=data/'analysis/risk_layer_diagnosis_20261019';assets=data/'model_runs/risk_layer_diagnosis_20261019'
if args.smoke:root=root/'smoke_verified';assets=assets/'smoke_verified'
root.mkdir(parents=True,exist_ok=True);assert not (root/'launch.json').exists(),'Inspect existing launch before retry'
query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
candidates=[]
for line in query.splitlines():
    index,uuid,free,util=[x.strip() for x in line.split(',')]
    if int(free)>=32768 and int(util)<=10:candidates.append((int(free),int(index),uuid))
assert candidates,'No sufficiently spare GPU; nothing launched'
free,index,uuid=max(candidates);env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
command=['nice','-n','19','ionice','-c','3',sys.executable,str(project/'scripts/diagnose_risk_layers.py'),
    '--base-experiment',str(data/'analysis/oracle_risk_distillation_20261019'),'--output',str(root),'--assets',str(assets)]
if args.smoke:command.append('--smoke')
with (root/'diagnosis.log').open('a') as log:
    process=subprocess.Popen(command,cwd=project,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,gpu=index,gpu_uuid=uuid,initial_free_mib=free,command=command)
(root/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
