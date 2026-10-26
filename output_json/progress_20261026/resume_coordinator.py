"""Adopt existing audited workers; permit four independent evaluation GPUs."""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('campaign',ROOT/'evaluate_campaign.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    lock=(ROOT/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'plan.json').read_text());rt=m.runtime(plan)
    old=json.loads((ROOT/'STATUS.json').read_text());states=old['jobs'];stable={};logs=[];children={}
    m.atomic(ROOT/'CONTROL_CHANGE.json',dict(time=m.stamp(),previous_pid=old['pid'],pid=os.getpid(),
        maximum_workers=4,reason='More idle GPUs available; adopt live workers without restarting them; checkpoints and evaluation code unchanged',
        controller_sha256=m.digest(__file__)))
    def publish():
        terminal=all(v['state'] in ['complete','failed'] for v in states.values())
        state='complete' if all(v['state']=='complete' for v in states.values()) else 'ended_with_failures' if terminal else 'running'
        m.atomic(ROOT/'STATUS.json',dict(state=state,time=m.stamp(),pid=os.getpid(),maximum_workers=4,jobs=states));m.summary(plan,states)
        return terminal
    while True:
        rt=m.runtime(plan)
        for name,value in states.items():
            if value['state']!='running':continue
            f=Path(value['audit'])/'worker_status.json';detail=json.loads(f.read_text()) if f.exists() else {}
            code=children[name].poll() if name in children else None
            if detail.get('state')=='complete':value.update(state='complete',result=detail,exit_code=0)
            elif detail.get('state') in ['failed','resource_exit'] or code is not None or not alive(value['pid']):
                resource=detail.get('state')=='resource_exit' or code==75
                value.update(state='pending' if resource and value['attempts']<3 else 'failed',result=detail,exit_code=75 if resource else code)
        if any(states[j['name']]['state']=='failed' for j in plan['jobs'] if j['split']=='val'):
            for j in plan['jobs']:
                if j['split']=='test' and states[j['name']]['state']=='pending':states[j['name']].update(state='failed',reason='Blocked by validation failure')
        if publish():return
        active=[v for v in states.values() if v['state']=='running']
        used={json.loads((Path(v['audit'])/'job.json').read_text())['gpu_uuid'] for v in active};cards=rt.gpus()
        for uid,c in cards.items():stable[uid]=stable.get(uid,0)+1 if uid not in used and c['utilization']<=60 and c['free_mib']>=26*1024 else 0
        for job in plan['jobs']:
            value=states[job['name']]
            if value['state']!='pending' or sum(v['state']=='running' for v in states.values())>=4:continue
            if job['split']=='test' and not all(states[j['name']]['state']=='complete' for j in plan['jobs'] if j['split']=='val'):continue
            options=[c for uid,c in cards.items() if uid not in used and stable[uid]>=2]
            if not options:continue
            c=min(options,key=lambda c:(bool(c['pids']),c['utilization'],-c['free_mib']));used.add(c['uuid']);value['attempts']+=1
            audit=ROOT/job['name']/f"attempt_{value['attempts']:02d}";audit.mkdir(parents=True,exist_ok=False)
            assigned=dict(job,gpu_uuid=c['uuid'],gpu_index=c['index'],audit=str(audit));m.atomic(audit/'job.json',assigned)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['uuid'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
            for key in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(key,None)
            log=(audit/'evaluation.log').open('ab');logs.append(log)
            cmd=['nice','-n','19','ionice','-c','3',plan['training']['python'],str(ROOT/'evaluate_campaign.py'),'worker','--plan',str(ROOT/'plan.json'),'--job',str(audit/'job.json')]
            proc=subprocess.Popen(cmd,cwd=plan['training']['worktree'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children[job['name']]=proc;value.update(state='running',pid=proc.pid,gpu_index=c['index'],audit=str(audit));publish();time.sleep(15)
        time.sleep(15)


if __name__=='__main__':
    try:main()
    except Exception as error:
        m.atomic(ROOT/'ERROR.resume.json',dict(error=str(error),traceback=traceback.format_exc(),time=m.stamp()));raise
