"""Read-only native full-split evaluation with shared GPU admission and audits."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def runtime(plan):
    spec=importlib.util.spec_from_file_location('shared_runtime',Path(plan['training']['worktree'])/'scripts/bottleneck_shared_runtime.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.check_pins(plan['training'])
    assert digest(__file__)==plan['evaluation_script_sha256'],'Evaluation script changed'
    return mod


def worker(plan_path,job_path):
    plan=json.loads(plan_path.read_text());job=json.loads(job_path.read_text());rt=runtime(plan)
    root=Path(job['audit']);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    status=lambda state,**kw:atomic(root/'worker_status.json',dict(state=state,time=stamp(),pid=os.getpid(),**kw))
    assert digest(job['checkpoint'])==job['checkpoint_sha256']
    import numpy as np
    import torch
    from radar_vla import pipeline
    assert Path(pipeline.__file__).resolve()==(Path(plan['training']['worktree'])/'src/pipeline.py').resolve()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.cuda.set_device(0)
    assert torch.cuda.is_bf16_supported()
    torch.cuda.set_per_process_memory_fraction(18*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    snap=rt.gpus()[job['gpu_uuid']]
    if snap['free_mib']<26*1024:
        status('resource_exit',reason='Admission headroom changed');return 75
    duty=[rt.duty_for(snap,os.getpid())];done=threading.Event();phase=['loading_checkpoint'];progress=[0,0]
    def guard():
        failures=0
        while not done.wait(5):
            try:
                c=rt.gpus()[job['gpu_uuid']];failures=0
                if c['free_mib']<8*1024:
                    status('resource_exit',reason='Free VRAM below8GiB; stopped only this worker');os._exit(75)
                duty[0]=rt.duty_for(c,os.getpid())
                atomic(root/'resource.json',dict(**c,duty=duty[0],time=stamp()))
            except Exception:
                failures+=1
                if failures>=3:
                    status('resource_exit',reason='Resource monitoring unavailable');os._exit(75)
    threading.Thread(target=guard,daemon=True).start()
    status('running',phase=phase[0]);original_loader=pipeline.make_loader
    def loader(*a,**kw):
        native=original_loader(*a,**kw)
        def report(index):
            if index%10==0 or index==len(native):
                progress[:]=[index,len(native)]
                atomic(root/'progress.json',dict(phase=phase[0],batches=index,total_batches=len(native),time=stamp()))
        return rt.PacedLoader(native,lambda:duty[0],torch.cuda.synchronize,report)
    pipeline.make_loader=loader
    original_validate=pipeline.validate
    def validate(*a,**kw):
        phase[0]='full_perception_and_teacher_forced_validation';status('running',phase=phase[0])
        return original_validate(*a,**kw)
    pipeline.validate=validate
    original_predictions=pipeline.prediction_rows
    def predict(*a,**kw):
        phase[0]='full_free_generation';status('running',phase=phase[0])
        rows=original_predictions(*a,**kw)
        with (out/'predictions.jsonl.partial').open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        (out/'predictions.jsonl.partial').replace(out/'predictions.jsonl')
        phase[0]='coverage_physical_and_instruction_metrics';status('running',phase=phase[0],predictions=len(rows))
        return rows
    pipeline.prediction_rows=predict
    try:
        with torch.no_grad():
            report=pipeline.evaluate_checkpoint(job['checkpoint'],plan['training']['manifest'],out/'report.json',
                split=job['split'],batch_size=1,device='cuda:0')
        assert report['epoch']==job['epoch'] and report['checkpoint_sha256']==job['checkpoint_sha256']
        assert report['samples']==plan['expected_splits'][job['split']]
        assert report['free_generation']['samples']==plan['expected_splits'][job['split']]
        with (out/'predictions.jsonl').open() as f:assert sum(1 for _ in f)==report['samples']
        status('complete',epoch=report['epoch'],samples=report['samples'],checkpoint_sha256=job['checkpoint_sha256'])
        return 0
    except Exception as error:
        status('failed',error=str(error),traceback=traceback.format_exc());traceback.print_exc();return 1
    finally:done.set()


def summary(plan,states):
    rows=[]
    for job in plan['jobs']:
        if states[job['name']]['state']!='complete':continue
        report=json.loads((Path(job['output'])/'report.json').read_text());g=report['free_generation']
        rows.append(dict(arm=job['arm'],split=job['split'],epoch=report['epoch'],samples=report['samples'],
            schema_valid_rate=g['schema_valid_rate'],ego_ade_m=g.get('ego_ade_m'),ego_fde_m=g.get('ego_final_horizon_fde_m'),
            ego_point_coverage=g.get('ego_point_coverage'),required_point_coverage=g.get('required_point_coverage'),
            required_ego_ade_m=g.get('required_ego_ade_m'),short_recall=g.get('short_recall'),
            detection=report.get('detection'),collision=report.get('collision'),physical_planning=report.get('physical_planning')))
    root=Path(plan['output_root']);atomic(root/'SUMMARY.json',rows)
    text=['# Full planning validation and test evaluation','',f'Updated: {stamp()}','',
          'Frozen native best_planning checkpoints selected during training. Test never selects models. '
          'Shared-GPU timing is not a dedicated latency benchmark. Offline collision metrics are not closed-loop safety claims.','',
          '| Arm | Split | Epoch | Samples | Valid | ADE m | FDE m | Required coverage | SHORT recall |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return '-' if x is None else f'{x:.4f}'
    for row in rows:text.append('| '+' | '.join([row['arm'],row['split'],str(row['epoch']),str(row['samples']),fmt(row['schema_valid_rate']),fmt(row['ego_ade_m']),fmt(row['ego_fde_m']),fmt(row['required_point_coverage']),fmt(row['short_recall'])])+' |')
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')


def coordinator(plan_path):
    plan=json.loads(plan_path.read_text());rt=runtime(plan);root=plan_path.parent
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'STATUS.json').exists(),'Refuse duplicate evaluation campaign'
    states={j['name']:dict(state='pending',attempts=0) for j in plan['jobs']};active={};stable={};logs=[]
    def publish():
        terminal=all(s['state'] in ['complete','failed'] for s in states.values())
        overall='complete' if all(s['state']=='complete' for s in states.values()) else 'ended_with_failures' if terminal else 'running'
        atomic(root/'STATUS.json',dict(state=overall,time=stamp(),pid=os.getpid(),jobs=states))
        summary(plan,states)
        return terminal
    try:
        while True:
            runtime(plan)
            for name,(proc,job) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                del active[name];record=states[name]
                f=Path(job['audit'])/'worker_status.json';detail=json.loads(f.read_text()) if f.exists() else {}
                record.update(state='complete' if code==0 and detail.get('state')=='complete' else 'pending' if code==75 and record['attempts']<3 else 'failed',exit_code=code,result=detail)
            if any(states[j['name']]['state']=='failed' for j in plan['jobs'] if j['split']=='val'):
                for j in plan['jobs']:
                    if j['split']=='test' and states[j['name']]['state']=='pending':
                        states[j['name']].update(state='failed',reason='Blocked by incomplete validation; test not started')
            if publish():return
            used={j['gpu_uuid'] for _,j in active.values()};cards=rt.gpus()
            for uid,c in cards.items():stable[uid]=stable.get(uid,0)+1 if uid not in used and c['utilization']<=60 and c['free_mib']>=26*1024 else 0
            for job in plan['jobs']:
                record=states[job['name']]
                if record['state']!='pending' or len(active)>=2:continue
                # Complete every validation job before exposing any test metric.
                if job['split']=='test' and not all(states[x['name']]['state']=='complete' for x in plan['jobs'] if x['split']=='val'):continue
                candidates=[c for uid,c in cards.items() if uid not in used and stable[uid]>=2]
                if not candidates:continue
                c=min(candidates,key=lambda c:(bool(c['pids']),c['utilization'],-c['free_mib']));used.add(c['uuid']);record['attempts']+=1
                audit=root/job['name']/f"attempt_{record['attempts']:02d}";audit.mkdir(parents=True,exist_ok=False)
                assigned=dict(job,gpu_uuid=c['uuid'],gpu_index=c['index'],audit=str(audit));atomic(audit/'job.json',assigned)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=c['uuid'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
                for key in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']:env.pop(key,None)
                log=(audit/'evaluation.log').open('ab');logs.append(log)
                cmd=['nice','-n','19','ionice','-c','3',plan['training']['python'],str(Path(__file__).resolve()),'worker','--plan',str(plan_path),'--job',str(audit/'job.json')]
                proc=subprocess.Popen(cmd,cwd=plan['training']['worktree'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[job['name']]=(proc,assigned);record.update(state='running',pid=proc.pid,gpu_index=c['index'],audit=str(audit));publish();time.sleep(15)
            time.sleep(15)
    except Exception as error:
        atomic(root/'ERROR.json',dict(error=str(error),traceback=traceback.format_exc(),time=stamp()));raise
    finally:
        for f in logs:f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','coordinator']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--job',type=Path);a=p.parse_args()
    if a.mode=='worker':sys.exit(worker(a.plan,a.job))
    coordinator(a.plan)
