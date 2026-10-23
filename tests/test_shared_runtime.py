import importlib.util
from pathlib import Path
import random


def runtime():
    path=Path(__file__).with_name('shared_runtime.py')
    if not path.exists(): path=Path(__file__).resolve().parents[1]/'scripts/bottleneck_shared_runtime.py'
    spec=importlib.util.spec_from_file_location('shared_runtime',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_pacing_preserves_order_count_rng_and_sleeps_after_batch():
    r=runtime();clock=[0.];sleeps=[];reports=[]
    random.seed(42);before=random.getstate()
    def sleep(t):sleeps.append(t);clock[0]+=t
    loader=r.PacedLoader([1,2,3,4],lambda:.25,lambda:None,reports.append,lambda:clock[0],sleep)
    out=[]
    for batch in loader:
        out.append(batch);clock[0]+=2
    assert out==[1,2,3,4] and len(loader)==4 and reports==[1,2,3,4]
    assert sleeps==[6.,6.,6.,6.]
    assert random.getstate()==before


def test_other_process_presence_throttles_but_never_stops():
    r=runtime()
    assert r.duty_for(dict(pids=[10],utilization=100),10)==1
    assert r.duty_for(dict(pids=[10,11],utilization=0),10)==.5
    assert r.duty_for(dict(pids=[10,11],utilization=70),10)==.25
    assert r.duty_for(dict(pids=[10,11],utilization=99),10)==.1


def test_dag_has_paired_readout_and_correct_sft_sources():
    r=runtime();jobs=r.jobs_for(dict(output_root='/tmp/experiments'))
    byname={j['name']:j for j in jobs}
    assert len(jobs)==11 and len(byname)==11
    assert byname['baseline_readout']['init']==byname['risk_readout']['init']
    assert byname['baseline_sft']['init']==byname['action_sft']['init']
    assert byname['baseline_readout_sft']['init'].endswith('/baseline/risk-readout/epoch_010.pt')
    for job in jobs:
        if job['stage']=='risk-readout':
            assert set(job['dependencies'])=={'baseline_grounding','assignment_grounding'}
        if job['stage']=='sft':
            assert {'baseline_readout','risk_readout','combined_readout'}<=set(job['dependencies'])
    visited=set()
    for job in jobs:
        assert set(job['dependencies'])<=visited
        visited.add(job['name'])


def test_atomic_json_and_streaming_digest(tmp_path):
    r=runtime();p=tmp_path/'status.json';r.atomic(p,dict(state='ready'))
    assert p.read_text().endswith('\n')
    import hashlib
    assert r.sha(p)==hashlib.sha256(p.read_bytes()).hexdigest()
    assert list(tmp_path.iterdir())==[p]
