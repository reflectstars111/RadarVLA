"""Isolated shared-GPU workers and a dependency-aware campaign coordinator.

Only run by explicit user authorization. No signals are sent to other processes.
Native training owns model losses, sampling, checkpointing and resume checks.
"""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def gpus():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=8)
    result = {}
    for line in raw.splitlines():
        i, uid, total, free, util = [s.strip() for s in line.split(',')]
        result[uid] = dict(index=int(i), uuid=uid, total_mib=int(total), free_mib=int(free), utilization=int(util), pids=[])
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=8)
    for line in raw.splitlines():
        if ',' not in line: continue
        uid, pid = [s.strip() for s in line.split(',')]
        if uid in result: result[uid]['pids'].append(int(pid))
    return result


def duty_for(gpu, own_pid):
    others = [pid for pid in gpu['pids'] if pid != own_pid]
    if not others: return 1.0
    return .1 if gpu['utilization'] >= 90 else .25 if gpu['utilization'] >= 65 else .5


class PacedLoader:
    def __init__(self, loader, duty, sync, report=lambda n: None, clock=time.monotonic, sleep=time.sleep):
        self.loader, self.duty, self.sync, self.report, self.clock, self.sleep = loader, duty, sync, report, clock, sleep

    def __len__(self): return len(self.loader)

    def __iter__(self):
        for index, batch in enumerate(self.loader, 1):
            self.sync(); began = self.clock()
            yield batch
            self.sync(); elapsed = self.clock() - began
            self.report(index)
            self.sleep(max(0., elapsed * (1. / self.duty() - 1.)))


def check_pins(plan):
    w = Path(plan['worktree'])
    for rel, expected in plan['source_files'].items():
        if sha(w / rel) != expected: raise RuntimeError('Pinned source changed: ' + rel)
    if sha(plan['profile']) != plan['profile_sha256']: raise RuntimeError('Pinned profile changed')
    if sha(__file__) != plan['runtime_sha256']: raise RuntimeError('Pinned worker changed')


def smoke(pipeline, config, plan, job, audit):
    import torch
    from radar_vla.data import RadarDataset, collate_batch
    from radar_vla.model import RadarVLAGrounder, ModelConfig
    torch.manual_seed(plan['seed']); random.seed(plan['seed'])
    dataset = RadarDataset(plan['manifest'], 'train')
    positive = next(i for i, r in enumerate(dataset.records) if r['local_risk']['pcol'][0])
    empty = next(i for i, r in enumerate(dataset.records) if not r['agents'])
    many = max(range(len(dataset)), key=lambda i: len(dataset.records[i]['agents']))
    selected = list(dict.fromkeys([positive, empty, many, 0]))
    while len(selected) < 4:
        selected.append(next(i for i in range(len(dataset)) if i not in selected))
    dataset.records = [dataset.records[i] for i in selected]; dataset._risk_cache.clear()
    grounder = RadarVLAGrounder(ModelConfig(**config['model'])).cuda()
    grounder.loss_weights = dict(config.get('loss_weights', {}))
    if job['init']:
        state = torch.load(job['init'], map_location='cpu', weights_only=True)
        if job['stage'] == 'risk-readout':
            pipeline._initialize_risk_readout(grounder, state, config, state['protocol']['data'])
        else: grounder.load_state_dict(state['grounder'])
        del state
    grounder.train(job['stage'] == 'grounding')
    if job['stage'] == 'risk-readout': grounder.configure_risk_readout_training()
    if job['stage'] == 'sft': grounder.requires_grad_(False)
    planner = pipeline.make_planner(config).cuda().train() if job['stage'] == 'sft' else None
    if planner is not None:
        # Native continuous-action SFT must preserve frozen perception.
        assert not any(p.requires_grad for p in grounder.parameters())
    passes = []
    count = 1 if planner is not None else 4
    batch = pipeline.to_device(collate_batch([dataset[i] for i in range(count)]), 'cuda:0')
    epochs = [1, max(2, grounder.config.detection_assignment_warmup_epochs + 1)] if job['arm'] == 'assignment' and job['stage'] == 'grounding' else [1]
    for epoch in epochs:
        grounder.training_epoch = epoch; grounder._assignment_diagnostics = {}
        with pipeline.autocast_context('cuda:0', 'bfloat16'):
            loss, components, pred = pipeline._loss(grounder, planner, batch)
        assert torch.isfinite(loss), 'Nonfinite real-sample smoke loss'
        loss.backward()
        gradients = [p.grad for p in list(grounder.parameters()) + (list(planner.parameters()) if planner is not None else []) if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert pred['risk'].shape == (count, 5) and pred['detection_logits'].shape == (count, 3456)
        passes.append(dict(epoch=epoch, loss=float(loss.detach()), assignment=dict(grounder._assignment_diagnostics)))
        grounder.zero_grad(set_to_none=True)
        if planner is not None: planner.zero_grad(set_to_none=True)
        del loss, components, pred
    result = dict(state='passed', source_commit=plan['source_commit'], samples=batch['sample_id'],
                  optimizer_updates=0, passes=passes, peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  time=now(), budget_gib=job['budget_gib'])
    atomic(audit / 'smoke.json', result)
    del batch, grounder, planner, dataset
    torch.cuda.empty_cache()
    return result


def worker(args):
    import torch
    from radar_vla import pipeline
    plan = json.loads(args.plan.read_text()); job = json.loads(args.job.read_text())
    audit = Path(job['audit']); output = Path(job['output']); audit.mkdir(parents=True, exist_ok=True)
    check_pins(plan)
    assert str(Path(pipeline.__file__).resolve()) == str((Path(plan['worktree']) / 'src/pipeline.py').resolve())
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('Selected GPU lacks native BF16')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(job['budget_gib'] * 1024**3 / total, 0)
    snapshot = gpus()[job['gpu_uuid']]
    if snapshot['free_mib'] < (job['budget_gib'] + 8) * 1024:
        atomic(audit / 'worker_status.json', dict(state='resource_exit', reason='Admission headroom changed', time=now())); return 75
    memory_duty = [duty_for(snapshot, os.getpid())]; done = threading.Event()
    atomic(audit / 'runtime.json', dict(pid=os.getpid(), gpu_uuid=job['gpu_uuid'], budget_gib=job['budget_gib'],
           torch=torch.__version__, cuda=torch.version.cuda, cpu_threads=torch.get_num_threads(), workers=0,
           nice=os.getpriority(os.PRIO_PROCESS, 0), ionice=subprocess.check_output(['ionice', '-p', str(os.getpid())], text=True).strip(), time=now()))
    def guard():
        failures = 0
        while not done.wait(5):
            try:
                snap = gpus()[job['gpu_uuid']]; failures = 0
                if snap['free_mib'] < 8 * 1024:
                    atomic(audit / 'worker_status.json', dict(state='resource_exit', reason='Free VRAM below8GiB; exit only this PID', time=now())); os._exit(75)
                memory_duty[0] = duty_for(snap, os.getpid())
                atomic(audit / 'resource.json', dict(**snap, duty=memory_duty[0], time=now()))
            except Exception:
                failures += 1
                if failures >= 3:
                    atomic(audit / 'worker_status.json', dict(state='resource_exit', reason='Resource monitor unavailable', time=now())); os._exit(75)
    threading.Thread(target=guard, daemon=True).start()
    try:
        atomic(audit / 'worker_status.json', dict(state='preflight', time=now(), pid=os.getpid()))
        if sha(plan['manifest']) != plan['manifest_sha256']: raise RuntimeError('Manifest digest changed')
        validation = json.loads((Path(plan['manifest']).parent / 'validation.json').read_text())
        if validation.get('status') != 'PASS' or validation.get('manifest_sha256') != plan['manifest_sha256']:
            raise RuntimeError('Missing matching PASS validation')
        config = pipeline.resolve_config(json.loads((Path(plan['worktree']) / f"configs/bottleneck_v1/{job['arm']}.json").read_text()))
        if job['stage'] == 'sft': config['language_model']['model_path'] = plan['qwen']
        smoke(pipeline, config, plan, job, audit)
        if args.smoke_only:
            atomic(audit / 'worker_status.json', dict(state='smoke_passed', time=now())); return 0
        # Preserve exactly the native sampling order, losses and optimizer updates.
        original_loader = pipeline.make_loader
        def loader(*a, **kw):
            def report(index):
                if index % 25 == 0:
                    atomic(audit / 'progress.json', dict(batches_in_loader=index, time=now(), pid=os.getpid()))
            return PacedLoader(original_loader(*a, **kw), lambda: memory_duty[0], torch.cuda.synchronize, report)
        pipeline.make_loader = loader
        execution = dict(source_files=plan['source_files'], profile_sha256=plan['profile_sha256'], runtime_sha256=plan['runtime_sha256'])
        if (output / 'execution.json').exists() and json.loads((output / 'execution.json').read_text()) != execution:
            raise RuntimeError('External runtime/provenance changed; refuse resume')
        original_write = pipeline.write_json
        def write(path, value):
            original_write(path, value)
            if Path(path).name == 'protocol.json': atomic(output / 'execution.json', execution)
        pipeline.write_json = write
        settings = plan['settings'][job['stage']]
        atomic(audit / 'worker_status.json', dict(state='preparing', pid=os.getpid(), time=now()))
        result = pipeline.run_training(plan['manifest'], job['output'], stage='sft' if job['stage'] == 'sft' else 'grounding',
            config=config, init_grounding=job['init'], resume=(output / 'last.pt').exists(),
            device='cuda:0', language_model_path=plan['qwen'] if job['stage'] == 'sft' else None,
            **{key: value for key, value in settings.items() if key not in {'risk_readout_only', 'freeze_grounder'}},
            risk_readout_only=job['stage'] == 'risk-readout', freeze_grounder=job['stage'] == 'sft')
        assert result['state'] == 'complete'
        atomic(audit / 'worker_status.json', dict(state='complete', training=result, time=now(), test_evaluated=False)); return 0
    except Exception as error:
        atomic(audit / 'worker_status.json', dict(state='failed', error=str(error), traceback=traceback.format_exc(), time=now()))
        traceback.print_exc(); return 1
    finally: done.set()


def jobs_for(plan):
    out = Path(plan['output_root']); jobs = []
    def add(name, arm, stage, deps=(), initial=None, override=None):
        jobs.append(dict(name=name, arm=arm, stage=stage, dependencies=list(deps), init=str(initial) if initial else None,
                         output=str(override or out / arm / stage), budget_gib=18 if stage == 'sft' else 6))
    base = out / 'baseline/grounding/epoch_020.pt'; assignment = out / 'assignment/grounding/epoch_020.pt'
    add('baseline_grounding', 'baseline', 'grounding')
    add('assignment_grounding', 'assignment', 'grounding')
    grounding_gate = ['baseline_grounding', 'assignment_grounding']
    add('baseline_readout', 'baseline', 'risk-readout', grounding_gate, base)
    add('risk_readout', 'risk', 'risk-readout', grounding_gate, base)
    add('combined_readout', 'combined', 'risk-readout', grounding_gate, assignment)
    readout_gate = ['baseline_readout', 'risk_readout', 'combined_readout']
    for arm in ['baseline', 'action', 'assignment', 'risk', 'combined']:
        dependency = 'baseline_grounding' if arm in ['baseline', 'action'] else 'assignment_grounding' if arm == 'assignment' else arm + '_readout'
        initial = base if arm in ['baseline', 'action'] else assignment if arm == 'assignment' else out / arm / 'risk-readout/epoch_010.pt'
        add(arm + '_sft', arm, 'sft', list(dict.fromkeys(readout_gate + [dependency])), initial)
    add('baseline_readout_sft', 'baseline', 'sft', readout_gate, out / 'baseline/risk-readout/epoch_010.pt', out / 'baseline_readout_sft')
    return jobs


def coordinator(args):
    plan = json.loads(args.plan.read_text()); root = args.plan.parent
    lock = (root / 'campaign.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_pins(plan)
    if (root / 'STATUS.json').exists():
        raise RuntimeError('Existing campaign state: inspect live workers before restarting; never duplicate a campaign')
    jobs = jobs_for(plan); states = {j['name']: dict(state='pending', attempts=0) for j in jobs}
    atomic(root / 'jobs.json', jobs)
    gate_log = (root / 'data_gate.log').open('ab')
    gate = subprocess.Popen(['nice','-n','19','ionice','-c','3',plan['python'],str(Path(__file__).resolve()),
                             'data-gate','--plan',str(args.plan)], cwd=plan['worktree'],
                            stdin=subprocess.DEVNULL, stdout=gate_log, stderr=subprocess.STDOUT)
    while gate.poll() is None:
        atomic(root / 'STATUS.json', dict(state='waiting_data', time=now(), data_gate_pid=gate.pid,
               reason='Waiting for manifest/PASS validation and representative radar arrays; no GPU worker admitted', jobs=states, test_evaluated=False))
        time.sleep(15)
    gate_log.close()
    if gate.returncode != 0:
        atomic(root / 'STATUS.json', dict(state='data_preflight_failed', time=now(), exit_code=gate.returncode, jobs=states)); return
    active = {}; stable = {}; loghandles = []
    pair_checked = False
    try:
        while True:
            check_pins(plan)
            for name, item in list(active.items()):
                proc, job = item
                code = proc.poll()
                if code is None: continue
                del active[name]
                if code == 0:
                    state = json.loads((Path(job['audit']) / 'worker_status.json').read_text())
                    states[name].update(state='complete' if state['state'] == 'complete' else 'failed', result=state)
                elif code == 75 and states[name]['attempts'] < 3:
                    output = Path(job['output'])
                    if output.exists() and not (output / 'last.pt').exists():
                        resolved = output.resolve(); assert resolved.is_relative_to(Path(plan['output_root']).resolve())
                        output.rename(output.with_name(output.name + f"_resource_attempt_{states[name]['attempts']}"))
                    states[name].update(state='pending', last_exit=code)
                else: states[name].update(state='failed', exit_code=code)
            for job in jobs:
                if any(states[d]['state'] in {'failed', 'blocked'} for d in job['dependencies']):
                    states[job['name']].update(state='blocked', reason='Upstream failure')
            if not pair_checked and all(states[n]['state'] == 'complete' for n in ['baseline_grounding','assignment_grounding']):
                initializations = [json.loads((Path(plan['output_root']) / arm / 'grounding/initialization.json').read_text()) for arm in ['baseline','assignment']]
                histories = [[json.loads(line) for line in (Path(plan['output_root']) / arm / 'grounding/metrics.jsonl').read_text().splitlines()] for arm in ['baseline','assignment']]
                paired = (initializations[0]['grounder_trainable_sha256'] == initializations[1]['grounder_trainable_sha256'] and
                          [x['training_order_by_rank'] for x in histories[0]] == [x['training_order_by_rank'] for x in histories[1]])
                atomic(root / 'PAIR_AUDIT.json', dict(passed=paired, time=now(), initializations=initializations,
                       semantics='Grounding baseline/assignment: exact trainable initialization and all epoch sample-order hashes'))
                pair_checked = True
                if not paired:
                    for value in states.values():
                        if value['state'] == 'pending': value.update(state='blocked', reason='Paired initialization/order audit failed')
            # Independent jobs may share the host, but never share a GPU here.
            used = {j['gpu_uuid'] for _, j in active.values()}
            available = gpus()
            for uid, card in available.items():
                acceptable = uid not in used and card['utilization'] <= 60
                stable[uid] = stable.get(uid, 0) + 1 if acceptable else 0
            for job in jobs:
                name = job['name']
                if states[name]['state'] != 'pending' or not all(states[d]['state'] == 'complete' for d in job['dependencies']): continue
                candidates = [card for uid, card in available.items() if uid not in used and stable.get(uid, 0) >= 2 and
                              card['free_mib'] >= (job['budget_gib'] + 8) * 1024]
                if not candidates: continue
                card = min(candidates, key=lambda c: (len(c['pids']) > 0, c['utilization'], -c['free_mib']))
                uid = card['uuid']; used.add(uid)
                states[name]['attempts'] += 1
                job = dict(job, gpu_uuid=uid, gpu_index=card['index'], audit=str(root / name / f"attempt_{states[name]['attempts']:02d}"))
                audit = Path(job['audit']); audit.mkdir(parents=True, exist_ok=True)
                jp = audit / 'job.json'; atomic(jp, job)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uid, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
                for key in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT']: env.pop(key, None)
                log = (audit / 'training.log').open('ab'); loghandles.append(log)
                proc = subprocess.Popen(['nice','-n','19','ionice','-c','3',plan['python'],str(Path(__file__).resolve()),'worker',
                                         '--plan',str(args.plan),'--job',str(jp)], cwd=plan['worktree'], env=env,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                active[name] = (proc, job); states[name].update(state='running', pid=proc.pid, gpu_uuid=uid, gpu_index=card['index'], audit=str(audit), output=job['output'])
                atomic(root / 'STATUS.json', dict(state='running', time=now(), jobs=states, test_evaluated=False))
                print('LAUNCHED', name, proc.pid, 'GPU', card['index'], flush=True)
                # Stagger launches; never reserve all cards at once.
                time.sleep(15)
            terminal = all(s['state'] in {'complete','failed','blocked'} for s in states.values())
            atomic(root / 'STATUS.json', dict(state='complete' if terminal and all(s['state']=='complete' for s in states.values()) else 'ended_with_failures' if terminal else 'running',
                   time=now(), jobs=states, test_evaluated=False))
            if terminal: return
            time.sleep(15)
    except Exception as error:
        atomic(root / 'STATUS.json', dict(state='coordinator_failed', time=now(), error=str(error), traceback=traceback.format_exc(), jobs=states))
        raise
    finally:
        for log in loghandles: log.close()


def data_gate(args):
    plan = json.loads(args.plan.read_text()); root = args.plan.parent
    check_pins(plan)
    atomic(root / 'DATA_GATE.json', dict(state='reading_validation', pid=os.getpid(), time=now()))
    manifest = Path(plan['manifest'])
    validation = json.loads((manifest.parent / 'validation.json').read_text())
    assert validation.get('status') == 'PASS' and validation.get('manifest_sha256') == plan['manifest_sha256'], 'No matching PASS validation'
    atomic(root / 'DATA_GATE.json', dict(state='hashing_manifest', pid=os.getpid(), time=now()))
    assert sha(manifest) == plan['manifest_sha256'], 'Manifest hash differs'
    with manifest.open() as stream:
        record = json.loads(next(line for line in stream if line.strip()))
    import numpy as np
    paths = record['sensors']['radar']; inspected = []
    atomic(root / 'DATA_GATE.json', dict(state='reading_radar_arrays', pid=os.getpid(), time=now()))
    for key in ['power','unfolded_doppler','doppler_valid']:
        values = paths.get(key, [])
        if isinstance(values, str): values = [values]
        for value in values:
            path = (manifest.parent / value).resolve()
            array = np.load(path, allow_pickle=False)
            inspected.append(dict(path=str(path), shape=list(array.shape)))
    assert inspected
    Path(plan['output_root']).mkdir(parents=True, exist_ok=True)
    atomic(root / 'DATA_GATE.json', dict(state='passed', pid=os.getpid(), time=now(), manifest_sha256=plan['manifest_sha256'], radar_arrays=inspected))
    print('Data gate passed; GPU workers may proceed', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode', choices=['worker','coordinator','data-gate']); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--job', type=Path); p.add_argument('--smoke-only', action='store_true'); a=p.parse_args()
    if a.mode == 'worker': sys.exit(worker(a))
    if a.mode == 'coordinator': coordinator(a)
    else: data_gate(a)
