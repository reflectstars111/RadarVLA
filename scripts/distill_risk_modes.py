"""Matched, finite Risk-head refinements on frozen features and a frozen Planner.

Run from the repository root. Review files go to --output; feature caches and
inference checkpoints go to --model-output and must not be exported or committed.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np
import torch

from radar_vla.data import RadarDataset, collate_batch
from radar_vla.losses import loss_grounding
from radar_vla.metrics import binary_metrics
from radar_vla.pipeline import (autocast_context, file_digest, load_models, save_checkpoint,
    source_fingerprint, to_device, write_json)
from radar_vla.risk_distillation import (configure_risk_head_training, first_mode_logits,
    is_risk_parameter, oracle_mode_kl, refinement_loss, risk_from_frozen_tokens)
from radar_vla.task_supervision import stage_available


def cpu_tree(value):
    if isinstance(value, torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value, dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list):return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):return tuple(cpu_tree(v) for v in value)
    return value


def frozen_digest(grounder, planner):
    digest=hashlib.sha256()
    for prefix,module in [('grounder',grounder),('planner',planner)]:
        for name,value in module.state_dict().items():
            if prefix=='grounder' and is_risk_parameter(name):continue
            digest.update((prefix+'.'+name+str(value.dtype)+str(tuple(value.shape))).encode())
            digest.update(value.detach().reshape(-1).cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def mode_metrics(logits, truth):
    chosen=logits.argmax(-1).numpy();truth=np.asarray(truth)
    tp=int(((truth==0)&(chosen==0)).sum());fp=int(((truth==1)&(chosen==0)).sum())
    positives=int((truth==0).sum());negatives=int((truth==1).sum())
    return dict(samples=len(truth),confusion=dict(short=dict(short=tp,long=positives-tp),
        long=dict(short=fp,long=negatives-fp)),short_recall=tp/positives if positives else None,
        short_precision=tp/(tp+fp) if tp+fp else None,
        balanced_accuracy=.5*(tp/positives+(negatives-fp)/negatives) if positives and negatives else None)


def summarize(cache, risk, risk_logits, mode):
    known=cache['risk_mask'].all(-1);truth=[0 if r['teacher_mode']=='short' else 1 for r in cache['rows']]
    complete=torch.nonzero(known).flatten();oracle=cache['oracle_mode_logits'][known]
    pred=mode[known];teacher_choice=oracle.argmax(-1);chosen=pred.argmax(-1)
    per=oracle.softmax(-1)*(oracle.log_softmax(-1)-pred.log_softmax(-1))
    per=per.sum(-1)
    oracle_recall={str(i):float((chosen[teacher_choice==i]==i).float().mean())
        if bool((teacher_choice==i).any()) else None for i in (0,1)}
    result=dict(samples=len(truth),complete_risk_samples=int(known.sum()),
        all_sft=mode_metrics(mode,truth),complete_risk=mode_metrics(pred,[truth[i] for i in complete.tolist()]),
        oracle=mode_metrics(oracle,[truth[i] for i in complete.tolist()]),
        oracle_agreement=float((chosen==teacher_choice).float().mean()),
        oracle_balanced_agreement=float(np.mean([v for v in oracle_recall.values() if v is not None])),
        oracle_short_agreement=oracle_recall['0'],oracle_long_agreement=oracle_recall['1'],
        oracle_mode_kl=float(per.mean()),risk={},
        probability_order_violations=int((risk[:,:2]>risk[:,1:3]).any(-1).sum()))
    for index,name in enumerate(['pcol_1s','pcol_2s','pcol_3s','dmin','areq']):
        valid=cache['risk_mask'][:,index];values=risk[valid,index];target=cache['risk_target'][valid,index]
        if index<3:result['risk'][name]=binary_metrics(values.tolist(),target.tolist())
        else:
            positive=target>0
            result['risk'][name]=dict(samples=len(target),mae=float((values-target).abs().mean()),
                positive_gt_samples=int(positive.sum()),
                positive_gt_mae=float((values[positive]-target[positive]).abs().mean()) if bool(positive.any()) else None)
    physical=loss_grounding(dict(risk=risk,risk_logits=risk_logits),
        dict(risk_target=cache['risk_target'],risk_mask=cache['risk_mask']))
    result['physical_loss']={k:float(physical[k]) for k in ['total','collision','distance','brake','monotonicity']}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','manifest','baseline','output','model-output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--lr',type=float,default=3e-4);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--temperature',type=float,default=1.);parser.add_argument('--mode-weight',type=float,default=1.)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.lr<=0 or args.mode_weight<=0 or args.temperature<=0:raise ValueError('Invalid experiment settings')
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'comparison.json').exists():raise ValueError('Completed experiment already exists; use a new output directory')
    args.model_output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.24,0)
    assert torch.cuda.mem_get_info()[0]>=32*1024**3,'Insufficient spare VRAM'
    finished=threading.Event()

    def status(phase,**kwargs):
        value=dict(state='running',phase=phase,pid=os.getpid(),test_evaluated=False,**kwargs)
        write_json(root/'status.json',value);print(json.dumps(value),flush=True)

    def guard():
        while not finished.wait(10):
            free=torch.cuda.mem_get_info()[0]
            if free<8*1024**3:
                write_json(root/'resource_guard.json',dict(pid=os.getpid(),free_bytes=free,reason='Our process exits below 8GiB free VRAM'))
                os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=guard,daemon=True).start()
    try:
        status('loading_frozen_base')
        previous=json.loads((args.baseline/'protocol.json').read_text())
        assert file_digest(args.checkpoint)==previous['checkpoint_sha256']
        assert file_digest(args.manifest)==previous['manifest_sha256']
        original={r['sample_id']:r for r in map(json.loads,(args.baseline/'samples.jsonl').open())}
        grounder,planner,state=load_models(args.checkpoint,'cuda:0')
        assert grounder.config.planner_ego_context and grounder.config.use_ego
        parameters=configure_risk_head_training(grounder,planner)
        template=cpu_tree(dict(protocol=state['protocol'],epoch=state['epoch'],grounder=state['grounder'],planner=state['planner']))
        del state;torch.cuda.empty_cache()
        initial_head={name:value.detach().cpu().clone() for name,value in grounder.state_dict().items() if is_risk_parameter(name)}
        sources=source_fingerprint()
        protocol=dict(schema='radar_vla_oracle_mode_distillation_v1',source_files=sources,
            script_sha256=file_digest(__file__),checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=previous['checkpoint_sha256'],manifest=str(args.manifest.resolve()),
            manifest_sha256=previous['manifest_sha256'],epochs=args.epochs,batch_size=args.batch_size,
            lr=args.lr,seed=args.seed,temperature=args.temperature,distillation_weight=args.mode_weight,
            physical_weights='existing DEFAULT_WEIGHTS; focal alpha=.25 gamma=2; unchanged',
            risk_projection='none',trainable_names=[n for n,p in grounder.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in parameters),
            frozen_planner=True,frozen_encoder=True,teacher_mode_only=True,teacher_stop_gradient=True,
            student_uses_gt=False,test_evaluated=False,epoch_selection='fixed final epoch; report every epoch; no automatic promotion')
        write_json(root/'protocol.json',protocol)
        before=frozen_digest(grounder,planner)
        protocol['frozen_tensor_sha256']=before;write_json(root/'protocol.json',protocol)
        caches={}
        for split in ['train','val']:
            dataset=RadarDataset(args.manifest,split,max_agents=None)
            dataset.records=[r for r in dataset.records if stage_available(r,'sft')]
            if split=='val':assert {r['sample_id'] for r in dataset.records}==set(original)
            pieces=[]
            for start in range(0,len(dataset),256):
                end=min(start+256,len(dataset));path=args.model_output/'cache'/f'{split}_{start:05d}.pt'
                path.parent.mkdir(exist_ok=True)
                expected=[r['sample_id'] for r in dataset.records[start:end]]
                if path.exists():
                    chunk=torch.load(path,map_location='cpu',weights_only=True)
                    assert chunk['source_files']==sources and chunk['checkpoint_sha256']==previous['checkpoint_sha256']
                    assert chunk['manifest_sha256']==previous['manifest_sha256']
                    assert [r['sample_id'] for r in chunk['rows']]==expected
                else:
                    packets=[]
                    with torch.no_grad(),autocast_context('cuda:0','bfloat16'):
                        for i in range(start,end):
                            record=dataset.records[i];batch=to_device(collate_batch([dataset[i]]),'cuda:0')
                            output=grounder(batch)
                            cached=risk_from_frozen_tokens(grounder,output['radar_tokens'])
                            for key in ['risk','risk_logits']:torch.testing.assert_close(cached[key],output[key],rtol=0,atol=0)
                            mode=first_mode_logits(planner,output['radar_tokens'],output['risk'],batch['instruction'])
                            oracle=torch.zeros_like(mode)
                            if bool(batch['risk_mask'].all()):
                                oracle=first_mode_logits(planner,output['radar_tokens'],batch['risk_target'],batch['instruction'])
                            if split=='val':
                                base=original[record['sample_id']]
                                np.testing.assert_allclose(output['risk'][0].float().cpu().numpy(),base['risk'],rtol=0,atol=1e-5)
                                assert ('short' if int(mode.argmax(-1))==0 else 'long')==base['plan']['mode']
                            packets.append(dict(row=dict(sample_id=record['sample_id'],episode_id=record['episode_id'],
                                teacher_mode=record['teacher_mode'],instruction=batch['instruction'][0]),
                                radar_tokens=output['radar_tokens'][0].cpu(),risk_target=batch['risk_target'][0].cpu(),
                                risk_mask=batch['risk_mask'][0].cpu(),risk=output['risk'][0].float().cpu(),
                                risk_logits=output['risk_logits'][0].float().cpu(),mode_logits=mode[0].cpu(),oracle_mode_logits=oracle[0].cpu()))
                            if (i+1)%100==0:status('caching_frozen_features',split=split,done=i+1,total=len(dataset))
                    chunk=dict(rows=[p['row'] for p in packets],source_files=sources,
                        checkpoint_sha256=previous['checkpoint_sha256'],manifest_sha256=previous['manifest_sha256'])
                    for key in ['radar_tokens','risk_target','risk_mask','risk','risk_logits','mode_logits','oracle_mode_logits']:
                        chunk[key]=torch.stack([p[key] for p in packets])
                    save_checkpoint(path,chunk)
                pieces.append(chunk)
                status('caching_frozen_features',split=split,done=end,total=len(dataset))
            cache=dict(rows=[r for p in pieces for r in p['rows']])
            for key in ['radar_tokens','risk_target','risk_mask','risk','risk_logits','mode_logits','oracle_mode_logits']:
                cache[key]=torch.cat([p[key] for p in pieces])
            caches[split]=cache
        assert not {r['episode_id'] for r in caches['train']['rows']} & {r['episode_id'] for r in caches['val']['rows']}
        baseline=summarize(caches['val'],caches['val']['risk'],caches['val']['risk_logits'],caches['val']['mode_logits'])
        oracle_reference=json.loads((args.baseline.parent.parent/'risk_input_followup_20261018/summary.json').read_text())
        assert baseline['complete_risk']['confusion']=={k:{p:v for p,v in value.items() if p!='invalid'}
            for k,value in oracle_reference['groups']['all']['modes']['predicted']['confusion'].items()}
        assert baseline['oracle']['confusion']=={k:{p:v for p,v in value.items() if p!='invalid'}
            for k,value in oracle_reference['groups']['all']['modes']['oracle']['confusion'].items()}
        write_json(root/'baseline.json',baseline)
        training=caches['train'];validation=caches['val'];variants={}

        def batch_at(cache,indices):
            value={key:cache[key][indices].to('cuda:0') for key in ['radar_tokens','risk_target','risk_mask','oracle_mode_logits']}
            value['instruction']=[cache['rows'][i]['instruction'] for i in indices.tolist()]
            return value

        def evaluate():
            risks=[];raw=[];modes=[]
            with torch.no_grad(),autocast_context('cuda:0','bfloat16'):
                for i in range(len(validation['rows'])):
                    batch=batch_at(validation,torch.tensor([i]));prediction=risk_from_frozen_tokens(grounder,batch['radar_tokens'])
                    logits=first_mode_logits(planner,batch['radar_tokens'],prediction['risk'],batch['instruction'])
                    risks.append(prediction['risk'].float().cpu());raw.append(prediction['risk_logits'].float().cpu());modes.append(logits.cpu())
            risk=torch.cat(risks);risk_logits=torch.cat(raw);mode=torch.cat(modes)
            return summarize(validation,risk,risk_logits,mode),risk,risk_logits,mode

        for variant,weight in [('physical_only',0.),('physical_plus_kl',args.mode_weight)]:
            with torch.no_grad():
                current_state=grounder.state_dict()
                for name,value in initial_head.items():current_state[name].copy_(value.to(current_state[name]))
            torch.manual_seed(args.seed)
            optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.01)
            history=[]
            for epoch in range(1,args.epochs+1):
                generator=torch.Generator().manual_seed(args.seed+epoch)
                order=torch.randperm(len(training['rows']),generator=generator)
                sums={key:0. for key in ['total','physical','mode_kl','collision','distance','brake','monotonicity']}
                began=time.monotonic()
                for start in range(0,len(order),args.batch_size):
                    indices=order[start:start+args.batch_size];batch=batch_at(training,indices)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context('cuda:0','bfloat16'):
                        losses=refinement_loss(grounder,planner,batch,weight,args.temperature)
                    assert bool(torch.isfinite(losses['total']))
                    losses['total'].backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                    assert all(p.grad is None for p in planner.parameters())
                    optimizer.step()
                    for key in sums:sums[key]+=float(losses[key].detach())*len(indices)
                    if (start//args.batch_size+1)%100==0:
                        status('training',variant=variant,epoch=epoch,done=min(start+len(indices),len(order)),total=len(order))
                status('validating_mode_and_physics',variant=variant,epoch=epoch)
                measured,risk,risk_logits,mode=evaluate()
                item=dict(epoch=epoch,training={k:v/len(order) for k,v in sums.items()},validation=measured,
                    elapsed_seconds=time.monotonic()-began)
                history.append(item);write_json(root/f'{variant}_history.json',history)
                saved=args.model_output/variant;saved.mkdir(exist_ok=True)
                save_checkpoint(saved/f'head_epoch_{epoch:03d}.pt',dict(risk_head=cpu_tree({n:v for n,v in grounder.state_dict().items() if is_risk_parameter(n)}),
                    optimizer=cpu_tree(optimizer.state_dict()),epoch=epoch,protocol=protocol,mode_weight=weight))
                print(json.dumps(dict(variant=variant,epoch=epoch,mode=measured['complete_risk'],oracle_kl=measured['oracle_mode_kl'],pcol3=measured['risk']['pcol_3s'])),flush=True)
            assert frozen_digest(grounder,planner)==before,'A frozen tensor changed'
            assert all(p.grad is None for n,p in grounder.named_parameters() if not is_risk_parameter(n))
            refinement=dict(variant=variant,epochs=args.epochs,mode_weight=weight,temperature=args.temperature,
                base_checkpoint_sha256=previous['checkpoint_sha256'],frozen_tensors_unchanged=True,protocol=protocol,history=history)
            output_checkpoint=args.model_output/variant/'refined_inference.pt'
            exported=dict(protocol=copy.deepcopy(template['protocol']),epoch=template['epoch'],
                grounder=cpu_tree(grounder.state_dict()),planner=template['planner'],
                risk_refinement=refinement,inference_only=True)
            exported['protocol']['source_files']=sources;exported['protocol']['risk_refinement']=dict(variant=variant,head_epochs=args.epochs,mode_weight=weight)
            save_checkpoint(output_checkpoint,exported)
            variants[variant]=dict(metrics=history[-1]['validation'],checkpoint=str(output_checkpoint.resolve()),
                checkpoint_sha256=file_digest(output_checkpoint),frozen_tensors_unchanged=True,epochs=args.epochs)
            with (root/f'{variant}_samples.jsonl').open('w') as stream:
                for i,row in enumerate(validation['rows']):
                    item=dict(row,risk=risk[i].tolist(),risk_target=validation['risk_target'][i].tolist(),
                        risk_mask=validation['risk_mask'][i].tolist(),mode_logits=mode[i].tolist(),
                        baseline_risk=validation['risk'][i].tolist(),baseline_mode_logits=validation['mode_logits'][i].tolist(),
                        oracle_mode_logits=validation['oracle_mode_logits'][i].tolist() if bool(validation['risk_mask'][i].all()) else None)
                    stream.write(json.dumps(item,allow_nan=False)+'\n')
            write_json(root/f'{variant}_result.json',variants[variant])
        assert source_fingerprint()==sources
        assert file_digest(args.checkpoint)==previous['checkpoint_sha256']
        write_json(root/'comparison.json',dict(protocol=protocol,baseline=baseline,variants=variants,
            cache_counts={s:len(c['rows']) for s,c in caches.items()},frozen_tensors_unchanged=True,
            original_checkpoint_unchanged=True,full_trajectory_evaluated=False,test_evaluated=False,
            automatic_checkpoint_promotion=False))
        write_json(root/'status.json',dict(state='complete',phase='risk_and_mode_comparison',pid=os.getpid(),test_evaluated=False))
    except BaseException as error:
        write_json(root/'status.json',dict(state='failed',pid=os.getpid(),error=str(error)));raise
    finally:finished.set()


if __name__=='__main__':main()
