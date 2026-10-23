"""A single follow-up control: Pcol3-only teacher includes censored-areq rows."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

import torch

from radar_vla.losses import loss_grounding
from radar_vla.pipeline import (autocast_context,file_digest,load_models,save_checkpoint,source_fingerprint,write_json)
from radar_vla.risk_distillation import (configure_risk_head_training,first_mode_logits,is_risk_parameter,
    oracle_mode_kl,risk_from_frozen_tokens)


def pcol3_teacher_condition(predicted, target, mask):
    """Keep predicted P1/P2/dmin/areq; replace only a valid GT P3."""
    if predicted.shape!=target.shape or mask.shape!=target.shape or target.shape[-1]!=5:
        raise ValueError('Risk tensors must have matching [B,5] shapes')
    known=mask[:,2].bool() & torch.isfinite(target[:,2])
    if bool(((target[known,2]<0)|(target[known,2]>1)).any()):raise ValueError('Invalid GT Pcol3')
    teacher=predicted.detach().clone();teacher[known,2]=target[known,2].to(teacher.dtype)
    return teacher,known


def pcol3_refinement_loss(grounder,planner,batch):
    prediction=risk_from_frozen_tokens(grounder,batch['radar_tokens'])
    physical=loss_grounding(prediction,dict(risk_target=batch['risk_target'],risk_mask=batch['risk_mask']))
    known=batch['risk_mask'][:,2]&torch.isfinite(batch['risk_target'][:,2])
    student=first_mode_logits(planner,batch['radar_tokens'],prediction['risk'],batch['instruction'])
    kl=oracle_mode_kl(student,batch['pcol3_teacher_logits'],known,1.)
    return dict(total=physical['total']+kl,physical=physical['total'],mode_kl=kl,
        collision=physical['collision'],distance=physical['distance'],brake=physical['brake'])


def main():
    # These are the canonical sibling script's bookkeeping, not another network.
    from distill_risk_modes import cpu_tree,frozen_digest,summarize,mode_metrics
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['base-experiment','output','model-output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();base=args.base_experiment;root=args.output
    root.mkdir(parents=True,exist_ok=True);args.model_output.mkdir(parents=True,exist_ok=True)
    assert not (root/'result.json').exists(),'Completed follow-up already exists'
    comparison=json.loads((base/'comparison.json').read_text());old=comparison['protocol']
    assert source_fingerprint()==old['source_files'],'Canonical neural source must stay fixed for this control'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.24,0)
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    finished=threading.Event()

    def status(phase,**kwargs):
        value=dict(state='running',phase=phase,pid=os.getpid(),**kwargs)
        write_json(root/'status.json',value);print(json.dumps(value),flush=True)

    def guard():
        while not finished.wait(10):
            if torch.cuda.mem_get_info()[0]<8*1024**3:
                write_json(root/'resource_guard.json',dict(reason='Our worker exits below 8GiB free VRAM',pid=os.getpid()))
                os.kill(os.getpid(),signal.SIGTERM)
    threading.Thread(target=guard,daemon=True).start()
    try:
        status('loading_same_frozen_base')
        assert file_digest(old['checkpoint'])==old['checkpoint_sha256']
        grounder,planner,state=load_models(old['checkpoint'],'cuda:0')
        parameters=configure_risk_head_training(grounder,planner)
        template=cpu_tree(dict(protocol=state['protocol'],epoch=state['epoch'],planner=state['planner']))
        del state;torch.cuda.empty_cache()
        before=frozen_digest(grounder,planner)
        protocol=dict(old,teacher='GT Pcol3 only; other four components remain base predictions',
            teacher_mask='valid Pcol3; censored areq does not exclude mode supervision',
            comparison_parent=str(base.resolve()),script_sha256=file_digest(__file__),
            canonical_base_script_sha256=old['script_sha256'],frozen_tensor_sha256=before)
        write_json(root/'protocol.json',protocol)
        assets=Path(comparison['variants']['physical_only']['checkpoint']).parent.parent
        caches={};teacher_stats={}
        for split in ['train','val']:
            chunks=[torch.load(path,map_location='cpu',weights_only=True) for path in sorted((assets/'cache').glob(split+'_*.pt'))]
            assert all(c['source_files']==old['source_files'] and c['checkpoint_sha256']==old['checkpoint_sha256'] for c in chunks)
            cache=dict(rows=[r for c in chunks for r in c['rows']])
            for key in ['radar_tokens','risk_target','risk_mask','risk','risk_logits','mode_logits','oracle_mode_logits']:
                cache[key]=torch.cat([c[key] for c in chunks])
            logits=[];knowns=[]
            with torch.no_grad(),autocast_context('cuda:0','bfloat16'):
                for i,row in enumerate(cache['rows']):
                    tokens=cache['radar_tokens'][i:i+1].to('cuda:0')
                    predicted=cache['risk'][i:i+1].to('cuda:0',dtype=tokens.dtype)
                    target=cache['risk_target'][i:i+1].to('cuda:0');mask=cache['risk_mask'][i:i+1].to('cuda:0')
                    teacher,known=pcol3_teacher_condition(predicted,target,mask)
                    value=first_mode_logits(planner,tokens,teacher,[row['instruction']])
                    logits.append(value.cpu());knowns.append(known.cpu())
                    if (i+1)%500==0:status('preparing_pcol3_teacher',split=split,done=i+1,total=len(cache['rows']))
            cache['pcol3_teacher_logits']=torch.cat(logits);known=torch.cat(knowns)
            truth=[0 if r['teacher_mode']=='short' else 1 for r in cache['rows']]
            teacher_stats[split]=mode_metrics(cache['pcol3_teacher_logits'][known],[truth[i] for i in torch.nonzero(known).flatten().tolist()])
            caches[split]=cache
            save_checkpoint(args.model_output/f'{split}_pcol3_teacher.pt',dict(logits=cache['pcol3_teacher_logits'],
                known=known,sample_ids=[r['sample_id'] for r in cache['rows']],protocol=protocol))
        write_json(root/'teacher_metrics.json',teacher_stats)
        train=caches['train'];val=caches['val'];history=[]

        def batch_at(cache,indices):
            value={key:cache[key][indices].to('cuda:0') for key in ['radar_tokens','risk_target','risk_mask','pcol3_teacher_logits']}
            value['instruction']=[cache['rows'][i]['instruction'] for i in indices.tolist()]
            return value

        optimizer=torch.optim.AdamW(parameters,lr=old['lr'],weight_decay=.01)
        for epoch in range(1,old['epochs']+1):
            order=torch.randperm(len(train['rows']),generator=torch.Generator().manual_seed(old['seed']+epoch))
            sums={k:0. for k in ['total','physical','mode_kl','collision','distance','brake']};began=time.monotonic()
            for start in range(0,len(order),old['batch_size']):
                indices=order[start:start+old['batch_size']];batch=batch_at(train,indices)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context('cuda:0','bfloat16'):losses=pcol3_refinement_loss(grounder,planner,batch)
                assert bool(torch.isfinite(losses['total']))
                losses['total'].backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                assert all(p.grad is None for p in planner.parameters());optimizer.step()
                for key in sums:sums[key]+=float(losses[key].detach())*len(indices)
                if (start//old['batch_size']+1)%100==0:status('training',epoch=epoch,done=start+len(indices),total=len(order))
            risks=[];raw=[];modes=[];status('validating',epoch=epoch)
            with torch.no_grad(),autocast_context('cuda:0','bfloat16'):
                for i in range(len(val['rows'])):
                    batch=batch_at(val,torch.tensor([i]));prediction=risk_from_frozen_tokens(grounder,batch['radar_tokens'])
                    mode=first_mode_logits(planner,batch['radar_tokens'],prediction['risk'],batch['instruction'])
                    risks.append(prediction['risk'].float().cpu());raw.append(prediction['risk_logits'].float().cpu());modes.append(mode.cpu())
            risk=torch.cat(risks);risk_logits=torch.cat(raw);mode=torch.cat(modes)
            measured=summarize(val,risk,risk_logits,mode)
            history.append(dict(epoch=epoch,training={k:v/len(order) for k,v in sums.items()},validation=measured,
                seconds=time.monotonic()-began));write_json(root/'history.json',history)
            save_checkpoint(args.model_output/f'head_epoch_{epoch:03d}.pt',dict(epoch=epoch,
                head=cpu_tree({n:v for n,v in grounder.state_dict().items() if is_risk_parameter(n)}),protocol=protocol))
            print(json.dumps(dict(epoch=epoch,mode=measured['complete_risk'],pcol3=measured['risk']['pcol_3s'])),flush=True)
        assert frozen_digest(grounder,planner)==before
        assert source_fingerprint()==old['source_files']
        checkpoint=args.model_output/'refined_inference.pt'
        refinement=dict(variant='physical_plus_pcol3_teacher_kl',epochs=old['epochs'],mode_weight=1.,temperature=1.,
            frozen_tensors_unchanged=True,protocol=protocol,history=history)
        exported=dict(protocol=copy.deepcopy(template['protocol']),epoch=template['epoch'],grounder=cpu_tree(grounder.state_dict()),
            planner=template['planner'],risk_refinement=refinement,inference_only=True)
        exported['protocol']['source_files']=old['source_files'];exported['protocol']['risk_refinement']=dict(
            variant=refinement['variant'],head_epochs=old['epochs'],mode_weight=1.,teacher='Pcol3 only')
        save_checkpoint(checkpoint,exported)
        with (root/'samples.jsonl').open('w') as stream:
            for i,row in enumerate(val['rows']):
                stream.write(json.dumps(dict(row,risk=risk[i].tolist(),risk_target=val['risk_target'][i].tolist(),
                    risk_mask=val['risk_mask'][i].tolist(),mode_logits=mode[i].tolist(),
                    full_oracle_mode_logits=val['oracle_mode_logits'][i].tolist() if bool(val['risk_mask'][i].all()) else None,
                    pcol3_teacher_mode_logits=val['pcol3_teacher_logits'][i].tolist()),allow_nan=False)+'\n')
        write_json(root/'result.json',dict(protocol=protocol,metrics=history[-1]['validation'],teacher_metrics=teacher_stats,
            checkpoint=str(checkpoint.resolve()),checkpoint_sha256=file_digest(checkpoint),
            frozen_tensors_unchanged=True,full_trajectory_evaluated=False,test_evaluated=False,automatic_promotion=False))
        write_json(root/'status.json',dict(state='complete',pid=os.getpid(),test_evaluated=False))
    except BaseException as error:
        write_json(root/'status.json',dict(state='failed',pid=os.getpid(),error=str(error)));raise
    finally:finished.set()


if __name__=='__main__':main()
