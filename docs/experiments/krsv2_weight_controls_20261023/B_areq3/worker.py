"""One isolated single-variable weight arm; never controls other workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

import torch
from radar_vla import pipeline
from radar_vla.data import RadarDataset,collate_batch
from radar_vla.model import ModelConfig,RadarVLAGrounder
from observer import Observer,atomic
from matching_reference import matched_objects


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    plan=json.loads(args.plan.read_text());root=args.plan.parent
    audit=root/('smoke' if args.smoke else 'run');audit.mkdir(exist_ok=True)
    output=Path(plan['output'])
    execution=dict(loss_weights=plan['loss_weights'],runtime_sha256=plan['runtime_sha256'],
                   config_sha256=plan['config_sha256'],source_files=plan['source_files'])
    execution_path=output/'execution.json'
    if execution_path.exists():
        if json.loads(execution_path.read_text())!=execution:
            raise ValueError('Weight/runtime execution differs; refusing resume')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    assert pipeline.source_fingerprint()==plan['source_files']
    assert pipeline.file_digest(plan['manifest'])==plan['manifest_sha256']
    assert pipeline.file_digest(plan['config'])==plan['config_sha256']
    for name,digest in plan['runtime_sha256'].items():assert pipeline.file_digest(root/name)==digest,name
    torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(plan['budget_gib']*1024**3/total,0)
    assert torch.cuda.mem_get_info()[0]>(plan['budget_gib']+8)*1024**3
    runtime=dict(pid=os.getpid(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),nice=os.getpriority(os.PRIO_PROCESS,0),
        ionice=subprocess.check_output(['ionice','-p',str(os.getpid())],text=True).strip(),
        cpu_threads=torch.get_num_threads(),data_workers=0,budget_gib=plan['budget_gib'],
        source_commit=plan['source_commit'],gpu_duty=plan.get('gpu_duty',1.),cudnn_deterministic=False,cudnn_benchmark=False,
        tf32_matmul=False,tf32_cudnn=True)
    atomic(audit/'runtime.json',runtime)
    done=threading.Event()
    def guard():
        while not done.wait(2):
            if torch.cuda.mem_get_info()[0]<8*1024**3:
                atomic(audit/'status.json',dict(state='resource_exit',reason='Less than8GiB free VRAM; only this worker exits',pid=os.getpid()))
                os._exit(75)
    threading.Thread(target=guard,daemon=True).start()
    try:
        config=json.loads(Path(plan['config']).read_text())
        assert config['loss_weights']==plan['loss_weights']
        if args.smoke:
            torch.manual_seed(plan['seed'])
            dataset=RadarDataset(plan['manifest'],'train')
            positive=next(i for i,r in enumerate(dataset.records) if r['local_risk']['pcol'][0])
            empty=next(i for i,r in enumerate(dataset.records) if not r['agents'])
            many=max(range(len(dataset)),key=lambda i:len(dataset.records[i]['agents']))
            selected=list(dict.fromkeys([positive,empty,many,0]))
            for i in range(len(dataset)):
                if len(selected)==4:break
                if i not in selected:selected.append(i)
            assert len(selected)==4
            dataset.records=[dataset.records[i] for i in selected];dataset._risk_cache.clear()
            model=RadarVLAGrounder(ModelConfig(**config['model'])).cuda().train()
            model.loss_weights=dict(config['loss_weights'])
            obs=Observer(pipeline,output,audit,matched_objects);obs.epoch=1
            batch=pipeline.to_device(collate_batch([dataset[i] for i in range(4)]),'cuda:0')
            with pipeline.autocast_context('cuda:0','bfloat16'):loss,components,pred=obs.loss(model,None,batch)
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(g.isfinite().all() for g in gradients)
            assert pred['risk'].shape==(4,5) and pred['detection_logits'].shape==(4,3456)
            validation=obs.validate(model,None,dataset,4,'cuda:0','bfloat16')
            atomic(audit/'result.json',dict(state='passed',samples=batch['sample_id'],loss=float(loss),
                validation_loss=validation['loss'],loss_weights=config['loss_weights'],runtime_sha256=plan['runtime_sha256'],
                optimizer_updates=0,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
            atomic(audit/'status.json',dict(state='passed'))
            print(json.dumps(dict(state='smoke_passed',loss=float(loss))),flush=True);return
        obs=Observer(pipeline,output,audit,matched_objects);obs.install()
        duty=float(plan.get('gpu_duty',1.))
        if not 0<duty<=1:raise ValueError('Invalid GPU duty')
        if duty<1:
            original_loader=pipeline.make_loader
            class PacedLoader:
                def __init__(self,loader):self.loader=loader
                def __len__(self):return len(self.loader)
                def __iter__(self):
                    for batch in self.loader:
                        torch.cuda.synchronize();began=time.monotonic()
                        yield batch
                        torch.cuda.synchronize()
                        delay=(time.monotonic()-began)*(1/duty-1)
                        if delay>0:time.sleep(delay)
            pipeline.make_loader=lambda *a,**kw:PacedLoader(original_loader(*a,**kw))
        original_write=pipeline.write_json
        def write_protocol(path,value):
            original_write(path,value)
            if Path(path).name=='protocol.json':atomic(execution_path,execution)
        pipeline.write_json=write_protocol
        atomic(audit/'status.json',dict(state='preparing',pid=os.getpid(),plan=str(args.plan),fresh_initialization=True,loss_weights=config['loss_weights']))
        result=pipeline.run_training(plan['manifest'],plan['output'],stage='grounding',epochs=20,
            batch_size=4,accumulation_steps=2,lr=2e-4,seed=42,device='cuda:0',precision='bfloat16',workers=0,
            config=config,init_grounding=None,resume=(output/'last.pt').exists(),validation_generation_samples=0,
            lr_schedule='flat_cosine',lr_decay_start_epoch=5,lr_min_ratio=.1,lr_schedule_unit='update')
        assert result['state']=='complete' and result['epoch']==20
        atomic(audit/'status.json',dict(state='complete',pid=os.getpid(),training=result,test_evaluated=False,
            automatic_weight_tuning_started=False))
    except BaseException as error:
        atomic(audit/'status.json',dict(state='failed',pid=os.getpid(),error=str(error),traceback=traceback.format_exc()))
        raise
    finally:done.set()


if __name__=='__main__':main()
