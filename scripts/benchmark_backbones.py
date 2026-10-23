"""Accepted-data forward/backward resource dry-run; never constructs an optimizer."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import torch
from radar_vla.data import RadarDataset, collate_batch
from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.losses import loss_grounding
from radar_vla.detection import decode_detections
from radar_vla.pipeline import autocast_context, file_digest, source_fingerprint, to_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbones', nargs='+', default=['cnn2d', 'rodnet_hg'])
    parser.add_argument('--legacy-checkpoint', type=Path)
    args = parser.parse_args()
    source_before=source_fingerprint()
    script_sha256=file_digest(__file__)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'results.json').exists(), 'Never overwrite finished measurements'
    acceptance = json.loads((args.manifest.parent/'acceptance.json').read_text())
    validation = json.loads((args.manifest.parent/'validation.json').read_text())
    assert acceptance['status']=='PASS' and validation['status']=='PASS'
    assert file_digest(args.manifest)==acceptance['manifest_sha256']==validation['manifest_sha256']
    selected = {}
    with args.manifest.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['split']=='test':
                continue
            n = len(record['agents'])
            cases = ['empty'] if n==0 else ['many'] if n>8 else []
            if record.get('mode_label')=='SHORT':
                cases.append('short')
            if record.get('mode_label')=='UNKNOWN':
                cases.append('unknown')
            for case in cases:
                selected.setdefault(case,record)
            if len(selected)==4:
                break
    assert set(selected)=={'empty','many','short','unknown'}
    manifest = args.output/'representative_samples.jsonl'
    unique = {r['sample_id']:r for r in selected.values()}
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in unique.values()))
    datasets = {split:RadarDataset(manifest,split,max_agents=None) for split in {r['split'] for r in unique.values()}}
    samples = {r['sample_id']:collate_batch([ds[i]]) for ds in datasets.values() for i,r in enumerate(ds.records)}
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.40)
    assert torch.cuda.mem_get_info()[0]>=32*1024**3
    done = threading.Event()
    def guard():
        while not done.wait(2):
            if torch.cuda.mem_get_info()[0]<8*1024**3:
                write_json(args.output/'guard.json',dict(reason='Self-exit below 8GiB free VRAM',pid=os.getpid()))
                os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=guard,daemon=True).start()
    config = json.loads(args.config.read_text())
    legacy_check = None
    if args.legacy_checkpoint:
        state=torch.load(args.legacy_checkpoint,map_location='cpu',weights_only=True,mmap=True)
        options=state['protocol']['config']['model']
        spec=importlib.util.spec_from_file_location('radar_vla._legacy_checkpoint_reference',
            Path(__file__).resolve().parents[1]/'tests/fixtures/legacy_cnn_model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        old=module.RadarVLAGrounder(module.ModelConfig(**options)).cuda().eval()
        new=RadarVLAGrounder(ModelConfig(**options)).cuda().eval()
        old.load_state_dict(state['grounder']);new.load_state_dict(state['grounder'])
        batch=to_device(next(iter(samples.values())),'cuda:0')
        with torch.no_grad(),autocast_context('cuda:0','bfloat16'):
            a,b=old(batch),new(batch)
        assert a.keys()==b.keys()
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        legacy_check=dict(checkpoint=str(args.legacy_checkpoint),sha256=file_digest(args.legacy_checkpoint),
            same_state_keys=list(old.state_dict())==list(new.state_dict()),all_outputs_bitwise_equal=True)
        del old,new,state,batch,a,b;torch.cuda.empty_cache()
    results = {}
    reference_anchors = {}
    for backbone in args.backbones:
        torch.manual_seed(42)
        options = dict(config['model'],backbone=backbone,
                       backbone_options=(config['model'].get('backbone_options')
                                         if config['model'].get('backbone','cnn2d')==backbone else None))
        m = RadarVLAGrounder(ModelConfig(**options)).cuda().eval()
        feature_shapes=[]
        shape_hook=m.encoder.register_forward_hook(lambda module,inputs,out: feature_shapes.append(list(out.shape)))
        initial = {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        rows = []
        for case,record in selected.items():
            batch = to_device(samples[record['sample_id']], 'cuda:0')
            assert tuple(batch['radar'].shape)==(1,4,2,256,107)
            m.eval()
            with torch.no_grad(), autocast_context('cuda:0','bfloat16'):
                m(batch) # warmup
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(3):
                torch.cuda.synchronize();began=time.perf_counter()
                with torch.no_grad(), autocast_context('cuda:0','bfloat16'):
                    pred=m(batch)
                torch.cuda.synchronize();times.append(time.perf_counter()-began)
            inference_peak = torch.cuda.max_memory_allocated()/1024**2
            assert pred['detection_logits'].shape==(1,3456)
            expected_tokens=m.config.num_queries+int(m.config.planner_ego_context)
            assert pred['radar_tokens'].shape==(1,expected_tokens,m.config.hidden_dim)
            assert pred['risk'].shape==(1,5)
            assert all(torch.isfinite(v).all() for v in pred.values())
            anchors = pred['detection_anchors'].detach().cpu()
            if record['sample_id'] in reference_anchors:
                torch.testing.assert_close(anchors,reference_anchors[record['sample_id']],rtol=0,atol=0)
            else:
                reference_anchors[record['sample_id']]=anchors
            decode_detections(pred,m.config,batch['future_times_s']) # unchanged decoder/NMS path
            del pred
            m.train();m.zero_grad(set_to_none=True)
            batch=dict(batch,radar=batch['radar'].detach().requires_grad_(True))
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
            with autocast_context('cuda:0','bfloat16'):
                pred=m(batch)
                loss=loss_grounding(pred,batch)['total']
            loss.backward();torch.cuda.synchronize()
            elapsed=time.perf_counter()-began
            assert torch.isfinite(loss)
            gradients=[p.grad for p in m.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            history_gradient=float(batch['radar'].grad[:,:-1].abs().sum())
            assert history_gradient>0
            for key,p in m.named_parameters():
                torch.testing.assert_close(p.detach().cpu(),initial[key],rtol=0,atol=0)
            row=dict(case=case,sample_id=record['sample_id'],split=record['split'],agents=len(record['agents']),
                input_shape=list(batch['radar'].shape),candidates=3456,planner_tokens=expected_tokens,loss=float(loss.detach()),
                backbone_feature_shape=feature_shapes[-1],
                inference_median_s=sorted(times)[1],inference_samples_s=times,inference_peak_allocated_mib=inference_peak,
                forward_backward_s=elapsed,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                history_input_gradient_l1=history_gradient,finite_gradients=True,coordinates_match_cnn=True)
            rows.append(row)
            # Restore all BatchNorm buffers; no checkpoint or original state mutates.
            m.load_state_dict(initial);m.zero_grad(set_to_none=True)
            del pred,loss,batch,gradients
            print(json.dumps(dict(backbone=backbone,**row)),flush=True)
        for key,v in m.state_dict().items():
            torch.testing.assert_close(v.detach().cpu(),initial[key],rtol=0,atol=0)
        results[backbone]=dict(model_config=options,parameters=sum(p.numel() for p in m.parameters()),
            backbone_parameters=sum(p.numel() for p in m.encoder.parameters()),rows=rows,
            optimizer_steps=0,parameters_and_buffers_restored=True)
        shape_hook.remove()
        del m,initial;torch.cuda.empty_cache()
        write_json(args.output/'progress.json',results)
    done.set()
    assert source_fingerprint()==source_before and file_digest(__file__)==script_sha256
    write_json(args.output/'results.json',dict(backbones=results,manifest=str(args.manifest),
        source_files=source_before,script_sha256=script_sha256,config_sha256=file_digest(args.config),
        legacy_checkpoint_regression=legacy_check,
        manifest_sha256=acceptance['manifest_sha256'],test_evaluated=False,optimizer_steps=0,
        device=torch.cuda.get_device_name(),precision='bfloat16',warmups=1,repeats=3,batch_size=1,
        cpu_threads=1,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,note='Untrained resource/gradient checks, no accuracy claim'))


if __name__=='__main__':
    main()
