"""Reproduce mode-risk intervention on the historical fixed validation cohort."""
import argparse
import json
from pathlib import Path
import torch
from radar_vla.data import collate_batch
from radar_vla.pipeline import load_models,to_device,autocast_context,file_digest
from radar_vla.visual_diagnosis import _single_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cohort',type=Path,default=Path('analysis/rebalanced_validation/mode_risk_epoch_003.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    old=json.loads(a.cohort.read_text());wanted={r['sample_id'] for r in old};records={}
    for line in a.manifest.open():
        r=json.loads(line)
        if r['sample_id'] in wanted:
            if r['split']!='val':raise ValueError('Intervention requires validation samples')
            records[r['sample_id']]=r
    if records.keys()!=wanted:raise ValueError('Missing cohort IDs')
    grounder,planner,state=load_models(a.checkpoint,a.device);rows=[]
    for i,prior in enumerate(old,1):
        r=records[prior['sample_id']];batch=to_device(collate_batch([_single_sample(r)]),a.device)
        with torch.no_grad(),autocast_context(a.device,state['protocol'].get('precision','float32')):
            pred=grounder(batch);known=batch['risk_mask'].bool() & torch.isfinite(batch['risk_target'])
            risk=torch.where(known,batch['risk_target'],pred['risk'])
            target=torch.tensor([[planner.tokenizer.token('<SHORT>')]],device=a.device)
            modes={}
            for name,condition in [('predicted',pred['risk']),('oracle',risk)]:
                logits=planner(pred['radar_tokens'],condition,batch['instruction'],target)[0,0]
                probs=logits[[planner.tokenizer.token('<SHORT>'),planner.tokenizer.token('<LONG>')]].float().softmax(0)
                modes[name]=dict(p_short_among_modes=float(probs[0]),mode='short' if probs[0]>=probs[1] else 'long')
        rows.append(dict(sample_id=r['sample_id'],episode_id=r['episode_id'],teacher_mode=r['teacher_mode'],
            risk_pred=pred['risk'][0].float().cpu().tolist(),risk_truth=batch['risk_target'][0].cpu().tolist(),
            risk_mask=known[0].cpu().tolist(),mode=modes))
        if i%10==0:print(f'mode risk {i}/{len(old)}',flush=True)
    a.output.write_text(json.dumps(dict(rows=rows,checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=file_digest(a.checkpoint),
        manifest_sha256=file_digest(a.manifest),cohort_sha256=file_digest(a.cohort),script_sha256=file_digest(Path(__file__)),
        note='oracle substitutes known risk dimensions only; unknown dimensions retain predicted values. First generated mode token is scored with BOS, no teacher prefix.'),indent=2)+'\n')

if __name__=='__main__':main()
