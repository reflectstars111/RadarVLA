from pathlib import Path
import json,hashlib,torch,gc,datetime
from radar_vla.task_supervision import stage_available
r=Path('/storage/ssd/metaiot_guest/luxinlan_radarvla_cache/evaluations/full_planning_20261026');training=json.loads(Path('/home/luxinlan/PublicWorkspace/Xinlan_Lu/.runs/RadarVLA-training-20261024/plan.json').read_text());output=Path(training['output_root'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
counts={'train':0,'val':0,'test':0};rawcounts=dict(counts)
with Path(training['manifest']).open() as f:
 for line in f:
  record=json.loads(line);rawcounts[record['split']]+=1
  if stage_available(record,'sft'):counts[record['split']]+=1
arms={'baseline':'baseline/sft','action':'action/sft','assignment':'assignment/sft','risk':'risk/sft','combined':'combined/sft','baseline_readout':'baseline_readout_sft'}
selected=[]
for arm,rel in arms.items():
 cp=output/rel/'best_planning.pt';state=torch.load(cp,map_location='cpu',weights_only=True)
 assert state['protocol']['stage']=='sft' and state['epoch']==json.loads((output/rel/'status.json').read_text())['best_quality_epoch']['planning']
 selected.append(dict(arm=arm,checkpoint=str(cp),checkpoint_sha256=sha(cp),epoch=state['epoch']))
 del state;gc.collect()
jobs=[dict(x,name=x['arm']+'_'+split,split=split,output=str(r/'results'/x['arm']/split)) for split in ['val','test'] for x in selected]
plan=dict(created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),training=training,evaluation_script_sha256=sha(r/'evaluate_campaign.py'),output_root=str(r),expected_splits=counts,raw_split_counts=rawcounts,selection='Freeze native best_planning checkpoints picked by training validation only; no reselection using full val or test',batch_size=1,risk_source='predicted',risk_projection='checkpoint default',max_concurrent_workers=2,workers=0,cpu_threads=1,budget_gib=18,minimum_free_vram_gib=8,offline_only=True,jobs=jobs)
(r/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(expected=counts,selected=selected),indent=2))
