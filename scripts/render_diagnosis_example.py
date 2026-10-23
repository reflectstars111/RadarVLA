"""Save one real turning-frame overlay using the corrected radar transform."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_vla.visual_diagnosis import DiagnosisSession
from radar_vla.pipeline import file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    records=[json.loads(l) for l in a.manifest.open()];records=[r for r in records if r['split']=='val']
    r=max(records,key=lambda r:abs(r['ego']['yaw_rate']))
    grounding=Path('analysis/retrained_grounding_v2/best.pt');sft=Path('analysis/retrained_sft_v2/best.pt')
    session=DiagnosisSession(a.manifest,grounding,sft,device=a.device,split='val')
    fig=session.diagnose(r['sample_id'],.9)
    fig.savefig(a.output_dir/'turning_overlay.png',dpi=120)
    plt.close(fig)
    (a.output_dir/'turning_overlay.json').write_text(json.dumps(dict(sample_id=r['sample_id'],episode_id=r['episode_id'],
        yaw_rate_radps=r['ego']['yaw_rate'],radar_pose=r['sensors']['radar']['poses_current_ego'][-1],
        threshold=.9,manifest_sha256=file_digest(a.manifest),grounding_sha256=file_digest(grounding),sft_sha256=file_digest(sft),
        script_sha256=file_digest(Path(__file__))),indent=2)+'\n')
    print('saved turning overlay',r['sample_id'])

if __name__=='__main__':main()
