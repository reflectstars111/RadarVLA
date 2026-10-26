"""Package completed evaluation artifacts; never touches NAS or checkpoints."""
from pathlib import Path
import hashlib
import json
import tarfile
import time

R=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


while True:
    if (R/'DELIVERY_STATUS.json').exists() and json.loads((R/'DELIVERY_STATUS.json').read_text())['state']=='complete':break
    time.sleep(15)
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
assert json.loads((R/'FINALIZER_STATUS.json').read_text())['state']=='complete'
files=sorted(p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.json','.jsonl','.md','.py','.log','.txt'] and p.name!='ARCHIVE_READY.json')
inventory={str(p.relative_to(R)):dict(size=p.stat().st_size,sha256=sha(p)) for p in files}
manifest=R/'ARCHIVE_MANIFEST.json';manifest.write_text(json.dumps(inventory,indent=2)+'\n');files.append(manifest)
target=R/'full_planning_eval_20261026.tar.gz';partial=R/'full_planning_eval_20261026.tar.gz.partial'
with tarfile.open(partial,'w:gz',compresslevel=3) as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(R)),recursive=False)
partial.replace(target)
(R/'ARCHIVE_READY.json').write_text(json.dumps(dict(state='complete',archive=str(target),size=target.stat().st_size,sha256=sha(target),files=len(files)),indent=2)+'\n')
