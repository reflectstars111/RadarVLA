"""Restore missing prepared files byte-for-byte from verified copies/saved products.

No radar processing is invoked. Every file must match its historical training
SHA256 before being published; existing files are verified and never replaced.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from radar_vla.agent50_migration import sha256,write_json
from radar_vla.collection_adapter import _product_frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();proof=json.loads(a.preflight.read_text())
    if sha256(a.protocol)!=proof['protocol_sha256']:raise ValueError('Protocol differs from verified preflight')
    expected=json.loads(a.protocol.read_text())['data']['sensor_files'];original=Path(proof['source_directory']);alt=Path(proof['alternative_directory'])
    if original.exists():
        # Re-runs only fill missing files; existing content must match below.
        if not original.is_dir():raise ValueError('Restore target is not a directory')
    original.mkdir(parents=True,exist_ok=True)
    issues={r['source'] for r in proof['issues']};methods={};raw_lookup={};records_by_mask={};jobs={}
    for line in a.manifest.open():
        r=json.loads(line);radar=r['sensors']['radar'];episode=Path(r['route_archive']).parent
        jobs[r['scene_id']]=episode.parent
        records_by_mask[radar['doppler_valid']]=r
    def link_verified(item):
        name,wanted=item;target=Path(name)
        if not target.is_relative_to(original):raise ValueError('Unexpected fingerprint path')
        if target.exists():
            if sha256(target)!=wanted:raise ValueError(f'Existing file differs; refusing overwrite: {target}')
            return name,'already_verified'
        if name in issues:return name,None
        source=alt/target.relative_to(original)
        if sha256(source)!=wanted:raise ValueError(f'Copy source changed: {source}')
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,target)
        return name,'verified_existing_copy'
    with ThreadPoolExecutor(max_workers=8) as pool:
        for name,method in pool.map(link_verified,expected.items()):
            if method:methods[name]=method
    missing=[name for name in expected if name not in methods]
    needed_jobs=set()
    for name in missing:
        rel=Path(name).relative_to(original)
        needed_jobs.add(rel.parts[1] if rel.parts[0]=='radar' else records_by_mask[name]['scene_id'])
    def product_index(job):
        return job,_product_frames(jobs[job])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for job,index in pool.map(product_index,sorted(needed_jobs)):
            raw_lookup[job]=index;print('indexed saved radar',job,flush=True)
    def array(name):
        rel=Path(name).relative_to(original)
        if rel.parts[0]=='radar':
            job=rel.parts[1];index=int(rel.name.split('_')[0]);folder=raw_lookup[job][index][1]
            if rel.name.endswith('_power.npy'):
                with np.load(folder/'folded_ra.npz') as f:return np.asarray(f['power_peak_ra'][:,::-1],np.float32)
            if rel.name.endswith('_doppler.npy'):
                with np.load(folder/'unfolded_ra.npz') as f:return np.asarray(f['doppler_unfolded_dense_ra_mps'][:,::-1],np.float32)
            raise ValueError('Unexpected radar filename')
        r=records_by_mask[name];planes=[]
        for power in r['sensors']['radar']['power']:
            index=int(Path(power).name.split('_')[0]);folder=raw_lookup[r['scene_id']][index][1]
            with np.load(folder/'unfolded_ra.npz') as f:planes.append(np.asarray(f['reliable_mask'][:,::-1],bool))
        return np.stack(planes)
    def restore_array(name):
        target=Path(name);target.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.verified_restore_',suffix='.npy',dir=target.parent);os.close(fd)
        try:
            np.save(tmp,array(name),allow_pickle=False)
            if sha256(tmp)!=expected[name]:raise ValueError(f'Saved-product recovery does not match original SHA256: {name}')
            if target.exists():
                if sha256(target)!=expected[name]:raise ValueError(f'Existing file differs: {name}')
            else:os.link(tmp,target)
        finally:Path(tmp).unlink(missing_ok=True)
        return name,'verified_saved_product'
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(name,method) in enumerate(pool.map(restore_array,missing),1):
            methods[name]=method
            if i%500==0:print('restored saved files',i,'/',len(missing),flush=True)
    # Verify all final paths independently after restoration.
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual=dict(pool.map(lambda name:(name,sha256(name)),expected))
    bad=[name for name in expected if actual[name]!=expected[name]]
    if bad:raise ValueError(f'Final restored files differ: {bad[:3]}')
    write_json(a.report,dict(status='PASS',files=len(expected),all_sha256_match=True,methods=dict(Counter(methods.values())),
        protocol_sha256=sha256(a.protocol),manifest_sha256=sha256(a.manifest),script_sha256=sha256(Path(__file__)),
        original_directory=str(original),raw_jobs=sorted(needed_jobs),
        files_sha256=actual,note='Missing prepared files restored to original byte hashes. No radar computation, no existing-file overwrite, no source array content modified.'))
    print('RESTORATION PASS',len(actual),dict(Counter(methods.values())),flush=True)

if __name__=='__main__':main()
