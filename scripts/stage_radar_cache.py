"""Create a byte-identical local mirror with one bounded read per radar file.

This does not create a PASS report or change a training plan. Run native dataset
validation on the completed local mirror before launching a fresh experiment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import time

RADAR_FIELDS = ('power', 'unfolded_doppler', 'doppler_valid')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(256*1024), b''):h.update(data)
    return h.hexdigest()


def atomic(path, value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


class Limit:
    def __init__(self, mib_per_second=4, opens_per_second=20, clock=time.monotonic, sleep=time.sleep, progress=None):
        if not all(math.isfinite(x) and x>0 for x in [mib_per_second,opens_per_second]):raise ValueError('Positive finite I/O budgets required')
        self.clock,self.sleep=clock,sleep
        self.started=clock();self.bytes=0;self.previous_open=None;self.next_read=self.started
        self.byte_rate=mib_per_second*1024**2;self.open_interval=1/opens_per_second
        self.progress=progress;self.last_report=-1e30;self.files=0;self.current_source=''

    def report(self,phase):
        if self.progress and self.clock()-self.last_report>=2:
            atomic(self.progress,dict(phase=phase,bytes=self.bytes,files=self.files,source=self.current_source,
                                     updated_unix_s=time.time()))
            self.last_report=self.clock()

    def opened(self):
        if self.previous_open is not None:
            self.sleep(max(0.,self.open_interval-(self.clock()-self.previous_open)))
        self.previous_open=self.clock()
        self.files+=1;self.report('OPEN')

    def consumed(self,n):
        self.bytes+=n
        # Idle time must not accumulate credits for a later NAS read burst.
        self.next_read=max(self.next_read,self.clock())+n/self.byte_rate
        self.sleep(max(0.,self.next_read-self.clock()))
        self.report('COPY')


def path_inside(root, relative):
    # Reject absolute references: byte-identical manifests must stay self-contained.
    relative=Path(relative)
    if relative.is_absolute():raise ValueError('Absolute radar reference requires an explicit new manifest version')
    path=Path(os.path.abspath(root/relative))
    if not path.is_relative_to(root):raise ValueError('Radar reference escapes source root')
    return path


def local_filesystem(path):
    """Do not let the cache accidentally live on the same network filesystem."""
    if os.name!='posix':return
    best=(0,'unknown')
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);mount=Path(left.split()[4].replace('\\040',' '));kind=right.split()[0]
        if path.is_relative_to(mount) and len(str(mount))>best[0]:best=(len(str(mount)),kind)
    if best[1] in {'nfs','nfs4','cifs','smb3','fuse.sshfs'}:raise ValueError('Cache destination is a network mount: '+best[1])


def copy_verified(source,target,expected,limit,source_open):
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(target)+'.partial');h=hashlib.sha256();offset=0;fresh=0
    if tmp.exists():
        with tmp.open('rb') as prefix:
            for block in iter(lambda:prefix.read(256*1024),b''):h.update(block);offset+=len(block)
    limit.current_source=str(source);limit.opened()
    with source_open(source) as inp,tmp.open('ab') as out:
        if offset:inp.seek(offset)
        for block in iter(lambda:inp.read(256*1024),b''):
            out.write(block);h.update(block);fresh+=len(block);limit.consumed(len(block))
    if h.hexdigest()!=expected:
        tmp.unlink();raise ValueError('Content differs from accepted digest: '+str(source))
    tmp.replace(target)
    return fresh


def _stage(source_root, manifest_relative, destination, expected_manifest, sensor_hashes,
          limit=None, min_free_gib=10, source_open=None, enforce_local=True):
    # Resolve the two roots once, never resolve/stat each repeated history reference.
    source_root=Path(source_root).resolve();destination=Path(destination).resolve()
    if destination==source_root or destination.is_relative_to(source_root):raise ValueError('Cache must be outside source tree')
    if enforce_local:local_filesystem(destination)
    source_manifest=path_inside(source_root,manifest_relative)
    local_root=destination/'root';local_manifest=local_root/source_manifest.relative_to(source_root)
    destination.mkdir(parents=True,exist_ok=True);local_manifest.parent.mkdir(parents=True,exist_ok=True)
    limit=limit or Limit();source_open=source_open or (lambda p:Path(p).open('rb'))
    identity=dict(schema='radar_vla_local_cache_v1',source_root=str(source_root),
                  manifest_relative=str(manifest_relative),manifest_sha256=expected_manifest,
                  sensor_inventory_sha256=hashlib.sha256(json.dumps(sensor_hashes,sort_keys=True).encode()).hexdigest())
    config=destination/'IDENTITY.json'
    if config.exists() and json.loads(config.read_text())!=identity:raise ValueError('Cache identity differs; use a new destination')
    atomic(config,identity)
    # This tool is for a stopped/pre-training cache, never an active dataset.
    (destination/'CACHE_READY.json').unlink(missing_ok=True)
    if local_manifest.exists():
        if digest(local_manifest)!=expected_manifest:raise ValueError('Cached manifest changed')
    else:
        copy_verified(source_manifest,local_manifest,expected_manifest,limit,source_open)
    # Parse the LOCAL copy once. Its original bytes and relative directory layout
    # are preserved, so no label or path rewriting is hidden in the cache.
    requests={};counts={};rows=0
    with local_manifest.open('rb') as stream:
        for line in stream:
            if not line.strip():continue
            record=json.loads(line);rows+=1;split=record['split'];counts[split]=counts.get(split,0)+1
            radar=record['sensors']['radar']
            for key in RADAR_FIELDS:
                values=radar.get(key,[])
                if isinstance(values,str):values=[values]
                for value in values:
                    absolute=Path(value)
                    if absolute.is_absolute():raise ValueError('Absolute radar paths are not safe in a byte-identical mirror')
                    absolute=Path(os.path.abspath(source_manifest.parent/absolute))
                    if not absolute.is_relative_to(source_root):raise ValueError('Radar file escapes source root')
                    expected=sensor_hashes.get(str(absolute))
                    if not expected:raise ValueError('No accepted digest for radar reference: '+str(absolute))
                    requests[str(absolute)]=expected
    atomic(destination/'PLAN.json',dict(**identity,rows=rows,splits=counts,unique_sensor_files=len(requests)))
    complete=0;read_files=0;bytes_read=0
    for original,expected in sorted(requests.items()):
        source=Path(original);target=local_root/source.relative_to(source_root)
        if target.is_symlink():raise ValueError('Cache target is a symlink')
        if target.exists() and digest(target)==expected:
            complete+=1;continue
        if shutil.disk_usage(destination).free<min_free_gib*1024**3:raise ValueError('Insufficient free local cache space')
        size=copy_verified(source,target,expected,limit,source_open)
        complete+=1;read_files+=1;bytes_read+=size
        if complete%100==0:
            atomic(destination/'PROGRESS.json',dict(state='copying',complete=complete,total=len(requests),
                   network_files_this_run=read_files,network_bytes_this_run=bytes_read))
    ready=dict(**identity,state='copied_and_hash_verified',manifest=str(local_manifest),rows=rows,
               splits=counts,unique_sensor_files=len(requests),network_files_this_run=read_files,
               network_sensor_bytes_this_run=bytes_read,
               validation='Not a PASS certificate; native validation on this local mirror is still required')
    # Interoperate with the deployed local-only RadarDataset transport. This
    # receipt preserves canonical source fingerprint keys without reopening NAS.
    assets={source:dict(path=str(local_root/Path(source).relative_to(source_root)),sha256=expected,
                       size=(local_root/Path(source).relative_to(source_root)).stat().st_size,
                       source_canonical=source) for source,expected in requests.items()}
    proof=dict(schema='radarvla_local_snapshot_v1',source_manifest=str(source_manifest),
               source_root=str(source_manifest.parent),source_stamp=None,manifest_sha256=expected_manifest,
               assets=assets,immutable_snapshot=True,
               import_method='rate-limited mirror verified against accepted training protocol')
    atomic(local_manifest.parent/'storage_receipt.json',proof)
    atomic(destination/'CACHE_READY.json',ready)
    return ready


def stage(source_root, manifest_relative, destination, expected_manifest, sensor_hashes,
          limit=None, min_free_gib=10, source_open=None, enforce_local=True):
    source_root=Path(source_root).resolve();destination=Path(destination).resolve()
    if destination==source_root or destination.is_relative_to(source_root):raise ValueError('Cache must be outside source tree')
    if enforce_local:local_filesystem(destination)
    destination.mkdir(parents=True,exist_ok=True)
    lock=destination/'.stage.lock'
    # A stale lock needs explicit PID inspection; do not launch duplicate readers.
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(descriptor,str(os.getpid()).encode())
        return _stage(source_root,manifest_relative,destination,expected_manifest,sensor_hashes,
                      limit,min_free_gib,source_open,enforce_local)
    finally:
        os.close(descriptor);lock.unlink()


def stage_assets(source_root,destination,hashes,limit,min_free_gib=10,enforce_local=True,source_open=None):
    source_root=Path(source_root).resolve();destination=Path(destination).resolve()
    if destination==source_root or destination.is_relative_to(source_root):raise ValueError('Assets must be cached outside source tree')
    if enforce_local:local_filesystem(destination)
    destination.mkdir(parents=True,exist_ok=True);root=destination/'files';root.mkdir(exist_ok=True)
    lock=destination/'.stage.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        os.write(fd,str(os.getpid()).encode());(destination/'ASSETS_READY.json').unlink(missing_ok=True)
        identity=dict(source=str(source_root),sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
        old=destination/'IDENTITY.json'
        if old.exists() and json.loads(old.read_text())!=identity:raise ValueError('Asset identity differs')
        atomic(old,identity);network=0
        for name,expected in sorted(hashes.items()):
            source=path_inside(source_root,name);target=root/source.relative_to(source_root)
            if target.is_symlink():raise ValueError('Asset target is a symlink')
            if target.exists() and digest(target)==expected:continue
            if shutil.disk_usage(destination).free<min_free_gib*1024**3:raise ValueError('Insufficient free local space')
            network+=copy_verified(source,target,expected,limit,source_open or (lambda p:p.open('rb')))
            atomic(destination/'PROGRESS.json',dict(state='copying',file=name,network_bytes_this_run=network))
        ready=dict(state='copied_and_hash_verified',root=str(root),identity=identity,files=len(hashes),network_bytes_this_run=network)
        atomic(destination/'ASSETS_READY.json',ready);return ready
    finally:os.close(fd);lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--manifest-relative',required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--accepted-protocol',type=Path,required=True,help='Local protocol.json from an accepted run; must contain data.manifest_sha256 and data.sensor_files')
    p.add_argument('--mib-per-second',type=float,default=4);p.add_argument('--opens-per-second',type=float,default=20)
    p.add_argument('--min-free-gib',type=float,default=10)
    p.add_argument('--progress',type=Path)
    p.add_argument('--qwen-protocol',type=Path);p.add_argument('--qwen-root',type=Path);p.add_argument('--qwen-destination',type=Path)
    a=p.parse_args();data=json.loads(a.accepted_protocol.read_text())['data']
    if any([a.qwen_protocol,a.qwen_root,a.qwen_destination]) and not all([a.qwen_protocol,a.qwen_root,a.qwen_destination]):
        p.error('Qwen staging requires all three Qwen arguments')
    limit=Limit(a.mib_per_second,a.opens_per_second,progress=a.progress)
    print(json.dumps(stage(a.source_root,a.manifest_relative,a.destination,data['manifest_sha256'],data['sensor_files'],
                          limit,a.min_free_gib),indent=2),flush=True)
    if a.qwen_protocol:
        hashes=json.loads(a.qwen_protocol.read_text())['language_model_files']
        print(json.dumps(stage_assets(a.qwen_root,a.qwen_destination,hashes,limit,a.min_free_gib),indent=2),flush=True)


if __name__=='__main__':main()
