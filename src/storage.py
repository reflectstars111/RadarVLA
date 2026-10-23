"""Bounded shared-storage transport; hot dataset reads use verified local snapshots.

Published prepared datasets are immutable snapshots. --refresh imports a changed
asset set explicitly. No pickle, shell evaluation, or archive path traversal.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

_LOCAL = {}
_FILES = {}


def absolute(path):
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def is_remote(path):
    path = str(absolute(path))
    prefixes = ['/storage/data/', '/home/luxinlan/data/', '/home/metaiot_guest/data/']
    prefixes += [p for p in os.environ.get('RADARVLA_REMOTE_PREFIXES','').split(os.pathsep) if p]
    if any(path.startswith(p.rstrip('/')+'/') or path == p.rstrip('/') for p in prefixes):
        return True
    try:
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left,right=line.split(' - ',1)
            mount=left.split()[4].replace('\\040',' ')
            if right.split()[0] in ('nfs','nfs4') and (path==mount or path.startswith(mount.rstrip('/')+'/')):
                return True
    except FileNotFoundError:
        pass
    return False


def cache_root():
    default='/storage/ssd/metaiot_guest/luxinlan_radarvla_cache'
    root=absolute(os.environ.get('RADARVLA_CACHE_ROOT',default))
    if is_remote(root):
        raise ValueError('Dataset cache must reside on a local filesystem')
    root.mkdir(parents=True,exist_ok=True)
    return root


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value,sort_keys=True)+'\n');os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def state_counts():
    try:
        text=Path('/proc/self/mountstats').read_text()
    except FileNotFoundError:
        return {}
    values={}
    for block in text.split('\ndevice '):
        if ' with fstype nfs' not in block:continue
        name=block.splitlines()[0]
        for line in block.splitlines():
            parts=line.split()
            if parts and parts[0]=='TEST_STATEID:': values[name]=int(parts[1])
    return values


class Transport:
    def __init__(self,root):
        self.root=Path(root)
        self.rate=float(os.environ.get('RADARVLA_NAS_MIB_S','16'))*1024**2
        self.opens=float(os.environ.get('RADARVLA_NAS_OPENS_S','64'))
        if self.rate<=0 or self.opens<=0:raise ValueError('Positive NAS I/O limits required')
        self.last_open=0.;self.started=time.monotonic();self.bytes=0;self.files=0
        self.local_bytes=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        self.health=state_counts();self.health_time=self.started;self.bad_since=None

    def progress(self,phase,**extra):
        atomic(self.root/'transfer_progress.json',dict(pid=os.getpid(),phase=phase,
            updated_unix_s=time.time(),bytes=self.bytes,files=self.files,**extra))

    def health_check(self):
        now=time.monotonic()
        if now-self.health_time<2:return
        counts=state_counts()
        high=any((n-self.health.get(k,n))/(now-self.health_time)>1000 for k,n in counts.items())
        self.bad_since=(self.bad_since or now) if high else None
        self.health,self.health_time=counts,now
        if self.bad_since and now-self.bad_since>=10:
            raise RuntimeError('NFS state recovery storm; transport paused, no automatic retry')

    def check_space(self,extra=0):
        reserve=float(os.environ.get('RADARVLA_CACHE_RESERVE_GIB','150'))*1024**3
        if shutil.disk_usage(self.root).free-extra<reserve:
            raise OSError('Local cache reserve would be violated; existing data preserved')
        maximum=float(os.environ.get('RADARVLA_CACHE_MAX_GIB','40'))*1024**3
        if self.local_bytes+extra>maximum:
            raise OSError('Local cache budget exceeded; no unverified data evicted')

    def copy(self,source,target):
        source,target=Path(source),Path(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        self.health_check()
        delay=1/self.opens-(time.monotonic()-self.last_open)
        if delay>0:time.sleep(delay)
        self.progress('OPEN',source=str(source))
        before=source.stat()
        if not is_remote(target):self.check_space(before.st_size)
        self.last_open=time.monotonic()
        h=hashlib.sha256();started=time.monotonic();copied=0
        tmp=target.with_name('.'+target.name+'.partial')
        try:
            with source.open('rb') as inp,tmp.open('wb') as out:
                while True:
                    block=inp.read(1024*1024)
                    if not block:break
                    out.write(block);h.update(block);copied+=len(block);self.bytes+=len(block)
                    delay=copied/self.rate-(time.monotonic()-started)
                    if delay>0:time.sleep(delay)
                    self.health_check();self.progress('COPY',source=str(source))
                out.flush();os.fsync(out.fileno())
            after=source.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Source changed during transfer: '+str(source))
            if is_remote(target):
                # One sequential readback of a bundle/checkpoint, never per-member
                # verification against NAS. Publish only after checksum equality.
                verify=hashlib.sha256();read=0;check_started=time.monotonic()
                self.progress('VERIFY_UPLOAD',source=str(target))
                with tmp.open('rb') as inp:
                    for block in iter(lambda:inp.read(1024*1024),b''):
                        verify.update(block);read+=len(block)
                        delay=read/self.rate-(time.monotonic()-check_started)
                        if delay>0:time.sleep(delay)
                        self.health_check();self.progress('VERIFY_UPLOAD',source=str(target))
                if verify.hexdigest()!=h.hexdigest():raise ValueError('NAS readback checksum mismatch')
            os.replace(tmp,target);self.files+=1
            if not is_remote(target):self.local_bytes+=copied
            return dict(sha256=h.hexdigest(),size=before.st_size,
                        source_mtime_ns=before.st_mtime_ns,path=str(target))
        finally:
            tmp.unlink(missing_ok=True)


def sensor_sources(manifest,source_root):
    """Deduplicate lexically before stat/resolve/open; no sensor I/O here."""
    found={}
    with Path(manifest).open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);radar=row['sensors']['radar']
            for key in ('power','unfolded_doppler','doppler_valid'):
                if key not in radar:continue
                items=radar[key] if isinstance(radar[key],list) else [radar[key]]
                for item in items:
                    if not isinstance(item,str):raise ValueError('Sensor paths must be strings')
                    path=absolute(Path(source_root)/item)
                    found[str(path)]=None
    return sorted(found)


def _stage(source,refresh=False):
    root=cache_root();key=hashlib.sha256(str(absolute(source)).encode()).hexdigest()[:24]
    target=root/'staging'/key;target.mkdir(parents=True,exist_ok=True)
    transfer=Transport(root)
    # One transport across all main/worktree processes of this user.
    with (root/'transport.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        transfer.progress('CHECK_MANIFEST',source=str(source))
        source=Path(source).resolve();stat=source.stat()
        location=root/'locations'/(key+'.json')
        old_manifest=Path(json.loads(location.read_text())['manifest']) if location.exists() else None
        old=receipt(old_manifest) if old_manifest else None
        if old and not refresh and old['source_stamp']==[stat.st_size,stat.st_mtime_ns]:
            # Assets are immutable published inputs. Verify local existence/size;
            # explicit refresh reimports edited source assets, never silent reuse.
            valid=sha(old_manifest)==old['manifest_sha256']
            for index,v in enumerate(old['assets'].values()):
                p=Path(v['path'])
                valid=valid and p.is_file() and p.stat().st_size==v['size'] and sha(p)==v['sha256']
                if index%128==0:transfer.progress('VERIFY_LOCAL_CACHE',dataset=str(old_manifest.parent))
            if valid:
                transfer.progress('CACHE_HIT',dataset=str(old_manifest.parent));return old_manifest
        entry=transfer.copy(source,target/'manifest.jsonl')
        sources=sensor_sources(target/'manifest.jsonl',source.parent)
        transfer.progress('OPEN',source=str(source.parent/'storage_index.json'))
        if (source.parent/'storage_index.json').is_file():
            transfer.copy(source.parent/'storage_index.json',target/'storage_index.json')
            index=json.loads((target/'storage_index.json').read_text())
            if index.get('manifest_sha256')!=entry['sha256']:
                raise ValueError('Archive index does not certify this manifest')
            assets=_archive_assets(index,sources,source.parent,target,root,transfer)
            return _publish_snapshot(root,key,target,source,stat,entry,assets,transfer)
        assets={}
        # Resume an interrupted transfer without treating its journal as PASS.
        journal=target/'stage_progress.json'
        previous=json.loads(journal.read_text()) if journal.exists() and not refresh else {}
        if previous.get('manifest_sha256')!=entry['sha256']:previous={}
        for item in sources:
            name=hashlib.sha256(item.encode()).hexdigest()
            dest=target/(name+'.npy')
            cached=previous.get('assets',{}).get(item)
            if cached and Path(cached['path']).is_file() and sha(cached['path'])==cached['sha256']:
                assets[item]=cached
            else:
                info=transfer.copy(item,dest)
                digest=info['sha256'];obj=root/'objects'/digest[:2]/(digest+'.npy')
                obj.parent.mkdir(parents=True,exist_ok=True)
                if not obj.exists() or sha(obj)!=digest:os.replace(dest,obj)
                else:dest.unlink()
                info['path']=str(obj)
                info['source_canonical']=str(Path(item).resolve())
                assets[item]=info
            if len(assets)%128==0:
                atomic(journal,dict(manifest_sha256=entry['sha256'],assets=assets))
        return _publish_snapshot(root,key,target,source,stat,entry,assets,transfer)


def _publish_snapshot(root,key,target,source,stat,entry,assets,transfer):
        proof=dict(schema='radarvla_local_snapshot_v1',source_manifest=str(source),
            source_root=str(source.parent),source_stamp=[stat.st_size,stat.st_mtime_ns],
            manifest_sha256=entry['sha256'],assets=assets,immutable_snapshot=True,
            imported_unix_s=time.time())
        snapshot=hashlib.sha256(json.dumps([entry['sha256'],
            {k:v['sha256'] for k,v in assets.items()}],sort_keys=True).encode()).hexdigest()
        final=root/'datasets'/snapshot
        final.mkdir(parents=True,exist_ok=True)
        if not (final/'storage_receipt.json').exists():
            shutil.copyfile(target/'manifest.jsonl',final/'manifest.jsonl')
            atomic(final/'storage_receipt.json',proof)
        atomic(root/'locations'/(key+'.json'),{'manifest':str(final/'manifest.jsonl')})
        (target/'stage_progress.json').unlink(missing_ok=True)
        transfer.progress('READY',dataset=str(final),unique_assets=len(assets))
        return final/'manifest.jsonl'


def _archive_assets(index,sources,source_root,target,root,transfer):
    if index.get('schema')!='radarvla_prepared_archives_v1':
        raise ValueError('Unknown archive storage schema')
    groups={}
    for source in sources:
        relative=str(Path(source).relative_to(source_root))
        archive=index['assets'][relative]
        groups.setdefault(archive,[]).append((source,relative))
    assets={}
    for key,items in groups.items():
        info=index['archives'][key]
        bundle=target/('bundle_'+hashlib.sha256(key.encode()).hexdigest()+'.tar')
        fetched=transfer.copy(source_root/info['path'],bundle)
        if fetched['sha256']!=info['sha256']:raise ValueError('Prepared archive checksum mismatch')
        with tarfile.open(bundle,'r:*') as archive:
            wanted={relative for _,relative in items}
            members={member.name:member for member in archive.getmembers() if member.name in wanted}
            if set(members)!=wanted:raise ValueError('Prepared archive missing sensor members')
            transfer.check_space(sum(m.size for m in members.values()))
            for source,relative in items:
                member=members[relative]
                if not member.isfile() or Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise ValueError('Unsafe prepared archive member')
                temporary=target/'unpack_sensor.partial';digest=hashlib.sha256()
                with archive.extractfile(member) as inp,temporary.open('wb') as out:
                    for block in iter(lambda:inp.read(1024*1024),b''):
                        digest.update(block);out.write(block)
                hexdigest=digest.hexdigest();obj=root/'objects'/hexdigest[:2]/(hexdigest+'.npy')
                obj.parent.mkdir(parents=True,exist_ok=True)
                if obj.exists() and sha(obj)==hexdigest:temporary.unlink()
                else:
                    temporary.replace(obj)
                    transfer.local_bytes+=member.size
                assets[source]=dict(path=str(obj),sha256=hexdigest,size=member.size,
                                    source_canonical=source,source_archive=key)
                transfer.progress('UNPACK_LOCAL',archive=key)
        bundle.unlink()
        transfer.local_bytes-=fetched['size']
    return assets


def _run_transport(arguments):
    root=cache_root()
    # No repeated creation of workers if an earlier hard-NFS syscall survives.
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        blocked=root/'blocked_worker.json'
        if blocked.exists():
            old=json.loads(blocked.read_text());proc=Path('/proc')/str(old['pid'])
            if proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':
                raise RuntimeError('Prior NAS worker remains blocked; no replacement spawned')
            blocked.unlink()
        progress=root/'transfer_progress.json'
        progress.unlink(missing_ok=True)
        logpath=root/'stage.log'
        with logpath.open('a') as log:
            command=[sys.executable,str(Path(__file__)),'--worker']+list(map(str,arguments))
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=last=time.monotonic();stamp=None;shown=0
            while process.poll() is None:
                time.sleep(.2)
                if progress.exists():
                    row=json.loads(progress.read_text());current=row['updated_unix_s']
                    if current!=stamp:last=time.monotonic();stamp=current
                    if time.monotonic()-shown>5:
                        print('Storage: %s | %d unique files | %.1f MiB copied'%(
                            row['phase'],row['files'],row['bytes']/1024**2),file=sys.stderr,flush=True)
                        shown=time.monotonic()
                limit=5 if progress.exists() and json.loads(progress.read_text())['phase'] in ('OPEN','CHECK_MANIFEST') else 30
                if time.monotonic()-last>limit:
                    os.killpg(process.pid,signal.SIGKILL)
                    atomic(blocked,dict(pid=process.pid,arguments=list(map(str,arguments)),time=time.time()))
                    raise TimeoutError('NAS transport stalled; see '+str(logpath))
            if process.returncode:raise RuntimeError('Dataset staging failed; see '+str(logpath))


def local_manifest(source,refresh=False,force=False):
    source=absolute(source)
    if not force and not is_remote(source):return source
    key=str(source)
    if key in _LOCAL and not refresh:return _LOCAL[key]
    _run_transport([source]+(['--refresh'] if refresh else []))
    root=cache_root();cache_key=hashlib.sha256(key.encode()).hexdigest()[:24]
    result=Path(json.loads((root/'locations'/(cache_key+'.json')).read_text())['manifest'])
    if not (result.parent/'storage_receipt.json').exists():raise RuntimeError('No complete storage receipt')
    _LOCAL[key]=result
    return result


def copy_file(source,target):
    """Bounded, globally serialized transport for manifests and bundle files."""
    _run_transport([absolute(source),'--copy-to',absolute(target)])
    _FILES.pop(str(absolute(target)),None)
    return absolute(target)


def ensure_local_capacity(size):
    Transport(cache_root()).check_space(int(size))


def local_file(source):
    source=absolute(source)
    if not is_remote(source):return source
    key=str(source)
    if key not in _FILES:
        target=cache_root()/'inputs'/hashlib.sha256(key.encode()).hexdigest()/source.name
        _FILES[key]=copy_file(source,target)
    return _FILES[key]


def compatible_protocol(old,new,proof_path=None):
    """Allow only the deployed I/O source change, never model/data/config drift."""
    if old==new:return True
    if {k:v for k,v in old.items() if k!='source_files'}!={k:v for k,v in new.items() if k!='source_files'}:
        return False
    path=Path(proof_path) if proof_path else Path(__file__).resolve().parents[1]/'.io_repair_receipt.json'
    if not path.is_file():return False
    proof=json.loads(path.read_text())
    before,after=proof.get('before',{}),proof.get('after',{})
    allowed={'storage.py','pipeline.py','data.py','__main__.py'}
    lhs,rhs=old.get('source_files',{}),new.get('source_files',{})
    changes={k for k in set(lhs)|set(rhs) if lhs.get(k)!=rhs.get(k)}
    return bool(changes) and changes<=allowed and any(all(
        lhs.get(k)==origin.get(k) and rhs.get(k)==after.get(k) for k in changes)
        for origin in [before]+proof.get('origins',[]))


def receipt(manifest):
    path=Path(manifest).parent/'storage_receipt.json'
    return json.loads(path.read_text()) if path.exists() else None


def asset_path(manifest,value,proof=None):
    proof=proof if proof is not None else receipt(manifest)
    if proof is None:
        path=Path(manifest).parent/value
        if is_remote(path):
            raise RuntimeError('Remote sensor in an unstaged local manifest; use radar_vla.storage to stage it')
        return path
    source=str(absolute(Path(proof['source_root'])/value))
    if source not in proof['assets']:raise ValueError('Asset absent from verified snapshot: '+source)
    return Path(proof['assets'][source]['path'])


def snapshot_fingerprint(manifest):
    proof=receipt(manifest)
    if proof is None:return None
    return dict(manifest_sha256=proof['manifest_sha256'],
                sensor_files={v.get('source_canonical',k):v['sha256'] for k,v in proof['assets'].items()})


class ArrayCache:
    """Bounded process-local cache for overlapping causal histories."""
    def __init__(self,limit=256*1024**2):self.limit=limit;self.bytes=0;self.items=OrderedDict()
    def load(self,path):
        import numpy as np
        key=str(path)
        if key in self.items:
            result=self.items.pop(key);self.items[key]=result;return result
        result=np.load(path,allow_pickle=False)
        if result.nbytes<=self.limit:
            while self.items and self.bytes+result.nbytes>self.limit:
                _,old=self.items.popitem(last=False);self.bytes-=old.nbytes
            result.setflags(write=False);self.items[key]=result;self.bytes+=result.nbytes
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--copy-to',type=Path)
    args=parser.parse_args()
    if args.copy_to:
        if args.worker:
            root=cache_root()
            with (root/'transport.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                transport=Transport(root);info=transport.copy(args.manifest,args.copy_to)
                transport.progress('COPIED',sha256=info['sha256'])
        else:copy_file(args.manifest,args.copy_to)
    elif args.worker:print(_stage(args.manifest,args.refresh),flush=True)
    else:print(local_manifest(args.manifest,args.refresh,force=True),flush=True)
