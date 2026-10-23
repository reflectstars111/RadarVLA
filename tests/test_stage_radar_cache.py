import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

module_path=Path(__file__).with_name('stage_radar_cache.py')
if not module_path.exists():module_path=Path(__file__).resolve().parents[1]/'scripts/stage_radar_cache.py'
spec=importlib.util.spec_from_file_location('cache',module_path)
cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)


def fixture(tmp_path):
    root=tmp_path/'nas';root.mkdir();(root/'prepared').mkdir()
    sensor=root/'radar.bin';sensor.write_bytes(b'unchanged-radar-bytes')
    rows=[dict(sample_id=str(i),split=s,labels=[.5,1,2],sensors=dict(radar=dict(
        power=['../radar.bin','../radar.bin'],unfolded_doppler='../radar.bin',doppler_valid='../radar.bin')))
        for i,s in enumerate(['train','val','test'])]
    raw=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode()
    manifest=root/'prepared/manifest.jsonl';manifest.write_bytes(raw)
    return root,manifest,sensor,raw,{str(sensor):hashlib.sha256(sensor.read_bytes()).hexdigest()}


class Unlimited:
    def opened(self):pass
    def consumed(self,n):pass


def test_unique_copy_exact_manifest_and_zero_network_on_resume(tmp_path):
    root,manifest,sensor,raw,hashes=fixture(tmp_path);opens=[]
    def read(p):opens.append(p);return p.open('rb')
    kwargs=dict(source_root=root,manifest_relative='prepared/manifest.jsonl',destination=tmp_path/'ssd',
                expected_manifest=hashlib.sha256(raw).hexdigest(),sensor_hashes=hashes,
                source_open=read,limit=Unlimited(),min_free_gib=0,enforce_local=False)
    result=cache.stage(**kwargs)
    assert opens==[manifest,sensor]
    assert Path(result['manifest']).read_bytes()==raw
    assert result['unique_sensor_files']==1 and result['rows']==3
    proof=json.loads((Path(result['manifest']).parent/'storage_receipt.json').read_text())
    assert proof['assets'][str(sensor)]['source_canonical']==str(sensor)
    assert proof['assets'][str(sensor)]['sha256']==hashes[str(sensor)]
    assert (tmp_path/'ssd/root/radar.bin').read_bytes()==sensor.read_bytes()
    opens.clear();second=cache.stage(**kwargs)
    assert opens==[] and second['network_files_this_run']==0
    assert 'Not a PASS' in result['validation']


def test_changed_sensor_never_publishes_ready_certificate(tmp_path):
    root,manifest,sensor,raw,hashes=fixture(tmp_path);sensor.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='differs from accepted'):
        cache.stage(root,'prepared/manifest.jsonl',tmp_path/'ssd',hashlib.sha256(raw).hexdigest(),hashes,
                    limit=Unlimited(),min_free_gib=0,enforce_local=False)
    assert not (tmp_path/'ssd/CACHE_READY.json').exists()


def test_absolute_reference_rejected_instead_of_silently_reading_nas(tmp_path):
    root,manifest,sensor,raw,hashes=fixture(tmp_path)
    row=json.loads(raw.splitlines()[0]);row['sensors']['radar']['power']=str(sensor)
    raw=(json.dumps(row)+'\n').encode();manifest.write_bytes(raw)
    with pytest.raises(ValueError,match='Absolute radar'):
        cache.stage(root,'prepared/manifest.jsonl',tmp_path/'ssd',hashlib.sha256(raw).hexdigest(),hashes,
                    limit=Unlimited(),min_free_gib=0,enforce_local=False)


def test_limits_enforce_bandwidth_and_open_interval():
    clock=[0.];sleeps=[]
    def sleep(t):sleeps.append(t);clock[0]+=t
    budget=cache.Limit(1,2,lambda:clock[0],sleep)
    budget.opened();budget.opened();assert clock[0]==.5
    budget.consumed(2*1024**2);assert clock[0]>=2
    assert sum(sleeps)>=2


def test_reject_source_tree_as_cache_destination(tmp_path):
    root,manifest,sensor,raw,hashes=fixture(tmp_path)
    with pytest.raises(ValueError,match='outside source'):
        cache.stage(root,'prepared/manifest.jsonl',root/'cache',hashlib.sha256(raw).hexdigest(),hashes,
                    limit=Unlimited(),min_free_gib=0,enforce_local=False)


def test_duplicate_stager_rejected_before_network_open(tmp_path):
    root,manifest,sensor,raw,hashes=fixture(tmp_path);dest=tmp_path/'ssd';dest.mkdir();(dest/'.stage.lock').write_text('123')
    with pytest.raises(FileExistsError):
        cache.stage(root,'prepared/manifest.jsonl',dest,hashlib.sha256(raw).hexdigest(),hashes,
                    limit=Unlimited(),min_free_gib=0,enforce_local=False,
                    source_open=lambda p:pytest.fail('No duplicate network scan allowed'))


def test_failed_recheck_invalidates_previous_ready_flag(tmp_path):
    root,manifest,sensor,raw,hashes=fixture(tmp_path);dest=tmp_path/'ssd'
    kwargs=dict(limit=Unlimited(),min_free_gib=0,enforce_local=False)
    cache.stage(root,'prepared/manifest.jsonl',dest,hashlib.sha256(raw).hexdigest(),hashes,**kwargs)
    (dest/'root/radar.bin').write_bytes(b'bad-local');sensor.write_bytes(b'bad-source')
    with pytest.raises(ValueError):
        cache.stage(root,'prepared/manifest.jsonl',dest,hashlib.sha256(raw).hexdigest(),hashes,**kwargs)
    assert not (dest/'CACHE_READY.json').exists()


def test_partial_file_resume_seeks_past_local_prefix(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'abcdefgh');target=tmp_path/'target'
    Path(str(target)+'.partial').write_bytes(b'abcd')
    fresh=cache.copy_verified(source,target,cache.digest(source),Unlimited(),lambda p:p.open('rb'))
    assert fresh==4 and target.read_bytes()==b'abcdefgh'


def test_model_asset_cache_has_no_control_files_inside_model_directory(tmp_path):
    source=tmp_path/'model';source.mkdir();(source/'config.json').write_bytes(b'{}')
    hashes={'config.json':cache.digest(source/'config.json')};dest=tmp_path/'model-cache'
    first=cache.stage_assets(source,dest,hashes,Unlimited(),min_free_gib=0,enforce_local=False)
    second=cache.stage_assets(source,dest,hashes,Unlimited(),min_free_gib=0,enforce_local=False,
                              source_open=lambda p:pytest.fail('No repeated network weight read'))
    assert second['network_bytes_this_run']==0
    assert [p.name for p in Path(first['root']).iterdir()]==['config.json']
