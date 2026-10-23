import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from radar_vla.raw_visualization import (RawSession, future_xy, sensor_to_ego,
    world_to_current, radar_vertices, road_pieces, render, inventory, build_widget)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def future(x):
    return dict(valid=[True, False, True]+[False]*27,
                world_xyzyaw=[[x, 21, 0, 90], None, [x, 23, 0, 90]]+[None]*27)


@pytest.fixture
def raw(tmp_path):
    root = tmp_path
    actor = dict(actor_id=1, actor_type='vehicle', pose=dict(position_m=[10,20,0], rotation_deg=[0,0,90]),
                 bbox=dict(center_actor_m=[0,0,0], extent_m=[2,1,1], rotation_actor_deg=[0,0,0]))
    agent = dict(actor, actor_id=2, pose=dict(position_m=[8,25,0], rotation_deg=[0,0,90]))
    sensor = dict(x=1.5,y=.2,z=2.4,roll=0,pitch=0,yaw=0)
    ep = root/'full/jobs/job/episode'
    write(ep/'metadata.json', dict(episode_id='ep', sensors=dict(semantic_lidar=sensor)))
    rows=[]
    road=dict(target_schema='ego_road_corridor_50m_v1', centerline_ego_m=[[1,0],[2,0],None,[4,1],[5,1],[7,2]],
              valid=[True,True,False,True,True,True], break_before=[False,False,False,False,True,False],
              width_m=[4,4,None,4,4,4], source_ids=[dict(road_id=7,section_id=0)]*6)
    for i, fid in enumerate([100,101,102]):
        folder=ep/'frames'/f'{i:06d}'
        write(folder/'frame.json',dict(carla_frame=fid,camera_frame=fid,lidar_frame=fid,index=i,episode_id='ep',
              sensors=dict(semantic_lidar='semantic_lidar.npy'),agents='agents.json',map='map.json',ego=actor))
        write(folder/'agents.json',dict(carla_frame=fid,agents=[agent]))
        write(folder/'map.json',dict(carla_frame=fid,map=road))
        np.save(folder/'semantic_lidar.npy',np.array([[2,-1,0,.9,2,10],[3,1,0,1,0,7]],float))
        window='000000_000001' if i<2 else '000002_000002'
        ri=i if i<2 else 0
        product=root/'full/jobs/job/radar'/window/'attempt_002'
        rf=product/'frames'/f'{ri:06d}';rf.mkdir(parents=True)
        shape=(3,3)
        np.savez(rf/'folded_ra.npz',power_peak_ra=np.ones(shape)*(i+1))
        np.savez(rf/'unfolded_ra.npz',doppler_unfolded_dense_ra_mps=np.full(shape,9+i),
                 reliable_mask=np.eye(3,dtype=bool))
        np.savez(product/'radar_axes.npz',range_m=[0,1,2],azimuth_deg=[-45,0,45])
        write(product/'complete.json',dict(status='PASS'))
        frames=[dict(carla_frame=100,source_index=0),dict(carla_frame=101,source_index=1)] if i<2 else [dict(carla_frame=102,source_index=2)]
        write(product/'radar_meta.json',dict(product_schema='dense_hann3d_v2',frames=frames,global_power_peak=10))
        rows.append(dict(carla_frame=fid,source_index=i,radar_frame_index=ri,job_id='job',episode_id='ep',
                         radar_product='/old/machine/RadarVLA/'+str(product.relative_to(root)),
                         split=['train','val','test'][i],weather='fog',category='normal'))
    (ep/'anchors.jsonl').write_text(''.join(json.dumps(dict(carla_frame=fid,future_steps=30,future_dt_s=.1,
        futures_by_actor_id={'1':future(10),'2':future(8)}))+'\n' for fid in [100,101,102]))
    (root/'full/dataset_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(root/'prepared/version/report.json',dict(status='PREPARED',prepared_rows=2,excluded_rows=1))
    write(root/'prepared/version/exclusions.jsonl',dict(reason='missing history'))
    return root


def test_rotated_ego_sensor_mount_and_polar_right_sign():
    pose=dict(position_m=[10,20,0],rotation_deg=[0,0,90])
    assert np.allclose(world_to_current([[10,25,0],[8,20,0]],pose),[[5,0,0],[0,2,0]])
    sensor=dict(x=1.5,y=.5,z=2.4,roll=0,pitch=0,yaw=90)
    assert np.allclose(sensor_to_ego([[2,0,0]],sensor),[[1.5,2.5,2.4]])
    sensor['yaw']=0
    v=radar_vertices(np.array([0,2]),np.array([-30,30]),sensor)
    assert v.shape==(3,3,3)
    assert v[-1,-1,1] > sensor['y']  # positive raw azimuth is right
    assert v[-1,0,1] < sensor['y']


def test_future_invalid_samples_break_lines():
    xy=future_xy(future(10),[10,20,0],dict(position_m=[10,20,0],rotation_deg=[0,0,90]))
    assert np.allclose(xy[0],[0,0])
    assert np.allclose(xy[1],[1,0])
    assert np.isnan(xy[2]).all()
    assert np.allclose(xy[3],[3,0])


def test_window_switch_and_relocated_root(raw):
    session=RawSession(raw)
    assert len(session.select())==3
    assert session.select(split='test')[0]['carla_frame']==102
    for row in session.index:
        sample=session.load(row)
        assert sample['speed'][0,0]==9+row['source_index']
    assert len(session._anchor_indices)==1
    sample=session.load(session.index[0])
    pieces=list(road_pieces(sample['road']))
    assert [len(p[0]) for p in pieces]==[2,1,2]
    fig=render(sample)
    dense=np.ma.asarray(fig.axes[2].collections[0].get_array())
    assert dense.count()==9
    assert np.all(dense==9)
    # All three panels carry identical current-actor boxes and future lines.
    assert [len(ax.patches) for ax in fig.axes[:3]]==[4,4,4]
    plt.close(fig)
    fig=render(sample,reliable_only=True)
    masked=np.ma.asarray(fig.axes[2].collections[0].get_array())
    assert masked.count()==3
    assert np.all(masked.compressed()==9)
    plt.close(fig)


@pytest.mark.parametrize('file,key',[('frame.json','lidar_frame'),('agents.json','carla_frame'),('map.json','carla_frame')])
def test_mismatch_reports_frame_and_file(raw,file,key):
    path=raw/'full/jobs/job/episode/frames/000000'/file
    value=json.loads(path.read_text());value[key]=999;write(path,value)
    session=RawSession(raw)
    with pytest.raises(ValueError,match=f'{file}.*expected CARLA frame 100'):
        session.load(session.index[0])


def test_missing_array_and_radar_mismatch(raw):
    session=RawSession(raw)
    product=session.product_path(session.index[0])
    value=json.loads((product/'radar_meta.json').read_text());value['frames'][0]['carla_frame']=999
    write(product/'radar_meta.json',value)
    with pytest.raises(ValueError,match='radar_meta.json.*expected CARLA frame 100'):
        session.load(session.index[0])
    session._radar_meta.cache_clear()
    value['frames'][0]['carla_frame']=100;write(product/'radar_meta.json',value)
    (product/'frames/000000/folded_ra.npz').unlink()
    with pytest.raises(ValueError,match='CARLA frame 100.*folded_ra.npz'):
        session.load(session.index[0])


def test_inventory_and_widget(raw,monkeypatch):
    report=inventory(raw)
    assert report['accepted_frames']==3
    assert report['radar_windows']==2
    assert not report['missing_completion_markers']
    assert report['prepared'][0]['exclusion_reasons']=={'missing history':1}
    import IPython.display
    monkeypatch.setattr(IPython.display,'display',lambda *args:None)
    widget=build_widget(RawSession(raw),raw/'export')
    assert widget.raw_bev_state['row']['carla_frame']==100
    assert not widget.raw_bev_state['error']
    widget.children[4].children[1].click()
    assert widget.raw_bev_state['row']['carla_frame']==101
    widget.children[4].children[2].click()
    assert list((raw/'export').glob('*101.png'))
    widget.children[0].children[0].value='test'
    assert widget.raw_bev_state['row']['carla_frame']==102


def test_cli_inventory_render_and_empty_selection(raw,capsys):
    from radar_vla.raw_visualization import main
    dest=raw/'cli_output'
    main(['inventory','--data-root',str(raw),'--output-dir',str(dest)])
    report=json.loads((dest/'inventory.json').read_text())
    assert report['jobs'][0]['frames']==3
    assert 'job' in (dest/'inventory.md').read_text()
    main(['render','--data-root',str(raw),'--output-dir',str(dest),'--split','test'])
    assert (dest/'job_102.png').is_file()
    assert json.loads((dest/'render_report.json').read_text())[0]['source_index']==2
    with pytest.raises(SystemExit) as exc:
        main(['render','--data-root',str(raw),'--weather','rain'])
    assert exc.value.code==2
