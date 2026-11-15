"""Prepare bounded train/validation prefixes, identities and new-model supports only."""
import argparse
import csv
import hashlib
import io
import json
from itertools import islice
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .imports import sha
from .graphs import geographic_neighbors, semantic_neighbors, physical_shortest_paths
from .data import fit_scaler, causal_fill, GraphWindows, batch_size

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT.parent/'amd-execution-evidence/m7/M7-v10-spatial-implementation-r1'


def prefix(stream, count):
    reader=csv.reader(stream);header=next(reader)
    rows=list(islice(reader,count))
    if len(rows)!=count:raise ValueError('short data prefix')
    return pd.DataFrame(rows,columns=header)


def save_domain(destination, x, y, valid, times, nodes, ends, scale, metadata, distance, graph_target=None, _final=False):
    destination.mkdir(exist_ok=False,parents=True)
    train_end,val_end,test_end=ends
    if len(x)!=(test_end if _final else val_end):raise ValueError('prefix endpoint mismatch')
    if len(set(nodes))!=len(nodes):raise ValueError('duplicate node id')
    semantic,info=semantic_neighbors((y if graph_target is None else graph_target)[:train_end],valid[:train_end],times[:train_end])
    geo=geographic_neighbors(distance)
    for name,array in dict(x=x,y=y,valid=valid,times=times,geo=geo,sem=semantic,distance=distance,
                           semantic_profile=info['profile'],semantic_constant=info['constant'],target_scale=scale).items():
        np.save(destination/(name+'.npy'),array,allow_pickle=False)
    metadata.update(nodes=len(nodes),node_order=list(nodes),endpoints=list(ends),train_last=str(times[train_end-1]),
                    validation_first=str(times[train_end]),last_read=str(times[-1]),
                    test_observations_loaded=_final,test_evaluation_count=0,
                    proposed_batch=batch_size(metadata.get('city',metadata['dataset'])),
                    graph=dict(geo='finite nonself nearest8; bidirectional union; self',
                               sem='train24h centered L2 cosine positive directed top8; self',
                               train_rows=train_end,code_sha=sha(ROOT/'ch4/graphs.py')),
                    files={p.name:sha(p) for p in destination.iterdir() if p.is_file()})
    (destination/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    return metadata


def urban(destination,_final=False):
    from utils.ch3_data import load
    protocol=json.loads((ROOT/'configs/ch3_baseline_ms_u96_oc01_v3.json').read_text())
    task=next(t for t in protocol['tasks'] if t['model']=='ModernTCN' and t['dataset']=='UrbanEV' and t['fold']==6 and t['h']==3)
    p=protocol['resolved_profiles'][task['id']]
    from utils.ch3_contract import digest
    capability=dict(purpose='ch3_formal_test',protocol_sha=digest(protocol),run_id=task['id']) if _final else None
    # This token is the frozen loader's internal API, not a claim of CH3 production.
    # Only the M7 whole-batch freeze gate calls this private final path.
    splits,info=load(protocol,task,resolved_profile=p,test_capability=capability)
    x,y=splits['train'].x,splits['train'].y
    ends=protocol['urban_folds'][5];times=pd.date_range('2022-09-01',periods=ends[2] if _final else ends[1],freq='h').to_numpy()
    nodes=list(info['node_order']);data_root=Path(protocol['datasets']['UrbanEV']['path'])
    frame=pd.read_csv(data_root/'distance.csv',index_col=None);frame.columns=frame.columns.astype(str)
    if list(frame.columns)!=nodes or frame.shape!=(len(nodes),len(nodes)):
        raise ValueError('Urban regional matrix implicit row order/header differs from volume node order')
    distance=frame.to_numpy(float)
    metadata=dict(dataset='UrbanEV',source=info,features=p['features'],target_idx=0,
                  distance_source=dict(path=str(data_root/'distance.csv'),sha256=sha(data_root/'distance.csv')),
                  split_policy='frozen fold6 split-local windows',upstream_preprocessing='original UrbanEV release; not a new untouched holdout')
    scale=np.asarray(info['scaler']['volume_scale']);mean=np.asarray(info['scaler']['volume_mean'])
    from utils.ch3_data import prefix_csv
    _,_,raw_target,graph_prefix=prefix_csv(data_root/'volume.csv',ends[0],nodes)
    metadata['graph_target_prefix']=graph_prefix
    return save_domain(destination,x,y,np.isfinite(y),times,nodes,ends,scale,metadata,distance,graph_target=raw_target,_final=_final)


def charged(destination,city,archive,_final=False):
    archive=Path(archive)
    acquisition=json.loads(archive.with_name(archive.name+'.acquisition.json').read_text())
    if not acquisition.get('verified'):raise ValueError('CHARGED v3 archive integrity not verified')
    times=pd.date_range('2023-04-01', '2023-10-01',inclusive='left',freq='h').to_numpy()
    total=len(times);ends=(int(.8*total),int(.9*total),total);stop=ends[2] if _final else ends[1]
    with zipfile.ZipFile(archive) as z:
        candidates=[n for n in z.namelist() if n.endswith('/volume.csv') and city in n.split('/') and 'remove_zero' not in n]
        if len(candidates)!=1:raise ValueError('ambiguous complete city path: '+repr(candidates))
        base=candidates[0].rsplit('/',1)[0]
        def table(name,rows=None):
            with z.open(base+'/'+name) as binary,io.TextIOWrapper(binary,encoding='utf-8-sig') as stream:
                return prefix(stream,rows) if rows is not None else pd.read_csv(stream)
        volume=table('volume.csv',stop);weather=table('weather.csv',stop)
        clock=pd.to_datetime(volume.iloc[:,0],errors='raise').to_numpy()
        if not np.array_equal(clock,times[:stop]):raise ValueError(city+' volume timestamp prefix differs from approved hourly axis')
        wc=pd.to_datetime(weather.iloc[:,0],errors='raise').to_numpy()
        if not np.array_equal(wc,clock):raise ValueError(city+' weather timestamp mismatch')
        nodes=list(volume.columns[1:]);v=volume.iloc[:,1:].apply(pd.to_numeric,errors='raise').to_numpy(float)
        weather_values=weather[['temp','humidity','pressure']].apply(pd.to_numeric,errors='raise').to_numpy(float)
        raw=np.concatenate((v[...,None],np.broadcast_to(weather_values[:,None],(stop,len(nodes),3))),-1)
        valid=np.isfinite(raw);mean,scale=fit_scaler(raw[:ends[0]],valid[:ends[0]],shared_features=(1,2,3))
        filled=causal_fill(raw,valid,mean);x=((filled-mean)/scale).astype(np.float32)
        y=((v-mean[0])/scale[0]).astype(np.float32)
        distance=table('distance.csv');distance=distance.set_index(distance.columns[0]);distance.index=distance.index.astype(str);distance.columns=distance.columns.astype(str)
        sites=table('sites.csv')
        site_column='site' if city in ('MEL','SPO') else 'site_id'
        if site_column not in sites:raise ValueError(city+' verified station identifier column absent')
        sites=sites.rename(columns={site_column:'site_id'});sites['site_id']=sites['site_id'].astype(str)
        if set(distance.index)!=set(nodes) or set(distance.columns)!=set(nodes) or set(sites['site_id'])!=set(nodes):
            raise ValueError(city+' complete station/distance/coordinate IDs mismatch')
        coordinates=sites.set_index('site_id').loc[nodes,['longitude','latitude']].to_numpy(float)
        if not np.isfinite(coordinates).all():raise ValueError(city+' nonfinite coordinates')
        metadata=dict(dataset='CHARGED',city=city,version='v3',doi='10.5281/zenodo.15814263',
                      source_archive=dict(path=str(archive),sha256=sha(archive),md5=acquisition['md5']),
                      member_base=base,station_id_column=site_column,features=['volume','temp','humidity','pressure'],target_idx=0,
                      target_unit='kWh',scaler=dict(mean=mean.tolist(),scale=scale.tolist(),population=True,weather_unique_time=True),
                      input_missing_by_variable=(~valid).sum((0,1)).tolist(),target_missing=int((~valid[...,0]).sum()),
                      split_policy='80/10/10 timestamp floors; complete labels; preceding history permitted',
                      distance_definition='official WGS84 ellipsoidal geodesic km',
                      upstream_preprocessing='release README: IQR correction and zero sequence interpolation; not end-to-end causal provenance')
        result=save_domain(destination,x,y,valid[...,0],clock,nodes,ends,np.asarray([scale[0]]),metadata,distance.loc[nodes,nodes].to_numpy(float),graph_target=v,_final=_final)
        np.save(destination/'coordinates.npy',coordinates,allow_pickle=False)
        result['files']['coordinates.npy']=sha(destination/'coordinates.npy')
        (destination/'metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('domain',choices=['UrbanEV','AMS','JHB','LOA','MEL','SPO','SZH','PEMS04','PEMS08']);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=urban(args.output) if args.domain=='UrbanEV' else pems(args.output,args.domain) if args.domain.startswith('PEMS') else charged(args.output,args.domain,EVIDENCE/'data/CHARGED-Hourly-r2.zip')
    print(json.dumps({k:result[k] for k in ['nodes','endpoints','train_last','last_read','proposed_batch']},ensure_ascii=False))


def npz_prefix(path,stop):
    """Decode only the admitted numeric prefix of the single C-order NPY member."""
    with zipfile.ZipFile(path) as archive:
        if archive.namelist()!=['data.npy']:raise ValueError('unbound NPZ keys')
        with archive.open('data.npy') as f:
            version=np.lib.format.read_magic(f)
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=reader(f)
            if fortran or dtype.hasobject or len(shape)!=3 or not 0<stop<=shape[0]:raise ValueError('NPZ layout')
            size=stop*shape[1]*shape[2]*dtype.itemsize
            block=f.read(size)
            if len(block)!=size:raise ValueError('short numeric prefix')
            return np.frombuffer(block,dtype=dtype).reshape(stop,*shape[1:]).copy(),dict(key='data',shape=list(shape),dtype=str(dtype),decoded_rows=stop)


def pems(destination,domain,_final=False):
    source=EVIDENCE/'data/ASTGCN-86efe627';root=source/'data'/domain
    rows,nodes,start=(16992,307,'2018-01-01') if domain=='PEMS04' else (17856,170,'2016-07-01')
    ends=(int(.6*rows),int(.8*rows),rows)
    stop=ends[2] if _final else ends[1]
    path=root/(domain.lower()+'.npz');raw,layout=npz_prefix(path,stop)
    if layout['shape']!=[rows,nodes,3]:raise ValueError('published PEMS metadata mismatch')
    valid=np.isfinite(raw);mean,scale=fit_scaler(raw[:ends[0]],valid[:ends[0]])
    x=((causal_fill(raw,valid,mean)-mean)/scale).astype(np.float32)
    y=((raw[:,:,0]-mean[0])/scale[0]).astype(np.float32)
    edges=pd.read_csv(root/'distance.csv')
    if list(edges)!=['from','to','cost']:raise ValueError('official distance columns')
    distance=physical_shortest_paths(nodes,edges.to_numpy())
    times=pd.date_range(start,periods=stop,freq='5min').to_numpy()
    metadata=dict(dataset=domain,features=['flow','occupancy','speed'],target_idx=0,
                  source_commit='86efe627f8d7a1d461be9446a3c9d0960c301d0b',source_npz=dict(path=str(path),sha256=sha(path),**layout),
                  channel_definition=dict(path=str(source/'README.md'),sha256=sha(source/'README.md')),
                  node_mapping='array axis indices 0..N-1; original lib/utils.py get_adjacency_matrix uses integer edge indices',
                  distance_source=dict(path=str(root/'distance.csv'),sha256=sha(root/'distance.csv')),
                  distance_definition='official distance weights; units and road-length interpretation unverified; undirected minimum parallel edge then shortest path',
                  scaler=dict(mean=mean.tolist(),scale=scale.tolist(),population=True),
                  input_missing_by_variable=(~valid).sum((0,1)).tolist(),
                  split_policy='60/20/20 timestamp floors; labels wholly within split; prior history permitted',
                  upstream_preprocessing='author-curated benchmark from Caltrans PeMS; release may include retrospective cleaning; original sensor IDs/GPS not supplied')
    return save_domain(destination,x,y,valid[:,:,0],times,[str(i) for i in range(nodes)],ends,np.asarray([scale[0]]),metadata,distance,raw[:,:,0],_final=_final)


if __name__=='__main__':main()
