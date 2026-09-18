"""Bounded observation-prefix loader. Test access requires a separate capability."""
import csv
import hashlib
import os
from contextlib import nullcontext
from pathlib import Path

from utils.ch3_contract import digest, profile


def prefix_csv(path, count, expected_features=None):
    import numpy as np
    if not isinstance(count,int) or count<1: raise ValueError('verified prefix endpoint required')
    rows=[]; times=[]; h=hashlib.sha256()
    access=nullcontext()
    if os.environ.get('AMD_RR_CONFIG'):
        from restricted_io_guard import permit_ch3_prefix
        access=permit_ch3_prefix(path,count)
    with access, Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream,strict=True)
        header=next(reader)
        if len(header)!=len(set(header)): raise ValueError('duplicate columns')
        if expected_features is not None and header[1:]!=expected_features:raise ValueError('ordered column binding mismatch')
        h.update((','.join(header)+'\n').encode())
        for i in range(count):
            try:row=next(reader)
            except StopIteration:raise ValueError(f'prefix shorter than declared {count}')
            if len(row)!=len(header):raise ValueError('row width mismatch')
            times.append(row[0]);rows.append([float(x) for x in row[1:]])
            h.update((','.join(row)+'\n').encode())
    values=np.asarray(rows,dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('nonfinite prefix')
    return header,times,values,dict(parsed_records=count,record_prefix_sha256=h.hexdigest(),
        fingerprint_semantics='UTF-8 normalized CSV records; not full-file SHA',test_observations_accessed=False)


def adjacent_timestamps(path, start=19041, stop=19048):
    """Diagnostic only: zero-based record indices, no numeric observations parsed."""
    if not 0<=start<stop<=42157:raise ValueError('Weather diagnostic prefix boundary')
    access=nullcontext()
    if os.environ.get('AMD_RR_CONFIG'):
        from restricted_io_guard import permit_ch3_prefix
        access=permit_ch3_prefix(path,stop)
    result=[]
    with access,Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream,strict=True);next(reader)
        for i in range(stop):
            row=next(reader)
            if i>=start:result.append(dict(record_index=i,raw_timestamp=row[0]))
    return dict(rows=result,records_traversed=stop,numeric_observations_parsed=0,
                test_observations_accessed=False,full_file_verified=False)


class Windows:
    def __init__(self,x,y,start,end,T,H,*,urban=False,label_horizon=None):
        self.x,self.y=x,y;self.start=start;self.end=end;self.T=T;self.H=H
        self.urban=urban;self.label_horizon=label_horizon
        self.nodes=x.shape[1] if urban else 1
        self.count=end-start-T-(label_horizon if urban else H)+1
        if self.count<=0:raise ValueError('empty complete-label split')
    def __len__(self):return self.count*self.nodes
    def __getitem__(self,index):
        import torch
        if not 0<=index<len(self):raise IndexError(index)
        w,node=divmod(index,self.nodes);i=self.start+w
        if self.urban:
            x=self.x[i:i+self.T,node,:]
            y=self.y[i+self.T+self.label_horizon-1,node:node+1].reshape(1,1)
        else:x=self.x[i:i+self.T];y=self.y[i+self.T:i+self.T+self.H]
        return torch.from_numpy(x),torch.from_numpy(y)


def verify_source_state(d, path=None):
    """Reuse the audited byte identity only while its recorded file state holds."""
    binding=d.get('source_admission',{}).get('local_state')
    if binding is None:return None
    s=Path(path or d['path']).stat()
    actual=dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
    if actual!=binding:raise ValueError('audited data file state changed; source review required')
    return actual


def validate_times(times, dataset, d):
    policy=d.get('time_index_policy','unique_nondecreasing')
    if policy not in ('unique_nondecreasing','weather_original_records'):
        raise ValueError('unknown time index policy')
    if policy=='weather_original_records' and dataset!='Weather':
        raise ValueError('record-index exception is Weather-only')
    if not times.is_monotonic_increasing or (not times.is_unique and policy!='weather_original_records'):
        raise ValueError('ordered unique time index required except approved Weather duplicates; reverse time forbidden')
    duplicates=times.duplicated()
    return dict(policy=policy,duplicate_timestamps=int(duplicates.sum()),
                duplicate_record_indices=[i for i,duplicate in enumerate(duplicates) if duplicate],
                strictly_regular_sampling_claimed=False)


def load(c,task,*,test_capability=None,path_override=None):
    import numpy as np
    import pandas as pd
    p=profile(c,task);d=c['datasets'][task['dataset']]
    final=test_capability is not None
    if final and test_capability!={'purpose':'ch3_formal_test','protocol_sha':digest(c),'run_id':task['id']}:
        raise PermissionError('explicit matching formal test capability required')
    if task['dataset']=='UrbanEV':
        train_end,val_end,test_end=c['urban_folds'][task['fold']-1]
        end=test_end if final else val_end
        root=Path(path_override or d['path']);raw={};provenance={};clock=None;node_names=None
        for file in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv'):
            header,times,values,info=prefix_csv(root/file,end)
            parsed=pd.DatetimeIndex(pd.to_datetime(times,errors='raise'))
            if parsed.tz is not None or not parsed.is_unique or not parsed.is_monotonic_increasing:raise ValueError('UrbanEV time policy')
            if clock is None:clock=parsed;node_names=header[1:]
            if not parsed.equals(clock):raise ValueError('UrbanEV prefix time mismatch')
            if file!='weather_central.csv' and header[1:]!=node_names:raise ValueError('node order mismatch')
            if len(node_names)!=275 and path_override is None:raise ValueError('official node count')
            raw[file]=(header,values);provenance[file]=info
        times=pd.DatetimeIndex(pd.to_datetime(clock,errors='raise'))
        expected=pd.date_range('2022-09-01',periods=end,freq='h')
        if not times.equals(expected):raise ValueError('UrbanEV hourly calendar binding')
        from utils.dataloader_urbanev import _standard_stats,_minmax_stats,_calendar_features
        volume=raw['volume.csv'][1];mean,scale=_standard_stats(volume[:train_end]);y=((volume-mean)/scale).astype('float32')
        parts={'volume':y};stats={'volume_mean':mean.tolist(),'volume_scale':scale.tolist()}
        for file,name in [('e_price.csv','e_price'),('s_price.csv','s_price')]:
            v=raw[file][1];lo,ran,safe=_minmax_stats(v[:train_end]);parts[name]=(v-lo)/safe
            stats[name]=dict(min=lo.tolist(),range=ran.tolist(),safe=safe.tolist())
        header,weather=raw['weather_central.csv'];order=[header[1:].index(k) for k in ['T','P','U']]
        weather=weather[:,order];wm,ws=_standard_stats(weather[:train_end]);weather=(weather-wm)/ws
        stats['weather']=dict(mean=wm.tolist(),scale=ws.tolist())
        for i,name in enumerate(['Ta','P','h']):parts[name]=np.broadcast_to(weather[:,i,None],volume.shape)
        calendar=_calendar_features(times)
        for i,name in enumerate(['hour_sin','hour_cos','weekday_sin','weekday_cos','is_weekend']):parts[name]=np.broadcast_to(calendar[:,i,None],volume.shape)
        x=np.stack([parts[n] for n in p['features']],axis=-1).astype('float32')
        splits={'train':Windows(x,y,0,train_end,p['T'],1,urban=True,label_horizon=task['h']),
                'validation':Windows(x,y,train_end,val_end,p['T'],1,urban=True,label_horizon=task['h'])}
        if final:splits['test']=Windows(x,y,val_end,test_end,p['T'],1,urban=True,label_horizon=task['h'])
        info=dict(prefixes=provenance,node_order=node_names,scaler=stats)
    else:
        if d['endpoints'] is None:raise ValueError('version-specific row count/endpoints Not verified; fit policy is already F=1')
        train_end,val_end,test_end=d['endpoints'];end=test_end if final else val_end
        inherited_state=verify_source_state(d) if path_override is None else None
        header,clock,values,info=prefix_csv(path_override or d['path'],end,d['features'])
        if path_override is None:verify_source_state(d)
        times=pd.DatetimeIndex(pd.to_datetime(clock,errors='raise',format=d.get('timestamp_format')))
        info['time_index']=validate_times(times,task['dataset'],d)
        info['source_admission']=d.get('source_admission',{})
        info['inherited_source_state_checked']=inherited_state is not None
        selected=[d['features'].index(n) for n in p['features']];v=values[:,selected]
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler().fit(v[:train_end]);x=scaler.transform(v).astype('float32');y=x[:,p['target_idx']:p['target_idx']+1]
        splits={'train':Windows(x,y,0,train_end,p['T'],p['pred_len']),
                'validation':Windows(x,y,train_end-p['T'],val_end,p['T'],p['pred_len'])}
        if final:splits['test']=Windows(x,y,val_end-p['T'],test_end,p['T'],p['pred_len'])
        info['scaler']={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()}
        info['scaler_fit_records']=train_end
    info.update(dataset=task['dataset'],fold=task['fold'],input_variant=task['input_variant'],target=p['features'][p['target_idx']],
                endpoints_read=[train_end,end],declared_test_end=test_end,full_file_verified=False,
                test_observations_accessed=final,window_counts={k:len(v) for k,v in splits.items()},
                cpu_array_bytes=x.nbytes+y.nbytes,time_first=str(times[0]),time_last_read=str(times[-1]),
                interval_counts={str(k):int(v) for k,v in pd.Series(times[1:]-times[:-1]).value_counts().items()})
    return splits,info


def batches(dataset,p,split,generator=None):
    from torch.utils.data import DataLoader
    training=split=='train'
    return DataLoader(dataset,batch_size=p['training']['batch' if training else 'eval_batch'],
                      shuffle=training,drop_last=training,num_workers=0,generator=generator)
