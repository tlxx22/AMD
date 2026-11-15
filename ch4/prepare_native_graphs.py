"""Materialize approved fixed relations; never accesses target/test values."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .imports import sha
from .native_graphs import charged_adjacency,save_fixed_artifacts
from .protocol import CITIES
from .prepare_data import ROOT,EVIDENCE


def prepare(domain,destination):
    prefix=EVIDENCE/'data'/(domain+('-prefix-r1' if domain.startswith('PEMS') else '-prefix-r2'))
    meta=json.loads((prefix/'metadata.json').read_text());nodes=meta['node_order'];n=len(nodes)
    identity=dict(domain=domain,reviewed_migration='user M7 fixed geographic graph amendment',
                  metadata_sha256=sha(prefix/'metadata.json'),node_order=nodes,
                  node_order_sha256=hashlib.sha256(json.dumps(nodes,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
                  demand_values_read=False,threshold_selected_using_validation_or_test=False,
                  graph_code_sha256=sha(ROOT/'ch4/native_graphs.py'),preparation_code_sha256=sha(__file__))
    pd_a=None
    if domain in CITIES:
        archive=Path(meta['source_archive']['path']);member=meta['member_base']+'/distance.csv'
        if sha(archive)!=meta['source_archive']['sha256']:raise ValueError('archive binding changed')
        with zipfile.ZipFile(archive) as z:raw=z.read(member)
        frame=pd.read_csv(io.BytesIO(raw),index_col=0);frame.index=frame.index.astype(str);frame.columns=frame.columns.astype(str)
        if frame.index.has_duplicates or frame.columns.has_duplicates or set(frame.index)!=set(nodes) or set(frame.columns)!=set(nodes):
            raise ValueError('official distance IDs differ from complete data node set')
        aligned=frame.loc[nodes,nodes]
        a,rule=charged_adjacency(aligned.to_numpy(),nodes,list(aligned.index),list(aligned.columns))
        identity.update(source_sha256=hashlib.sha256(raw).hexdigest(),source_archive=meta['source_archive'],source_member=member,
                        original_row_ids=frame.index.tolist(),original_column_ids=frame.columns.tolist(),
                        alignment='explicit ID reindex to target columns',rule=rule,
                        input_type='already Gaussian weighted adjacency; PDFormer must not Gaussianize again')
    elif domain=='UrbanEV':
        protocol=json.loads((ROOT/'configs/ch3_baseline_ms_u96_oc01_v3.json').read_text())
        path=Path(protocol['datasets']['UrbanEV']['path'])/'adj.csv';frame=pd.read_csv(path)
        if list(frame.columns.astype(str))!=nodes or frame.shape!=(n,n):raise ValueError('Urban adjacency implicit rows/header mismatch')
        a=frame.to_numpy(dtype=np.float32)
        identity.update(source_sha256=sha(path),source_path=str(path),rule='original adj.csv unchanged; implicit row order equals header/volume order',
                        input_type='original topology; no global symmetrization or Gaussian transform')
    else:
        path=Path(meta['distance_source']['path'])
        if sha(path)!=meta['distance_source']['sha256']:raise ValueError('published edge binding changed')
        edges=pd.read_csv(path);a=np.zeros((n,n),dtype=np.float32)
        for i,j,cost in edges.itertuples(index=False,name=None):
            if int(i)!=i or int(j)!=j or not (0<=i<n and 0<=j<n) or not np.isfinite(cost) or cost<0:raise ValueError('invalid published edge')
            a[int(i),int(j)]=1
        # PDFormer PEMS profile's bidir=True is applied only to its derived copy.
        pd_a=np.maximum(a,a.T)
        identity.update(source_sha256=sha(path),source_path=str(path),rule='directed published edges -> binary connection; cost retained only in source',
                        pdformer_derivation='PEMS author profile bidir=True applied to connection copy; hop path; no Gaussian distance conversion',
                        cost_unit='unverified',input_type='connection, not distance strength')
    refs=save_fixed_artifacts(destination,a,identity,non_pems=not domain.startswith('PEMS'),pd_adjacency=pd_a)
    result=json.loads((Path(destination)/'identity.json').read_text())
    print(json.dumps(dict(domain=domain,files=len(refs),diagnostics=result['diagnostics'],failures=result['failures']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('domain',choices=['UrbanEV',*CITIES,'PEMS04','PEMS08']);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare(args.domain,args.output)
