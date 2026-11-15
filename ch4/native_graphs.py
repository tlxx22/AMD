"""Source-native graph preprocessing given an explicitly reviewed adjacency.

The distance-to-adjacency migration is a separate required input. This module
does not silently reuse the proposed model's top-k/cosine supports.
"""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import connected_components
from .imports import sha
from .runtime_io import write_new


def doubletransition(adjacency):
    a=np.asarray(adjacency,dtype=np.float32)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all() or (a<0).any():raise ValueError('finite nonnegative square adjacency')
    def transition(x):
        degree=x.sum(1);inverse=np.divide(1.,degree,out=np.zeros_like(degree),where=degree!=0)
        return (inverse[:,None]*x).astype(np.float32)
    return np.stack((transition(a),transition(a.T)))


def d2_doubletransition(adjacency):
    """D2STGNN load_adj transposes both row-normalized supports."""
    return doubletransition(adjacency).transpose(0,2,1).copy()


def charged_adjacency(distance,node_order,row_ids,column_ids):
    d=np.asarray(distance,dtype=np.float64)
    if list(row_ids)!=list(node_order) or list(column_ids)!=list(node_order):
        raise ValueError('distance node order mismatch')
    if len(set(node_order))!=len(node_order) or d.shape!=(len(node_order),len(node_order)):
        raise ValueError('distance dimensions/duplicate IDs')
    if not np.isfinite(d).all() or (d<0).any() or not np.array_equal(d,d.T) or np.any(np.diag(d)!=0):
        raise ValueError('distance must be finite, nonnegative, symmetric, zero diagonal')
    sigma=float(d.std(ddof=0))
    if not sigma>0:raise ValueError('undefined distance scale')
    weights=np.exp(-np.square(d/sigma))
    return threshold_weights(weights),dict(sigma=sigma,ddof=0,scale_population='all matrix elements including diagonal',
        formula='exp(-(D/sigma)^2); weights < 0.1 set to zero',threshold=.1,compute_dtype='float64',cache_dtype='float32',
        diagonal='one; retained in shared weighted adjacency',off_diagonal_zero_distance='valid')


def threshold_weights(weights):
    weights=np.asarray(weights,dtype=np.float64).copy()
    weights[weights<.1]=0
    return weights.astype(np.float32)


def stdn_binary(adjacency,*,non_pems):
    binary=(np.asarray(adjacency)>0).astype(np.float64)
    if non_pems:np.fill_diagonal(binary,0)
    return binary


def topology_diagnostics(adjacency):
    edge=np.asarray(adjacency)>0;edge=edge.copy();np.fill_diagonal(edge,False)
    out=edge.sum(1);incoming=edge.sum(0);n=len(edge)
    return dict(nodes=n,external_directed_edges=int(edge.sum()),density=float(edge.sum()/max(1,n*(n-1))),
                out_degree=dict(min=int(out.min()),max=int(out.max()),mean=float(out.mean()),
                                quantiles=np.quantile(out,[0,.25,.5,.75,1]).tolist()),
                no_outgoing_external_neighbor=int((out==0).sum()),no_incoming_external_neighbor=int((incoming==0).sum()),
                isolated_nodes=int(((out+incoming)==0).sum()),
                weak_components=int(connected_components(sp.csr_matrix(edge),directed=True,connection='weak')[0]),
                strong_components=int(connected_components(sp.csr_matrix(edge),directed=True,connection='strong')[0]))


def laplacian_encoding(adjacency,dimension):
    # Exact source PDFormer/STDN normalization, eig sorting and isolated-node skip.
    a=sp.coo_matrix(adjacency);degree=np.asarray(a.sum(1));isolated=int((degree==0).sum())
    inverse=np.zeros_like(degree,dtype=float);np.power(degree,-.5,out=inverse,where=degree!=0)
    d=sp.diags(inverse.flatten());lap=sp.eye(a.shape[0])-a.dot(d).transpose().dot(d)
    values,vectors=np.linalg.eig(lap.toarray());order=values.argsort()
    result=np.real(vectors[:,order])[:,isolated+1:dimension+isolated+1]
    if result.shape[1]!=dimension:raise ValueError('source Laplacian dimension not supported by node/isolated count')
    if not np.isfinite(result).all():raise ValueError('source Laplacian contains nonfinite values')
    return result.astype(np.float32)


def source_hops(adjacency):
    support=np.asarray(adjacency)>0
    hop=shortest_path(sp.csr_matrix(support),directed=True,unweighted=True)
    return np.minimum(hop,511)


def train_dtw(raw_target,valid,steps_per_day):
    from fastdtw import fastdtw
    raw=np.asarray(raw_target,dtype=np.float64)
    if not np.asarray(valid).all():raise ValueError('native DTW/pattern source has no missing-target rule; explicit review required')
    days=len(raw)//steps_per_day
    if not days:raise ValueError('complete train day absent')
    profile=raw[:days*steps_per_day].reshape(days,steps_per_day,-1).mean(0)
    n=raw.shape[1];dtw=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            distance,_=fastdtw(profile[:,i,None],profile[:,j,None],radius=6)
            dtw[i,j]=dtw[j,i]=distance
    return dtw


def validate_pattern_fit(model):
    if model._X_fit is None or not np.isfinite(model.inertia_) or model.n_iter_<=0:
        raise ValueError('KShape returned no accepted successful fit (_X_fit/inertia/n_iter); failure phase requires native evidence')
    if model.cluster_centers_.shape!=(16,3,1) or not np.isfinite(model.cluster_centers_).all():
        raise ValueError('source KShape produced invalid pattern prototypes')
    parameters=model.get_params()
    if any(parameters[k]!=v for k,v in dict(n_clusters=16,max_iter=5,random_state=2024).items()):
        raise ValueError('unapproved KShape configuration')
    if set(np.unique(model.labels_))!=set(range(16)):raise ValueError('KShape cluster occupancy')
    from .pattern_initialization import array_sha
    if not isinstance(parameters['init'],np.ndarray) or parameters['init'].shape!=(16,3,1):
        raise ValueError('explicit approved candidate initialization required')
    parameters=dict(parameters,init='explicit-candidate-centers',init_array_sha256=array_sha(parameters['init']))
    result=dict(successful_fit=True,inertia=float(model.inertia_),n_iter=int(model.n_iter_),
                cluster_counts=np.bincount(model.labels_,minlength=16).tolist(),parameters=parameters)
    if hasattr(model,'rollback_record'):
        from .kshape_rollback import validate_rollback_record
        result['rollback_policy']=model.rollback_record()
        validate_rollback_record(result['rollback_policy'],result)
    return result


def fit_patterns(standardized_target,legal_train_windows,steps_per_day,*,cand_key_days=14,initialization_stage=None,policy_stage=None):
    from .kshape_rollback import RollbackKShape
    from .pattern_candidates import candidate_indices,build_candidates
    indices=candidate_indices(legal_train_windows,cand_key_days*steps_per_day)
    candidates=build_candidates(standardized_target,indices)
    from .pattern_initialization import select_initial_centers
    centers,initialization=select_initial_centers(candidates)
    if initialization_stage:initialization_stage(centers,initialization)
    model=RollbackKShape(n_clusters=16,max_iter=5,random_state=2024,init=centers)
    try:model.fit(candidates)
    finally:
        if policy_stage:policy_stage(model.rollback_record())
    return model.cluster_centers_,validate_pattern_fit(model)


def train_dtw_and_patterns(raw_target,standardized_target,valid,steps_per_day,legal_train_windows,*,cand_key_days=14,stage=None):
    dtw=train_dtw(raw_target,valid,steps_per_day)
    if stage:stage('dtw_matrix',dtw)
    centers,_=fit_patterns(standardized_target,legal_train_windows,steps_per_day,cand_key_days=cand_key_days)
    if stage:stage('pattern_keys',centers)
    return dtw,centers


def save_fixed_artifacts(destination,adjacency,identity,*,non_pems,pd_adjacency=None):
    if not identity.get('reviewed_migration') or not identity.get('source_sha256'):raise PermissionError('explicit relation provenance/decision required')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    a=np.asarray(adjacency,dtype=np.float32)
    pd_a=a if pd_adjacency is None else np.asarray(pd_adjacency,dtype=np.float32)
    binary=stdn_binary(a,non_pems=non_pems)
    values=dict(base_adjacency=a,gwn_supports=doubletransition(a),d2_supports=d2_doubletransition(a),adj_mx=pd_a,sh_mx=source_hops(pd_a),stdn_adjacency=binary)
    failures={}
    for model,key,graph,dimension in [('PDFormer','laplacian8',pd_a,8),('STDN','laplacian32',binary,32)]:
        try:values[key]=laplacian_encoding(graph,dimension)
        except (ValueError,np.linalg.LinAlgError) as error:failures[model]=str(error)
    for key,value in values.items():np.save(destination/(key+'.npy'),value,allow_pickle=False)
    write_new(destination/'identity.json',dict(identity=identity,code_sha256=sha(__file__),
         consumer_conventions=dict(GraphWaveNet='[R(A),R(A.T)]',D2STGNN='[R(A).T,R(A.T).T]; native dynamic mask +1e-7 unchanged',
             PDFormer='adj_mx is weight; no repeated Gaussian; hops from A>0',STDN='binary without diagonal' if non_pems else 'published directed binary edges'),
         diagnostics=topology_diagnostics(a),stdn_diagnostics=topology_diagnostics(binary),failures=failures,
         files={p.name:sha(p) for p in destination.iterdir()}))
    return {key:dict(path=str(destination/(key+'.npy')),sha256=sha(destination/(key+'.npy'))) for key in values}
