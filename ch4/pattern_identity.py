"""Adopt a native pattern cache only through a complete explicit identity."""
import math
from pathlib import Path
import numpy as np
from .imports import sha
from .runtime_io import read
from .pattern_candidates import validate_candidate_identity
from .pattern_initialization import RULE,TOL,array_sha,native_witness,library_identity

# Exact historical producers retained without rewriting or claiming the new policy.
LEGACY_SUCCESS_IDS={
    'UrbanEV':'c9b8c62fae79f0fbfb2031bc5673adff3c6ed908208ed3bb1390365278fefdbb',
    'AMS':'bc2d1940e382b9a09c9fd3bd9b599fb42ab25f7573ae416985287725c3b644a7',
    'JHB':'3af2016a3314d2fc76b76ab2f8a3823409ddfa2195ab7874ad6345ebd6cf7ee7',
    'LOA':'0c6d3a916d0dcf462d48737ddd43ab744c7faf0d71d7f48df5c5e2b5cc9206c1'}


def validate_policy_provenance(reference,identity,domain):
    if 'rollback_policy_version' not in identity:
        if reference['sha256']!=LEGACY_SUCCESS_IDS.get(domain) or 'rollback_policy' in identity['fit'] or 'rollback_policy_identity' in identity:
            raise ValueError('unrecognized historical policy-free producer')
        return 'historical-native-update-no-new-policy'
    from .kshape_rollback import POLICY,validate_rollback_record
    if identity['rollback_policy_version']!=POLICY:raise ValueError('rollback version mismatch')
    policy=read(checked_ref(identity['rollback_policy_identity']))
    if policy!=identity['fit']['rollback_policy']:raise ValueError('rollback sidecar mismatch')
    validate_rollback_record(policy,identity['fit'])
    return POLICY


def checked_ref(reference):
    path=Path(reference['path'])
    if sha(path)!=reference['sha256']:raise ValueError('explicit cache reference changed: '+str(path))
    return path


def validate_initialization_identity(reference,candidate_reference,selection,metadata_path):
    identity=read(checked_ref(reference));details=identity['initialization']
    if identity['candidate_identity']!=candidate_reference or identity['domain']!=selection['domain']:
        raise ValueError('initialization candidate identity mismatch')
    if details['rule']!=RULE or details['tolerance']!=TOL or details['producer_sha256']!=sha(Path(__file__).with_name('pattern_initialization.py')):
        raise ValueError('initialization rule/source mismatch')
    if details['library']!=library_identity():raise ValueError('initialization library identity mismatch')
    indices=np.asarray(details['selected_indices'])
    if indices.shape!=(16,) or indices.dtype.kind not in 'iu' or len(set(indices.tolist()))!=16 or np.any(indices<0) or np.any(indices>=selection['candidate_sequences']):
        raise ValueError('initial center indices invalid')
    if details['input_candidates']!=selection['candidate_sequences'] or details['fit_pool_filtered'] or details['extra_normalization'] or details['seed_search']:
        raise ValueError('initialization changed approved pool')
    centers=np.load(checked_ref(identity['centers']),allow_pickle=False)
    if centers.shape!=(16,3,1) or not np.isfinite(centers).all() or array_sha(centers)!=details['centers_array_sha256']:
        raise ValueError('initial centers identity mismatch')
    norms=np.linalg.norm(centers,axis=(1,2))
    centered=np.linalg.norm(centers-centers.mean(axis=1,keepdims=True),axis=(1,2))
    if details['n_clusters']!=16 or np.any(norms*norms<1e-9) or np.any(centered<=TOL*np.maximum(1.,norms)):
        raise ValueError('degenerate initial center identity')
    y=np.load(Path(metadata_path).parent/'y.npy',mmap_mode='r',allow_pickle=False)
    expected=np.stack([y[selection['indices'][int(i)//selection['nodes']]:selection['indices'][int(i)//selection['nodes']]+3,int(i)%selection['nodes']] for i in indices])[:,:,None]
    if not np.array_equal(centers,expected):raise ValueError('initial centers not from approved candidate pool')
    if native_witness(centers)!=details['witness']:raise ValueError('initial ownership witness mismatch')
    return details


def validate_pattern_identity(reference,domain,metadata_path,fixed_reference):
    path=checked_ref(reference);identity=read(path);metadata=read(metadata_path)
    selection=validate_candidate_identity(identity['candidate_identity'],domain,metadata_path)
    if identity['candidate_selection']!=selection:raise ValueError('pattern candidate selection mismatch')
    initialization=validate_initialization_identity(identity['initialization_identity'],identity['candidate_identity'],selection,metadata_path)
    expected=dict(domain=domain,train_rows=metadata['endpoints'][0],node_order=metadata['node_order'],
        metadata_sha256=sha(metadata_path),train_only=True,validation_or_test_fitting=False,
        dtw_radius=6,pattern_length=3,clusters=16,max_iter=5,seed=2024,
        cand_key_days=21 if domain=='PEMS08' else 14,steps_per_day=288 if domain.startswith('PEMS') else 24)
    for key,value in expected.items():
        if identity[key]!=value:raise ValueError('pattern identity mismatch: '+key)
    if identity['fixed_graph_identity']!=fixed_reference:raise ValueError('pattern/fixed graph identity mismatch')
    checked_ref(fixed_reference)
    fit=identity['fit'];params=fit['parameters']
    validate_policy_provenance(reference,identity,domain)
    if fit['successful_fit'] is not True or not math.isfinite(fit['inertia']) or not 1<=fit['n_iter']<=5:
        raise ValueError('native KShape fit not successful')
    if len(fit['cluster_counts'])!=16 or min(fit['cluster_counts'])<1:raise ValueError('unoccupied cluster')
    if any(params[k]!=v for k,v in dict(n_clusters=16,max_iter=5,random_state=2024).items()):raise ValueError('KShape parameter identity')
    if params.get('init')!='explicit-candidate-centers' or params.get('init_array_sha256')!=initialization['centers_array_sha256']:
        raise ValueError('fit initialization binding mismatch')
    expected_count=min(identity['cand_key_days']*identity['steps_per_day'],metadata['endpoints'][0]-23 if domain=='UrbanEV' or domain.startswith('PEMS') else metadata['endpoints'][0]-12)
    if identity['candidate_windows']!=expected_count or sum(fit['cluster_counts'])!=expected_count*metadata['nodes']:
        raise ValueError('pattern candidate coverage')
    if identity['legal_train_windows']!=selection['legal_train_windows']:raise ValueError('legal train range mismatch')
    for original,expected_sha in identity['code_sha256'].items():
        if sha(path.parent/'producer-source'/Path(original).name)!=expected_sha:raise ValueError('producer source snapshot')
    for key,ref in identity['input_files'].items():
        checked_ref(ref)
        if ref['sha256']!=metadata['files'][key+'.npy']:raise ValueError('pattern train prefix identity')
    artifacts=identity['artifacts']
    if set(artifacts)!={'dtw_matrix','pattern_keys'}:raise ValueError('pattern artifact coverage')
    n=metadata['nodes']
    for key,shape in [('dtw_matrix',(n,n)),('pattern_keys',(16,3,1))]:
        value=np.load(checked_ref(artifacts[key]),allow_pickle=False)
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('native pattern array schema')
        if key=='pattern_keys' and 'rollback_policy' in fit:
            from .kshape_rollback import digest
            if digest(value)!=fit['rollback_policy']['final_state']['centers']:
                raise ValueError('pattern output differs from accepted rollback/native state')
    return artifacts
