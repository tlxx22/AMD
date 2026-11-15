"""One deterministic candidate-medoid initialization; no fit or pool filtering."""
import hashlib
from pathlib import Path
import numpy as np
from .imports import sha

RULE = 'native-sbd-farthest-first-v1'
TOL = 1e-12  # fixed float64 distance/degeneracy tolerance, not a searched parameter


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()


def library_identity():
    import tslearn
    from tslearn.clustering import kshape
    from tslearn.metrics import cycc
    return dict(tslearn=tslearn.__version__, numpy=np.__version__,
                files={str(Path(m.__file__)):sha(m.__file__) for m in (kshape,cycc)})


def distances_to_center(x, center, norms):
    """Vectorized native normalized_cc FFT/lags/denominator, length three only.

    No centering, variance scaling or alteration of candidate values. A single
    center column avoids a candidate-by-candidate distance matrix.
    """
    center_norm=np.linalg.norm(center)
    denominator=norms*center_norm
    denominator=np.where(denominator < 1e-9, np.inf, denominator)
    fft=np.fft.fft(x[:,:,0],8,axis=1)
    cc=np.fft.ifft(fft*np.conj(np.fft.fft(center[:,0],8)),axis=1).real
    cc=np.concatenate((cc[:,-2:],cc[:,:3]),axis=1)
    return 1.-np.max(cc,axis=1)/denominator


def native_witness(centers):
    """Actual KShape distance and _assign on 16 pool members, never fit/update."""
    from tslearn.clustering import KShape
    centers=np.asarray(centers,dtype=np.float64)
    model=KShape(n_clusters=16,max_iter=5,random_state=2024,init=centers)
    model.cluster_centers_=centers.copy()
    model.norms_=np.linalg.norm(centers,axis=(1,2))
    model.norms_centroids_=model.norms_.copy()
    distances=model._cross_dists(centers)
    if distances.shape!=(16,16) or not np.isfinite(distances).all():raise ValueError('native initialization distance schema')
    off=distances.copy();np.fill_diagonal(off,np.inf)
    margin=off.min(axis=1)-np.diag(distances)
    if np.any(np.abs(np.diag(distances))>TOL) or np.any(margin<=TOL):
        raise ValueError('shape-equivalent or ambiguous initial centers')
    model._assign(centers)  # original EmptyClusterError and argmin semantics
    if not np.array_equal(model.labels_,np.arange(16)):raise ValueError('initial witness ownership changed')
    return dict(labels=model.labels_.tolist(),counts=np.bincount(model.labels_,minlength=16).tolist(),
                min_ownership_margin=float(margin.min()),pairwise_distances=distances.tolist(),
                off_diagonal_ties_with_self=int(np.sum(off<=np.diag(distances)[:,None]+TOL)),
                initial_nonempty=True,full_fit_success=False)


def select_initial_centers(candidates):
    x=np.asarray(candidates,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(3,1) or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite [candidate,3,1] input required')
    norms=np.linalg.norm(x,axis=(1,2))
    centered=np.linalg.norm(x-x.mean(axis=1,keepdims=True),axis=(1,2))
    # Eligibility is only for centers. The unchanged full x is passed to KShape.fit.
    eligible=(norms*norms>=1e-9)&(centered>TOL*np.maximum(1.,norms))
    if np.count_nonzero(eligible)<16:raise ValueError('fewer than 16 nondegenerate center candidates')
    selected=[int(np.flatnonzero(eligible)[0])];nearest=np.full(len(x),np.inf)
    tie_counts=[];selected_distances=[]
    while len(selected)<16:
        nearest=np.minimum(nearest,distances_to_center(x,x[selected[-1]],norms))
        maximum=float(np.max(nearest[eligible]))
        if not np.isfinite(maximum) or maximum<=TOL:raise ValueError('fewer than 16 separable native-SBD shapes')
        tied=np.flatnonzero(eligible & (nearest>=maximum-TOL))
        chosen=int(tied[0]);selected.append(chosen)
        tie_counts.append(int(len(tied)));selected_distances.append(float(nearest[chosen]))
    centers=x[selected].copy()
    witness=native_witness(centers)
    return centers,dict(rule=RULE,tolerance=TOL,n_clusters=16,selected_indices=selected,
        input_candidates=len(x),eligible_center_candidates=int(eligible.sum()),
        zero_or_constant_ineligible=int((~eligible).sum()),tie_counts=tie_counts,
        selected_min_distances=selected_distances,centers_array_sha256=array_sha(centers),
        witness=witness,producer_sha256=sha(__file__),library=library_identity(),
        fit_pool_filtered=False,extra_normalization=False,seed_search=False)
