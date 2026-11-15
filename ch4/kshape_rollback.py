"""Local KShape adaptation: only post-update EmptyClusterError may terminate safely.

The normal loop follows tslearn 0.6.3 kshape.py:159-192, including its original
inertia rollback. Distance/extraction/update/post-fit implementations are inherited.
"""
import hashlib
import copy
import numpy as np
from tslearn.clustering import KShape
from tslearn.clustering.utils import EmptyClusterError
from .imports import sha

POLICY = 'post-update-empty-rollback-stop-v1'
NATIVE_SHA = '371d7d56af5a44a0498190d9da0a75a0de2f492ce8e356172860ad2f73e9dd26'
STATE = ('cluster_centers_', 'norms_centroids_', 'labels_', 'inertia_', '_iter', 'n_iter_')


def digest(value):
    a=np.ascontiguousarray(value)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def capture(model):
    return {key:copy.deepcopy(getattr(model,key)) for key in STATE if hasattr(model,key)}


def restore(model,state):
    for key in STATE:
        if key in state:setattr(model,key,copy.deepcopy(state[key]))
        elif hasattr(model,key):delattr(model,key)


def summary(state):
    counts=np.bincount(state['labels_'],minlength=16) if state.get('labels_') is not None else np.zeros(16,dtype=int)
    return dict(centers=digest(state['cluster_centers_']),norms=digest(state['norms_centroids_']),
        labels=digest(state['labels_']),inertia=float(state['inertia_']),cluster_counts=counts.tolist(),
        centers_finite=bool(np.isfinite(state['cluster_centers_']).all()),
        norms_finite=bool(np.isfinite(state['norms_centroids_']).all()),
        inertia_finite=bool(np.isfinite(state['inertia_'])),
        iter_present='_iter' in state,iter_value=state.get('_iter'),n_iter=state.get('n_iter_'))


class RollbackKShape(KShape):
    def fit(self,X,y=None):
        from tslearn.clustering import kshape
        if sha(kshape.__file__)!=NATIVE_SHA:raise ValueError('unreviewed native KShape source')
        self.policy_attempts_=[]
        return super().fit(X,y)

    def _cross_dists(self,X):
        distances=super()._cross_dists(X)
        if not np.isfinite(distances).all():raise FloatingPointError('nonfinite native SBD distances')
        return distances

    def _finite_centers(self):
        if not np.isfinite(self.cluster_centers_).all() or not np.isfinite(self.norms_centroids_).all():
            raise FloatingPointError('nonfinite native center update')

    def _legal_state(self):
        self._finite_centers()
        if not np.isfinite(self.inertia_):raise FloatingPointError('nonfinite native inertia')
        if set(np.unique(self.labels_))!=set(range(self.n_clusters)):
            raise ValueError('saved/restored state is not fully occupied')

    def rollback_record(self):
        from tslearn.clustering import kshape
        result=dict(version=POLICY,adapter_sha256=sha(__file__),native_source_sha256=sha(kshape.__file__),
                    attempts=copy.deepcopy(getattr(self,'policy_attempts_',[])))
        if getattr(self,'_X_fit',None) is not None and getattr(self,'n_iter_',0)>0:
            try:self._legal_state()
            except (ValueError,FloatingPointError):result['final_state_invalid']=True
            else:result['final_state']=summary(capture(self))
        return result

    def _fit_one_init(self,X,rs):
        if not hasattr(self,'policy_attempts_'):self.policy_attempts_=[]
        record=dict(attempt=len(self.policy_attempts_)+1,triggered=False,early_terminated=False,
                    legal_updates_completed=0,retained_legal_updates=0,attempted_updates=0,stop_reason='not_completed')
        self.policy_attempts_.append(record)
        # Unchanged native initialization and first assignment, outside the new catch.
        if hasattr(self.init,'__array__'):self.cluster_centers_=self.init.copy()
        elif self.init=='random':self.cluster_centers_=X[rs.choice(X.shape[0],self.n_clusters)].copy()
        else:raise ValueError("Value %r for parameter 'init' is invalid" % self.init)
        self.norms_centroids_=np.linalg.norm(self.cluster_centers_,axis=(1,2))
        self._assign(X)
        self._legal_state()
        old_inertia=np.inf
        for it in range(self.max_iter):
            previous=capture(self)
            old_cluster_centers=self.cluster_centers_.copy()
            record['attempted_updates']=it+1
            self._update_centroids(X)  # native extraction/update; exceptions are not caught
            self._finite_centers()
            try:
                self._assign(X)  # only this post-update native empty check is adapted
            except EmptyClusterError:
                self._finite_centers()
                if not np.isfinite(self.inertia_):raise FloatingPointError('nonfinite state cannot be accepted by rollback')
                counts=np.bincount(self.labels_,minlength=self.n_clusters)
                empty=np.flatnonzero(counts==0).tolist()
                if not empty:raise  # do not reinterpret an unrelated exception as empty assignment
                record.update(triggered=True,trigger_iteration=it+1,empty_clusters=empty,
                    before=summary(previous),failed=summary(capture(self)),failed_update_inertia_computed=False)
                restore(self,previous)
                self._legal_state()
                record['restored']=summary(capture(self))
                if record['restored']!=record['before']:raise AssertionError('incomplete empty-cluster state restoration')
                if record['legal_updates_completed']==0:
                    record['stop_reason']='first_update_empty_no_legal_update'
                    raise  # inherited fit failure/retry behavior remains; initial allocation is not success
                self._iter=record['legal_updates_completed']
                record.update(early_terminated=True,stop_reason='post_update_empty_rollback',
                              retained_legal_updates=record['legal_updates_completed'])
                return self
            self._legal_state()
            record['legal_updates_completed']+=1
            record['retained_legal_updates']=record['legal_updates_completed']
            if self.verbose:print('%.3f' % self.inertia_,end=' --> ')
            # Exact native normal convergence/inertia rollback path; not the new policy.
            if np.abs(old_inertia-self.inertia_)<self.tol or (old_inertia-self.inertia_<0):
                self.cluster_centers_=old_cluster_centers
                self._assign(X)
                self._legal_state()
                record.update(stop_reason='native_inertia_rollback',retained_legal_updates=record['legal_updates_completed']-1)
                break
            old_inertia=self.inertia_
        else:record['stop_reason']='native_max_iter'
        if self.verbose:print('')
        self._iter=it+1  # unchanged native n_iter semantics on normal paths
        return self


def validate_rollback_record(policy,fit):
    if policy['version']!=POLICY or policy['adapter_sha256']!=sha(__file__) or policy['native_source_sha256']!=NATIVE_SHA:
        raise ValueError('rollback policy/source identity')
    if not policy['attempts']:raise ValueError('missing rollback attempt record')
    last=policy['attempts'][-1];legal=last['legal_updates_completed'];attempted=last['attempted_updates']
    if not 1<=legal<=attempted<=5:raise ValueError('invalid legal update accounting')
    final=policy['final_state']
    if final['centers']['shape']!=[16,3,1] or not all(final[k] for k in ('centers_finite','norms_finite','inertia_finite')):
        raise ValueError('invalid final rollback state')
    if final['cluster_counts']!=fit['cluster_counts'] or final['inertia']!=fit['inertia'] or final['n_iter']!=fit['n_iter']:
        raise ValueError('rollback final fit state mismatch')
    if last['triggered']:
        if not last['early_terminated'] or last['stop_reason']!='post_update_empty_rollback':raise ValueError('rollback is not native convergence')
        if last['trigger_iteration']!=attempted or attempted!=legal+1 or fit['n_iter']!=legal or last['retained_legal_updates']!=legal:
            raise ValueError('failed update counted as completed')
        if not last['empty_clusters'] or last['before']!=last['restored']:raise ValueError('missing complete restoration proof')
        for key in ('centers','norms','labels','inertia','cluster_counts'):
            if final[key]!=last['restored'][key]:raise ValueError('final state differs from restored state')
    elif last['early_terminated'] or last['stop_reason'] not in ('native_inertia_rollback','native_max_iter'):
        raise ValueError('normal path termination mislabelled')
    elif attempted!=legal or fit['n_iter']!=attempted or last['retained_legal_updates']!=legal-int(last['stop_reason']=='native_inertia_rollback'):
        raise ValueError('native normal update accounting changed')
