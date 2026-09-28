"""Read-only diagnostic sidecars, independent of any numerical admission policy.

No model calls, optimizer calls, backward calls, randomness or backend changes.
Legacy FullNumericStateWriter's 'bounded' layout tag does not grant a tolerance.
"""
import hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def rng_value(generator):
    import random,numpy as np,torch
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
        torch_cpu=torch.get_rng_state().tolist(),
        torch_cuda=[x.tolist()for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
        data_generator=generator.get_state().tolist())

class DiagnosticCapture:
    def __init__(self,model,opt,out,digest_state,generator):
        from ch3_runner import FullNumericStateWriter
        self.model,self.opt,self.generator=model,opt,generator
        self.root=Path(out)/'diagnostic-state';self.root.mkdir(exist_ok=False)
        # Initial state has no gradients/moments; keep its own schema.
        self.initial_dir=self.root/'initial';self.initial_dir.mkdir()
        self.steps_dir=self.root/'steps';self.steps_dir.mkdir()
        layout={'id':'urban-diagnostic-raw-state-v1'}
        self.initial_writer=FullNumericStateWriter(model,opt,self.initial_dir,digest_state,layout)
        self.step_writer=FullNumericStateWriter(model,opt,self.steps_dir,digest_state,layout)
        self.rows=[]
    def capture(self,step):
        from ch3_runner import dump
        before=rng_value(self.generator)
        writer=self.initial_writer if step==0 else self.step_writer
        row=writer.capture(step)
        after=rng_value(self.generator)
        if before!=after:raise RuntimeError('diagnostic capture consumed RNG')
        meta=dict(step=step,rng=before,
            absent_gradients=[n for n,p in self.model.named_parameters()if p.grad is None],
            empty_optimizer_state=[n for n,p in self.model.named_parameters()if not self.opt.state.get(p,{})])
        meta_path=self.root/f'meta-{step}.json';dump(meta_path,meta)
        row.update(meta_file=str(meta_path),meta_sha=sha(meta_path))
        self.rows.append(row);dump(self.root/'index.json',dict(diagnostic_only=True,admission_granted=False,points=self.rows))
        return row

def tensor_metrics(reference,observed,*,exact=False,near_zero=1e-12):
    """Symmetric elementwise relative error; no epsilon-added/floored denominator.
    Values at or below near_zero have no reported relative ratio.
    near_zero is reporting metadata, NEVER an acceptance threshold.
    """
    import numpy as np
    a=np.asarray(reference);b=np.asarray(observed)
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('dtype/shape mismatch')
    finite=bool(np.isfinite(a).all()and np.isfinite(b).all())
    changed=a!=b;count=int(np.count_nonzero(changed))
    out=dict(dtype=str(a.dtype),shape=list(a.shape),elements=int(a.size),different_elements=count,
        finite=finite,exact_required=exact,exact_equal=bool(a.tobytes()==b.tobytes()),
        denominator='max(abs(reference_i),abs(observed_i)); ratios omitted where denominator <= 1e-12',near_zero_threshold=near_zero)
    if not finite:
        out.update(reference_max_abs=None,observed_max_abs=None,max_abs_diff=None,max_relative_diff=None,near_zero_elements=None,near_zero_changed=None)
        return out
    if exact and a.dtype.kind in "biu":
        # Python integers avoid float64 rounding of int64 state.
        av=[int(x)for x in a.flat];bv=[int(x)for x in b.flat]
        dif=[abs(x-y)for x,y in zip(av,bv)];den=[max(abs(x),abs(y))for x,y in zip(av,bv)]
        out.update(reference_max_abs=max(map(abs,av),default=0),observed_max_abs=max(map(abs,bv),default=0),
            max_abs_diff=max(dif,default=0),max_relative_diff=None,near_zero_elements=sum(x==0 for x in den),near_zero_changed=0)
    else:
        aa=a.astype('float64');bb=b.astype('float64');diff=np.abs(aa-bb);den=np.maximum(np.abs(aa),np.abs(bb));valid=den>near_zero
        out.update(reference_max_abs=float(np.max(np.abs(aa)))if aa.size else 0.,observed_max_abs=float(np.max(np.abs(bb)))if bb.size else 0.,
            max_abs_diff=float(np.max(diff))if diff.size else 0.,max_relative_diff=float(np.max(diff[valid]/den[valid]))if valid.any()else None,
            near_zero_elements=int((~valid).sum()),near_zero_changed=int(np.count_nonzero(changed&~valid)))
    return out

def checked(row,root):
    root=Path(root).resolve()
    for file_key,sha_key in [('schema_file','schema_sha'),('data_file','data_sha'),('meta_file','meta_sha')]:
        p=Path(row[file_key])
        if not p.resolve().is_relative_to(root)or p.is_symlink()or sha(p)!=row[sha_key]:raise ValueError('diagnostic payload binding/path')
    schema=read(row['schema_file'])
    if Path(row['data_file']).stat().st_size!=schema['total_bytes']or row['bytes']!=schema['total_bytes']:raise ValueError('payload size')
    return schema,read(row['meta_file'])

def compare_point(x,y,root):
    import numpy as np
    sx,mx=checked(x,root);sy,my=checked(y,root)
    if x['step']!=y['step']:raise ValueError('step identity')
    # Schema/optimizer groups/non-tensor state mismatches are explicit evidence.
    metadata_equal=(sx==sy and mx==my)
    ex={e['path']:e for e in sx['entries']};ey={e['path']:e for e in sy['entries']};rows=[]
    dtypes={'torch.float32':'<f4','torch.float64':'<f8','torch.float16':'<f2','torch.int64':'<i8','torch.int32':'<i4','torch.int16':'<i2','torch.int8':'i1','torch.uint8':'u1','torch.bool':'?'}
    with open(x['data_file'],'rb')as fa,open(y['data_file'],'rb')as fb:
        for name in sorted(set(ex)|set(ey)):
            a,b=ex.get(name),ey.get(name)
            if a is None or b is None or any(a[k]!=b[k]for k in ('dtype','shape','numel','nbytes','mode')):
                rows.append(dict(name=name,structure_equal=False,reference=a,observed=b));continue
            if a['dtype']not in dtypes:raise ValueError('unsupported diagnostic dtype: '+a['dtype'])
            fa.seek(a['offset']);fb.seek(b['offset']);ra=fa.read(a['nbytes']);rb=fb.read(b['nbytes'])
            va=np.frombuffer(ra,dtype=dtypes[a['dtype']]).reshape(a['shape']);vb=np.frombuffer(rb,dtype=dtypes[b['dtype']]).reshape(b['shape'])
            exact=a['mode']=='exact'
            metrics=tensor_metrics(va,vb,exact=exact)
            metrics.update(name=name,source_dtype=a['dtype'],structure_equal=True,byte_equal=ra==rb)
            rows.append(metrics)
    return dict(step=x['step'],metadata_equal=metadata_equal,rng_equal=mx.get('rng')==my.get('rng'),
        param_groups_equal=sx.get('optimizer_groups')==sy.get('optimizer_groups'),
        optimizer_non_tensor_equal=sx.get('optimizer_non_tensor_state')==sy.get('optimizer_non_tensor_state'),
        absent_gradients_equal=mx.get('absent_gradients')==my.get('absent_gradients'),
        missing_optimizer_state_equal=mx.get('empty_optimizer_state')==my.get('empty_optimizer_state'),
        tensors=rows,identical=metadata_equal and all(r.get('byte_equal',False)for r in rows))

def compare_traces(reference,actual,root):
    for k in ('id','profile_sha','initial_rng','batch_ids','steps'):
        if reference[k]!=actual[k]:raise ValueError('diagnostic identity/RNG/batch mismatch: '+k)
    x,y=reference['urban_diagnostic_trace'],actual['urban_diagnostic_trace']
    if [p['step']for p in x]!=list(range(7))or [p['step']for p in y]!=list(range(7)):raise ValueError('seven state captures required')
    points=[compare_point(a,b,root)for a,b in zip(x,y)]
    first=next((p for p in points if not p['identical']),None)
    return dict(diagnostic_only=True,admission_granted=False,first_different_step=first['step']if first else None,
        earliest_different_tensors=[r['name']for r in first['tensors']if not r.get('byte_equal',False)]if first else [],
        points=points,scalar_losses_equal=[p['loss']for p in reference['trajectory']]==[p['loss']for p in actual['trajectory']],
        validation_equal=reference['validation']==actual['validation'],final_rng_equal=reference['final_rng']==actual['final_rng'],
        inference_limit='Earliest stored state is post-update; this does not uniquely identify an operator. No tolerance/admission is inferred.')
