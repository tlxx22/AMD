"""Project AMD family and each baseline's own author source; target-only loss."""
import hashlib
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

from utils.ch3_contract import digest, profile


def build(c, task):
    import torch
    p=profile(c,task); name=task['model']; s=p['structure']
    torch.manual_seed(2024)
    if name in ('AMD','J','N','S'):
        from models.tsAMD_enhanced import AMDEnhanced
        s2=name in ('J','S'); thls=name in ('J','N')
        return AMDEnhanced((p['T'],p['C']),p['pred_len'],s['n_block'],s['dropout'],
            s['patch'],s['k'],s['c'],s['alpha'],None,norm=True,layernorm=s['layernorm'],
            target_idx=p['target_idx'],teb_context_dim=32,task_mode='target_exogenous',aux_idx=p['aux_idx'],
            ch3_contract=p,use_sonnet_mvca=s2,
            sonnet_feature_schema=p['features'] if s2 else None,
            sonnet_schema_fingerprint=digest(p['features']) if s2 else None,
            module_init_seed=2024 if s2 else None,use_target_history_local_shape=thls,
            local_shape_kernel_small=s['kernel_small'] if thls else None,
            local_shape_kernel_large=s['kernel_large'] if thls else None,
            local_shape_init_seed=2024 if thls else None)
    src=c['sources'][name]
    for path,expected in src['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('author source identity mismatch: '+path)
    # Each baseline runs in its own process. Never reuse another model package.
    for module in ('models','layers','model','utils'):
        loaded=sys.modules.get(module)
        if loaded is not None:
            if module=='utils':
                # Our helper is already loaded, but author utils/masking must
                # resolve in the author's root without replacing helper modules.
                if hasattr(loaded,'__path__'): loaded.__path__=[str(Path(src['root'])/'utils')]
            elif module=='models':
                loaded.__path__=[str(Path(src['root'])/'models')]
    sys.path.insert(0,src['root'])
    module=importlib.import_module(src['module'])
    if str(Path(module.__file__).resolve())!=src['entry']:raise RuntimeError('foreign model import')
    options=dict(s,task_name='long_term_forecast',seq_len=p['T'],pred_len=p['pred_len'],
                 enc_in=p['C'],dec_in=p['C'],c_out=p['C'],label_len=0,
                 features='MS' if name=='TimeXer' else 'M')
    # TimeMixer channel-independent reshape needs native C outputs, not c_out=1.
    model=module.Model(SimpleNamespace(**options))
    for mod in list(sys.modules.values()):
        path=getattr(mod,'__file__',None)
        if path and str(Path(path).resolve()).startswith(src['root']+'/'):
            if str(Path(path).resolve()) not in src['files']:
                raise RuntimeError('unbound author import: '+path)
    return model


def target_prediction(model, x, p):
    name=p['model']
    if name=='TimeXer':
        order=p['aux_idx']+[p['target_idx']]
        x=x[:,:,order]
        output=model(x,None,None,None)
    elif name in ('iTransformer','TimeMixer'):output=model(x,None,None,None)
    else:output=model(x)
    aux=output[1] if isinstance(output,tuple) and name in ('AMD','J','N','S') else None
    prediction=output[0] if isinstance(output,tuple) else output
    if name not in ('AMD','J','N','S','TimeXer'):
        prediction=prediction[:,:,p['target_idx']:p['target_idx']+1]
    if tuple(prediction.shape)!=(x.shape[0],p['pred_len'],1):raise ValueError('target output shape mismatch')
    return prediction,aux


def reversible_target_order(p):
    order=p['aux_idx']+[p['target_idx']]
    return order,[order.index(i) for i in range(p['C'])]
