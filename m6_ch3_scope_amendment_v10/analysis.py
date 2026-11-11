"""Fixed complete-grid averaging and five-model cache statistics; no model calls."""
import numpy as np
from .contract import PAIRS, MODELS
from m6_ch3_test_reporting_v10.analysis import PositionErrors


def sensitivity_means(rows):
    pairs=((3,4),(5,4),(7,4),(5,2),(5,6))
    expected={(d,h,G,R) for d,hs in [('UrbanEV',(3,6,9,12)),('NP',(24,))] for h in hs for G,R in pairs}
    keys=[(r['dataset'],r['H'],r['G'],r['R']) for r in rows]
    if len(keys)!=25 or len(set(keys))!=25 or set(keys)!=expected:raise ValueError('complete unique 25 cells required; missing H never shrinks denominator')
    out=[]
    for d,hs in [('UrbanEV',(3,6,9,12)),('NP',(24,))]:
        for G,R in pairs:
            selected=[next(r for r in rows if (r['dataset'],r['H'],r['G'],r['R'])==(d,h,G,R)) for h in hs]
            if any(r['fold']!=(6 if d=='UrbanEV' else 1) or r['P']!=(1 if d=='UrbanEV' else 24) or r['seed']!=2024 for r in selected):raise ValueError('wrong cell identity')
            values={split+'_'+metric:float(np.mean([r[split+'_'+metric] for r in selected])) for split in ('validation','test') for metric in ('mse','mae')}
            if not np.isfinite(list(values.values())).all():raise ValueError('nonfinite sensitivity')
            out.append(dict(dataset=d,G=G,R=R,H=list(hs),aggregation='unrounded equal H arithmetic mean; not pooled SSE or mean percentage',**values))
    return out


def case_rows(caches,plan,pre,selected_nodes):
    if set(caches)!=set(MODELS) or selected_nodes!=['1162','1083','1120','969']:raise ValueError('five models and locked original four required')
    ref=caches['UV'];rows=[]
    for m,a in caches.items():
        for k in ('truth','physical_truth','indices','target_record_indices','origin_record_indices','node_indices'):
            if not np.array_equal(a[k],ref[k]):raise ValueError('five-model common case '+k)
        if len(a['truth'])!=1680:raise ValueError('fixed ten x 168')
    for i,(node,idx) in enumerate(zip(plan['nodes'],plan['node_indices'])):
        r=dict(node=node,node_index=idx,train_mean=pre['train_mean'][idx],train_std=pre['train_std'][idx],split='test',target_start=plan['target_times'][0],target_end=plan['target_times'][-1],selected_for_plot=node in selected_nodes,selection_origin='original two-baseline comparison; no reranking against added baselines')
        sl=slice(i*168,(i+1)*168)
        for m,a in caches.items():
            for prefix,pk,tk in [('', 'prediction','truth'),('physical_','physical_prediction','physical_truth')]:
                meter=PositionErrors(1);meter.add(a[pk][sl],a[tk][sl]);v=meter.result()
                for metric in ('mse','mae'):r[m+'_'+prefix+metric]=v['overall_'+metric]
        for m in MODELS[:-1]:
            for metric in ('mse','mae'):
                b,u=r[m+'_'+metric],r['UV_'+metric];r[m+'_'+metric+'_win']=u<b
                r[m+'_'+metric+'_difference']=u-b;r[m+'_'+metric+'_improvement_percent']=100*(b-u)/b if b else None
        rows.append(r)
    return rows
