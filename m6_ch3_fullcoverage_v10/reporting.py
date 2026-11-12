"""Complete machine-readable data; display choices are always explicit arguments."""
from pathlib import Path
import numpy as np
from .contract import read,sha,load_ref,write_new,verify_sources
from .analysis import MODELS,MARKETS,HS,validate_cache,merge_caches,region_rows,explicit_models,PositionErrors
from m6_ch3_supplement_v10.plots import table,figure


def ref(p):return dict(path=str(p),sha256=sha(p))


def resolve_inputs(c,folder):
    from .queue import checked
    s=load_ref(c['sources_ref']);verify_sources(s);inputs={}
    for cell,x in s['inherited'].items():
        # Flags in the sealed source describe its upstream attempt, not this
        # consumer. Preserve them explicitly while marking this use as a reuse.
        entry=dict(x)
        entry['upstream_reference_flags']={k:x[k] for k in ('inherited','new_forward') if k in x}
        entry.update(inherited=True,new_forward=0)
        inputs[cell]=[entry]
    for j in c['jobs'][:-1]:
        r=checked(c,folder,j);out=Path(folder)/j['id'];source=read(out/'source.json')
        inputs.setdefault(j['cell'],[]).append(dict(inherited=False,source=source,job=j,receipt_ref=ref(Path(folder)/'receipts'/(j['id']+'.json')),runtime_ref=ref(Path(folder)/'runtime.json'),result_ref=ref(out/'result.json'),prediction_ref=ref(out/'predictions.npz')))
    if set(inputs)!=set(s['rows']):raise ValueError('all54 source unions required')
    return inputs


def arrays(x):
    p=x['prediction_ref']
    if sha(p['path'])!=p['sha256']:raise ValueError('prediction cache changed')
    with np.load(p['path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}


def cache_metrics(a,bs):
    meter=PositionErrors(a['prediction'].shape[1])
    for i in range(0,len(a['prediction']),bs):meter.add(a['prediction'][i:i+bs],a['truth'][i:i+bs])
    return meter.result()


def completed(c,folder,*,source_folder=None):
    folder=Path(folder);out=folder/'data';out.mkdir(exist_ok=False);(out/'merged').mkdir();s=load_ref(c['sources_ref']);inputs=resolve_inputs(c,folder if source_folder is None else source_folder);lineage={};positions=[];regional=[];wide=[]
    for d in MARKETS:
        common=None;market={}
        for model in MODELS:
            cell=f'P-{d}-{model}';parts=inputs[cell]
            if len(parts)!=1:raise ValueError('EPF full cache exactly once')
            x=parts[0];a=arrays(x);row=s['rows'][cell];validate_cache(row,a,list(range(row['test_arithmetic']['full_test_samples'])))
            keys={k:a[k] for k in ('truth','physical_truth','indices','target_times','origin_times','node_indices')}
            if common is None:common=keys
            elif any(not np.array_equal(common[k],keys[k]) for k in common):raise ValueError('EPF common six-model sample/truth')
            metrics=cache_metrics(a,row['profile']['training']['eval_batch']);saved=load_ref(x['result_ref'])['metrics']
            if metrics!=saved:raise ValueError('cache production-order metrics must reconstruct exactly')
            market[model]=metrics
            for k in range(24):positions.append(dict(dataset=d,model=model,split='test',position=k+1,count=metrics['count_per_position'],sse=metrics['sse'][k],sae=metrics['sae'][k],mse=metrics['mse'][k],mae=metrics['mae'][k],overall_mse=metrics['overall_mse'],overall_mae=metrics['overall_mae'],unit='train-standardized target'))
            lineage[cell]=dict(parts=parts,source=row,cache_ref=x['prediction_ref'],kind='whole_reference' if x['inherited'] else 'new_full_test')
        write_new(out/f'EPF-{d}-six-model-data.json',dict(dataset=d,split='test',P=24,metrics=market,display_models=None,final_figure=False))
    for h in HS:
        plan=load_ref(s['plans'][f'H{h}-all275']);common=None;pernode={}
        for model in MODELS:
            cell=f'C-H{h}-{model}';parts=inputs[cell];row=s['rows'][cell]
            a=merge_caches(row,plan,[arrays(x) for x in parts])
            keys={k:a[k] for k in ('truth','physical_truth','indices','target_times','origin_times','node_indices')}
            if common is None:common=keys
            elif any(not np.array_equal(common[k],keys[k]) for k in common):raise ValueError('UrbanEV common six-model sample/truth')
            if len(parts)==2:
                if h!=6 or model=='PatchTST' or sum(x['inherited'] for x in parts)!=1:raise ValueError('only granted H6 partial reuse')
                path=out/'merged'/(cell+'.npz');np.savez(path,**a);cache=ref(path)
            elif len(parts)==1 and not parts[0]['inherited']:cache=parts[0]['prediction_ref']
            else:raise ValueError('UrbanEV source coverage')
            rows=region_rows(h,model,a,plan);regional.extend(rows)
            for r in rows:
                w=pernode.setdefault(r['node'],dict(H=h,node=r['node'],node_index=r['node_index'],target_start=r['target_start'],target_end=r['target_end'],split='test'))
                for unit in ('standardized','physical'):
                    for metric in ('mse','mae'):w[model+'_'+unit+'_'+metric]=r[unit+'_'+metric]
            lineage[cell]=dict(parts=parts,source=row,cache_ref=cache,plan_ref=s['plans'][f'H{h}-all275'],kind='partial_reference_plus_missing' if len(parts)==2 else 'new_full_week')
        wide.extend(pernode.values())
    if len(lineage)!=54 or len(regional)!=6600 or len(wide)!=1100 or len(positions)!=720:raise ValueError('full data coverage')
    table(out/'region-metrics-6600.csv',regional);write_new(out/'region-metrics-6600.json',regional);table(out/'region-wide-1100.csv',wide)
    table(out/'EPF-position-all.csv',positions);write_new(out/'EPF-position-all.json',positions)
    write_new(out/'source-ledger.json',lineage)
    write_new(out/'display-policy.json',dict(regions_selected=[],final_region_models=None,final_EPF_models=None,automatic_ranking=False,notes='Region H/node/model lists and final EPF models are user/ChatGPT decisions after data. Prior four displays remain historical; no new eight-case promise.'))
    return dict(job=c['jobs'][-1],logical_cells=54,region_metric_cells=6600,region_wide_rows=1100,position_cells=30,model_calls=0,regions_selected=[],whole_references=15,partial_references=5)


def region_plot(data_folder,output,*,H,node_ids,models):
    """Offline caller MUST specify H, nodes and ordered curves; never searches."""
    models=explicit_models(models);root=Path(data_folder);ledger=read(root/'source-ledger.json')
    if H not in HS or not node_ids or len(set(node_ids))!=len(node_ids):raise ValueError('explicit H and unique node_ids')
    plan=load_ref(ledger[f'C-H{H}-UV']['plan_ref'])
    if not set(node_ids)<=set(plan['nodes']):raise ValueError('node outside dataset')
    # Validate all inputs before creating any output; no model/loader access.
    caches={m:arrays(dict(prediction_ref=ledger[f'C-H{H}-{m}']['cache_ref'])) for m in models}
    out=Path(output);out.mkdir(exist_ok=False)
    for node in node_ids:
        i=plan['nodes'].index(node);sl=slice(i*168,(i+1)*168);base=caches[models[0]];series={'truth':base['physical_truth'][sl,0,0].tolist()}
        for m in models:
            if not np.array_equal(caches[m]['physical_truth'][sl],base['physical_truth'][sl]):raise ValueError('display common truth')
            series[m]=caches[m]['physical_prediction'][sl,0,0].tolist()
        figure(out/f'UrbanEV-H{H}-node{node}',plan['target_times'],series,'target time; rolling P1',f'UrbanEV H{H} node {node}; original volume; explicit display')
    write_new(out/'selection.json',dict(H=H,node_ids=node_ids,models=models,automatic_selection=False,new_forward=0))


def epf_plot(data_folder,output,*,market,models):
    models=explicit_models(models)
    if market not in MARKETS:raise ValueError('explicit market')
    data=read(Path(data_folder)/f'EPF-{market}-six-model-data.json');out=Path(output);out.mkdir(exist_ok=False)
    for metric in ('mse','mae'):figure(out/f'{market}-{metric}',range(1,25),{m:data['metrics'][m][metric] for m in models},'position k within fixed P24',market+' '+metric+'; train-standardized target')
    write_new(out/'selection.json',dict(market=market,models=models,new_forward=0,automatic_selection=False))
