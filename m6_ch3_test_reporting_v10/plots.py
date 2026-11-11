"""Cache-only reporting; all points/nodes retained before four display cases."""
from pathlib import Path
import json
import numpy as np
from .contract import load_ref, read, write_new, sha
from .analysis import PositionErrors, rank_cases
from m6_ch3_supplement_v10.plots import table, figure


def completed(config, folder):
    folder=Path(folder);out=folder/'plots';out.mkdir(exist_ok=False)
    ledger=load_ref(config['sources_ref']);results={};lineage={}
    for job in config['jobs']:
        receipt=read(folder/'receipts'/(job['id']+'.json'))
        if receipt['job']!=job or receipt['runtime_sha256']!=sha(folder/'runtime.json'):raise ValueError('cache receipt identity')
        for name,s in receipt['files'].items():
            p=(folder/name).resolve()
            if not p.is_relative_to((folder/job['id']).resolve()) or sha(p)!=s:raise ValueError('cache changed')
        r=read(folder/job['id']/'result.json');results[job['id']]=r
        if r!=receipt['result']:raise ValueError('cache result/receipt mismatch')
        lineage[job['id']]=dict(producer_commit=r['producer_commit'],weights_ref=r['weights_ref'],best_epoch=r['best_epoch'],
            original_source=ledger['rows'][job['source_key']],evaluation_runtime_ref=dict(path=str(folder/'runtime.json'),sha256=sha(folder/'runtime.json')),
            evaluation_receipt_ref=dict(path=str(folder/'receipts'/(job['id']+'.json')),sha256=sha(folder/'receipts'/(job['id']+'.json'))),
            result_ref=dict(path=str(folder/job['id']/'result.json'),sha256=sha(folder/job['id']/'result.json')),split='test',new_test_access=True)
    write_new(out/'source-ledger.json',lineage)
    sensitivity=[]
    for dataset in ('UrbanEV','PJM'):
        for G,R in ((3,4),(5,4),(7,4),(5,2),(5,6)):
            if (G,R)==(5,4):
                source=ledger['centers'][dataset];test=load_ref(source['metrics_ref'])
                hist=[json.loads(s) for s in Path(source['history_ref']['path']).read_text().splitlines()]
                val=next(h['validation'] for h in hist if h['epoch']==source['best_epoch'])
                provenance='original formal test reference; new forward 0';ref=source['metrics_ref'];test_metrics={k:test[k] for k in ('mse','mae','sse','sae','elements')}
            else:
                key=f'ST-{dataset}-G{G}-R{R}';source=ledger['rows'][key];val=load_ref(source['metrics_ref'])['validation'];r=results[key]
                test_metrics=dict(mse=r['metrics']['overall_mse'],mae=r['metrics']['overall_mae'],sse=sum(r['metrics']['sse']),sae=sum(r['metrics']['sae']),elements=r['elements'])
                provenance='first test of frozen validation-best sensitivity';ref=lineage[key]['result_ref']
            sensitivity.append(dict(dataset=dataset,G=G,R=R,best_epoch=source['best_epoch'],validation_mse=val['mse'],validation_mae=val['mae'],test_mse=test_metrics['mse'],test_mae=test_metrics['mae'],test_sse=test_metrics['sse'],test_sae=test_metrics['sae'],test_elements=test_metrics['elements'],provenance=provenance,source=json.dumps(source,ensure_ascii=False),test_result_ref=json.dumps(ref,ensure_ascii=False)))
        for axis,values in [('G',[3,5,7]),('R',[2,4,6])]:
            selected=[next(r for r in sensitivity if r['dataset']==dataset and r[axis]==v and r['R' if axis=='G' else 'G']==(4 if axis=='G' else 5)) for v in values]
            for split in ('validation','test'):
                for metric in ('mse','mae'):
                    figure(out/f'sensitivity-{dataset}-{axis}-{split}-{metric}',values,{'UV':[r[split+'_'+metric] for r in selected]},axis,split+' '+metric+' (train-standardized)')
    table(out/'sensitivity-all-points.csv',sensitivity);write_new(out/'sensitivity-all-points.json',sensitivity)
    positions=[]
    for dataset in ('NP','BE','DE'):
        for metric in ('mse','mae'):
            series={m:results[f'PT-{dataset}-{m}']['metrics'][metric] for m in ('ModernTCN','TimeMixer','UV')}
            figure(out/f'{dataset}-test-position-{metric}',range(1,25),series,'k within fixed P24',metric+' (train-standardized target)')
            for k in range(24):
                row=dict(dataset=dataset,split='test',position=k+1,metric=metric,**{m:v[k] for m,v in series.items()})
                for m in ('ModernTCN','TimeMixer'):
                    b=series[m][k];u=series['UV'][k];row[m+'_improvement_percent']=100*(b-u)/b if b else None;row[m+'_UV_win']=u<b
                positions.append(row)
    table(out/'test-position-all.csv',positions)
    write_new(out/'test-position-summary.json',{d:{m:dict(wins={metric:sum(results[f'PT-{d}-UV']['metrics'][metric][k]<results[f'PT-{d}-{m}']['metrics'][metric][k] for k in range(24)) for metric in ('mse','mae')},baseline=results[f'PT-{d}-{m}']['metrics'],UV=results[f'PT-{d}-UV']['metrics']) for m in ('ModernTCN','TimeMixer')} for d in ('NP','BE','DE')})
    pre=load_ref(config['preselection_ref']);plan=read(folder/'case-plan.json');caches={}
    for m in ('ModernTCN','TimeMixer','UV'):
        with np.load(folder/f'CT-UrbanEV-{m}'/'predictions.npz',allow_pickle=False) as a:caches[m]={k:a[k] for k in a.files}
    rows=[]
    for i,(node,idx) in enumerate(zip(plan['nodes'],plan['node_indices'])):
        item=dict(node=node,node_index=idx,train_mean=pre['train_mean'][idx],train_std=pre['train_std'][idx],target_start=plan['target_times'][0],target_end=plan['target_times'][-1],split='test',metric_space='train-standardized')
        for m,c in caches.items():
            sl=slice(i*168,(i+1)*168)
            if not np.array_equal(c['truth'][sl],caches['UV']['truth'][sl]):raise ValueError('case truth alignment')
            for prefix,pk,tk in [('', 'prediction','truth'),('physical_','physical_prediction','physical_truth')]:
                meter=PositionErrors(1);meter.add(c[pk][sl],c[tk][sl]);r=meter.result()
                item[m+'_'+prefix+'mse']=r['overall_mse'];item[m+'_'+prefix+'mae']=r['overall_mae']
        rows.append(item)
    ranked=rank_cases(rows);table(out/'ten-region-comparison.csv',ranked);write_new(out/'ten-region-comparison.json',ranked)
    for row in ranked[:4]:
        i=plan['node_indices'].index(row['node_index']);sl=slice(i*168,(i+1)*168)
        series={'truth':caches['UV']['physical_truth'][sl,0,0].tolist()}
        series.update({m:c['physical_prediction'][sl,0,0].tolist() for m,c in caches.items()})
        figure(out/f'case-{row["node"]}-test',plan['target_times'],series,'test target time; rolling H6/P1',f'UrbanEV node {row["node"]}; selected from random10; wins {row["strict_wins"]}/4; original volume')
    write_new(out/'display-selection.json',dict(rule=pre['ranking_rule'],selected_nodes=[r['node'] for r in ranked[:4]],description='Typical/advantage display subset of the ten sampled middle-demand nodes; not a random representative sample of all UrbanEV',new_forward=0))
    return dict(model_evaluations=20,center_references=2,sensitivity_points=10,position_markets=3,case_nodes=10,case_plots=4,new_test_accesses=20,new_training=0,cache_only_plotting=True)
