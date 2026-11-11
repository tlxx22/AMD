"""Offline only: 25 sensitivity cells, 15 position cells, five-model ten-node caches."""
import json
from pathlib import Path
import numpy as np
from .contract import read,sha,load_ref,write_new,MODELS,PAIRS
from .analysis import sensitivity_means,case_rows
from m6_ch3_supplement_v10.plots import table,figure
from m6_ch3_test_reporting_v10.analysis import PositionErrors


def completed(config,folder):
    folder=Path(folder);out=folder/'plots';out.mkdir(exist_ok=False);s=load_ref(config['sources_ref']);lineage={};results={};cache_paths={}
    for key,x in s['inherited'].items():
        receipt=load_ref(x['receipt_ref']);r=load_ref(x['result_ref'])
        if receipt['result']!=r or receipt['runtime_sha256']!=x['runtime_ref']['sha256']:raise ValueError('inherited result/runtime')
        load_ref(x['runtime_ref'])
        if sha(x['prediction_ref']['path'])!=x['prediction_ref']['sha256']:raise ValueError('inherited prediction SHA')
        results[key]=r;cache_paths[key]=x['prediction_ref']['path'];lineage[key]=dict(inherited=True,new_forward=0,**x)
    for j in config['jobs']:
        if j['phase'] not in ('S','P','C'):continue
        root=folder/j['id'];r=read(root/'result.json');source=read(root/'source.json');receipt=read(folder/'receipts'/(j['id']+'.json'))
        if receipt['job']!=j or receipt['result']!=r or receipt['runtime_sha256']!=sha(folder/'runtime.json'):raise ValueError('new cache receipt identity')
        for p,d in receipt['files'].items():
            if not (folder/p).resolve().is_relative_to(root.resolve()) or sha(folder/p)!=d:raise ValueError('new cache changed')
        def ref(p):return dict(path=str(p),sha256=sha(p))
        results[j['id']]=r;cache_paths[j['id']]=str(root/'predictions.npz')
        lineage[j['id']]=dict(inherited=False,source=source,receipt_ref=ref(folder/'receipts'/(j['id']+'.json')),result_ref=ref(root/'result.json'),runtime_ref=ref(folder/'runtime.json'),prediction_ref=ref(root/'predictions.npz'))
    cells=[]
    for dataset,hs in [('UrbanEV',(3,6,9,12)),('NP',(24,))]:
        for h in hs:
            for G,R in ((3,4),(5,4),(7,4),(5,2),(5,6)):
                if (G,R)==(5,4):
                    source=s['centers'][f'{dataset}-H{h}'];test=load_ref(source['metrics_ref'])
                    hist=[json.loads(x) for x in Path(source['history_ref']['path']).read_text().splitlines()]
                    val=next(x['validation'] for x in hist if x['epoch']==source['best_epoch']);tm={m:test[m] for m in ('mse','mae')};origin='original formal center; new forward 0';binding=source
                else:
                    key=f'ST-UrbanEV-G{G}-R{R}' if dataset=='UrbanEV' and h==6 else f'ST-{dataset}-H{h}-G{G}-R{R}'
                    source=lineage[key]['source'];val=load_ref(source['metrics_ref'])['validation'];r=results[key];tm={m:r['metrics']['overall_'+m] for m in ('mse','mae')}
                    origin='old H6 training/validation/test reference; new forward 0' if h==6 and dataset=='UrbanEV' else 'new train/validation-best plus separate first test';binding=lineage[key]
                cells.append(dict(dataset=dataset,input='F4' if dataset=='UrbanEV' else 'MS',fold=6 if dataset=='UrbanEV' else 1,H=h,P=1 if dataset=='UrbanEV' else 24,G=G,R=R,seed=2024,best_epoch=source['best_epoch'],producer_commit=source['producer_commit'],validation_mse=val['mse'],validation_mae=val['mae'],test_mse=tm['mse'],test_mae=tm['mae'],origin=origin,source=json.dumps(binding,ensure_ascii=False)))
    means=sensitivity_means(cells);table(out/'sensitivity-25-cells.csv',cells);write_new(out/'sensitivity-25-cells.json',cells);table(out/'sensitivity-means.csv',means);write_new(out/'sensitivity-means.json',means)
    for d in ('UrbanEV','NP'):
        for axis,values in [('G',[3,5,7]),('R',[2,4,6])]:
            rows=[next(r for r in means if r['dataset']==d and r[axis]==v and r['R' if axis=='G' else 'G']==(4 if axis=='G' else 5)) for v in values]
            for split in ('validation','test'):
                for metric in ('mse','mae'):figure(out/f'sensitivity-{d}-{axis}-{split}-{metric}',values,{'UV':[r[split+'_'+metric] for r in rows]},axis,f'{d} '+('four H equal mean ' if d=='UrbanEV' else 'fixed P24 ')+split+' '+metric)
    position=[];summary={}
    for d in ('NP','BE','DE'):
        mr={};base=None
        for m in MODELS:
            key=f'P-{d}-{m}' if m in ('AMD','TimeXer') else f'PT-{d}-{m}';r=results[key];mr[m]=r['metrics']
            with np.load(cache_paths[key],allow_pickle=False) as a:
                current={k:a[k] for k in ('truth','indices','target_record_indices','origin_record_indices')}
                if base is None:base=current
                elif any(not np.array_equal(current[k],base[k]) for k in base):raise ValueError('five-model full test sample identity')
        summary[d]={}
        for metric in ('mse','mae'):
            series={m:mr[m][metric] for m in MODELS};figure(out/f'{d}-five-model-test-position-{metric}',range(1,25),series,'k within fixed P24',metric+' (train-standardized target)')
            for k in range(24):
                r=dict(dataset=d,split='test',position=k+1,metric=metric,**{m:series[m][k] for m in MODELS})
                for m in MODELS[:-1]:
                    b,u=series[m][k],series['UV'][k];r[m+'_improvement_percent']=100*(b-u)/b if b else None;r[m+'_UV_win']=u<b
                position.append(r)
            summary[d][metric]={m:dict(UV_wins=sum(series['UV'][k]<series[m][k] for k in range(24)),baseline_overall=mr[m]['overall_'+metric],UV_overall=mr['UV']['overall_'+metric]) for m in MODELS[:-1]}
    table(out/'five-model-position-all.csv',position);write_new(out/'five-model-position-summary.json',summary)
    plan=load_ref(s['case_plan_ref']);pre=load_ref(s['old_preselection_ref']);selected=load_ref(s['display_selection_ref'])['selected_nodes'];caches={}
    for m in MODELS:
        key=('C-UrbanEV-' if m in ('AMD','TimeXer') else 'CT-UrbanEV-')+m
        with np.load(cache_paths[key],allow_pickle=False) as a:caches[m]={k:a[k] for k in a.files}
    rows=case_rows(caches,plan,pre,selected);table(out/'ten-region-five-model.csv',rows);write_new(out/'ten-region-five-model.json',rows)
    totals={}
    for m,a in caches.items():
        totals[m]={}
        for space,pk,tk in [('standardized','prediction','truth'),('physical','physical_prediction','physical_truth')]:
            meter=PositionErrors(1);meter.add(a[pk],a[tk]);totals[m][space]=meter.result()
    write_new(out/'ten-region-summary.json',totals)
    for node in selected:
        i=plan['nodes'].index(node);sl=slice(i*168,(i+1)*168);series={'truth':caches['UV']['physical_truth'][sl,0,0].tolist()}
        series.update({m:caches[m]['physical_prediction'][sl,0,0].tolist() for m in MODELS})
        figure(out/f'case-{node}-five-model-test',plan['target_times'],series,'target time; rolling H6/P1',f'UrbanEV {node}; original volume; fixed prior display subset')
    write_new(out/'display-selection.json',dict(selected_nodes=selected,origin=s['display_selection_ref'],reranked=False,description='Prior four selected against ModernTCN/TimeMixer; no guarantee versus AMD/TimeXer'))
    write_new(out/'source-ledger.json',dict(evaluations=lineage,centers=s['centers'],case_plan_ref=s['case_plan_ref'],main_index=s['main_index']))
    return dict(job=config['jobs'][-1],sensitivity_cells=25,position_cells=15,case_models=5,case_nodes=10,case_plots=4,cache_only=True,new_forward=0)
