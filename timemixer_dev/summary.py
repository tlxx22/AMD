"""Read-only saved-result aggregation; complete coverage before scientific gates."""
from pathlib import Path
import json
from .artifacts import audit

def summarize(config,root,attempt,main_metric=None):
    rows={}
    if attempt is None:
        from .worker import RUN_NAMESPACE
        registry=(Path(root) if config['purpose']=='synthetic' else RUN_NAMESPACE)/'completed'
        paths=[]
        if registry.exists():
            for p in registry.iterdir():
                if p.name.endswith('.tmp'):continue
                value=json.loads(p.read_text())
                if value.get('status')=='completed' and value['receipt']['manifest']['identity']['config_sha']==__import__('utils.ch3_contract',fromlist=['digest']).digest(config):paths.append(Path(value['receipt']['path']))
    else:
        folder=Path(root)/attempt/'runs';paths=list(folder.iterdir()) if folder.exists() else []
    recorded_metrics=set()
    for p in paths:
            receipt=audit(p);m=receipt['metrics'];task=receipt['manifest']['identity']['task']
            if task['id'] in rows:raise ValueError('duplicate result identity')
            if receipt['manifest']['identity']['config_sha'] != __import__('utils.ch3_contract',fromlist=['digest']).digest(config):raise ValueError('summary configuration identity')
            rows[task['id']]=(task,m)
            recorded_metrics.add(receipt['manifest']['identity']['gate_main_metric'])
    if not set(rows)<={t['id'] for t in config['tasks']}:raise ValueError('foreign summary task identity')
    if main_metric is not None and recorded_metrics!={main_metric}:raise ValueError('main metric differs from recorded startup contracts')
    def mean(items):
        return {metric:sum(m[metric] for m in items)/len(items) for metric in ('mse','mae')} if items else None
    urban={};inputs={};domains={}
    for arm in ('B','U','V','UV'):
        items=[v for k,v in config['existing_B'].items() if '|UrbanEV|' in k] if arm=='B' else [m for t,m in rows.values() if t['goal']==1 and t['arm']==arm]
        urban[arm]=dict(count=len(items),mean=mean(items),complete=len(items)==8)
    for variant in ('F1','F2','F3','F4'):
        items=[m for t,m in rows.values() if t['dataset']=='UrbanEV' and t['arm']=='UV' and t['input_variant']==variant]
        inputs[variant]=dict(count=len(items),mean=mean(items),complete=len(items)==8)
    for ds in ('ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange','PJM','NP','BE','FR','DE'):
        tasks=[t for t in config['tasks'] if t['goal']==2 and t['dataset']==ds];items=[rows[t['id']][1] for t in tasks if t['id'] in rows];base=[config['existing_B'][t['baseline_cell_id']] for t in tasks]
        domains[ds]=dict(count=len(items),required=len(tasks),candidate=mean(items),B=mean(base),complete=len(items)==len(tasks))
    complete=len(rows)==len(config['tasks']) and all(x['complete'] for x in urban.values()) and all(x['complete'] for x in inputs.values()) and all(x['complete'] for x in domains.values())
    result=dict(required=len(config['tasks']),completed=len(rows),coverage_complete=complete,Urban_module=urban,Urban_inputs=inputs,nonUrban=domains,main_metric=main_metric,scientific_gate='Not evaluated',cross_dataset_raw_MSE_average=False,F4_reused_without_new_run=True)
    if complete and main_metric in ('mse','mae'):
        u={k:v['mean'][main_metric] for k,v in urban.items()};f={k:v['mean'][main_metric] for k,v in inputs.items()}
        wins=sum(v['candidate'][main_metric]<v['B'][main_metric] for v in domains.values())
        gates=dict(goal1=u['UV']<u['U']<u['B'] and u['UV']<u['V']<u['B'],goal3=f['F4']<f['F2']<f['F1'] and f['F4']<f['F3']<f['F1'],goal2_improved_datasets=wins,goal2=wins>=8)
        result.update(gates=gates,numerical_conditions_met=bool(gates['goal1'] and gates['goal2'] and gates['goal3']),scientific_gate='Pending result review',result_review='Pending')
    return result
