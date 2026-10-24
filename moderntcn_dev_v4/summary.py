"""Fresh v4 completion and explicitly attributed v3 V reuse; single-fold current goals."""
from pathlib import Path
import json,math
from .training_config import validate,digest
from .gate_policy import validate_policy,validate_config_policy
from .artifacts import audit
from .scope import expected_tasks,verified_v_records,MARKETS
METRICS=('mse','mae')
def checked(m):
    if any(type(m[k]) not in (float,int) or not math.isfinite(m[k]) or m[k]<0 for k in METRICS):raise ValueError('invalid saved metric')
    return m
def aggregate(config,records,gate_policy=None,*,reuse_records=None):
    validate(config)
    if gate_policy is not None and gate_policy!=config['gate_policy']:raise ValueError('gate policy mismatch')
    tasks={t['id']:t for t in config['tasks']};rows={}
    for t,m in records:
        if t['id'] in rows or tasks.get(t['id'])!=t or m['task_id']!=t['id']:raise ValueError('duplicate/foreign summary identity')
        rows[t['id']]=checked(m)
    fresh_count=len(rows)
    if config['purpose']=='synthetic':
        return dict(required=len(tasks),completed=fresh_count,scientific_gate='Not evaluated',result_review='Pending')
    reused=verified_v_records(config) if reuse_records is None else reuse_records
    expected={t['id']:t for t in expected_tasks(('V',))}
    if len(reused)!=9 or len({t['id'] for t,m in reused})!=9:raise ValueError('nine explicit V references required')
    for t,m in reused:
        if expected.get(t['id'])!=t or m['task_id']!=t['id']:raise ValueError('equivalent V reference identity')
        rows[t['id']]=checked(m)
    all_tasks=expected_tasks(('U','V','UV'))
    def cell(ds,h):
        ts=[t for t in all_tasks if t['dataset']==ds and t['h']==h]
        b=config['existing_B'][ts[0]['baseline_pair_cell_id']]
        return dict(B=checked(b),**{t['arm']:rows.get(t['id']) for t in ts})
    urban={str(h):cell('UrbanEV',h) for h in (3,6,9,12)}
    means={a:({k:math.fsum(v[a][k] for v in urban.values())/4 for k in METRICS} if all(v[a] is not None for v in urban.values()) else None) for a in ('B','U','V','UV')}
    coverage=all(means[a] is not None for a in ('U','V','UV'))
    g1=dict(coverage_complete=coverage,numerical_status='Not evaluated',result_review='Pending',positive_contribution_supported=None)
    if coverage:
        g1.update(numerical_status='Ablation evaluated',metrics={k:{'UV<U':means['UV'][k]<means['U'][k],'UV<V':means['UV'][k]<means['V'][k]} for k in METRICS})
        g1['positive_contribution_supported']=all(x for d in g1['metrics'].values() for x in d.values())
    epf={ds:cell(ds,24) for ds in MARKETS};complete=all(v['UV'] is not None for v in epf.values())
    g2=dict(coverage_complete=complete,denominator=10,required_winning_cells=7,numerical_status='Not evaluated',result_review='Pending')
    if complete:
        metrics={k:dict(winners=[ds for ds,v in epf.items() if v['UV'][k]<v['B'][k]],denominator=5) for k in METRICS}
        for x in metrics.values():x.update(winner_count=len(x['winners']),win_rate=len(x['winners'])/5)
        wins=sum(x['winner_count'] for x in metrics.values());g2.update(metrics=metrics,winning_cells=wins,numerical_status='Met' if wins>=7 else 'Not met',numerical_conditions_met=wins>=7)
    g3=dict(coverage_complete=False,numerical_status='Not evaluated',result_review='Pending',reason='fold6 F1/F2/F3 scope retained but not executed; v4 F4 only')
    return dict(required=18,completed=fresh_count,V_equivalent_reuse=9,evaluation_cells_available=len(rows),engineering_completion='Complete' if fresh_count==18 else 'Partial',scientific_gate='Pending result review' if fresh_count==18 else 'Not evaluated',result_review='Pending',goals=dict(goal1=g1,goal2=g2,goal3=g3),fold6=dict(per_horizon=urban,equal_four_H_mean=means),EPF=epf,gate_policy=config['gate_policy'])
def summarize(config,root,attempt,gate_policy=None):
    validate(config);records=[]
    if attempt is None:
        from .worker import RUN_NAMESPACE
        registry=(Path(root) if config['purpose']=='synthetic' else RUN_NAMESPACE)/'completed';paths=[]
        for p in registry.iterdir() if registry.exists() else []:
            if p.name.endswith('.tmp'):continue
            v=json.loads(p.read_text())
            if v.get('status')=='completed' and v['receipt']['manifest']['identity']['config_sha']==digest(config):paths.append(Path(v['receipt']['path']))
    else:
        folder=Path(root)/attempt/'runs';paths=list(folder.iterdir()) if folder.exists() else []
    for p in paths:
        r=audit(p);identity=r['manifest']['identity'];t=identity['task']
        if identity['config_sha']!=digest(config) or r['manifest']['task_id']!=t['id'] or identity['gate_policy']!=config['gate_policy']:raise ValueError('summary source/config/policy identity')
        records.append((t,r['metrics']))
    return aggregate(config,records,gate_policy)
