"""Sixteen input cells, separate from the completed 27 module cells."""
import math
from pathlib import Path
import json
from .training_config import validate,digest,load_ref
from .scope import HORIZONS,verified_f4_records
from .artifacts import audit
from moderntcn_dev_v10.summary import checked

def goal3(cells):
    if set(cells)!={'F1','F2','F3','F4'} or any(set(r)!=set(HORIZONS) for r in cells.values()):raise ValueError('exact 16-cell input grid required')
    for row in cells.values():
        for m in row.values():
            if m is not None:checked(m)
    complete=all(m is not None for row in cells.values() for m in row.values())
    means={v:{k:math.fsum(row[h][k] for h in HORIZONS)/4 for k in ('mse','mae')} for v,row in cells.items()} if complete else None
    comparisons={k:{'F4<F2':means['F4'][k]<means['F2'][k],'F2<F1':means['F2'][k]<means['F1'][k],'F4<F3':means['F4'][k]<means['F3'][k],'F3<F1':means['F3'][k]<means['F1'][k]} for k in ('mse','mae')} if complete else None
    met=all(v for row in comparisons.values() for v in row.values()) if complete else None
    return dict(coverage_complete=complete,required_cells=16,equal_four_H_mean=means,comparisons=comparisons,numerical_status=('Met' if met else 'Not met') if complete else 'Not evaluated',result_review='Pending')

def aggregate(config,records,gate_policy=None,*,f4_records=None):
    validate(config)
    if gate_policy is not None and gate_policy!=config['gate_policy']:raise ValueError('input policy mismatch')
    tasks={t['id']:t for t in config['tasks']};seen=set();cells={v:{h:None for h in HORIZONS} for v in ('F1','F2','F3','F4')}
    for t,m in records:
        if t['id'] in seen or tasks.get(t['id'])!=t or m['task_id']!=t['id']:raise ValueError('duplicate/foreign input result')
        seen.add(t['id']);cells[t['input_variant']][t['h']]=checked(m)
    if config['purpose']=='synthetic':return dict(required=len(tasks),fresh_completed=len(seen),scientific_gate='Not evaluated',result_review='Pending')
    actual=verified_f4_records(config)
    if f4_records is not None and f4_records!=actual:raise ValueError('forged F4 reference')
    for t,m in actual:cells['F4'][t['h']]=checked(m)
    return dict(required=12,completed=len(seen),fresh_completed=len(seen),F4_equivalent_reuse=4,input_evaluation_cells_available=len(seen)+4,module_evaluation_cells_not_added=27,per_variant_horizon=cells,goal3=goal3(cells),module_goals_source=config['completed_module_goals_ref'],module_goals=load_ref(config['completed_module_goals_ref']),engineering_completion='Complete' if len(seen)==12 else 'Partial',scientific_gate='Pending result review' if len(seen)==12 else 'Not evaluated',result_review='Pending',gate_policy=config['gate_policy'])

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
        if identity['config_sha']!=digest(config) or identity['gate_policy']!=config['gate_policy']:raise ValueError('input summary source/config/policy identity')
        records.append((t,r['metrics']))
    return aggregate(config,records,gate_policy)
