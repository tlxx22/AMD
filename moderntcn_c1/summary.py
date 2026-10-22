"""New complete revision and paired EPF comparison; never overwrite old tables."""
import csv
import math
from pathlib import Path
from .contract import *
from .artifacts import audit

def summarize(c,auth,folder):
    folder=Path(folder);rows=[]
    if len(list((folder/'runs').iterdir()))!=53 or len(list((folder/'receipts').glob('*.json')))!=53:raise ValueError('53 exact saved results required')
    for t in c['tasks']:
        r=audit(folder/'runs'/t['id'],c,t,auth['commit']);old=c['parents'][t['id']]['old_cell'];r.update(dataset=t['dataset'],fold=t['fold'],H=t['h'],task=t['task'],old_run_id=old['run_id'],old_result_ref=c['parents'][t['id']]['result'],new_result_ref=ref(folder/'runs'/t['id']/'result.json'),percent_change={k:100*(r['metrics'][k]-float(old[k]))/float(old[k])if float(old[k])!=0 else None for k in ('mse','mae')});rows.append(r)
    with MAPPING.open() as stream:originals=list(csv.DictReader(stream))
    new_by_old={r['old_run_id']:r for r in rows}
    if len(originals)!=371 or len(new_by_old)!=53:raise ValueError('371 round2/53 replacement membership')
    effective=[]
    for old in originals:
        row=dict(old)
        if old['run_id']in new_by_old:
            x=new_by_old[old['run_id']];r=read(x['new_result_ref']['path']);row.update(run_id=x['task_id'],mse=str(r['mse']),mae=str(r['mae']),epochs=str(x['epochs']),best_epoch=str(x['best_epoch']),early_stopped=str(x['epochs']<int(old['max_epochs'])),producer=auth['commit'],protocol_sha=digest(c),profile_sha=r['profile_sha'],data_sha=r['data_sha'],origin='paper_C1_r8_v1_fixed_rerun',result_path=x['new_result_ref']['path'],result_sha256=x['new_result_ref']['sha256'])
        effective.append(row)
    with (folder/'round2-C1-effective-371.csv').open('x',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(originals[0]));writer.writeheader();writer.writerows(effective)
    write_new(folder/'old-to-new-53.json',rows)
    # Upstream was technically verified before this chain. Recheck saved metric/receipt bytes at use.
    up=Path(UPSTREAM['folder']);uv={}
    for p in (up/'receipts').glob('*.json'):
        r=read(p);t=r['manifest']['identity']['task']
        if t['dataset']in EPF and t['arm']=='UV':
            saved=Path(r['path'])/'metrics.json';m=read(saved)
            if m!=r['metrics']:raise ValueError('upstream UV metrics drift')
            uv[t['dataset']]=dict(metrics=m,ref=ref(saved))
    comparison=[]
    for d in EPF:
        b=next(x for x in rows if x['dataset']==d)
        if d not in uv:raise ValueError('complete five-market UV required; denominator cannot shrink')
        for k in ('mse','mae'):
            v=uv[d]['metrics'][k];bv=b['metrics'][k]
            if not math.isfinite(v):raise ValueError('nonfinite UV')
            comparison.append(dict(dataset=d,H=24,metric=k,UV=v,B_ref=bv,B_cfg=float(c['parents'][b['task_id']]['old_cell'][k]),strict_win=v<bv,percent_change=100*(v-bv)/bv if bv else None,UV_ref=uv[d]['ref'],B_ref_file=b['new_result_ref']))
    wins=sum(x['strict_win']for x in comparison)
    return dict(revision=ID,technical_complete=True,result_review='Pending',old_total=371,replaced_cells=53,unchanged_cells=318,actual_run_epochs=sum(r['epochs']for r in rows),actual_optimizer_updates=sum(r['updates']for r in rows),formal_test_calls=53,replacement_mapping_ref=ref(folder/'old-to-new-53.json'),effective_table_ref=ref(folder/'round2-C1-effective-371.csv'),goal2=dict(B_ref='paper_C1_r8_v1',B_cfg='cfg_adapted_v1',denominator=10,wins=wins,status='Met' if wins>=7 else 'Not met',mse_wins=sum(x['strict_win']for x in comparison if x['metric']=='mse'),mae_wins=sum(x['strict_win']for x in comparison if x['metric']=='mae'),cells=comparison),goal1='module ablation separately; baseline chain not an effect gate',goal3='Not evaluated')
