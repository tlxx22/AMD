"""Read-only paired result index; no ranking, selection, training or test execution."""
import json
from pathlib import Path
from utils.ch3_contract import digest
from utils.ch3_native_recovery_records import bound,ref
from utils import ch3_type1_tasks as s
from utils.ch3_type1_upstream import OLD_RESULT

def observed(path,task_id=None,protocol=None,profile_sha=None):
    path=Path(path)
    if not path.exists():return dict(path=str(path),status='not_available')
    if path.is_symlink()or path.resolve()!=path:raise ValueError('no result aliases')
    r=bound(ref(path));m=bound(ref(path.with_name('manifest.json')))
    if r.get('id')!=m['task']['id'] or (task_id and r.get('id')!=task_id)or (protocol and r.get('scientific_protocol')!=protocol)or (profile_sha and r.get('profile_sha')!=profile_sha):raise ValueError('result source identity mismatch')
    return dict(path=str(path),ref=ref(path),manifest_ref=ref(path.with_name('manifest.json')),task_id=r['id'],profile_sha=m['identity']['profile_sha'],science_execution_commit=m['identity']['commit'],mse=r['mse'],mae=r['mae'],status='complete',seed=r['seed'],std='N/A')
def result_index(configs):
    from utils.ch3_round2_amendment import summary,key
    revision=summary();cell_bank={r['cell_id']:r for r in revision.get('cells',[])}
    rows=[];base=s.RESULT.parent
    for stage,c in configs.items():
        if stage=='M_AMEND':continue
        for prior,t in zip(s.selected(stage),c['tasks']):
            current=s.context(c)['result_root']/('formal-'+t['model'])/t['id']/'result.json'
            row=dict(ring=stage,model=t['model'],dataset=t['dataset'],fold=t['fold'],H=t['h'],seed=2024,std='N/A',new=observed(current,t['id'],s.PROTOCOL,digest(s.profile(c,t))),v3=observed(OLD_RESULT/('M' if stage=='M_ALL' else 'MS')/('formal-'+t['model'])/prior['id']/'result.json',prior['id'],'baseline-unified96-onecycle001-v3'),historical_T168=[])
            # Daily comparison follows the sealed round-two source rule, never test ranking.
            row['second_round_revised']=cell_bank.get(key(t),dict(status='revision_pending'))
            if stage=='EPF_ALL':
                legacy_id=t['model']+'-'+t['dataset']+'-MS-f1-h24-s2024'
                paths=[base/('formal-'+t['model'])/legacy_id/'result.json',base/'supplements/epf4-timemixer-v1'/('formal-'+t['model'])/legacy_id/'result.json']
                for root in (base/'revisions').glob('native-time-mark-v*'):
                    candidate=root/('formal-'+t['model'])/(t['model']+'-'+t['dataset']+'-MS-f1-h24-s2024-native-tmark-v1')/'result.json';paths.append(candidate)
                for path in paths:
                    if not path.exists():continue
                    m=bound(ref(path.with_name('manifest.json')))
                    if m['profile'].get('T')==168 and all(m['task'].get(k)==t[k]for k in ('model','dataset','fold','h','seed')):row['historical_T168'].append(observed(path))
            rows.append(row)
    return dict(protocol=s.PROTOCOL,result_review='pending',rows=rows,coverage=dict(UrbanEV='28/168: folds1,2 and H3,12 only',EPF='all 35 fixed model/market pairs',M='all 168 fixed all-channel tasks; new ETT have no v3 result'),legacy_paths_include_supplements=True,history_test_seen=True,no_automatic_main_table_replacement=True,no_old_new_selection=True,aggregation='separate dataset/fold/H, never average raw MSE across domains')
