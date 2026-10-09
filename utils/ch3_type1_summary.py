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
    value=dict(path=str(path),ref=ref(path),manifest_ref=ref(path.with_name('manifest.json')),task_id=r['id'],profile_sha=m['identity']['profile_sha'],science_execution_commit=m['identity']['commit'],mse=r['mse'],mae=r['mae'],status='complete',seed=r['seed'],std='N/A')
    if m['identity'].get('resource_mode')is not None:value.update(resource_mode=m['identity']['resource_mode'],resource_contract_ref=m['identity']['resource_contract_ref'])
    if m['identity'].get('startup_hardware_ref'):value['startup_hardware_ref']=m['identity']['startup_hardware_ref']
    return value
def result_index(configs):
    from utils.ch3_round2_amendment import summary,key
    revision=summary();cell_bank={r['cell_id']:r for r in revision.get('cells',[])}
    rows=[];base=s.RESULT.parent
    for stage,c in configs.items():
        if stage not in ('URBAN_SUBSET','EPF_ALL','M_ALL'):continue
        for prior,t in zip(s.selected(stage),c['tasks']):
            current=s.context(c)['result_root']/('formal-'+t['model'])/t['id']/'result.json'
            row=dict(ring=stage,model=t['model'],dataset=t['dataset'],fold=t['fold'],H=t['h'],seed=2024,std='N/A',new=observed(current,t['id'],c['baseline_unified']['id'],digest(s.profile(c,t))),v3=observed(OLD_RESULT/('M' if stage=='M_ALL' else 'MS')/('formal-'+t['model'])/prior['id']/'result.json',prior['id'],'baseline-unified96-onecycle001-v3'),historical_T168=[])
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
    urban=len(configs['URBAN_SUBSET']['tasks'])
    urban_coverage={28:'28/168: folds1,2 and H3,12 only',84:'84/168: folds1–6 and H3,12 only',168:'168/168: folds1–6 and H3,6,9,12'}
    if urban not in urban_coverage:raise ValueError('unknown Urban coverage cannot be summarized')
    fixed=revision.get('adoption_policy_ref')
    value=dict(protocol=s.PROTOCOL,result_review='pending',rows=rows,coverage=dict(UrbanEV=urban_coverage[urban],EPF='all 35 fixed model/market pairs',M='all 168 fixed all-channel tasks'),legacy_paths_include_supplements=True,history_test_seen=True,no_automatic_main_table_replacement=not bool(fixed),no_metric_based_layer_selection=True,main_table_policy_ref=fixed,fixed_encoder_adoption=revision.get('fixed_encoder_adoption','complete'if fixed else None),no_old_new_selection=True,aggregation='separate dataset/fold/H, never average raw MSE across domains')
    from utils.ch3_type1_chain import PROBE_RECOVERY
    if PROBE_RECOVERY and hasattr(PROBE_RECOVERY,'resource_binding'):value.update(PROBE_RECOVERY.resource_binding())
    from utils import ch3_reviewed_concurrency as concurrency
    if concurrency.enabled():value.update(concurrency_policy=concurrency.POLICY,concurrency_plan_ref=PROBE_RECOVERY.contract()['concurrency_plan_ref'],probe_admission_claim=False)
    return value
