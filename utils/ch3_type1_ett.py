"""Source-bound incremental ETT migration; no model or data observation import."""
import copy
from functools import lru_cache
from utils.ch3_contract import ROOT,digest
from utils.ch3_native_recovery_records import bound
from utils.ch3_type1_scaled import configuration

NEW_DATASETS=('ETTh2','ETTm1','ETTm2')
M_DATASETS=('ETTh1','Weather','Exchange')+NEW_DATASETS
PACKAGE=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-type1-followup-v2'
TEMPLATE_REF=dict(path=str(PACKAGE/'reviewed-147-candidate/workspace-bytes/configs/ch3_type1_m_all_v2.json'),sha256='9c8b109244f2921faefd8c611e804af665db21ef564f6a9a30fc3740b3876c04')
SOURCE_REF=dict(path=str(PACKAGE/'ett-data-source-audit.json'),sha256='8feff1bcc8059132b01e762290619bf68ca8654ad1af5c4900b82e0fd11fea1c')
PAPER_REF=dict(path=str(PACKAGE/'paper-table7-source.json'),sha256='a8817c2acb0a21e6c60cae5e714f5b7944d9920ffb235eea05d9e81d298b010f')

@lru_cache(maxsize=1)
def template():return bound(TEMPLATE_REF)

@lru_cache(maxsize=1)
def sources():
    value=bound(SOURCE_REF);paper=bound(PAPER_REF)
    if value['paper_ref']!=PAPER_REF or set(value['datasets'])!=set(NEW_DATASETS):raise ValueError('exact independently admitted ETT sources')
    if any(paper['rows'][d]!=dict(train_batch=128,epochs=10)for d in NEW_DATASETS):raise ValueError('fixed Table 7 train batch/epochs')
    return value

def source_task(t):
    rows=[x for x in template()['tasks']if x['model']==t['model']and x['dataset']=='ETTh1'and x['h']==t['h']]
    if len(rows)!=1:raise ValueError('unique same-model same-H ETTh1 M template')
    return rows[0]

def dataset(name):
    if name not in NEW_DATASETS:raise ValueError('new ETT dataset only')
    return copy.deepcopy(sources()['datasets'][name]['dataset'])

def profile(t):
    before=template()['resolved_profiles'][source_task(t)['id']];p=copy.deepcopy(before);p['dataset']=t['dataset']
    evidence=sources()['datasets'][t['dataset']]
    if 'freq'in p['structure']:p['structure']['freq']=evidence['timeF_freq']
    if p.get('time_mark'):
        from utils.ch3_time_marks import policy
        p['time_mark']=policy(p)
    windows=dataset(t['dataset'])['endpoints'][0]-p['T']-p['pred_len']+1
    p['training']['scheduler']=configuration(p['training']['epochs'],windows//p['training']['batch'])
    return p

def parent_ref(t):
    old=source_task(t)
    return dict(task_id=old['id'],profile_sha=digest(template()['resolved_profiles'][old['id']]),config_ref=TEMPLATE_REF)

def numeric_policy(model,name):
    rule=copy.deepcopy(template()['baseline_unified']['numeric_policies'][model+'-ETTh1'])
    if rule is not None:
        rule['dataset']=name;rule['id']=rule['id'].replace('etth1',name.lower())
    return rule
