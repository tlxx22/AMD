"""Exact 53-task C.1 contract. Standard library only; no data/model access."""
import copy
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from utils.ch3_contract import digest, step_arithmetic

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
ID = 'moderntcn-paper-C1-r8-round2-v1'
BRANCH = 'M6-ModernTCN-C1-baseline-v1'
CONFIG = ROOT/'configs/moderntcn_c1_round2_rerun_v1.json'
EVIDENCE = BASE/'amd-execution-evidence/m6/M6-ModernTCN-C1-round2-rerun-v1'
RESULT_ROOT = BASE/'amd-execution-evidence/m6/moderntcn-c1-round2-runs/paper-C1-r8-round2-v1'
ATTEMPT = 'moderntcn-C1-r8-round2-first-r1'
PYTHON = BASE.parent/'miniconda/envs/amd/bin/python'
MAPPING = BASE/'amd-execution-evidence/m6/m6-original-round2-round3-result-review-v1-20261015/round2-raw-tasks.csv'
MAPPING_SHA = '3bc76e32e8290e785e2eac95fbea99b2d4e56831124c1e123270dfbdeae417a8'
PARENTS = [BASE/'AMD-m-baselines-v1/configs/ch3_baseline_ms_u96_oc01_v3.json', BASE/'AMD-type1-followup-v1/configs/ch3_round2_m_batch128_etth1_numeric_r1.json', BASE/'AMD-type1-followup-v1/configs/ch3_round2_m_amend1_weather20_patience10_v1.json']
EPF = ('PJM','NP','BE','FR','DE')
M_DOMAINS = ('ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange')
STRUCTURE = dict(num_blocks=[1], dims=[64,64,64,64], ffn_ratio=8, large_size=[51], small_size=[5], patch_size=8, patch_stride=4)
POLICY = dict(B_ref='paper_C1_r8_v1',B_cfg='cfg_adapted_v1',goal1='UV versus w/o U and w/o V; each original fold, four H equally weighted, both metrics; no U<B or V<B requirement',goal2=dict(datasets=list(EPF),H=24,metrics=['mse','mae'],denominator=10,minimum_strict_wins=7,reference='paired C1 B only',ties_win=False,missing_reduce_denominator=False,separate_metric_threshold=False),goal3='unchanged Urban input ablation; this baseline rerun does not evaluate it',dispatch_depends_on='technical completion only')
LOCK = BASE/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu/project-gpu0.lock'
UP_ROOT = BASE/'amd-execution-evidence/m6/moderntcn-uv-v1-development-runs/urbanev-f6-epf-3arm-uv-v1-extension-v1'
UP_ATTEMPT = 'urbanev-f6-epf-3arm-moderntcn-uv-v1-extension-r1'
UPSTREAM = dict(folder=str(UP_ROOT/UP_ATTEMPT),
    auth=dict(path=str(BASE/'amd-execution-evidence/m6/M6-ModernTCN-UV-v1-extension-f6-epf-launch-authorization-r1/AUTH.json'),sha256='23ad4385142af999ff46142ac875b29934c6d1ce47b4e7af3359c051517f6991'),
    config=dict(path=str(BASE/'ModernTCN-U+V/configs/moderntcn_uv_extension_f6_epf_3arm_v1.json'),sha256='fd0bdca9637ea19858ba60ec7f63136fc881846f3e288ce59392dd7feeb418b4'),
    runtime=dict(path=str(UP_ROOT/UP_ATTEMPT/'runtime-609d6c5d74ab47d4860cbcb367965bef.json'),sha256='cfa6ec00e2466c6ac96cf5d9b9df093f35eaa51854e6e43825f218ec6ef6e525'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def bound(r):
    if Path(r['path']).is_symlink() or sha(r['path'])!=r['sha256']:raise ValueError('bound bytes changed: '+r['path'])
    return read(r['path'])
def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,timeout=30).strip()

def tasks():
    cells=[('UrbanEV','MS','F4',f,h)for f in range(1,7)for h in (3,6,9,12)]
    cells += [(d,'MS','MS',1,24)for d in EPF]
    cells += [(d,'M','M',1,h)for d in M_DOMAINS for h in (96,192,336,720)]
    return [dict(id=f'ModernTCN-C1-r8-round2-{d}-{v}-f{f}-h{h}-s2024',model='ModernTCN',dataset=d,task=k,input_variant=v,fold=f,h=h,seed=2024,metric_scope='all_channels' if k=='M' else 'target_only')for d,k,v,f,h in cells]

def waves(c):
    ids=[t['id'] for t in c['tasks']]
    return [ids[i:i+4]for i in range(0,24,4)]+[ids[24:28],ids[28:29]]+[ids[i:i+4]for i in range(29,53,4)]

def profile(c,t):
    if c.get('c1_baseline')!=ID or t not in tasks():raise ValueError('exact C1 registered task')
    return copy.deepcopy(c['resolved_profiles'][t['id']])

def derive():
    if sha(MAPPING)!=MAPPING_SHA:raise ValueError('round2 final mapping changed')
    cs=[read(p)for p in PARENTS]
    with MAPPING.open() as stream:rows=[r for r in csv.DictReader(stream)if r['model']=='ModernTCN']
    if len(rows)!=53:raise ValueError('53 ModernTCN reference cells required')
    c=dict(c1_baseline=ID,contract='ch3-target-ms-formal-v2',backbone_variant='paper_C1_r8_v1',tasks=tasks(),resolved_profiles={},datasets={},urban_folds=cs[0]['urban_folds'],sources={'ModernTCN':cs[0]['sources']['ModernTCN']},parents={},data_bindings={},data_stat={},global_lock=str(LOCK),upstream=UPSTREAM,max_runs=53,max_run_epochs=860,max_optimizer_steps=2164920,execution_authorized=False,preparation_head=git('rev-parse','HEAD'),round2_mapping=ref(MAPPING),parent_protocols=[ref(p)for p in PARENTS],source_binding={})
    diffs=[]
    for t in c['tasks']:
        rows1=[r for r in rows if r['dataset']==t['dataset']and int(r['fold'])==t['fold']and int(r['H'])==t['h']and r['task']==t['task']]
        if len(rows1)!=1:raise ValueError('unique final round2 cell')
        r=rows1[0];rp=Path(r['result_path'])
        if sha(rp)!=r['result_sha256']:raise ValueError('parent result SHA')
        result=read(rp);manifest=read(rp.parent/'manifest.json')
        i=0 if t['task']=='MS' else 1 if t['dataset']in ('ETTh1','Exchange') else 2
        pc=cs[i];pt=next(x for x in pc['tasks']if x['model']=='ModernTCN'and x['dataset']==t['dataset']and x['fold']==t['fold']and x['h']==t['h'])
        parent=pc['resolved_profiles'][pt['id']]
        if parent!=manifest['profile']or digest(parent)!=r['profile_sha']or digest(manifest['metadata'])!=r['data_sha']:raise ValueError('final parent profile/data mismatch '+t['id'])
        if any(result[k]!=float(r[k]) for k in ('mse','mae')):raise ValueError('parent mapped metrics')
        p=copy.deepcopy(parent);p['structure'].update(copy.deepcopy(STRUCTURE));c['resolved_profiles'][t['id']]=p
        c['datasets'][t['dataset']]=copy.deepcopy(pc['datasets'][t['dataset']])
        c['parents'][t['id']]=dict(result=ref(rp),manifest=ref(rp.parent/'manifest.json'),protocol=ref(PARENTS[i]),task=pt,profile_sha=digest(parent),data_sha=digest(manifest['metadata']),old_cell=r,old_profile=parent)
        c['data_bindings'].setdefault(t['dataset'],{})[t['id']]=digest(manifest['metadata'])
        diffs.append(dict(task_id=t['id'],changes={k:dict(before=parent['structure'][k],after=v)for k,v in STRUCTURE.items()if parent['structure'][k]!=v},training_changed=False,data_changed=False))
    for name,d in c['datasets'].items():
        paths=[Path(d['path'])/n for n in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv')]if name=='UrbanEV'else [Path(d['path'])]
        for p in paths:c['data_stat'][str(p)]={k:getattr(p.stat(),k)for k in ('st_dev','st_ino','st_size','st_mtime_ns')}
    c['waves']=waves(c);c['comparison_policy']=copy.deepcopy(POLICY)
    validate(c,check_sources=False)
    return c,diffs

def source_binding():
    # Bind actual small execution dependency closure, including modules imported after author build.
    paths=['ch3_runner.py','models/__init__.py','models/ch3_adapter.py','utils/__init__.py','utils/ch3_contract.py','utils/ch3_data.py','utils/ch3_onecycle.py','utils/ch3_m_tasks.py','utils/ch3_time_marks.py','utils/dataloader_urbanev.py','utils/feature_schema.py','moderntcn_c1_entry.py','scripts/ch3/start_moderntcn_c1.sh']
    paths += [str(p.relative_to(ROOT))for p in sorted((ROOT/'moderntcn_c1').glob('*.py'))]
    return {p:sha(ROOT/p)for p in paths}

def validate(c,check_sources=True):
    if c.get('c1_baseline')!=ID or c['tasks']!=tasks()or c.get('backbone_variant')!='paper_C1_r8_v1' or c['waves']!=waves(c):raise ValueError('exact C1 scope/waves')
    if set(c['resolved_profiles'])!={t['id']for t in tasks()}or set(c['parents'])!=set(c['resolved_profiles']):raise ValueError('profile membership')
    if c['upstream']!=UPSTREAM or c['global_lock']!=str(LOCK):raise ValueError('upstream/mutex identity')
    if c.get('comparison_policy')!=POLICY:raise ValueError('C1 comparison policy identity')
    epochs=updates=0
    for t in c['tasks']:
        p=profile(c,t);parent=c['parents'][t['id']]['old_profile'];expected=copy.deepcopy(parent);expected['structure'].update(STRUCTURE)
        if p!=expected or digest(parent)!=c['parents'][t['id']]['profile_sha']:raise ValueError('parent science drift')
        tr=p['training'];d=t['dataset'];ne=20 if d=='UrbanEV'or d in EPF or d=='Weather'else 10;pat=10 if d=='Weather'else 5 if d=='UrbanEV'or d in EPF else None
        if tr['epochs']!=ne or tr['patience']!=pat or tr['batch']!=128 or tr['eval_batch']!=128 or tr['lr']!=.01 or tr['seed']!=2024 or tr['dtype']!='float32' or tr['weight_decay']!=1e-7 or not tr['from_scratch']:raise ValueError('frozen second-round training')
        s=p['structure'];drop=.1 if d=='UrbanEV'else .4 if d=='Weather'else .2 if d=='Exchange'else .3
        if s['dropout']!=drop or s['head_dropout']!=(.6 if d=='Exchange'else 0)or any(s[k]!=v for k,v in dict(revin=1,affine=0,subtract_last=0,decomposition=0,individual=0,use_multi_scale=False,small_kernel_merged=False).items()):raise ValueError('native structure policy')
        if p['T']!=(12 if d=='UrbanEV'else 96)or p['pred_len']!=(1 if d=='UrbanEV'else t['h']):raise ValueError('L/P/H')
        if t['task']=='M' and (p['task']!='M'or p['supervised_channels']!=list(range(p['C']))):raise ValueError('M all-channel supervision')
        if d in EPF and (p['C']!=3 or p['target_idx']!=2 or p['aux_idx']!=[0,1]):raise ValueError('EPF receiver')
        ar=step_arithmetic(c,t)
        from utils.ch3_onecycle import configuration
        if tr['scheduler']!=configuration(ne,ar['train_batches']):raise ValueError('OneCycle arithmetic')
        epochs+=ne;updates+=ar['max_optimizer_steps']
    if (len(tasks()),epochs,updates)!=(53,860,2164920)or (c['max_runs'],c['max_run_epochs'],c['max_optimizer_steps'])!=(53,860,2164920):raise ValueError('hard batch budget')
    if check_sources and c['source_binding']!=source_binding():raise ValueError('execution source changed')
    return c
