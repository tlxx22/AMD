"""Isolated 84-run M contract. Parent MS profiles are immutable provenance."""
import copy,json,hashlib,subprocess
from pathlib import Path
from utils.ch3_contract import ROOT,CONTRACT,digest
BASE='5341fbcb7c9f4f97658728d79b1af5487f7d38c3'
ID='m-baselines-v1';PROBE='m-baselines-probe-v1'
MODELS=('AMD','DLinear','PatchTST','iTransformer','TimeMixer','ModernTCN','TimeXer')
DOMAINS=('ETTh1','Weather','Exchange');HS=(96,192,336,720)
PACKAGE=ROOT.parent/'amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1'
RESULT=ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu/m-tasks/m-baselines-v1'
DATA={'ETTh1':(512,7,[8640,11520,14400],6),'Weather':(512,21,[36887,42157,52696],1),'Exchange':(96,8,[5311,6071,7588],7)}
CAPS={'adam':1512,'forward':2016,'backward':1512}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def bound(r):
 p=Path(r['path'])
 if p.is_symlink() or sha(p)!=r['sha256']:raise ValueError('M source path/SHA mismatch')
 x=json.loads(p.read_text())
 if sha(p)!=r['sha256']:raise ValueError('M source changed during read')
 return x

def tasks():
 return [dict(id=f'{m}-{d}-M-f1-h{h}-s2024',model=m,dataset=d,input_variant='M',task='M',metric_scope='all_channels',fold=1,h=h,group=f'{m}-{d}-M',profile=f'{m}-{d}-M-h{h}',seed=2024)for m in MODELS for d in DOMAINS for h in HS]

def resolved(c,t):
 if t not in tasks():raise ValueError('outside exact 84 M tasks')
 parent=c['m_experiment']['parent_profiles'][t['id']];p=copy.deepcopy(parent['profile'])
 p.update(task='M',input_variant='M',metric_scope='all_channels',supervised_channels=list(range(p['C'])),
    parent_MS_profile=dict(run_id=parent['id'],sha256=parent['sha256'],commit=BASE,protocol_sha=c['m_experiment']['parent_protocol_sha']),
    output_order=list(p['features']),from_scratch=True)
 if t['model']=='AMD':p['aux_idx']=[];p['native_task_mode']='parallel_multivariate'
 if t['model']=='TimeXer':p['native_features']='M';p['native_n_vars']=p['C']
 return p

def validate(c):
 from utils.ch3_contract import generate_groups,RSS_PROBE_POLICY
 if c['contract']!=CONTRACT or c['tasks']!=tasks()or c['groups']!=generate_groups(tasks()):raise ValueError('exact M contract/tasks/groups')
 if set(c['datasets'])!=set(DOMAINS)or c['m_experiment']['id']!=ID:raise ValueError('M domain scope')
 if c['execution']['evidence']!=str(RESULT)or c['execution']['fixture']!=str(PACKAGE/'fixtures'):raise ValueError('M isolated roots')
 if c['execution']['probe']['rss_policy']!=RSS_PROBE_POLICY:raise ValueError('M RSS policy changed')
 raw=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':configs/ch3_formal_profiles.json'])
 if hashlib.sha256(raw).hexdigest()!=c['m_experiment']['parent_config_sha256']:raise ValueError('parent frozen config')
 parent=json.loads(raw)
 if c['sources']!={m:parent['sources'][m]for m in MODELS if m in parent['sources']}or c['training_common']!=parent['training_common']:raise ValueError('M author sources/common optimizer changed')
 if c['m_experiment']['numeric_policies']!=make_policies(parent)or c['m_experiment']['parent_protocol_sha']!=digest(parent):raise ValueError('M per-scope numerical policy/parent protocol changed')
 if c['m_experiment']['project_lock']!=str(Path(parent['execution']['evidence'])/'project-gpu0.lock')or c['m_experiment']['gpu_caps']!=CAPS or c['m_experiment']['cpu_forward_cap']!=64:raise ValueError('M shared resource lock/independent caps')
 # Resolve parent using its own frozen functions, no module shadowing or model import.
 code=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':utils/ch3_contract.py'])
 ns={'__file__':str(ROOT/'utils/ch3_contract.py')};exec(compile(code,'frozen-MS-contract','exec'),ns)
 for d,(T,C,ends,target)in DATA.items():
  if c['datasets'][d]!=parent['datasets'][d]or c['datasets'][d]['T']!=T or c['datasets'][d]['endpoints']!=ends or len(c['datasets'][d]['features'])!=C or c['datasets'][d]['features'].index(c['datasets'][d]['target'])!=target:raise ValueError('data/schema/endpoint changed')
 for t in tasks():
  pid=t['id'].replace('-M-f','-MS-f');pt=next(x for x in parent['tasks']if x['id']==pid);p=ns['profile'](parent,pt);saved=c['m_experiment']['parent_profiles'][t['id']]
  if saved!={'id':pid,'profile':p,'sha256':digest(p)}:raise ValueError('parent MS effective profile changed')
  q=resolved(c,t)
  if q['training']!=p['training']or q['structure']!=p['structure']or q['T']!=p['T']or q['training']['epochs']!=10:raise ValueError('unauthorized M hyperparameter change')
 if len(c['tasks'])!=84 or sum(resolved(c,t)['training']['epochs']for t in tasks())!=840:raise ValueError('M run/epoch budget')
 return c

def numeric_policy(c,t):
 if t not in tasks():raise ValueError('numeric task scope')
 row=c['m_experiment']['numeric_policies'][t['group']]
 # Each bounded proposal is copied from only this model/domain's accepted MS policy,
 # renamed/re-scoped to M; no claim of M measurements or approval is made.
 return row['rule']

def make_policies(parent):
 result={}
 for m in MODELS:
  for d in DOMAINS:
   old=next((x for x in parent['execution']['probe']['numeric_equivalence']if x['model']==m and x['dataset']==d),None)
   rule=copy.deepcopy(old)if old else None
   if rule:rule['id']=m.lower()+'-'+d.lower()+'-M-full-state-proposed-v1';rule['task']='M'
   result[f'{m}-{d}-M']=dict(rule=rule,status='Proposed',basis='same model/domain MS numerical-rule basis; M scope requires new review and measurements'if old else'parent model/domain exact rule; no global exact reset',parent_rule=old,parent_rule_sha=digest(old),no_M_admission_inherited=True)
 return result

def result_path(t):
 p=RESULT/('formal-'+t['model'])/t['id']
 if p.resolve()!=p or p.is_symlink():raise ValueError('M result path symlink/foreign namespace')
 return p
def metric_totals(errors,features,target_idx):
 """Pure numerical aggregation helper used in no-model tests."""
 import numpy as np
 if not errors:raise ValueError('empty metrics')
 sse=np.zeros(len(features),dtype='float64');sae=sse.copy();n=0
 for e in errors:
  e=np.asarray(e,dtype='float64')
  if e.ndim!=3 or e.shape[2]!=len(features)or not np.isfinite(e).all():raise ValueError('M error shape/finite')
  sse+=(e*e).sum(axis=(0,1));sae+=abs(e).sum(axis=(0,1));n+=e.shape[0]*e.shape[1]
 if not n:raise ValueError('empty elements')
 channels=[dict(index=i,name=name,mse=float(sse[i]/n),mae=float(sae[i]/n),sse=float(sse[i]),sae=float(sae[i]),elements=n)for i,name in enumerate(features)]
 return dict(mse=float(sse.sum()/(n*len(features))),mae=float(sae.sum()/(n*len(features))),sse=float(sse.sum()),sae=float(sae.sum()),elements=n*len(features),channels=channels,MS_target_diagnostic=channels[target_idx])

def evaluate_all(model,batches,p,device):
 import torch
 from models.ch3_adapter import target_prediction
 from ch3_runner import finite
 model.eval();sse=torch.zeros(p['C'],dtype=torch.float64);sae=sse.clone();n=0
 with torch.no_grad():
  for x,y in batches:
   pred,_=target_prediction(model,x.to(device),p)
   if tuple(y.shape)!=tuple(pred.shape):raise ValueError('M label/output shape mismatch; no broadcasting')
   e=pred-y.to(device);finite(e);sse+=e.double().square().sum((0,1)).cpu();sae+=e.double().abs().sum((0,1)).cpu();n+=e.shape[0]*e.shape[1]
 if not n:raise ValueError('empty evaluation')
 channels=[dict(index=i,name=name,mse=float(sse[i]/n),mae=float(sae[i]/n),sse=float(sse[i]),sae=float(sae[i]),elements=n)for i,name in enumerate(p['features'])]
 return dict(mse=float(sse.sum()/(n*p['C'])),mae=float(sae.sum()/(n*p['C'])),sse=float(sse.sum()),sae=float(sae.sum()),elements=n*p['C'],channels=channels,MS_target_diagnostic=channels[p['target_idx']])
