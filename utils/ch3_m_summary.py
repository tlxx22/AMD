"""M JSON summary/technical handoff; optional exact 168-checkpoint CPU audit."""
import json,math,hashlib,subprocess
from pathlib import Path
from utils.ch3_contract import digest,profile,step_arithmetic,BestState
from utils import ch3_m_tasks as scope

def verify_result(c,t,r):
 p=profile(c,t);n=step_arithmetic(c,t)['test_windows_arithmetic_only']*p['pred_len']
 if any(r.get(k)!=v for k,v in dict(id=t['id'],task='M',metric_scope='all_channels',input_variant='M',purpose='ch3_formal',protocol_sha=digest(c),profile_sha=digest(p),parent_MS_profile=p['parent_MS_profile'],from_scratch=True,seed=2024,metric_space='train-standardized').items()):raise ValueError('M result identity')
 if type(r.get('best_epoch'))is not int or not 1<=r['best_epoch']<=10:raise ValueError('M best epoch')
 for row,elems in [(r,n*p['C'])]+[(x,n)for x in r.get('channels',[])]:
  if row.get('elements')!=elems or any(not isinstance(row.get(k),(int,float))or not math.isfinite(row[k])or row[k]<0 for k in ('sse','sae','mse','mae'))or row['mse']!=row['sse']/elems or row['mae']!=row['sae']/elems:raise ValueError('M metric finite/count/aggregation')
 if len(r.get('channels',[]))!=p['C']or [(x['index'],x['name'])for x in r['channels']]!=list(enumerate(p['features']))or r['MS_target_diagnostic']!=r['channels'][p['target_idx']]:raise ValueError('M original channel order/diagnostic target')
 if abs(sum(x['sse']for x in r['channels'])-r['sse'])>max(1e-12,abs(r['sse'])*1e-14)or abs(sum(x['sae']for x in r['channels'])-r['sae'])>max(1e-12,abs(r['sae'])*1e-14):raise ValueError('M per-channel totals')
 if r.get('final_test',{}).get('calls')!=1 or r['final_test'].get('selected')!='best.pt'or r['final_test'].get('epoch')!=r['best_epoch']:raise ValueError('one final test selected by validation only')
 return r

def technical_group(c,model,a):
 from utils.ch3_m_execution import CONTROL
 from m6_remaining_entry import same
 root=CONTROL/model;control=json.loads((root/'controller.json').read_text())
 if same(control):raise ValueError('M group still active')
 if any(p.name in ('STOP','failure.json')or p.name.startswith(('staging','running','failed','recovery'))for p in root.iterdir()):raise ValueError('M retained failed/staging')
 ids=[t['id']for t in c['tasks']if t['model']==model];done=json.loads((root/'complete.json').read_text())
 if done.get('task_ids')!=ids or done.get('protocol_sha')!=digest(c)or done.get('commit')!=a['commit']:raise ValueError('M group complete identity')
 sources={};totals=dict(epochs=0,adam=0,forward=0,backward=0);worker_ids={}
 def load(p):sources[str(p)]=scope.ref(p);return scope.bound(sources[str(p)])
 for t in [t for t in c['tasks']if t['model']==model]:
  out=scope.result_path(t);p=profile(c,t);ar=step_arithmetic(c,t);m=load(out/'manifest.json');r=verify_result(c,t,load(out/'result.json'));b=load(out/'budget.json');runtime=load(out/'runtime.json');config=load(out/'config.json');i=m['identity']
  expected=dict(run_id=t['id'],profile_sha=digest(p),data_sha=digest(m['metadata']),commit=a['commit'],protocol_sha=digest(c),source=a['code'],task='M',metric_scope='all_channels',parent_MS_profile=p['parent_MS_profile'],from_scratch=True)
  if any(i.get(k)!=v for k,v in expected.items())or m['task']!=t or m['profile']!=p or a['data_bindings'][t['dataset']][t['id']]!=i['data_sha']or config['approval']!=a or config['resume']is not False:raise ValueError('M manifest/config/data actual binding')
  hp=out/'history.jsonl';sources[str(hp)]=scope.ref(hp);hist=[json.loads(x)for x in hp.read_text().splitlines()];best=BestState(p['training']['patience'])
  if not 1<=len(hist)<=10:raise ValueError('M epoch bound')
  for n,row in enumerate(hist,1):
   val=row['validation'];elements=ar['validation_windows']*p['pred_len']*p['C']
   if best.stopped or row['epoch']!=n or row['steps']!=n*ar['train_batches']or val['elements']!=elements or any(not math.isfinite(val[k])for k in ('mse','mae','sse','sae'))or val['mse']!=val['sse']/elements or val['mae']!=val['sae']/elements:raise ValueError('M history/budget/all-channel selection')
   best.update(val['mse'],n)
   if row['best_epoch']!=best.epoch:raise ValueError('M tie retains earlier best')
  if best.epoch!=r['best_epoch']or(len(hist)<10 and not best.stopped):raise ValueError('M best/stopping')
  steps=hist[-1]['steps'];fwd=len(hist)*(ar['train_batches']+math.ceil(ar['validation_windows']/p['training']['eval_batch']))+math.ceil(ar['test_windows_arithmetic_only']/p['training']['eval_batch']);counts=dict(adam=steps,backward=steps,forward=fwd)
  if b['counts']!=counts or runtime['error']is not None or b['by_pid']!={str(runtime['pid']):counts}:raise ValueError('M actual accounting/runtime')
  if(out/'best.pt').is_symlink()or(out/'best.pt').stat().st_nlink!=1 or scope.sha(out/'best.pt')!=r['final_test']['sha256']:raise ValueError('M final best SHA/path changed')
  worker_ids[t['id']]=str(runtime['pid']);totals['epochs']+=len(hist)
  for k,v in counts.items():totals[k]+=v
 report=scope.bound(a['resource_report']);waves=[]
 for g in [x for x in c['groups']if x['model']==model]:
  from utils.ch3_m_execution import wave_ids
  waves+=wave_ids(g['task_ids'],report['decisions'][g['id']]['concurrency'])
 for n,ids in enumerate(waves):
  d=root/('wave-'+str(n));pr=load(d/'process.json')
  if not pr['resource_admission']or not pr['exit_transitions_resolved']or pr['failure']is not None or pr['returncodes']!=[0]*len(ids)or set(pr['process_peaks'])!={worker_ids[r]for r in ids}:raise ValueError('M wave ownership/resource/exit')
  with (d/'memory.jsonl').open()as f:first=json.loads(next(f))
  for run in ids:
   pid=worker_ids[run];ticks=first['owned_pid_metadata'][pid]['start_ticks']
   if same(dict(pid=int(pid),start_ticks=str(ticks))):raise ValueError('M worker original still live')
 return dict(model=model,status='technical-complete',result_review='pending',task_ids=[t['id']for t in c['tasks']if t['model']==model],totals=totals,sources=sources,checkpoint_integrity='deferred to final CPU audit; saved-best byte binding checked')

def summarize(c):
 panels={};index=[];missing=[];channels=[];target=[];execution=[]
 for t in c['tasks']:
  p=scope.result_path(t)/'result.json';profile_row=profile(c,t);ar=step_arithmetic(c,t);actual={};sources={}
  for name in ('budget','runtime'):
   f=p.parent/(name+'.json')
   if f.exists():sources[name]=scope.ref(f);actual[name]=scope.bound(sources[name])
  failure=p.parent/'failure.json'
  if failure.exists():sources['failure']=scope.ref(failure);actual['failure']=scope.bound(sources['failure'])
  execution.append(dict(run_id=t['id'],model=t['model'],dataset=t['dataset'],H=t['h'],parent_MS_profile=profile_row['parent_MS_profile'],training=profile_row['training'],author_source=c['sources'].get(t['model']),max_optimizer_steps=ar['max_optimizer_steps'],max_epochs=profile_row['training']['epochs'],actual=actual,sources=sources))
  if not p.exists():
   missing.append(t['id']);failed='failure'in actual or actual.get('runtime',{}).get('error')is not None
   index.append(dict(key=['M',t['dataset'],t['model'],t['h'],2024],path=str(p.parent),status='failed'if failed else'not-run'if not p.parent.exists()else'incomplete',result_review='unverified'));continue
  r=verify_result(c,t,json.loads(p.read_text()));m=scope.bound(scope.ref(p.parent/'manifest.json'));identity=m['identity']
  if identity.get('run_id')!=t['id']or identity.get('profile_sha')!=digest(profile(c,t))or identity.get('data_sha')!=digest(m['metadata'])or identity.get('task')!='M':raise ValueError('M summary actual identity')
  index.append(dict(key=['M',t['dataset'],t['model'],t['h'],2024],status='completed-pending-review',result_review='pending',result=scope.ref(p),manifest=scope.ref(p.parent/'manifest.json'),identity=identity))
  panels.setdefault(t['dataset'],{}).setdefault(t['model'],[]).append(dict(H=t['h'],mse=r['mse'],mae=r['mae'],best_epoch=r['best_epoch'],source=scope.ref(p)))
  channels+=[dict(dataset=t['dataset'],model=t['model'],H=t['h'],**row)for row in r['channels']];target.append(dict(dataset=t['dataset'],model=t['model'],H=t['h'],**r['MS_target_diagnostic']))
 macro={}
 for d,models in panels.items():
  macro[d]={m:dict(mse=sum(x['mse']for x in rows)/4,mae=sum(x['mae']for x in rows)/4)for m,rows in models.items()if len(rows)==4}
 ranking={d:[dict(model=m,**v)for m,v in sorted(rows.items(),key=lambda item:(item[1]['mse'],item[1]['mae'],item[0]))]for d,rows in macro.items()}
 return dict(task='M',metric_scope='all_channels',panels=panels,four_H_macro=macro,baseline_ranking=ranking,ranking_basis='within one dataset; only models with all four H results; result review pending',channel_rows=channels,MS_target_diagnostic=target,index=index,missing=missing,execution=execution,result_review='pending',cross_dataset_mean=None,MS_ranking_included=False,J_competitiveness_evaluated=False)

def audit_checkpoints(c,out):
 """Future completion audit: exact M best/last only, CPU one object at a time."""
 import torch
 from ch3_runner import environment_binding,code_binding
 out=Path(out);out.mkdir(exist_ok=False);rows=[]
 if torch.cuda.is_initialized()or __import__('os').environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('M checkpoint audit CPU-only environment')
 torch.set_num_threads(1)
 if len(c['tasks'])!=84:raise ValueError('84 audit tasks')
 for t in c['tasks']:
  root=scope.result_path(t);manifest=json.loads((root/'manifest.json').read_text());result=verify_result(c,t,json.loads((root/'result.json').read_text()));history=[json.loads(x)for x in(root/'history.jsonl').read_text().splitlines()]
  for name in ('best.pt','last.pt'):
   p=root/name
   if p.is_symlink()or p.resolve()!=p or p.stat().st_nlink!=1:raise ValueError('M checkpoint exact owned path required before load')
   before=scope.sha(p);x=torch.load(p,map_location='cpu')
   if x['identity']!=manifest['identity']or x['epoch']!=(result['best_epoch']if name=='best.pt'else history[-1]['epoch'])or x['steps']!=x['epoch']*step_arithmetic(c,t)['train_batches']:raise ValueError('M checkpoint identity/epoch/step')
   if not x.get('model')or not x.get('optimizer')or not {'python','numpy','torch','cuda','generator'}<=set(x.get('rng',{})):raise ValueError('M checkpoint/RNG structure')
   opt=x['optimizer'];ids=[i for g in opt['param_groups']for i in g['params']]
   if len(ids)!=len(set(ids))or not set(opt['state'])<=set(ids):raise ValueError('M optimizer parameter identity')
   for state in opt['state'].values():
    if not {'step','exp_avg','exp_avg_sq'}<=set(state)or state['exp_avg'].shape!=state['exp_avg_sq'].shape or not 0<float(state['step'])<=x['steps']or float(state['step'])%1:raise ValueError('M optimizer moments/step structure')
   def check(v):
    if torch.is_tensor(v):
     if v.device.type!='cpu'or(v.is_floating_point()and not torch.isfinite(v).all()):raise ValueError('M checkpoint finite/CPU')
    elif isinstance(v,dict):
     for z in v.values():check(z)
    elif isinstance(v,(list,tuple)):
     for z in v:check(z)
   check(x);del x
   system=subprocess.check_output(['sha256sum',str(p)],text=True).split()[0]
   if before!=scope.sha(p)or before!=system:raise ValueError('M checkpoint byte changed')
   rows.append(dict(run=t['id'],file=name,path=str(p),sha256=before,status='Passed'))
   (out/'progress.json').write_text(json.dumps(dict(rows=rows,expected=168,reviewed=False),ensure_ascii=False,indent=2)+'\n')
 (out/'checkpoint-audit.json').write_text(json.dumps(dict(rows=rows,reviewed=False,source=code_binding(),environment=environment_binding(),model_construction=0,forward=0,checkpoint_loads=168),ensure_ascii=False,indent=2)+'\n')
 return rows

def estimate(c):
 """MS anchors are measured; M planning ranges require this M probe first."""
 from utils.ch3_m_execution import PROBE_ROOT,wave_ids
 E=scope.RESULT.parents[1];rows=[];parent_total=0.;m_total=[0.,0.];all_M=True
 for g in c['groups']:
  parent=[]
  if g['model']!='TimeXer':
   base=E/'revisions/timemixer-fixedlr-v2/formal-TimeMixer'if g['model']=='TimeMixer'else E/('formal-'+g['model'])
   for run in g['task_ids']:
    p=base/run.replace('-M-f','-MS-f')/'runtime.json'
    if not p.exists():raise ValueError('registered completed parent runtime missing: '+str(p))
    value=scope.bound(scope.ref(p))
    if value.get('error')is not None:raise ValueError('parent runtime failed')
    parent.append(dict(run_id=run.replace('-M-f','-MS-f'),runtime=scope.ref(p),elapsed=value['elapsed']))
  anchor=max(x['elapsed']for x in parent)if parent else None
  if anchor is not None:parent_total+=anchor
  r=dict(model=g['model'],dataset=g['dataset'],parent_MS=parent,parent_q4_slowest_seconds=anchor,TimeXer_parent='not read: old queue still executing; no complete registered anchor'if not parent else None,max_optimizer_steps=sum(step_arithmetic(c,t)['max_optimizer_steps']for t in c['tasks']if t['group']==g['id']),M_short_measurement=None,final_q=None,initialization=None,stable_updates=None,validation_pair=None,estimated_seconds_range=None)
  done=PROBE_ROOT/'complete.json'
  if done.exists():
   raw=scope.bound(scope.ref(done));decision=raw['decisions'][g['id']];q=decision['concurrency'];phase='serial'if q==1 else'q'+str(q);cost={};measures=[]
   for run in g['task_ids']:
    t=next(t for t in c['tasks']if t['id']==run);tr=scope.bound(scope.ref(PROBE_ROOT/g['id']/phase/run/'trajectory.json'));timing=tr['M_timing'];stable=timing['update_seconds'][2:];ar=step_arithmetic(c,t);epochs=profile(c,t)['training']['epochs'];vb=math.ceil(ar['validation_windows']/profile(c,t)['training']['eval_batch']);tb=math.ceil(ar['test_windows_arithmetic_only']/profile(c,t)['training']['eval_batch'])
    # Engineering extrapolation: stable train-step range, measured 2-batch eval pair.
    # Pair/full/tail shape differences and six-step sample uncertainty remain explicit.
    cost[run]=[timing['initialization_seconds']+ar['max_optimizer_steps']*min(stable)+(epochs*vb+tb)*timing['validation_seconds']/2,timing['initialization_seconds']+ar['max_optimizer_steps']*max(stable)+(epochs*vb+tb)*timing['validation_seconds']]
    measures.append(dict(run_id=run,source=scope.ref(PROBE_ROOT/g['id']/phase/run/'trajectory.json'),**timing))
   band=[sum(max(cost[run][k]for run in ids)for ids in wave_ids(g['task_ids'],q))for k in (0,1)]
   r.update(M_short_measurement=measures,final_q=q,estimated_seconds_range=band,initialization='measured separately',stable_updates='steps3-6; synthetic update/input digest included',validation_pair='measured existing full+tail pair; full-epoch extrapolation heuristic, not statistical bound')
   for k in (0,1):m_total[k]+=band[k]
  else:all_M=False
  rows.append(r)
 return dict(rows=rows,parent_MS_runtime_count=sum(len(r['parent_MS'])for r in rows),parent_q4_reference_seconds=parent_total,parent_reference_is_M_measurement=False,M_total_seconds_range=m_total if all_M else None,status='M measurements pending'if not all_M else'M synthetic planning extrapolation; excludes CSV I/O and full-state capture cost',formula='sum per fixed wave max(initialization + stable update cost*max steps + full validation/final test estimate); never worker package / 6',uncertainties=['M supervision changes cost; TimeXer M changes native n_vars/global tokens','six steps cannot establish long-run steady state or full validation cost','early stopping may shorten actual epochs; bounds use maxima','GPU final q is unknown before M admission','state-capture/launch costs excluded from stable-update estimate'],no_cross_dataset_metric_mean=True)
