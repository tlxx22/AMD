"""Exact preregistered synthetic M forward checks, one author per process."""
import sys,json,os,time
from pathlib import Path
TOOL=Path(__file__).resolve().parent;sys.path.insert(0,str(TOOL));sys.path.insert(0,str(TOOL.parents[1]))
from utils.ch3_contract import read_profiles,validate_manifest
from utils import ch3_m_execution as e
from utils import ch3_m_tasks as s
from ch3_runner import dump
from resource_budget import initialize
from m5_formal_entry import spawn

def main():
 c=validate_manifest(read_profiles());root=e.CPU_ROOT;root.mkdir(exist_ok=False)
 plan=dict(id='m-baselines-cpu-shapes-v1',ids=[e.CPU_ID],order=list(s.MODELS),cases={m:e.cpu_tasks(c,m)for m in s.MODELS},forward_cap=64,planned_forward=54,adam=0,backward=0,cuda_initialized=0,source={})
 from ch3_runner import code_binding
 plan['source']=code_binding();dump(root/'preregistration.json',plan)
 initialize(root/'shared-budget.json','M_cpu_shapes',dict(adam=0,forward=64,backward=0,seconds=3600,rss=4*1024**3));executions=[]
 try:
  for model in s.MODELS:
   conf=e.make_cpu_config(c,model);p,h=spawn(conf)
   try:code=p.wait(timeout=600)
   finally:
    if p.poll()is None:p.terminate();p.wait(timeout=30)
    h.close()
   executions.append(dict(model=model,pid=p.pid,exit_code=code,tasks=conf['cpu_tasks']))
   dump(root/'execution.json',dict(executions=executions,model_forward=json.loads((root/'shared-budget.json').read_text())['counts'],remaining_models=[m for m in s.MODELS if m not in[x['model']for x in executions]]))
   if code:raise RuntimeError('CPU exact shape check failed; cost retained')
 finally:
  dump(root/'final.json',dict(executions=executions,budget=json.loads((root/'shared-budget.json').read_text()),planned=54,success=len(executions)==7 and all(x['exit_code']==0 for x in executions),GPU=0,real_observations=0,checkpoints=0))
if __name__=='__main__':main()
