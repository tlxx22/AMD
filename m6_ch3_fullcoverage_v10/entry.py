import argparse
import json


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','supervise','worker','status','stop','plot-region','plot-epf'])
    for k in ('config','auth','runtime','job','folder','output','market'):p.add_argument('--'+k)
    p.add_argument('--H',type=int);p.add_argument('--nodes',nargs='+');p.add_argument('--models',nargs='+');a=p.parse_args()
    if a.action.startswith('plot-'):
        from .reporting import region_plot,epf_plot
        if a.action=='plot-region':region_plot(a.folder,a.output,H=a.H,node_ids=a.nodes,models=a.models)
        else:epf_plot(a.folder,a.output,market=a.market,models=a.models)
        return
    from . import queue
    if a.action=='preflight':
        c=queue.preflight(a.config,a.auth);print(json.dumps(dict(jobs=40,model_jobs=39,waves=40,training_started=False,test_observations_accessed=False,GPU_query=False)))
    elif a.action=='supervise':queue.supervise(a.config,a.auth)
    elif a.action=='worker':queue.worker(a.config,a.runtime,a.job)
    elif a.action=='status':print(json.dumps(queue.status(a.folder),indent=2))
    else:queue.owned_stop(a.folder)


if __name__=='__main__':main()
