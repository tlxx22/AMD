import argparse
import json


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','supervise','worker','status','stop'])
    for k in ('config','auth','runtime','job','folder'):p.add_argument('--'+k)
    a=p.parse_args();from . import queue
    if a.action=='preflight':
        c=queue.preflight(a.config,a.auth);print(json.dumps(dict(jobs=len(c['jobs']),waves=len(c['waves']),training_started=False,test_observations_accessed=False,GPU_query=False)))
    elif a.action=='supervise':queue.supervise(a.config,a.auth)
    elif a.action=='worker':queue.worker(a.config,a.runtime,a.job)
    elif a.action=='status':print(json.dumps(queue.status(a.folder),indent=2))
    else:queue.owned_stop(a.folder)


if __name__=='__main__':main()
