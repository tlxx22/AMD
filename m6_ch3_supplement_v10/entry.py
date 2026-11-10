"""Command dispatcher. Preparation never invokes workload actions."""
import argparse
import json
from .contract import read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preflight', 'supervise', 'worker', 'status', 'stop'])
    p.add_argument('--config')
    p.add_argument('--auth')
    p.add_argument('--runtime')
    p.add_argument('--job')
    p.add_argument('--folder')
    args = p.parse_args()
    from . import queue
    if args.action == 'preflight':
        result = queue.preflight(args.config, args.auth)
        print(json.dumps(dict(training_started=False, GPU_query=False, jobs=len(result['config']['jobs']))))
    elif args.action == 'supervise':
        queue.supervise(args.config, args.auth)
    elif args.action == 'worker':
        queue.run_job(args.config, args.runtime, args.job)
    elif args.action == 'stop':
        queue.owned_stop(args.folder)
    else:
        print(json.dumps(queue.status(args.folder), indent=2))


if __name__ == '__main__':
    main()
