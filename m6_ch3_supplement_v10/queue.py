"""One finite owned queue; remote Git is checked only before runtime creation."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import math
from .contract import ROOT, SCOPE, read, sha, load_ref, write_new, validate_permission
from moderntcn_input_ablation_v10.launcher import proc_start, gpu_startup


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, timeout=30).strip()


def preflight(config_path, auth_path):
    auth = read(auth_path)
    c = validate_permission(config_path, auth)
    head = git('rev-parse', 'HEAD')
    if head != auth['commit'] or git('status', '--porcelain') or git('rev-parse', '@{upstream}') != head:
        raise PermissionError('clean bound HEAD/tracking required')
    remote = git('ls-remote', 'origin', 'refs/heads/'+git('branch', '--show-current')).split()
    if not remote or remote[0] != head:
        raise PermissionError('remote HEAD mismatch')
    if (Path(c['result_root'])/c['attempt']).exists():
        raise FileExistsError('attempt already used')
    return dict(training_started=False, GPU_query=False, config=c, auth=auth)


def owned_stop(folder):
    folder = Path(folder)
    owner = read(folder/'owner.json')
    runtime = read(folder/'runtime.json')
    c = load_ref(runtime['config_ref'])
    if runtime.get('scope') != SCOPE or owner['runtime_sha256'] != sha(folder/'runtime.json') or folder.resolve() != (Path(c['result_root'])/c['attempt']).resolve():
        raise PermissionError('STOP requires this exact owned supplement attempt')
    if proc_start(owner['pid']) != owner['start']:
        raise ValueError('owner PID/start no longer matches')
    if not (folder/'STOP').exists():
        write_new(folder/'STOP', dict(owner=owner, requested=time.time()))


def local_context(config_path, runtime_path):
    runtime = read(runtime_path)
    auth = read(runtime['auth_ref']['path'])
    if sha(runtime['auth_ref']['path']) != runtime['auth_ref']['sha256']:
        raise PermissionError('AUTH changed')
    c = validate_permission(config_path, auth)
    if runtime['commit'] != auth['commit'] or git('rev-parse', 'HEAD') != auth['commit'] or git('status', '--porcelain'):
        raise PermissionError('local committed source changed')
    owner = read(Path(runtime_path).parent/'owner.json')
    if owner['runtime_sha256'] != sha(runtime_path) or proc_start(owner['pid']) != owner['start']:
        raise PermissionError('owned runtime required')
    return c


def validate_result(config, job, result):
    phase = job['phase']
    if phase == 'E':
        if result['warmup_calls'] != 50 or result['timed_calls'] != 500 or result['batch'] != job['batch'] or len(result['groups']) != 5:
            raise ValueError('measurement coverage')
        if [g['group'] for g in result['groups']] != list(range(5)) or any(g['calls'] != 100 or not math.isfinite(g['wall_seconds']) or g['wall_seconds'] <= 0 for g in result['groups']):
            raise ValueError('measurement group integrity')
    elif phase == 'S':
        t = config['sensitivity_prototypes'][job['dataset']]
        epochs = result['epochs']
        if not 1 <= epochs <= 20 or result['updates'] != epochs*t['arithmetic']['train_batches'] or result['test_calls'] != 0 or result['status'] != 'validation_only_completed':
            raise ValueError('sensitivity completion arithmetic')
        if result['routes'] != dict(train=result['updates'], validation=epochs*t['arithmetic']['eval_batches']['validation']) or not 1 <= result['best_epoch'] <= epochs:
            raise ValueError('sensitivity validation/best coverage')
        if not all(math.isfinite(result['validation'][k]) for k in ('mse', 'mae')):
            raise ValueError('nonfinite saved validation')
    elif phase in ('C', 'P'):
        from .readonly import source
        expected = 336 if phase == 'C' else source(config, job)['data']['window_counts']['validation']
        if result['presentations'] != expected or result['forward_calls'] != (expected+127)//128 or result['test'] != 0 or result['metrics']['count_per_position'] != expected:
            raise ValueError('read-only sample/batch coverage')
    elif result.get('status') != 'completed' or result.get('test') != 0:
        raise ValueError('offline publication incomplete')


def run_job(config_path, runtime_path, job_id):
    c = local_context(config_path, runtime_path)
    job = next(j for j in c['jobs'] if j['id'] == job_id)
    folder = Path(runtime_path).parent
    def stop():
        if (folder/'STOP').exists():
            raise InterruptedError('owned STOP')
    stop()
    if job['phase'] == 'S':
        from .sensitivity import run
        result = run(c, job, folder, stop, read(runtime_path)['gpu']['uuid'])
    elif job['phase'] in ('E', 'C', 'P'):
        from .readonly import run
        result = run(c, job, folder, stop)
    else:
        from .plots import completed
        result = completed(c, folder)
    validate_result(c, job, result)
    output = folder/('plots' if job['phase'] == 'O' else job_id)
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    if not files:
        raise ValueError('no saved job artifacts')
    write_new(folder/'receipts'/(job_id+'.json'), dict(job=job, result=result, files=files, runtime_sha256=sha(runtime_path)))


def supervise(config_path, auth_path):
    admitted = preflight(config_path, auth_path)
    c, auth = admitted['config'], admitted['auth']
    folder = Path(c['result_root'])/c['attempt']
    with open(c['global_lock'], 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise PermissionError('explicit GPU 0 required')
        gpu = gpu_startup(auth)
        # Do not stop unrelated jobs. Refuse polluted measurements.
        for retry in range(3):
            try:
                active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True, timeout=60)
                break
            except subprocess.TimeoutExpired:
                if retry == 2:
                    raise
        if any(auth['gpu_uuid'] in line for line in active.splitlines()):
            raise RuntimeError('unrelated GPU compute load; no process was stopped')
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'logs').mkdir()
        (folder/'receipts').mkdir()
        runtime_path = folder/'runtime.json'
        write_new(runtime_path, dict(scope=c['scope'], commit=auth['commit'], config_ref=dict(path=str(config_path), sha256=sha(config_path)), auth_ref=dict(path=str(auth_path), sha256=sha(auth_path)), gpu=gpu, waves=c['waves']))
        write_new(folder/'owner.json', dict(pid=os.getpid(), start=proc_start(os.getpid()), runtime_sha256=sha(runtime_path)))
        children = []
        try:
            from .readonly import prepare_case
            prepare_case(c, folder)
            for wave_number, wave in enumerate(c['waves'], 1):
                if (folder/'STOP').exists():
                    raise InterruptedError('owned STOP')
                children = []
                for job in wave:
                    log = (folder/'logs'/(job+'.log')).open('x')
                    child = subprocess.Popen([sys.executable, '-m', 'm6_ch3_supplement_v10.entry', 'worker', '--config', str(config_path), '--runtime', str(runtime_path), '--job', job], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    log.close()
                    children.append((job, child))
                write_new(folder/f'children-wave-{wave_number}.json', [dict(job=j, pid=p.pid, start=proc_start(p.pid)) for j, p in children])
                while any(p.poll() is None for _, p in children):
                    if any(p.poll() not in (None, 0) for _, p in children) and not (folder/'STOP').exists():
                        write_new(folder/'STOP', dict(reason='owned child technical failure'))
                    time.sleep(1)
                if any(p.returncode != 0 for _, p in children):
                    raise RuntimeError('wave failed; subsequent stages withheld')
                for j, _ in children:
                    receipt = read(folder/'receipts'/(j+'.json'))
                    if receipt['runtime_sha256'] != sha(runtime_path) or receipt['job']['id'] != j:
                        raise ValueError('receipt identity')
                    validate_result(c, receipt['job'], receipt['result'])
                    for name, expected in receipt['files'].items():
                        target = (folder/name).resolve()
                        if not target.is_relative_to(folder.resolve()) or sha(target) != expected:
                            raise ValueError('job payload checksum changed')
                write_new(folder/f'wave-{wave_number}-complete.json', dict(jobs=wave))
            if (folder/'STOP').exists():
                raise InterruptedError('STOP received before final publication')
            write_new(folder/'complete.json', dict(technical_complete=True, jobs=len(c['jobs']), fresh_sensitivity=8, test=0, scientific_claim='no replacement of main table'))
        except BaseException:
            if any(p.poll() is None for _, p in children):
                if not (folder/'STOP').exists():
                    write_new(folder/'STOP', dict(reason='supervisor technical failure; owned children stop at safe boundary'))
                for _, child in children:
                    child.wait()
            write_new(folder/'failure.json', dict(traceback=traceback.format_exc()))
            raise


def status(folder):
    folder = Path(folder)
    receipts = [read(p) for p in sorted((folder/'receipts').glob('*.json'))]
    by_phase = {phase: sum(r['job']['phase'] == phase for r in receipts) for phase in ('E', 'C', 'P', 'S', 'O')}
    epochs = {}
    for path in sorted(folder.glob('S-*/history.jsonl')):
        rows = path.read_text().splitlines()
        if rows:
            try:
                last = json.loads(rows[-1])
                epochs[path.parent.name] = {k: last[k] for k in ('epoch', 'steps', 'best_epoch', 'bad')}
            except json.JSONDecodeError:
                epochs[path.parent.name] = dict(partial_last_line=True)
    active = []
    for path in sorted(folder.glob('children-wave-*.json')):
        for child in read(path):
            try:
                if proc_start(child['pid']) == child['start']:
                    active.append(child)
            except (FileNotFoundError, ProcessLookupError):
                pass
    return dict(completed_by_phase=by_phase, fresh_completed=by_phase['S'],
                waves_completed=len(list(folder.glob('wave-*-complete.json'))),
                epochs=epochs, active_owned_children=active,
                complete=(folder/'complete.json').exists(), failure=(folder/'failure.json').exists(),
                STOP=(folder/'STOP').exists())
