"""Narrow EL7 3.10 process-leader/driver PID binding; no CUDA operations.

Linux v3.10 kernel/sched/debug.c proc_sched_show_task prints task_struct.pid.
This is NOT a portable NSpid resolver. Unknown kernel/proc semantics fail closed.
"""
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
import time
import uuid as uuid_module

from .contract import write_new

SUPPORTED_KERNEL = '3.10.0-1127.el7.x86_64'
COMMAND = ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader']


def capture_self(destination):
    """Capture fresh evidence, leaving partial raw reads on any exception."""
    destination.update(local_pid=os.getpid(), kernel=platform.release(),
                       main_thread=threading.current_thread() is threading.main_thread())
    for name in ('stat', 'status', 'sched'):
        destination[name] = Path('/proc/self', name).read_text()
    destination['pid_namespace'] = os.readlink('/proc/self/ns/pid')
    return destination


def parse_identity(raw):
    if raw['kernel'] != SUPPORTED_KERNEL or raw['main_thread'] is not True:
        raise ValueError('unsupported kernel or non-leader calling thread')
    local = raw['local_pid']
    if not isinstance(local, int) or local <= 0:
        raise ValueError('invalid local PID')
    status = {}
    for line in raw['status'].splitlines():
        key, sep, value = line.partition(':')
        if sep:
            if key in status:
                raise ValueError('duplicate proc status field')
            status[key] = value.strip()
    if 'NSpid' in status or 'NStgid' in status:
        raise ValueError('unexpected namespace fields for reviewed proc semantics')
    if int(status['Pid']) != local or int(status['Tgid']) != local:
        raise ValueError('status PID/Tgid is not the current leader')
    stat_pid, tail = raw['stat'].split(' (', 1)
    fields = tail.rsplit(') ', 1)[1].split()
    start = int(fields[19])  # stat field 22, after state (field 3)
    if int(stat_pid) != local or start <= 0:
        raise ValueError('stat PID/start mismatch')
    namespace = raw['pid_namespace']
    if not re.fullmatch(r'pid:\[\d+\]', namespace):
        raise ValueError('PID namespace identity unavailable')
    first = raw['sched'].splitlines()[0]
    match = re.fullmatch(r'.+ \(([1-9]\d*), #threads: ([1-9]\d*)\)', first)
    if match is None:
        raise ValueError('invalid proc sched leader header')
    return dict(local_pid=local, tgid=local, start=start, pid_namespace=namespace,
                driver_pid=int(match.group(1)), kernel=raw['kernel'],
                mapping_source='/proc/self/sched: proc_sched_show_task p->pid',
                numbering_space='initial-kernel-task-pid/NVML-host-pid')


def parse_gpu_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        parts = [v.strip() for v in line.split(',')]
        if len(parts) != 2 or not re.fullmatch(r'[1-9]\d*', parts[0]) or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', parts[1]):
            raise ValueError('unparseable GPU process record')
        row = dict(pid=int(parts[0]), gpu_uuid=parts[1])
        if row in rows:
            raise ValueError('duplicate GPU process record')
        rows.append(row)
    return rows


def query_processes(attempts):
    """Exactly timeout60, at most three tries, only TimeoutExpired retries."""
    for number in range(1, 4):
        record = dict(command=list(COMMAND), attempt=number, timeout_seconds=60,
                      started_unix=time.time(), returncode=None, stdout='', stderr='')
        attempts.append(record)
        try:
            result = subprocess.run(COMMAND, capture_output=True, text=True, timeout=60)
            record.update(returncode=result.returncode, stdout=result.stdout,
                          stderr=result.stderr, ended_unix=time.time())
            if result.returncode != 0:
                raise RuntimeError('GPU process query nonzero exit')
            return result.stdout
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
            record.update(stdout=decoded(exc.stdout), stderr=decoded(exc.stderr),
                          ended_unix=time.time(), error='TimeoutExpired')
            if number == 3:
                raise
        except Exception as exc:
            record.update(ended_unix=time.time(), error=type(exc).__name__)
            raise


def check_gpu_identity(gpu_uuid, evidence_dir, stage, expect_context):
    """Persist accepted AND rejected raw evidence before returning/raising.

    Empty before CUDA; exactly the mapped leader after CUDA. A local PID is
    never a second whitelist entry in the driver numbering space.
    """
    folder = Path(evidence_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(stage+'-'+uuid_module.uuid4().hex+'.json')
    evidence = dict(schema='m6-gpu-pid-boundary-v1', stage=stage,
                    target_uuid=gpu_uuid, expect_context=expect_context,
                    started_unix=time.time(), before={}, after={}, queries=[],
                    records=[], target_pids=[], self_pids=[], foreign_pids=[],
                    classification='unknown', accepted=False)
    try:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', stage) or type(expect_context) is not bool:
            raise ValueError('invalid boundary stage/context')
        if not re.fullmatch(r'GPU-[A-Za-z0-9-]+', gpu_uuid):
            raise ValueError('invalid target UUID')
        capture_self(evidence['before'])
        identity = parse_identity(evidence['before'])
        evidence['identity'] = identity
        stdout = query_processes(evidence['queries'])
        capture_self(evidence['after'])
        if identity != parse_identity(evidence['after']):
            raise ValueError('process identity changed across GPU query')
        records = parse_gpu_rows(stdout)
        evidence['records'] = records
        pids = sorted(r['pid'] for r in records if r['gpu_uuid'] == gpu_uuid)
        own = identity['driver_pid']
        evidence.update(target_pids=pids, self_pids=[p for p in pids if p == own],
                        foreign_pids=[p for p in pids if p != own])
        if expect_context:
            if evidence['foreign_pids']:
                evidence['classification'] = 'foreign'
                raise RuntimeError('foreign target GPU process; no process stopped')
            if pids != [own]:
                raise RuntimeError('mapped self GPU record missing after context creation')
            evidence['classification'] = 'self'
        else:
            if pids:
                evidence['classification'] = 'foreign' if evidence['foreign_pids'] else 'unexpected-self-context'
                raise RuntimeError('target GPU must be empty before CUDA context')
            evidence['classification'] = 'empty-before-context'
        evidence['accepted'] = True
    except Exception as exc:
        evidence.update(error_type=type(exc).__name__, error=str(exc), ended_unix=time.time())
        write_new(path, evidence)
        raise
    evidence['ended_unix'] = time.time()
    write_new(path, evidence)
    return dict(evidence_path=str(path), identity=evidence['identity'],
                target_pids=evidence['target_pids'], classification=evidence['classification'])


def measure_with_identity(measurement, gpu_uuid, evidence_dir):
    """Management checks stay outside the complete warmup/timing callable."""
    before = check_gpu_identity(gpu_uuid, evidence_dir, 'before-timing', True)
    result = measurement()
    # Keep measured groups/cost even if the post-boundary detects foreign load.
    # This file is evidence, not a successful job receipt.
    write_new(Path(evidence_dir)/'measurement-completed.json', result)
    after = check_gpu_identity(gpu_uuid, evidence_dir, 'after-timing', True)
    result['gpu_process_boundary'] = dict(before=before, after=after,
                                         telemetry='boundary only; shared mutex held throughout')
    return result
