"""Supervisor-owned worker lifecycle and persisted-progress accounting only."""
import datetime
import json
from pathlib import Path
import time
from .imports import sha
from .runtime_io import proc_start,read,write_new


def stamp():
    unix=time.time()
    return dict(utc=datetime.datetime.fromtimestamp(unix,datetime.timezone.utc).isoformat(),unix=unix)


def log_position(path):
    path=Path(path)
    if not path.exists():return dict(offset=0,inode=None)
    stat=path.stat()
    return dict(offset=stat.st_size,inode=stat.st_ino)


def begin(folder,task,phase):
    folder=Path(folder)
    return dict(schema_version='m7-worker-lifecycle-v1',task_id=task,phase=phase,
                started_at=stamp(),supervisor_monotonic_start=time.monotonic(),
                worker_log_path=str(folder/'logs'/(task+'-'+phase+'.log')),
                access_log_path=str(folder/task/'train-access.jsonl'),
                history_path=str(folder/task/'history.jsonl'),
                access_start=log_position(folder/task/'train-access.jsonl'),
                history_start=log_position(folder/task/'history.jsonl'),
                config_sha256=sha(folder/'config.json'),runtime_sha256=sha(folder/'runtime.json'))


def attach(folder,record,child):
    record['pid']=child.pid
    try:record['proc_start']=proc_start(child.pid)
    except (FileNotFoundError,ProcessLookupError):record['proc_start']=None
    write_new(Path(folder)/'lifecycles'/(record['task_id']+'-'+record['phase']+'-start.json'),record)


def progress(path,position,kind):
    """Count only complete, valid new records; never deserialize checkpoints."""
    path=Path(path);errors=[];count=0;last=None;maximum=None
    if not path.exists():
        if position['offset']:errors.append('log missing after nonzero starting offset')
        return dict(count=0,last=None,max_update=None,sha256=None,errors=errors)
    stat=path.stat()
    if position['inode'] is not None and position['inode']!=stat.st_ino:
        errors.append('log inode changed; new-worker range is ambiguous')
        return dict(count=0,last=None,max_update=None,sha256=sha(path),errors=errors)
    if stat.st_size<position['offset']:
        errors.append('log truncated below starting offset')
        return dict(count=0,last=None,max_update=None,sha256=sha(path),errors=errors)
    with path.open('rb') as stream:
        stream.seek(position['offset'])
        for number,line in enumerate(stream,1):
            if not line.endswith(b'\n'):
                errors.append(f'new record {number}: truncated line');continue
            try:
                row=json.loads(line)
                if not isinstance(row,dict):raise ValueError('record is not an object')
                if kind=='updates':
                    if row.get('event') not in ('before_forward','after_forward','optimizer_completed'):
                        raise ValueError('missing/unknown access event')
                    if row['event']!='optimizer_completed':continue
                value=row['updates' if kind=='updates' else 'epoch'];update=row['updates']
                if type(value) is not int or value<=0 or type(update) is not int or update<0:
                    raise ValueError('invalid epoch/update integer')
                if last is not None and value<=last:raise ValueError('non-increasing/duplicate completed record')
                if maximum is not None and update<maximum:raise ValueError('cumulative update regressed')
                if last is not None and value>last+1:
                    errors.append(f'new record {number}: missing {kind} range {last+1}..{value-1} (between {last} and {value})')
                count+=1;last=value;maximum=update if maximum is None else max(maximum,update)
            except (ValueError,KeyError,TypeError,UnicodeError) as error:
                errors.append(f'new record {number}: {type(error).__name__}: {error}')
    return dict(count=count,last=last,max_update=maximum,sha256=sha(path),errors=errors)


def record_exception(folder,task,phase,error):
    write_new(Path(folder)/'worker-errors'/(task+'-'+phase+'.json'),
              dict(exception_type=type(error).__module__+'.'+type(error).__qualname__,
                   exception_message=str(error),stop_interruption=isinstance(error,InterruptedError),recorded_at=stamp()))


def finish(folder,record,returncode):
    """Called after poll/wait confirms exit. elapsed ends at observed exit."""
    folder=Path(folder);name=record['task_id']+'-'+record['phase']
    ended=stamp();elapsed=time.monotonic()-record['supervisor_monotonic_start']
    errors=[];exception_type=None;exception_message=None;exception_source=None
    reason='signal_exit' if returncode<0 else 'nonzero_exit' if returncode else None
    stop=None
    if (folder/'STOP').exists():
        try:stop=dict(path=str(folder/'STOP'),sha256=sha(folder/'STOP'),record=read(folder/'STOP'))
        except (OSError,ValueError) as error:
            stop=dict(path=str(folder/'STOP'),unreadable=str(error));errors.append('STOP record unreadable: '+str(error))
    diagnostic=folder/'worker-errors'/(name+'.json')
    if diagnostic.exists():
        try:
            detail=read(diagnostic);exception_type=detail['exception_type'];exception_message=detail['exception_message']
            exception_source='worker_structured_record'
            reason='stop_interruption' if detail['stop_interruption'] and stop is not None else 'worker_exception'
        except (OSError,ValueError,KeyError,TypeError) as error:errors.append('worker exception record invalid: '+str(error))
    receipt=folder/'receipts'/(name+'.json')
    if not returncode and reason is None:
        try:
            value=read(receipt)
            if not isinstance(value,dict):raise ValueError('receipt root must be a dict')
            if not isinstance(value.get('files'),dict):raise ValueError('receipt files must be a dict')
            if value['task_id']!=record['task_id'] or value['phase']!=record['phase']:raise ValueError('receipt task/phase mismatch')
            for rel,expected in value['files'].items():
                if Path(rel).is_absolute() or '..' in Path(rel).parts or not rel.startswith(record['task_id']+'/'):
                    raise ValueError('receipt outside task')
                if sha(folder/rel)!=expected:raise ValueError('receipt file changed: '+rel)
        except (OSError,ValueError,KeyError,TypeError) as error:
            reason='missing_success_receipt' if not receipt.exists() else 'invalid_success_receipt'
            exception_type=type(error).__module__+'.'+type(error).__qualname__;exception_message=str(error)
            exception_source='supervisor_receipt_validation'
    worker_log=Path(record['worker_log_path'])
    try:log_sha=sha(worker_log)
    except OSError as error:log_sha=None;errors.append('worker log unreadable: '+str(error))
    result=dict(record,ended_at=ended,elapsed_seconds=elapsed,returncode=returncode,
                elapsed_definition='supervisor monotonic before Popen to observed exit, includes polling delay',
                failure_reason=reason,exception_type=exception_type,exception_message=exception_message,
                exception_source=exception_source,worker_log_sha256=log_sha,stop_record=stop)
    failure_ref=None
    if reason is not None:
        values=[]
        for key,kind in [('access','updates'),('history','epochs')]:
            path=record['access_log_path' if key=='access' else 'history_path']
            try:value=progress(path,record[key+'_start'],kind)
            except OSError as error:value=dict(count=0,last=None,max_update=None,sha256=None,errors=[str(error)])
            values.append(value);errors.extend(key+': '+error for error in value['errors'])
        access,history=values
        result.update(access_log_sha256=access['sha256'],history_sha256=history['sha256'],
                      completed_updates=access['count'],completed_epochs=history['count'],
                      last_completed_epoch=history['last'],last_confirmed_total_update=access['max_update'],
                      accounting_complete=not errors,accounting_errors=errors,
                      accounting_basis='new complete persisted optimizer_completed/history records only; not checkpoint recoverability')
        path=folder/'failures'/(name+'.json');write_new(path,result)
        failure_ref=dict(path=str(path),sha256=sha(path))
    write_new(folder/'lifecycles'/(name+'-end.json'),dict(result,failure_receipt=failure_ref))
    return failure_ref
