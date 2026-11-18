"""Supervisor-owned worker lifecycle and persisted-progress accounting only."""
import datetime
import hashlib
import json
from pathlib import Path
import time
from .imports import sha
from .runtime_io import proc_start,process_group_identity,read,write_new


def stamp():
    unix=time.time()
    return dict(utc=datetime.datetime.fromtimestamp(unix,datetime.timezone.utc).isoformat(),unix=unix)


def log_position(path):
    path=Path(path)
    if not path.exists():return dict(offset=0,inode=None)
    stat=path.stat()
    return dict(offset=stat.st_size,inode=stat.st_ino)


def snapshot_sha(path,size):
    """Hash only a captured finite extent, including for unreaped live writers."""
    digest=hashlib.sha256();remaining=size
    with Path(path).open('rb') as stream:
        while remaining:
            data=stream.read(min(remaining,1024*1024))
            if not data:raise OSError('file truncated during bounded snapshot')
            digest.update(data);remaining-=len(data)
    return digest.hexdigest()


def begin(folder,task,phase):
    folder=Path(folder)
    if phase not in ('train','test'):raise ValueError('lifecycle phase')
    return dict(schema_version='m7-worker-lifecycle-v2',task_id=task,phase=phase,
                started_at=stamp(),supervisor_monotonic_start=time.monotonic(),
                worker_log_path=str(folder/'logs'/(task+'-'+phase+'.log')),
                access_log_path=str(folder/task/(phase+'-access.jsonl')),
                history_path=str(folder/task/'history.jsonl'),
                access_start=log_position(folder/task/(phase+'-access.jsonl')),
                history_start=log_position(folder/task/'history.jsonl'),
                config_sha256=sha(folder/'config.json'),runtime_sha256=sha(folder/'runtime.json'))


def attach(folder,record,child):
    record['pid']=child.pid
    try:
        record.update(process_group_identity(child.pid))
        if record['pgid']!=child.pid or record['session']!=child.pid:raise PermissionError('worker requires independent session/group')
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
        return dict(count=0,last=None,max_update=None,sha256=snapshot_sha(path,stat.st_size),bytes_hashed=stat.st_size,errors=errors)
    if stat.st_size<position['offset']:
        errors.append('log truncated below starting offset')
        return dict(count=0,last=None,max_update=None,sha256=snapshot_sha(path,stat.st_size),bytes_hashed=stat.st_size,errors=errors)
    with path.open('rb') as stream:
        stream.seek(position['offset'])
        remaining=stat.st_size-position['offset'];number=0
        while remaining:
            line=stream.readline(remaining);number+=1
            if not line:errors.append('log truncated during bounded read');break
            remaining-=len(line)
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
    return dict(count=count,last=last,max_update=maximum,sha256=snapshot_sha(path,stat.st_size),bytes_hashed=stat.st_size,errors=errors)


def record_exception(folder,task,phase,error):
    write_new(Path(folder)/'worker-errors'/(task+'-'+phase+'.json'),
              dict(exception_type=type(error).__module__+'.'+type(error).__qualname__,
                   exception_message=str(error),stop_interruption=isinstance(error,InterruptedError),recorded_at=stamp()))


def test_progress(path,position):
    """before batch b (0-based) pairs with after batch b+1, never same b.

    Samples are synchronous origins, not nodes or prediction positions. Unknown
    records stay unknown; only complete pairs contribute confirmed completion.
    """
    path=Path(path);errors=[];started={};done={};last=None
    result=dict(started_batches=0,completed_batches=0,started_samples=0,completed_samples=0,
                unconfirmed_batches=[],unconfirmed_samples=None,additional_unobserved_amount_unknown=True,
                batch_ranges=[],sha256=None,errors=errors)
    if not path.exists():errors.append('test access log missing');return result
    stat=path.stat();result['sha256']=snapshot_sha(path,stat.st_size);result['bytes_hashed']=stat.st_size
    if position['inode'] is not None and stat.st_ino!=position['inode']:errors.append('test access inode changed');return result
    if stat.st_size<position['offset']:errors.append('test access truncated below starting offset');return result
    with path.open('rb') as stream:
        stream.seek(position['offset'])
        remaining=stat.st_size-position['offset'];number=0
        while remaining:
            line=stream.readline(remaining);number+=1
            if not line:errors.append('test log truncated during bounded read');break
            remaining-=len(line)
            try:
                if not line.endswith(b'\n'):raise ValueError('truncated line')
                row=json.loads(line)
                if not isinstance(row,dict) or row.get('phase')!='test':raise ValueError('not a test event')
                b=row['batch']
                if type(b) is not int:raise ValueError('batch is not integer')
                if row['event']=='before_forward':
                    if b<0 or b in started:raise ValueError('invalid/duplicate before batch')
                    origins=row['origins']
                    if not isinstance(origins,list) or not origins:raise ValueError('missing origin count')
                    if last is not None and b!=last+1:errors.append(f'new record {number}: before sequence discontinuity {last}->{b}')
                    started[b]=len(origins);last=b
                elif row['event']=='after_forward':
                    index=b-1
                    if index not in started or index in done:raise ValueError('after without unique matching before b-1')
                    if type(row['samples']) is not int or row['samples']!=started[index]:raise ValueError('after sample count mismatch')
                    done[index]=row['samples']
                else:raise ValueError('unknown test event')
            except (ValueError,KeyError,TypeError,UnicodeError) as error:
                errors.append(f'new record {number}: {type(error).__name__}: {error}')
    pending=sorted(set(started)-set(done))
    if not started:errors.append('no valid test before_forward records')
    if pending:errors.append('started test batches lack complete after_forward: '+str(pending))
    # Observed sample ranges are relative to this worker's newly logged origins;
    # a sequence gap does not imply an invented missing sample count.
    offset=0
    for b,n in started.items():
        result['batch_ranges'].append(dict(batch_index=b,observed_sample_range=[offset,offset+n],samples=n,completed=b in done));offset+=n
    result.update(started_batches=len(started),completed_batches=len(done),started_samples=sum(started.values()),
                  completed_samples=sum(done.values()),unconfirmed_batches=pending,unconfirmed_samples=sum(started[b] for b in pending),
                  additional_unobserved_amount_unknown=bool(errors))
    return result


def finish(folder,record,returncode):
    """Called after poll/wait confirms exit. elapsed ends at observed exit."""
    folder=Path(folder);name=record['task_id']+'-'+record['phase']
    ended=stamp();elapsed=time.monotonic()-record['supervisor_monotonic_start']
    errors=[];exception_type=None;exception_message=None;exception_source=None
    reason='exit_unconfirmed' if returncode is None else 'signal_exit' if returncode<0 else 'nonzero_exit' if returncode else None
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
            if returncode is not None:reason='stop_interruption' if detail['stop_interruption'] and stop is not None else 'worker_exception'
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
    try:
        log_bytes=worker_log.stat().st_size;log_sha=snapshot_sha(worker_log,log_bytes)
    except OSError as error:log_sha=None;errors.append('worker log unreadable: '+str(error))
    result=dict(record,ended_at=ended,elapsed_seconds=elapsed,returncode=returncode,
                exit_confirmed=returncode is not None,
                elapsed_definition='supervisor monotonic before Popen to observed exit or bounded cleanup end if exit unconfirmed',
                failure_reason=reason,exception_type=exception_type,exception_message=exception_message,
                exception_source=exception_source,worker_log_sha256=log_sha,stop_record=stop)
    result.update(log_hash_scope='captured byte prefixes; whole files only if closed and unchanged',worker_log_bytes=locals().get('log_bytes'))
    failure_ref=None
    if record['phase']=='test':
        success_receipt=value if reason is None else None
        try:value=test_progress(record['access_log_path'],record['access_start'])
        except OSError as error:value=dict(errors=[str(error)],sha256=None,started_batches=0,completed_batches=0,started_samples=0,completed_samples=0,unconfirmed_batches=[],unconfirmed_samples=None,batch_ranges=[])
        errors.extend('test access: '+error for error in value['errors'])
        if success_receipt is not None:
            try:
                reported=success_receipt['result']
                if (value['completed_batches'],value['completed_samples'])!=(reported['batches'],reported['origins']):
                    errors.append('test access totals differ from success receipt batches/origins')
            except (KeyError,TypeError):errors.append('test success receipt lacks batch/sample totals')
        if returncode is None:errors.append('worker exit unconfirmed; log may still change')
        result.update(test_progress=value,access_log_sha256=value['sha256'],history_sha256=None,
                      access_log_bytes=value.get('bytes_hashed'),history_bytes=None,
                      completed_updates=None,completed_epochs=None,last_completed_epoch=None,last_confirmed_total_update=None,
                      accounting_complete=not errors,accounting_errors=errors,
                      accounting_basis='test before b / after b+1 complete pairs and observed synchronous-origin sample ranges; no optimizer accounting')
        if errors and reason is None:
            reason='test_accounting_incomplete';result['failure_reason']=reason
    elif reason is not None:
        values=[]
        for key,kind in [('access','updates'),('history','epochs')]:
            path=record['access_log_path' if key=='access' else 'history_path']
            try:value=progress(path,record[key+'_start'],kind)
            except OSError as error:value=dict(count=0,last=None,max_update=None,sha256=None,errors=[str(error)])
            values.append(value);errors.extend(key+': '+error for error in value['errors'])
        access,history=values
        if returncode is None:errors.append('worker exit unconfirmed; log may still change')
        result.update(access_log_sha256=access['sha256'],history_sha256=history['sha256'],
                      access_log_bytes=access.get('bytes_hashed'),history_bytes=history.get('bytes_hashed'),
                      completed_updates=access['count'],completed_epochs=history['count'],
                      last_completed_epoch=history['last'],last_confirmed_total_update=access['max_update'],
                      accounting_complete=not errors,accounting_errors=errors,
                      accounting_basis='new complete persisted optimizer_completed/history records only; not checkpoint recoverability')
    if reason is not None:
        path=folder/'failures'/(name+'.json');write_new(path,result)
        failure_ref=dict(path=str(path),sha256=sha(path))
    write_new(folder/'lifecycles'/(name+'-end.json'),dict(result,failure_receipt=failure_ref))
    return failure_ref
