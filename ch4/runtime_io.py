"""Small M7 file/process helpers; no historical launcher import side effects."""
import json
import os
from pathlib import Path
import signal
import tempfile
import time

# Exception/STOP cleanup only; no normal run/epoch timeout.
COOPERATIVE_STOP_SECONDS=10.0
TERM_STOP_SECONDS=5.0
KILL_REAP_SECONDS=5.0
PROCESS_POLL_SECONDS=.05


def proc_start(pid):
    text=Path('/proc/'+str(pid)+'/stat').read_text()
    return text[text.rfind(')')+2:].split()[19]


def process_group_identity(pid):
    text=Path('/proc/'+str(pid)+'/stat').read_text();parts=text[text.rfind(')')+2:].split()
    return dict(pid=int(text.split(' ',1)[0]),proc_start=parts[19],state=parts[0],
                pgid=int(parts[2]),session=int(parts[3]))


def bounded_shutdown(entries):
    """Only registered Popen children; all slots progress together within 20s.

    The caller publishes STOP first. Identity disagreement never grants a kill.
    Unreapable processes are reported, not converted to successful termination.
    """
    began=time.monotonic();events=[]
    def wait_until(deadline):
        while any(e['child'].poll() is None for e in entries) and time.monotonic()<deadline:
            time.sleep(min(PROCESS_POLL_SECONDS,max(0,deadline-time.monotonic())))
    def send(sig):
        for entry in entries:
            child=entry['child'];record=entry['record']
            if child.poll() is not None:continue
            item=dict(pid=child.pid,signal=int(sig),monotonic=time.monotonic())
            try:
                current=process_group_identity(child.pid)
                item['observed']=current
                if record.get('pid')!=child.pid or record.get('proc_start')!=current['proc_start'] or current['pid']!=child.pid or current['pgid']!=child.pid or current['session']!=child.pid:
                    raise PermissionError('owned child/start/independent process group mismatch')
                if record.get('pgid')!=child.pid or record.get('session')!=child.pid:
                    raise PermissionError('missing registered independent group')
                os.killpg(child.pid,sig);item['sent']=True
            except (OSError,ValueError,IndexError) as error:
                item.update(sent=False,error_type=type(error).__name__,error=str(error),returncode_recheck=child.poll())
            events.append(item)
    wait_until(began+COOPERATIVE_STOP_SECONDS)
    send(signal.SIGTERM);wait_until(time.monotonic()+TERM_STOP_SECONDS)
    send(signal.SIGKILL);wait_until(time.monotonic()+KILL_REAP_SECONDS)
    return dict(elapsed_seconds=time.monotonic()-began,events=events,
                children=[dict(pid=e['child'].pid,proc_start=e['record'].get('proc_start'),returncode=e['child'].poll(),
                               exit_confirmed=e['child'].poll() is not None) for e in entries],
                all_exited=all(e['child'].poll() is not None for e in entries))


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name);identity=os.fstat(descriptor)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        # Same filesystem, atomic visibility, and no replacement of existing identity.
        os.link(temporary,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        try:current=temporary.lstat()
        except FileNotFoundError:current=None
        if current is not None and (current.st_dev,current.st_ino)==(identity.st_dev,identity.st_ino):
            temporary.unlink()
            directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)


def publish_complete(path,value):
    """Publish closed, fsynced bytes without replacing an existing completion."""
    path=Path(path);staging=path.with_name(path.name+'.staging')
    with staging.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(staging,path)  # atomic, same directory/filesystem; refuses overwrite
    staging.unlink()
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
