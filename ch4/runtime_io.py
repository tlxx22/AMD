"""Small M7 file/process helpers; no historical launcher import side effects."""
import json
import os
from pathlib import Path


def proc_start(pid):
    text=Path('/proc/'+str(pid)+'/stat').read_text()
    return text[text.rfind(')')+2:].split()[19]


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


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
