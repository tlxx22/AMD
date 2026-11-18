"""CPU-only regression for control publication; no model/data/GPU imports."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from ch4 import runtime_io as io


def paused_writer(path,value,paused,release,fail=False):
    original=io.json.dump
    def slow(value,stream,**kwargs):
        stream.write('{');stream.flush();paused.set()
        if not release.wait(5):raise TimeoutError('fixture release')
        if fail:raise RuntimeError('fixture before publication')
        stream.seek(0);stream.truncate();return original(value,stream,**kwargs)
    try:
        with patch.object(io.json,'dump',side_effect=slow):io.write_new(path,value)
    except RuntimeError:
        if not fail:raise


def waiting_reader(path,ready,result):
    ready.set();deadline=time.monotonic()+5
    while not Path(path).exists():
        if time.monotonic()>deadline:raise TimeoutError('fixture publication')
        time.sleep(.002)
    result.put(io.read(path))


class AtomicControls(unittest.TestCase):
    def test_owner_runtime_and_stop_not_visible_until_complete(self):
        ctx=mp.get_context('fork')
        with tempfile.TemporaryDirectory() as temp:
            for name in ('runtime.json','owner.json','STOP'):
                path=Path(temp)/name;value={'pid':os.getpid(),'start':io.proc_start(os.getpid()),'payload':'x'*50000}
                paused=ctx.Event();release=ctx.Event();result=ctx.Queue();children=[]
                try:
                    writer=ctx.Process(target=paused_writer,args=(path,value,paused,release));writer.start();children.append(writer)
                    self.assertTrue(paused.wait(5));self.assertFalse(path.exists())
                    for _ in range(2):
                        ready=ctx.Event();reader=ctx.Process(target=waiting_reader,args=(path,ready,result));reader.start();children.append(reader);self.assertTrue(ready.wait(5))
                    self.assertFalse(path.exists());release.set()
                    self.assertEqual(result.get(timeout=5),value);self.assertEqual(result.get(timeout=5),value)
                    for child in children:child.join(5);self.assertEqual(child.exitcode,0)
                    self.assertEqual(io.read(path),value);self.assertEqual(list(Path(temp).glob('.*.tmp')),[])
                finally:
                    release.set()
                    for child in children:
                        if child.is_alive():child.terminate()
                        child.join(5)
    def test_failure_duplicate_and_foreign_temp_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'STOP';foreign=root/'.STOP.foreign.tmp';foreign.write_text('owned elsewhere')
            with self.assertRaises(ValueError):io.write_new(path,{'bad':float('nan')})
            self.assertFalse(path.exists());self.assertEqual(list(root.iterdir()),[foreign])
            io.write_new(path,{'original':1});before=path.read_bytes()
            with self.assertRaises(FileExistsError):io.write_new(path,{'replacement':2})
            self.assertEqual(path.read_bytes(),before);self.assertEqual(foreign.read_text(),'owned elsewhere')
            with patch.object(io.os,'link',side_effect=OSError('synthetic publication failure')):
                with self.assertRaises(OSError):io.write_new(root/'owner.json',{'pid':1})
            self.assertFalse((root/'owner.json').exists());self.assertEqual(set(root.iterdir()),{path,foreign})
    def test_process_writer_failure_never_publishes_GO(self):
        ctx=mp.get_context('fork')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'owner.json';paused=ctx.Event();release=ctx.Event()
            child=ctx.Process(target=paused_writer,args=(path,{'a':1},paused,release,True));child.start()
            try:
                self.assertTrue(paused.wait(5));self.assertFalse(path.exists());release.set();child.join(5)
                self.assertEqual(child.exitcode,0);self.assertFalse(path.exists());self.assertEqual(list(Path(temp).iterdir()),[])
            finally:
                release.set()
                if child.is_alive():child.terminate()
                child.join(5)
    def test_actual_stop_identity_and_runtime_fields(self):
        from ch4 import queue
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);runtime={'config_sha256':'fixture','commit':'fixture'};io.write_new(folder/'runtime.json',runtime)
            owner={'pid':os.getpid(),'start':io.proc_start(os.getpid()),'runtime_sha256':hashlib.sha256((folder/'runtime.json').read_bytes()).hexdigest()}
            io.write_new(folder/'owner.json',owner);queue.owned_stop(folder)
            stop=io.read(folder/'STOP');self.assertEqual(stop['owner'],owner);self.assertIsInstance(stop['requested'],float)
            before=(folder/'STOP').read_bytes();queue.owned_stop(folder);self.assertEqual((folder/'STOP').read_bytes(),before)
    def test_complete_serialization_unchanged_and_fsync_precedes_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'runtime.json';value={'unicode':'中文','a':[1,True,None]};events=[]
            real_link=io.os.link;real_fsync=io.os.fsync
            def sync(fd):events.append('fsync');return real_fsync(fd)
            def link(src,dst):
                self.assertIn('fsync',events);self.assertEqual(io.read(src),value);events.append('publish');return real_link(src,dst)
            with patch.object(io.os,'fsync',side_effect=sync),patch.object(io.os,'link',side_effect=link):io.write_new(p,value)
            self.assertEqual(p.read_text(),json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
            self.assertGreater(len(events)-events.index('publish'),1)

if __name__=='__main__':unittest.main(verbosity=2)
