"""Synthetic subprocess producer for supervisor integration; no model/data load."""
import argparse
import os
import signal
import time
from pathlib import Path
import numpy as np
from ch4.runtime_io import read,write_new,proc_start
from ch4.imports import sha
from ch4.queue import claim_test
from ch4.lifecycle import record_exception


def main():
    p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--folder');p.add_argument('--task');p.add_argument('--phase')
    args=p.parse_args();root=Path(args.folder);config=read(root/'config.json');owner=read(root/'owner.json')
    assert config['fixture_only'] is True and args.command=='worker'
    assert os.getppid()==owner['pid'] and proc_start(owner['pid'])==owner['start']
    task=next(t for t in config['tasks'] if t['id']==args.task);folder=root/args.task
    with (root/'fixture-events.jsonl').open('a') as f:f.write(args.phase+' '+args.task+'\n')
    if args.phase=='test':
        binding=claim_test(root,args.task)
        if task.get('fail_phase')=='test' or task.get('test_failure_mode'):
            path=folder/'test-access.jsonl'
            path.write_text('{"event":"before_forward","phase":"test","batch":0,"origins":[12,13]}\n'
                            '{"event":"after_forward","phase":"test","batch":1,"samples":2,"valid":2}\n'
                            '{"event":"before_forward","phase":"test","batch":1,"origins":[14]}\n')
            if task.get('test_failure_mode')=='corrupt':
                with path.open('a') as stream:stream.write('{"event":')
            try:raise RuntimeError('synthetic test failure after partial access')
            except BaseException as error:record_exception(root,args.task,args.phase,error);raise
    if task.get('fail_phase')==args.phase or task.get('failure_mode'):
        folder.mkdir(exist_ok=True)
        with (folder/'train-access.jsonl').open('a') as access:
            for update in ((101,103) if task.get('failure_mode')=='sequence_gap' else (101,102,103)):
                access.write('{"event":"optimizer_completed","updates":'+str(update)+'}\n');access.flush()
            if task.get('failure_mode')=='corrupt':access.write('{"event":');access.flush()
        with (folder/'history.jsonl').open('a') as history:
            history.write('{"epoch":7,"updates":102}\n');history.flush()
            if task.get('failure_mode')=='sequence_gap':history.write('{"epoch":9,"updates":103}\n');history.flush()
        try:
            mode=task.get('failure_mode','exception')
            if mode in ('ignore_stop','ignore_term'):
                if mode=='ignore_term':signal.signal(signal.SIGTERM,signal.SIG_IGN)
                write_new(root/(args.task+'-ready.json'),dict(pid=os.getpid()))
                while True:time.sleep(.02)
            if mode=='supervisor_signal':
                os.kill(os.getppid(),signal.SIGTERM)
                mode='wait_stop'
            if mode in ('stop','wait_stop'):
                if mode=='stop':write_new(root/'STOP',dict(reason='synthetic user STOP'))
                deadline=time.monotonic()+10
                while not (root/'STOP').exists():
                    if time.monotonic()>deadline:raise TimeoutError('fixture STOP timeout')
                    time.sleep(.02)
                raise InterruptedError('owned STOP')
            if mode=='signal':os.kill(os.getpid(),signal.SIGTERM)
            if mode=='missing_receipt':return
            if mode=='invalid_receipt':
                value=dict(task_id=args.task,phase=args.phase,files=task.get('invalid_files',[]))
                if 'invalid_root' in task:value=task['invalid_root']
                write_new(root/'receipts'/(args.task+'-'+args.phase+'.json'),value)
                return
            raise RuntimeError('deliberate synthetic technical failure')
        except BaseException as error:
            record_exception(root,args.task,args.phase,error);raise
    if args.phase=='train':
        assert not (root/'frozen-best.json').exists();folder.mkdir()
        for name in ('best.pt','last.pt'):(folder/name).write_bytes(b'synthetic checkpoint bytes, never deserialized')
        write_new(folder/'resolved.json',task)
        (folder/'history.jsonl').write_text('{"epoch":1,"updates":1}\n')
        (folder/'train-access.jsonl').write_text('{"synthetic":true,"phase":"train/validation"}\n')
        result=dict(task_id=args.task,training_complete=True,test_accesses=0,epochs=1,updates=1,elapsed_seconds=.01,
                    binding=dict(synthetic=True,task=args.task),best_sha256=sha(folder/'best.pt'),last_sha256=sha(folder/'last.pt'))
        write_new(folder/'train-result.json',result)
    else:
        assert all((root/t['id']/'train-result.json').exists() for t in config['tasks'])
        predictions=folder/'test_predictions';predictions.mkdir()
        for key,array in dict(prediction=np.ones((1,1,1)),truth=np.full((1,1,1),.5),mask=np.ones((1,1,1),dtype=bool),
                              origin=np.array([12]),node_ids=np.array(['synthetic-node']),target_times=np.array([['2023-01-01']],dtype='datetime64[D]')).items():
            np.save(predictions/(key+'.npy'),array,allow_pickle=False)
        (folder/'test-access.jsonl').write_text('{"event":"before_forward","phase":"test","batch":0,"origins":[12]}\n'
                                               '{"event":"after_forward","phase":"test","batch":1,"samples":1,"valid":1}\n')
        metric=dict(mse=.25,mae=.5,sse=.25,sae=.5,count=1)
        result=dict(formal_train_standardized=metric,auxiliary_physical_units=metric,positions_train_standardized=[metric],origins=1,batches=1)
        write_new(folder/'test-result.json',dict(result,task_id=args.task,best_sha256=binding['best_sha256']))
    write_new(root/'receipts'/(args.task+'-'+args.phase+'.json'),dict(task_id=args.task,phase=args.phase,result=result,
        files={str(p.relative_to(root)):sha(p) for p in folder.rglob('*') if p.is_file()}))


if __name__=='__main__':main()
