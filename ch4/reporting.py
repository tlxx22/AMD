"""Complete-scope MSE/MAE tables. No winner/case selection."""
import csv
import math
import subprocess
import hashlib
from pathlib import Path
from .imports import sha
from .runtime_io import read,write_new,publish_complete
from .protocol import validate_plan


def checked_metric(value):
    if not isinstance(value['count'],int) or value['count']<=0:raise ValueError('metric count')
    for key in ('mse','mae','sse','sae'):
        if not math.isfinite(value[key]) or value[key]<0:raise ValueError('metric finite/nonnegative')
    if value['mse']!=value['sse']/value['count'] or value['mae']!=value['sae']/value['count']:
        raise ValueError('metric aggregation identity')


def verify_task_set(config,folder):
    folder=Path(folder);names=[t['id'] for t in config['tasks']]
    if len(names)!=len(set(names)):raise ValueError('duplicate tasks')
    if read(folder/'config.json')!=config:raise ValueError('persisted config differs')
    frozen=read(folder/'frozen-best.json')
    if frozen['config_sha256']!=sha(folder/'config.json'):raise ValueError('freeze config binding')
    if len(frozen['tasks'])!=len(names) or {r['task_id'] for r in frozen['tasks']}!=set(names):raise ValueError('frozen task coverage')
    if {p.name for p in (folder/'claims').iterdir()}!={n+'.json' for n in names}:raise ValueError('claim coverage')
    if {p.name for p in (folder/'receipts').iterdir()}!={n+'-'+phase+'.json' for n in names for phase in ('train','test')}:
        raise ValueError('receipt coverage')
    for required in ('owner.json','runtime.json'):
        if not (folder/required).is_file():raise ValueError('missing '+required)
    return {r['task_id']:r for r in frozen['tasks']}


def verify_payload(folder,files,*,with_controls):
    folder=Path(folder)
    expected=set(files)|({'checksums.sha256'} if with_controls else set())
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('sealed file set changed')
    for name,expected_sha in files.items():
        p=folder/name
        if p.is_symlink() or sha(p)!=expected_sha:raise ValueError('sealed file identity changed: '+name)


def csv_new(path,rows,fields):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def seal(config,folder):
    validate_plan(config)  # production coverage remains exactly 216; fixtures mock admission only
    folder=Path(folder);standard=[];physical=[];positions=[];costs=[];ledger=[]
    if any((folder/p).exists() for p in ('STOP','failure.json','complete.json','complete.json.staging')):raise ValueError('failed/stopped/already sealed attempt')
    frozen=verify_task_set(config,folder)
    for task in config['tasks']:
        name=task['id'];base=dict(task=name,model=task['model'],domain=task['domain'],H=task['H'],P=task['P'],seed=2024)
        train=read(folder/name/'train-result.json');result=read(folder/name/'test-result.json')
        if train['task_id']!=name or result['task_id']!=name or train['training_complete'] is not True or train['test_accesses']!=0:
            raise ValueError('task result identity/completion')
        if train['binding']!=frozen[name]['training_binding']:raise ValueError('frozen training binding')
        if sha(folder/name/'last.pt')!=train['last_sha256'] or train['best_sha256']!=frozen[name]['best_sha256']:raise ValueError('training checkpoint identity')
        claim=read(folder/'claims'/(name+'.json'))
        if claim['task_id']!=name or claim['freeze_sha256']!=sha(folder/'frozen-best.json'):raise ValueError('claim freeze identity')
        if result['best_sha256']!=frozen[name]['best_sha256'] or sha(folder/name/'best.pt')!=result['best_sha256']:raise ValueError('frozen best changed')
        if not (folder/'claims'/(name+'.json')).exists():raise ValueError('missing one-time test claim')
        for phase in ('train','test'):
            receipt=read(folder/'receipts'/(name+'-'+phase+'.json'))
            if receipt['task_id']!=name or receipt['phase']!=phase:raise ValueError('receipt task/phase')
            expected_result=train if phase=='train' else {k:v for k,v in result.items() if k not in ('task_id','best_sha256')}
            if receipt['result']!=expected_result:raise ValueError('receipt result differs from task result')
            required={'resolved.json','history.jsonl','best.pt','last.pt','train-result.json','train-access.jsonl'}
            if phase=='test':required|={'test-result.json','test-access.jsonl',*[f'test_predictions/{k}.npy' for k in ('prediction','truth','mask','origin','node_ids','target_times')]}
            if not {name+'/'+r for r in required}.issubset(receipt['files']):raise ValueError('required task file missing from receipt')
            if phase=='test' and set(receipt['files'])!={str(p.relative_to(folder)) for p in (folder/name).rglob('*') if p.is_file()}:
                raise ValueError('test receipt task file coverage')
            for rel,expected in receipt['files'].items():
                if Path(rel).is_absolute() or '..' in Path(rel).parts or not rel.startswith(name+'/'):raise ValueError('receipt file outside task')
                if sha(folder/rel)!=expected:raise ValueError('receipt bytes changed')
        if read(folder/name/'resolved.json')!=task:raise ValueError('task resolved configuration')
        for key in ('formal_train_standardized','auxiliary_physical_units'):checked_metric(result[key])
        if len(result['positions_train_standardized'])!=task['P']:raise ValueError('prediction position coverage')
        for row in result['positions_train_standardized']:checked_metric(row)
        standard.append(dict(base,scale='train-standardized',mse=result['formal_train_standardized']['mse'],mae=result['formal_train_standardized']['mae']))
        physical.append(dict(base,scale='physical auxiliary',mse=result['auxiliary_physical_units']['mse'],mae=result['auxiliary_physical_units']['mae']))
        if task['P']==12:
            positions.extend(dict(base,position=i+1,mse=row['mse'],mae=row['mae']) for i,row in enumerate(result['positions_train_standardized']))
        costs.append(dict(base,epochs=train['epochs'],updates=train['updates'],worker_elapsed_seconds=train['elapsed_seconds']))
        ledger.append(dict(base,train_result_sha256=sha(folder/name/'train-result.json'),test_result_sha256=sha(folder/name/'test-result.json'),
                           best_sha256=result['best_sha256'],data_metadata_sha256=task['metadata_sha256'],producer=str(folder/name)))
    fields=['task','model','domain','H','P','seed']
    csv_new(folder/'comparison-standardized.csv',standard,fields+['scale','mse','mae'])
    csv_new(folder/'comparison-physical-auxiliary.csv',physical,fields+['scale','mse','mae'])
    csv_new(folder/'pems-position-standardized.csv',positions,fields+['position','mse','mae'])
    csv_new(folder/'actual-training-costs.csv',costs,fields+['epochs','updates','worker_elapsed_seconds'])
    averages=[]
    for model in config['models']:
        rows=[r for r in standard if r['model']==model and r['domain']=='UrbanEV']
        if {r['H'] for r in rows}!={3,6,9,12}:raise ValueError('Urban four-H coverage')
        averages.append(dict(model=model,scale='train-standardized four-H equal weight',mse=sum(r['mse'] for r in rows)/4,mae=sum(r['mae'] for r in rows)/4))
    csv_new(folder/'urban-four-H-standardized.csv',averages,['model','scale','mse','mae'])
    write_new(folder/'source-ledger.json',ledger)
    # All generated tables and ledger are closed before collecting identities.
    if verify_task_set(config,folder)!=frozen:raise ValueError('frozen set changed while writing summaries')
    for task in config['tasks']:
        name=task['id']
        if sha(folder/name/'best.pt')!=frozen[name]['best_sha256']:raise ValueError('best changed before seal')
        claim=read(folder/'claims'/(name+'.json'))
        if claim['task_id']!=name or claim['freeze_sha256']!=sha(folder/'frozen-best.json'):raise ValueError('claim changed before seal')
        for phase in ('train','test'):
            for rel,expected in read(folder/'receipts'/(name+'-'+phase+'.json'))['files'].items():
                if sha(folder/rel)!=expected:raise ValueError('receipt bytes changed before seal')
    for filename,expected_rows in [('comparison-standardized.csv',standard),('comparison-physical-auxiliary.csv',physical),
            ('pems-position-standardized.csv',positions),('actual-training-costs.csv',costs),('urban-four-H-standardized.csv',averages)]:
        with (folder/filename).open(newline='') as stream:actual_rows=list(csv.DictReader(stream))
        if actual_rows!=[{k:str(v) for k,v in row.items()} for row in expected_rows]:raise ValueError('summary differs from verified results: '+filename)
    if read(folder/'source-ledger.json')!=ledger:raise ValueError('source ledger differs')
    files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    verify_payload(folder,files,with_controls=False)
    write_new(folder/'manifest.json',dict(files=files,scope=config['scope'],
        coverage='all payload including generated summaries; excludes manifest/checksums/complete controls'))
    files_with_manifest=dict(files,**{'manifest.json':sha(folder/'manifest.json')})
    with (folder/'checksums.sha256').open('x') as f:
        for p,h in files_with_manifest.items():f.write(h+'  '+p+'\n')
    checksum_sha=sha(folder/'checksums.sha256')
    verify_payload(folder,files_with_manifest,with_controls=True)
    verification=subprocess.run(['sha256sum','--strict','-c','checksums.sha256'],cwd=folder,
                                capture_output=True,text=True,check=True)
    verify_payload(folder,files_with_manifest,with_controls=True)
    if sha(folder/'checksums.sha256')!=checksum_sha:raise ValueError('checksum control changed')
    publish_complete(folder/'complete.json',dict(training_tasks=len(costs),formal_test_evaluations=len(standard),
              epochs=sum(r['epochs'] for r in costs),updates=sum(r['updates'] for r in costs),
              worker_elapsed_sum_seconds=sum(r['worker_elapsed_seconds'] for r in costs),
              elapsed_sum_is_gpu_hours=False,manifest_sha256=sha(folder/'manifest.json'),checksums_sha256=checksum_sha,
              python_verified_files=len(files_with_manifest),system_checksum_returncode=verification.returncode,
              system_checksum_stdout_sha256=hashlib.sha256(verification.stdout.encode()).hexdigest(),
              coverage='checksums includes payload+manifest; excludes itself and complete; complete binds both controls; no self hash',
              scientific_review='Pending'))
