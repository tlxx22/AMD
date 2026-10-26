"""Candidate-only immutable publication with Python/system SHA agreement."""
import json
import os
from pathlib import Path
import subprocess
from .training_state import file_sha,atomic_json
from .training_state import SCHEMA
from utils.ch3_contract import digest,BestState
from .spec import CANDIDATE_ID, REVISION, validate_revision

def audit(path):
    path=Path(path)
    if not path.is_dir() or path.is_symlink():raise ValueError('completed directory required')
    seal=json.loads((path/'seal.json').read_text())
    if seal.get('schema')!='moderntcn-uv-artifact-v1' or seal.get('complete') is not True or seal['checksums_sha256']!=file_sha(path/'checksums.sha256'):raise ValueError('seal/checksum identity')
    expected={}
    for line in (path/'checksums.sha256').read_text().splitlines():
        h,name=line.split('  ',1)
        if '/' in name or name.startswith('.') or name in expected:raise ValueError('checksum path/schema')
        expected[name]=h
    actual={p.name:file_sha(p) for p in path.iterdir() if p.is_file() and p.name not in ('checksums.sha256','seal.json')}
    if actual!=expected or any(not p.is_file() or p.is_symlink() for p in path.iterdir()):raise ValueError('artifact payload membership/SHA')
    subprocess.run(['sha256sum','--check','checksums.sha256'],cwd=path,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    m=json.loads((path/'manifest.json').read_text());result=json.loads((path/'metrics.json').read_text())
    if m.get('identity',{}).get('candidate_id') != CANDIDATE_ID:
        raise ValueError('ModernTCN v8 artifact candidate identity required')
    validate_revision(m['identity']['structural_spec'])
    if m['identity']['task'].get('structural_revision') != REVISION:
        raise ValueError('ModernTCN v8 artifact task revision required')
    if m['status']!='completed' or m['identity_sha']!=seal['identity_sha'] or result['test_calls']!=1 or result['identity_sha']!=seal['identity_sha']:raise ValueError('completed/test identity')
    if not {'config.json','spec.json','environment.json','command.json','worker.log','history.jsonl','updates.jsonl','best.pt','last.pt','test-claim.json','test-result.json','metrics.json','manifest.json'}<=set(expected):raise ValueError('incomplete artifact')
    history=[json.loads(l) for l in (path/'history.jsonl').read_text().splitlines()]
    updates=[json.loads(l) for l in (path/'updates.jsonl').read_text().splitlines()]
    if len(history)!=m['epochs'] or len(updates)!=m['steps'] or m['steps']>m['max_steps'] or result['steps']!=m['steps'] or result['epochs']!=m['epochs']:raise ValueError('completion progress accounting')
    if history[-1]['best_epoch']!=m['best']['epoch'] or result['best_epoch']!=m['best']['epoch']:raise ValueError('completion best identity')
    for name in ('best.pt','last.pt'):
        if json.loads((path/(name+'.sha256')).read_text())!={'sha256':file_sha(path/name),'schema':SCHEMA}:raise ValueError('checkpoint sidecar checksum mismatch')
    if json.loads((path/'test-result.sha256').read_text())!={'sha256':file_sha(path/'test-result.json')}:raise ValueError('saved test sidecar checksum mismatch')
    if json.loads((path/'test-result.json').read_text())!=result:raise ValueError('metrics/saved-test mismatch')
    claim=json.loads((path/'test-claim.json').read_text())
    if claim!={'identity_sha':m['identity_sha'],'calls':1,'best_epoch':m['best']['epoch']}:raise ValueError('test claim identity/count')
    identity=m['identity'];config=json.loads((path/'config.json').read_text());data=json.loads((path/'data-metadata.json').read_text());test_data=json.loads((path/'test-data-metadata.json').read_text())
    if digest(config)!=identity['config_sha'] or digest(identity)!=m['identity_sha'] or json.loads((path/'identity.json').read_text())!=identity:raise ValueError('artifact configuration/run identity')
    if identity.get('schema')=='moderntcn-uv-run-identity-v2':
        from .gate_policy import validate_config_policy,validate_policy
        validate_config_policy(config);validate_policy(identity.get('gate_policy'))
        if 'gate_main_metric' in identity or identity['gate_policy']!=config['gate_policy']:raise ValueError('artifact gate policy identity')
    if digest(data)!=identity['data_sha'] or digest(test_data)!=result['test_data_sha'] or data['scaler']!=test_data['scaler'] or data['test_observations_accessed'] is not False or test_data['test_observations_accessed'] is not True:raise ValueError('data identity/scaler/test roles')
    if json.loads((path/'spec.json').read_text())!=identity['structural_spec'] or json.loads((path/'environment.json').read_text())!=identity['environment']:raise ValueError('spec/environment identity')
    tr=identity['profile']['training'];best=BestState(tr['patience']);per_epoch=tr['scheduler']['steps_per_epoch']
    for n,row in enumerate(history,1):
        if best.stopped or row['epoch']!=n or row['steps']!=n*per_epoch:raise ValueError('epoch/stop history accounting')
        improved=best.update(row['validation']['mse'],n)
        if row['best_epoch']!=best.epoch or row['bad']!=best.bad or row['improved']!=improved:raise ValueError('strict validation best history')
    if vars(best)!=m['best'] or (len(history)!=tr['epochs'] and not best.stopped):raise ValueError('incomplete cap/early-stop progress')
    return dict(path=str(path),identity_sha=seal['identity_sha'],metrics=result,manifest=m,seal_sha256=file_sha(path/'seal.json'))

def publish(staging,final,manifest,metrics):
    staging,final=Path(staging),Path(final)
    if final.exists():raise FileExistsError('completed identity cannot be overwritten')
    atomic_json(staging/'manifest.json',manifest);atomic_json(staging/'metrics.json',metrics)
    files=sorted(p.name for p in staging.iterdir() if p.is_file() and p.name not in ('checksums.sha256','seal.json'))
    if any(p.is_symlink() or not p.is_file() or p.name.endswith('.tmp') for p in staging.iterdir()):raise ValueError('flat complete regular artifact files required')
    python={name:file_sha(staging/name) for name in files}
    output=subprocess.check_output(['sha256sum','--',*files],cwd=staging,text=True)
    system=dict((name,h) for h,name in (line.split('  ',1) for line in output.splitlines()))
    if system!=python:raise ValueError('Python/system checksum disagreement')
    (staging/'checksums.sha256').write_text(''.join(python[n]+'  '+n+'\n' for n in files))
    atomic_json(staging/'seal.json',dict(schema='moderntcn-uv-artifact-v1',complete=True,identity_sha=manifest['identity_sha'],checksums_sha256=file_sha(staging/'checksums.sha256')))
    audit(staging)
    if final.exists():raise FileExistsError('publication destination exists')
    final.parent.mkdir(parents=True,exist_ok=True);os.rename(staging,final)
    fd=os.open(final.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return audit(final)
