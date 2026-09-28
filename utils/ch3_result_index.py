"""Join only SHA-anchored accepted audit JSON, never original weights or live results."""
import copy,hashlib,json,math
from pathlib import Path
from utils.ch3_contract import digest,profile,read_profiles
BASE=Path('/public/home/yueweiting/大论文/amd-execution-evidence/m6')
ANCHORS={
 'AMD':('m6-amd-integrity-4w2coy1u','dbea91f446b596b94edceffbabff40059f28fbe01f82aa77fd0d0862173d7e48'),
 'PatchTST':('m6-patchtst-integrity-5dyeg0u5','55a1c3bfcd309308667cdb670872387e235346d76042525f9cfce04c014f40e4'),
 'J':('m6-j-completed-review-iuxcb4m0','48110f34ed3a32738438dded8f89a171af97792ff83120602ad796e9eb62f410')}
PACKAGE=BASE/'m6-epf4-timemixer-y5k7elwc'
CURRENT=PACKAGE/'boundary-index-v1'
DEFAULT_RECEIPTS=CURRENT/'accepted-receipts.json'
PREVIOUS_RECEIPTS=PACKAGE/'meter-closeout-v1/accepted-receipts.json'
PREVIOUS_SHA='a46e8a8df24bbecb70e316434a1f5dd17b85f96eb5af4f6af9c9e1d9d326bce2'
FROZEN_CONFIG_SHA='314ceca949dc9999fbbb9a6903fa97b6541f6e984dc25db5e6bfe7af3b74e038'
FORMAL_APPROVAL_SHA='4eed4c9fd2eb5dbd21201054cd3ad132daa383517a2d9826404f815cb1e79529'
class VerifiedReceipts(list):
    """Constructed after reconstruction against the fixed accepted evidence anchors."""

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def accepted_files(model,names):
    directory,expected=ANCHORS[model];root=BASE/directory;index=root/'evidence-index.json'
    if sha(index)!=expected:raise ValueError('accepted audit index changed')
    table=json.loads(index.read_text());out={};refs=[dict(path=str(index),sha256=expected)]
    for name in names:
        if name not in table:raise ValueError('file absent from accepted evidence index')
        entry=table[name];want=entry['sha256']if isinstance(entry,dict)else entry;p=root/name
        if sha(p)!=want:raise ValueError('accepted source SHA mismatch: '+str(p))
        out[name]=json.loads(p.read_text());refs.append(dict(path=str(p),sha256=want))
    return out,refs

def validate_manifest_identity(raw,expected_sha,task,expected_profile,approval,parent_protocol):
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('manifest accepted SHA mismatch')
    doc=json.loads(raw);identity=doc['identity']
    fields=('run_id','profile_sha','data_sha','commit','protocol_sha')
    if any(not isinstance(identity.get(k),str) or not identity[k] for k in fields):raise ValueError('missing actual manifest identity field')
    if identity['purpose']!='ch3_formal' or identity['run_id']!=task['id'] or doc['task']!=task:raise ValueError('actual task/run conflict')
    if doc['profile']!=expected_profile or identity['profile_sha']!=digest(doc['profile']):raise ValueError('actual/expected profile conflict')
    if identity['input_variant']!=task['input_variant']:raise ValueError('actual input variant conflict')
    if identity['data_sha']!=digest(doc['metadata']) or identity['data_sha']!=approval['data_bindings'][task['dataset']][task['id']]:raise ValueError('actual metadata/approved data conflict')
    if identity['commit']!=approval['commit'] or identity['protocol_sha']!=parent_protocol or identity['protocol_sha']!=approval['protocol_sha']:raise ValueError('actual commit/protocol conflict')
    if identity['source']!=approval['code']:raise ValueError('actual code binding conflict')
    return {k:identity[k]for k in fields}

def validate_identity_collection(records,expected_ids):
    ids=[r['run_id']for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):raise ValueError('duplicate/missing/foreign manifest identity')

def build_receipts(c):
    if sha(PREVIOUS_RECEIPTS)!=PREVIOUS_SHA:raise ValueError('prior accepted receipt changed')
    previous=json.loads(PREVIOUS_RECEIPTS.read_text());records=copy.deepcopy(previous['records'])
    if len(records)!=267 or len({r['id']for r in records})!=267:raise ValueError('prior 267 accepted records')
    j,refs=accepted_files('J',['source-sha256.json']);fingerprints=j['source-sha256.json']
    frozen_path=CURRENT/'frozen-config.json';approval_path=CURRENT/'original-formal-approval.json'
    if sha(frozen_path)!=FROZEN_CONFIG_SHA or sha(approval_path)!=FORMAL_APPROVAL_SHA:raise ValueError('original frozen config/approval changed')
    frozen=read_profiles(frozen_path);approval=json.loads(approval_path.read_text())
    if digest(frozen)!=c['extension']['parent_protocol_sha']:raise ValueError('frozen parent protocol mismatch')
    tasks={t['id']:t for t in frozen['tasks']if t['model']=='J'}
    exact={str(Path(c['execution']['evidence'])/'formal-J'/r/'manifest.json') for r in tasks}
    accepted={p for p in fingerprints if '/formal-J/'in p and p.endswith('/manifest.json')}
    if len(tasks)!=113 or exact!=accepted:raise ValueError('113 exact accepted J manifest scope')
    extracted=json.loads((CURRENT/'j-identities.json').read_text());validate_identity_collection(extracted['records'],tasks)
    entries={r['run_id']:r for r in extracted['records']}
    for row in records:
        if row['id']not in tasks:continue # retain the already-bound 154 verbatim
        run=row['id'];task=tasks[run];mp=str(Path(c['execution']['evidence'])/'formal-J'/run/'manifest.json');h=fingerprints[mp]
        snapshot=CURRENT/'manifests'/(run+'.json');raw=snapshot.read_bytes()
        actual=validate_manifest_identity(raw,h,task,profile(frozen,task),approval,c['extension']['parent_protocol_sha'])
        entry=entries[run]
        if entry['identity']!=actual or entry['manifest_path']!=mp or entry['manifest_sha256']!=h or entry['snapshot_path']!=str(snapshot):raise ValueError('identity excerpt/source mismatch')
        if row['manifest_sha256']!=h or row['expected_profile_sha']!=actual['profile_sha']or row['expected_protocol_sha']!=actual['protocol_sha']:raise ValueError('accepted result and manifest identity disagree')
        row.update({k:v for k,v in actual.items()if k!='run_id'})
        row.update(status='success',missing_actual_bindings=[],identity_basis='actual SHA-verified manifest identity; source/config/metadata crosscheck; no checkpoint/result recomputation',identity_verification='verified from accepted manifest SHA',identity_extract_review='pending ChatGPT review of new excerpt',identity_source=dict(path=mp,sha256=h,snapshot_path=str(snapshot)),identity_evidence=[dict(path=str(frozen_path),sha256=FROZEN_CONFIG_SHA),dict(path=str(approval_path),sha256=FORMAL_APPROVAL_SHA)]+refs)
    return dict(format='accepted-audit-receipts-v1',parent_protocol_sha=c['extension']['parent_protocol_sha'],records=records,source_files=previous['source_files']+[dict(path=str(PREVIOUS_RECEIPTS),sha256=PREVIOUS_SHA)]+refs,original_artifact_reads=0,manifest_snapshot_reads=113,TimeMixer_result_reads=0,missing={},identity_extract_review='pending ChatGPT; existing result acceptance unchanged')

def validate_receipt_document(c,document,expected):
    if document!=expected:raise ValueError('receipt differs from reconstructed accepted sources; handwritten review not accepted')
    return VerifiedReceipts(document['records'])

def load_receipts(c,path=DEFAULT_RECEIPTS):
    document=json.loads(Path(path).read_text());return validate_receipt_document(c,document,build_receipts(c))
