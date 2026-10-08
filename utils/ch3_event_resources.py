"""M6 exclusive execution: one launch hardware grant, no runtime GPU queries."""
import contextlib
import hashlib
import hmac
import json
import math
import os
import subprocess
import time
from pathlib import Path
from utils.ch3_contract import digest
from utils.ch3_native_recovery_records import bound, exclusive, ref

MODE = 'exclusive_gpu_event_driven_v1'
_PRESTART = None
_LAST = None


def enabled():
    from utils import ch3_type1_chain as q
    return q.PROBE_RECOVERY is not None and q.PROBE_RECOVERY.resource_binding().get('resource_mode') == MODE


def frozen_hardware():
    from utils import ch3_type1_chain as q
    return bound(q.ENVIRONMENT_REF)['hardware']


def validate_hardware(value):
    expected = frozen_hardware()
    if value != expected or value['cpu_affinity'] != sorted(os.sched_getaffinity(0)) or value['threads'] != 4:
        raise ValueError('fixed GPU/environment/CPU affinity binding changed')
    return value


@contextlib.contextmanager
def prestart(authorization):
    """Only public prepare/start with valid authorization, before owned workers."""
    global _PRESTART, _LAST
    from utils import ch3_type1_chain as q
    q.validate_start(authorization)
    if q.s.RESULT.exists():raise PermissionError('no GPU query after execution namespace exists')
    started = time.monotonic()
    raw = subprocess.check_output(['nvidia-smi','-i','0',
        '--query-gpu=uuid,name,memory.total,driver_version,memory.used,memory.free,memory.reserved',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    finished = time.monotonic()
    rows = raw.strip().splitlines()
    if len(rows) != 1:raise ValueError('exact fixed GPU0 startup query')
    uuid,name,total,driver,used,free,reserved = [x.strip() for x in rows[0].split(',')]
    numbers = {k:float(v)*1024**2 for k,v in [('total',total),('used',used),('free',free),('driver_reserved',reserved)]}
    sample = dict(time=finished,uuid=uuid,device='cuda:0',resource_mode=MODE,query_timeout=10.,query_elapsed=finished-started,**numbers)
    from tools.restricted_regression.m5_formal_entry import resource_assessment
    assessment = resource_assessment(sample,[],resource_mode=MODE)
    hardware = validate_hardware(dict(gpu=', '.join([uuid,name,total,driver]),cpu_affinity=sorted(os.sched_getaffinity(0)),threads=4))
    if not assessment['admission']or sample['free']<=assessment['reserve']:raise ValueError('startup GPU accounting/headroom query rejected')
    value = dict(purpose='M6_event_startup_hardware_v1',hardware=hardware,sample=sample,assessment=assessment,
        resource_contract_ref=q.PROBE_RECOVERY.resource_binding()['resource_contract_ref'],
        closure_commit=authorization['closure_commit'],authorization_sha=digest(authorization))
    _PRESTART = value
    try:yield value
    finally:_LAST=value;_PRESTART=None


def seal_startup(start_ref):
    from utils import ch3_type1_chain as q
    a=bound(start_ref);secret=os.environ.get(q.SECRET,'')
    if not secret or _LAST is None or _LAST['authorization_sha']!=digest(a) or _LAST['closure_commit']!=q.closure():
        raise PermissionError('startup grant requires this validated launch boundary')
    body=dict(_LAST,authorization_ref=start_ref,owner=bound(ref(q.CONTROL/'controller.json'))['owner'])
    body['mac']=hmac.new(secret.encode(),digest(body).encode(),hashlib.sha256).hexdigest()
    return exclusive(q.CONTROL/'startup-hardware.json',body)


def startup_ref():
    from utils import ch3_type1_chain as q
    return ref(q.CONTROL/'startup-hardware.json')


def validate_startup(value,producer=None,live=False):
    from utils import ch3_type1_chain as q
    body=bound(value);contract=q.PROBE_RECOVERY.resource_binding()
    if (body.get('purpose')!='M6_event_startup_hardware_v1' or body.get('resource_contract_ref')!=contract['resource_contract_ref']
        or body.get('authorization_sha')!=digest(bound(body['authorization_ref']))
        or body.get('closure_commit')!=bound(body['authorization_ref'])['closure_commit']):raise PermissionError('exact startup hardware/authorization binding')
    validate_hardware(body['hardware'])
    from tools.restricted_regression.m5_formal_entry import resource_assessment
    if (body['sample'].get('resource_mode')!=MODE or body['sample']['uuid']!=body['hardware']['gpu'].split(',')[0].strip()
        or body['assessment']!=resource_assessment(body['sample'],[],resource_mode=MODE) or not body['assessment']['admission']):raise ValueError('startup hardware evidence safety')
    if producer is not None and (producer.get('startup_hardware_ref')!=value or producer.get('hardware')!=body['hardware']
        or producer.get('commit')!=body['closure_commit'] or producer.get('start_authorization_ref')!=body['authorization_ref']):raise PermissionError('producer startup grant mismatch')
    if live:
        secret=os.environ.get(q.SECRET,'');payload={k:v for k,v in body.items()if k!='mac'}
        controller=bound(ref(q.CONTROL/'controller.json'))
        if (value!=startup_ref() or body['owner']!=controller['owner'] or body['authorization_ref']!=controller['authorization']
            or not q.same(body['owner']) or not secret or not hmac.compare_digest(body.get('mac',''),hmac.new(secret.encode(),digest(payload).encode(),hashlib.sha256).hexdigest())):raise PermissionError('startup hardware from current owned lifecycle only')
    return body


def hardware_binding():
    if _PRESTART is not None:return validate_hardware(_PRESTART['hardware'])
    return validate_startup(startup_ref(),live=True)['hardware']


def allocator_inputs(config):
    # Bootstrap is called only after the real installed guard validated config.
    from restricted_io_guard import require_installed
    if require_installed()is not config or config.get('resource_mode')!=MODE:raise PermissionError('verified event worker required')
    producer=config.get('approval')if config['purpose']=='ch3_probe'else bound(config['formal_permit_ref'])
    if config.get('resource_contract_ref')!=producer.get('resource_contract_ref') or config.get('startup_hardware_ref')!=producer.get('startup_hardware_ref'):raise PermissionError('worker allocator startup grant mismatch')
    sample=validate_startup(config['startup_hardware_ref'],producer,live=True)['sample']
    return sample['free'],sample['total']


def validate_receipt(value,producer,ids):
    from utils.ch3_m_execution import wave_passed
    if (producer.get('resource_mode')!=MODE or any(value.get(k)!=producer.get(k) for k in ('resource_mode','resource_contract_ref','startup_hardware_ref'))
        or not wave_passed(value) or value.get('owned_workers_exited')is not True or value.get('failure_kind')is not None
        or value.get('startup_hardware_ref') is None or value.get('original_failures') or value.get('cleanup_terminations')
        or value.get('runtime_gpu_queries')!=0 or value.get('telemetry')!='not_collected_startup_only'
        or not math.isfinite(value.get('elapsed',float('nan'))) or value['elapsed']<0):raise ValueError('event resource/task lifecycle gate')
    validate_startup(producer['startup_hardware_ref'],producer)
    if any(value.get(k)is not None for k in ('process_peaks','cpu_peaks','process_attribution','external_occupancy_known','whole_card_peak','actual_interval_min','actual_interval_max','fresh_post_exit_sample','sample_count')):raise ValueError('event mode must not invent telemetry')
    workers=value.get('worker_lifecycles',[])
    if (not ids or [w.get('task_id')for w in workers]!=ids or value['returncodes']!=[0]*len(ids)
        or len({w.get('pid')for w in workers})!=len(ids)
        or any(set(w)!={'pid','start_ticks','task_id','returncode'} or type(w['pid'])is not int or not str(w['start_ticks']).isdigit() or w['returncode']!=0 for w in workers)):
        raise ValueError('exact owned lifecycle/task/exit receipt')
    return {str(w['pid']):dict(pid=w['pid'],start_ticks=str(w['start_ticks']),task_id=w['task_id'])for w in workers}
