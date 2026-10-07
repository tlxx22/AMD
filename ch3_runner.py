"""Shared formal runner, fixed-wave scheduler and six-step resource worker.

No business import on module import. Formal start requires external approval,
clean reviewed closure, data bindings and a reviewed probe decision.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from utils.ch3_contract import (ROOT, PROFILE_FILE, digest, read_profiles, validate_manifest,
                               profile, task_by_id, BestState, summarize)


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def code_binding():
    files=['ch3_runner.py','utils/ch3_m6.py','tests/test_ch3_m6.py','utils/ch3_contract.py','utils/ch3_data.py','models/ch3_adapter.py',
           'models/tsAMD.py','models/tsAMD_enhanced.py','models/common.py','models/tsmoe.py',
           'utils/dataloader_urbanev.py','utils/feature_schema.py']
    files += ['tests/test_ch3_formal.py','scripts/ch3/start_model.sh','scripts/ch3/start_probe.sh',
              'tools/restricted_regression/m5_formal_entry.py','tools/restricted_regression/restricted_io_guard.py',
              'tools/restricted_regression/sitecustomize.py','tools/restricted_regression/bundle.sha256']
    if (ROOT/'utils/ch3_extension.py').exists():
        files += ['utils/ch3_extension.py','m6_extension_entry.py','tests/test_m6_extension.py','scripts/ch3/start_supplement.sh',
                  'utils/ch3_extension_probe.py','m6_probe_entry.py','scripts/ch3/start_extension_probe.sh',
                  'utils/ch3_result_index.py','tests/test_m6_closeout.py','tests/test_m6_boundaries.py',
                  'utils/ch3_revision.py','m6_revision_entry.py','scripts/ch3/start_timemixer_revision.sh','tests/test_m6_revision.py',
                  'utils/ch3_revision_probe.py','m6_revision_probe_entry.py','scripts/ch3/start_revision_probe.sh','tests/test_m6_revision_probe.py']
    files += ['utils/ch3_urban_diagnostic.py','utils/ch3_urban_capture.py','m6_urban_diagnostic_entry.py',
              'scripts/ch3/start_urban_diagnostic.sh','tests/test_m6_urban_diagnostic.py']
    files += ['utils/ch3_urban_confirmation.py','utils/ch3_admission_merge.py','m6_urban_confirmation_entry.py',
              'scripts/ch3/start_urban_confirmation.sh','tests/test_m6_urban_confirmation.py']
    files += ['utils/ch3_remaining.py','m6_remaining_entry.py','scripts/ch3/start_remaining_models.sh','tests/test_m6_remaining.py']
    if (ROOT/'utils/ch3_m_tasks.py').exists():
        files += ['tools/restricted_regression/m6_m_cpu_entry.py','utils/ch3_m_tasks.py','utils/ch3_m_execution.py','utils/ch3_m_summary.py','m6_m_entry.py','scripts/ch3/start_m_baselines.sh','scripts/ch3/start_m_probe.sh','tests/test_m6_m_tasks.py']
        files += ['utils/ch3_m_launch.py','utils/ch3_m_handoff.py','m6_m_handoff_entry.py','scripts/ch3/start_m_handoff.sh','tests/test_m6_m_handoff.py']
        files += ['utils/ch3_m_auto.py','tests/test_m6_m_auto.py']
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'models/modules').glob('*.py'))]
    if (ROOT/'utils/ch3_native_chain.py').exists():
        files += ['utils/ch3_time_marks.py','utils/ch3_native_tasks.py','utils/ch3_native_execution.py',
                  'utils/ch3_native_chain.py','m6_native_chain_entry.py','scripts/ch3/start_native_time_mark_chain.sh',
                  'configs/ch3_native_time_mark_profiles.json','tests/test_m6_native_time_marks.py','tests/test_m6_native_chain.py',
                  'utils/ch3_ms_retirement.py','tests/test_m6_native_retirement.py','tests/test_m6_native_v3.py','tests/test_m6_native_v4.py']
    if (ROOT/'utils/ch3_native_recovery.py').exists():
        files += ['utils/ch3_native_recovery.py','utils/ch3_native_recovery_records.py','utils/ch3_native_recovery_execution.py',
                  'm6_native_recovery_entry.py','scripts/ch3/start_native_time_mark_recovery.sh','tests/test_m6_native_recovery.py']
    if (ROOT/'utils/ch3_baseline_unified_tasks.py').exists():
        files += ['utils/ch3_onecycle.py','utils/ch3_baseline_unified_tasks.py','utils/ch3_baseline_unified_execution.py',
                  'utils/ch3_baseline_unified_chain.py','m6_baseline_unified_entry.py','scripts/ch3/start_baseline_unified.sh',
                  'configs/ch3_baseline_ms_u96_oc01.json','configs/ch3_baseline_m_u96_oc01.json',
                  'tests/test_ch3_onecycle.py','tests/test_m6_baseline_unified.py']
        files += ['configs/ch3_baseline_ms_u96_oc01_v2.json','configs/ch3_baseline_m_u96_oc01_v2.json',
                  'tests/test_m6_baseline_unified_v2.py']
    if (ROOT/'utils/ch3_baseline_unified_tasks.py').exists():
        files += ['configs/ch3_baseline_ms_u96_oc01_v3.json','configs/ch3_baseline_m_u96_oc01_v3.json',
                  'tests/test_m6_baseline_unified_v3.py']
    if (ROOT/'utils/ch3_type1_tasks.py').exists():
        files += ['utils/ch3_type1.py','utils/ch3_type1_tasks.py','utils/ch3_type1_chain.py','utils/ch3_type1_upstream.py',
                  'utils/ch3_type1_execution.py','utils/ch3_type1_summary.py','m6_type1_followup_entry.py','scripts/ch3/start_type1_followup.sh',
                  'configs/ch3_type1_urban_subset.json','configs/ch3_type1_epf_all.json','tests/test_m6_type1_followup.py']
    if (ROOT/'utils/ch3_type1_scaled.py').exists():
        files += ['utils/ch3_type1_scaled.py','configs/ch3_type1_urban_subset_v2.json','configs/ch3_type1_epf_all_v2.json',
                  'configs/ch3_type1_m_all_v2.json','tests/test_m6_type1_followup_v2.py','tests/ch3_historical_type1_v1.py']
    if (ROOT/'utils/ch3_type1_ett.py').exists():
        files += ['utils/ch3_type1_ett.py','tests/test_m6_type1_ett.py','tests/ch3_historical_type1_v2_147.py']
    if (ROOT/'utils/ch3_round2_amendment.py').exists():
        files += ['utils/ch3_round2_amendment.py','configs/ch3_round2_m_amend1.json',
                  'configs/ch3_type1_urban_subset_v3.json','configs/ch3_type1_epf_all_v3.json',
                  'configs/ch3_type1_m_all_v3.json','tests/test_m6_round2_amendment.py',
                  'tests/ch3_historical_type1_v2_231.py']
    if (ROOT/'utils/ch3_ms_seal_recovery.py').exists():
        files += ['utils/ch3_ms_seal_recovery.py','tests/test_m6_ms_seal_recovery.py','configs/ch3_round2_m_batch128_recovery1.json',
                  'configs/ch3_round2_m_amend1_recovery1.json','configs/ch3_type1_urban_subset_v3_recovery1.json',
                  'configs/ch3_type1_epf_all_v3_recovery1.json','configs/ch3_type1_m_all_v3_recovery1.json']
    if (ROOT/'utils/ch3_probe_schema_recovery.py').exists():
        files += ['utils/ch3_probe_schema_recovery.py','m6_probe_schema_recovery_entry.py',
                  'scripts/ch3/start_probe_schema_recovery.sh','tests/test_m6_probe_schema_recovery.py']
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}


def environment_binding():
    from importlib.metadata import version
    return dict(python=sys.version, executable=sys.executable,
                packages={k:version(k) for k in ('torch','numpy','pandas','scipy','scikit-learn','einops')})


def hardware_binding():
    return dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip(),
                cpu_affinity=sorted(os.sched_getaffinity(0)),threads=4)


def validate_probe_report(c, report):
    if report.get('purpose')=='m6_remaining_resource_carry_forward_v1':
        from utils.ch3_remaining import validate_resources
        return validate_resources(c,report)
    if report.get('purpose')=='m6_merged_resource_admission_v1':
        from utils.ch3_admission_merge import validate_merged
        return validate_merged(c,report)
    if report.get('protocol_sha')!=digest(c) or report.get('Q')!=sum(g['q'] for g in c['groups']):
        raise ValueError('probe configuration/worker coverage mismatch')
    if set(report.get('decisions',{}))!={g['id'] for g in c['groups']}:
        raise ValueError('probe group coverage mismatch')
    for group in c['groups']:
        decision=report['decisions'][group['id']]
        if decision.get('status')=='Passed':
            verified_waves(group,decision)
            from utils.ch3_contract import numeric_probe_policy
            rule=numeric_probe_policy(c,task_by_id(c,group['representatives'][0]))
            if rule and decision['concurrency']>1:
                checked=decision.get(str(decision['concurrency']),{}).get('numerical_comparisons',[])
                if len(checked)!=group['q']:raise ValueError('numeric admission coverage missing')
                import math
                for row in checked:
                    if (row.get('passed') is not True or row.get('mode')!='bounded_numeric'
                            or row.get('policy_sha')!=digest(rule) or row.get('rtol')!=0
                            or row.get('exact_residual_state') is not True):
                        raise ValueError('numeric admission policy mismatch')
                    if rule['kind']=='named_tensor':
                        maxima=list(row.get('tensor_max_abs',{}).values())
                        scalar=row.get('scalar_normalized_max_abs')
                        if (len(maxima)!=4 or any(v is None or not math.isfinite(v) or v<0 or v>rule['atol'] for v in maxima)
                                or scalar is None or not math.isfinite(scalar) or scalar<0 or scalar>rule['metric_atol']):
                            raise ValueError('named numeric admission exceeds bound')
                    elif rule['kind']=='full_float_state':
                        state=row.get('state_max_abs')
                        scalar=row.get('validation_max_abs') if rule.get('loss_rtol',0)>0 else row.get('scalar_normalized_max_abs')
                        ratio=row.get('loss_error_ratio') if rule.get('loss_rtol',0)>0 else 0.0
                        if (state is None or not math.isfinite(state) or state<0 or state>rule['state_atol']
                                or scalar is None or not math.isfinite(scalar) or scalar<0 or scalar>rule['metric_atol']
                                or ratio is None or not math.isfinite(ratio) or ratio<0 or ratio>1):
                            raise ValueError('full-state numeric admission exceeds bound')
                    else:raise ValueError('unknown numeric policy kind')
    return report

def validate_model_completion(c, model, report):
    expected=[t['id'] for t in c['tasks'] if t['model']==model]
    if (not expected or report.get('protocol_sha')!=digest(c) or report.get('model')!=model
            or report.get('task_ids')!=expected):raise ValueError('foreign/incomplete model completion')
    groups={g['id']:g for g in c['groups'] if g['model']==model}
    if set(report.get('results',{}))!=set(groups):raise ValueError('model summary coverage mismatch')
    for key,value in report['results'].items():
        if value.get('input_variant')!=groups[key]['input_variant']:raise ValueError('model input identity mismatch')
    return report


def kernel_admission_reasons(c):
    """Conditional relaxation requires reviewed same-operand backend evidence."""
    item=c['execution']['probe'].get('kernel_admission')
    if not item:return ['kernel attribution and paired numeric review missing']
    try:
        p=Path(item['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('kernel evidence SHA mismatch')
        d=json.loads(p.read_text());expected={'TimeMixer/Exchange','ModernTCN/Weather','ModernTCN/ECL','TimeMixer/Weather'}
        if set(d['scopes'])!=expected:raise ValueError('kernel scope coverage mismatch')
        for v in d['scopes'].values():
            if (v.get('backend_demonstrated') is not True or v.get('paired_passed') is not True
                    or v.get('flags_restored') is not True or v.get('exact_identity') is not True):
                raise ValueError('kernel/paired conditions not met')
        return []
    except (OSError,KeyError,ValueError,TypeError) as exc:return ['kernel admission: '+str(exc)]


def preflight(c,model,approval=None,probe=False):
    validate_manifest(c)
    if 'm_experiment' in c:
        from utils.ch3_m_execution import authorization_reasons
        return authorization_reasons(c,approval,probe=probe)
    if model is not None and model not in {t['model'] for t in c['tasks']}:raise ValueError('model group not registered')
    reasons=[]
    if approval and approval.get('queue_id')=='m6-remaining-models-v1':
        from utils.ch3_remaining import child_reasons
        reasons.extend(child_reasons(c,model,approval))
    from utils.ch3_contract import training_blockers
    for task in c['tasks']:
        if model is None or task['model']==model:
            reasons.extend(training_blockers(c,task))
    reasons=list(dict.fromkeys(reasons))
    from utils.ch3_data import verify_source_state
    for name,d in c['datasets'].items():
        try:verify_source_state(d)
        except (OSError,ValueError) as exc:reasons.append(name+': '+str(exc))
        if probe:
            reasons.extend(name+': '+reason for reason in d['mandatory_blockers'])
            if name!='UrbanEV' and d['endpoints'] is None:reasons.append(name+': probe tail endpoints missing')
    if probe:
        reasons.extend(kernel_admission_reasons(c))
        reasons.extend(c['execution']['probe'].get('mandatory_blockers',[]))
        if 'extension' in c:
            from utils.ch3_extension_probe import scope_module
            reasons.extend(scope_module(approval).authorization_reasons(c,approval))
        else:
            from utils.ch3_contract import followup_limits
            try:followup_limits(c,approval)
            except (ValueError,KeyError,TypeError) as exc:reasons.append('follow-up: '+str(exc))
    if not probe and c.get('timemixer_revision') and model=='TimeMixer' and not (approval and approval.get('extension_batch')==c.get('extension',{}).get('id')):
        from utils.ch3_revision import authorization_reasons as revision_authorization_reasons
        reasons+=revision_authorization_reasons(c,approval)
    if git('status','--porcelain','--untracked-files=all'):reasons.append('reviewed clean closure required')
    if approval is None:reasons.append('explicit review/closure authorization missing')
    else:
        if approval.get('commit')!=git('rev-parse','HEAD'):reasons.append('approval commit mismatch')
        if approval.get('protocol_sha')!=digest(c):reasons.append('approval configuration mismatch')
        if approval.get('code')!=code_binding():reasons.append('approval source mismatch')
        if approval.get('environment')!=environment_binding():reasons.append('approval environment mismatch')
        if approval.get('hardware')!=hardware_binding():reasons.append('approval hardware/CPU allocation mismatch')
        if probe:
            if approval.get('purpose')!='ch3_resource_probe' or approval.get('reviewed') is not True:reasons.append('probe review not approved')
        else:
            if approval.get('purpose')!='ch3_formal' or approval.get('structure_frozen') is not True or approval.get('m6_authorized') is not True:
                reasons.append('structure freeze and M6 authorization required')
            from utils.ch3_m6 import scoped_model_tasks
            for domain in {t['dataset'] for t in scoped_model_tasks(c,model,approval)}:
                unresolved=c['datasets'][domain]['mandatory_blockers']
                if unresolved:reasons.append(domain+': '+','.join(unresolved))
                if domain not in approval.get('data_bindings',{}):reasons.append(domain+': reviewed data prefix binding missing')
            if not approval.get('probe_report_sha'):reasons.append('reviewed resource decisions missing')
            from utils.ch3_m6 import approval_reasons
            reasons.extend(approval_reasons(c,model,approval))
    for src in c['sources'].values():
        if subprocess.check_output(['git','-C',src['repository'],'rev-parse','HEAD'],text=True).strip()!=src['commit']:
            reasons.append('author commit changed: '+src['repository'])
        for p,h in src['files'].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:reasons.append('author bytes changed: '+p)
    return reasons


class GPULock:
    def __init__(self,c):self.path=Path(c['baseline_unified']['project_lock']) if 'baseline_unified'in c else Path(c['m_experiment']['project_lock']) if 'm_experiment'in c else Path(c['execution']['evidence'])/'project-gpu0.lock';self.handle=None
    def __enter__(self):
        self.handle=self.path.open('a+')
        try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.handle.close();raise RuntimeError('another project model/probe owns GPU0')
        return self
    def __exit__(self,*args):fcntl.flock(self.handle,fcntl.LOCK_UN);self.handle.close()


def tensor_digest(value):
    import torch
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode())
            if a.numel():h.update(memoryview(a.numpy()).cast('B'))
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()


class ReusableTensorDigest:
    """Hash state through fixed CPU storage; never change values or RNG."""
    def __init__(self, template):
        import torch
        schemas={}
        def register(v):
            if torch.is_tensor(v):
                if v.layout!=torch.strided:raise ValueError('digest requires dense tensors')
                schemas[(v.dtype,tuple(v.shape))]=v.numel()
            elif isinstance(v,dict):
                for x in v.values():register(x)
            elif isinstance(v,(tuple,list)):
                for x in v:register(x)
        register(template);schemas[(torch.float32,())]=1
        sizes={}
        for (dtype,shape),n in schemas.items():sizes[dtype]=max(sizes.get(dtype,0),n)
        self.buffers={dtype:torch.empty(n,dtype=dtype,device='cpu') for dtype,n in sizes.items()}
        for b in self.buffers.values():b.zero_()
        self.views={};self.bytes={};self.headers={}
        for (dtype,shape),n in schemas.items():
            key=(dtype,shape);v=self.buffers[dtype][:n].view(shape)
            self.views[key]=v;self.bytes[key]=memoryview(v.numpy()).cast('B') if n else None
            self.headers[key]=str((dtype,shape)).encode()
        self.storage_bytes=sum(b.numel()*b.element_size() for b in self.buffers.values())
        self.buffer_allocations=len(self.buffers)
    def copy_bytes(self,v):
        import torch
        if not torch.is_tensor(v):raise TypeError('tensor required')
        key=(v.dtype,tuple(v.shape))
        if key not in self.views or v.layout!=torch.strided:raise ValueError('unregistered state tensor schema')
        if v.numel():self.views[key].copy_(v.detach(),non_blocking=False)
        return self.bytes[key],self.headers[key]
    def __call__(self, value):
        import torch
        h=hashlib.sha256()
        def add(v):
            if torch.is_tensor(v):
                raw,header=self.copy_bytes(v);h.update(header)
                if raw is not None:h.update(raw)
            elif isinstance(v,dict):
                for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
            elif isinstance(v,(list,tuple)):
                for x in v:add(x)
            else:h.update(repr(v).encode())
        add(value);return h.hexdigest()


def _numeric_json_safe(v):
    if v is None or isinstance(v,(str,int,float,bool)):return v
    if isinstance(v,(list,tuple)):return [_numeric_json_safe(x) for x in v]
    if isinstance(v,dict):return {str(k):_numeric_json_safe(x) for k,x in sorted(v.items(),key=lambda z:str(z[0]))}
    return repr(v)


class FullNumericStateWriter:
    """Write full floating state through the same preallocated CPU buffers.

    Floating model parameters/buffers, gradients and Adam moments are bounded;
    optimizer step, nonfloating tensors and optimizer structure stay exact.
    """
    def __init__(self,model,opt,out,digest_state,rule):
        self.model,self.opt,self.out,self.digest,self.rule=model,opt,Path(out),digest_state,rule
        self.params=list(model.named_parameters());self.buffers=list(model.named_buffers())
        self.names={id(v):k for k,v in self.params};self.entries=None;self.schema=None;self.schema_path=self.out/'numeric-full-schema.json';self.schema_sha=None
    def _group_meta(self):
        rows=[]
        for group in self.opt.param_groups:
            row={}
            for k,v in group.items():row[k]=[self.names[id(x)] for x in v] if k=='params' else _numeric_json_safe(v)
            rows.append(row)
        if getattr(self.opt,'_ch3_onecycle',None):
            from utils.ch3_onecycle import optimizer_groups_static
            rows=optimizer_groups_static(rows)
        return rows
    def _build(self):
        import torch
        specs=[];access=[];offset=0;non_tensor={}
        def add(path,tensor,mode,getter):
            nonlocal offset
            if not torch.is_tensor(tensor) or tensor.layout!=torch.strided:raise ValueError('full numeric state requires dense tensor')
            n=tensor.numel()*tensor.element_size();specs.append(dict(path=path,dtype=str(tensor.dtype),shape=list(tensor.shape),mode=mode,offset=offset,nbytes=n,numel=tensor.numel()))
            access.append(getter);offset+=n
        for name,t in self.params:add('model/parameter/'+name,t,'bounded' if t.is_floating_point() else 'exact',lambda t=t:t)
        for name,t in self.buffers:add('model/buffer/'+name,t,'bounded' if t.is_floating_point() else 'exact',lambda t=t:t)
        for name,p in self.params:
            if p.grad is not None:add('gradient/'+name,p.grad,'bounded' if p.grad.is_floating_point() else 'exact',lambda p=p:p.grad)
        for name,p in self.params:
            row={}
            for key,value in sorted(self.opt.state.get(p,{}).items(),key=lambda z:str(z[0])):
                if torch.is_tensor(value):
                    mode='exact' if key=='step' or not value.is_floating_point() else 'bounded'
                    add('optimizer/'+name+'/'+str(key),value,mode,lambda p=p,key=key:self.opt.state[p][key])
                else:row[str(key)]=_numeric_json_safe(value)
            if row:non_tensor[name]=row
        schema=dict(policy_id=self.rule['id'],entries=specs,optimizer_groups=self._group_meta(),optimizer_non_tensor_state=non_tensor,total_bytes=offset)
        self.entries,self.schema=access,schema
        dump(self.schema_path,schema);self.schema_sha=hashlib.sha256(self.schema_path.read_bytes()).hexdigest()
    def _verify_structure(self):
        import torch
        expected=self.schema['entries'];actual=[];non_tensor={};offset=0
        def add(path,tensor,mode):
            nonlocal offset
            if not torch.is_tensor(tensor):raise ValueError('full numeric tensor disappeared')
            n=tensor.numel()*tensor.element_size();actual.append(dict(path=path,dtype=str(tensor.dtype),shape=list(tensor.shape),mode=mode,offset=offset,nbytes=n,numel=tensor.numel()));offset+=n
        for name,t in self.params:add('model/parameter/'+name,t,'bounded' if t.is_floating_point() else 'exact')
        for name,t in self.buffers:add('model/buffer/'+name,t,'bounded' if t.is_floating_point() else 'exact')
        for name,p in self.params:
            if p.grad is not None:add('gradient/'+name,p.grad,'bounded' if p.grad.is_floating_point() else 'exact')
        for name,p in self.params:
            row={}
            for key,value in sorted(self.opt.state.get(p,{}).items(),key=lambda z:str(z[0])):
                if torch.is_tensor(value):add('optimizer/'+name+'/'+str(key),value,'exact' if key=='step' or not value.is_floating_point() else 'bounded')
                else:row[str(key)]=_numeric_json_safe(value)
            if row:non_tensor[name]=row
        if actual!=expected or self._group_meta()!=self.schema['optimizer_groups'] or non_tensor!=self.schema['optimizer_non_tensor_state']:
            raise ValueError('full numeric state structure changed')
    def capture(self,step):
        if self.entries is None:self._build()
        else:self._verify_structure()
        path=self.out/f'numeric-full-step-{step}.bin';h=hashlib.sha256()
        with path.open('xb') as f:
            for getter in self.entries:
                tensor=getter()
                if tensor is None:raise ValueError('full numeric state tensor missing')
                if tensor.is_floating_point():finite(tensor)
                raw,_=self.digest.copy_bytes(tensor)
                if raw is not None:f.write(raw);h.update(raw)
        if path.stat().st_size!=self.schema['total_bytes']:raise ValueError('full numeric sidecar size mismatch')
        return dict(step=step,schema_file=str(self.schema_path),schema_sha=self.schema_sha,data_file=str(path),data_sha=h.hexdigest(),bytes=path.stat().st_size)


def _compare_full_numeric_files(rule,reference,actual):
    import math,numpy as np
    if reference.get('schema_sha')!=actual.get('schema_sha'):return dict(passed=False,state_max_abs=float('inf'),exact=False,failures=['schema digest mismatch'],compared_elements=0)
    for row in (reference,actual):
        sp=Path(row['schema_file']);dp=Path(row['data_file'])
        if hashlib.sha256(sp.read_bytes()).hexdigest()!=row['schema_sha'] or hashlib.sha256(dp.read_bytes()).hexdigest()!=row['data_sha']:raise ValueError('numeric sidecar checksum mismatch')
        if dp.stat().st_size!=row['bytes']:raise ValueError('numeric sidecar length mismatch')
    sa=json.loads(Path(reference['schema_file']).read_text());sb=json.loads(Path(actual['schema_file']).read_text())
    if sa!=sb:return dict(passed=False,state_max_abs=float('inf'),exact=False,failures=['schema content mismatch'],compared_elements=0)
    dtype_map={'torch.float16':np.dtype('<f2'),'torch.float32':np.dtype('<f4'),'torch.float64':np.dtype('<f8')}
    maxima=0.0;count=0;failures=[];exact=True
    with Path(reference['data_file']).open('rb') as fa,Path(actual['data_file']).open('rb') as fb:
        for e in sa['entries']:
            off,n=e['offset'],e['nbytes'];fa.seek(off);fb.seek(off)
            if e['mode']=='exact':
                if hashlib.sha256(fa.read(n)).digest()!=hashlib.sha256(fb.read(n)).digest():exact=False;failures.append('exact '+e['path'])
                continue
            if e['mode']!='bounded' or e['dtype'] not in dtype_map:raise ValueError('unsupported bounded numeric dtype/mode')
            dt=dtype_map[e['dtype']];remaining=e['numel'];local=0.0
            while remaining:
                take=min(remaining,65536);aa=np.fromfile(fa,dtype=dt,count=take);bb=np.fromfile(fb,dtype=dt,count=take)
                if len(aa)!=take or len(bb)!=take:raise ValueError('truncated full numeric sidecar')
                if not np.isfinite(aa).all() or not np.isfinite(bb).all():raise ValueError('nonfinite full numeric state')
                d=np.abs(aa.astype(np.float64)-bb.astype(np.float64))
                if d.size:local=max(local,float(d.max()))
                remaining-=take
            maxima=max(maxima,local);count+=e['numel']
            if local>rule['state_atol']:failures.append(e['path']+' exceeds state tolerance')
    return dict(passed=not failures,state_max_abs=maxima,exact=exact,failures=failures,compared_elements=count)

def cpu_tree(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def save_state(path,model,opt,identity,best,epoch,steps,generator,scheduler=None):
    import torch,random,numpy as np
    value=dict(schema='ch3-state-v1',identity=identity,model=cpu_tree(model.state_dict()),
               optimizer=cpu_tree(opt.state_dict()),best=vars(best),epoch=epoch,steps=steps,
               rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                        generator=generator.get_state()))
    if scheduler is not None:
        if scheduler.updates!=steps:raise ValueError('checkpoint scheduler step accounting')
        value.update(schema=getattr(scheduler,'checkpoint_schema','ch3-state-v2-onecycle'),scheduler=scheduler.state_dict())
    tmp=Path(str(path)+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def restore_state(path,model,opt,identity,generator,scheduler=None):
    import torch,random,numpy as np
    state=torch.load(path,map_location='cpu')
    schema=getattr(scheduler,'checkpoint_schema','ch3-state-v2-onecycle') if scheduler is not None else 'ch3-state-v1'
    if state.get('schema')!=schema or state.get('identity')!=identity:raise ValueError('resume identity mismatch; no cross-protocol recovery')
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    if scheduler is not None:scheduler.load_state_dict(state['scheduler'],state['steps'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    generator.set_state(state['rng']['generator'])
    best=BestState();best.__dict__.update(state['best']);epoch,steps=state['epoch'],state['steps']
    del state
    return best,epoch,steps


def formal_identity(c, task, metadata, approval):
    p=profile(c,task)
    result=dict(purpose='ch3_formal',run_id=task['id'],input_variant=task['input_variant'],
                protocol_sha=digest(c),profile_sha=digest(p),data_sha=digest(metadata),
                source=code_binding(),commit=approval['commit'],**(__import__('utils.ch3_revision',fromlist=['identity_fields']).identity_fields(c,task,approval)))
    if 'baseline_unified' in c:
        result.update(scientific_protocol=c['baseline_unified']['id'],task=task['task'],metric_scope=task['metric_scope'],
                      scheduler_sha=digest(p['training']['scheduler']),parent_profile=c['baseline_unified']['parent_refs'][task['id']],from_scratch=True)
    if c.get('native_time_mark'):
        result.update(time_mark=p.get('time_mark'),time_mark_protocol=c['native_time_mark']['id'])
        if p.get('time_mark'):
            result.update(time_mark_mode=p['time_mark']['mode'],time_mark_freq=p['time_mark']['freq'],time_mark_features=p['time_mark']['features'],time_mark_source=p['time_mark']['source'])
        if task.get('parent_run_id'):
            result.update(parent_run_id=task['parent_run_id'],revision='native-time-mark-v1',parent_profile_sha=p['parent_profile_sha'],new_profile_sha=digest(p))
    if approval.get('recovery_scope'):
        result.update(**{k:approval[k] for k in ('science_baseline_commit','controller_execution_commit','worker_execution_commit','science_computation_fingerprint')},
                      recovery_scope=approval['recovery_scope'])
    return result


def audit_resume(out,approval,run,protocol=None):
    """Metadata and approved fingerprints before any torch deserialization."""
    out=Path(out);record=approval.get('resume_audits',{}).get(run)
    if record is None:raise PermissionError('explicit audit of retained run required')
    mode=record.get('mode')
    names={'complete':['result.json','manifest.json'],
           'resume':['manifest.json','last.pt','best.pt','history.jsonl'],
           'not_started':[]}.get(mode)
    if names is None:raise ValueError('unsupported recovery mode')
    if mode=='not_started':
        if out.exists():raise FileExistsError('cannot fresh-restart retained output')
        return mode
    # Reject foreign protocol/input identity from metadata before any checkpoint read.
    c=read_profiles() if protocol is None else protocol;task=task_by_id(c,run)
    if c.get('type1_followup') and mode=='resume' and record.get('test_access_status')!='not_accessed':raise PermissionError('unknown or previously accessed formal test; automatic recovery prohibited')
    manifest=json.loads((out/'manifest.json').read_text())
    identity=manifest['identity']
    if (identity.get('protocol_sha')!=digest(c) or identity.get('profile_sha')!=digest(profile(c,task))
            or identity.get('input_variant')!=task['input_variant']
            or manifest.get('task')!=task or manifest.get('profile')!=profile(c,task)):
        raise ValueError('foreign protocol/input identity before checkpoint access')
    for name in names:
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=record.get('sha256',{}).get(name):raise ValueError('recovery audit bytes mismatch: '+name)
    if manifest['identity']['run_id']!=run or manifest['identity']['purpose']!='ch3_formal':raise ValueError('foreign recovery identity')
    if (out/'result.json').exists() and mode!='complete':raise ValueError('completed run cannot resume')
    return mode


def finite(t):
    import torch
    if not bool(torch.isfinite(t).all()):raise FloatingPointError('nonfinite tensor')


def update(model,opt,x,y,p,device,x_mark_enc=None):
    import torch
    from models.ch3_adapter import target_prediction
    model.train();opt.zero_grad(set_to_none=True)
    if x_mark_enc is not None:finite(x_mark_enc)
    pred,aux=target_prediction(model,x.to(device),p,x_mark_enc.to(device) if x_mark_enc is not None else None);target=y.to(device)
    if tuple(pred.shape)!=tuple(target.shape):raise ValueError('supervision shape mismatch; broadcasting forbidden')
    finite(pred);loss=torch.nn.functional.mse_loss(pred,target)
    if aux is not None:finite(aux);loss=loss+aux
    finite(loss);loss.backward()
    active=[]
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite(parameter.grad)
            if bool(torch.count_nonzero(parameter.grad)):active.append(parameter)
    if not active:raise FloatingPointError('no target/loss gradient')
    opt.step()
    for parameter in model.parameters():finite(parameter)
    return float(loss.detach())


def evaluate(model,batches,p,device):
    if p.get('task')=='M':
        from utils.ch3_m_tasks import evaluate_all
        return evaluate_all(model,batches,p,device)
    import torch
    from models.ch3_adapter import target_prediction
    model.eval();sse=sae=0.;count=0
    with torch.no_grad():
        from utils.ch3_time_marks import batch_parts
        for batch in batches:
            x,y,mark=batch_parts(batch)
            pred,_=target_prediction(model,x.to(device),p,mark.to(device) if mark is not None else None);err=pred-y.to(device);finite(err)
            sse+=float(err.double().square().sum());sae+=float(err.double().abs().sum());count+=err.numel()
    if count==0:raise ValueError('empty evaluation')
    return dict(mse=sse/count,mae=sae/count,sse=sse,sae=sae,elements=count)


def init_training(c,task,device):
    import torch,random,numpy as np
    from models.ch3_adapter import build
    p=profile(c,task);torch.set_num_threads(4);random.seed(2024);np.random.seed(2024);torch.manual_seed(2024)
    model=build(c,task).to(device);t=p['training']
    opt=torch.optim.Adam(model.parameters(),lr=t['lr'],betas=tuple(t['betas']),eps=t['eps'],weight_decay=t['weight_decay'])
    return p,model,opt,torch.Generator().manual_seed(2024)


def numeric_probe_snapshot(model, opt, rule, step, digest_state=None):
    """Small CPU values for one named tensor; all residual state stays exact."""
    import torch
    digest_state=digest_state or tensor_digest
    if rule.get('kind')!='named_tensor':raise ValueError('named numeric snapshot requires named_tensor policy')
    name=rule['parameter'];params=dict(model.named_parameters())
    if name not in params:raise ValueError('approved numeric parameter missing')
    parameter=params[name];raw=opt.state_dict();ids=[]
    for objects,stored in zip(opt.param_groups,raw['param_groups']):
        if len(objects['params'])!=len(stored['params']):raise ValueError('optimizer mapping mismatch')
        ids += [sid for obj,sid in zip(objects['params'],stored['params']) if obj is parameter]
    if len(ids)!=1:raise ValueError('numeric parameter optimizer identity is not unique')
    sid=ids[0];state=raw['state'].get(sid,{})
    if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('unapproved optimizer state schema')
    def pack(v):
        if not torch.is_tensor(v) or str(v.dtype)!=rule['dtype'] or list(v.shape)!=rule['shape']:
            raise ValueError('numeric tensor dtype/shape changed')
        finite(v)
        return dict(dtype=str(v.dtype),shape=list(v.shape),values=v.detach().cpu().reshape(-1).tolist())
    values=dict(parameter=pack(parameter),gradient=pack(parameter.grad),
                exp_avg=pack(state['exp_avg']),exp_avg_sq=pack(state['exp_avg_sq']))
    residual=dict(raw);residual['state']=dict(raw['state']);residual['state'][sid]={k:v for k,v in state.items() if k not in ('exp_avg','exp_avg_sq')}
    return dict(step=step,parameter_name=name,optimizer_parameter_id=sid,
                tensors=values,
                exact_model=digest_state({k:v for k,v in model.state_dict().items() if k!=name}),
                exact_optimizer=digest_state(residual),
                exact_gradients=digest_state({k:v.grad for k,v in params.items() if k!=name and v.grad is not None}))


def _probe_validation_metrics(c, task, value, tail=None):
    """Validate the evaluation contract; project M's aggregate without dropping evidence."""
    import math
    if task not in c['tasks']:raise ValueError('foreign validation task')
    p=profile(c,task);is_m=p.get('task')=='M'  # Same routing as evaluate().
    if task.get('task','MS')!=p.get('task','MS'):raise ValueError('MS/M task/profile conflict')
    fields={'mse','mae','sse','sae','elements'}
    if not isinstance(value,dict) or set(value)!=(fields|{'channels','MS_target_diagnostic'} if is_m else fields):
        raise ValueError('validation schema changed')
    def totals(row,expected=None):
        n=row['elements']
        if type(n)is not int or n<=0 or (expected is not None and n!=expected):raise ValueError('validation element count mismatch')
        for k in ('mse','mae','sse','sae'):
            if isinstance(row[k],bool) or not isinstance(row[k],(int,float)) or not math.isfinite(row[k]) or row[k]<0:
                raise ValueError('nonfinite or nonnumeric validation metric')
        if row['mse']!=row['sse']/n or row['mae']!=row['sae']/n:raise ValueError('validation aggregation inconsistent')
    totals(value)
    if is_m:
        C=p['C'];names=p['features']
        if (type(C)is not int or C<=0 or len(names)!=C or len(set(names))!=C
            or p.get('metric_scope')!='all_channels' or task.get('metric_scope')!='all_channels'
            or p.get('supervised_channels')!=list(range(C)) or p.get('output_order')!=names):
            raise ValueError('M all-channel contract mismatch')
        channels=value['channels']
        if not isinstance(channels,list) or len(channels)!=C or value['elements']%C:raise ValueError('M validation channel count')
        n=value['elements']//C
        if tail is not None:
            v=p['training']['eval_batch']
            if type(tail)is not int or not 0<=tail<v:raise ValueError('validation tail identity')
            if n!=(v+(tail or v))*p['pred_len']:raise ValueError('M probe validation shape/elements')
        for i,row in enumerate(channels):
            if not isinstance(row,dict) or set(row)!=fields|{'index','name'} or type(row['index'])is not int or row['index']!=i or row['name']!=names[i]:
                raise ValueError('M validation channel order/schema')
            totals(row,n)
        index=p['target_idx']
        if type(index)is not int or not 0<=index<C or value['MS_target_diagnostic']!=channels[index]:raise ValueError('M target diagnostic identity')
        for k in ('sse','sae'):
            # CPU reduction order may round differently. This is binary64
            # accumulation integrity, not a serial/parallel admission tolerance.
            total=math.fsum(row[k] for row in channels)
            if abs(value[k]-total)>C*math.ulp(total):raise ValueError('M channel aggregation inconsistent')
    return {k:value[k] for k in fields}


def compare_probe_trajectories(c, task, reference, actual):
    """Fail closed on identity; compare exact, named or full-state bounded paths."""
    import math
    from utils.ch3_contract import numeric_probe_policy
    rule=numeric_probe_policy(c,task)
    identity=('id','profile_sha','initial','initial_rng','batch_ids','validation_tail','steps','final_rng')
    for key in identity:
        if key not in reference or key not in actual or reference[key]!=actual[key]:raise ValueError('exact identity/RNG/batch mismatch: '+key)
    if reference['id']!=task['id'] or reference['profile_sha']!=digest(profile(c,task)):raise ValueError('foreign task profile')
    if reference.get('finite') is not True or actual.get('finite') is not True:raise ValueError('finite flag missing')
    if reference['steps']!=6 or len(reference['batch_ids'])!=6:raise ValueError('incomplete short trajectory')
    exact_fields=('trajectory','validation','final')
    if any(k not in x for x in (reference,actual) for k in exact_fields):raise ValueError('required trajectory evidence missing')
    if any(len(x['trajectory'])!=6 for x in (reference,actual)):raise ValueError('six-step coverage missing')
    a=_probe_validation_metrics(c,task,reference['validation'],reference['validation_tail'])
    b=_probe_validation_metrics(c,task,actual['validation'],actual['validation_tail'])
    raw_equal=all(reference[k]==actual[k] for k in exact_fields)
    if rule is None:return dict(passed=raw_equal,mode='exact',bitwise_equal=raw_equal,reason=None if raw_equal else 'exact numerical mismatch')
    if reference.get('numeric_policy_sha')!=digest(rule) or actual.get('numeric_policy_sha')!=digest(rule):raise ValueError('missing or foreign numeric policy evidence')
    failures=[];scalar_max=0.0
    def number(v):
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('nonfinite or nonnumeric comparison value')
        return float(v)
    def compare_number(a,b,label,limit):
        nonlocal scalar_max
        delta=abs(number(a)-number(b));scalar_max=max(scalar_max,delta)
        if delta>limit:failures.append(label)
    metric_limit=rule['metric_atol'];loss_max=0.0;loss_ratio=0.0
    for step,(ta,tb) in enumerate(zip(reference['trajectory'],actual['trajectory']),1):
        if set(ta)!=set(tb) or set(ta)!={'step','loss','state','optimizer'} or ta['step']!=step or tb['step']!=step:raise ValueError('trajectory schema changed')
        la,lb=number(ta['loss']),number(tb['loss'])
        delta=abs(la-lb);loss_max=max(loss_max,delta)
        allowance=rule.get('loss_atol',metric_limit)+rule.get('loss_rtol',0.0)*max(abs(la),abs(lb))
        loss_ratio=max(loss_ratio,delta/allowance)
        compare_number(la,lb,'step%d loss'%step,allowance)
    scalar_max=0.0
    if set(a)!={'mse','mae','sse','sae','elements'} or set(a)!=set(b):raise ValueError('validation schema changed')
    if type(a['elements']) is not int or type(b['elements']) is not int or a['elements']<=0 or a['elements']!=b['elements']:raise ValueError('validation element count mismatch')
    for x in (a,b):
        for key,total in (('mse','sse'),('mae','sae')):
            if number(x[key])!=number(x[total])/x['elements']:raise ValueError('validation aggregation inconsistent')
    for key in ('mse','mae'):compare_number(a[key],b[key],'validation '+key,metric_limit)
    for key in ('sse','sae'):compare_number(a[key]/a['elements'],b[key]/b['elements'],'normalized validation '+key,metric_limit)
    validation_max=scalar_max;scalar_max=max(scalar_max,loss_max)
    if rule['kind']=='named_tensor':
        traces=[reference.get('numeric_trace'),actual.get('numeric_trace')]
        if any(not isinstance(x,list) or len(x)!=6 for x in traces):raise ValueError('numeric trace missing')
        maxima={k:0.0 for k in rule['tensor_fields']};count=0
        for step,(x,y) in enumerate(zip(*traces),1):
            required={'step','parameter_name','optimizer_parameter_id','tensors','exact_model','exact_optimizer','exact_gradients'}
            if set(x)!=required or set(y)!=required:raise ValueError('numeric evidence schema changed')
            if x['step']!=step or y['step']!=step or x['parameter_name']!=rule['parameter'] or y['parameter_name']!=rule['parameter']:raise ValueError('numeric step/parameter mismatch')
            for key in ('optimizer_parameter_id','exact_model','exact_optimizer','exact_gradients'):
                if x[key]!=y[key]:failures.append('step%d exact %s'%(step,key))
            if set(x['tensors'])!=set(rule['tensor_fields']) or set(y['tensors'])!=set(rule['tensor_fields']):raise ValueError('numeric tensor coverage mismatch')
            for key in rule['tensor_fields']:
                aa,bb=x['tensors'][key],y['tensors'][key]
                if set(aa)!={'dtype','shape','values'} or set(bb)!={'dtype','shape','values'}:raise ValueError('packed tensor schema')
                if any(z['dtype']!=rule['dtype'] or z['shape']!=rule['shape'] for z in (aa,bb)):raise ValueError('numeric shape/dtype mismatch')
                size=math.prod(rule['shape'])
                if len(aa['values'])!=size or len(bb['values'])!=size:raise ValueError('numeric element count mismatch')
                delta=max((abs(number(u)-number(v)) for u,v in zip(aa['values'],bb['values'])),default=0.);count+=size;maxima[key]=max(maxima[key],delta)
                if delta>rule['atol']:failures.append('step%d %s exceeds absolute tolerance'%(step,key))
        return dict(passed=not failures,mode='bounded_numeric',policy_id=rule['id'],policy_sha=digest(rule),atol=rule['atol'],metric_atol=metric_limit,rtol=0.0,equal_nan=False,bitwise_equal=raw_equal and all(v==0 for v in maxima.values()),tensor_max_abs=maxima,scalar_normalized_max_abs=scalar_max,compared_elements=count,exact_residual_state=not any('exact' in f for f in failures),failures=failures)
    if rule['kind']=='full_float_state':
        traces=[reference.get('full_numeric_trace'),actual.get('full_numeric_trace')]
        if any(not isinstance(x,list) or len(x)!=6 for x in traces):raise ValueError('full numeric trace missing')
        state_max=0.0;count=0;exact=True
        for step,(x,y) in enumerate(zip(*traces),1):
            if x.get('step')!=step or y.get('step')!=step:raise ValueError('full numeric step mismatch')
            row=_compare_full_numeric_files(rule,x,y);state_max=max(state_max,row['state_max_abs']);count+=row['compared_elements'];exact=exact and row['exact'];failures.extend('step%d '%step+f for f in row['failures'])
        return dict(passed=not failures,mode='bounded_numeric',policy_id=rule['id'],policy_sha=digest(rule),state_atol=rule['state_atol'],metric_atol=metric_limit,rtol=0.0,equal_nan=False,bitwise_equal=raw_equal and state_max==0 and exact,state_max_abs=state_max,scalar_normalized_max_abs=scalar_max,validation_max_abs=validation_max,loss_max_abs=loss_max,loss_error_ratio=loss_ratio,loss_atol=rule.get('loss_atol',metric_limit),loss_rtol=rule.get('loss_rtol',0.0),compared_elements=count,exact_residual_state=exact and not any(f.startswith('step') and 'exact ' in f for f in failures),failures=failures)
    raise ValueError('unknown numeric policy kind')

def memory_growth_review(memory):
    """Material growth AND rate AND no longer-window plateau, not mere monotonic RSS."""
    import math
    from utils.ch3_contract import RSS_PROBE_POLICY as rule
    needed=('allocated','rss_before_hash','rss_after_hash')
    if any(any(k not in r or not isinstance(r[k],(int,float)) or not math.isfinite(r[k]) or r[k]<0 for k in needed) for r in memory):
        raise ValueError('invalid memory observation')
    if len(memory)<4:return dict(blocked=False,needs_long_window=True,triggers=[],scope='insufficient memory observations')
    recent=memory[-4:]
    gpu_growth=all(b['allocated']>a['allocated'] for a,b in zip(recent,recent[1:]))
    long_enough=len(memory)>=rule['long_total_updates']
    window=memory[rule['warmup_updates']:] if long_enough else recent
    rss=[r['rss_before_hash'] for r in window];net=max(0,rss[-1]-rss[0]);rate=net/(len(rss)-1)
    continuous=all(b['rss_before_hash']>a['rss_before_hash'] for a,b in zip(recent,recent[1:]))
    material=net>rule['min_cumulative_bytes'] and rate>rule['min_average_bytes_per_step']
    tail=[r['rss_before_hash'] for r in memory[-rule['plateau_window']:]]
    plateau=(long_enough and max(tail)-min(tail)<=rule['plateau_range_bytes'] and abs(tail[-1]-tail[0])/(len(tail)-1)<=rule['plateau_abs_slope_bytes_per_step'])
    cpu_growth=continuous and material and long_enough and not plateau
    needs_long=continuous and material and not long_enough
    triggers=(['allocated'] if gpu_growth else [])+(['rss_material_long_no_plateau'] if cpu_growth else [])
    return dict(blocked=bool(triggers),needs_long_window=needs_long,triggers=triggers,
        observed_updates=len(memory),rss_continuous_recent=continuous,rss_net_bytes=net,rss_average_bytes_per_step=rate,
        long_window_observed=long_enough,plateau=plateau,after_hash_grows=all(b['rss_after_hash']>a['rss_after_hash'] for a,b in zip(recent,recent[1:])),
        policy_id=rule['id'],scope='six-step screen or 24-update window; not proof of lifetime memory stability')


class FrozenConvReplay:
    """Diagnostic only: isolate one Conv1d with frozen operands and grad_output."""
    def __init__(self,model,name,repetitions=8):
        import torch
        module=dict(model.named_modules()).get(name)
        if not isinstance(module,torch.nn.Conv1d):raise ValueError('exact Conv1d diagnostic path missing: '+name)
        if repetitions not in (4,8):raise ValueError('bounded kernel replay count')
        self.module=module;self.name=name;self.saved={};self.repetitions=repetitions
        def hook(mod,inputs,output):
            if self.saved:return
            self.saved={'x':inputs[0].detach().clone(),'w':mod.weight.detach().clone(),
                        'b':mod.bias.detach().clone() if mod.bias is not None else None}
            def upstream(g):self.saved['g']=g.detach().clone()
            output.register_hook(upstream)
        self.handle=module.register_forward_hook(hook)
    def run(self):
        import torch
        from resource_budget import INSTANCE
        self.handle.remove();v=self.saved
        if set(v)!={'x','w','b','g'}:raise ValueError('frozen upstream missing')
        original=dict(enabled=torch.backends.cudnn.enabled,benchmark=torch.backends.cudnn.benchmark,
                      deterministic=torch.backends.cudnn.deterministic,allow_tf32=torch.backends.cudnn.allow_tf32)
        identity=tensor_digest(v);w=v['w'].requires_grad_(True);results={}
        for label,det in [('current',original['deterministic']),('deterministic',True)]:
            grads=[];outs=[]
            with torch.backends.cudnn.flags(enabled=original['enabled'],benchmark=original['benchmark'],deterministic=det,allow_tf32=original['allow_tf32']):
                for _ in range(self.repetitions):
                    INSTANCE.sample(torch);INSTANCE.charge('forward','frozen-Conv1d-replay')
                    y=self.module._conv_forward(v['x'],w,v['b'])
                    g=torch.autograd.grad(y,w,grad_outputs=v['g'])[0];torch.cuda.synchronize();finite(g);finite(y)
                    grads.append(g.detach().cpu());outs.append(tensor_digest(y))
            results[label]=dict(gradient_hashes=[tensor_digest(g) for g in grads],forward_hashes=outs,
                repeated_gradient_max_abs=max(float((g.double()-grads[0].double()).abs().max()) for g in grads[1:]),
                all_finite=True)
        restored=dict(enabled=torch.backends.cudnn.enabled,benchmark=torch.backends.cudnn.benchmark,
                      deterministic=torch.backends.cudnn.deterministic,allow_tf32=torch.backends.cudnn.allow_tf32)
        if original!=restored or tensor_digest(v)!=identity:raise ValueError('diagnostic changed flags or operands')
        current_varies=len(set(results['current']['gradient_hashes']))>1
        deterministic_exact=len(set(results['deterministic']['gradient_hashes']))==1
        return dict(repetitions_per_mode=self.repetitions,conv_name=self.name,input_shape=list(v['x'].shape),input_stride=list(v['x'].stride()),
            weight_shape=list(w.shape),upstream_shape=list(v['g'].shape),operand_sha=identity,
            original_flags=original,restored_flags=restored,results=results,
            backend_nondeterminism_demonstrated=current_varies and deterministic_exact,
            scope='frozen operands Conv1d backward; no concurrent optimizer or RNG; no global training setting changed')


def rss_window_worker(c,task,out,device='cuda:0'):
    """One bounded 24-update diagnostic; no per-step parameter snapshots/hashing."""
    import torch
    from utils.ch3_contract import RSS_PROBE_POLICY
    p,model,opt,generator=init_training(c,task,device);rows=[]
    def rss():return next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
    def batch(n):return torch.randn(n,p['T'],p['C'],generator=generator),torch.randn(n,p['pred_len'],1,generator=generator)
    n=c['datasets'][task['dataset']]['endpoints'];v=p['training']['eval_batch'];tail=(n[1]-n[0]-p['pred_len']+1)%v
    for i in range(RSS_PROBE_POLICY['long_total_updates']):
        x,y=batch(p['training']['batch']);update(model,opt,x,y,p,device)
        if i==1:evaluate(model,[batch(v),batch(tail or v)],p,device)
        del x,y;torch.cuda.synchronize();r=rss()
        rows.append(dict(step=i+1,allocated=torch.cuda.memory_allocated(),rss_before_hash=r,rss_after_hash=r))
    review=memory_growth_review(rows)
    result=dict(id=task['id'],profile_sha=digest(p),memory=rows,review=review,updates=24,
                data='synthetic only; no full epoch; no per-step state hashing',finite=True)
    dump(Path(out)/'rss-window.json',result)
    if review['blocked'] or review['needs_long_window']:raise RuntimeError('material RSS growth still has no plateau')
    return result

def probe_worker(c,task,out,device='cuda:0',capture_states=False,backend_name=None,backend_repetitions=8,urban_diagnostic=False,urban_confirmation=False):
    import torch
    m_clock=time.monotonic()if task.get('task')=='M' or c.get('native_time_mark') else None
    out=Path(out);p,model,opt,generator=init_training(c,task,device)
    from utils.ch3_onecycle import build
    scheduler=build(opt,p);scheduler_trace=[]
    m_timing=dict(initialization_seconds=time.monotonic()-m_clock,update_seconds=[],validation_seconds=None,includes_synthetic_batch_and_digest=True)if m_clock is not None else None
    from utils.ch3_contract import numeric_probe_policy
    numeric_rule=numeric_probe_policy(c,task);numeric_trace=[None]*6 if numeric_rule and numeric_rule['kind']=='named_tensor' else None
    import random,numpy as np
    def rng():return tensor_digest((random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],generator.get_state()))
    def rss():return next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
    def capture(label):
        if capture_states:torch.save(dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),gradients={k:cpu_tree(v.grad) for k,v in model.named_parameters() if v.grad is not None}),out/(label+'.pt'))
    kernel_replay=FrozenConvReplay(model,backend_name,backend_repetitions) if backend_name else None
    state_digest=ReusableTensorDigest(model.state_dict());full_writer=FullNumericStateWriter(model,opt,out,state_digest,numeric_rule) if numeric_rule and numeric_rule['kind']=='full_float_state' else None
    full_trace=[None]*6 if full_writer else None
    M_full_writer=None;M_full_trace=[]
    if (p.get('task')=='M' or p.get('time_mark') or scheduler is not None) and full_writer is None:
        M_full_writer=FullNumericStateWriter(model,opt,out,state_digest,dict(id=task['group']+'-exact-full-state'))
    initial=state_digest(model.state_dict());initial_rng=rng();capture('initial')
    diagnostic=None
    if urban_confirmation:
        from utils.ch3_urban_confirmation import policy
        if policy(c,task) is None:raise ValueError('foreign confirmation task')
    if urban_diagnostic or urban_confirmation:
        from utils.ch3_urban_capture import DiagnosticCapture
        diagnostic=DiagnosticCapture(model,opt,out,state_digest,generator)
        diagnostic.capture(0)
    b=p['training']['batch'];v=p['training']['eval_batch'];d=c['datasets'][task['dataset']]
    if task['dataset']=='UrbanEV':a,z,_=c['urban_folds'][task['fold']-1];validation_samples=(z-a-p['T']-task['h']+1)*275
    else:
        if d['endpoints'] is None:raise ValueError('version endpoints required for actual validation tail')
        a,z,_=d['endpoints'];validation_samples=z-a-p['pred_len']+1
    tail=validation_samples%v
    def batch(size):
        from utils.ch3_time_marks import synthetic
        pair=(torch.randn(size,p['T'],p['C'],generator=generator),torch.randn(size,p['pred_len'],p['C'] if p.get('task')=='M' else 1,generator=generator))
        mark=synthetic(p,size)
        return pair+(mark,) if mark is not None else pair
    trajectory=[None]*6;batch_ids=[None]*6;memory=[None]*6
    for step in range(6):
        if m_timing is not None:m_update_start=time.monotonic()
        from utils.ch3_time_marks import batch_parts
        current_batch=batch(b);x,y,mark=batch_parts(current_batch);batch_ids[step]=tensor_digest(current_batch);loss=update(model,opt,x,y,p,device,mark)
        if scheduler is not None:
            scheduler.after_successful_update();scheduler_trace.append(scheduler.state_dict())
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        if m_timing is not None:m_timing['update_seconds'].append(time.monotonic()-m_update_start)
        before=rss();allocated=torch.cuda.memory_allocated() if str(device).startswith('cuda') else 0
        trajectory[step]=dict(step=step+1,loss=loss,state=state_digest(model.state_dict()),optimizer=state_digest(opt.state_dict()))
        if numeric_trace is not None:numeric_trace[step]=numeric_probe_snapshot(model,opt,numeric_rule,step+1,state_digest)
        if full_writer is not None:full_trace[step]=full_writer.capture(step+1)
        if M_full_writer is not None:M_full_trace.append(M_full_writer.capture(step+1))
        after=rss()
        if step==1:
            if m_timing is not None:m_validation_start=time.monotonic()
            if urban_confirmation:
                validation_batches=[batch(v),batch(tail or v)]
                validation_batch_ids=[tensor_digest(pair)for pair in validation_batches]
                validation=evaluate(model,validation_batches,p,device)
            else:validation=evaluate(model,[batch(v),batch(tail or v)],p,device)
        del x,y,mark,current_batch
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        if step==1 and m_timing is not None:m_timing['validation_seconds']=time.monotonic()-m_validation_start
        memory[step]=dict(step=step+1,allocated=allocated,reserved=torch.cuda.memory_reserved() if str(device).startswith('cuda') else 0,rss_before_hash=before,rss_after_hash=after,rss=rss())
        capture('step-'+str(step+1))
        if diagnostic is not None:diagnostic.capture(step+1)
    result=dict(id=task['id'],profile_sha=digest(p),initial=initial,initial_rng=initial_rng,batch_ids=batch_ids,trajectory=trajectory,validation=validation,validation_tail=tail,steps=6,final=state_digest(model.state_dict()),final_rng=rng(),memory=memory,allocated=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,reserved=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0,affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),finite=True,diagnostic_state_capture=capture_states,memory_review=memory_growth_review(memory),state_digest_storage_bytes=state_digest.storage_bytes,state_digest_buffer_allocations=state_digest.buffer_allocations,state_digest_policy='preallocated-per-dtype-v1; original digest byte semantics')
    if diagnostic is not None:result['urban_diagnostic_trace']=diagnostic.rows
    if urban_confirmation:
        from utils.ch3_urban_confirmation import cached_endpoint,policy
        target=dict(index=p['target_idx'],pred_len=p['pred_len'],C=p['C'],metric_space='train-standardized target-only')
        endpoint=cached_endpoint(model,validation_batches,lambda bs:evaluate(model,bs,p,device),
            lambda:dict(rng=rng(),model=state_digest(model.state_dict()),optimizer=state_digest(opt.state_dict())),tensor_digest)
        endpoint['target']=target
        result['urban_confirmation']=dict(policy_sha=digest(policy(c,task)),evaluations={'2':dict(metrics=validation,batch_ids=validation_batch_ids,target=target),'6':endpoint})
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        result['endpoint_resources']=dict(allocated=torch.cuda.memory_allocated()if str(device).startswith('cuda')else 0,rss=rss())
        result['allocated']=torch.cuda.max_memory_allocated()if str(device).startswith('cuda')else 0
        result['reserved']=torch.cuda.max_memory_reserved()if str(device).startswith('cuda')else 0
    if p.get('task')=='M' or p.get('time_mark') or scheduler is not None:
        result['M_full_state_trace']=full_trace if full_trace else M_full_trace
        result['M_timing']=m_timing
        result['time_mark']=p.get('time_mark')
    if scheduler is not None:result['scheduler_trace']=scheduler_trace
    if numeric_rule:
        result['numeric_policy_sha']=digest(numeric_rule)
        if numeric_trace is not None:result['numeric_trace']=numeric_trace
        if full_trace is not None:result['full_numeric_trace']=full_trace
    dump(out/'trajectory.json',result)
    if kernel_replay:dump(out/'kernel-replay.json',kernel_replay.run())
    if result['memory_review']['blocked']:raise RuntimeError('material long-window memory growth or GPU allocation growth')
    if result['memory_review']['needs_long_window']:raise RuntimeError('RSS requires 24-update confirmation; not evidence of a leak')
    return result

def formal_worker(c,task,out,approval,resume=False):
    import torch
    from utils.ch3_data import load,batches
    out=Path(out);p=profile(c,task)
    datasets,metadata=load(c,task)
    binding=approval['data_bindings'][task['dataset']]
    if binding.get(task['id'])!=digest(metadata):raise ValueError('data prefix/scaler/task binding mismatch')
    identity=formal_identity(c,task,metadata,approval)
    if p.get('task')=='M':identity.update(task='M',metric_scope='all_channels',parent_MS_profile=p['parent_MS_profile'],from_scratch=True)
    if (out/'result.json').exists():raise FileExistsError('completed run cannot restart')
    p,model,opt,generator=init_training(c,task,'cuda:0');best=BestState(p['training']['patience']);epoch=steps=0
    from utils.ch3_onecycle import build
    scheduler=build(opt,p)
    if resume and approval.get('recovery_scope'):
        from utils.ch3_native_recovery_execution import resume_before_training
        best,epoch,steps=resume_before_training(c,task,out,approval,model,opt,generator)
        dump(out/'manifest.json',dict(identity=identity,task=task,profile=p,metadata=metadata,
             resume_source_ref=__import__('utils.ch3_native_recovery_records',fromlist=['ref']).ref(out/'resume-source.json')))
    elif resume:
        audit_resume(out,approval,task['id'],protocol=c if c.get('type1_followup') else None)
        if json.loads((out/'manifest.json').read_text())['identity']!=identity:raise ValueError('manifest identity mismatch before checkpoint load')
        best,epoch,steps=restore_state(out/'last.pt',model,opt,identity,generator,scheduler)
    elif (out/'last.pt').exists():raise FileExistsError('staging retained; audit then explicit resume required')
    if not resume:dump(out/'manifest.json',dict(identity=identity,task=task,profile=p,metadata=metadata))
    train=batches(datasets['train'],p,'train',generator);val=batches(datasets['validation'],p,'validation')
    for epoch_index in range(epoch+1,p['training']['epochs']+1):
        if best.stopped:break
        from utils.ch3_time_marks import batch_parts
        for batch in train:
            x,y,mark=batch_parts(batch);update(model,opt,x,y,p,'cuda:0',mark);steps+=1
            if scheduler is not None:scheduler.after_successful_update()
        metrics=evaluate(model,val,p,'cuda:0')
        improved=best.update(metrics['mse'],epoch_index)
        if scheduler is not None and hasattr(scheduler,'after_epoch'):
            scheduler.after_epoch(epoch_index,continuing=not best.stopped and epoch_index<p['training']['epochs'])
        if improved:save_state(out/'best.pt',model,opt,identity,best,epoch_index,steps,generator,scheduler)
        save_state(out/'last.pt',model,opt,identity,best,epoch_index,steps,generator,scheduler)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch_index,steps=steps,validation=metrics,best_epoch=best.epoch))+'\n')
    del train,val,datasets,opt
    saved=torch.load(out/'best.pt',map_location='cpu')
    expected_best_identity=identity
    if approval.get('recovery_scope') and resume:
        from utils.ch3_native_recovery_execution import best_identity
        expected_best_identity=best_identity(c,task,out,approval,identity)
    if saved['identity']!=expected_best_identity:raise ValueError('best identity mismatch')
    model.load_state_dict(saved['model'],strict=True);del saved
    # Only the separately approved formal worker can enter this final test path.
    datasets,test_metadata=load(c,task,test_capability={'purpose':'ch3_formal_test','protocol_sha':digest(c),'run_id':task['id']})
    result=evaluate(model,batches(datasets['test'],p,'test'),p,'cuda:0')
    result.update(id=task['id'],input_variant=task['input_variant'],purpose='ch3_formal',protocol_sha=digest(c),best_epoch=best.epoch,
                  seed=2024,std=None,metric_space='train-standardized',stability='Not evaluated')
    if p.get('task')=='M' or p.get('time_mark') or scheduler is not None:
        result.update(task=p.get('task','MS'),metric_scope='all_channels' if p.get('task')=='M' else 'target_only',profile_sha=digest(p),data_sha=digest(metadata),commit=approval['commit'],from_scratch=True,final_test=dict(calls=1,selected='best.pt',sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),epoch=best.epoch))
        if p.get('task')=='M':result['parent_MS_profile']=p['parent_MS_profile']
        if task.get('parent_run_id'):result.update(parent_run_id=task['parent_run_id'],revision='native-time-mark-v1',time_mark=p['time_mark'],parent_profile_sha=p['parent_profile_sha'])
    if 'baseline_unified' in c:
        result.update(scientific_protocol=c['baseline_unified']['id'],task=task['task'],metric_scope=task['metric_scope'],
                      scheduler_sha=digest(p['training']['scheduler']),scheduler_updates=steps)
    if approval.get('recovery_scope'):
        result.update(**{k:approval[k] for k in ('science_baseline_commit','controller_execution_commit','worker_execution_commit','science_computation_fingerprint')},
                      recovery_scope=approval['recovery_scope'],from_scratch=not resume)
        if resume:result['resume_source_ref']=__import__('utils.ch3_native_recovery_records',fromlist=['ref']).ref(out/'resume-source.json')
    dump(out/'result.json',result)


def verified_waves(group,decision):
    q=decision.get('concurrency')
    if q not in (1,2,4) or q>group['q']:raise ValueError('unverified concurrency')
    if decision.get('representatives')!=group['representatives'] or decision.get('status')!='Passed':raise ValueError('group evidence mismatch')
    return [wave[i:i+q] for wave in group['waves'] for i in range(0,len(wave),q)]


def dispatch(waves,command,out,*,placeholder=False):
    """Fixed waves, no refill; own subprocesses only; retain failed output."""
    out=Path(out);active=[]
    def stop(sig,frame):
        for p in active:
            if p.poll() is None:p.terminate()
        raise InterruptedError('safe-stop requested; staging retained')
    old=signal.signal(signal.SIGTERM,stop)
    try:
        for number,wave in enumerate(waves):
            if (out/'STOP').exists():raise InterruptedError('STOP requested between waves')
            active=[];handles=[]
            for run in wave:
                log=(out/(run+'.log')).open('x',encoding='utf-8');handles.append(log)
                active.append(subprocess.Popen(command(run),stdout=log,stderr=subprocess.STDOUT))
            codes=[p.wait() for p in active]
            for f in handles:f.close()
            dump(out/f'wave-{number}.json',dict(tasks=wave,returncodes=codes))
            if any(codes):raise RuntimeError('wave failed; no automatic fresh rerun')
    finally:
        for p in active:
            if p.poll() is None:p.terminate();p.wait()
        signal.signal(signal.SIGTERM,old)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('action',choices=['preflight','dry-run','start','status','logs','complete','safe-stop','worker'],nargs='?',default='preflight')
    parser.add_argument('--approval');parser.add_argument('--probe-report');parser.add_argument('--output');parser.add_argument('--run-id');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();c=validate_manifest(read_profiles());out=Path(args.output or (Path(c['execution']['evidence'])/('formal-'+str(args.model))))
    if c.get('timemixer_revision') and args.model=='TimeMixer' and args.action!='worker':
        from m6_revision_entry import cli
        return cli()
    if 'extension' in c and args.model in ('DLinear','iTransformer','ModernTCN','TimeXer') and args.action!='worker':
        from m6_extension_entry import cli
        return cli()
    if args.action in ['status','logs','complete','safe-stop']:
        if args.action=='safe-stop':
            state=json.loads((out/'controller.json').read_text());pid=state['pid']
            current=Path(f'/proc/{pid}/stat').read_text().split()[21]
            if current!=state['start_ticks']:raise RuntimeError('PID reused; refusing signal')
            os.kill(pid,signal.SIGTERM)
        else:
            complete=(out/'complete.json').exists()
            if complete:validate_model_completion(c,args.model,json.loads((out/'complete.json').read_text()))
            logs=sorted(str(p) for p in out.glob('*/worker.log'))
            control=json.loads((out/'controller.json').read_text()) if (out/'controller.json').exists() else {}
            proc=Path('/proc')/str(control.get('pid','missing'))/'stat'
            try:running=proc.read_text().split()[21]==control.get('start_ticks')
            except OSError:running=False
            print(json.dumps({'output':str(out),'complete':complete,'running':running,
                              'completed_runs':len(list(out.glob('*/result.json'))),
                              'progress':str(out/'progress.json'),'logs':logs},ensure_ascii=False))
        return
    approval=json.loads(Path(args.approval).read_text()) if args.approval else None
    reasons=preflight(c,args.model,approval)
    if args.action in ['preflight','dry-run']:
        print(json.dumps(dict(model=args.model,blocked=reasons,tasks=[t['id'] for t in c['tasks'] if t['model']==args.model]),indent=2))
        if args.action=='preflight' and reasons:raise SystemExit(2)
        return
    if reasons:raise RuntimeError('; '.join(reasons))
    if args.action=='worker':
        from restricted_io_guard import require_installed
        state=require_installed()
        if state.get('purpose')!='ch3_formal':raise PermissionError('formal guarded worker required')
        return formal_worker(c,task_by_id(c,args.run_id),out,approval,args.resume)
    report=json.loads(Path(args.probe_report).read_text())
    validate_probe_report(c,report)
    if hashlib.sha256(Path(args.probe_report).read_bytes()).hexdigest()!=approval['probe_report_sha']:raise ValueError('probe report SHA mismatch')
    if report['protocol_sha']!=digest(c) or report['code']!=code_binding():raise ValueError('probe source/config mismatch')
    if report['hardware']!=hardware_binding() or report['environment']!=environment_binding():raise ValueError('probe hardware/environment mismatch')
    # Reject before creating artifacts; start one model group only.
    waves=[]
    for group in c['groups']:
        if group['model']==args.model:waves.extend(verified_waves(group,report['decisions'][group['id']]))
    recovery={}
    if args.resume:
        if (out/'complete.json').exists():raise FileExistsError('completed group cannot resume')
        for t in c['tasks']:
            if t['model']==args.model:recovery[t['id']]=audit_resume(out/t['id'],approval,t['id'])
        waves=[[run for run in wave if recovery[run]!='complete'] for wave in waves]
        waves=[wave for wave in waves if wave]
    controller_out=out/('recovery-'+str(time.time_ns())) if args.resume else out
    with GPULock(c):
        if not args.resume:out.mkdir(parents=True,exist_ok=False)
        controller_out.mkdir(exist_ok=True)
        dump(out/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],model=args.model))
        from utils.ch3_m6 import run_formal_waves
        run_formal_waves(c,waves,controller_out,out,approval,recovery)
        rows=[json.loads((out/t['id']/'result.json').read_text()) for t in c['tasks'] if t['model']==args.model]
        dump(out/'complete.json',dict(model=args.model,protocol_sha=digest(c),
             task_ids=[t['id'] for t in c['tasks'] if t['model']==args.model],results=summarize(c,rows)))


if __name__=='__main__':main()
