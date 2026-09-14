"""Exact saved-snapshot rounding analysis plus bounded same-layout primitive replay."""
import argparse
import csv
import ctypes
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback

REVISION = 'revin-exact-products-roundoff-v1.1'
CARRIED = dict(forward=119, backward=34, adam=0)
U = F(1, 2**24)
MIN_NORMAL = F(1, 2**126)
MAX_FINITE = F((2**24-1)*2**104)
PRIOR_SHA = '928628f2521a24d3432165f66ed711400d87558e8db47d83ec1db79b351dcf8b'


def read(p):
    return json.loads(Path(p).read_text())


def write_new(p,x):
    payload=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with Path(p).open('x') as f:
        f.write(payload)


def frac(x):
    assert math.isfinite(x)
    assert struct.unpack('<f',struct.pack('<f',x))[0]==x, 'snapshot is not an exact float32 value'
    return F.from_float(x)


def rn32(x):
    """Integer, round-to-nearest ties-to-even; no host float multiplication used."""
    if not x:return F(0)
    sign=-1 if x<0 else 1;x=abs(x);n,d=x.numerator,x.denominator
    exponent=n.bit_length()-d.bit_length()
    if (F(2)**exponent)>x:exponent-=1
    quantum=max(exponent-23,-149)
    scaled=x/(F(2)**quantum)
    q,r=divmod(scaled.numerator,scaled.denominator)
    if 2*r>scaled.denominator or (2*r==scaled.denominator and q%2):q+=1
    rounded=sign*F(q)*(F(2)**quantum)
    if abs(rounded)>MAX_FINITE:raise ArithmeticError('float32 overflow in exact replay')
    return rounded


def gamma(m):
    assert m>=0 and m*U<1
    return m*U/(1-m*U)


def exact_record(x):
    return dict(numerator=str(x.numerator),denominator=str(x.denominator),value=float(x))


def assumption_record(values):
    nonzero=[abs(x) for x in values if x]
    assert all(MIN_NORMAL<=x<=MAX_FINITE for x in nonzero), 'non-normal operands/results invalidate relative model'
    total=sum(nonzero,F(0));assert total<=MAX_FINITE,'possible overflowing partial sum'
    denominator_bits=max((x.denominator.bit_length()-1 for x in nonzero),default=0)
    quantum=F(1,2**denominator_bits)
    assert quantum>=MIN_NORMAL,'nonzero partial sum could enter subnormal range'
    return dict(nonzero=len(nonzero),zeros=len(values)-len(nonzero),min_nonzero=float(min(nonzero)) if nonzero else None,
                max_abs=float(max(nonzero)) if nonzero else 0.,sum_abs=float(total),
                partial_sum_quantum_lower_bound=float(quantum),subnormal_or_overflow_possible=False,
                FTZ_relevance='No nonzero operand/product/possible rounded-product partial sum reaches subnormal range; FTZ cannot affect these values')


def decomposition(high,rounded,actual,product_bound):
    n=len(high);assert n==len(rounded)
    sh=sum(high,F(0));sr=sum(rounded,F(0));sa=frac(actual)
    absh=sum((abs(x) for x in high),F(0));absr=sum((abs(x) for x in rounded),F(0))
    reduction_bound=gamma(n-1)*absr
    assert abs(sr-sh)<=product_bound
    assert abs(sa-sr)<=reduction_bound
    return dict(terms=n,positive_terms=sum(x>0 for x in high),negative_terms=sum(x<0 for x in high),
        product_min=float(min(high)),product_max=float(max(high)),
        positive_sum=float(sum((x for x in high if x>0),F(0))),negative_sum=float(sum((x for x in high if x<0),F(0))),
        sum_abs=float(absh),cancellation_ratio=float(absh/abs(sh)) if sh else None,
        S_high=exact_record(sh),S_rounded=exact_record(sr),S_actual=actual,
        product_rounding_difference=float(sr-sh),reduction_difference=float(sa-sr),total_difference=float(sa-sh),
        product_bound=float(product_bound),m=n-1,m_basis='At most N-1 nontrivial additions in a reduction of these N leaves; zero-initialization additions exact; not an inferred exact kernel order',
        gamma_m=float(gamma(n-1)),reduction_bound=float(reduction_bound),total_bound=float(product_bound+reduction_bound),
        bound_to_abs_final=float((product_bound+reduction_bound)/abs(sa)) if sa else None,
        rounded_products_assumptions=assumption_record(rounded),within_derived_bounds=True)


def selfcheck():
    samples=[F(0),F(1),-F(1),F(1)+F(1,2**24),F(1)+F(3,2**24),F(1,2**126),F(1,2**149),F(0.1)]
    for x in samples:
        expected=struct.unpack('<f',struct.pack('<f',float(x)))[0]
        assert float(rn32(x))==expected
    assert U==F(1,2**24) and gamma(1023)==1023*U/(1-1023*U)
    return dict(status='integer RN ties-to-even and arithmetic no-model selfcheck Passed',samples=len(samples),business_API_calls=0)


def worker(config):
    from restricted_io_guard import require_installed
    from current_policy import require_scope
    from acceptance_driver import verify_seal
    from resource_budget import INSTANCE
    import torch
    state=require_installed();require_scope(state,'new_cuda');assert state['access_policy']=='synthetic_regression'
    assert config['revision']==REVISION
    verify_seal(read(config['seal_file']));torch.set_num_threads(1)
    session=Path(config['session_root']);snap=Path(config['snapshots'])
    stored={}
    def get(side,key):
        k=side+'_'+key
        if k not in stored:stored[k]=read(snap/(k+'.json'))
        return stored[k]
    def values(side,key):return get(side,key)['values']
    def tensor(side,key):
        x=get(side,key);v=torch.tensor(x['values'],dtype=torch.float32)
        t=torch.empty_strided(tuple(x['layout']['shape']),tuple(x['layout']['stride']),dtype=torch.float32)
        t.copy_(v);assert tuple(t.stride())==tuple(x['layout']['stride'])
        return t
    result=dict(revision=REVISION,status='running',models_constructed=0,backward=0,optimizer=0)
    monitor_stop=threading.Event();monitor_errors=[]
    def alarm(signum,frame):raise TimeoutError('roundoff analysis time/resource bound')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
    def monitor():
        while not monitor_stop.wait(.5):
            try:
                INSTANCE.sample(torch)
                if sum(p.stat().st_size for p in session.rglob('*') if p.is_file())>config['limits']['output']:
                    raise RuntimeError('output limit exceeded')
            except BaseException as exc:
                monitor_errors.append(str(exc));INSTANCE.stop(exc);os.kill(os.getpid(),signal.SIGALRM);return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        INSTANCE.current('analysis/saved_RevIN_rounding')
        INSTANCE.charge('forward','independent_exact_complete_RevIN_rounding_reference')
        libc=ctypes.CDLL(None);libc.fegetround.restype=ctypes.c_int
        fe_rounding_mode=libc.fegetround();assert fe_rounding_mode==0,'rounding mode is not FE_TONEAREST on this Linux platform'
        for key in ['norm_pre_affine','norm_cotangent','norm_mul_gout0','norm_mul_gin1','denorm_input','denorm_cotangent','stdev']:
            assert values('A',key)==values('AMD',key),key
        assert values('A','norm_cotangent')==values('A','norm_mul_gout0')
        assert values('A','norm_weight_saved')==[1.]*7==values('AMD','norm_weight_saved')
        assert values('A','denorm_divisor')==[1.]*7 and values('AMD','denorm_divisor')==[1.]
        for side in ('A','AMD'):
            assert values(side,'denorm_input')==values(side,'denorm_numerator'), 'bias subtraction not exact zero in this observed initial-state case'
        q=values('A','norm_pre_affine');g=values('A','norm_mul_gout0');norm_actual=values('A','norm_mul_gin1')
        norm_rows=[];norm_high=[];norm_rounded=[];norm_bounds=[]
        with (session/'norm-products.csv').open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['channel','batch','time','q','g','exact_num','exact_den','rounded32','rounding_error'])
            for c in range(7):
                hp=[];rp=[]
                for b in range(2):
                    for t in range(512):
                        x,y=frac(q[b][t][c]),frac(g[b][t][c]);assert not x or abs(x)>=MIN_NORMAL;assert not y or abs(y)>=MIN_NORMAL
                        p=x*y;r=rn32(p);assert abs(r-p)<=U*abs(p)
                        # Exact product of two f32 normals has <=48 significand bits; cross-check host conversion.
                        assert float(r)==struct.unpack('<f',struct.pack('<f',float(p)))[0]
                        hp.append(p);rp.append(r);writer.writerow([c,b,t,float(x),float(y),p.numerator,p.denominator,float(r),float(r-p)])
                bound=U*sum(map(abs,hp),F(0));row=decomposition(hp,rp,norm_actual[c],bound);row['channel']=c
                norm_rows.append(row);norm_high.append(sum(hp,F(0)));norm_rounded.append(rp)
                norm_bounds.append(bound+gamma(1023)*sum(map(abs,rp),F(0)))
        write_new(session/'norm-exact-analysis.json',norm_rows)
        # Same-kernel replay only checks the formula, axes, order/layout and saved values; exact arithmetic above is independent.
        INSTANCE.charge('forward','norm_same_layout_primitive_replay')
        qt=tensor('A','norm_pre_affine');gt=tensor('A','norm_mul_gout0');products=gt*qt
        expected=torch.empty((2,512,7),dtype=torch.float32)
        for c in range(7):
            expected[:,:,c]=torch.tensor([float(v) for v in norm_rounded[c]],dtype=torch.float32).reshape(2,512)
        norm_replay=products.sum_to_size(7)
        norm_checks=dict(products_correctly_rounded_all=torch.equal(products,expected),sum_matches_saved=torch.equal(norm_replay,torch.tensor(norm_actual)),
                         product_stride=list(products.stride()),input_q_stride=list(qt.stride()),input_g_stride=list(gt.stride()),sum_axes=[0,1],
                         scope='Same installed reduction replay verifies formula/axes/layout only, not an independent kernel proof')
        write_new(session/'norm-primitive-replay.json',norm_checks)
        assert norm_checks['products_correctly_rounded_all'] and norm_checks['sum_matches_saved']
        full=read(snap/'paired-gradients.json')['weight_gradients'];den_reports={};total_reports={}
        std=values('A','stdev');go=values('A','denorm_cotangent');p=values('A','denorm_input')
        assert all(v==0 for batch in go for row in batch for v in row[:6])
        # The scalar constant is the exact binary64 eps*eps used by Python before tensor addition; saved f32 denominator=1.
        eps2=F.from_float(1e-5*1e-5);ideal_d=F(1)+eps2
        for side in ('A','AMD'):
            mapping=list(range(7)) if side=='A' else [6]*7
            saved=values(side,'denorm_div_gin1');expanded=saved if side=='A' else [0.]*6+saved
            channels=range(7) if side=='A' else [6]
            den_high=[F(0)]*7;den_ideal=[F(0)]*7;den_bounds=[F(0)]*7;all_rows=[]
            term_arrays={};gd_expected=torch.empty((2,96,7),dtype=torch.float32);prod_expected=torch.empty((2,96,7),dtype=torch.float32)
            with (session/(side+'-denorm-products.csv')).open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(['channel','batch','time','mapped_weight','g_output','std','p','stage1_rounded32','stage2_rounded32','high_num','high_den'])
                for c in range(7):
                    j=mapping[c]
                    for b in range(2):
                        for t in range(96):
                            gv,sv,pv=frac(go[b][t][c]),frac(std[b][0][j]),frac(p[b][t][c])
                            first=gv*sv;r1=rn32(first);high=-first*pv;second=-r1*pv;r2=rn32(second)
                            assert abs(r1-first)<=U*abs(first) and abs(r2-second)<=U*abs(second)
                            assert float(r1)==values(side,'denorm_div_gout0')[b][t][c]
                            gd_expected[b,t,c]=float(r1);prod_expected[b,t,c]=float(r2)
                            term_arrays.setdefault(j,[]).append((high,r2,first,pv,second))
                            writer.writerow([c,b,t,j,float(gv),float(sv),float(pv),float(r1),float(r2),high.numerator,high.denominator])
                for j in channels:
                    terms=term_arrays[j];hp=[v[0] for v in terms];rp=[v[1] for v in terms]
                    # Two distinct f32 multiplications, two divisions by saved1 and negation are exact.
                    bound1=U*sum((abs(v[2]*v[3]) for v in terms),F(0));bound2=U*sum((abs(v[4]) for v in terms),F(0))
                    row=decomposition(hp,rp,expanded[j],bound1+bound2)
                    sh=sum(hp,F(0));ideal=sh/(ideal_d*ideal_d);input_error=sh-ideal
                    row.update(channel=j,first_multiply_bound=float(bound1),second_multiply_bound=float(bound2),
                        input_denominator_ideal=str(ideal_d),saved_denominator=1.,denominator_input_rounding_difference=float(input_error),
                        complete_denorm_bound=float(bound1+bound2+gamma(len(terms)-1)*sum(map(abs,rp),F(0))+abs(input_error)),
                        divisions_by_saved_one_exact=True,stage1_matches_saved=True)
                    all_rows.append(row);den_high[j]=sh;den_ideal[j]=ideal
                    den_bounds[j]=bound1+bound2+gamma(len(terms)-1)*sum(map(abs,rp),F(0))+abs(input_error)
            write_new(session/(side+'-denorm-exact-analysis.json'),all_rows)
            INSTANCE.charge('forward','denorm_same_layout_primitive_replay_'+side)
            gout=tensor(side,'denorm_cotangent');st=tensor(side,'stdev')
            st=st if side=='A' else st[:,:,6:7]
            gd=gout*st;pn=tensor(side,'denorm_numerator');divisor=tensor(side,'denorm_divisor')
            rp=(-gd)*(pn/divisor)/divisor
            ds=rp.sum_to_size(7 if side=='A' else 1)
            dc=dict(stage1_correctly_rounded_and_matches_saved=torch.equal(gd,gd_expected),
                    stage2_correctly_rounded_all=torch.equal(rp,prod_expected),sum_matches_saved=torch.equal(ds,torch.tensor(saved)),
                    reduction_axes=[0,1] if side=='A' else [0,1,2],reduction_leaves=192 if side=='A' else 1344,
                    product_stride=list(rp.stride()),scope='Same-kernel precision-matched replay; not independent kernel correctness evidence')
            write_new(session/(side+'-denorm-primitive-replay.json'),dc)
            assert dc['stage1_correctly_rounded_and_matches_saved'] and dc['stage2_correctly_rounded_all'] and dc['sum_matches_saved']
            totals=[];actual=full['left' if side=='A' else 'right']
            for c in range(7):
                intermediate=frac(norm_actual[c])+frac(expanded[c]);final=rn32(intermediate);assert float(final)==actual[c]
                ref=norm_high[c]+den_ideal[c];rounding=final-intermediate;assert abs(rounding)<=U*abs(intermediate)
                bound=norm_bounds[c]+den_bounds[c]+U*abs(intermediate);error=frac(actual[c])-ref
                assert abs(error)<=bound
                totals.append(dict(channel=c,norm=norm_actual[c],denorm=expanded[c],final=actual[c],exact_reference=exact_record(ref),
                    final_add_rounding_difference=float(rounding),final_add_bound=float(U*abs(intermediate)),
                    full_actual_minus_reference=float(error),full_bound=float(bound),
                    bound_to_abs_gradient=float(bound/abs(frac(actual[c]))) if actual[c] else None,
                    final_add_matches_record=True,within_derived_bound=True))
            write_new(session/(side+'-full-shared-gradient.json'),totals)
            den_reports[side]=dict(channels=all_rows,replay=dc);total_reports[side]=totals
        result.update(status='complete_derivative_and_roundoff_conditions_Passed',reference_policy='derived roundoff; cross-entry atol/rtol unchanged',
            norm=norm_rows,norm_replay=norm_checks,denorm=den_reports,total=total_reports,FE_rounding_mode=fe_rounding_mode,unit_roundoff=float(U),
            reference_input='Actual saved MulBackward q and full norm cotangent; q was not recomputed in float64',
            independent_reference_precision='Exact Fraction products and exact rational sums; only printed decimal summaries rounded to binary64',
            assumptions='Normal nonzero inputs/products; sum_abs excludes overflow; common product quantum excludes subnormal partial sums; eager separate product/reduce; divisions by saved1 exact',
            limitations='Bounds are conservative and can exceed small cancelled gradients; does not claim high relative accuracy or whole-training stability. Replay uses installed kernels, independent product/sum math does not.')
    except BaseException:
        result.update(status='analysis_failed',traceback=traceback.format_exc())
    finally:
        monitor_stop.set();thread.join(timeout=2);signal.alarm(0);INSTANCE.sample(torch)
        result['budget']=read(config['budget_file']);result['actual']={k:result['budget']['counts'][k]-CARRIED[k] for k in CARRIED}
        result['monitor_errors']=monitor_errors
        result['denials']=[x for x in (json.loads(l) for l in Path(config['audit_log']).read_text().splitlines()) if x['event']=='denied']
        verify_seal(read(config['seal_file']));write_new(config['report_file'],result)
        print(json.dumps({k:result[k] for k in ['status','actual','monitor_errors']}),flush=True)
    return 0 if result['status']=='complete_derivative_and_roundoff_conditions_Passed' and not result['denials'] and not monitor_errors else 1


def launch(repo,evidence):
    from acceptance_driver import validate_inputs,file_seal,verify_seal
    from run_restricted import VERSION,TOOL_ROOT
    from resource_budget import initialize
    import current_policy
    repo=Path(repo).resolve();evidence=Path(evidence).resolve();validate_inputs(repo)
    prior=evidence.parent/'m4611-revin-equivalence-b7u7ow_0';assert hashlib.sha256((prior/'revin-diagnosis-conclusion.json').read_bytes()).hexdigest()==PRIOR_SHA
    assert read(prior/'revin-diagnosis-conclusion.json')['cumulative']==CARRIED
    for line in (prior/'diagnostic-evidence.sha256').read_text().splitlines():
        wanted,rel=line.split('  ',1);assert hashlib.sha256((prior/rel).read_bytes()).hexdigest()==wanted
    session=evidence/'analysis-01';session.mkdir(exist_ok=False);(session/'fixtures').mkdir()
    limits=dict(forward=151,backward=38,adam=0,seconds=180,method_seconds=180,rss=8*1024**3,reserved=4*1024**3,output=1024**3)
    budget=initialize(session/'budget.json','new_cuda/saved_roundoff_analysis',limits)
    with budget.state() as x:x['counts']=dict(CARRIED);x['carried_counts']=dict(CARRIED);x['full_authorized_limits']=dict(forward=512,backward=80,adam=16)
    seal=file_seal(repo)
    for p in (prior/'diagnostic-02').glob('*.json'):seal[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    seal[str(evidence/'authorization.json')]=hashlib.sha256((evidence/'authorization.json').read_bytes()).hexdigest();write_new(session/'sealed-inputs.json',seal)
    config=dict(version=VERSION,repo=str(repo),tool_root=str(TOOL_ROOT),session_root=str(session),revision=REVISION,
        snapshots=str(prior/'diagnostic-02'),audit_log=str(session/'audit.jsonl'),forbidden_roots=[str(repo/'data'),str(repo/'artifacts')],
        stage='new_cuda',access_policy='synthetic_regression',restriction_policy_id=current_policy.POLICY_ID,
        approval_sha256=current_policy.APPROVAL_SHA,budget_file=str(session/'budget.json'),limits=limits,
        seal_file=str(session/'sealed-inputs.json'),report_file=str(session/'report.json'),owner_pid=os.getpid(),business_bootstrap=True)
    path=session/'guard-config.json';write_new(path,config)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker-config',str(path)];write_new(session/'command.json',command)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',PYTHONPATH=os.pathsep.join([str(TOOL_ROOT),str(repo/'tests'),str(repo)]),
        AMD_RR_CONFIG=str(path),AMD_RR_CONFIG_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),TMPDIR=str(session/'fixtures'),CUDA_VISIBLE_DEVICES='')
    start=time.monotonic()
    with (session/'execution.log').open('x') as log:
        p=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_new(session/'process.json',dict(pid=p.pid,command=command,started_monotonic=start))
        try:code=p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    verify_seal(seal);x=dict(exit_code=code,elapsed_seconds=time.monotonic()-start,budget=read(session/'budget.json'))
    write_new(session/'exit.json',x);print(json.dumps({k:v for k,v in x.items() if k!='budget'}));return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo');p.add_argument('--evidence');p.add_argument('--worker-config');p.add_argument('--selfcheck',action='store_true')
    a=p.parse_args()
    if a.selfcheck:print(json.dumps(selfcheck()))
    else:raise SystemExit(worker(read(a.worker_config)) if a.worker_config else launch(a.repo,a.evidence))
