"""Non-executable third-round preparation; user selection and later review required.

No activation of the width queue, GPU query, permit generation or controller is
reachable here. Future execution must be implemented against the reviewed
selected-width configuration and adoption boundary, rather than use a default.
"""
import argparse,json
from pathlib import Path
from utils.ch3_contract import ROOT,profile
from utils.ch3_native_recovery_records import bound

CONTRACT_REF={'path': '/public/home/yueweiting/大论文/amd-execution-evidence/m6/m6-epf4-timemixer-y5k7elwc/m-baselines-v1/baseline-unified-v3-ms-seal-m128-recovery1/moderntcn-etth1-numeric-diagnosis-v1/baseline-numeric-admission-v1/patchtst-depth-urban6-repair-v1/patchtst-enc2-adoption-v1/patch-enc1-identity-settle-repair-v1/whole-card-monitor-v1/worker-guard-binding-fix-v1/runtime-no-telemetry-v1/patchtst-width-urban-audit-v1/separate-entrypoints-v1/third-round-preparation.contract.json', 'sha256': '7eef30632fb5cf110beae5105a6fe2261a8232b2af6e20e4c7be69c205daa93d'}
PACKAGE=Path(CONTRACT_REF['path']).parent
ENTRY=ROOT/'m6_third_round_after_selection_entry.py'
WRAPPER=ROOT/'scripts/ch3/start_third_round_after_selection.sh'
RESULT=ROOT.parent/'amd-execution-evidence/m6/m6-formal-launch-dhozikhu/baseline-unified-v3-ms-seal-m128-recovery1-third-after-width-selection-r1'
LOG=PACKAGE/'third-round-launcher.log'
SESSION='ch3-m6-third-after-width-selection-r1'
STAGES=('URBAN_SUBSET','EPF_ALL','M_ALL')
CURRENT_STAGE_RUNS={'URBAN_SUBSET':84,'EPF_ALL':35,'M_ALL':168}
EXECUTION_ACTIONS=('arm','start','preflight','prepare-launch','prepare-permission','probe-child','group-child','safe-stop')


def preparation():
    v=bound(CONTRACT_REF)
    if (v.get('purpose')!='third_round_after_width_selection_non_executable_v1'
        or v.get('executable')is not False or v.get('execution_permitted')is not False
        or any(v.get(k)is not None for k in('selected_nonWeather_d_model','selection_ref','reviewed_config_refs','reviewed_closure','start_authorization_ref','new_main_index_ref','new_main_boundary_ref','entry_A_complete_ref'))
        or v.get('stage_order')!=list(STAGES) or v.get('stage_runs')!=CURRENT_STAGE_RUNS
        or v.get('third_formal')!=287 or v.get('retained_depth_formal')!=48
        or v.get('Weather_d_model')!=128 or v.get('Weather_full_parent_unchanged')is not True
        or v.get('GPU_preflight_allowed')is not False or v.get('creates_results')is not False
        or any(v.get(k)is not True for k in('does_not_use_A_start_authorization','does_not_call_A','does_not_receive_automatic_A_completion'))):
        raise PermissionError('exact non-executable third-round preparation only; no inferred selection')
    future=v['only_after_user_selection']
    if (future.get('allowed_nonWeather_widths')!=[4,8]
        or future.get('M_ALL_PatchTST_nonWeather_tasks')!=20
        or future.get('new_main_index')!={'total':371,'switched_nonWeather_PatchTST':20,'preserved_Weather':4,'preserved_other':347,'preserved_total':351}
        or future.get('fresh_third_permission_required')is not True):
        raise PermissionError('exact future selected-width adoption boundary required')
    source=bound(v['source_ref']);refs=source['refs']
    if (source.get('depth_completed_new_formal')!=48 or source.get('old_completed_new_formal')!=196
        or source.get('third_formal')!=0 or source.get('third_round_paused')is not True
        or v['r6_Urban_complete_ref']!=refs['probe/URBAN_SUBSET/complete.json']
        or v['r6_failure_ref']!=refs['queue/controller/failure.json']
        or v['r6_fixed_enc2_index_ref']!=refs['round2-amendment/queue/round2-main-results.patchtst-enc2-v1.json']
        or v['r6_fixed_enc2_boundary_ref']!=refs['round2-amendment/queue/round2-boundary.patchtst-enc2-v1.json']
        or v['retained_depth_boundaries']!={k:refs[k]for k in('PATCH_ENC1','PATCH_ENC2')}):
        raise PermissionError('retained r6 completion/probe references must be exact')
    # Parent profiles are source evidence only, never executable B configs.
    c=bound(v['parent_config_refs']['M_ALL'])
    for t in c['tasks']:
        if t['model']=='PatchTST'and t['dataset']=='Weather':
            p=profile(c,t)
            if [p['structure'][k]for k in('d_model','n_heads','e_layers','d_ff')]!=[128,16,2,256]or(p['training']['epochs'],p['training']['patience'])!=(20,10):
                raise PermissionError('original full Weather128 parent must remain')
    return v


def status():
    v=preparation()
    return dict(scope='m6-third-round-after-width-selection-preparation-v1',state='NON_EXECUTABLE_WAIT_USER_SELECTION_AND_REVIEW',
        blocked=list(v['required_future_gates']),READY_TO_ARM_HANDOFF=False,READY_FOR_GPU_EXECUTION=False,
        execution_permitted=False,selected_nonWeather_d_model=None,executable_config_refs=None,
        source_ref=v['source_ref'],contract_ref=CONTRACT_REF,planned_stage_runs=CURRENT_STAGE_RUNS,remaining_formal_runs=287,
        planned_resume_at='URBAN_AUTO_AUDIT_AFTER_REVIEWED_SOURCE_ADOPTION',saved_Urban_probe_ref=v['r6_Urban_complete_ref'],
        old_failure_preserved=True,retained_depth_formal_runs=48,redispatch_depth=False,
        future_main_index=v['only_after_user_selection']['new_main_index'],Weather_d_model=128,
        executable_permission_bound=False,permission_exists=(PACKAGE/'third-round-start-review.json').exists(),result_root=str(RESULT),log=str(LOG),session=SESSION,
        result_exists=RESULT.exists()or RESULT.is_symlink(),log_exists=LOG.exists()or LOG.is_symlink(),
        GPU_preflight_executed=False,automatic_call_A=False,result_review='pending')


def cli(argv=None):
    p=argparse.ArgumentParser(description='Preparation only; B cannot start before user selection and later byte review/closure.')
    p.add_argument('action',choices=('dry-run','plan','status','logs','complete','second-round','third-round',*EXECUTION_ACTIONS),nargs='?',default='dry-run')
    p.add_argument('--approval');p.add_argument('--approval-sha')
    a=p.parse_args(argv)
    v=status()
    if a.action in EXECUTION_ACTIONS:
        v['requested_action']=a.action;v['refusal']='No executable B permission or selected-width configuration exists; current entry never launches, queries GPU, writes a result or signals a process.'
        # Any A/old approval argument is deliberately not opened or consumed.
        print(json.dumps(v,ensure_ascii=False,indent=2));return 2
    if a.action=='logs':v['logs']=[str(LOG)];v['materialized']=False
    if a.action in('second-round','third-round'):v['results_available']=False;v['new_main_index_generated']=False
    print(json.dumps(v,ensure_ascii=False,indent=2))
    return 2 if a.action=='complete'else 0
