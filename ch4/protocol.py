"""Approved dataset-level protocol, including the latest 20/40 patience amendment."""
MODELS=('AMD','DLinear','ModernTCN','PatchTST','TimeMixer','TimeXer','iTransformer',
        'PDFormer','HimNet','TESTAM','D2STGNN','MegaCRN','STDN','STAEformer','GraphWaveNet','AGCRN',
        'ModernTCN-UV-v10','ch4_v10_pdformer_him_v1')
CITIES=('AMS','JHB','LOA','MEL','SPO','SZH')


def resource_admission_policy():
    """Resource measurements select concurrency; no serial/concurrent equivalence gate."""
    return dict(
        schema='m7-resources-only-concurrency-v1',
        selection_basis='measured_resources_and_throughput',
        selection_metrics=['gpu_memory','safety_headroom','oom_risk','step_time',
                           'end_to_end_throughput','gpu_busy','resource_contention'],
        warmup_updates=5,measured_updates=20,max_unstable_timing_repeats=1,
        gpu_hours_cap=4,concurrency_sequence=[1,2,4],
        q8_requires='measured headroom and continuing total throughput benefit',
        concurrency_numerical_equivalence=dict(
            status='not_executed_by_user_policy',
            statement='本轮按用户批准不执行；基于任务独立执行机制、既有证据及差异风险评估接受'),
        serial_concurrent_tensor_comparisons=False,
        persistent_step_state_dumps=False,
        renamed_equivalence_probes=False,
        retained_protections=['finite','oom','process_ownership','checkpoint_resume_identity',
                              'data_identity','test_access_boundary','artifact_integrity'],
        freeze=['batch','concurrency','device','threads','task_order'],
        concurrency_only_oom='reduce_concurrency_first_keep_scientific_batch',
        singleton_oom='approved_domain_wide_batch_halving')


def validate_resource_admission(plan):
    policy=resource_admission_policy()
    if plan.get('resource_admission_policy')!=policy:
        raise ValueError('approved resources-only concurrency policy required')
    if plan.get('resource_frozen'):
        resource=plan.get('resource_plan')
        if not isinstance(resource,dict):raise ValueError('frozen resource plan missing')
        for key in ('selection_basis','concurrency_numerical_equivalence','persistent_step_state_dumps'):
            if resource.get(key)!=policy[key]:raise ValueError('resource admission policy mismatch: '+key)


def dataset_protocol(domain):
    if domain not in ('UrbanEV','PEMS04','PEMS08',*CITIES):raise ValueError('unregistered domain')
    pems=domain.startswith('PEMS')
    batch=16 if pems else {'AMS':4,'SZH':8}.get(domain,32)
    return dict(batch=batch,eval_batch=batch,batch_candidates=[batch//(2**i) for i in range(batch.bit_length())],
                max_epochs=200 if pems else 50,patience=40 if pems else 20,
                optimizer='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=1e-4 if pems else 1e-5,
                amsgrad=False,foreach=False,clip_norm=5.,min_delta=0.,
                scheduler=dict(name='OneCycleLR',max_lr=.001,pct_start=.10,anneal_strategy='cos',
                               div_factor=25.,final_div_factor=10000.,three_phase=False,
                               cycle_momentum=True,base_momentum=.85,max_momentum=.95),
                scheduler_unit='successful_optimizer_update',main_loss='train_standardized_masked_mse',
                best='strict_less_train_standardized_validation_mse',formal_metrics=['mse','mae'],
                amp=False,dtype='float32',accumulation=1,seed=2024,train_shuffle=True,drop_last=False,
                eval_shuffle=False,validation_frequency='each_completed_epoch',
                horizon_curriculum=False,future_target_feedback=False,workers=0,
                tf32_matmul=False,tf32_cudnn=False,cudnn_benchmark=False,cudnn_deterministic=True,
                deterministic_algorithms=False,threads=1)


def resolved_optimization(name,domain):
    if name not in MODELS:raise ValueError('model excluded from current scope')
    cfg=dataset_protocol(domain);cfg['cap']=cfg['max_epochs']
    cfg['auxiliary']=({'native_moe':1.} if name=='AMD' else
                      {'worst_avoidance':.5,'best_choice':.5,'is_quantile':False,
                       'warmup_epoch':0,'use_uncertainty':False} if name=='TESTAM' else
                      {'memory_triplet':.01,'memory_compact':.01,'triplet_margin':1.} if name=='MegaCRN' else {})
    return cfg


def validate_plan(plan):
    import math
    expected={(d,h,m) for d,h in [('UrbanEV',h) for h in (3,6,9,12)]+[(d,1) for d in (*CITIES,'PEMS04','PEMS08')] for m in MODELS}
    tasks=plan['tasks']
    if len(tasks)!=216 or {(t['domain'],t['H'],t['model']) for t in tasks}!=expected:raise ValueError('exact216 task coverage')
    domain_batches={}
    for task in tasks:
        domain=task['domain'];actual=task['training'];expected_cfg=resolved_optimization(task['model'],domain)
        if actual['batch'] not in expected_cfg['batch_candidates'] or actual['eval_batch']!=actual['batch']:raise ValueError('approved domain batch sequence')
        if domain in domain_batches and domain_batches[domain]!=actual['batch']:raise ValueError('different batches inside one domain')
        domain_batches[domain]=actual['batch']
        expected_cfg.update(batch=actual['batch'],eval_batch=actual['batch'],threads=actual['threads'])
        if not isinstance(actual['threads'],int) or actual['threads']<1 or actual!=expected_cfg:raise ValueError('uniform approved training protocol')
        windows=task['endpoints'][0]-12-task['H']-task['P']+2
        steps=math.ceil(windows/actual['batch'])
        if (task['train_origins'],task['steps_per_epoch'],task['max_updates'])!=(windows,steps,steps*actual['max_epochs']):raise ValueError('window/update budget')
    if sum(t['training']['max_epochs'] for t in tasks)!=16200:raise ValueError('epoch budget')
    validate_resource_admission(plan)
    return True
