"""Generate the full comparison ledger without model construction or execution."""
import argparse
import ast
import configparser
import json
import math
from pathlib import Path
import subprocess
import yaml
from .imports import sha
from .protocol import MODELS,CITIES,resolved_optimization,validate_plan,resource_admission_policy
from .dependencies import execution_seeds,import_closure,execution_assets
from .pattern_identity import checked_ref,validate_pattern_identity
from .pattern_candidates import validate_candidate_identity

ROOT=Path(__file__).resolve().parents[1]
E=ROOT.parent/'amd-execution-evidence/m7/M7-v10-spatial-implementation-r1'
CORES={'PDFormer':'libcity/model/traffic_flow_prediction/PDFormer.py','HimNet':'model/HimNet.py',
       'TESTAM':'model.py','D2STGNN':'models/model.py','MegaCRN':'model/MegaCRN.py','STDN':'model.py',
       'STAEformer':'model/STAEformer.py','GraphWaveNet':'model.py','AGCRN':'model/AGCRN.py'}


def ref(path):return dict(path=str(path),sha256=sha(path))


def structure(model,domain,c,p,n,h):
    source=E/'sources'/model;sp=domain if domain.startswith('PEMS') else None
    if model in ('ModernTCN-UV-v10','ch4_v10_pdformer_him_v1'):
        value=dict(time=dict(C=c,P=p,L=12,D=64,K=12,blocks=1,ffn_ratio=1,patch_size=1,patch_stride=1,
                             large_kernel=7,small_kernel=3,dropout=.1,head_dropout=0.,revin=True,affine=False,
                             merge_kernels=False,U=True,V=True,G=5,R=4,initialization='fresh',joint_training=True))
        if model.startswith('ch4_'):value['space']=dict(d=64,heads_per_branch=4,head_dim=16,layers=1,r=8,node_embedding_dim=16,
            mlp=[82,32,8],hidden='GELU',output='tanh',geo_offsets=[0,1],sem_offsets=[0],attention_dropout=0.,output_dropout=0.,
            Wp_bias=False,QKV_bias=False,W0LR_bias=False,Wo_bias=False,MLP_bias=True,Wo_init='zero',W0_init='identity',
            other_linear_init='Xavier uniform',node_embedding_init_std=.02,spatial_rng_seed=39025)
        return value
    if model not in CORES:
        historical=json.loads((ROOT/'ch4/temporal_sources.json').read_text())[model]
        return dict(selection='CH3 current adopted second-round UrbanEV MS structure; nine-spatial-baseline fallback explicitly inapplicable',
                    values=historical['profile']['structure'],task_overrides=dict(N=n,C=c,L=12,P=p,target_idx=0),
                    source_profile_ref=ref(ROOT/'ch4/temporal_sources.json'),time_mark_frequency='t' if sp else 'h')
    if model in ('HimNet','STAEformer'):
        key=sp or ('METRLA' if model=='HimNet' else 'PEMS08');path=source/f'model/{model}.yaml'
        values=yaml.safe_load(path.read_text())[key]['model_args'].copy()
        values.update(num_nodes=n,input_dim=c+2,out_steps=p,output_dim=1,steps_per_day=288 if sp else 24)
        if model=='HimNet':values['use_teacher_forcing']=False
        else:values.update(in_steps=12,use_mixed_proj=True)
        return dict(config=ref(path),key=key,values=values,calendar_layout='target,TOD,DOW,all_auxiliary')
    if model=='D2STGNN':
        path=source/f'configs/{sp or "METR-LA"}.yaml';v=yaml.safe_load(path.read_text())['model_args'].copy();v.pop('batch_size')
        v.update(num_nodes=n,num_feat=c,seq_length=12,output_seq_len=p,gap=min(3,p),steps_per_day=288 if sp else 24,
                 use_pre=False,dy_graph=True,sta_graph=True,fk_dim=256,output_hidden=512)
        return dict(config=ref(path),values=v,calendar_layout='all_fields,TOD,DOW')
    if model=='PDFormer':
        path=source/'libcity/config/model/traffic_state_pred/PDFormer.json';cfg=json.loads(path.read_text());refs=[ref(path)]
        if sp:
            extra=source/('PeMS08.json' if sp=='PEMS08' else 'PeMS04.json');cfg.update(json.loads(extra.read_text()));refs.append(ref(extra))
        keys=['embed_dim','skip_dim','geo_num_heads','sem_num_heads','t_num_heads','mlp_ratio','qkv_bias','drop','attn_drop','drop_path',
              's_attn_size','t_attn_size','enc_depth','type_ln','type_short_path','bidir','far_mask_delta','dtw_delta']
        v={k:cfg[k] for k in keys};v.update(input_window=12,output_window=p,output_dim=1,lape_dim=8,
              feature_dim=c+8,ext_dim=8,add_time_in_day=True,add_day_in_week=True,use_curriculum_learning=False,
              cluster_method=cfg.get('cluster_method','kshape'),cand_key_days=cfg.get('cand_key_days',14),n_cluster=16,cluster_max_iter=5)
        return dict(configs=refs,values=v,calendar_layout='all_fields,TOD,DOW_onehot7')
    if model=='STDN':
        path=source/'conf/PEMSD4_1dim_12.conf';cfg=configparser.ConfigParser();cfg.read(path)
        v={k:int(cfg['Training'][k]) for k in ['L','K','d','reference','order']}
        v.update(N=n,C=c,L_history=12,P=p,time_slice_size=5 if sp else 60,lape_dim=32,bn_decay=.1,
                 output_length_adapter='identity' if p==12 else 'Linear(12,1) before native calendar decoder')
        return dict(config=ref(path),reason='author default train.py --config',values=v)
    if model=='MegaCRN':return dict(config=ref(source/'model/traintest_MegaCRN.py'),values=dict(num_nodes=n,input_dim=c,output_dim=1,horizon=p,
                                    rnn_units=64,num_layers=1,cheb_k=3,ycov_dim=1,mem_num=20,mem_dim=64,use_curriculum_learning=False))
    if model=='AGCRN':return dict(config=ref(source/f'model/{"PEMSD8" if sp=="PEMS08" else "PEMSD4"}_AGCRN.conf'),
                                 values=dict(num_nodes=n,input_dim=c,output_dim=1,horizon=p,rnn_units=64,num_layers=2,cheb_k=2,
                                             embed_dim=2 if sp=='PEMS08' else 10,default_graph=True,initialization='source Run.py Xavier/uniform'))
    if model=='GraphWaveNet':return dict(config=ref(source/'README.md'),entry=ref(source/'train.py'),
        values=dict(nodes=n,in_dim=c+1,out_dim=p,dropout=.3,gcn_bool=True,addaptadj=True,randomadj=True,aptonly=False,adjtype='doubletransition',
                    residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2))
    if model=='TESTAM':return dict(config=ref(source/'train.py'),values=dict(num_nodes=n,in_dim=c+1,out_dim=1,dropout=.3,hidden_size=32,layers=3,
                      prob_mul=False,max_time_index=288 if sp else 24,native_slots=12,selected_slots=list(range(12)) if p==12 else [h-1],
                      memory_hidden=32,memory_size=20,nodewise=False,ind_proj=True,attention_type='attention'))
    raise ValueError(model)


def registry():
    rows={}
    for name,core in CORES.items():
        root=E/'sources'/name;acq=json.loads((E/'sources'/(name+'-acquisition.json')).read_text())['result']
        if name=='MegaCRN':
            archive=json.loads((E/'sources/MegaCRN-archive-identity.json').read_text())
            commit=archive['commit'];url='https://github.com/deepkashiwa20/MegaCRN'
        else:commit=acq['commit'];url=acq['url']
        # Hash only source/config/entry files, never bundled datasets or weights.
        files={str(p):sha(p) for p in root.rglob('*') if p.is_file() and '.git' not in p.parts
               and not any(k in p.parts for k in ('data','datasets','raw_data','libcity/cache'))
               and p.suffix in ('.py','.yaml','.conf','.json')}
        files[str(root/core)]=sha(root/core)
        rows[name]=dict(commit=commit,url=url,core=ref(root/core),files=files,
                        acquired=True,original_entry_executed=False,adapter=ref(ROOT/'ch4/baselines.py'))
    historical=json.loads((ROOT/'ch4/temporal_sources.json').read_text())
    for name in MODELS[:7]:rows[name]=dict(historical_source=historical[name]['source'],producer_commit=historical[name]['producer_commit'],
                                         source_profile_identity=ref(ROOT/'ch4/temporal_sources.json'))
    rows['AMD'].update(current_commit='ee8ea3f32e87ed48608d21991a4fc60d06891c23',
                       actual_entry=ref(ROOT/'models/tsAMD_enhanced.py'),frozen_core=ref(ROOT/'models/tsAMD.py'),
                       configuration='all CH3 enhancement toggles off; adopted source profile')
    for name in MODELS[-2:]:rows[name]=dict(parent='ee8ea3f32e87ed48608d21991a4fc60d06891c23',core=ref(ROOT/'ch4/model.py'),spatial=ref(ROOT/'ch4/spatial.py'))
    return rows


def prepare(destination):
    tasks=[]
    native_registry=json.loads((ROOT/'ch4/pattern_registry.json').read_text())
    for domain,h in [('UrbanEV',h) for h in (3,6,9,12)]+[(city,1) for city in CITIES]+[('PEMS04',1),('PEMS08',1)]:
        path=E/'data'/(domain+('-prefix-r1' if domain.startswith('PEMS') else '-prefix-r2'))
        meta=json.loads((path/'metadata.json').read_text());n=meta['nodes'];p=12 if domain.startswith('PEMS') else 1;c=len(meta['features'])
        start,val,end=meta['endpoints'];train_windows=start-12-h-p+2
        val_windows=val-start-(12+h+p-2 if domain=='UrbanEV' else p-1)
        test_windows=end-val-(12+h+p-2 if domain=='UrbanEV' else p-1)
        for model in MODELS:
            cfg=resolved_optimization(model,domain);steps=math.ceil(train_windows/cfg['batch'])
            blockers=['GPU capacity/throughput and final batch/concurrency not yet reviewed']
            artifacts={}
            native_refs=native_registry[domain]
            required={'PDFormer':['adj_mx','sh_mx','laplacian8'],
                      'D2STGNN':['d2_supports'],'STDN':['laplacian32'],'GraphWaveNet':['gwn_supports']}.get(model,[])
            identity_path=checked_ref(native_refs['fixed_graph_identity'])
            identity=json.loads(identity_path.read_text())
            if identity['identity']['domain']!=domain or identity['identity']['node_order']!=meta['node_order'] or identity['identity']['metadata_sha256']!=sha(path/'metadata.json'):
                raise ValueError('fixed graph metadata identity')
            for key in required:
                native=identity_path.parent/(key+'.npy')
                if key+'.npy' in identity['files']:
                    artifact=dict(path=str(native),sha256=identity['files'][key+'.npy']);checked_ref(artifact);artifacts[key]=artifact
                else:blockers.append('missing native artifact: '+key)
            if required:
                if model in identity['failures']:blockers.append('native graph failure: '+identity['failures'][model])
                if model=='PDFormer':
                    validate_candidate_identity(native_refs['candidate_identity'],domain,path/'metadata.json')
                    if native_refs['pattern_identity'] is None:blockers.append('native train-only pattern fit not completed')
                    else:
                        artifacts.update(validate_pattern_identity(native_refs['pattern_identity'],domain,path/'metadata.json',native_refs['fixed_graph_identity']))
            tasks.append(dict(id=f'{domain}-H{h}-{model}-s2024',domain=domain,model=model,C=c,N=n,P=p,H=h,L=12,target_idx=0,seed=2024,
                              node_order=meta['node_order'],features=meta['features'],endpoints=meta['endpoints'],data_path=str(path),metadata_sha256=sha(path/'metadata.json'),
                              train_origins=train_windows,validation_origins=val_windows,test_origins=test_windows,
                              steps_per_epoch=steps,max_updates=steps*cfg['max_epochs'],training=cfg,structure=structure(model,domain,c,p,n,h),
                              native_artifacts=artifacts,native_identities=native_refs if required else {},readiness_blockers=blockers))
    sources=registry();temporal=json.loads((ROOT/'ch4/temporal_sources.json').read_text())
    seeds=execution_seeds(ROOT,E/'sources',temporal);closure=import_closure(seeds)
    binding=dict(closure['files']);source_snapshots={}
    assets=execution_assets(ROOT,E/'sources',temporal);binding.update(assets)
    for name in ('ch4/temporal_sources.json','ch4/pattern_registry.json','configs/ch3_baseline_ms_u96_oc01_v3.json'):binding[str(ROOT/name)]=sha(ROOT/name)
    for task in tasks:
        profile=task['structure']
        for reference in profile.get('configs',[])+([profile['config']] if 'config' in profile else []):
            binding[reference['path']]=reference['sha256']
    for task in tasks:
        metadata_path=Path(task['data_path'])/'metadata.json';binding[str(metadata_path)]=task['metadata_sha256']
        meta=json.loads(metadata_path.read_text())
        for key in ('source_npz','source_archive','distance_source','channel_definition'):
            if key in meta:binding[meta[key]['path']]=meta[key]['sha256']
        if task['domain']=='UrbanEV':
            protocol=json.loads((ROOT/'configs/ch3_baseline_ms_u96_oc01_v3.json').read_text())
            raw=Path(protocol['datasets']['UrbanEV']['path'])
            for name in ('volume.csv','e_price.csv','s_price.csv','weather_central.csv','distance.csv','adj.csv'):
                path=str(raw/name)
                if path not in binding:binding[path]=sha(path)
        for reference in task['native_identities'].values():
            if reference:binding[reference['path']]=reference['sha256']
        pattern_reference=task['native_identities'].get('pattern_identity')
        if task['model']=='PDFormer' and pattern_reference:
            pattern=json.loads(checked_ref(pattern_reference).read_text())
            initialization_reference=pattern['initialization_identity']
            initialization=json.loads(checked_ref(initialization_reference).read_text())
            for reference in (initialization_reference,initialization['centers']):
                checked_ref(reference);binding[reference['path']]=reference['sha256']
            if pattern.get('rollback_policy_identity'):
                reference=pattern['rollback_policy_identity'];checked_ref(reference);binding[reference['path']]=reference['sha256']
        for artifact in task['native_artifacts'].values():binding[artifact['path']]=artifact['sha256']
    for row in sources.values():
        source_snapshots.update(row.get('files',{}))
        if 'historical_source' in row:source_snapshots.update(row['historical_source'].get('files',{}))
    value=dict(scope='m7-v10-spatial-216-v1',models=list(MODELS),tasks=tasks,sources=sources,source_binding=binding,
               dependency_identity=dict(runtime_python=closure,runtime_assets=assets,import_roots=[dict(file=str(p),root=str(r)) for p,r in seeds],
                                        source_snapshots=source_snapshots,historical_only=['retired PatchSTG; never an execution entry']),
               environment=ref(E.parent/'M7-v10-spatial-engineering-repair-r2/environment-r3.json'),
               training_jobs=len(tasks),max_run_epochs=sum(t['training']['max_epochs'] for t in tasks),
               proposed_max_optimizer_updates=sum(t['max_updates'] for t in tasks),formal_test_evaluations=len(tasks),
               resource_admission_policy=resource_admission_policy(),
               resource_frozen=False,resource_plan=None,execution_authorized=False,
               preparation_blockers=sorted(set(b for t in tasks for b in t['readiness_blockers']))+['engineering review','finite resource probe/review','Git closure and actual AUTH'],
               retired=['PatchSTG'],budget_note='candidate batches; freeze after capacity admission; no extra training scope')
    if (len(tasks),value['max_run_epochs'])!=(216,16200):raise AssertionError('scope arithmetic')
    validate_plan(value)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:value[k] for k in ['training_jobs','max_run_epochs','proposed_max_optimizer_updates']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);prepare(parser.parse_args().output)
