"""36 forward / 4 backward / 0 optimizer steps; explicit real-prefix policy."""
from contextlib import ExitStack, contextmanager
from io import StringIO
import json
from pathlib import Path
import time

HASHES = {
    'volume.csv':'a55a095ce75af33c59aece2643d5d71b5cd5a0dc73bb97bc553f0a48f40ace32',
    'e_price.csv':'0076d03b8e400c3e911789e2c7ffb7dd0d44a4414247ead676b508def95bcef4',
    's_price.csv':'d125783e042024157f38d1749232696ea2aa893c61fc31672a3c54374498d3dc',
    'weather_central.csv':'da8c16dcc6a25eadc97ca062998b5dbb01efbb4569efdd693ac98fb5bbc6d065',
    'adj.csv':'93100d3b042086159387ec069efbaf411b90298cdf8a7ada64de214c6bdb5c00',
    'distance.csv':'3630642ddce0e4aac440804c134f3424614ce2bd34fc7bcadd1bc1a3de0d303e',
    'inf.csv':'03c9830965e9e99b29adfb8cceed0eba98d37631f514273cb3fe61f80d63de7c'}
OBSERVATIONS = {'volume.csv','e_price.csv','s_price.csv','weather_central.csv'}


@contextmanager
def prefix_policy(config, counters):
    import restricted_io_guard as guard
    import current_policy
    current_policy.require_scope(config, 'real_prefix')
    import main as runner
    import pandas as pd
    import utils.dataloader_urbanev as data
    from unittest import mock
    root = Path(config['repo'])/'data/UrbanEV/data'
    original_open, original_parser = Path.open, pd.read_csv
    original_hash, original_frame, original_header = data._file_sha256, data._read_csv_frame, data._read_csv_header
    original_dataset = runner.TemporalRegionDataset
    active = {'rows':None, 'path':None, 'hash':False}
    class BoundedText:
        def __init__(self, file, count): self.file, self.count, self.used = file, count, 0
        def __enter__(self): return self
        def __exit__(self,*args): self.file.close()
        def __iter__(self): return self
        def __next__(self):
            if self.used >= self.count:
                counters['test_read'] += 1
                raise RuntimeError('prefix text read would cross approved row limit')
            result = next(self.file); self.used += 1; return result
        def readline(self,*args): return next(self)
        def read(self,*args):
            counters['test_read'] += 1
            raise RuntimeError('unbounded observation text read refused')
    def opening(path, mode='r', *args, **kwargs):
        handle = original_open(path,mode,*args,**kwargs)
        permitted = getattr(guard._LOCAL, 'permitted_real_file', None)
        if path.parent==root and path.name=='inf.csv' and permitted and permitted[1]=='node_metadata':
            return handle
        if path.parent==root and not active['hash']:
            if active['path']!=str(path) or active['rows'] is None:
                handle.close();raise RuntimeError('real read not bound to approved prefix operation')
            return BoundedText(handle,active['rows'])
        return handle
    @contextmanager
    def access(path, role, rows=None):
        if path.parent!=root or path.name not in HASHES:
            raise RuntimeError('real file outside exact M1 allowlist')
        previous=dict(active);active.update(path=str(path),rows=rows,hash=role=='byte_hash')
        try:
            with guard.permit_real_file(path,role): yield
        finally:active.update(previous)
    def hashing(path):
        with access(path,'byte_hash'):value=original_hash(path)
        if value!=HASHES[path.name]:raise RuntimeError('frozen UrbanEV byte identity changed')
        counters['byte_hash_calls']+=1
        return value
    def header(path):
        with access(path,'header',1):return original_header(path)
    def frame(path,*,row_stop=None,**kwargs):
        if path.name not in OBSERVATIONS or row_stop not in (1,3909):
            raise RuntimeError('only production 1/3909-row prefix reader is allowed')
        with access(path,'observation_prefix',row_stop+1):
            result=original_frame(path,row_stop=row_stop,**kwargs)
        counters['prefix_reads']+=1
        return result
    def parsing(source,*args,**kwargs):
        if isinstance(source,(str,Path)) and Path(source).parent==root:
            path=Path(source)
            if path.name!='inf.csv' or kwargs.get('usecols')!=['TAZID']:
                counters['test_parse']+=1;raise RuntimeError('direct real observation parser refused')
            # Metadata only: retain the real production parser and TAZID selection.
            with guard.permit_real_file(path,'node_metadata'):
                return original_parser(source,*args,**kwargs)
        if isinstance(source,StringIO):
            if len(source.getvalue().splitlines())>3910:
                counters['test_parse']+=1;raise RuntimeError('numeric parser prefix exceeds 3909')
        return original_parser(source,*args,**kwargs)
    def dataset(*args,**kwargs):
        split=kwargs['split']
        if split=='test':counters['test_construct']+=1;raise RuntimeError('test Dataset forbidden')
        counters[split+'_construct']+=1
        return original_dataset(*args,**kwargs)
    def no_evaluate(*args,**kwargs):
        counters['test_evaluate']+=1;raise RuntimeError('no complete evaluator in single-batch probe')
    with ExitStack() as stack:
        for obj,name,value in [(Path,'open',opening),(pd,'read_csv',parsing),
                               (data,'_file_sha256',hashing),(data,'_read_csv_header',header),
                               (data,'_read_csv_frame',frame),(runner,'TemporalRegionDataset',dataset),
                               (runner,'evaluate',no_evaluate)]:
            stack.enter_context(mock.patch.object(obj,name,value))
        yield


def args_for(runner,root,horizon,enabled,batch):
    values=['--implementation_variant',runner.THLS_IMPLEMENTATION_VARIANT,
        '--development_protocol_id',runner.THLS_DEVELOPMENT_PROTOCOL,
        '--training_protocol_id',runner.STANDARD_TRAINING_PROTOCOL,
        '--ablation_id',runner.THLS_ABLATION_ID if enabled else runner.THLS_CONTROL_ABLATION_ID,
        '--use_target_history_local_shape',str(enabled).lower(),'--data',str(root),
        '--dataset_id','UrbanEV','--task_mode','target_exogenous','--feature_type','MS',
        '--target','volume','--feature_preset','F4','--fold','6','--seq_len','12',
        '--pred_len','1','--label_horizon',str(horizon),'--patch','12',
        '--batch_size',str(batch),'--seed','2024','--num_threads','4','--progress','false',
        '--n_block','1','--alpha','0','--mix_layer_num','3','--mix_layer_scale','2',
        '--norm','true','--layernorm','true','--dropout','.1','--teb_context_dim','32',
        '--use_pmcr','false','--use_teb','false']
    if enabled:values+=['--local_shape_init_seed','2024']
    return runner.prepare_args(runner.parse_args(values))


def run_probe(config):
    import torch
    import main as runner
    from resource_budget import INSTANCE
    from acceptance_driver import write_new, read_json
    counters=dict(test_read=0,test_parse=0,test_construct=0,test_iterate=0,test_evaluate=0,
                  train_construct=0,validation_construct=0,prefix_reads=0,byte_hash_calls=0)
    results=[];root=Path(config['repo'])/'data/UrbanEV/data'
    if not torch.cuda.is_available():raise RuntimeError('required CUDA probe Pending; no CPU substitution')
    torch.set_num_threads(4)
    with prefix_policy(config,counters):
        for horizon in (3,6,9,12):
            batches=[];states=[]
            for enabled in (False,True):
                args=args_for(runner,root,horizon,enabled,2)
                generator=torch.Generator().manual_seed(2024)
                runtime=runner._build_urbanev_runtime_data(args,generator)
                assert runtime.test_data is None and runtime.n_feature==11
                assert runtime.backend.raw.num_nodes==275 and len(runtime.backend.raw.timestamps)==3909
                assert runtime.backend.features.shape==(3909,275,11)
                runner._validate_loader_contract(args,runtime,runtime.preprocessing)
                schema=runner._build_target_exogenous_schema_contract(args,runtime.preprocessing)
                assert schema['target_indices']==[0] and schema['aux_idx']==list(range(1,11))
                train_batch=next(iter(runtime.train_data));val_batch=next(iter(runtime.val_data))
                batches.append((train_batch,val_batch));states.append(generator.get_state())
                for device in ('cpu','cuda'):
                    runner.set_seed(2024)
                    model=runner._build_model(args,runtime).to(device).eval()
                    for split,(x,y) in zip(('train','validation'),(train_batch,val_batch)):
                        INSTANCE.current(f'real_h{horizon}_{"N" if enabled else "A"}_{device}_{split}')
                        with torch.no_grad():pred,aux,state=model(x.to(device),return_state_source=True)
                        adapted=runner._prediction_for_loss(pred,y.to(device),task_mode=runner.TARGET_EXOGENOUS)
                        assert tuple(x.shape)==(2,12,11) and tuple(adapted.shape)==(2,1)
                        assert torch.isfinite(pred).all() and torch.isfinite(aux).all() and torch.isfinite(state).all()
                        assert tuple(state.shape)==(2,56) and torch.count_nonzero(state[:,-32:])==0
                        # Independent sequential sample identifies the scaler's node.
                        view=runtime.val_data.dataset;_,label=view[0];meta=view.metadata(0)
                        raw=runtime.backend.inverse_transform_target(label,node_position=meta['node_position'])
                        error=abs(float(raw.item())-float(runtime.backend.raw.volume[meta['label_idx'],meta['node_position']]))
                        assert error<1e-4
                        results.append(dict(horizon=horizon,arm='N' if enabled else 'A',device=device,split=split,
                            input_shape=list(x.shape),target_shape=list(adapted.shape),finite=True,inverse_max_abs=error))
                    del model
                if horizon in (3,12):
                    args=args_for(runner,root,horizon,enabled,128)
                    # Same actual Dataset, independently seeded first train batch.
                    loader=torch.utils.data.DataLoader(runtime.train_data.dataset,batch_size=128,shuffle=True,
                                drop_last=True,generator=torch.Generator().manual_seed(2024),num_workers=0)
                    x,y=next(iter(loader));runner.set_seed(2024)
                    model=runner._build_model(args,runtime).cuda().train()
                    INSTANCE.current(f'real_h{horizon}_{"N" if enabled else "A"}_cuda_backward')
                    pred,aux=model(x.cuda());adapted=runner._prediction_for_loss(pred,y.cuda(),task_mode=runner.TARGET_EXOGENOUS)
                    loss=(adapted-y.cuda()).square().mean()+aux;assert torch.isfinite(loss)
                    loss.backward()
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    results.append(dict(horizon=horizon,arm='N' if enabled else 'A',device='cuda',batch=128,
                        training_path=True,finite=True,optimizer_steps=0))
                    del model
                del runtime
            assert torch.equal(states[0],states[1])
            assert all(torch.equal(a,b) for pair_a,pair_b in zip(batches[0],batches[1]) for a,b in zip(pair_a,pair_b))
    INSTANCE.sample(torch)
    assert all(counters[k]==0 for k in ('test_read','test_parse','test_construct','test_iterate','test_evaluate'))
    budget=read_json(config['budget_file']);assert budget['counts']==dict(forward=36,backward=4,adam=0)
    write_new(config['report_file'],dict(success=True,results=results,access=counters,budget=budget,
        checkpoint_reads=0,full_evaluations=0,development_artifacts=0))
    return 0
