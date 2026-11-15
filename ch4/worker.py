"""One bound task. Only the queue grants train/validation or one-test execution."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import GraphWindows,MetricTotals
from .model import Chapter4Model
from .baselines import NativeTemporal,NativeSpatial,TEMPORAL
from .optimization import optimizer_and_scheduler,training_loss
from .state import capture,restore,save_checkpoint,update_best
from .imports import sha
from .runtime_io import write_new,read


def log(path,value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(dict(time=time.time(),**value),allow_nan=False)+'\n');stream.flush()


def load_arrays(folder,metadata):
    result={}
    for name in ('x','y','valid','times','geo','sem','target_scale'):
        path=Path(folder)/(name+'.npy')
        if sha(path)!=metadata['files'][path.name]:raise ValueError('data identity changed: '+name)
        result[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    return result


def dataset(task,arrays,split):
    train,val,test=task['endpoints'];start,stop={'train':(0,train),'validation':(train,val),'test':(val,test)}[split]
    return GraphWindows(arrays['x'],arrays['y'],arrays['valid'],arrays['times'],start,stop,
                        prediction=task['P'],offset=task['H'],history_within_split=task['domain']=='UrbanEV')


def make_model(task,arrays):
    name=task['model']
    if name in ('ModernTCN-UV-v10','ch4_v10_pdformer_him_v1'):
        return Chapter4Model(task['C'],task['P'],task['N'],spatial=name.startswith('ch4_'))
    if name in TEMPORAL:return NativeTemporal(name,task['C'],task['P'],'t' if task['domain'].startswith('PEMS') else 'h')
    artifacts={}
    for key,ref in task.get('native_artifacts',{}).items():
        if sha(ref['path'])!=ref['sha256']:raise ValueError('native relation binding changed')
        artifacts[key]=np.load(ref['path'],allow_pickle=False)
    return NativeSpatial(name,task['C'],task['P'],task['N'],task['domain'],artifacts,offset=task['H'])


def predict(model,batch,task,arrays,device,step):
    if isinstance(model,Chapter4Model):
        support=[torch.as_tensor(np.array(arrays[k],copy=True),device=device) for k in ('geo','sem')]
        return model(batch['x'],*support,batch['available']),None
    return model(batch,step=step)


def move(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


@torch.no_grad()
def evaluate(model,loader,task,arrays,device,access,stop,phase,save=None):
    model.eval();standard=MetricTotals();physical=MetricTotals();positions=[MetricTotals() for _ in range(task['P'])]
    scale=torch.as_tensor(np.array(arrays['target_scale'],copy=True),dtype=torch.float64,device=device)
    batches=0;origins=0
    if save is not None:
        save=Path(save);save.mkdir(exist_ok=False)
        shape=(len(loader.dataset),task['P'],task['N'])
        outputs={k:np.lib.format.open_memmap(save/(k+'.npy'),mode='w+',dtype=dtype,shape=shape)
                 for k,dtype in [('prediction',np.float32),('truth',np.float32),('mask',bool)]}
        origin_array=np.lib.format.open_memmap(save/'origin.npy',mode='w+',dtype=np.int64,shape=(shape[0],))
    for batch in loader:
        stop();batch=move(batch,device);log(access,dict(event='before_forward',phase=phase,batch=batches,origins=batch['origin'].tolist()))
        y,_=predict(model,batch,task,arrays,device,0)
        standard.update(y,batch['y'],batch['mask']);physical.update(y.double()*scale,batch['y'].double()*scale,batch['mask'])
        for p,metric in enumerate(positions):metric.update(y[:,p],batch['y'][:,p],batch['mask'][:,p])
        if save is not None:
            end=origins+len(y)
            for k,v in [('prediction',y),('truth',batch['y']),('mask',batch['mask'])]:outputs[k][origins:end]=v.cpu().numpy()
            origin_array[origins:end]=batch['origin'].cpu().numpy()
        origins+=len(y);batches+=1
        log(access,dict(event='after_forward',phase=phase,batch=batches,samples=len(y),valid=int(batch['mask'].sum())))
    if origins!=len(loader.dataset):raise ValueError('evaluation sample coverage')
    if save is not None:
        for value in outputs.values():value.flush()
        origin_array.flush()
        np.save(save/'node_ids.npy',np.asarray(task['node_order']),allow_pickle=False)
        origin_values=np.asarray(loader.dataset.origins)
        labels=origin_values[:,None]+task['H']-1+np.arange(task['P'])[None]
        np.save(save/'target_times.npy',arrays['times'][labels],allow_pickle=False)
    return dict(formal_train_standardized=standard.result(),auxiliary_physical_units=physical.result(),
                positions_train_standardized=[m.result() for m in positions],origins=origins,batches=batches)


def run(task,folder,phase,stop,binding,*,test_folder=None,resume=None,resume_review=None,device='cuda:0'):
    """Called only after queue context validation; no implicit test or source loader."""
    folder=Path(folder);cfg=task['training'];device=torch.device(device)
    torch.set_num_threads(cfg['threads']);random.seed(2024);np.random.seed(2024);torch.manual_seed(2024)
    torch.backends.cuda.matmul.allow_tf32=cfg['tf32_matmul'];torch.backends.cudnn.allow_tf32=cfg['tf32_cudnn']
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(cfg['deterministic_algorithms'])
    if device.type=='cuda':torch.cuda.manual_seed_all(2024)
    prefix=Path(test_folder or task['data_path']);meta=read(prefix/'metadata.json')
    if test_folder is None and sha(prefix/'metadata.json')!=task['metadata_sha256']:raise ValueError('metadata changed')
    arrays=load_arrays(prefix,meta);model=make_model(task,arrays).to(device);access=folder/(phase+'-access.jsonl')
    generator=torch.Generator().manual_seed(2024)
    if phase=='test':
        checkpoint_path=folder/'best.pt'
        if sha(checkpoint_path)!=binding['best_sha256']:raise ValueError('frozen best changed')
        checkpoint=torch.load(checkpoint_path,map_location='cpu')
        if checkpoint['binding']!=binding['training_binding']:raise ValueError('best task binding')
        model.load_state_dict(checkpoint['model'],strict=True)
        loader=DataLoader(dataset(task,arrays,'test'),batch_size=cfg['eval_batch'],shuffle=False,drop_last=False,num_workers=0)
        result=evaluate(model,loader,task,arrays,device,access,stop,'test',folder/'test_predictions')
        write_new(folder/'test-result.json',dict(task_id=task['id'],best_sha256=binding['best_sha256'],**result));return result
    train=DataLoader(dataset(task,arrays,'train'),batch_size=cfg['batch'],shuffle=True,drop_last=False,num_workers=0,generator=generator)
    validation=DataLoader(dataset(task,arrays,'validation'),batch_size=cfg['eval_batch'],shuffle=False,drop_last=False,num_workers=0)
    if len(train)!=task['steps_per_epoch']:raise ValueError('loader update budget differs from resolved task')
    opt,sch=optimizer_and_scheduler(model,cfg,len(train));state=dict(epoch=0,updates=0,best_mse=None,best_epoch=None,stale=0)
    if resume is not None:
        # A reviewed queue capability is required; no automatic technical recovery.
        if not resume_review or resume_review.get('approved') is not True:raise PermissionError('resume review required')
        if sha(resume)!=resume_review['checkpoint_sha256']:raise PermissionError('reviewed recovery checkpoint changed')
        best_ref=resume_review['best_ref']
        if sha(best_ref['path'])!=best_ref['sha256']:raise PermissionError('reviewed prior best changed')
        state=restore(torch.load(resume,map_location='cpu'),model,opt,sch,generator,binding,device)
        if state['epoch']>=cfg['max_epochs'] or state['stale']>=cfg['patience']:raise ValueError('completed task cannot resume')
        import shutil
        with (folder/'best.pt').open('xb') as dst,Path(best_ref['path']).open('rb') as src:shutil.copyfileobj(src,dst)
        write_new(folder/'recovery-source.json',resume_review)
    started=time.monotonic();write_new(folder/'resolved.json',task)
    for epoch in range(state['epoch']+1,cfg['max_epochs']+1):
        model.train();loss_sums=dict(main=0.,auxiliary=0.);valid_count=0;lr_trace=[];beta_trace=[]
        for batch in train:
            stop();batch=move(batch,device);opt.zero_grad(set_to_none=True)
            log(access,dict(event='before_forward',phase='train',epoch=epoch,update=state['updates'],origins=batch['origin'].tolist()))
            prediction,extra=predict(model,batch,task,arrays,device,state['updates'])
            loss,parts=training_loss(prediction,batch['y'],batch['mask'],extra,cfg,task['P'])
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            lr=opt.param_groups[0]['lr'];beta=opt.param_groups[0]['betas'][0]
            count=int(batch['mask'].sum());norm_value=float(norm)
            opt.step();sch.step();state['updates']+=1
            log(access,dict(event='optimizer_completed',phase='train',epoch=epoch,updates=state['updates'],lr=lr,beta1=beta,
                            grad_norm=norm_value,valid=count,**parts))
            valid_count+=count
            for k in loss_sums:loss_sums[k]+=parts[k]*count
            lr_trace.append(lr);beta_trace.append(beta)
        metrics=evaluate(model,validation,task,arrays,device,access,stop,'validation')
        state['epoch']=epoch
        improved,early=update_best(state,metrics['formal_train_standardized']['mse'],epoch,cfg['patience'])
        payload=capture(model,opt,sch,generator,state,binding,device)
        save_checkpoint(folder/'last.pt',payload)
        if improved:save_checkpoint(folder/'best.pt',payload)
        log(folder/'history.jsonl',dict(epoch=epoch,updates=state['updates'],losses={k:v/valid_count for k,v in loss_sums.items()},
                                       validation=metrics,lr=lr_trace,beta1=beta_trace,strict_best=improved,stale=state['stale']))
        if early:break
    result=dict(task_id=task['id'],training_complete=True,epochs=state['epoch'],updates=state['updates'],best_epoch=state['best_epoch'],
                best_validation_mse=state['best_mse'],best_sha256=sha(folder/'best.pt'),last_sha256=sha(folder/'last.pt'),
                elapsed_seconds=time.monotonic()-started,stop_reason='patience' if early else 'cap',test_accesses=0,binding=binding)
    write_new(folder/'train-result.json',result);return result
