"""One explicitly selected CPU synthetic adapter case; never reads task data."""
import sys
import numpy as np
import torch
from ch4.baselines import NativeTemporal, NativeSpatial, TEMPORAL
from ch4.optimization import resolved_optimization,training_loss
from ch4.native_graphs import doubletransition,d2_doubletransition


def run(name):
    torch.set_num_threads(1);torch.manual_seed(2024)
    for p in ([12] if name=='TESTAM' else [1,12]):
        n=4
        native_input_grad=name not in ('TimeXer','iTransformer')
        x=torch.randn(2,12,n,3,requires_grad=native_input_grad)
        batch=dict(x=x,y=torch.randn(2,p,n),mask=torch.ones(2,p,n,dtype=torch.bool),
                   history_calendar=torch.stack((torch.arange(12)/24,torch.ones(12)),1)[None].repeat(2,1,1),
                   future_calendar=torch.stack((torch.arange(12,12+p)/24,torch.ones(p)),1)[None].repeat(2,1,1),
                   time_marks=torch.randn(2,12,4))
        batch['mask'][0,0,0]=False;batch['y'][0,0,0]=float('nan');batch['y'][0,0,1]=0.
        directed=np.array([[1,2,0,0],[0,1,4,0],[1,0,1,0],[0,0,0,0]],dtype=np.float32)
        artifact=dict(gwn_supports=doubletransition(directed),d2_supports=d2_doubletransition(directed),laplacian32=np.zeros((n,32)),
                      adj_mx=np.eye(n),sh_mx=np.zeros((n,n)),dtw_matrix=np.eye(n),
                      pattern_keys=np.zeros((16,3,1)),laplacian8=np.zeros((n,8)))
        model=NativeTemporal(name,3,p) if name in TEMPORAL else NativeSpatial(name,3,p,n,artifacts=artifact)
        if name in ('D2STGNN','GraphWaveNet'):
            consumed=model.model._model_args['adjs'] if name=='D2STGNN' else model.model.supports
            key='d2_supports' if name=='D2STGNN' else 'gwn_supports'
            for actual,expected in zip(consumed,artifact[key]):
                torch.testing.assert_close(actual,torch.as_tensor(expected),atol=0,rtol=0)
        first=[]
        hook=model.model.register_forward_pre_hook(lambda m,args:first.append((args[0]['X'] if isinstance(args[0],dict) else args[0]).detach().clone()))
        y,extra=model(batch,step=0);hook.remove()
        assert y.shape==(2,p,n) and torch.isfinite(y).all()
        loss,parts=training_loss(y,batch['y'],batch['mask'],extra,resolved_optimization(name,'UrbanEV'),p)
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if native_input_grad:
            assert x.grad is not None and torch.isfinite(x.grad).all()
            grad=x.grad.abs().sum((0,1,2))
        else:
            grad=None
            expected=x.detach().permute(0,2,1,3).reshape(8,12,3)
            if name=='TimeXer':expected=expected[:,:,[1,2,0]]
            assert torch.equal(first[0],expected)
        # Both native channel-independent profiles receive all fields; their
        # unchanged target path has zero auxiliary influence. This is reported.
        if native_input_grad and name not in ('AMD','DLinear'):assert (grad>0).all(),grad
        elif name in ('AMD','DLinear'):
            assert first[0].shape[-1]==3
            assert torch.equal(first[0],x.detach().permute(0,2,1,3).reshape(8,12,3))
        model.eval()
        with torch.no_grad():
            eval_y,_=model(batch,step=0)
            altered=dict(batch,y=torch.full_like(batch['y'],12345.))
            second,_=model(altered,step=0)
            torch.testing.assert_close(eval_y,second,atol=0,rtol=0)
        print(name,p,list(y.shape),parts,'input_gradient',grad.tolist() if grad is not None else 'source in-place normalization: parameter backward + exact input mapping checked',flush=True)


if __name__=='__main__':run(sys.argv[1])
