"""Unified prediction MSE with separately accounted native method auxiliaries."""
import torch
from .data import masked_mse
from .protocol import resolved_optimization


def optimizer_and_scheduler(model,cfg,steps_per_epoch):
    if steps_per_epoch<1:raise ValueError('empty training loader')
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],
                         weight_decay=cfg['weight_decay'],amsgrad=False,foreach=False)
    args={k:v for k,v in cfg['scheduler'].items() if k!='name'}
    scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,total_steps=cfg['max_epochs']*steps_per_epoch,**args)
    return opt,scheduler


def training_loss(prediction,target,mask,extra,cfg,length):
    main=masked_mse(prediction[:,:length],target[:,:length],mask[:,:length]);aux=main.new_zeros(())
    if cfg['auxiliary'].get('native_moe'):
        if extra is None:raise ValueError('native AMD auxiliary loss missing')
        aux=cfg['auxiliary']['native_moe']*extra
    elif 'worst_avoidance' in cfg['auxiliary']:
        if extra is None:raise ValueError('TESTAM expert/gate outputs required')
        gate=extra['gate'].squeeze(-2);experts=extra['experts'].squeeze(-2)
        truth=target.permute(0,2,1);valid=mask.permute(0,2,1)
        error=(experts[valid]-truth[valid].unsqueeze(-1)).square()
        gates=gate[valid]
        chosen=gates.argmax(-1,keepdim=True);worst=error.argmax(-1,keepdim=True);best=error.argmin(-1,keepdim=True)
        selected=torch.zeros_like(gates).scatter_(-1,chosen,1.)
        residual=error.scatter(-1,worst,0.);den=residual.sum(-1,keepdim=True)
        # Source division has a 0/0 case for perfect experts; its continuous zero-loss extension is zero.
        scaling=torch.where(den>0,residual/den.clamp_min(torch.finfo(error.dtype).tiny),torch.zeros_like(residual))*(1-selected)
        worst_label=torch.where(worst==chosen,scaling,selected).detach()
        best_label=torch.where(best==chosen,selected,torch.zeros_like(gates).scatter_(-1,best,1.)).detach()
        log=gates.clamp_min(torch.finfo(gates.dtype).tiny).log()
        aux=-.5*(worst_label*log).mean()-.5*(best_label*log).mean()
        main=error.mean()  # source supervision on all three native experts, now squared error
    elif 'memory_triplet' in cfg['auxiliary']:
        if extra is None:raise ValueError('MegaCRN memory tensors missing')
        query,pos,neg=extra['query'],extra['pos'],extra['neg']
        aux=.01*torch.nn.functional.triplet_margin_loss(query,pos.detach(),neg.detach(),margin=1.)+.01*torch.nn.functional.mse_loss(query,pos.detach())
    return main+aux,dict(main=float(main.detach()),auxiliary=float(aux.detach()))
