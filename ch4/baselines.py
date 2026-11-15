"""Native baseline adapters. Predictions and main losses use train-standardized units."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import torch
from torch import nn
from .imports import author_module
from .protocol import dataset_protocol

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent/'amd-execution-evidence/m7/M7-v10-spatial-implementation-r1/sources'
TEMPORAL = ('AMD','DLinear','ModernTCN','PatchTST','TimeMixer','TimeXer','iTransformer')


class NativeTemporal(nn.Module):
    def __init__(self, name, channels, prediction, frequency='h'):
        super().__init__()
        info=json.loads((ROOT/'ch4/temporal_sources.json').read_text())[name]
        self.name,self.channels,self.prediction=name,channels,prediction
        p=copy.deepcopy(info['profile']);s=p['structure'];src=info['source']
        self.source=info
        if name=='AMD':
            from models.tsAMD_enhanced import AMDEnhanced
            p.update(C=channels,T=12,pred_len=prediction,target_idx=0,aux_idx=list(range(1,channels)))
            p['features']=['target']+[f'aux_{i}' for i in range(1,channels)]
            self.model=AMDEnhanced((12,channels),prediction,s['n_block'],s['dropout'],s['patch'],
                                  s['k'],s['c'],s['alpha'],None,norm=True,layernorm=s['layernorm'],
                                  target_idx=0,teb_context_dim=32,task_mode='target_exogenous',aux_idx=list(range(1,channels)),
                                  ch3_contract=p,use_sonnet_mvca=False,use_target_history_local_shape=False)
        else:
            options=dict(s,task_name='long_term_forecast',seq_len=12,pred_len=prediction,enc_in=channels,
                         dec_in=channels,c_out=channels,label_len=0,features='MS' if name=='TimeXer' else 'M')
            if name in ('TimeMixer','TimeXer','iTransformer'):
                options['freq']=frequency
            if name=='iTransformer':options['output_attention']=False
            with author_module(src['root'],src['module'],src['files']) as module:
                self.model=module.Model(SimpleNamespace(**options))

    def forward(self,batch,**kwargs):
        x=batch['x'];b,l,n,c=x.shape
        x=x.permute(0,2,1,3).reshape(b*n,l,c)
        marks=batch.get('time_marks')
        if self.name in ('TimeMixer','TimeXer','iTransformer'):
            if marks is None:raise ValueError('explicit historical source timeF marks required')
            marks=marks[:,None].expand(-1,n,-1,-1).reshape(b*n,l,-1)
        if self.name=='TimeXer':x=x[:,:,[*range(1,c),0]]
        output=self.model(x,marks,None,None) if self.name in ('TimeMixer','TimeXer','iTransformer') else self.model(x)
        auxiliary=None
        if isinstance(output,tuple):
            output,extra=output[0],output[1]
            if self.name=='AMD':auxiliary=extra
        if self.name not in ('AMD','TimeXer'):output=output[:,:,:1]
        if output.shape!=(b*n,self.prediction,1):raise ValueError('native temporal output shape')
        return output[:,:,0].reshape(b,n,self.prediction).permute(0,2,1),auxiliary


def native_calendar_input(batch):
    x=batch['x'];cal=batch['history_calendar'][:,:,None].expand(-1,-1,x.shape[2],-1)
    # Native HimNet/STAEformer read indices 1 and 2 as TOD/DOW. All auxiliary fields retained.
    return torch.cat((x[...,:1],cal,x[...,1:]),-1)


class NativeSpatial(nn.Module):
    def __init__(self,name,channels,prediction,nodes,domain='UrbanEV',artifacts=None,offset=1):
        super().__init__()
        import yaml
        self.name,self.prediction,self.nodes=name,prediction,nodes
        self.offset=offset
        self.steps_per_day=288 if domain.startswith('PEMS') else 24
        self.artifacts=artifacts or {}
        root=SOURCE_ROOT/name
        domain_profile=domain if domain in ('PEMS04','PEMS08') else None
        if name in ('HimNet','STAEformer'):
            default='METRLA' if name=='HimNet' else 'PEMS08'
            cfg=yaml.safe_load((root/f'model/{name}.yaml').read_text())[domain_profile or default]
            args=copy.deepcopy(cfg['model_args']);args.update(num_nodes=nodes,input_dim=channels+2,out_steps=prediction,output_dim=1)
            if name=='STAEformer':args.update(in_steps=12,steps_per_day=self.steps_per_day)
            if name=='HimNet':args.update(steps_per_day=self.steps_per_day,use_teacher_forcing=False)
            import_root=ROOT/'ch4/vendor/HimNet' if name=='HimNet' else root
            with author_module(import_root,'model.'+name) as mod:
                self.model=getattr(mod,name)(**args)
            if name=='HimNet':
                self.teacher_decay=args['tf_decay_steps']
        elif name=='D2STGNN':
            args=yaml.safe_load((root/f'configs/{domain_profile or "METR-LA"}.yaml').read_text())['model_args']
            args.pop('batch_size',None)
            if 'd2_supports' not in self.artifacts:raise ValueError('D2STGNN transposed doubletransition supports missing')
            args.update(num_nodes=nodes,num_feat=channels,seq_length=12,output_seq_len=prediction,
                        gap=min(3,prediction),steps_per_day=self.steps_per_day,
                        adjs=[torch.as_tensor(v,dtype=torch.float32) for v in self.artifacts['d2_supports']])
            with author_module(ROOT/'ch4/vendor/D2STGNN','models.model') as mod:self.model=mod.D2STGNN(**args)
        elif name=='STDN':
            import configparser
            cfg=configparser.ConfigParser();cfg.read(root/'conf/PEMSD4_1dim_12.conf')
            cfg['Data'].update(num_of_vertices=str(nodes),time_slice_size=str(1440//self.steps_per_day))
            cfg['Training'].update(num_his='12',num_pred=str(prediction),in_channels=str(channels),out_channels='1')
            if 'laplacian32' not in self.artifacts:raise ValueError('STDN native Laplacian positional encoding missing')
            self.register_buffer('laplacian',torch.as_tensor(self.artifacts['laplacian32'],dtype=torch.float32))
            with author_module(ROOT/'ch4/vendor/STDN','model') as mod:self.model=mod.make_model(cfg)
        elif name=='PDFormer':
            required={'adj_mx','sh_mx','dtw_matrix','pattern_keys','laplacian8'}
            if not required.issubset(self.artifacts):raise ValueError('PDFormer native train-only DTW/pattern/Laplacian artifacts missing')
            cfg=json.loads((root/'libcity/config/model/traffic_state_pred/PDFormer.json').read_text())
            if domain_profile:cfg.update(json.loads((root/('PeMS08.json' if domain=='PEMS08' else 'PeMS04.json')).read_text()))
            cfg.update(input_window=12,output_window=prediction,output_dim=1,max_epoch=dataset_protocol(domain)['max_epochs'],
                       device=torch.device('cpu'),lape_dim=8,dataset=domain,use_curriculum_learning=False)
            feature=dict(self.artifacts,num_nodes=nodes,feature_dim=channels+8,ext_dim=8,num_batches=1,scaler=None)
            self.register_buffer('laplacian',torch.as_tensor(self.artifacts['laplacian8'],dtype=torch.float32))
            with author_module(ROOT/'ch4/vendor/PDFormer','libcity.model.traffic_flow_prediction.PDFormer') as mod:self.model=mod.PDFormer(cfg,feature)
        elif name=='MegaCRN':
            # Explicitly approved author entry defaults; training is independent.
            with author_module(ROOT/'ch4/vendor/MegaCRN','model.MegaCRN') as mod:
                self.model=mod.MegaCRN(nodes,channels,1,prediction,64,num_layers=1,cheb_k=3,
                                      ycov_dim=1,mem_num=20,mem_dim=64,cl_decay_steps=2000,
                                      use_curriculum_learning=False)
        elif name=='AGCRN':
            args=SimpleNamespace(num_nodes=nodes,input_dim=channels,output_dim=1,horizon=prediction,
                                 rnn_units=64,num_layers=2,cheb_k=2,embed_dim=2 if domain=='PEMS08' else 10,default_graph=True)
            with author_module(root,'model.AGCRN') as mod:self.model=mod.AGCRN(args)
            # Official Run.py initialization, required for bare FloatTensor weight pools.
            for parameter in self.model.parameters():
                if parameter.dim()>1:nn.init.xavier_uniform_(parameter)
                else:nn.init.uniform_(parameter)
        elif name=='GraphWaveNet':
            if 'gwn_supports' not in self.artifacts:raise ValueError('GraphWaveNet row-normalized doubletransition supports missing')
            supports=[torch.as_tensor(v,dtype=torch.float32) for v in self.artifacts['gwn_supports']]
            with author_module(ROOT/'ch4/vendor/GraphWaveNet','model') as mod:
                self.model=mod.gwnet(torch.device('cpu'),nodes,dropout=.3,supports=supports,gcn_bool=True,
                                     addaptadj=True,aptinit=None,in_dim=channels+1,out_dim=prediction,
                                     residual_channels=32,dilation_channels=32,skip_channels=256,
                                     end_channels=512,kernel_size=2,blocks=4,layers=2)
        elif name=='TESTAM':
            if prediction not in (1,12) or not 1<=offset<=12 or (prediction==12 and offset!=1):
                raise ValueError('TESTAM native 12 slots: select one H slot or full P12')
            with author_module(root,'model') as mod:
                self.model=mod.TESTAM(nodes,dropout=.3,in_dim=channels+1,out_dim=1,hidden_size=32,layers=3,
                                      prob_mul=False,max_time_index=self.steps_per_day)
        else:
            raise ValueError('native adapter not resolved: '+name)

    def forward(self,batch,step=0,**kwargs):
        x=batch['x'];b,l,n,c=x.shape;extra=None
        if self.name=='HimNet':
            future=batch['future_calendar'][:,:,None].expand(-1,-1,n,-1)
            y=self.model(native_calendar_input(batch),future,labels=None,batches_seen=step,label_mask=None)
            y=y[...,0]
        elif self.name=='MegaCRN':
            future=batch['future_calendar'][:,:,:1,None].transpose(2,3).expand(-1,-1,n,-1)
            result=self.model(x,future,labels=None,batches_seen=step,label_mask=None)
            y=result[0][...,0];extra=dict(query=result[2],pos=result[3],neg=result[4])
        elif self.name=='STAEformer':y=self.model(native_calendar_input(batch))[...,0]
        elif self.name=='AGCRN':y=self.model(x,None,teacher_forcing_ratio=0)[...,0]
        elif self.name=='D2STGNN':
            cal=batch['history_calendar'][:,:,None].expand(-1,-1,n,-1)
            y=self.model(torch.cat((x,cal),-1)).transpose(1,2)
        elif self.name=='STDN':
            cal=torch.cat((batch['history_calendar'],batch['future_calendar']),1)
            te=torch.stack((cal[...,1],(cal[...,0]*self.steps_per_day).round()),-1)
            y=self.model(x,te,self.laplacian,'train' if self.training else 'test')[...,0]
        elif self.name=='PDFormer':
            cal=batch['history_calendar']
            dow=torch.nn.functional.one_hot(cal[...,1].long(),7).to(x.dtype)
            ext=torch.cat((cal[...,:1],dow),-1)[:,:,None].expand(-1,-1,n,-1)
            for name in ['geo_mask','sem_mask','pattern_keys']:
                setattr(self.model,name,getattr(self.model,name).to(x.device))
            y=self.model({'X':torch.cat((x,ext),-1)},lap_mx=self.laplacian)[...,0]
        else:
            tod=batch['history_calendar'][:,:,:1,None].transpose(2,3).expand(-1,-1,n,-1)
            signal=torch.cat((x,tod),-1).permute(0,3,2,1)
            if self.name=='GraphWaveNet':
                if self.model.supports is not None:self.model.supports=[v.to(x.device) for v in self.model.supports]
                y=self.model(torch.nn.functional.pad(signal,(1,0,0,0)))[:,:,:,0]
            else:
                result=self.model(signal,gate_out=kwargs.get('diagnostic_aux',False))
                slot=slice(self.offset-1,self.offset) if self.prediction==1 else slice(None)
                if isinstance(result,tuple):
                    result,gate,experts=result
                    extra=dict(gate=gate[:,:,slot],experts=experts[:,:,slot])
                result=result[...,slot]
                y=result[:,0].permute(0,2,1)
        if y.shape!=(b,self.prediction,n):raise ValueError('native spatial output shape '+str(y.shape))
        return y,extra
