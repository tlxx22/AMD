"""Native Model subclass: two explicit first-block additions, no global hooks/cache."""
from types import SimpleNamespace
import torch
from .source import load_author
from .modules import WindowStateModulation,ConditionalTemporalFilter,stats

class ModernTCNUV(load_author().Model):
    def __init__(self,spec):
        v=spec.value
        super().__init__(SimpleNamespace(**v['native_options']))
        self.spec=spec;self.use_u=v['use_u'];self.use_v=v['use_v']
        # CPU-only substreams restore caller RNG, do not touch CUDA RNG.
        for enabled,offset,name,ctor in [(self.use_u,17011,'u',lambda:WindowStateModulation(v['C'],v['D'],v['target_idx'],v['aux_idx'])),(self.use_v,17021,'v',lambda:ConditionalTemporalFilter(v['L'],v['D'],v['N'],v['native_options']['large_size'][0]))]:
            if enabled:
                saved=torch.random.get_rng_state()
                try:
                    torch.random.default_generator.manual_seed(v['module_init_seed']+offset);setattr(self,name,ctor())
                finally:torch.random.set_rng_state(saved)
    def _block(self,H,X,Z,diagnostics):
        block=self.model.stages[0].blocks[0];B,C,D,N=H.shape
        x=block.dw(H.reshape(B,C*D,N)).reshape(B,C,D,N)
        if self.use_v:
            delta=self.v(H,Z,diagnostics);native=x
            x=x+delta
        if diagnostics is not None:pre=x.detach().clone()
        x=block.norm(x.reshape(B*C,D,N))
        if diagnostics is not None:
            diagnostics['native_BN']={'input':stats(pre),'output':stats(x)}
            if self.use_v and not self.training:
                # Local eval affine response computed from this forward's delta;
                # no additional BN invocation or running-stat mutation.
                scale=block.norm.weight/(block.norm.running_var+block.norm.eps).sqrt()
                diagnostics['V']['post_BN_delta']=stats(delta*scale[None,None,:,None])
        x=x.reshape(B,C*D,N)
        x=block.ffn1drop1(block.ffn1pw1(x));x=block.ffn1act(x);x=block.ffn1drop2(block.ffn1pw2(x))
        x=x.reshape(B,C,D,N)
        if self.use_u:x=self.u(x,X,diagnostics)
        x=x.permute(0,2,1,3).reshape(B,D*C,N)
        x=block.ffn2drop1(block.ffn2pw1(x));x=block.ffn2act(x);x=block.ffn2drop2(block.ffn2pw2(x))
        return H+x.reshape(B,D,C,N).permute(0,2,1,3)
    def forward(self,x,te=None,return_diagnostics=False):
        if te is not None:raise ValueError('native second-round ModernTCN uses no extra marks')
        if not self.use_u and not self.use_v:
            y=super().forward(x,te)
            return (y,{}) if return_diagnostics else y
        native=self.model;B,L,C=x.shape
        if (L,C)!=(self.spec.value['L'],self.spec.value['C']):raise ValueError('history shape')
        diagnostics={} if return_diagnostics else None
        # Exact native RevIN -> stem -> one block -> head -> denorm order.
        Z=native.revin_layer(x,'norm').permute(0,2,1)
        stem=Z.reshape(B*C,1,L)
        if native.patch_size!=native.patch_stride:
            stem=torch.cat((stem,stem[:,:,-1:].repeat(1,1,native.patch_size-native.patch_stride)),-1)
        stem=native.downsample_layers[0](stem);D,N=stem.shape[1:]
        H=stem.reshape(B,C,D,N)
        H=self._block(H,x,Z,diagnostics)
        y=native.head(H).permute(0,2,1)
        y=native.revin_layer(y,'denorm')
        return (y,diagnostics) if return_diagnostics else y
