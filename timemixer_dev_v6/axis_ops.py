"""Exact torch layout equivalents for the six fixed author einops expressions.

The existing amd environment has no einops. No installation/global aliases: this
local, literal-only compatibility layer changes neither algorithms nor weights.
"""
def rearrange(x,pattern,**sizes):
    if pattern=='b (seg_num seg_len) d -> (b d seg_num) seg_len':
        b,L,C=x.shape;p=sizes['seg_len'];return x.permute(0,2,1).reshape(b*C*(L//p),p)
    if pattern=='(b d seg_num) d_model -> b d seg_num d_model':
        b,C=sizes['b'],sizes['d'];return x.reshape(b,C,x.shape[0]//(b*C),x.shape[-1])
    if pattern=='b ts_d seg_num d_model -> (b ts_d) seg_num d_model':
        b,C,N,D=x.shape;return x.reshape(b*C,N,D)
    if pattern=='(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model':
        b=sizes['b'];BC,N,D=x.shape;C=BC//b;return x.reshape(b,C,N,D).permute(0,2,1,3).reshape(b*N,C,D)
    if pattern=='(b seg_num) ts_d d_model -> b ts_d seg_num d_model':
        b=sizes['b'];BN,C,D=x.shape;return x.reshape(b,BN//b,C,D).permute(0,2,1,3)
    raise ValueError('expression outside fixed extracted author layout: '+pattern)

def repeat(x,pattern,**sizes):
    if pattern!='seg_num factor d_model -> (repeat seg_num) factor d_model':raise ValueError('expression outside fixed router repeat')
    N,K,D=x.shape;b=sizes['repeat'];return x[None].expand(b,N,K,D).reshape(b*N,K,D)
