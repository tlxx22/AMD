"""Same-level parallel branches; inherited native construction/RevIN/head unchanged."""
import torch
from moderntcn_dev_v1.model import ModernTCNUV as _NativeWiring
from .modules import stats

class ModernTCNUV(_NativeWiring):
    def _block(self,H,X,Z,diagnostics):
        block=self.model.stages[0].blocks[0];B,C,D,N=H.shape
        x=block.dw(H.reshape(B,C*D,N)).reshape(B,C,D,N)
        x=block.norm(x.reshape(B*C,D,N)).reshape(B,C*D,N)
        x=block.ffn1drop1(block.ffn1pw1(x));x=block.ffn1act(x)
        h=block.ffn1drop2(block.ffn1pw2(x)).reshape(B,C,D,N)
        # Both branches consume the same unmodified, once-dropped h.
        delta_v=self.v(h,Z,diagnostics) if self.use_v else None
        h_u=self.u(h,X,diagnostics) if self.use_u else h
        s=h_u+delta_v if self.use_v else h_u
        if diagnostics is not None:
            diagnostics['parallel_post_ffn1']={
                'axes':'B,C,D,N','h':stats(h),
                'delta_U':stats(h_u-h), 'delta_V':stats(delta_v) if delta_v is not None else None,
                'merged':stats(s),'merged_delta':stats(s-h),
                'V_filter_input':'shared_post_ffn1_h','V_router_input':'post_RevIN_Z'}
        x=s.permute(0,2,1,3).reshape(B,D*C,N)
        x=block.ffn2drop1(block.ffn2pw1(x));x=block.ffn2act(x)
        x=block.ffn2drop2(block.ffn2pw2(x))
        return H+x.reshape(B,D,C,N).permute(0,2,1,3)
