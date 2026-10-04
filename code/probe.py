"""Collect reached-state data and controlled numerical references. No model training."""
from pathlib import Path
import sys, time, json, platform, os
import numpy as np
import torch
import torch.nn.functional as F
from smollm import Model
ROOT=Path(__file__).resolve().parents[1]
class Probe(Model):
    def __init__(self,path,mode='fp32'):
        super().__init__(path); self.mode=mode; self.records=[];self.last_input=None
    def attn(self,q,k,v,causal):
        # Identical weights and full, unpruned attention in every variant.
        self.current=(q[:,-1].clone(),k.clone(),v.clone()) if causal else None
        self.q_last=q[:,-1].clone()
        if self.mode=='bf16_attention':
            return super().attn(q.bfloat16(),k.bfloat16(),v.bfloat16(),causal).float()
        if self.mode=='fp64_attention':
            return super().attn(q.double(),k.double(),v.double(),causal).float()
        return super().attn(q,k,v,causal)
    @torch.inference_mode()
    def capture(self,ids,capacity=None,save_last=True):
        n=len(ids);capacity=capacity or n
        x=self.embed[torch.tensor(ids)];cache=[];records=[]
        for l,w in enumerate(self.w):
            a=self.rms(x,w['an'])
            q=self.rotary(F.linear(a,w['q']).reshape(n,self.heads,self.hd).transpose(0,1))
            k=self.rotary(F.linear(a,w['k']).reshape(n,self.kvheads,self.hd).transpose(0,1))
            v=F.linear(a,w['v']).reshape(n,self.kvheads,self.hd).transpose(0,1)
            if l==self.layers-1: x_in=x[-1].clone()
            # Avoid copies performed by instrumentation on long arrays.
            md={'fp32':torch.float32,'bf16_attention':torch.bfloat16,'fp64_attention':torch.float64}[self.mode]
            o=Model.attn(self,q.to(md),k.to(md),v.to(md),True).float().transpose(0,1).reshape(n,self.d)
            x=x+F.linear(o,w['o'])
            if l==self.layers-1: ffn_in=x[-1].clone()
            a=self.rms(x,w['fn']);x=x+F.linear(F.silu(F.linear(a,w['gate']))*F.linear(a,w['up']),w['down'])
            kc=torch.empty(self.kvheads,capacity,self.hd);vc=torch.empty_like(kc);kc[:,:n].copy_(k);vc[:,:n].copy_(v);cache.append((kc,vc))
            records.append({'x_norm':float(x[-1].double().norm()),'query_norms':q[:,-1].double().norm(dim=-1).tolist()})
            if l==self.layers-1 and save_last:
                np.savez_compressed(ROOT/'data'/f'last_{self.mode}.npz',q=q[:,-1].numpy(),k=k.numpy(),v=v.numpy(),o=o[-1].numpy(),x_in=x_in.numpy(),ffn_in=ffn_in.numpy(),x=x[-1].numpy(),ids=np.array(ids))
        self.length=n
        logits=F.linear(self.rms(x[-1],self.gamma),self.cls)
        return logits,cache,records

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'fp32'
    torch.set_num_threads(4)
    s=time.perf_counter();m=Probe(os.environ.get('MODEL_PATH','/mnt/data/SmolLM2-135M-Instruct.gguf.bin'),mode);load=time.perf_counter()-s
    ids=np.load(ROOT/'data/all_token_ids.npy')[:7168].tolist()
    s=time.perf_counter();z,c,rec=m.capture(ids,8192);tm=time.perf_counter()-s
    np.save(ROOT/'data'/f'logits_{mode}.npy',z.numpy())
    x=np.load(ROOT/'data'/f'last_{mode}.npz')['x'].astype(np.float64)
    nw=np.load(ROOT/'data/norm_witness.npz');a=(nw['row1'].astype(np.float64)-nw['row2'].astype(np.float64))*nw['gamma'].astype(np.float64)
    A=np.mean(x*x)+m.eps;j=a/np.sqrt(A)-x*(a@x)/(m.d*A**1.5)
    r=dict(mode=mode,model_sha256=m.sha256,torch=torch.__version__,load_seconds=load,prefill_including_snapshot_seconds=tm,hidden_norm=float(np.linalg.norm(x)),rms=float(np.sqrt(A)),readout_pair_gradient_screen=float(np.linalg.norm(j)),records=rec)
    (ROOT/'results'/f'probe_{mode}.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='records'}),flush=True)
    # Tail weights are separate so all formal finite-array checks need no full model.
    w=m.w[-1]
    np.savez_compressed(ROOT/'data'/'tail_weights.npz',**{k:t.numpy() for k,t in w.items()},gamma=m.gamma.numpy(),cls=m.cls.numpy(),eps=m.eps)
if __name__=='__main__': main()
