"""Query-informed centroid partitions on reached final-layer states.
Every score, sorting operation, and raw value read is charged, not hidden.
"""
from pathlib import Path
import json,time,math
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
def partition(q,k,v,M,kind='scores',singles=64):
    n,d=k.shape;s=k.double()@q.double()/math.sqrt(d)
    if kind=='time':
        order=torch.arange(n);ns=0
    else:
        order=torch.argsort(s,descending=True,stable=True);ns=min(singles,M-1)
    groups=[order[i:i+1] for i in range(ns)]
    groups+=list(torch.tensor_split(order[ns:],M-ns))
    ks=torch.stack([k[g].double().mean(0) for g in groups]);vs=torch.stack([v[g].double().mean(0) for g in groups])
    counts=torch.tensor([len(g) for g in groups],dtype=torch.float64)
    scores=ks@q.double()/math.sqrt(d)+counts.log()
    pi=scores.softmax(0); out=pi@vs
    # Screen of the coordinate-box and true projected-score certificates.
    rad=torch.stack([((k[g].double()-ks[j]).abs().max(0).values*q.double().abs()).sum()/math.sqrt(d) for j,g in enumerate(groups)])
    projected=torch.stack([(s[g]-(ks[j]@q.double()/math.sqrt(d))).abs().max() for j,g in enumerate(groups)])
    rho=torch.stack([(v[g].double()-vs[j]).norm(dim=-1).max() for j,g in enumerate(groups)])
    D=(vs-out).norm(dim=-1)
    def bound(r):return float((pi*(r.expm1()*rho+(r.expm1()-r)*D)).sum())
    exact=s.softmax(0)@v.double()
    return out.float(),{'groups':M,'kind':kind,'singletons':ns,'attention_l2_error_screen':float((out-exact).norm()),'box_certificate_screen':bound(rad),'projected_certificate_screen':bound(projected),'projected_max_radius':float(projected.max())},groups

def tail(o,x_in,w):
    x=x_in+F.linear(o,w['o']);a=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+float(w['eps']))*w['fn']
    x=x+F.linear(F.silu(F.linear(a,w['gate']))*F.linear(a,w['up']),w['down'])
    a=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+float(w['eps']))*w['gamma']
    return F.linear(a,w['cls']),x

def divergence(pz,qz):
    p=pz.double().softmax(-1);q=qz.double().softmax(-1);delta=pz.double()-qz.double()
    # Stable expression avoids cancellation between close logsumexp values.
    dc=delta-(q*delta).sum();kl=torch.log1p((q*torch.expm1(dc)).sum())
    return dict(tv=float((p-q).abs().sum()/2),kl_QP=float(kl),logit_osc=float(delta.max()-delta.min()))

def main():
    torch.set_num_threads(4);z=np.load(ROOT/'data/last_fp32.npz');tw=np.load(ROOT/'data/tail_weights.npz');w={k:torch.from_numpy(tw[k]) if tw[k].shape else float(tw[k]) for k in tw.files}
    O=torch.from_numpy(z['o']);xi=torch.from_numpy(z['x_in']);q=torch.from_numpy(z['q']);k=torch.from_numpy(z['k']);v=torch.from_numpy(z['v']);ref=torch.from_numpy(np.load(ROOT/'data/logits_fp32.npy'));base,hx=tail(O,xi,w)
    print('tail replay max',float((base-ref).abs().max()),'TV',divergence(ref,base),flush=True)
    # Source prefill uses a matrix of 7168 rows, while tail uses one row. Treat
    # the single-row unmodified tail as the common intervention reference.
    np.save(ROOT/'data/logits_tail_reference.npy',base.numpy())
    np.save(ROOT/'data/hidden_tail_reference.npy',hx.numpy())
    results=[]
    for kind,sing in [('time',0),('scores',0),('scores',64),('scores',256)]:
      s=time.perf_counter();outs=[];hr=[]
      for h in range(9):
        out,r,groups=partition(q[h],k[h//3],v[h//3],716,kind,sing);outs.append(out);r['head']=h;hr.append(r)
        if kind=='scores' and sing==64:
          out1=O.clone();out1[h*64:(h+1)*64]=out
          logits,_=tail(out1,xi,w);np.save(ROOT/'data'/f'logits_single_h{h}.npy',logits.numpy())
          np.savez_compressed(ROOT/'data'/f'partition_h{h}.npz',order=torch.cat(groups).numpy(),counts=np.array([len(g) for g in groups]),output=out.numpy())
          r['one_head_divergence']=divergence(base,logits)
      logits,x=tail(torch.cat(outs),xi,w);elapsed=time.perf_counter()-s
      r=dict(kind=kind,singletons=sing,seconds_including_certificate_screen=elapsed,divergence=divergence(base,logits),head_records=hr)
      np.save(ROOT/'data'/f'logits_{kind}_{sing}.npy',logits.numpy());results.append(r);print(json.dumps(r),flush=True)
    (ROOT/'results/interventions.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
