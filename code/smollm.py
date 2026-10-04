"""Frozen SmolLM2 Q8_0 export, dequantized exactly into binary32 weights.

Uses adjacent-pair RoPE, as required by the permuted Q/K rows in GGUF.
PyTorch operators define this executable; equality with llama.cpp or the
original BF16 checkpoint is NOT asserted. The verifier consumes saved bits.
"""
from __future__ import annotations
import time, math, json
from pathlib import Path
import numpy as np
import regex
import torch
import torch.nn.functional as F
from gguf_reader import GGUF

EXPECTED_SHA='5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83'

class Tokenizer:
    def __init__(self,g):
        meta=g.meta; self.tokens=meta['tokenizer.ggml.tokens'];self.lookup={x:i for i,x in enumerate(self.tokens)}
        self.ranks={tuple(x.split(' ')):i for i,x in enumerate(meta['tokenizer.ggml.merges'])}
        bs=list(range(ord('!'),ord('~')+1))+list(range(0xa1,0xad))+list(range(0xae,0x100));cs=bs[:];n=0
        for b in range(256):
            if b not in bs:bs.append(b);cs.append(256+n);n+=1
        self.be={b:chr(c) for b,c in zip(bs,cs)};self.bd={chr(c):b for b,c in zip(bs,cs)}
        self.pattern=regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        self.special={self.tokens[i]:i for i,t in enumerate(meta['tokenizer.ggml.token_type']) if t in (3,4)}
        self.cache={}
    def bpe(self,piece):
        if piece in self.cache:return self.cache[piece]
        p=list(''.join(self.be[b] for b in piece.encode('utf8')))
        while len(p)>1:
            choices=[(self.ranks.get((p[i],p[i+1]),10**10),i) for i in range(len(p)-1)]
            rank,i=min(choices)
            if rank==10**10:break
            p[i:i+2]=[p[i]+p[i+1]]
        out=[self.lookup[s] for s in p];self.cache[piece]=out;return out
    def encode(self,text,special=True):
        ids=[]
        pieces=regex.split('('+'|'.join(regex.escape(s) for s in sorted(self.special,key=len,reverse=True))+')',text) if special else [text]
        for piece in pieces:
            if piece in self.special:ids.append(self.special[piece]);continue
            # SmolLM first isolates individual digits, then applies GPT-2 splitting.
            for segment in regex.split(r'(\p{N})',piece):
                if not segment:continue
                for m in self.pattern.finditer(segment):ids.extend(self.bpe(m.group()))
        return ids
    def decode(self,ids):
        out=bytearray()
        for i in ids:
            s=self.tokens[int(i)]
            if s in self.special:out.extend(s.encode())
            else:out.extend(self.bd[c] for c in s)
        return out.decode('utf8',errors='replace')

class Model:
    def __init__(self,path):
        g=GGUF(path)
        if g.sha256!=EXPECTED_SHA:raise ValueError('checkpoint SHA256 mismatch')
        self.sha256=g.sha256;m=g.meta;self.tokenizer=Tokenizer(g);self.metadata=m
        self.d=m['llama.embedding_length'];self.ff=m['llama.feed_forward_length'];self.layers=m['llama.block_count'];self.heads=m['llama.attention.head_count'];self.kvheads=m['llama.attention.head_count_kv'];self.hd=self.d//self.heads;self.context=m['llama.context_length'];self.vocab=len(m['tokenizer.ggml.tokens']);self.group=self.heads//self.kvheads;self.eps=m['llama.attention.layer_norm_rms_epsilon']
        def t(n):return torch.from_numpy(g.tensor(n))
        self.embed=t('token_embd.weight');self.cls=t('output.weight') if 'output.weight' in g.tensors else self.embed
        self.gamma=t('output_norm.weight');self.w=[]
        for l in range(self.layers):
            self.w.append({k:t(f'blk.{l}.{n}.weight') for k,n in {'q':'attn_q','k':'attn_k','v':'attn_v','o':'attn_output','an':'attn_norm','fn':'ffn_norm','gate':'ffn_gate','up':'ffn_up','down':'ffn_down'}.items()})
        inv=1/(m['llama.rope.freq_base']**(torch.arange(0,self.hd,2,dtype=torch.float32)/self.hd))
        angles=torch.arange(self.context,dtype=torch.float32)[:,None]*inv[None,:];self.cos=angles.cos();self.sin=angles.sin()
    def rms(self,x,w):return x*torch.rsqrt((x*x).mean(-1,keepdim=True)+self.eps)*w
    def rotary(self,x,start=0):
        n=x.shape[1];c=self.cos[start:start+n][None,:,:];s=self.sin[start:start+n][None,:,:];y=torch.empty_like(x)
        y[...,0::2]=x[...,0::2]*c-x[...,1::2]*s;y[...,1::2]=x[...,0::2]*s+x[...,1::2]*c;return y
    def attn(self,q,k,v,causal):
        # GQA grouping avoids materializing a repeated cache and keeps CPU flash SDPA.
        outputs=[]
        for h in range(self.kvheads):
            qs=q[h*self.group:(h+1)*self.group].unsqueeze(0)
            ks=k[h:h+1].expand(self.group,-1,-1).unsqueeze(0);vs=v[h:h+1].expand(self.group,-1,-1).unsqueeze(0)
            outputs.append(F.scaled_dot_product_attention(qs,ks,vs,is_causal=causal).squeeze(0))
        return torch.cat(outputs,dim=0)
    @torch.inference_mode()
    def prefill(self,ids,save=None,capacity=None):
        n=len(ids);capacity=capacity or n
        if not (0<n<=capacity<=self.context):raise ValueError('native context exceeded')
        x=self.embed[torch.tensor(ids)];cache=[];times=[]
        for l,w in enumerate(self.w):
            st=time.perf_counter();a=self.rms(x,w['an'])
            q=self.rotary(F.linear(a,w['q']).reshape(n,self.heads,self.hd).transpose(0,1))
            k=self.rotary(F.linear(a,w['k']).reshape(n,self.kvheads,self.hd).transpose(0,1))
            v=F.linear(a,w['v']).reshape(n,self.kvheads,self.hd).transpose(0,1)
            projection=time.perf_counter()-st
            if save:
                np.savez_compressed(Path(save)/f'layer{l:02d}.npz',q=q[:,-1:].transpose(0,1).numpy(),k=k.transpose(0,1).numpy(),v=v.transpose(0,1).numpy(),ids=np.array(ids),model_sha256=self.sha256,query_position=n-1)
            st=time.perf_counter();o=self.attn(q,k,v,True).transpose(0,1).reshape(n,self.d);attention=time.perf_counter()-st
            st=time.perf_counter();x=x+F.linear(o,w['o']);a=self.rms(x,w['fn']);x=x+F.linear(F.silu(F.linear(a,w['gate']))*F.linear(a,w['up']),w['down']);rest=time.perf_counter()-st
            st=time.perf_counter();kc=torch.empty(self.kvheads,capacity,self.hd);vc=torch.empty_like(kc);kc[:,:n].copy_(k);vc[:,:n].copy_(v);cache.append((kc,vc));allocation=time.perf_counter()-st
            times.append(dict(projection=projection,attention=attention,rest=rest,cache_allocation=allocation))
        self.length=n
        return F.linear(self.rms(x[-1],self.gamma),self.cls),cache,times
    @torch.inference_mode()
    def step(self,token,cache):
        pos=self.length
        if pos>=cache[0][0].shape[1]:raise ValueError('cache capacity exceeded')
        x=self.embed[int(token)].unsqueeze(0);times=[]
        for l,w in enumerate(self.w):
            st=time.perf_counter();a=self.rms(x,w['an'])
            q=self.rotary(F.linear(a,w['q']).reshape(1,self.heads,self.hd).transpose(0,1),pos)
            k=self.rotary(F.linear(a,w['k']).reshape(1,self.kvheads,self.hd).transpose(0,1),pos)
            v=F.linear(a,w['v']).reshape(1,self.kvheads,self.hd).transpose(0,1)
            kc,vc=cache[l];kc[:,pos:pos+1].copy_(k);vc[:,pos:pos+1].copy_(v);projection=time.perf_counter()-st
            st=time.perf_counter();o=self.attn(q,kc[:,:pos+1],vc[:,:pos+1],False).transpose(0,1).reshape(1,self.d);attention=time.perf_counter()-st
            st=time.perf_counter();x=x+F.linear(o,w['o']);a=self.rms(x,w['fn']);x=x+F.linear(F.silu(F.linear(a,w['gate']))*F.linear(a,w['up']),w['down']);rest=time.perf_counter()-st
            times.append(dict(projection=projection,attention=attention,rest=rest))
        self.length+=1
        return F.linear(self.rms(x[0],self.gamma),self.cls),times

if __name__=='__main__':
    import sys
    torch.set_num_threads(4);m=Model(sys.argv[1]);tok=m.tokenizer
    text='<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n';ids=tok.encode(text);assert tok.decode(ids)==text
    print('ids',ids,flush=True);logits,cache,times=m.prefill(ids,capacity=len(ids)+32);out=[]
    for _ in range(32):
        nxt=int(logits.argmax());out.append(nxt);logits,_=m.step(nxt,cache)
        if nxt==2:break
    print(tok.decode(out),flush=True)
