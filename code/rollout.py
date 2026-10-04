"""A state-preserving, oracle-checked terminal-attention diagnostic.

All lower-layer computation and KV writes are dense. Only the final attention
read is replaced. A paired dense tail supplies the reference logits required
by the sound certificate. This is not an acceleration claim.
"""
from pathlib import Path
import argparse,time,json,hashlib,math,secrets,resource,bisect,itertools
import numpy as np
import torch
import torch.nn.functional as TF
from smollm import Model
from fastlaw import law
ROOT=Path(__file__).resolve().parents[1]

def compress(q,k,v,singles=256):
 n=k.shape[1];M=n//10;outputs=[]
 for h in range(q.shape[0]):
  K=k[h//3].double();V=v[h//3].double();Q=q[h,0].double();s=K@Q/8
  order=torch.argsort(s,descending=True,stable=True);ns=min(singles,M-1)
  tail=order[ns:];g=M-ns;width,rem=divmod(len(tail),g)
  ks=[K[order[:ns]]];vs=[V[order[:ns]]];cs=[torch.ones(ns,dtype=torch.float64)]
  at=0
  for num,w in [(rem,width+1),(g-rem,width)]:
   if num:
    idx=tail[at:at+num*w];at+=num*w
    ks.append(K[idx].reshape(num,w,64).mean(1));vs.append(V[idx].reshape(num,w,64).mean(1));cs.append(torch.full((num,),w,dtype=torch.float64))
  kc=torch.cat(ks);vc=torch.cat(vs);cnt=torch.cat(cs)
  outputs.append(((kc@Q/8+cnt.log()).softmax(0)@vc).float())
 return torch.cat(outputs).unsqueeze(0),M

class Paired(Model):
 def tail(self,x,o,w):
  x=x+TF.linear(o,w['o']);a=self.rms(x,w['fn'])
  x=x+TF.linear(TF.silu(TF.linear(a,w['gate']))*TF.linear(a,w['up']),w['down'])
  return TF.linear(self.rms(x[0],self.gamma),self.cls)
 @torch.inference_mode()
 def begin(self,ids,capacity=8192):
  n=len(ids);x=self.embed[torch.tensor(ids)];cache=[]
  for l,w in enumerate(self.w):
   a=self.rms(x,w['an']);q=self.rotary(TF.linear(a,w['q']).reshape(n,self.heads,64).transpose(0,1))
   k=self.rotary(TF.linear(a,w['k']).reshape(n,self.kvheads,64).transpose(0,1));v=TF.linear(a,w['v']).reshape(n,self.kvheads,64).transpose(0,1)
   kc=torch.empty(self.kvheads,capacity,64);vc=torch.empty_like(kc);kc[:,:n]=k;vc[:,:n]=v;cache.append((kc,vc))
   if l==29:
    self.length=n;return (x[-1:].clone(),q[:,-1:].clone(),k,v),cache
   o=self.attn(q,k,v,True).transpose(0,1).reshape(n,self.d);x=x+TF.linear(o,w['o']);a=self.rms(x,w['fn']);x=x+TF.linear(TF.silu(TF.linear(a,w['gate']))*TF.linear(a,w['up']),w['down'])
 @torch.inference_mode()
 def advance(self,token,cache):
  pos=self.length;x=self.embed[int(token)].unsqueeze(0)
  for l,w in enumerate(self.w):
   a=self.rms(x,w['an']);q=self.rotary(TF.linear(a,w['q']).reshape(1,self.heads,64).transpose(0,1),pos)
   k=self.rotary(TF.linear(a,w['k']).reshape(1,self.kvheads,64).transpose(0,1),pos);v=TF.linear(a,w['v']).reshape(1,self.kvheads,64).transpose(0,1)
   kc,vc=cache[l];kc[:,pos:pos+1]=k;vc[:,pos:pos+1]=v;k=kc[:,:pos+1];v=vc[:,:pos+1]
   if l==29:
    self.length+=1;return (x,q,k,v)
   o=self.attn(q,k,v,False).transpose(0,1).reshape(1,self.d);x=x+TF.linear(o,w['o']);a=self.rms(x,w['fn']);x=x+TF.linear(TF.silu(TF.linear(a,w['gate']))*TF.linear(a,w['up']),w['down'])
 @torch.inference_mode()
 def reference(self,state):
  x,q,k,v=state;o=self.attn(q,k,v,False).transpose(0,1).reshape(1,self.d);return self.tail(x,o,self.w[-1])
 @torch.inference_mode()
 def candidate(self,state,singles):
  x,q,k,v=state;o,M=compress(q,k,v,singles);return self.tail(x,o,self.w[-1]),M

def divergence(pz,qz):
 p=pz.astype(np.float64);q=qz.astype(np.float64);p=np.exp(p-p.max());p/=p.sum();q=np.exp(q-q.max());q/=q.sum()
 return float(np.abs(p-q).sum()/2)

def run(args):
 torch.set_num_threads(4);start=time.perf_counter();m=Paired(args.model);load=time.perf_counter()-start
 if args.prompt_ids:
  ids=np.load(args.prompt_ids,allow_pickle=False).tolist();assert len(ids)==7168
 elif args.prompt_file:
  source=Path(args.prompt_file).read_text(encoding='utf8');ids=m.tokenizer.encode(source)[:7168]
  assert len(ids)==7168
 else:ids=np.load(ROOT/'data/all_token_ids.npy')[:7168].tolist()
 t=time.perf_counter();state,cache=m.begin(ids);prefill=time.perf_counter()-t
 tag=args.tag or f'rollout_s{args.singles}_t{args.tokens}';records=[];tokens=[];words=[];ledger=0.;accepted=0
 replay=json.loads(Path(args.replay).read_text())['records'] if args.replay else None
 np.save(ROOT/'data'/(tag+'_prompt_ids.npy'),np.array(ids))
 streams=(ROOT/'results'/(tag+'.jsonl')).open('w');decode_start=time.perf_counter()
 for t in range(args.tokens):
  ts=time.perf_counter();p=m.reference(state).numpy();p=p if np.isfinite(p).all() else np.zeros_like(p);ref_seconds=time.perf_counter()-ts
  ts=time.perf_counter();q,M=m.candidate(state,args.singles);q=q.numpy();q=q if np.isfinite(q).all() else np.zeros_like(q);cand_seconds=time.perf_counter()-ts
  ts=time.perf_counter()
  try: counts,cert=law(q,p)
  except (ValueError,RuntimeError) as e:
   cert={'admit':False,'reason':str(e)};counts=None
  admit=cert['admit']
  if not admit:
   # The generic exact implementation is the defined rare numerical fallback.
   try:counts,rcert=law(p)
   except (ValueError,RuntimeError):
    from categorical import weights
    z=weights(p);counts=z['counts'];rcert={'high_precision':True}
   cert['reference_sampler']=rcert
  check_seconds=time.perf_counter()-ts
  ts=time.perf_counter();word=int(replay[t]['uniform64']) if replay else secrets.randbits(64)
  cumulative=list(itertools.accumulate(map(int,counts)))
  assert cumulative[-1]==1<<64
  tok=bisect.bisect_right(cumulative,word)
  if replay:assert tok==replay[t]['token'], 'Recorded-word replay changed the token'
  sample_seconds=time.perf_counter()-ts;tokens.append(tok);words.append(word)
  num=int(cert.get('kl_numerator',0));den=int(cert.get('kl_denominator',0));klub=num/den if den else None
  ledger+=(klub or 0) if admit else 0;accepted+=admit
  row={'step':t,'length':m.length,'groups_per_head':M,'accepted':admit,'token':tok,'uniform64':str(word),'reference_logit_sha256':hashlib.sha256(p.tobytes()).hexdigest(),'candidate_logit_sha256':hashlib.sha256(q.tobytes()).hexdigest(),'counts_min':min(map(int,counts)),'candidate_TV_screen':divergence(p,q),'kl_upper_screen':klub,'certificate':cert,'seconds':{'reference_terminal':ref_seconds,'candidate_terminal':cand_seconds,'law_and_check':check_seconds,'sample':sample_seconds}}
  if t in [0,1,2,3,31,63,127,255,511,767,1023]:
   np.savez_compressed(ROOT/'data'/(tag+f'_step{t:04d}.npz'),reference=p,candidate=q,counts=counts,**{'q':state[1][:,0].numpy()})
  streams.write(json.dumps(row)+'\n');streams.flush();records.append(row)
  if t%32==0:print(t,accepted,'TV',row['candidate_TV_screen'],'KLbound',klub,'ledger',ledger,'sec',time.perf_counter()-decode_start,flush=True)
  if t+1<args.tokens:
   ts=time.perf_counter();state=m.advance(tok,cache);row['seconds']['lower_layers_and_updates']=time.perf_counter()-ts
 decode=time.perf_counter()-decode_start;streams.close()
 np.save(ROOT/'data'/(tag+'_tokens.npy'),np.array(tokens));(ROOT/'results'/(tag+'_text.txt')).write_text(m.tokenizer.decode(tokens))
 totals={k:sum(r['seconds'].get(k,0) for r in records) for k in set(k for r in records for k in r['seconds'])}
 report={'tokens':args.tokens,'prompt_tokens':7168,'prompt_source':str(args.prompt_ids or args.prompt_file or 'earlier technical research note'),'singletons':args.singles,'accepted':accepted,'fallbacks':args.tokens-accepted,'accepted_nonzero_logit_changes':sum(r['accepted'] and r['reference_logit_sha256']!=r['candidate_logit_sha256'] for r in records),'load_seconds':load,'prefill_seconds':prefill,'decode_all_overheads_seconds':decode,'total_with_load_seconds':time.perf_counter()-start,'phase_seconds':totals,'sum_accepted_KL_bounds_screen':ledger,'candidate_tv_screen_quantiles':np.quantile([r['candidate_TV_screen'] for r in records],[0,.5,.9,.99,1]).tolist(),'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'uniform_per_token_KL_budget':'1/600000000','arithmetic':'Binary32 reference/candidate logits; 96-bit integer exponent intervals; integer moment admission; 64-bit integer sampler','reference':'Full FP32 attention; final prompt layer queried only at final prompt position; no pruning','checkpoint_sha256':m.sha256,'records':records}
 (ROOT/'results'/(tag+'.json')).write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='records'},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--tokens',type=int,default=1024);ap.add_argument('--prompt-file');ap.add_argument('--prompt-ids');ap.add_argument('--tag');ap.add_argument('--replay');ap.add_argument('--singletons',dest='singles',type=int,default=256);ap.add_argument('--model',default='/mnt/data/SmolLM2-135M-Instruct.gguf.bin');a=ap.parse_args();assert 0<a.tokens<=1024;run(a)
