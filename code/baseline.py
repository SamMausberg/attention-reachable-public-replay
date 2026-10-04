"""Matched dense reference, with the same certified integer sampler.
No centroid construction or candidate branch is executed.
"""
from pathlib import Path
import json,time,bisect,itertools,resource,argparse
import numpy as np
import torch
from rollout import Paired,ROOT
from fastlaw import law
from categorical import weights

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trace',default='technical_replay');ap.add_argument('--model',default='/mnt/data/SmolLM2-135M-Instruct.gguf.bin');a=ap.parse_args()
 torch.set_num_threads(4);trace=json.loads((ROOT/'results'/(a.trace+'.json')).read_text());ids=np.load(ROOT/'data'/(a.trace+'_prompt_ids.npy')).tolist();start=time.perf_counter();m=Paired(a.model);load=time.perf_counter()-start
 t=time.perf_counter();state,cache=m.begin(ids);prefill=time.perf_counter()-t;records=[];tokens=[];s=time.perf_counter()
 for j,r in enumerate(trace['records']):
  t=time.perf_counter();z=m.reference(state).numpy();z=z if np.isfinite(z).all() else np.zeros_like(z);forward=time.perf_counter()-t
  t=time.perf_counter()
  try:c,_=law(z)
  except (ValueError,RuntimeError):c=weights(z)['counts']
  sampler=time.perf_counter()-t;word=int(r['uniform64']);cdf=list(itertools.accumulate(map(int,c)));tok=bisect.bisect_right(cdf,word);tokens.append(tok)
  rec={'step':j,'reference_terminal_seconds':forward,'law_seconds':sampler,'same_token_as_checked':tok==r['token']};records.append(rec)
  if j+1<len(trace['records']):
   t=time.perf_counter();state=m.advance(tok,cache);rec['lower_layers_and_updates_seconds']=time.perf_counter()-t
  if j%128==0:print(j,time.perf_counter()-s,flush=True)
 decode=time.perf_counter()-s
 result={'trace':a.trace,'tokens':len(tokens),'same_tokens':sum(r['same_token_as_checked'] for r in records),'first_different_token':next((r['step'] for r in records if not r['same_token_as_checked']),None),'load_seconds':load,'prefill_seconds':prefill,'decode_seconds':decode,'total_with_load_seconds':time.perf_counter()-start,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'records':records}
 (ROOT/'results'/('baseline_'+a.trace+'.json')).write_text(json.dumps(result,indent=2));np.save(ROOT/'data'/('baseline_'+a.trace+'_tokens.npy'),np.array(tokens));print({k:v for k,v in result.items() if k!='records'},flush=True)
if __name__=='__main__':main()
