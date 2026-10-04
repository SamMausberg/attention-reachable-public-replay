"""Rational readout derivative and exact finite-array packing checks.
A derivative lower bound applies to every positive-radius Euclidean enclosure
of the recorded reached state, not to an arbitrary discrete reachable set.
"""
from pathlib import Path
from fractions import Fraction as F
import numpy as np,json,math,time,heapq
from audit import mass_lower,geometry,load_lib,groups_required
ROOT=Path(__file__).resolve().parents[1]

def ff(x):return F(float(x))
def main():
 start=time.perf_counter();w=np.load(ROOT/'data/tail_weights.npz');z=np.load(ROOT/'data/norm_witness.npz')
 print(z.files,flush=True)
 x=np.load(ROOT/'data/hidden_tail_reference.npy');eps=ff(w['eps']);d=len(x)
 # Derivative of one readout row difference, evaluated at the exact dyadic x.
 b=[(ff(a)-ff(c))*ff(g) for a,c,g in zip(z['row1'],z['row2'],z['gamma'])]
 xx=[ff(t) for t in x];s=sum(t*t for t in xx);bb=sum(t*t for t in b);bx=sum(a*c for a,c in zip(b,xx));a=s/d+eps
 grad2=(a*a*bb-2*a*bx*bx/d+bx*bx*s/(d*d))/(a*a*a)
 C=F(28,100);assert grad2>=C*C
 # All weighted rows are exact products of binary32 values, hence exactly
 # representable in binary64. Round their absolute values UP on a dyadic grid.
 cls=w['cls'];gamma=w['gamma'];assert cls.dtype==gamma.dtype==np.float32;assert np.array_equal(cls[int(z['row_indices'][0])],z['row1']);assert np.array_equal(cls[int(z['row_indices'][1])],z['row2']);assert np.array_equal(gamma,z['gamma']);bits=16;Q=1<<bits
 A=cls.astype(np.float64)*gamma.astype(np.float64)[None,:]
 bound=int(np.ceil(np.abs(A).max()*Q));assert d*bound*bound<2**62
 Ai=np.ceil(np.abs(A)*Q).astype(np.int64)
 sq=(Ai*Ai).sum(axis=1);maxsq=int(sq.max());normhi=F(math.isqrt(maxsq)+1,Q)
 # radius-one enclosure excludes the origin. Certify ||x|| >= 948.
 assert s>=948**2
 radius=F(1);rms_lower2=F(947**2,d)+eps
 upper=F(math.ceil(float(2*normhi/(float(rms_lower2)**.5))*1000)+1,1000)
 assert upper*upper*rms_lower2>=4*normhi*normhi
 cols=z['columns'];proj_squares=[sum(ff(t)**2 for t in c) for c in cols[-9:]]
 assert all(t>=9 for t in proj_squares)
 # Stronger joint projection lower bound: a rational-vector Rayleigh witness.
 O=w['o'].astype(np.float64);rng=np.random.default_rng(20261003);v=rng.normal(size=O.shape[1])
 for _ in range(35):
  v=O.T@(O@v);v/=np.linalg.norm(v)
 vi=np.round(v*(1<<14)).astype(np.int64)
 # Every stored O element is a dyadic. Exact products via Fraction are modest.
 out=[sum(ff(t)*int(y) for t,y in zip(row,vi)) for row in O]
 ray=sum(t*t for t in out)/sum(int(y)**2 for y in vi)
 op=F(math.floor(math.sqrt(float(ray))*100),100)
 assert ray>=op*op
 np.savez_compressed(ROOT/'data/reachable_norm_witness.npz',x=x,row1=z['row1'],row2=z['row2'],gamma=z['gamma'],eps=w['eps'],projection=w['o'],projection_vector=vi)
 normrec={'derivative_squared_exact':str(grad2),'readout_lower':str(C),'readout_gradient_screen':math.sqrt(float(grad2)), 'hidden_squared_exact':str(s),'hidden_norm_screen':math.sqrt(float(s)),'ball_radius':str(radius),'ball_readout_upper':str(upper),'weighted_row_norm_upper':str(normhi),'all_head_projection_lower':'3','joint_projection_lower':str(op),'joint_projection_rayleigh_squared':str(ray),'norm_check_seconds':time.perf_counter()-start}
 print({k:v for k,v in normrec.items() if 'exact' not in k and 'squared' not in k},flush=True)
 (ROOT/'results/reachable_norm.json').write_text(json.dumps(normrec,indent=2))
 del A,Ai,cls,cols,O
 snap=np.load(ROOT/'data/last_fp32.npz');lib=load_lib();n=snap['k'].shape[1];curves=[];recs=[]
 for h in range(9):
  t=time.perf_counter();q=np.ascontiguousarray(snap['q'][h]);k=np.ascontiguousarray(snap['k'][h//3]);v=np.ascontiguousarray(snap['v'][h//3])
  lower,Z,sl,su,den=mass_lower(q,k);ki,vi,qi=geometry(q,k,v);order=np.argsort(su,kind='stable')[::-1].astype(np.int64).copy();indices=np.empty(n,dtype=np.int64)
  eta=F(1,2);nu=eta;c=F(1,20);kt=math.ceil(eta*8*(1<<20));vt=math.ceil(nu*(1<<10));num=lib.greedy(ki,vi,qi,order,n,64,kt,vt,indices);indices=indices[:num].copy()
  assert lib.verify(ki,vi,qi,indices,n,64,kt,vt,num)==1
  vals=sorted([lower[int(i)] for i in indices],reverse=True);tails=[sum(vals)]
  for val in vals:tails.append(tails[-1]-val)
  curve=[F(t,Z)*c for t in tails];curves.append(curve)
  tau=F(1,250)/(C*3);floor=curve[716]
  r={'head':h,'n':n,'clique_size':num,'eta':str(eta),'nu':str(nu),'cost_lower':str(c),'tau_from_local_lower':str(tau),'groups_required':groups_required(lower,Z,indices,tau,c),'B_lower_at_716_exact':str(floor),'B_lower_at_716':float(floor),'critical_readout_gain_at_716':float(F(1,250)/(3*floor)),'check_seconds':time.perf_counter()-t}
  recs.append(r);print(r,flush=True)
  np.savez_compressed(ROOT/'data'/f'local_packing_h{h}.npz',q=q,k=k,v=v,indices=indices)
 # Convex resource allocation: squared tails are decreasing discrete convex.
 # Exact heap ordering and a final discrete-dual check prove global optimality.
 functions=[[t*t for t in cv] for cv in curves];maxbudget=(9*n)//10;alloc=[0]*9
 heap=[]
 for h,fu in enumerate(functions):heapq.heappush(heap,(-(fu[0]-fu[1]),h))
 aggregate=[]
 current=sum(f[0] for f in functions)
 target2=(F(1,250)/(C*op))**2
 floor_group_budget=None
 for budget in range(0,sum(len(f)-1 for f in functions)+1):
  if budget in [maxbudget,9*717] or budget%300==0:
   aggregate.append({'groups':budget,'sum_squared_lower':float(current),'osc_lower':float(C*op)*math.sqrt(float(current))})
  if current<=target2 and floor_group_budget is None:floor_group_budget=budget
  if budget==maxbudget:
   chosen=alloc.copy();atbudget=current
   # For every coordinate, next saving <= lambda <= last chosen saving.
   nexts=[fu[alloc[h]]-fu[alloc[h]+1] for h,fu in enumerate(functions) if alloc[h]+1<len(fu)]
   lasts=[fu[alloc[h]-1]-fu[alloc[h]] for h,fu in enumerate(functions) if alloc[h]>0]
   assert max(nexts)<=min(lasts)
   dual=(max(nexts)+min(lasts))/2
  if not heap:break
  neg,h=heapq.heappop(heap);current+=neg;alloc[h]+=1
  j=alloc[h];fu=functions[h]
  if j+1<len(fu):heapq.heappush(heap,(-(fu[j]-fu[j+1]),h))
  # Need only up to the first passing budget beyond the tenfold point.
  if floor_group_budget is not None and budget>maxbudget+100 and budget>=floor_group_budget+100:break
 joint={'group_budget':maxbudget,'dense_interactions':9*n,'optimal_allocation':chosen,'squared_certificate_floor_exact':str(atbudget),'dual_lambda_exact':str(dual),'squared_allowed':str(target2),'declared_osc_floor':float(C*op)*math.sqrt(float(atbudget)),'required_groups_lower':floor_group_budget,'curve':aggregate}
 report={'norm':normrec,'heads':recs,'joint':joint,'total_check_seconds':time.perf_counter()-start}
 (ROOT/'results/reachable_packing.json').write_text(json.dumps(report,indent=2));print('JOINT',joint,flush=True)
if __name__=='__main__':main()
