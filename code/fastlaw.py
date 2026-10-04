"""Integer-checked variance certificate and bounded-error 64-bit sampler."""
import ctypes,math,json
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import numpy as np
lib=ctypes.CDLL(str(Path(__file__).with_name('law96.so')))
u=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
lib.law96.argtypes=[u,ctypes.c_int,u,u,ctypes.c_int,i,ctypes.c_int64,ctypes.c_uint64,ctypes.c_int,ctypes.c_uint64,ctypes.c_int,u,ctypes.c_char_p,ctypes.c_int]
lib.law96.restype=ctypes.c_int
@lru_cache(None)
def table(bits):
 S=1<<96;x=F(1,1<<bits);term=part=F(1)
 for j in range(1,40):
  term*=-x/j;part+=term
  if j%2:lo=part
  else:hi=part
  if j>2 and abs(term)<F(1,1<<150):
   if j%2:hi=part+term*(-x/(j+1))
   else:lo=part+term*(-x/(j+1))
   break
 a=lo.numerator*S//lo.denominator;b=-(-hi.numerator*S//hi.denominator)
 ls=[];us=[]
 for j in range(64):
  ls.extend([a&((1<<64)-1),a>>64]);us.extend([b&((1<<64)-1),b>>64])
  a=a*a//S;b=(b*b+S-1)//S
 return np.array(ls,np.uint64),np.array(us,np.uint64)

def bits_of(z):
 b=z.view(np.uint32);e=(b>>23)&255
 if np.any((e==0)&((b&0x7fffff)!=0)):raise ValueError('Subnormal logit: use high-precision fallback')
 e=e[(b&0x7fffffff)!=0]
 return max(1,150-int(e.min(initial=150)))

def law(z,reference=None,budget_den=600000000):
 z=np.ascontiguousarray(z,dtype=np.float32)
 if z.ndim!=1 or not 2<=len(z)<=65536 or not np.isfinite(z).all():raise ValueError('Finite logits required')
 B=bits_of(z)
 if B>40:raise ValueError('Rare fine-scale logit: use high-precision fallback')
 zi=np.ldexp(z.astype(np.float64),B)
 if np.max(np.abs(zi),initial=0)>2**61:raise ValueError('Grid overflow')
 zi=zi.astype(np.int64);js=(int(zi.max())-zi).astype(np.uint64);lo,hi=table(B)
 if reference is None:delta=np.zeros(len(z),np.int64);dbits=1;center=0;ran=0
 else:
  reference=np.asarray(reference,dtype=np.float32);dbits=max(B,bits_of(reference))
  if dbits>40:raise ValueError('Moment precision guard')
  # Each binary32 operand is exact in binary64, as is this bounded difference.
  dd=np.ldexp(reference.astype(np.float64)-z.astype(np.float64),dbits)
  if np.max(np.abs(dd),initial=0)>2**44:raise ValueError('Moment overflow guard')
  delta=dd.astype(np.int64);prob=np.exp(z.astype(np.float64)-float(z.max()));prob/=prob.sum()
  center=int(round(float(prob@delta.astype(np.float64))));assert abs(center)<2**45;ran=int(delta.max())-int(delta.min())
 counts=np.empty(len(z),np.uint64);buf=ctypes.create_string_buffer(2000)
 code=lib.law96(js,len(z),lo,hi,64,delta,center,ran,dbits,budget_den,int(np.argmax(z)),counts,buf,2000)
 if code:raise RuntimeError(f'Integer law error {code}')
 rec=json.loads(buf.value);rec.update(logit_grid_bits=B,delta_grid_bits=dbits,center=center,osc_numerator=ran,osc_denominator=1<<dbits)
 # Guard the exact lengths and common-scale exponent accumulation used in C++.
 assert len(z)<=65536 and max(js)<1<<63
 assert sum(map(int,counts))==1<<64
 if not rec['sampler_ok']:raise ValueError('Exponential enclosure requires high-precision fallback')
 return counts,rec
