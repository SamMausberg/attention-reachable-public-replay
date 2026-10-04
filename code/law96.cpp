// Fixed-point exponential intervals and an exactly specified integer sampler.
// All arithmetic below is unsigned integer arithmetic; no fast-math is used.
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <exception>
using boost::multiprecision::uint256_t;
using boost::multiprecision::uint128_t;
extern "C" int law96(const uint64_t* js, int n, const uint64_t* plo,
 const uint64_t* phi, int plen, const int64_t* delta, int64_t center,
 uint64_t range, int dbits, uint64_t budget_den, int pivot,
 uint64_t* counts, char* buf, int cap) {
 try {
  if(n<2 || plen<1 || plen>64 || dbits<0 || dbits>45 || pivot<0 || pivot>=n) return -1;
  const uint256_t S=uint256_t(1)<<96, mask=S-1, D=uint256_t(1)<<64;
  std::vector<uint256_t> L(n), U(n);uint256_t zl=0,zu=0,ss=0;
  for(int i=0;i<n;i++) {
   uint256_t lo=S,hi=S;uint64_t j=js[i];int b=0;
   while(j) {
    if(b>=plen)return -2;
    if(j&1) {
     uint256_t a=uint256_t(plo[2*b+1])<<64; a+=plo[2*b];
     uint256_t c=uint256_t(phi[2*b+1])<<64; c+=phi[2*b];
     lo=(lo*a)>>96;hi=(hi*c+mask)>>96;
    }
    ++b;j>>=1;
   }
   L[i]=lo;U[i]=hi;zl+=lo;zu+=hi;
   // Signed subtraction is guarded by the Python caller.
   int64_t dt=delta[i]-center;uint256_t ad=dt<0?uint64_t(-dt):uint64_t(dt);
   ss+=hi*ad*ad;
  }
  if(zl==0 || zl>zu)return -3;
  uint256_t sum=0;
  for(int i=0;i<n;i++) if(i!=pivot) {
   uint256_t c=(L[i]<<64)/zl;if(c>=D)return -4;
   counts[i]=c.convert_to<uint64_t>();sum+=c;
  }
  if(sum==0 || sum>D)return -5;
  counts[pivot]=(D-sum).convert_to<uint64_t>();
  uint256_t unit=uint256_t(1)<<dbits;
  bool small=uint256_t(range)<unit;
  uint256_t den=small ? 2*zl*unit*(unit-range):uint256_t(0);
  bool admit=small && ss*budget_den<=den;
  bool sampler_ok=(zu-zl)*1000000000000ULL<=zl;
  std::string text="{\"ZL\":\""+zl.str()+"\",\"ZU\":\""+zu.str()+
   "\",\"kl_numerator\":\""+ss.str()+"\",\"kl_denominator\":\""+den.str()+
   "\",\"admit\":"+(admit?"true":"false")+",\"sampler_ok\":"+(sampler_ok?"true":"false")+"}";
  if((int)text.size()+1>cap)return -6;
  std::memcpy(buf,text.c_str(),text.size()+1);return 0;
 }catch(const std::exception&){return -7;}
}
