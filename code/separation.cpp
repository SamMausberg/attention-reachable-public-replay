#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
// All integer overflow preconditions are checked in audit.py before invocation.
// K and V contain floors on a 2^-10 grid. Q is floor(abs(q)*2^10).
static bool edge(const int64_t *K,const int64_t *V,const int64_t *Q,
                 int /*n*/,int d,int i,int j,int64_t kt,int64_t vt) {
    int64_t kd=0,vd=0;
    for(int c=0;c<d;++c) {
        const int64_t a=std::llabs(K[(int64_t)i*d+c]-K[(int64_t)j*d+c]);
        if(a>1) kd+=Q[c]*(a-1);
        const int64_t b=std::llabs(V[(int64_t)i*d+c]-V[(int64_t)j*d+c]);
        if(b>1) vd=std::max(vd,b-1);
        if(kd>=kt && vd>=vt) return true;
    }
    return kd>=kt && vd>=vt;
}
extern "C" int greedy(const int64_t *K,const int64_t *V,const int64_t *Q,
                       const int64_t *order,int n,int d,int64_t kt,int64_t vt,int64_t *out) {
    int count=0;
    for(int t=0;t<n;++t) {
        int i=(int)order[t];bool ok=true;
        for(int s=0;s<count;++s) if(!edge(K,V,Q,n,d,i,(int)out[s],kt,vt)){ok=false;break;}
        if(ok) out[count++]=i;
    }
    return count;
}
extern "C" int verify(const int64_t *K,const int64_t *V,const int64_t *Q,
                       const int64_t *indices,int n,int d,int64_t kt,int64_t vt,int count) {
    if(n<0 || d<=0 || count<0 || count>n || kt<=0 || vt<=0) return 0;
    std::vector<bool> seen(n,false);
    for(int i=0;i<count;++i) {int64_t a=indices[i];if(a<0||a>=n||seen[(size_t)a]) return 0;seen[(size_t)a]=true;}
    for(int i=0;i<count;++i)for(int j=0;j<i;++j)
        if(!edge(K,V,Q,n,d,(int)indices[i],(int)indices[j],kt,vt))return 0;
    return 1;
}
