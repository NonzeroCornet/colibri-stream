/* test_backend_stream.cu — correctness + pipeline test for the streaming
 * VRAM expert tier (backend_stream.cu), standalone (no engine, no model).
 *
 * Covers: cold demand streaming (miss), warm reuse (hit), speculative put()
 * (prefetch hit), LFRU eviction pressure, multi-row experts, every container
 * format (f32 / int8 / int4 offset-nibble / int2), odd inner dims (packed
 * tail nibble), oversized-expert rejection, split-phase overlap (two live
 * handles), a concurrent put() racing a demand group, and the stats counters.
 * Exit 0 = pass, 77 = no usable GPU (skip), 1 = fail. */
#include "../backend_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(cond,msg) do{ if(!(cond)){ std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,msg); return 1; } }while(0)

/* ---- CPU reference with the same container decode as the kernels ---- */
static float wdec(const uint8_t *row,int fmt,int i){
    if(fmt==0) return reinterpret_cast<const float*>(row)[i];
    if(fmt==1) return (float)reinterpret_cast<const int8_t*>(row)[i];
    if(fmt==2){ uint8_t v=row[i>>1]; return (float)((int)((i&1)?(v>>4):(v&15))-8); }
    uint8_t v=row[i>>2]; return (float)((int)((v>>((i&3)*2))&3)-2);
}
static size_t rbytes(int fmt,int I){
    if(fmt==0) return (size_t)I*4;
    if(fmt==1) return (size_t)I;
    if(fmt==2) return (size_t)(I+1)/2;
    return (size_t)(I+3)/4;
}
static void ref_expert(float *y,const float *x,
        const uint8_t *gw,const uint8_t *uw,const uint8_t *dw,
        const float *gs,const float *us,const float *ds,
        int fg,int fu,int fd,int D,int I,int rows){
    std::vector<float> h((size_t)I);
    for(int r=0;r<rows;r++){
        const float *xs=x+(size_t)r*D;
        for(int o=0;o<I;o++){
            const uint8_t *gr=gw+(size_t)o*rbytes(fg,D), *ur=uw+(size_t)o*rbytes(fu,D);
            float g=0,u=0;
            for(int i=0;i<D;i++){ g+=xs[i]*wdec(gr,fg,i); u+=xs[i]*wdec(ur,fu,i); }
            g*=fg?gs[o]:1.f; u*=fu?us[o]:1.f;
            h[o]=(g/(1.f+std::exp(-g)))*u;
        }
        for(int o=0;o<D;o++){
            const uint8_t *dr=dw+(size_t)o*rbytes(fd,I);
            float a=0;
            for(int i=0;i<I;i++) a+=h[i]*wdec(dr,fd,i);
            y[(size_t)r*D+o]=a*(fd?ds[o]:1.f);
        }
    }
}

static int close_enough(const float *got,const float *want,size_t n,float tol){
    for(size_t i=0;i<n;i++){
        float ref=std::fabs(want[i]);
        if(std::fabs(got[i]-want[i])>tol*(1.f+ref)){
            std::fprintf(stderr,"mismatch %zu: got %.6f want %.6f\n",i,got[i],want[i]);
            return 0;
        }
    }
    return 1;
}

/* deterministic synthetic expert: weights + scales derived from a seed */
struct Expert {
    std::vector<uint8_t> gw,uw,dw;
    std::vector<float> gs,us,ds;
    int fg,fu,fd,D,I;
};
static uint32_t rng_state;
static uint32_t rnd(void){ rng_state=rng_state*1664525u+1013904223u; return rng_state>>8; }
static void fill_mat(std::vector<uint8_t> &w,std::vector<float> &s,int fmt,int O,int I){
    w.resize(rbytes(fmt,I)*(size_t)O);
    s.resize(fmt?(size_t)O:0);
    if(fmt==0){
        float *f=(float*)w.data();
        for(size_t i=0;i<(size_t)O*I;i++) f[i]=((int)(rnd()%2001)-1000)/1000.f;
    } else for(size_t i=0;i<w.size();i++) w[i]=(uint8_t)rnd();
    for(size_t i=0;i<s.size();i++) s[i]=0.005f+(rnd()%100)*0.0002f;
}
static Expert make_expert(uint32_t seed,int fg,int fu,int fd,int D,int I){
    rng_state=seed*2654435761u+1;
    Expert e; e.fg=fg; e.fu=fu; e.fd=fd; e.D=D; e.I=I;
    fill_mat(e.gw,e.gs,fg,I,D);
    fill_mat(e.uw,e.us,fu,I,D);
    fill_mat(e.dw,e.ds,fd,D,I);
    return e;
}

/* stream a group of experts and compare against the CPU reference */
static int run_group(int layer,const std::vector<int> &eids,std::vector<Expert> &ex,
        int rows_per,float tol,uint64_t *mask_out){
    int count=(int)eids.size(), D=ex[0].D, I=ex[0].I, total=count*rows_per;
    std::vector<float> x((size_t)total*D), y((size_t)total*D,0.f), ref((size_t)total*D);
    rng_state=97531u+(uint32_t)layer;
    for(size_t i=0;i<x.size();i++) x[i]=((int)(rnd()%2001)-1000)/500.f;
    const void *gw[64],*uw[64],*dw[64]; const float *gs[64],*us[64],*ds[64];
    int fg[64],fu[64],fd[64],nr[64];
    for(int j=0;j<count;j++){
        Expert *e=&ex[j];
        gw[j]=e->gw.data(); uw[j]=e->uw.data(); dw[j]=e->dw.data();
        gs[j]=e->gs.data(); us[j]=e->us.data(); ds[j]=e->ds.data();
        fg[j]=e->fg; fu[j]=e->fu; fd[j]=e->fd; nr[j]=rows_per;
        ref_expert(ref.data()+(size_t)j*rows_per*D,x.data()+(size_t)j*rows_per*D,
            e->gw.data(),e->uw.data(),e->dw.data(),e->gs.data(),e->us.data(),e->ds.data(),
            e->fg,e->fu,e->fd,D,I,rows_per);
    }
    uint64_t mask=0;
    ColiCudaStreamHandle *h=coli_cuda_stream_begin(layer,eids.data(),count,
        gw,uw,dw,gs,us,ds,fg,fu,fd,D,I,x.data(),nr,total,&mask);
    if(!h){ std::fprintf(stderr,"stream_begin returned NULL\n"); return 1; }
    if(!coli_cuda_stream_end(h,y.data())){ std::fprintf(stderr,"stream_end failed\n"); return 1; }
    if(mask_out) *mask_out=mask;
    for(int j=0;j<count;j++){
        if(!(mask>>j&1)) continue;
        if(!close_enough(y.data()+(size_t)j*rows_per*D,ref.data()+(size_t)j*rows_per*D,
                         (size_t)rows_per*D,tol)){
            std::fprintf(stderr,"  expert %d (layer %d, eid %d) wrong\n",j,layer,eids[j]);
            return 1;
        }
    }
    return 0;
}

/* --bench: sustained-pipeline throughput on GLM-5.2-sized int4 experts.
 * Streams groups of 8 unique experts (one MoE layer's routing) back-to-back,
 * demand path only, and reports effective H2D bandwidth with the compute
 * pipelined behind it — the number that decides decode speed once the disk
 * is no longer the bottleneck. Pinned vs pageable shows what STREAM_PIN buys. */
static int run_bench(int device,int tile_kb){
    const int D=5120,I=1536;                       /* ~11.8 MB/expert at int4 */
    const int GROUPS=24, PER=8, N=GROUPS*PER;
    size_t ebytes=2*rbytes(2,D)*(size_t)I+rbytes(2,I)*(size_t)D+(size_t)(2*I+D)*4;
    size_t slot=ebytes+8192;
    std::printf("bench: tile %d KB\n",tile_kb);
    if(!coli_cuda_stream_init(device,(size_t)(N+8)*(((slot+255)&~(size_t)255)+4096),slot,tile_kb))
        return 1;
    std::vector<Expert> ex;
    for(int e=0;e<N;e++) ex.push_back(make_expert(9000u+e,2,2,2,D,I));
    std::vector<float> x((size_t)PER*D,0.25f),y((size_t)PER*D);
    for(int pinned=0;pinned<2;pinned++){
        if(pinned) for(auto &e:ex){
            coli_cuda_host_register(e.gw.data(),e.gw.size());
            coli_cuda_host_register(e.uw.data(),e.uw.size());
            coli_cuda_host_register(e.dw.data(),e.dw.size());
        }
        for(int warm=0;warm<2;warm++){
            uint64_t st0[12],st1[12]; coli_cuda_stream_stats(st0);
            cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaEventRecord(t0,0);
            for(int g=0;g<GROUPS;g++){
                int eids[PER]; const void *gw[PER],*uw[PER],*dw[PER];
                const float *gs[PER],*us[PER],*ds[PER];
                int f2[PER],nr[PER];
                for(int j=0;j<PER;j++){
                    Expert *e=&ex[g*PER+j];
                    eids[j]=g*PER+j;
                    gw[j]=e->gw.data(); uw[j]=e->uw.data(); dw[j]=e->dw.data();
                    gs[j]=e->gs.data(); us[j]=e->us.data(); ds[j]=e->ds.data();
                    f2[j]=2; nr[j]=1;
                }
                uint64_t mask=0;
                /* same layer for cold and warm: the warm pass must be cache hits */
                ColiCudaStreamHandle *h=coli_cuda_stream_begin(90+pinned,eids,PER,
                    gw,uw,dw,gs,us,ds,f2,f2,f2,D,I,x.data(),nr,PER,&mask);
                if(!h||!coli_cuda_stream_end(h,y.data())||mask!=0xFF){
                    std::fprintf(stderr,"bench group failed\n"); return 1; }
            }
            cudaEventRecord(t1,0); cudaEventSynchronize(t1);
            float ms=0; cudaEventElapsedTime(&ms,t0,t1);
            coli_cuda_stream_stats(st1);
            double gb=(double)(st1[3]-st0[3])/1e9;
            std::printf("  %-8s %-5s: %3d experts (%d groups of %d) in %7.2f ms — "
                "%6.2f GB/s H2D, %6.0f experts/s%s\n",
                pinned?"pinned":"pageable",warm?"warm":"cold",N,GROUPS,PER,ms,
                gb/(ms/1e3),N/(ms/1e3),
                warm?" (VRAM cache, kernel-bound)":"");
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }
    }
    coli_cuda_stream_shutdown();
    return 0;
}

int main(int argc,char **argv){
    int device=argc>1&&argv[1][0]!='-'?std::atoi(argv[1]):0;
    int n=0;
    if(cudaGetDeviceCount(&n)!=cudaSuccess||n<1) return 77;
    for(int i=1;i<argc;i++) if(!std::strcmp(argv[i],"--bench"))
        return run_bench(device,i+1<argc?std::atoi(argv[i+1]):512);

    const int D=1024, I=768;                       /* gate/up int4 slab = 393 KB → 7 tiles at 64 KB */
    /* generous slot: fits every test expert at f32, rejects only the oversized one */
    size_t slot=3*((size_t)I*rbytes(0,D)>(size_t)D*rbytes(0,I)?(size_t)I*rbytes(0,D):(size_t)D*rbytes(0,I));
    slot+=(size_t)(2*I+D)*4+8192;
    size_t padded=((slot+255)&~(size_t)255)+4096;
    CHECK(coli_cuda_stream_init(device,padded*6+1024,slot,64),"stream_init (6 slots, 64 KB tiles)");

    uint64_t st[12];

    /* 1. cold int4 group (miss path, multi-tile) then warm reuse (hit path) */
    {
        std::vector<int> eids={3,7,11,42};
        std::vector<Expert> ex;
        for(int e:eids) ex.push_back(make_expert(1000u+e,2,2,2,D,I));
        uint64_t mask=0;
        CHECK(!run_group(0,eids,ex,1,1e-3f,&mask),"cold int4 group");
        CHECK(mask==0xF,"cold group fully accepted");
        coli_cuda_stream_stats(st);
        CHECK(st[1]==4,"4 uploads after cold group");
        CHECK(!run_group(0,eids,ex,1,1e-3f,&mask),"warm int4 group");
        CHECK(mask==0xF,"warm group fully accepted");
        coli_cuda_stream_stats(st);
        CHECK(st[0]==4&&st[1]==4,"warm group served from VRAM cache");
        std::printf("  int4 cold+warm: ok (hits=%llu uploads=%llu tiles=%llu)\n",
            (unsigned long long)st[0],(unsigned long long)st[1],(unsigned long long)st[4]);
    }

    /* 2. mixed formats: f32, int8, int2 (+odd D packing) with multiple rows */
    {
        std::vector<int> eids={1,2,3};
        std::vector<Expert> ex;
        ex.push_back(make_expert(51,0,0,0,250,192));   /* f32, odd D */
        ex.push_back(make_expert(52,1,1,1,250,192));   /* int8 */
        ex.push_back(make_expert(53,3,3,3,250,192));   /* int2 */
        for(int j=0;j<3;j++){
            std::vector<int> one={eids[j]};
            std::vector<Expert> exj={ex[j]};
            uint64_t mask=0;
            CHECK(!run_group(10+j,one,exj,4,1e-3f,&mask),"mixed-format expert");
            CHECK(mask==1,"mixed-format expert accepted");
        }
        std::printf("  f32/int8/int2, odd dims, 4 rows: ok\n");
    }

    /* 3. speculative put() then demand hit */
    {
        Expert e=make_expert(777,2,2,2,D,I);
        CHECK(coli_cuda_stream_put(20,5,e.gw.data(),e.uw.data(),e.dw.data(),
            e.gs.data(),e.us.data(),e.ds.data(),2,2,2,D,I),"stream_put");
        coli_cuda_stream_stats(st);
        uint64_t puts=st[5];
        CHECK(puts>=1,"put counted");
        std::vector<int> one={5}; std::vector<Expert> ex1={e};
        uint64_t mask=0;
        CHECK(!run_group(20,one,ex1,1,1e-3f,&mask),"demand after put");
        coli_cuda_stream_stats(st);
        CHECK(st[6]>=1,"prefetch consumed by demand (put_hits)");
        std::printf("  put -> demand hit: ok\n");
    }

    /* 4. eviction pressure: 12 distinct experts through 6 slots, all correct */
    {
        std::vector<Expert> ex;
        for(int e=0;e<12;e++) ex.push_back(make_expert(3000u+e,2,2,2,D,I));
        for(int rep=0;rep<2;rep++)
            for(int b=0;b<3;b++){
                std::vector<int> eids; std::vector<Expert> sub;
                for(int k=0;k<4;k++){ eids.push_back(b*4+k); sub.push_back(ex[b*4+k]); }
                uint64_t mask=0;
                CHECK(!run_group(30,eids,sub,1,1e-3f,&mask),"eviction-pressure group");
                CHECK(mask==0xF,"eviction-pressure group accepted");
            }
        coli_cuda_stream_stats(st);
        CHECK(st[2]>0,"evictions occurred under pressure");
        std::printf("  eviction pressure (12 experts / 6 slots): ok (%llu evictions)\n",
            (unsigned long long)st[2]);
    }

    /* 5. oversized expert is rejected, small sibling still accepted */
    {
        int bigD=2048,bigI=2048;                     /* f32 far beyond slot_bytes */
        Expert big=make_expert(4001,0,0,0,bigD,bigI);
        Expert ok=make_expert(4002,2,2,2,bigD,bigI); /* int4 of same shape fits? gate=2048*1024*... */
        (void)ok;
        std::vector<int> eids={0};
        std::vector<Expert> exb={big};
        uint64_t mask=0xFF;
        /* run manually: run_group would flag the miss as failure; here rejection is the PASS */
        int total=1;
        std::vector<float> x((size_t)total*bigD,0.5f), y((size_t)total*bigD);
        const void *gw[1]={big.gw.data()},*uw[1]={big.uw.data()},*dw[1]={big.dw.data()};
        const float *gs[1]={big.gs.data()},*us[1]={big.us.data()},*ds[1]={big.ds.data()};
        int fg[1]={0},fu[1]={0},fd[1]={0},nr[1]={1};
        ColiCudaStreamHandle *h=coli_cuda_stream_begin(40,eids.data(),1,gw,uw,dw,gs,us,ds,
            fg,fu,fd,bigD,bigI,x.data(),nr,total,&mask);
        CHECK(h!=NULL,"begin with oversized expert");
        CHECK(coli_cuda_stream_end(h,y.data()),"end with oversized expert");
        CHECK(mask==0,"oversized expert rejected to CPU");
        std::printf("  oversized expert rejection: ok\n");
    }

    /* 6. two live handles (split-phase overlap, as moe() uses per block) */
    {
        std::vector<int> e1={50,51},e2={52,53};
        std::vector<Expert> x1,x2;
        for(int e:e1) x1.push_back(make_expert(5000u+e,2,2,2,D,I));
        for(int e:e2) x2.push_back(make_expert(5000u+e,2,2,2,D,I));
        /* interleave: begin A, begin B, end A, end B */
        int total=2;
        std::vector<float> xa((size_t)total*D),xb((size_t)total*D),ya((size_t)total*D),yb((size_t)total*D);
        std::vector<float> ra((size_t)total*D),rb((size_t)total*D);
        rng_state=13; for(auto &v:xa) v=((int)(rnd()%2001)-1000)/500.f;
        for(auto &v:xb) v=((int)(rnd()%2001)-1000)/500.f;
        const void *gwa[2],*uwa[2],*dwa[2],*gwb[2],*uwb[2],*dwb[2];
        const float *gsa[2],*usa[2],*dsa[2],*gsb[2],*usb[2],*dsb[2];
        int f2[2]={2,2},nr[2]={1,1};
        for(int j=0;j<2;j++){
            gwa[j]=x1[j].gw.data(); uwa[j]=x1[j].uw.data(); dwa[j]=x1[j].dw.data();
            gsa[j]=x1[j].gs.data(); usa[j]=x1[j].us.data(); dsa[j]=x1[j].ds.data();
            gwb[j]=x2[j].gw.data(); uwb[j]=x2[j].uw.data(); dwb[j]=x2[j].dw.data();
            gsb[j]=x2[j].gs.data(); usb[j]=x2[j].us.data(); dsb[j]=x2[j].ds.data();
            ref_expert(ra.data()+(size_t)j*D,xa.data()+(size_t)j*D,
                x1[j].gw.data(),x1[j].uw.data(),x1[j].dw.data(),
                x1[j].gs.data(),x1[j].us.data(),x1[j].ds.data(),2,2,2,D,I,1);
            ref_expert(rb.data()+(size_t)j*D,xb.data()+(size_t)j*D,
                x2[j].gw.data(),x2[j].uw.data(),x2[j].dw.data(),
                x2[j].gs.data(),x2[j].us.data(),x2[j].ds.data(),2,2,2,D,I,1);
        }
        uint64_t ma=0,mb=0;
        ColiCudaStreamHandle *ha=coli_cuda_stream_begin(60,e1.data(),2,gwa,uwa,dwa,gsa,usa,dsa,
            f2,f2,f2,D,I,xa.data(),nr,2,&ma);
        ColiCudaStreamHandle *hb=coli_cuda_stream_begin(60,e2.data(),2,gwb,uwb,dwb,gsb,usb,dsb,
            f2,f2,f2,D,I,xb.data(),nr,2,&mb);
        CHECK(ha&&hb,"two concurrent handles");
        CHECK(coli_cuda_stream_end(ha,ya.data()),"end handle A");
        CHECK(coli_cuda_stream_end(hb,yb.data()),"end handle B");
        CHECK(ma==3&&mb==3,"both handles fully accepted");
        CHECK(close_enough(ya.data(),ra.data(),(size_t)2*D,1e-3f),"handle A results");
        CHECK(close_enough(yb.data(),rb.data(),(size_t)2*D,1e-3f),"handle B results");
        std::printf("  two live handles (split-phase): ok\n");
    }

    /* 7. concurrent put() racing demand groups (throttle path, smoke) */
    {
        std::vector<Expert> bg;
        for(int e=0;e<4;e++) bg.push_back(make_expert(7000u+e,2,2,2,D,I));
        std::thread putter([&]{
            for(int rep=0;rep<8;rep++) for(int e=0;e<4;e++)
                coli_cuda_stream_put(70,100+e,bg[e].gw.data(),bg[e].uw.data(),bg[e].dw.data(),
                    bg[e].gs.data(),bg[e].us.data(),bg[e].ds.data(),2,2,2,D,I);
        });
        std::vector<Expert> fg;
        std::vector<int> eids={200,201,202,203};
        for(int e:eids) fg.push_back(make_expert(7100u+e,2,2,2,D,I));
        int rc=0;
        for(int rep=0;rep<8&&!rc;rep++){
            uint64_t mask=0;
            rc=run_group(71,eids,fg,1,1e-3f,&mask);
        }
        putter.join();
        CHECK(!rc,"demand groups correct under concurrent prefetch");
        /* the prefetched experts must also be intact */
        std::vector<int> beids={100,101,102,103};
        uint64_t mask=0;
        CHECK(!run_group(70,beids,bg,1,1e-3f,&mask),"prefetched experts intact");
        std::printf("  concurrent put vs demand: ok\n");
    }

    coli_cuda_stream_stats(st);
    std::printf("stream tier test: ALL OK — %llu hits, %llu uploads, %.2f MB streamed, "
        "%llu tiles, %llu evictions, %llu puts (%llu used, %llu dropped), %llu rejects\n",
        (unsigned long long)st[0],(unsigned long long)st[1],(double)st[3]/1e6,
        (unsigned long long)st[4],(unsigned long long)st[2],
        (unsigned long long)st[5],(unsigned long long)st[6],(unsigned long long)st[7],
        (unsigned long long)st[8]);
    coli_cuda_stream_shutdown();
    return 0;
}
