/* backend_stream.cu — streaming GPU expert tier: VRAM as an execution cache.
 *
 * Design (docs/streaming-tier.md): instead of fitting experts into VRAM, the
 * GPU is treated as a streaming compute device over the SSD → RAM → VRAM → SM
 * hierarchy. VRAM holds a pool of fixed-size expert slots managed as a
 * score-based cache (LFRU: frequency<<8 | recency, same policy as tier.h);
 * expert weights are streamed into slots as tile-chunked async copies on
 * dedicated copy streams while the compute stream consumes experts whose
 * bytes have already landed (per-slot CUDA events: one after gate+up, one
 * after down — so gate/up matmuls start while down is still on the PCIe bus).
 *
 * Pipelining: all kernels run on one exec stream, all needed-now transfers on
 * copy_hi, background prefetch on copy_lo. While the exec stream computes
 * expert j, copy_hi is already filling expert j+1's slot — the double-buffer
 * of the design generalized to N slots. Cache hits are issued BEFORE misses so
 * resident experts never wait behind an in-flight transfer.
 *
 * PCIe scheduler: copy_lo transfers (speculative/predicted experts, pushed by
 * the router-lookahead pilot thread) are chunked into tiles and yield between
 * tiles whenever a needed-now group is active (g_active), so a prefetch burst
 * delays a demand transfer by at most one tile (~tens of µs), approximating
 * priority preemption on hardware whose copy queues are FIFO.
 *
 * Weight formats are the CPU container formats decoded in-kernel (int4 =
 * offset nibbles, value = (nibble)-8; int2 = 2-bit-2; int8 = signed bytes;
 * f32 raw) — no upload-time conversion pass, unlike the resident tier.
 *
 * Thread safety: cache metadata under one mutex; uploads run outside the lock
 * on slots reserved via busy bits (4 handle bits + 1 upload bit), so a demand
 * group (main thread) and a pilot prefetch (background thread) stream
 * concurrently on their own copy streams. coli_cuda_stream_put() returns only
 * after its bytes are on the device, so the caller's slab may be recycled
 * immediately (no host-lifetime hazard for speculative loads).
 */
#include "backend_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <thread>

#define ST_MAX_HANDLES 4
#define ST_MAX_GROUP   64

typedef struct {
    uint64_t key;                    /* (layer<<32)|eid; UINT64_MAX = empty */
    uint8_t *base;                   /* device slab (slot_bytes) */
    size_t gw, uw, dw, gs, us, ds;   /* view offsets into base */
    int fg, fu, fd, D, I;
    uint32_t freq, last;             /* LFRU score inputs (tier.h semantics) */
    int busy;                        /* bit h = active handle h; bit 4 = upload */
    int valid;                       /* weights complete and published */
    int from_put;                    /* prefetched, not yet demanded (stats) */
    cudaEvent_t ev_gu, ev_d;         /* gate+up bytes landed / down bytes landed */
} StreamSlot;

typedef struct ColiCudaStreamHandle {
    int in_use, failed, idx;
    int count, D, I, total_rows;
    int ok[ST_MAX_GROUP];
    int slot_of[ST_MAX_GROUP];
    float *d_x, *d_gate, *d_up, *d_y; size_t x_cap, gate_cap, up_cap, y_cap;
    float *h_x, *h_y; size_t hx_cap, hy_cap;      /* pinned staging */
} Handle;

static struct {
    int inited, device, nslots;
    size_t slot_bytes, tile_bytes;
    StreamSlot *slots;
    cudaStream_t exec, copy_hi, copy_lo;
    std::mutex mx;
    std::atomic<int> active;                       /* live demand groups */
    uint32_t clock;
    int evict_since_decay;
    std::atomic<unsigned long long> hits, misses, evictions, bytes, tiles,
        puts, put_hits, put_drops, rejects;
    Handle handles[ST_MAX_HANDLES];
} S;

static int st_ok(cudaError_t err, const char *what){
    if(err==cudaSuccess) return 1;
    std::fprintf(stderr,"[STREAM] %s: %s\n",what,cudaGetErrorString(err));
    return 0;
}

__host__ __device__ static size_t st_row_bytes(int fmt,int I){
    if(fmt==0) return (size_t)I*sizeof(float);
    if(fmt==1) return (size_t)I;
    if(fmt==2) return (size_t)(I+1)/2;
    if(fmt==3) return (size_t)(I+3)/4;
    return 0;
}

/* CPU container decode (NOT the resident tier's converted form): int4 slabs
 * store offset nibbles low-first, int2 stores (2-bit)-2, int8 signed bytes. */
__device__ static float st_w(const uint8_t *row,int fmt,int i){
    if(fmt==0) return reinterpret_cast<const float*>(row)[i];
    if(fmt==1) return (float)reinterpret_cast<const int8_t*>(row)[i];
    if(fmt==2){ uint8_t v=row[i>>1]; return (float)((int)((i&1)?(v>>4):(v&15))-8); }
    uint8_t v=row[i>>2]; return (float)((int)((v>>((i&3)*2))&3)-2);
}

/* gate/up dual projection for rows [row0,row0+R): grid (I,R), block 256. */
__global__ static void st_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int fg,int fu,int D,int I,int row0){
    int o=blockIdx.x, s=blockIdx.y;
    const float *xs=x+(size_t)(row0+s)*D;
    const uint8_t *gr=gw+(size_t)o*st_row_bytes(fg,D);
    const uint8_t *ur=uw+(size_t)o*st_row_bytes(fu,D);
    float ga=0,ua=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x){ float xv=xs[i]; ga+=xv*st_w(gr,fg,i); ua+=xv*st_w(ur,fu,i); }
    __shared__ float gp[256],upv[256];
    gp[threadIdx.x]=ga; upv[threadIdx.x]=ua; __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){
        if(threadIdx.x<n){ gp[threadIdx.x]+=gp[threadIdx.x+n]; upv[threadIdx.x]+=upv[threadIdx.x+n]; }
        __syncthreads();
    }
    if(!threadIdx.x){ size_t z=(size_t)(row0+s)*I+o;
        gate[z]=gp[0]*(fg?gs[o]:1.f); up[z]=upv[0]*(fu?us[o]:1.f); }
}

/* int4 fast path: consume each packed byte once (offset-nibble decode). */
__global__ static void st_gate_up_w4(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int D,int I,int row0){
    int o=blockIdx.x, s=blockIdx.y;
    const float *xs=x+(size_t)(row0+s)*D;
    size_t rb=(size_t)(D+1)/2;
    const uint8_t *gr=gw+(size_t)o*rb, *ur=uw+(size_t)o*rb;
    float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){
        uint8_t gv=gr[b], uv=ur[b]; int i=b*2;
        float x0=xs[i], x1=(i+1<D)?xs[i+1]:0.f;
        ga+=x0*(float)((int)(gv&15)-8)+x1*(float)((int)(gv>>4)-8);
        ua+=x0*(float)((int)(uv&15)-8)+x1*(float)((int)(uv>>4)-8);
    }
    __shared__ float gp[256],upv[256];
    gp[threadIdx.x]=ga; upv[threadIdx.x]=ua; __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){
        if(threadIdx.x<n){ gp[threadIdx.x]+=gp[threadIdx.x+n]; upv[threadIdx.x]+=upv[threadIdx.x+n]; }
        __syncthreads();
    }
    if(!threadIdx.x){ size_t z=(size_t)(row0+s)*I+o;
        gate[z]=gp[0]*gs[o]; up[z]=upv[0]*us[o]; }
}

__global__ static void st_silu_mul(float *gate,const float *up,size_t off,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ float v=gate[off+i]; gate[off+i]=(v/(1.f+expf(-v)))*up[off+i]; }
}

__global__ static void st_down(float *y,const float *act,
        const uint8_t *dw,const float *ds,int fd,int I,int D,int row0){
    int o=blockIdx.x, s=blockIdx.y;
    const float *as=act+(size_t)(row0+s)*I;
    const uint8_t *dr=dw+(size_t)o*st_row_bytes(fd,I);
    float a=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x) a+=as[i]*st_w(dr,fd,i);
    __shared__ float p[256];
    p[threadIdx.x]=a; __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){
        if(threadIdx.x<n) p[threadIdx.x]+=p[threadIdx.x+n];
        __syncthreads();
    }
    if(!threadIdx.x) y[(size_t)(row0+s)*D+o]=p[0]*(fd?ds[o]:1.f);
}

__global__ static void st_down_w4(float *y,const float *act,
        const uint8_t *dw,const float *ds,int I,int D,int row0){
    int o=blockIdx.x, s=blockIdx.y;
    const float *as=act+(size_t)(row0+s)*I;
    const uint8_t *dr=dw+(size_t)o*((size_t)(I+1)/2);
    float a=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){
        uint8_t v=dr[b]; int i=b*2;
        a+=as[i]*(float)((int)(v&15)-8);
        if(i+1<I) a+=as[i+1]*(float)((int)(v>>4)-8);
    }
    __shared__ float p[256];
    p[threadIdx.x]=a; __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){
        if(threadIdx.x<n) p[threadIdx.x]+=p[threadIdx.x+n];
        __syncthreads();
    }
    if(!threadIdx.x) y[(size_t)(row0+s)*D+o]=p[0]*ds[o];
}

/* ---- cache policy: recency-primary, frequency tiebreak.
 * The RAM hot-store (tier.h) is frequency-primary because it holds the
 * long-term hot set; the STREAMING cache's dominant signal is temporal
 * locality — the demand stream itself. A frequency-primary score here
 * self-evicts fresh entrants whenever the pool is full of stale
 * higher-frequency slots (LFU pollution: a full working-set turnover then
 * re-streams forever, measured 24 GB/s of avoidable PCIe in the bench).
 * Frequency still breaks ties within an age bucket, and it is halved every
 * nslots evictions so conversation heat decays instead of fossilizing. */
static uint64_t st_score(const StreamSlot *sl,uint32_t clock){
    uint32_t age=clock-sl->last, recent=age<0xFFFF?0xFFFF-age:0;
    uint32_t freq=sl->freq<0xFFFF?sl->freq:0xFFFF;
    return ((uint64_t)recent<<16)|freq;
}

/* caller holds S.mx */
static StreamSlot *st_lookup(uint64_t key){
    for(int i=0;i<S.nslots;i++)
        if(S.slots[i].key==key) return &S.slots[i];
    return NULL;
}

/* caller holds S.mx; returns a reserved (busy|16, invalid) slot or NULL */
static StreamSlot *st_acquire(uint64_t key){
    StreamSlot *best=NULL;
    for(int i=0;i<S.nslots;i++){
        StreamSlot *sl=&S.slots[i];
        if(sl->busy) continue;
        if(sl->key==UINT64_MAX){ best=sl; break; }        /* free slot wins */
        if(!best||st_score(sl,S.clock)<st_score(best,S.clock)) best=sl;
    }
    if(!best) return NULL;
    if(best->key!=UINT64_MAX){
        S.evictions.fetch_add(1,std::memory_order_relaxed);
        /* decay: every nslots evictions halve all frequencies (tier_decay
         * semantics) so stale conversation heat ages out of the tiebreak */
        if(++S.evict_since_decay>=S.nslots){
            S.evict_since_decay=0;
            for(int i=0;i<S.nslots;i++) S.slots[i].freq>>=1;
        }
    }
    best->key=key; best->valid=0; best->from_put=0; best->busy|=16;
    return best;
}

static size_t st_align(size_t n){ return (n+255)&~(size_t)255; }

/* Tile-chunked async H2D. throttle=1 (prefetch lane): yield between tiles
 * while a demand group is active, so speculative traffic never delays a
 * needed-now transfer by more than one tile. */
static int st_copy_tiles(uint8_t *dst,const void *src,size_t n,cudaStream_t cs,int throttle){
    const uint8_t *s=(const uint8_t*)src;
    size_t off=0;
    while(off<n){
        size_t chunk=n-off<S.tile_bytes?n-off:S.tile_bytes;
        if(throttle) while(S.active.load(std::memory_order_relaxed)>0) std::this_thread::yield();
        if(!st_ok(cudaMemcpyAsync(dst+off,s+off,chunk,cudaMemcpyHostToDevice,cs),"tile upload")) return 0;
        S.tiles.fetch_add(1,std::memory_order_relaxed);
        off+=chunk;
    }
    S.bytes.fetch_add(n,std::memory_order_relaxed);
    return 1;
}

/* Streams one expert into a reserved slot. Layout: [gw|uw|gs|us|dw|ds], with
 * ev_gu recorded once gate+up (weights+scales) are on the bus and ev_d after
 * down — the gate/up matmuls start while down is still transferring. */
static int st_upload(StreamSlot *sl,cudaStream_t cs,int throttle,
        const void *gw,const void *uw,const void *dw,
        const float *gs,const float *us,const float *ds,
        int fg,int fu,int fd,int D,int I){
    size_t gb=st_row_bytes(fg,D)*(size_t)I, ub=st_row_bytes(fu,D)*(size_t)I;
    size_t db=st_row_bytes(fd,I)*(size_t)D;
    size_t gsb=fg?(size_t)I*4:0, usb=fu?(size_t)I*4:0, dsb=fd?(size_t)D*4:0;
    if(!gb||!ub||!db||(fg&&!gs)||(fu&&!us)||(fd&&!ds)) return 0;
    size_t o=0;
    sl->gw=o; o+=st_align(gb); sl->uw=o; o+=st_align(ub);
    sl->gs=o; o+=st_align(gsb); sl->us=o; o+=st_align(usb);
    sl->dw=o; o+=st_align(db); sl->ds=o; o+=st_align(dsb);
    if(o>S.slot_bytes) return 0;
    sl->fg=fg; sl->fu=fu; sl->fd=fd; sl->D=D; sl->I=I;
    if(!st_copy_tiles(sl->base+sl->gw,gw,gb,cs,throttle)) return 0;
    if(!st_copy_tiles(sl->base+sl->uw,uw,ub,cs,throttle)) return 0;
    if(gsb&&!st_copy_tiles(sl->base+sl->gs,gs,gsb,cs,throttle)) return 0;
    if(usb&&!st_copy_tiles(sl->base+sl->us,us,usb,cs,throttle)) return 0;
    if(!st_ok(cudaEventRecord(sl->ev_gu,cs),"gate/up event")) return 0;
    if(!st_copy_tiles(sl->base+sl->dw,dw,db,cs,throttle)) return 0;
    if(dsb&&!st_copy_tiles(sl->base+sl->ds,ds,dsb,cs,throttle)) return 0;
    if(!st_ok(cudaEventRecord(sl->ev_d,cs),"down event")) return 0;
    return 1;
}

static int st_reserve_dev(float **p,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1;
    if(*p) cudaFree(*p);
    *p=NULL; *cap=0;
    if(!st_ok(cudaMalloc(p,bytes),"stream scratch")) return 0;
    *cap=bytes; return 1;
}
static int st_reserve_pinned(float **p,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1;
    if(*p) cudaFreeHost(*p);
    *p=NULL; *cap=0;
    if(!st_ok(cudaMallocHost(p,bytes),"stream pinned staging")) return 0;
    *cap=bytes; return 1;
}

extern "C" int coli_cuda_stream_init(int device,size_t vram_bytes,size_t slot_bytes,int tile_kb){
    if(S.inited) return 0;
    if(!st_ok(cudaSetDevice(device),"stream device")) return 0;
    slot_bytes=st_align(slot_bytes)+4096;
    if(tile_kb<64) tile_kb=64;
    if(tile_kb>4096) tile_kb=4096;
    int nslots=(int)(vram_bytes/slot_bytes);
    if(nslots<2){ std::fprintf(stderr,"[STREAM] budget %.2f GB < 2 slots of %.1f MB — tier disabled\n",
        vram_bytes/1e9,slot_bytes/1e6); return 0; }
    if(nslots>8192) nslots=8192;
    S.slots=(StreamSlot*)std::calloc((size_t)nslots,sizeof(StreamSlot));
    if(!S.slots) return 0;
    int got=0;
    for(int i=0;i<nslots;i++){
        StreamSlot *sl=&S.slots[i];
        if(cudaMalloc(&sl->base,slot_bytes)!=cudaSuccess){ cudaGetLastError(); break; }
        if(cudaEventCreateWithFlags(&sl->ev_gu,cudaEventDisableTiming)!=cudaSuccess||
           cudaEventCreateWithFlags(&sl->ev_d,cudaEventDisableTiming)!=cudaSuccess){
            cudaFree(sl->base); sl->base=NULL; break; }
        sl->key=UINT64_MAX; got++;
    }
    if(got<2){
        for(int i=0;i<got;i++){ cudaFree(S.slots[i].base);
            cudaEventDestroy(S.slots[i].ev_gu); cudaEventDestroy(S.slots[i].ev_d); }
        std::free(S.slots); S.slots=NULL;
        std::fprintf(stderr,"[STREAM] VRAM slot allocation failed — tier disabled\n");
        return 0;
    }
    if(!st_ok(cudaStreamCreateWithFlags(&S.exec,cudaStreamNonBlocking),"exec stream")||
       !st_ok(cudaStreamCreateWithFlags(&S.copy_hi,cudaStreamNonBlocking),"copy stream")||
       !st_ok(cudaStreamCreateWithFlags(&S.copy_lo,cudaStreamNonBlocking),"prefetch stream")){
        return 0;
    }
    S.device=device; S.nslots=got; S.slot_bytes=slot_bytes;
    S.tile_bytes=(size_t)tile_kb*1024;
    S.inited=1;
    std::fprintf(stderr,"[STREAM] VRAM execution cache: %d slots x %.1f MB (%.2f GB), tile %d KB\n",
        got,slot_bytes/1e6,(double)got*slot_bytes/1e9,tile_kb);
    return 1;
}

extern "C" void coli_cuda_stream_shutdown(void){
    if(!S.inited) return;
    cudaSetDevice(S.device);
    cudaStreamSynchronize(S.exec);
    cudaStreamSynchronize(S.copy_hi);
    cudaStreamSynchronize(S.copy_lo);
    for(int i=0;i<S.nslots;i++){
        cudaFree(S.slots[i].base);
        cudaEventDestroy(S.slots[i].ev_gu); cudaEventDestroy(S.slots[i].ev_d);
    }
    std::free(S.slots); S.slots=NULL;
    for(int h=0;h<ST_MAX_HANDLES;h++){
        Handle *H=&S.handles[h];
        if(H->d_x)cudaFree(H->d_x); if(H->d_gate)cudaFree(H->d_gate);
        if(H->d_up)cudaFree(H->d_up); if(H->d_y)cudaFree(H->d_y);
        if(H->h_x)cudaFreeHost(H->h_x); if(H->h_y)cudaFreeHost(H->h_y);
        std::memset(H,0,sizeof(*H));
    }
    cudaStreamDestroy(S.exec); cudaStreamDestroy(S.copy_hi); cudaStreamDestroy(S.copy_lo);
    S.inited=0; S.nslots=0;
}

extern "C" int coli_cuda_host_register(void *p,size_t bytes){
    if(!p||!bytes) return 0;
    if(cudaHostRegister(p,bytes,cudaHostRegisterPortable)==cudaSuccess) return 1;
    cudaGetLastError();                          /* clear sticky error: pageable still works */
    return 0;
}
extern "C" void coli_cuda_host_unregister(void *p){
    if(!p) return;
    if(cudaHostUnregister(p)!=cudaSuccess) cudaGetLastError();
}

/* Demand path, split-phase (mirrors the Metal moe_block begin/end pattern):
 * begin() returns after every transfer and kernel is ENQUEUED — the CPU is
 * free to read the next experts from disk or run the shared expert while the
 * GPU pipeline drains. y rows are packed in call order (rejected experts'
 * rows are left unwritten; accepted_mask tells the caller which to trust). */
extern "C" ColiCudaStreamHandle *coli_cuda_stream_begin(int layer,const int *eids,int count,
        const void *const *gw,const void *const *uw,const void *const *dw,
        const float *const *gs,const float *const *us,const float *const *ds,
        const int *fg,const int *fu,const int *fd,
        int D,int I,const float *x,const int *nrows,int total_rows,
        uint64_t *accepted_mask){
    if(accepted_mask) *accepted_mask=0;
    if(!S.inited||count<1||count>ST_MAX_GROUP||total_rows<1||D<1||I<1||!x||!nrows) return NULL;
    if(!st_ok(cudaSetDevice(S.device),"stream device")) return NULL;
    Handle *h=NULL;
    {
        std::lock_guard<std::mutex> lk(S.mx);
        for(int i=0;i<ST_MAX_HANDLES;i++) if(!S.handles[i].in_use){
            h=&S.handles[i]; h->in_use=1; h->idx=i; break; }
        if(h) S.active.fetch_add(1,std::memory_order_relaxed);
    }
    if(!h) return NULL;
    h->failed=0; h->count=count; h->D=D; h->I=I; h->total_rows=total_rows;
    size_t xb=(size_t)total_rows*D*sizeof(float), ib=(size_t)total_rows*I*sizeof(float);
    if(!st_reserve_dev(&h->d_x,&h->x_cap,xb)||!st_reserve_dev(&h->d_y,&h->y_cap,xb)||
       !st_reserve_dev(&h->d_gate,&h->gate_cap,ib)||!st_reserve_dev(&h->d_up,&h->up_cap,ib)){
        h->failed=1;
    }
    if(!h->failed&&(!st_reserve_pinned(&h->h_x,&h->hx_cap,xb)||!st_reserve_pinned(&h->h_y,&h->hy_cap,xb)))
        h->failed=1;
    if(!h->failed){
        std::memcpy(h->h_x,x,xb);
        if(!st_ok(cudaMemcpyAsync(h->d_x,h->h_x,xb,cudaMemcpyHostToDevice,S.exec),"x upload"))
            h->failed=1;
    }
    /* resolve every expert first (hits issued before misses: a resident expert
     * must never queue its kernels behind another expert's PCIe transfer) */
    int row0[ST_MAX_GROUP], hit[ST_MAX_GROUP];
    {
        int r=0;
        for(int j=0;j<count;j++){ row0[j]=r; r+=nrows[j]; h->ok[j]=0; h->slot_of[j]=-1; hit[j]=0; }
    }
    if(!h->failed){
        std::lock_guard<std::mutex> lk(S.mx);
        for(int j=0;j<count;j++){
            if(nrows[j]<1) continue;
            uint64_t key=((uint64_t)(uint32_t)layer<<32)|(uint32_t)eids[j];
            StreamSlot *sl=st_lookup(key);
            if(sl&&sl->valid&&sl->D==D&&sl->I==I){
                sl->freq++; sl->last=++S.clock; sl->busy|=(1<<h->idx);
                if(sl->from_put){ sl->from_put=0; S.put_hits.fetch_add(1,std::memory_order_relaxed); }
                h->slot_of[j]=(int)(sl-S.slots); hit[j]=1;
                S.hits.fetch_add(1,std::memory_order_relaxed);
            }
        }
    }
    /* pass 1: cache hits — kernels only, no transfer dependency */
    for(int pass=0;pass<2&&!h->failed;pass++){
        for(int j=0;j<count&&!h->failed;j++){
            if(nrows[j]<1||h->ok[j]) continue;
            if(pass==0&&!hit[j]) continue;
            if(pass==1&&hit[j]) continue;
            StreamSlot *sl=NULL;
            if(pass==1){
                /* NULL slabs = the caller believed this expert was VRAM-resident
                 * (stream_query) and skipped its disk read, but the slot was
                 * evicted in between: reject to the CPU-fallback path, never
                 * upload from a NULL/stale pointer. */
                if(!gw[j]||!uw[j]||!dw[j]){ S.rejects.fetch_add(1,std::memory_order_relaxed); continue; }
                uint64_t key=((uint64_t)(uint32_t)layer<<32)|(uint32_t)eids[j];
                {
                    std::lock_guard<std::mutex> lk(S.mx);
                    sl=st_acquire(key);
                    if(sl) sl->busy|=(1<<h->idx);
                }
                if(!sl){ S.rejects.fetch_add(1,std::memory_order_relaxed); continue; }
                if(!st_upload(sl,S.copy_hi,0,gw[j],uw[j],dw[j],gs[j],us[j],ds[j],
                              fg[j],fu[j],fd[j],D,I)){
                    /* drain any tiles already in flight before the slot can be
                     * reacquired: a late stale tile must never overwrite a
                     * successor's upload into the same device slab */
                    cudaStreamSynchronize(S.copy_hi);
                    std::lock_guard<std::mutex> lk(S.mx);
                    sl->key=UINT64_MAX; sl->valid=0; sl->busy&=~(1<<h->idx)&~16;
                    S.rejects.fetch_add(1,std::memory_order_relaxed);
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lk(S.mx);
                    sl->valid=1; sl->busy&=~16; sl->freq=1; sl->last=++S.clock;
                }
                h->slot_of[j]=(int)(sl-S.slots);
                S.misses.fetch_add(1,std::memory_order_relaxed);
            } else {
                sl=&S.slots[h->slot_of[j]];
            }
            int R=nrows[j];
            if(!st_ok(cudaStreamWaitEvent(S.exec,sl->ev_gu,0),"gate/up wait")){ h->failed=1; break; }
            if(sl->fg==2&&sl->fu==2)
                st_gate_up_w4<<<dim3((unsigned)I,(unsigned)R),256,0,S.exec>>>(
                    h->d_gate,h->d_up,h->d_x,sl->base+sl->gw,sl->base+sl->uw,
                    (const float*)(sl->base+sl->gs),(const float*)(sl->base+sl->us),D,I,row0[j]);
            else
                st_gate_up<<<dim3((unsigned)I,(unsigned)R),256,0,S.exec>>>(
                    h->d_gate,h->d_up,h->d_x,sl->base+sl->gw,sl->base+sl->uw,
                    (const float*)(sl->base+sl->gs),(const float*)(sl->base+sl->us),
                    sl->fg,sl->fu,D,I,row0[j]);
            size_t n=(size_t)R*I, off=(size_t)row0[j]*I;
            st_silu_mul<<<(unsigned)((n+255)/256),256,0,S.exec>>>(h->d_gate,h->d_up,off,n);
            if(!st_ok(cudaStreamWaitEvent(S.exec,sl->ev_d,0),"down wait")){ h->failed=1; break; }
            if(sl->fd==2)
                st_down_w4<<<dim3((unsigned)D,(unsigned)R),256,0,S.exec>>>(
                    h->d_y,h->d_gate,sl->base+sl->dw,(const float*)(sl->base+sl->ds),I,D,row0[j]);
            else
                st_down<<<dim3((unsigned)D,(unsigned)R),256,0,S.exec>>>(
                    h->d_y,h->d_gate,sl->base+sl->dw,(const float*)(sl->base+sl->ds),
                    sl->fd,I,D,row0[j]);
            if(!st_ok(cudaGetLastError(),"stream kernel launch")){ h->failed=1; break; }
            h->ok[j]=1;
        }
    }
    if(accepted_mask&&!h->failed){
        uint64_t m=0;
        for(int j=0;j<count;j++) if(h->ok[j]) m|=1ULL<<j;
        *accepted_mask=m;
    }
    if(h->failed){
        /* leave slot metadata consistent; caller must still call end() */
        if(accepted_mask) *accepted_mask=0;
        for(int j=0;j<count;j++) h->ok[j]=0;
    }
    return h;
}

extern "C" int coli_cuda_stream_end(ColiCudaStreamHandle *h,float *y){
    if(!h||!h->in_use) return 0;
    cudaSetDevice(S.device);
    int ok=!h->failed;
    if(ok){
        size_t yb=(size_t)h->total_rows*h->D*sizeof(float);
        if(!st_ok(cudaMemcpyAsync(h->h_y,h->d_y,yb,cudaMemcpyDeviceToHost,S.exec),"y download")||
           !st_ok(cudaStreamSynchronize(S.exec),"stream synchronize")) ok=0;
        if(ok&&y) std::memcpy(y,h->h_y,yb);
    } else {
        cudaStreamSynchronize(S.exec);           /* drain partial work before releasing slots */
    }
    {
        std::lock_guard<std::mutex> lk(S.mx);
        for(int j=0;j<h->count;j++) if(h->slot_of[j]>=0)
            S.slots[h->slot_of[j]].busy&=~(1<<h->idx);
        h->in_use=0;
        S.active.fetch_sub(1,std::memory_order_relaxed);
    }
    return ok;
}

/* Residency probe: 1 if (layer,eid) is VRAM-resident right now. Lets the
 * engine skip the DISK read for experts the execution cache already holds —
 * VRAM acts as a true cache tier above RAM, not just above compute. The
 * answer can go stale (a concurrent prefetch may evict the slot before the
 * demand group arrives); begin() rejects a miss with NULL slabs instead of
 * uploading garbage, and the caller falls back to a disk load. */
extern "C" int coli_cuda_stream_query(int layer,int eid){
    if(!S.inited) return 0;
    uint64_t key=((uint64_t)(uint32_t)layer<<32)|(uint32_t)eid;
    std::lock_guard<std::mutex> lk(S.mx);
    StreamSlot *sl=st_lookup(key);
    return sl&&sl->valid;
}

/* Speculative path (router-lookahead pilot / heat prefetch): synchronous —
 * returns once the bytes are on the device, so the caller's slab can be
 * recycled immediately. Runs on the low-priority copy stream and yields to
 * demand groups between tiles. */
extern "C" int coli_cuda_stream_put(int layer,int eid,
        const void *gw,const void *uw,const void *dw,
        const float *gs,const float *us,const float *ds,
        int fg,int fu,int fd,int D,int I){
    if(!S.inited) return 0;
    if(!st_ok(cudaSetDevice(S.device),"stream device")) return 0;
    uint64_t key=((uint64_t)(uint32_t)layer<<32)|(uint32_t)eid;
    StreamSlot *sl=NULL;
    {
        std::lock_guard<std::mutex> lk(S.mx);
        StreamSlot *ex=st_lookup(key);
        if(ex&&ex->valid){ ex->freq++; ex->last=++S.clock; return 1; }
        if(ex) return 0;                          /* upload already in flight */
        sl=st_acquire(key);
        if(!sl){ S.put_drops.fetch_add(1,std::memory_order_relaxed); return 0; }
    }
    if(!st_upload(sl,S.copy_lo,1,gw,uw,dw,gs,us,ds,fg,fu,fd,D,I)||
       !st_ok(cudaEventSynchronize(sl->ev_d),"prefetch synchronize")){
        cudaStreamSynchronize(S.copy_lo);        /* drain stale tiles before releasing */
        std::lock_guard<std::mutex> lk(S.mx);
        sl->key=UINT64_MAX; sl->valid=0; sl->busy&=~16;
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(S.mx);
        sl->valid=1; sl->busy&=~16; sl->freq=1; sl->last=++S.clock; sl->from_put=1;
    }
    S.puts.fetch_add(1,std::memory_order_relaxed);
    return 1;
}

/* st[0]=hits st[1]=misses st[2]=evictions st[3]=bytes_streamed st[4]=tiles
 * st[5]=prefetch_puts st[6]=prefetch_hits st[7]=prefetch_drops st[8]=rejects
 * st[9]=nslots st[10]=slot_bytes st[11]=reserved */
extern "C" void coli_cuda_stream_stats(uint64_t st[12]){
    if(!st) return;
    st[0]=S.hits.load(); st[1]=S.misses.load(); st[2]=S.evictions.load();
    st[3]=S.bytes.load(); st[4]=S.tiles.load(); st[5]=S.puts.load();
    st[6]=S.put_hits.load(); st[7]=S.put_drops.load(); st[8]=S.rejects.load();
    st[9]=(uint64_t)S.nslots; st[10]=(uint64_t)S.slot_bytes; st[11]=0;
}
