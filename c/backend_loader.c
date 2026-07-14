/* backend_loader.c — Windows runtime loader for coli_cuda.dll.
 *
 * Why this exists: the engine is built with MinGW-w64 (gcc), but CUDA kernels
 * must be compiled with MSVC + nvcc. We cannot link a CUDA .o into a gcc binary
 * reliably across the MSVC/MinGW ABI, and nvcc requires cl.exe as its host
 * compiler. The clean cross-toolchain split is: build the CUDA backend into a
 * standalone coli_cuda.dll with nvcc+MSVC, then load it here at runtime via
 * LoadLibrary/GetProcAddress. The host (glm.exe) never links cudart directly.
 *
 * On Linux this file is not compiled (the Makefile links backend_cuda.o
 * directly). On Windows, when COLI_CUDA is defined, glm.c calls the
 * coli_cuda_* wrappers below, which forward through function pointers resolved
 * from the DLL at first use. If the DLL is absent, every call safely returns
 * the "not initialized" sentinel (0 / no-op) and the engine falls back to CPU.
 *
 * ABI note: ColiCudaTensor* is opaque to the host (it stores the pointer,
 * never dereferences it), so the MSVC-allocated struct is safe to pass across
 * the boundary as an opaque handle. All scalar types (int, size_t, pointers)
 * agree between MSVC and MinGW-w64 on x86-64.
 */
#ifdef _WIN32

#include <stdio.h>
#include <stddef.h>
#include <windows.h>

#include "backend_cuda.h"

/* Function-pointer typedefs matching each exported symbol. */
typedef int            (*fn_init)(const int *devices, int count);
typedef void           (*fn_shutdown)(void);
typedef int            (*fn_device_count)(void);
typedef int            (*fn_device_at)(int index);
typedef int            (*fn_mem_info)(int device, size_t *free_bytes, size_t *total_bytes);
typedef void           (*fn_stats)(int device, size_t *tensor_count, size_t *tensor_bytes);
typedef void           (*fn_group_stats)(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                         double *h2d_ms, double *kernel_ms, double *d2h_ms);
typedef int            (*fn_expert_mlp)(ColiCudaTensor *gate, ColiCudaTensor *up,
                                        ColiCudaTensor *down, float *y, const float *x, int S);
typedef int            (*fn_expert_group)(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                                          ColiCudaTensor *const *downs, const int *rows, int count,
                                          float *y, const float *x);
typedef int            (*fn_attention_absorb)(ColiCudaTensor *kv_b, float *ctx, const float *q,
                                              const float *latent, const float *rope, int H, int Q,
                                              int R, int V, int K, int T, float attention_scale);
typedef int            (*fn_tensor_upload)(ColiCudaTensor **tensor, const void *weights,
                                           const float *scales, int fmt, int I, int O, int device);
typedef int            (*fn_matmul)(ColiCudaTensor **tensor, float *y, const float *x,
                                    const void *weights, const float *scales,
                                    int fmt, int S, int I, int O, int device);
typedef void           (*fn_tensor_free)(ColiCudaTensor *tensor);
typedef size_t         (*fn_tensor_bytes)(const ColiCudaTensor *tensor);
typedef int            (*fn_tensor_device)(const ColiCudaTensor *tensor);
/* streaming expert tier (backend_stream.cu) */
typedef int            (*fn_stream_init)(int device, size_t vram_bytes, size_t slot_bytes, int tile_kb);
typedef void           (*fn_stream_shutdown)(void);
typedef int            (*fn_host_register)(void *ptr, size_t bytes);
typedef void           (*fn_host_unregister)(void *ptr);
typedef ColiCudaStreamHandle *(*fn_stream_begin)(int layer, const int *eids, int count,
                                const void *const *gw, const void *const *uw, const void *const *dw,
                                const float *const *gs, const float *const *us, const float *const *ds,
                                const int *fg, const int *fu, const int *fd,
                                int D, int I, const float *x, const int *nrows, int total_rows,
                                uint64_t *accepted_mask);
typedef int            (*fn_stream_end)(ColiCudaStreamHandle *handle, float *y);
typedef int            (*fn_stream_put)(int layer, int eid,
                                const void *gw, const void *uw, const void *dw,
                                const float *gs, const float *us, const float *ds,
                                int fg, int fu, int fd, int D, int I);
typedef int            (*fn_stream_query)(int layer, int eid);
typedef void           (*fn_stream_stats)(uint64_t st[12]);

/* Resolved pointers, plus a flag so we attempt the load at most once. */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_init            init;
    fn_shutdown        shutdown;
    fn_device_count    device_count;
    fn_device_at       device_at;
    fn_mem_info        mem_info;
    fn_stats           stats;
    fn_group_stats     group_stats;
    fn_expert_mlp      expert_mlp;
    fn_expert_group    expert_group;
    fn_attention_absorb attention_absorb;
    fn_tensor_upload   tensor_upload;
    fn_matmul          matmul;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;
    /* streaming tier: resolved OPTIONALLY so an older coli_cuda.dll still
     * provides the resident tier (stream_available gates the new calls). */
    int stream_available;
    fn_stream_init     stream_init;
    fn_stream_shutdown stream_shutdown;
    fn_host_register   host_register;
    fn_host_unregister host_unregister;
    fn_stream_begin    stream_begin;
    fn_stream_end      stream_end;
    fn_stream_put      stream_put;
    fn_stream_query    stream_query;
    fn_stream_stats    stream_stats;
} g_cuda;

/* Resolve the DLL and all 11 symbols. Returns 1 on success, 0 otherwise.
 * Idempotent: the first call (success or fail) sticks; later calls are no-ops
 * that return the cached result. The engine treats a 0 return as "CUDA
 * unavailable" and falls back to the CPU path without aborting. */
static int coli_cuda_load(void){
    if(g_cuda.loaded) return g_cuda.available;
    g_cuda.loaded = 1;

    /* Search the model directory first (so a DLL shipped next to the model
     * wins), then the engine directory, then the default DLL search path. */
    g_cuda.dll = LoadLibraryA("coli_cuda.dll");
    if(!g_cuda.dll){
        fprintf(stderr, "[CUDA] coli_cuda.dll not found; GPU tier disabled "
                        "(CPU path remains active).\n");
        return 0;
    }

    #define RESOLVE(name, type) \
        /* GetProcAddress returns FARPROC (void(*)(void)); casting it to a   \
         * specific function-pointer type is the standard LoadLibrary idiom. \
         * -Wcast-function-type flags it but it is safe: the DLL exported     \
         * the symbol with extern "C" and the exact signature we expect. */   \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_cuda.name){ \
            fprintf(stderr, "[CUDA] coli_cuda.dll missing symbol coli_cuda_" #name "\n"); \
            FreeLibrary(g_cuda.dll); g_cuda.dll=NULL; return 0; }

    RESOLVE(init,           fn_init)
    RESOLVE(shutdown,       fn_shutdown)
    RESOLVE(device_count,   fn_device_count)
    RESOLVE(device_at,      fn_device_at)
    RESOLVE(mem_info,       fn_mem_info)
    RESOLVE(stats,          fn_stats)
    RESOLVE(group_stats,    fn_group_stats)
    RESOLVE(expert_mlp,     fn_expert_mlp)
    RESOLVE(expert_group,   fn_expert_group)
    RESOLVE(attention_absorb, fn_attention_absorb)
    RESOLVE(tensor_upload,  fn_tensor_upload)
    RESOLVE(matmul,         fn_matmul)
    RESOLVE(tensor_free,    fn_tensor_free)
    RESOLVE(tensor_bytes,   fn_tensor_bytes)
    RESOLVE(tensor_device,  fn_tensor_device)
    #undef RESOLVE

    /* Streaming-tier symbols are optional (older DLLs lack them): resolve all
     * or none, and only disable the STREAM feature — never the whole backend. */
    #define RESOLVE_OPT(name, type) \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_cuda.name) stream_ok = 0;
    {
        int stream_ok = 1;
        RESOLVE_OPT(stream_init,     fn_stream_init)
        RESOLVE_OPT(stream_shutdown, fn_stream_shutdown)
        RESOLVE_OPT(host_register,   fn_host_register)
        RESOLVE_OPT(host_unregister, fn_host_unregister)
        RESOLVE_OPT(stream_begin,    fn_stream_begin)
        RESOLVE_OPT(stream_end,      fn_stream_end)
        RESOLVE_OPT(stream_put,      fn_stream_put)
        RESOLVE_OPT(stream_query,    fn_stream_query)
        RESOLVE_OPT(stream_stats,    fn_stream_stats)
        g_cuda.stream_available = stream_ok;
        if(!stream_ok)
            fprintf(stderr, "[CUDA] coli_cuda.dll predates the streaming tier; "
                            "COLI_STREAM will be unavailable.\n");
    }
    #undef RESOLVE_OPT

    g_cuda.available = 1;
    return 1;
}

/* ---- Public wrappers: match backend_cuda.h signatures exactly.
 * Each forwards to the resolved pointer; if the DLL never loaded, return the
 * "not initialized" result the engine already handles (init returns 0, matmul
 * returns 0 so the caller marks the tensor cuda_failed and uses CPU). ---- */

int coli_cuda_init(const int *devices, int count){
    if(!coli_cuda_load()) return 0;
    return g_cuda.init(devices, count);
}

void coli_cuda_shutdown(void){
    if(g_cuda.available && g_cuda.shutdown) g_cuda.shutdown();
}

int coli_cuda_device_count(void){
    if(!g_cuda.available) return 0;
    return g_cuda.device_count();
}

int coli_cuda_device_at(int index){
    if(!g_cuda.available) return -1;
    return g_cuda.device_at(index);
}

int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_cuda.available){ if(free_bytes)*free_bytes=0; if(total_bytes)*total_bytes=0; return 0; }
    return g_cuda.mem_info(device, free_bytes, total_bytes);
}

void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    if(!g_cuda.available){ if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0; return; }
    g_cuda.stats(device, tensor_count, tensor_bytes);
}

void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(!g_cuda.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_cuda.group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_mlp(gate, up, down, y, x, S);
}

int coli_cuda_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs, const int *rows, int count,
                           float *y, const float *x){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group(gates, ups, downs, rows, count, y, x);
}

int coli_cuda_attention_absorb(ColiCudaTensor *kv_b, float *ctx, const float *q,
                               const float *latent, const float *rope, int H, int Q,
                               int R, int V, int K, int T, float attention_scale){
    if(!g_cuda.available) return 0;
    return g_cuda.attention_absorb(kv_b, ctx, q, latent, rope, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights,
                            const float *scales, int fmt, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_upload(tensor, weights, scales, fmt, I, O, device);
}

int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.matmul(tensor, y, x, weights, scales, fmt, S, I, O, device);
}

void coli_cuda_tensor_free(ColiCudaTensor *tensor){
    if(g_cuda.available && g_cuda.tensor_free) g_cuda.tensor_free(tensor);
}

size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_bytes(tensor);
}

int coli_cuda_tensor_device(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return -1;
    return g_cuda.tensor_device(tensor);
}

/* ---- streaming tier wrappers: absent symbols degrade to "unavailable" ---- */

int coli_cuda_stream_init(int device, size_t vram_bytes, size_t slot_bytes, int tile_kb){
    if(!g_cuda.available || !g_cuda.stream_available) return 0;
    return g_cuda.stream_init(device, vram_bytes, slot_bytes, tile_kb);
}

void coli_cuda_stream_shutdown(void){
    if(g_cuda.available && g_cuda.stream_available) g_cuda.stream_shutdown();
}

int coli_cuda_host_register(void *ptr, size_t bytes){
    if(!g_cuda.available || !g_cuda.stream_available) return 0;
    return g_cuda.host_register(ptr, bytes);
}

void coli_cuda_host_unregister(void *ptr){
    if(g_cuda.available && g_cuda.stream_available) g_cuda.host_unregister(ptr);
}

ColiCudaStreamHandle *coli_cuda_stream_begin(int layer, const int *eids, int count,
        const void *const *gw, const void *const *uw, const void *const *dw,
        const float *const *gs, const float *const *us, const float *const *ds,
        const int *fg, const int *fu, const int *fd,
        int D, int I, const float *x, const int *nrows, int total_rows,
        uint64_t *accepted_mask){
    if(!g_cuda.available || !g_cuda.stream_available){
        if(accepted_mask) *accepted_mask = 0;
        return NULL;
    }
    return g_cuda.stream_begin(layer, eids, count, gw, uw, dw, gs, us, ds,
                               fg, fu, fd, D, I, x, nrows, total_rows, accepted_mask);
}

int coli_cuda_stream_end(ColiCudaStreamHandle *handle, float *y){
    if(!g_cuda.available || !g_cuda.stream_available) return 0;
    return g_cuda.stream_end(handle, y);
}

int coli_cuda_stream_put(int layer, int eid,
        const void *gw, const void *uw, const void *dw,
        const float *gs, const float *us, const float *ds,
        int fg, int fu, int fd, int D, int I){
    if(!g_cuda.available || !g_cuda.stream_available) return 0;
    return g_cuda.stream_put(layer, eid, gw, uw, dw, gs, us, ds, fg, fu, fd, D, I);
}

int coli_cuda_stream_query(int layer, int eid){
    if(!g_cuda.available || !g_cuda.stream_available) return 0;
    return g_cuda.stream_query(layer, eid);
}

void coli_cuda_stream_stats(uint64_t st[12]){
    if(!g_cuda.available || !g_cuda.stream_available){
        if(st) for(int i = 0; i < 12; i++) st[i] = 0;
        return;
    }
    g_cuda.stream_stats(st);
}

#endif /* _WIN32 */
