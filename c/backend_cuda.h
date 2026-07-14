#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* COLI_CUDA_DLLEXPORT marks functions exported from coli_cuda.dll on Windows.
 * Define COLI_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
#define COLI_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define COLI_CUDA_DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
COLI_CUDA_DLLEXPORT int coli_cuda_init(const int *devices, int count);
COLI_CUDA_DLLEXPORT void coli_cuda_shutdown(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_at(int index);
COLI_CUDA_DLLEXPORT int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* device < 0 returns aggregate statistics for all configured devices. */
COLI_CUDA_DLLEXPORT void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* Upload without executing, so capacity failures happen during model startup. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2.
 * The first successful call uploads W and its row scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
COLI_CUDA_DLLEXPORT int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

COLI_CUDA_DLLEXPORT void coli_cuda_tensor_free(ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

/* ---- Streaming expert tier (backend_stream.cu): VRAM as an execution cache.
 * Experts are streamed on demand into a pool of fixed-size VRAM slots as
 * tile-chunked async copies overlapped with compute (double-buffered across
 * slots); slots are retained under a score-based LFRU policy so temporally
 * local experts become cache hits. See docs/streaming-tier.md. ---- */

/* Opaque in-flight demand group (one 64-expert MoE block submit). */
typedef struct ColiCudaStreamHandle ColiCudaStreamHandle;

/* Carve `vram_bytes` of `device` into slots of `slot_bytes` (one expert:
 * gate+up+down weights and scales, 256B-aligned each). tile_kb sets the DMA
 * chunk size (64..4096, the preemption granularity of the PCIe scheduler). */
COLI_CUDA_DLLEXPORT int coli_cuda_stream_init(int device, size_t vram_bytes,
                                              size_t slot_bytes, int tile_kb);
COLI_CUDA_DLLEXPORT void coli_cuda_stream_shutdown(void);

/* Pin host RAM (expert slabs) so streamed uploads are true async DMA.
 * Failure is harmless: unpinned slabs still copy, just slower. */
COLI_CUDA_DLLEXPORT int coli_cuda_host_register(void *ptr, size_t bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_host_unregister(void *ptr);

/* Demand path, split-phase like the Metal block submit: begin() enqueues
 * every transfer + kernel and returns immediately (the CPU overlaps disk
 * reads / shared-expert work with the GPU pipeline); end() synchronizes and
 * fills y[total_rows,D] (rows packed in call order). Weight pointers are the
 * CPU container slabs (int4 offset-nibble / int2 / int8 / f32) and must stay
 * valid until end() returns. accepted_mask bit j = expert j computed on GPU;
 * rejected experts (slot too small, cache exhausted) stay on the CPU path.
 * end() returning 0 means the WHOLE group must be recomputed on the CPU. */
COLI_CUDA_DLLEXPORT ColiCudaStreamHandle *coli_cuda_stream_begin(
        int layer, const int *eids, int count,
        const void *const *gw, const void *const *uw, const void *const *dw,
        const float *const *gs, const float *const *us, const float *const *ds,
        const int *fg, const int *fu, const int *fd,
        int D, int I, const float *x, const int *nrows, int total_rows,
        uint64_t *accepted_mask);
COLI_CUDA_DLLEXPORT int coli_cuda_stream_end(ColiCudaStreamHandle *handle, float *y);

/* Residency probe: 1 if (layer,eid) is VRAM-resident now. The engine uses it
 * to SKIP the disk read for cache-held experts (VRAM as a tier above RAM);
 * a stale answer is safe — begin() rejects NULL-slab misses to the CPU path. */
COLI_CUDA_DLLEXPORT int coli_cuda_stream_query(int layer, int eid);

/* Speculative path (router-lookahead pilot): synchronous tiled upload on the
 * low-priority copy stream, yielding to demand groups between tiles. Returns
 * once the bytes are on the device — the caller's slab may be recycled
 * immediately after. Returns 1 when the expert is (now) VRAM-resident. */
COLI_CUDA_DLLEXPORT int coli_cuda_stream_put(int layer, int eid,
        const void *gw, const void *uw, const void *dw,
        const float *gs, const float *us, const float *ds,
        int fg, int fu, int fd, int D, int I);

/* st[0]=hits st[1]=misses st[2]=evictions st[3]=bytes_streamed st[4]=tiles
 * st[5]=prefetch_puts st[6]=prefetch_hits st[7]=prefetch_drops st[8]=rejects
 * st[9]=nslots st[10]=slot_bytes st[11]=reserved */
COLI_CUDA_DLLEXPORT void coli_cuda_stream_stats(uint64_t st[12]);

#ifdef __cplusplus
}
#endif

#endif
