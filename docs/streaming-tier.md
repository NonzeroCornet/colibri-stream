# The Streaming GPU Expert Tier (`COLI_STREAM=1`)

VRAM as an **execution cache**, not storage. This tier implements the
"ColiStream" design: instead of deciding which experts
*fit* in VRAM, the GPU is treated as a streaming compute device at the top of
one managed hierarchy —

```
SSD   →   RAM   →   VRAM   →   SMs
 L3        L2        L1       compute
```

— where expert weights flow through as **tile-chunked async DMA** and the
scheduler's job is to keep the PCIe bus and the tensor units busy at the same
time. It complements (does not replace) the existing resident CUDA tier: the
measured long-term hot set still gets pinned VRAM residency via
`CUDA_EXPERT_GB`/`PIN`; the streaming tier serves **everything else** — the
cold and warm tail that used to be CPU-only.

## Quick start

```bash
# Linux
make CUDA=1
COLI_CUDA=1 COLI_STREAM=1 SNAP=/nvme/glm52_i4 ./glm 64 4 4

# Windows (MinGW host + MSVC/nvcc DLL)
make cuda-dll CUDA_ARCH=sm_120     # coli_cuda.dll (from a vcvars64 shell)
make glm.exe CUDA_DLL=1 ARCH=native
$env:COLI_CUDA="1"; $env:COLI_STREAM="1"
python coli chat --model D:\glm52_i4
```

Knobs — flags on `glm` and on every `coli` subcommand (flags win over the
env equivalents, kept for compatibility):

| flag | env | default | meaning |
|---|---|---|---|
| `--stream` / `--no-stream` | `COLI_STREAM` | off | enable the tier (implies the CUDA backend on device 0) |
| `--stream-vram GB` | `STREAM_VRAM_GB` | auto (free − 2 GB) | VRAM slot-pool budget, carved **after** the resident tier |
| `--stream-tile-kb KB` | `STREAM_TILE_KB` | 512 | DMA chunk size = the PCIe scheduler's preemption grain (64–4096) |
| `--stream-pin-gb GB` | `STREAM_PIN_GB` | 8 | budget for page-locking host expert slabs (true async DMA) |
| `PILOT=1 PILOT_REAL=1` | (env only) | off | router-lookahead prefetch now also feeds the VRAM cache |

```powershell
# examples
./glm --stream --stream-vram 8 --stream-tile-kb 2048 64 4 4
python coli chat --model D:\glm52_i4 --stream --topp 0.7
```

Composes with `PIPE=1` (async disk loads), `PILOT_REAL=1`, `--topp`, MTP, and
the resident tier. Works on Windows through the runtime-DLL split: the eight
new `coli_cuda_stream_*`/`coli_cuda_host_*` symbols resolve **optionally**, so
an old `coli_cuda.dll` still provides the resident tier.

## Architecture mapping

How each element of the design document landed in code
(`c/backend_stream.cu` + the `g_stream` paths in `c/glm.c`):

| design element | implementation |
|---|---|
| VRAM compute buffers A/B, "swap after completion" | Generalized to an N-slot pool: while the exec stream computes expert *j*, the copy stream is already filling expert *j+1*'s slot. Double-buffering is the N=2 case; the pool makes the same overlap cover a whole MoE block plus prefetch. |
| Tile streaming / micro-streaming experts | Every upload is chunked into `STREAM_TILE_KB` copies. Two CUDA events split each expert: kernels for **gate/up start while down is still on the bus** (gate+up are ⅔ of the bytes, transferred first). Compute is event-driven, not upload-then-execute. |
| Four async pipeline stages (SSD→RAM→PCIe→GPU) | SSD→RAM stays on colibrì's proven paths (`PIPE` I/O pool, pilot thread). RAM→VRAM is the new copy-stream stage. GPU compute is the exec stream. The MoE block submits in **two phases**: RAM-resident experts stream immediately, disk misses follow as their preads land — PCIe transfers overlap disk reads exactly like the Metal backend's early submit. |
| Expert scheduler / PCIe queue priorities | Demand groups ride a dedicated high-priority copy stream. Speculative uploads ride a second stream and **yield between tiles** whenever a demand group is active, so a prefetch burst delays needed-now bytes by at most one tile (~tens of µs). That is the practical form of "transfers may be canceled": CUDA cannot abort an in-flight copy, so preemption granularity = tile size. |
| Prefetch source 1 — current router | The demand path itself (`stream_begin` per 64-expert block). |
| Prefetch source 2 — layer predictor | `PILOT_REAL=1`: the pilot's cross-layer loads now also push the expert into VRAM (`stream_put`) while the pilot still owns the slab exclusively — no lifetime race with the LRU. |
| Prefetch source 3 — temporal locality | The cache itself: experts used this token stay resident, so the "4 4 7 4 4 7" pattern becomes VRAM hits with **zero** transferred bytes (measured 88–90 % hit rate in decode on the oracle model). |
| Prefetch source 4 — conversation heat / long-term learning | Split across tiers by design: `.coli_usage` + `PIN`/`CUDA_EXPERT_GB` place the persistent hot set as *resident* tensors at startup; inside the streaming cache, frequency is the eviction tiebreak. |
| Score-based cache policy | Recency-primary, frequency-tiebreak, frequency halved every `nslots` evictions (`tier_decay` semantics). **Deliberate deviation** from the document's frequency-heavy sum: a frequency-primary score self-evicts fresh entrants once the pool fills with stale high-frequency slots (LFU pollution — measured as a full working set re-streaming at 24 GB/s forever instead of hitting cache). Transfer cost is a constant (fixed-size slots) and drops out of the comparison. |
| Quantization strategy table | Already colibrì's layout (int8/int4 dense, int4/int2 experts). The stream kernels decode the **CPU container formats in-kernel** (int4 offset-nibble, int2, int8, f32) — no conversion pass, any mix streams as-is. INT1.5/INT1 adaptive compression: future work. |
| "Never let the tensor cores idle" | Cache hits are issued to the exec stream **before** misses so a resident expert never queues behind another expert's PCIe transfer. |

## Measured (RTX 5070 Ti 16 GB, PCIe 5.0 x16, Windows 11, CUDA 13.1)

Standalone pipeline benchmark, GLM-5.2-shaped int4 experts (11.8 MB each),
groups of 8 (one layer's routing), demand path only
(`backend_stream_test --bench 2048`):

| path | throughput |
|---|---|
| cold stream, pinned host slabs | **26.6 GB/s H2D — 2,249 experts/s** with compute pipelined behind the copies |
| cold stream, pageable slabs | 19.0 GB/s (what you get past `STREAM_PIN_GB`) |
| warm (VRAM cache hit) | **~16,000 experts/s**, zero PCIe traffic, kernel-bound |

Back-of-envelope for real GLM-5.2 decode (75 layers × 8 experts =
600 expert-applications/token): a fully RAM-resident expert set streamed
through this tier supports ~3.7 tok/s at the demand-only ceiling; every point
of VRAM-cache hit rate moves the blend toward the ~26 tok/s warm ceiling.
The 512 KB→2 MB tile sweep moved cold throughput 24.8→27 GB/s on this box
(WDDM per-submission overhead); Linux should sit closer to the wire.

Real-model A/B on the same box (744B int4 container, cold-ish cache, greedy
16 tokens, `DRAFT=0 AUTOPIN=0 PIPE=1` on both sides, interleaved with
controls): CPU path 277.3 s vs `COLI_STREAM=1` 260.3 s — **+6.5 % end-to-end
on a disk-bound workload** where the drive (measured 0.5 GB/s cold random
reads, flat from 8 to 32 threads — the engine saturates it) accounts for
~85 % of wall time. The compute-side ceiling is therefore small by physics;
the tier's larger wins need either a faster drive or a warm cache. Fairness
notes: MTP defaults differ between configs (`DRAFT` auto-disables under
CUDA) — an A/B without pinning `DRAFT=0` hands the CPU side a cold-cache MTP
penalty (1216 vs 750 expert-loads/token) and inflates the stream advantage to
a misleading 1.48×; and greedy token paths can fork between kernels (#100),
so per-run hit rates are not directly comparable.

Correctness: token-exact vs the `transformers` oracle — teacher-forcing
**32/32** and greedy generation **20/20** with the tier active, including
under `PIPE=1` and `PILOT_REAL=1` (`SNAP=./glm_tiny TF=1 COLI_CUDA=1
COLI_STREAM=1 ./glm 64 16 16`). The standalone suite
(`make cuda-stream-test`) covers all four weight formats, odd packed dims,
eviction pressure, oversized-expert rejection (the int8 MTP row falls back to
CPU cleanly), two concurrent handles, and a prefetch thread racing demand
groups. As with the resident tier and Metal, GPU accumulation order is not
bit-identical to the CPU path — same forward, slightly different rounding
(#100 applies unchanged).

## Failure containment

Every stage degrades to the existing CPU path, never to a wrong answer:
budget too small → tier disabled at startup with a message; slot too small
for an expert → that expert rejected via `accepted_mask`, computed on CPU;
cache exhausted by busy slots → same; any CUDA error inside a group →
`stream_end` returns 0 and the **whole group** is recomputed on the CPU from
the same slabs (still valid: LRU/ws recycling happens after the drain);
`stream_put` failure → the prediction is simply not cached. Host page-locking
failure is harmless (pageable copies, just slower).

## Files

- `c/backend_stream.cu` — slot pool, tile engine, streams/events, cache policy, stats
- `c/backend_cuda.h` — the 8 new exports (`coli_cuda_stream_*`, `coli_cuda_host_*`)
- `c/backend_loader.c` — optional resolution on Windows (old DLLs keep working)
- `c/glm.c` — `g_stream` paths: two-phase submit in `moe()`, pilot `stream_put`
  hook, slab page-locking in `expert_load`, init/stats
- `c/tests/test_backend_stream.cu` — correctness suite + `--bench [tile_kb]`
