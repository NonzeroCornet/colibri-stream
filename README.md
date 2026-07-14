<h1 align="center">ColiStream</h1>
<p align="center"><b>A streaming execution-cache architecture for Mixture-of-Experts inference on consumer GPUs.</b></p>
<p align="center"><i>Stop asking what fits in VRAM. Keep every engine of the memory hierarchy busy on a single stream of expert tiles.</i></p>

---

ColiStream runs **GLM-5.2 — a 744B-parameter MoE — on one consumer GPU** by
treating the GPU as a *streaming compute device* instead of a container to
fill. VRAM is not storage; it is an **execution cache** at the top of one
managed hierarchy:

```
SSD   →   RAM   →   VRAM   →   Tensor cores
 L3        L2        L1         compute
```

Expert weights flow through as tile-chunked asynchronous DMA. Compute is
event-driven at sub-expert granularity — matrix multiplication starts while
later tiles of the same expert are still on the PCIe bus. A two-lane bus
scheduler bounds speculative-prefetch interference to a single tile. A
recency-primary, decayed-frequency policy keeps temporally local experts
resident. The scheduler's objective is simple: **no hardware engine — NVMe
queue, PCIe copy engine, SMs — idles while there is work anywhere in the
pipeline.**

```
$ ./coli chat --stream
  🐦 ColiStream — GLM-5.2 · 744B MoE · int4 · streaming GPU
  ✓ ready in 32s · resident 9.9 GB · VRAM execution cache 727 slots
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## Why streaming

A 744B MoE activates only ~40B parameters per token, and only ~11 GB of those
change token to token (the routed experts: 75 MoE layers × 8 of 256 experts,
~19 MB each at int4). The 370 GB expert store lives on disk; the ~10 GB dense
core lives in RAM. The open question was the GPU: a *synchronous* NVMe→VRAM
copy per expert would just move the bottleneck to PCIe — so prior engines left
streamed experts on the CPU. ColiStream's answer is that the copy is only a
bottleneck if it is synchronous:

- **Tile streaming** — experts transfer as 512 KB chunks (`--stream-tile-kb`);
  two CUDA events per expert let the gate/up matmuls run while the down
  projection is still on the bus.
- **N-slot double buffering** — while the GPU computes expert *j*, the copy
  engine fills expert *j+1*'s slot; a pool of ~700 slots (16 GB card)
  generalizes the classic A/B swap across a whole MoE layer plus prefetch.
- **Two-phase demand submit** — RAM-resident experts start their PCIe
  transfers *while* the async I/O pool is still reading the disk misses.
- **Bounded speculation** — a router-lookahead pilot predicts the next
  layer's experts (71.6% recall) and uploads them on a low-priority lane that
  yields between tiles: worst-case demand delay ≈ 20 µs, ever.
- **Execution-cache retention** — MoE routing repeats; repeated experts are
  served from VRAM with **zero** bus traffic under a recency-primary,
  decayed-frequency replacement policy (frequency-primary provably pollutes —
  see the paper).
- **Failure containment** — every miss, race, oversize, or CUDA error falls
  back to the CPU expert path. Output correctness never depends on the cache,
  the scheduler, or the predictor being right.

## Measured (RTX 5070 Ti 16 GB, PCIe 5.0 x16, Windows 11, CUDA 13.1)

| metric | value |
|---|---|
| cold expert streaming, compute fully hidden | **26.6–27.1 GB/s — ~2,250 experts/s** |
| warm (VRAM execution-cache hit) | **~16,000 experts/s**, zero PCIe traffic |
| replacement-policy ablation (LFU-primary vs ColiStream) | **7.6×** |
| end-to-end vs optimized AVX-512 CPU path, disk-bound workload | **1.065×** (Amdahl compute-ceiling: 1.049×) |
| correctness vs `transformers` oracle | token-exact: TF **32/32**, greedy **20/20** |

Decode ceilings implied by the pipeline on this card: ~3.7 tok/s streaming
everything cold from RAM, ~26 tok/s from the warm execution cache — the gap
to a slow disk is a hardware upgrade, not a software promise. Full
methodology, models, and fairness notes: the
[ColiStream paper](paper/colistream.pdf) and
[docs/streaming-tier.md](docs/streaming-tier.md).

## Quick start

You need: a CUDA GPU (tested on sm_120), gcc with OpenMP (Linux/macOS or
MinGW-w64 on Windows), ~16 GB RAM minimum, and the ~370 GB int4 model
container on a local NVMe.

```bash
# Linux
cd c
make CUDA=1
./glm --stream 64 4 4                     # engine self-test / batch mode
python coli chat --model /nvme/glm52_i4 --stream
```

```powershell
# Windows 11 (MinGW host + MSVC/nvcc CUDA DLL)
cd c
make cuda-dll CUDA_ARCH=sm_120            # coli_cuda.dll (from a vcvars64 shell)
make glm.exe CUDA_DLL=1 ARCH=native
python coli chat --model D:\glm52_i4 --stream --topp 0.7
```

All ColiStream settings are flags (env-var equivalents remain for scripts):

| flag | default | meaning |
|---|---|---|
| `--stream` / `--no-stream` | off | enable the streaming tier (implies the CUDA backend) |
| `--stream-vram GB` | auto (free − 2 GB) | VRAM slot-pool budget |
| `--stream-tile-kb KB` | 512 | DMA chunk = the bus scheduler's preemption grain (64–4096) |
| `--stream-pin-gb GB` | 8 | host page-lock budget for true async DMA |

Get the model (pre-quantized int4 container with int8 MTP heads, ~370 GB):
**https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp** — or
convert from the FP8 original with `./coli convert` (resumable, never needs
the full 756 GB checkpoint on disk at once).

Useful companions: `--topp 0.7` (30–40% fewer expert loads), `PIPE=1` (async
disk I/O pool), `PILOT=1 PILOT_REAL=1` (router-lookahead prefetch into the
execution cache), `coli plan` / `coli doctor` (placement planning and
read-only readiness checks before loading 370 GB).

## Verify it yourself

```bash
cd c
make cuda-stream-test                     # standalone GPU suite: formats, eviction,
                                          # oversize rejection, concurrent prefetch
./backend_stream_test --bench 2048        # pipeline throughput on GLM-shaped experts

pip install torch transformers safetensors
python tools/make_glm_oracle.py           # 2.4 MB tiny model with the real architecture
SNAP=./glm_tiny TF=1 ./glm --stream 64 16 16    # expect "32/32 positions"
```

The streaming tier is validated token-exactly against the `transformers`
reference with async I/O and speculative prefetch active. Like any GPU path,
accumulation order differs from the CPU path, so greedy argmax ties can fork
the token stream — every emitted token remains the argmax of a valid forward.

## How it's built

```
paper/                      the ColiStream paper (architecture, proofs, measurements)
c/
├── backend_stream.cu       the streaming tier: slot pool, tile engine, two copy
│                           lanes, replacement policy, statistics
├── glm.c                   single-file GLM-5.2 engine (host integration: two-phase
│                           submit, residency probe, pilot hook, page-lock budget)
├── backend_cuda.{h,cu}     resident CUDA tier (pinned hot experts) + shared API
├── backend_loader.c        Windows runtime-DLL loader (old DLLs degrade gracefully)
├── coli                    CLI: chat / run / serve / plan / doctor / bench / convert
├── tests/                  dependency-free C, CUDA and Python tests
└── tools/                  converter, oracle generator, benchmark fixtures
docs/streaming-tier.md      design→implementation mapping, knobs, measurements
```

The runtime is C and CUDA with zero mandatory dependencies (Python only for
the one-time converter and the OpenAI-compatible HTTP gateway). CPU-only
builds (`make`) remain fully functional: ColiStream is an opt-in tier, and
everything degrades to the same validated CPU engine.

## Foundation and credits

ColiStream is built on the excellent
[colibrì](https://github.com/JustVugg/colibri) engine by JustVugg, which
contributes the CPU inference core this project extends: the faithful
GLM-5.2 forward (MLA attention with compressed KV-cache, DSA sparse
attention, sigmoid router), the int8/int4/int2 quantization containers and
AVX-512/AVX2 kernels, native MTP speculative decoding, the RAM LRU +
learned-pin expert tiers, KV persistence, the byte-level BPE tokenizer, and
the FP8→int4 converter. See the upstream repository for the full engine
documentation and its community benchmark program. What ColiStream adds is
the architecture described above and in the paper: the VRAM execution cache,
tile streaming with event-driven partial compute, the two-lane PCIe
scheduler, the RAM-above-VRAM residency probe, and the replacement-policy
analysis.

## License

Apache 2.0, same as the upstream engine. GLM-5.2 weights are released by
Z.ai under MIT.
