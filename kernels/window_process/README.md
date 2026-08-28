# Fused window process kernels

`torch.roll` + `window_partition`, and its inverse, fused into a single pass.

Both are pure data movement, so the eager path materialises the tensor twice —
`torch.roll` writes a copy, then `window_partition` permutes and calls
`.contiguous()` for another — where the fused kernel writes once. Peak memory
drops with the copy that is no longer made.

Enable with `--fused_window_process` (see `get_started.md`), or by passing
`fused_window_process=True` to the model.

## Build

```bash
cd kernels/window_process
python setup.py install
```

The same command builds on ROCm; see [Building on AMD](#building-on-amd).

## Usage

```python
from window_process import WindowProcess, WindowProcessReverse

# (B, H, W, C) -> (B * nH * nW, window_h, window_w, C)
windows = WindowProcess.apply(x, B, H, W, C, -shift_size, window_size)

# and back
x = WindowProcessReverse.apply(windows, B, H, W, C, shift_size, window_size)
```

`shift_size` and `window_size` accept an `int` (isotropic) or an `(h, w)` pair:

```python
windows = WindowProcess.apply(x, B, H, W, C, (-2, -8), (4, 16))
```

`shift_size` is the negated `torch.roll` shift on the partition path, which is
why the call sites pass `-shift_size` forward and `+shift_size` back.

## Constraints

| | |
|---|---|
| dtype | float64, float32, float16, bfloat16 |
| layout | contiguous, channels last in memory: `(..., C)` |
| tiling | `H % window_h == 0` and `W % window_w == 0` |
| shift | `abs(shift) < window size`, per axis |
| size | `B * H * W * C < 2^31` — offsets are computed in `int` |
| grid | `B * nH * nW <= 65535` and `H <= 65535` — CUDA `grid.z` and `grid.y` |

All of these are enforced by `TORCH_CHECK` before the launch.

Note that `nH != nW` follows from `H != W` alone: with a square window,
`nH = H / window_size` and `nW = W / window_size`. A non-square window is not
required to reach that case, and `img_size` is documented as `int | tuple(int)`.

## Tests

| file | needs a GPU | what it covers |
|---|---|---|
| `test_index_math.py` | no | the index arithmetic of all four kernels, transcribed to PyTorch in `reference.py` |
| `hipify_check.py` | no | CUDA → HIP translation is complete, and every launch carries a stream |
| `unit_test.py` | yes | parity with the eager path across dtype × shape × shift, forward and backward |
| `test_model_parity.py` | yes | a `SwinTransformer` gives identical logits and gradients either way |
| `benchmark.py` | yes | wall time and peak memory, fused vs eager |

```bash
python test_index_math.py      # runs anywhere
python hipify_check.py         # runs anywhere
python unit_test.py            # skips without a GPU or the built extension
python test_model_parity.py
python benchmark.py
```

The first two run in CI on every push, on a CPU runner.

The kernels perform no arithmetic on the values, only gathers, so parity is
asserted with `torch.equal` on every dtype including float16 and bfloat16: any
deviation is an indexing error rather than a rounding one. For the same reason
`gradcheck` adds nothing, even though float64 does dispatch.

`reference.py` transcribes the kernels' `input_offset` computation to vectorised
PyTorch. That is not an approximation of the kernels — the offset computation is
their entire logic — which is what makes them testable on a CPU, with no
compiled extension and no GPU of either vendor.

## Validation

Parity is asserted against the eager path on an **RTX 3080** (torch
`2.8.0+cu129`, CUDA 12.9) and an **AMD Instinct MI300X** (gfx942, torch
`2.12.0+rocm7.14.0`): `unit_test.py` 15/15 and `test_model_parity.py` 4/4 on
both, bit-exact, over square and non-square images and windows including
coprime `nH`/`nW`. `compute-sanitizer` (memcheck, initcheck, synccheck) is clean
on the RTX 3080 across all four kernels, forward and backward; ROCm has no
equivalent run.

## Performance

`benchmark.py`, forward + backward, batch 192, 200 iterations after 10 warm-up.
The `partition` direction is shown; `merge` tracks it within 6% at every point
on the RTX 3080 and within 1% on the MI300X. `benchmark.py` prints the full
matrix, both directions and all four configurations.

**RTX 3080 (10 GB)**

| config | dtype | eager ms | fused ms | speedup | eager MiB | fused MiB |
|---|---|---:|---:|---:|---:|---:|
| stage 1 56×56 w7 | float32 | 7.137 | 2.398 | 2.98× | 1543.5 | 882.0 |
| stage 1 56×56 w7 | float16 | 5.223 | 2.083 | 2.51× | 771.8 | 441.0 |
| stage 1 56×56 w7 | bfloat16 | 5.074 | 1.948 | 2.60× | 771.8 | 441.0 |
| stage 2 28×28 w7 | float32 | 3.711 | 1.156 | 3.21× | 771.8 | 441.0 |
| stage 2 28×28 w7 | float16 | 2.552 | 0.700 | 3.65× | 392.0 | 224.0 |
| stage 2 28×28 w7 | bfloat16 | 2.783 | 0.726 | 3.84× | 392.0 | 224.0 |

Non-square configurations (32×16 with a square window, 16×64 with a 4×16 window)
land in the same 2.5–3.0× band.

**AMD Instinct MI300X**

| config | dtype | eager ms | fused ms | speedup | eager MiB | fused MiB |
|---|---|---:|---:|---:|---:|---:|
| stage 1 56×56 w7 | float32 | 0.911 | 0.598 | 1.52× | 1543.5 | 882.0 |
| stage 1 56×56 w7 | float16 | 0.652 | 0.588 | 1.11× | 771.8 | 441.0 |
| stage 1 56×56 w7 | bfloat16 | 0.646 | 0.589 | 1.10× | 771.8 | 441.0 |
| stage 2 28×28 w7 | float32 | 0.449 | 0.188 | 2.38× | 771.8 | 441.0 |
| stage 2 28×28 w7 | float16 | 0.323 | 0.152 | 2.12× | 392.0 | 224.0 |
| stage 2 28×28 w7 | bfloat16 | 0.320 | 0.152 | 2.10× | 392.0 | 224.0 |

Non-square configurations land in a 1.03–1.44× band.

Memory is identical across vendors, as it must be — the allocations are the
same. Time is not. The eager baseline is what moved: 7.137 ms at stage 1 on the
3080 against 0.911 ms here, so both paths compress toward a floor and the ratio
closes with them. The fused rows also stop scaling with dtype on the MI300X
(0.598 ms float32 against 0.588 float16, for half the bytes) while the eager
path still does, so at these sizes the fused kernel is no longer bandwidth bound
on CDNA. Block width is not the cause — that is measured below.

`bfloat16` is dispatched directly. The upstream kernel raises on a `bfloat16`
input, so the fused path previously needed an fp32 round trip.

## Building on AMD

`setup.py` needs no change. `CUDAExtension` hipifies its own sources when
`torch.version.hip` is set, substitutes `hipcc`, and derives `--offload-arch`
from `PYTORCH_ROCM_ARCH`:

```bash
PYTORCH_ROCM_ARCH=gfx942 python setup.py install
```

Do not pass `--offload-arch` through `extra_compile_args`: PyTorch's
`_get_rocm_arch_flags()` skips its own detection as soon as it sees one, which
pins the build to a single GPU.

Two things are worth knowing before building:

- **`__ldg` has no HIP overload for `c10::Half`**, and half is in the dispatch.
  Reads go through the `SWIN_WP_LDG` macro for that reason: on NVIDIA it expands
  to the same `__ldg(ptr)` tokens as before, on AMD to a plain load. The hint is
  advisory on both, so no result changes. `hipify_check.py` cannot catch this
  and says so — it is a symbol-level check, and a symbol can translate cleanly
  and still not compile.

- **AMD's ROCm 7.14 PyTorch images ship the SDK as pip wheels** with no
  `/opt/rocm` tree, and those wheels carry no rocThrust, hipSPARSE, hipBLAS,
  hipBLASLt or hipSOLVER headers — which torch's own headers include when
  compiling device code. In that state *no* PyTorch HIP extension compiles; a
  file whose only content is `#include <ATen/ATen.h>` fails the same way.
  Installing the matching dev packages is enough, and touches no source:

  ```bash
  apt-get install -y amdrocm-blas-dev7.14 amdrocm-hipblas-common-dev7.14 \
                     amdrocm-sparse-dev7.14 amdrocm-solver-dev7.14 \
                     librocthrust-dev librocprim-dev
  export CPLUS_INCLUDE_PATH=/opt/rocm/core-7.14/include:/usr/include
  ```

### Portability

The kernels use no shared memory, no `__syncthreads()` and no warp-level
primitives. Nothing depends on the warp or wavefront width, so CDNA's 64-wide
wavefront against NVIDIA's 32-wide warp affects occupancy and nothing else. The
three block widths in `best_block_dim()` are multiples of 64, so neither
platform schedules a partial wave.

Those thresholds were tuned on NVIDIA, then measured on CDNA by rebuilding with
`-DSWIN_WP_BLOCK_DIM=N` (stage 1 / stage 2, float32, MI300X, fused time):

| block width | stage 1 | stage 2 |
|---|---:|---:|
| 64 (what the heuristic picks here) | **0.598 ms** | **0.188 ms** |
| 256 | 0.632 ms | 0.212 ms |
| 1024 | 1.230 ms | 0.326 ms |

The NVIDIA-tuned choice wins on CDNA too, and widening hurts monotonically — at
1024 the fused path becomes *slower than eager* (0.74×). The grid is fixed by
the window geometry, so threads past `C` have nothing to do; swin-tiny's `C` is
96 at stage 1, and a 1024-wide block leaves 928 lanes idle.

The one substantive difference between the two platforms is the launch stream.
Upstream passes `0`, the null stream; this version passes
`at::cuda::getCurrentCUDAStream()`, which hipify rewrites to the current HIP
stream. `hipify_check.py` verifies that, prints the full symbol mapping, and
fails if anything is left untranslated.
