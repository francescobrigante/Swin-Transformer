# Fused window process kernels

`torch.roll` + `window_partition` and its inverse, fused into a single pass.
Both are pure data movement, so the eager path materialises the tensor twice —
`torch.roll` writes a copy, then `window_partition` permutes and calls
`.contiguous()` for another — where the fused kernel writes once.

Enable with `--fused_window_process` (see `get_started.md`), or by passing
`fused_window_process=True` to the model.

## Build

```bash
cd kernels/window_process
python setup.py install
```

The same command builds on ROCm. `CUDAExtension` hipifies its sources when
`torch.version.hip` is set, so there is no separate HIP source tree and no
change to `setup.py`. Select the target architecture with the standard
environment variable, for example MI300X:

```bash
PYTORCH_ROCM_ARCH=gfx942 python setup.py install
```

Do not pass `--offload-arch` through `extra_compile_args`: PyTorch's
`_get_rocm_arch_flags()` skips its own detection as soon as it sees one, which
pins the build to a single GPU.

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

All of these are enforced by `TORCH_CHECK` before the launch. Violating them
used to read the wrong element, or read out of bounds, without any error.

Note that `nH != nW` follows from `H != W` alone: with a square window,
`nH = H / window_size` and `nW = W / window_size`. A non-square window is not
required, and `img_size` is documented as `int | tuple(int)`.

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

The kernels perform no arithmetic on the values, only gathers, so parity is
asserted with `torch.equal` on every dtype including float16 and bfloat16: any
deviation is an indexing error rather than a rounding one. For the same reason
`gradcheck` adds nothing, even though float64 does dispatch.

`reference.py` transcribes the kernels' `input_offset` computation to vectorised
PyTorch. That is not an approximation of the kernels — the offset computation is
their entire logic — so it makes their correctness testable on a CPU, with no
compiled extension.

## Performance

`benchmark.py`, the fused kernel against the `torch.roll` + `window_partition`
it replaces, forward and backward, on an **RTX 3080 (10 GB)**, torch
`2.8.0+cu129` / CUDA 12.9, batch 192, 200 iterations after 10 warm-up:

| config | op | dtype | eager ms | fused ms | speedup | eager MiB | fused MiB |
|---|---|---|---:|---:|---:|---:|---:|
| stage 1 56×56 w7 | partition | float32 | 7.137 | 2.398 | 2.98× | 1543.5 | 882.0 |
| stage 1 56×56 w7 | partition | float16 | 5.223 | 2.083 | 2.51× | 771.8 | 441.0 |
| stage 1 56×56 w7 | partition | bfloat16 | 5.074 | 1.948 | 2.60× | 771.8 | 441.0 |
| stage 1 56×56 w7 | merge | float32 | 7.191 | 2.405 | 2.99× | 1323.0 | 882.0 |
| stage 1 56×56 w7 | merge | float16 | 5.104 | 2.013 | 2.54× | 661.5 | 441.0 |
| stage 1 56×56 w7 | merge | bfloat16 | 5.150 | 1.998 | 2.58× | 661.5 | 441.0 |
| stage 2 28×28 w7 | partition | float32 | 3.711 | 1.156 | 3.21× | 771.8 | 441.0 |
| stage 2 28×28 w7 | partition | float16 | 2.552 | 0.700 | 3.65× | 392.0 | 224.0 |
| stage 2 28×28 w7 | partition | bfloat16 | 2.783 | 0.726 | 3.84× | 392.0 | 224.0 |
| stage 2 28×28 w7 | merge | float32 | 3.907 | 1.222 | 3.20× | 661.5 | 441.0 |
| stage 2 28×28 w7 | merge | float16 | 2.650 | 0.735 | 3.60× | 336.0 | 224.0 |
| stage 2 28×28 w7 | merge | bfloat16 | 2.630 | 0.714 | 3.68× | 336.0 | 224.0 |
| non-square 32×16 w8 | partition | float32 | 1.218 | 0.417 | 2.92× | 252.0 | 144.0 |
| non-square 32×16 w8 | partition | float16 | 0.870 | 0.335 | 2.59× | 126.0 | 72.0 |
| non-square 32×16 w8 | partition | bfloat16 | 0.871 | 0.340 | 2.56× | 126.0 | 72.0 |
| non-square 32×16 w8 | merge | float32 | 1.224 | 0.414 | 2.95× | 216.0 | 144.0 |
| non-square 32×16 w8 | merge | float16 | 0.872 | 0.346 | 2.52× | 108.0 | 72.0 |
| non-square 32×16 w8 | merge | bfloat16 | 0.867 | 0.343 | 2.53× | 108.0 | 72.0 |
| non-square window 16×64 w4×16 | partition | float32 | 2.434 | 0.814 | 2.99× | 504.0 | 288.0 |
| non-square window 16×64 w4×16 | partition | float16 | 1.681 | 0.658 | 2.55× | 252.0 | 144.0 |
| non-square window 16×64 w4×16 | partition | bfloat16 | 1.692 | 0.660 | 2.56× | 252.0 | 144.0 |
| non-square window 16×64 w4×16 | merge | float32 | 2.402 | 0.801 | 3.00× | 432.0 | 288.0 |
| non-square window 16×64 w4×16 | merge | float16 | 1.708 | 0.658 | 2.59× | 216.0 | 144.0 |
| non-square window 16×64 w4×16 | merge | bfloat16 | 1.707 | 0.655 | 2.61× | 216.0 | 144.0 |

The path is memory bound, so the gain tracks the second materialisation the
eager path pays (`torch.roll` writes a copy, `window_partition` calls
`.contiguous()` for another) and the fused kernel does not; peak memory drops
with it. `bfloat16` is dispatched directly — the upstream kernel raises on a
`bfloat16` input, so the fused path previously needed an fp32 round trip.

## Validation

CUDA is validated end to end. The ROCm path is verified statically only — no AMD
GPU was available.

| check | where | result |
|---|---|---|
| index math (`reference.py` vs the composed PyTorch ops) | CPU (Apple M1), re-run on the RTX 3080 host | 9/9 |
| CUDA → HIP translation (`hipify_check.py`) | CPU | complete, every launch keeps its stream |
| compile the extension | RTX 3080 · torch 2.8.0+cu129 · CUDA 12.9 · MSVC 14.44 | builds, no source change |
| kernel parity (`unit_test.py`) | RTX 3080 | 15/15 — 4 kernels × fwd/bwd × {f64, f32, f16, bf16} × 13 shapes × 3 shifts, bit-exact |
| model parity (`test_model_parity.py`) | RTX 3080 | 4/4 — logits and gradients identical, square and non-square (tall and wide) |
| bug on the compiled kernel | RTX 3080 | pre-fix: a stock `img_size=(256,128)` model gives different logits with the fused path, and `compute-sanitizer` reports an out-of-bounds read; fixed: bit-identical to eager |
| `compute-sanitizer` (memcheck, initcheck, synccheck), fixed kernel | RTX 3080 | 0 errors — all 4 kernels, fwd + bwd, every shape above |
| ROCm runtime (build + parity on an AMD GPU) | — | not run — no AMD hardware |

The shapes span every image/window combination: square image with a square
window, square image with a non-square window, and non-square images with square
and with non-square windows, including cases where `nH` and `nW` are coprime.

The one substantive ROCm difference is the launch stream. `torch.utils.hipify`
run on the upstream sources and on this branch:

```c
// upstream, hipified            -- 5th arg is 0: the HIP null stream
hipLaunchKernelGGL((kernel<scalar_t>), dim3(grid), dim3(block), 0, 0, ...);

// this branch, hipified         -- 5th arg is the current stream
hipLaunchKernelGGL((kernel<scalar_t>), dim3(grid), dim3(block), 0,
                   at::cuda::getCurrentCUDAStream(), ...);
```

Older hipify (torch <= 2.8) rewrites that getter to
`at::hip::getCurrentHIPStreamMasqueradingAsCUDA()`; torch >= 2.9 keeps the
`at::cuda::` spelling and resolves it through the compatibility headers. Both are
the current stream; only the `0` on the upstream side is wrong.

`setup.py` is unchanged: `CUDAExtension` hipifies its own sources when
`torch.version.hip` is set (`cpp_extension.py`, the `IS_HIP_EXTENSION` branch),
substitutes `hipcc`, and derives `--offload-arch` from `PYTORCH_ROCM_ARCH` —
passing an arch through `extra_compile_args` disables that detection.

## Portability

The kernels use no shared memory, no `__syncthreads()` and no warp-level
primitives. Nothing depends on the warp or wavefront width, so CDNA's 64-wide
wavefront against NVIDIA's 32-wide warp affects occupancy and nothing else. The
three block widths in `best_block_dim()` are multiples of 64, so neither
platform schedules a partial wave; the thresholds were tuned on NVIDIA and are
not measured on CDNA. Compile with `-DSWIN_WP_BLOCK_DIM=N` to override them.

`hipify_check.py` prints the full mapping and fails if anything is left
untranslated.
