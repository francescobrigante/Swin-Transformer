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

## Portability

The kernels use no shared memory, no `__syncthreads()` and no warp-level
primitives. Nothing depends on the warp or wavefront width, so CDNA's 64-wide
wavefront against NVIDIA's 32-wide warp affects occupancy and nothing else. The
three block widths in `best_block_dim()` are multiples of 64, so neither
platform schedules a partial wave; the thresholds were tuned on NVIDIA and are
not measured on CDNA. Compile with `-DSWIN_WP_BLOCK_DIM=N` to override them.

`hipify_check.py` prints the full mapping and fails if anything is left
untranslated.
