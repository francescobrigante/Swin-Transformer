# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# Written by Francesco Brigante
# --------------------------------------------------------
# Correctness tests for the index arithmetic of the fused window kernels.
#
# These run on CPU against reference.py and need neither a GPU nor the compiled
# extension, so they can gate the index math in CI. The GPU parity tests for the
# compiled kernels live in unit_test.py.
#
#   python -m unittest kernels.window_process.test_index_math -v
# --------------------------------------------------------

import unittest

import torch

import reference as ref


# (H, W, window_h, window_w) -- covers square and non-square feature maps, and
# square and non-square windows.
SHAPES = [
    (8, 8, 4, 4),      # nH == nW == 2   square grid, square window
    (16, 8, 4, 4),     # nH=4, nW=2      taller than wide
    (8, 16, 4, 4),     # nH=2, nW=4      wider than tall
    (8, 32, 4, 8),     # nH=2, nW=4      non-square window
    (32, 8, 8, 4),     # nH=4, nW=2      non-square window, taller than wide
    (24, 24, 4, 8),    # nH=6, nW=3      square grid, non-square window
    (12, 8, 4, 4),     # nH=3, nW=2      coprime counts, both > 1
    (12, 20, 4, 4),    # nH=3, nW=5      coprime counts, nH < nW
    (8, 8, 8, 8),      # nH == nW == 1   single window
    (12, 12, 4, 4),    # nH == nW == 3   odd window count
]

BATCH = 2
CHANNELS = 3


def _shifts(window_h, window_w):
    """W-MSA (no shift), SW-MSA (half window), and an asymmetric shift_h != shift_w
    that only the per-axis signature on this branch can express."""
    return [(0, 0), (window_h // 2, window_w // 2), (1, 2)]


def _make_spatial(B, H, W, C):
    return torch.arange(B * H * W * C, dtype=torch.float32).view(B, H, W, C)


def _make_windows(B, H, W, C, window_h, window_w):
    n = B * (H // window_h) * (W // window_w)
    return torch.arange(n * window_h * window_w * C, dtype=torch.float32).view(
        n, window_h, window_w, C)


class TestKernelIndexMath(unittest.TestCase):
    """Each kernel must equal the composition of PyTorch ops it replaces.

    The kernel `shift_h`/`shift_w` parameters are the negated torch.roll shift on
    the partition path: callers pass -shift_size to the forward kernels and
    +shift_size to the reverse ones. These identities pin that convention down.
    """

    def test_k1_equals_roll_then_partition(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    x = _make_spatial(BATCH, H, W, CHANNELS)
                    got = ref.roll_and_window_partition_forward(x, (sh, sw), (wh, ww))
                    want = ref.window_partition(
                        torch.roll(x, shifts=(sh, sw), dims=(1, 2)), (wh, ww))
                    self.assertTrue(torch.equal(got, want))

    def test_k2_equals_reverse_then_unroll(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    g = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    got = ref.roll_and_window_partition_backward(g, (sh, sw), (wh, ww), H, W)
                    want = torch.roll(
                        ref.window_reverse(g, (wh, ww), H, W), shifts=(-sh, -sw), dims=(1, 2))
                    self.assertTrue(torch.equal(got, want))

    def test_k3_equals_reverse_then_roll(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    x = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    got = ref.window_merge_and_roll_forward(x, (sh, sw), (wh, ww), H, W)
                    want = torch.roll(
                        ref.window_reverse(x, (wh, ww), H, W), shifts=(sh, sw), dims=(1, 2))
                    self.assertTrue(torch.equal(got, want))

    def test_k4_equals_unroll_then_partition(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    g = _make_spatial(BATCH, H, W, CHANNELS)
                    got = ref.window_merge_and_roll_backward(g, (sh, sw), (wh, ww))
                    want = ref.window_partition(
                        torch.roll(g, shifts=(-sh, -sw), dims=(1, 2)), (wh, ww))
                    self.assertTrue(torch.equal(got, want))

    def test_backward_kernels_invert_their_forward(self):
        """K1/K2 and K3/K4 are permutations, so each pair must round-trip."""
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    x = _make_spatial(BATCH, H, W, CHANNELS)
                    k1 = ref.roll_and_window_partition_forward(x, (sh, sw), (wh, ww))
                    self.assertTrue(torch.equal(
                        ref.roll_and_window_partition_backward(k1, (sh, sw), (wh, ww), H, W), x))

                    w = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    k3 = ref.window_merge_and_roll_forward(w, (sh, sw), (wh, ww), H, W)
                    self.assertTrue(torch.equal(
                        ref.window_merge_and_roll_backward(k3, (sh, sw), (wh, ww)), w))


class TestUpstreamRowStrideBug(unittest.TestCase):
    """window_merge_and_roll_forward used `* nH` where the row stride is `* nW`.

    Windows are laid out row-major as `b * nH * nW + wrow * nW + wcol`, so the
    stride between consecutive window rows is nW. The two agree exactly when
    nH == nW, which is why the bug never surfaced: every model in this repository
    is trained on square images.
    """

    def _legacy_index(self, B, H, W, sh, sw, wh, ww):
        return ref.window_merge_and_roll_forward_index(
            B, H, W, sh, sw, wh, ww, legacy_row_stride=True)

    def test_legacy_is_correct_only_on_square_grids(self):
        for H, W, wh, ww in SHAPES:
            nH, nW = H // wh, W // ww
            if nH != nW:
                continue
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw), nH=nH, nW=nW):
                    x = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    want = ref.window_merge_and_roll_forward(x, (sh, sw), (wh, ww), H, W)
                    legacy = ref.window_merge_and_roll_forward(
                        x, (sh, sw), (wh, ww), H, W, legacy_row_stride=True)
                    self.assertTrue(torch.equal(legacy, want))

    def test_legacy_reads_out_of_bounds_when_nH_greater_than_nW(self):
        """nH > nW makes the miscomputed offset exceed the input, an illegal read."""
        checked = 0
        for H, W, wh, ww in SHAPES:
            nH, nW = H // wh, W // ww
            if nH <= nW:
                continue
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw), nH=nH, nW=nW):
                    numel = BATCH * nH * nW * wh * ww
                    legacy_max = int(self._legacy_index(BATCH, H, W, sh, sw, wh, ww).max())
                    self.assertGreaterEqual(legacy_max, numel)
                    checked += 1
        self.assertGreater(checked, 0, "no nH > nW shape exercised")

    def test_legacy_returns_wrong_values_when_nH_less_than_nW(self):
        """nH < nW keeps the offset in bounds, so the corruption is silent."""
        checked = 0
        for H, W, wh, ww in SHAPES:
            nH, nW = H // wh, W // ww
            if nH >= nW:
                continue
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw), nH=nH, nW=nW):
                    numel = BATCH * nH * nW * wh * ww
                    legacy_max = int(self._legacy_index(BATCH, H, W, sh, sw, wh, ww).max())
                    self.assertLess(legacy_max, numel)

                    x = _make_windows(BATCH, H, W, CHANNELS, wh, ww)
                    want = ref.window_merge_and_roll_forward(x, (sh, sw), (wh, ww), H, W)
                    legacy = ref.window_merge_and_roll_forward(
                        x, (sh, sw), (wh, ww), H, W, legacy_row_stride=True)
                    self.assertFalse(torch.equal(legacy, want))
                    checked += 1
        self.assertGreater(checked, 0, "no nH < nW shape exercised")


class TestIntraWindowModuloIsEquivalent(unittest.TestCase):
    """The upstream intra-window modulo omits `% H` / `% W`, which is a no-op here.

    `(y - s + H) % window_h` and `((y - s + H) % H) % window_h` differ by a
    multiple of H, and H is a multiple of window_h whenever the launcher's
    `nH = H / window_h` is exact -- which it must be for the kernel to be valid
    at all. The explicit form is kept for readability, not for correctness.
    """

    def test_forms_agree_under_exact_divisibility(self):
        for H, W, wh, ww in SHAPES:
            for sh, sw in _shifts(wh, ww):
                with self.subTest(H=H, W=W, wh=wh, ww=ww, shift=(sh, sw)):
                    explicit = ref.window_merge_and_roll_forward_index(
                        BATCH, H, W, sh, sw, wh, ww, legacy_intra_modulo=False)
                    legacy = ref.window_merge_and_roll_forward_index(
                        BATCH, H, W, sh, sw, wh, ww, legacy_intra_modulo=True)
                    self.assertTrue(torch.equal(explicit, legacy))


if __name__ == '__main__':
    unittest.main(verbosity=2)
