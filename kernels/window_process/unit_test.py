# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Parity tests for the compiled kernels against the PyTorch ops they replace.
# Requires a CUDA device and the built extension; the index math itself is
# covered without either by test_index_math.py.
#
#   python unit_test.py
# --------------------------------------------------------

import unittest

import torch

import reference as ref

try:
    from window_process import WindowProcess, WindowProcessReverse
    EXTENSION_AVAILABLE = True
except ImportError:                                  # extension not built here
    WindowProcess = WindowProcessReverse = None
    EXTENSION_AVAILABLE = False


CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = unittest.skipIf(
    not (CUDA_AVAILABLE and EXTENSION_AVAILABLE),
    'requires a CUDA device and the compiled swin_window_process extension')


def available_dtypes():
    """float64 and float32 come from AT_DISPATCH_FLOATING_TYPES; the rest are added."""
    dtypes = [torch.float64, torch.float32, torch.float16]
    if CUDA_AVAILABLE and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


# (B, H, W, C, window_h, window_w)
SHAPES = [
    (2, 56, 56, 96, 7, 7),     # nH == nW == 8   the ImageNet configuration
    (2, 32, 16, 64, 8, 8),     # nH=4,  nW=2     out-of-bounds read before the fix
    (2, 16, 32, 64, 8, 8),     # nH=2,  nW=4     silent corruption before the fix
    (2, 16, 64, 64, 4, 16),    # nH=4,  nW=4     non-square window
    (2, 64, 16, 64, 16, 4),    # nH=4,  nW=4     non-square window, transposed
    (2, 24, 24, 32, 24, 24),   # nH == nW == 1   a single window
]


def shifts_for(window_h, window_w):
    """W-MSA (no shift) and SW-MSA (half window), the two regimes Swin uses."""
    return [(0, 0), (window_h // 2, window_w // 2)]


def pyt_forward(x, shift, window):
    """torch.roll followed by window_partition -- what WindowProcess fuses."""
    shift_h, shift_w = ref.to_pair(shift)
    if shift_h or shift_w:
        x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
    return ref.window_partition(x, window)


def reverse_pyt_forward(windows, shift, window, H, W):
    """window_reverse followed by torch.roll -- what WindowProcessReverse fuses."""
    shift_h, shift_w = ref.to_pair(shift)
    x = ref.window_reverse(windows, window, H, W)
    if shift_h or shift_w:
        x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))
    return x


def leaf(tensor, requires_grad=True):
    return tensor.clone().detach().requires_grad_(requires_grad).cuda()


def spatial(B, H, W, C, dtype):
    return torch.randn((B, H, W, C), dtype=dtype)


def windowed(B, H, W, C, window_h, window_w, dtype):
    n = B * (H // window_h) * (W // window_w)
    return torch.randn((n, window_h, window_w, C), dtype=dtype)


@requires_cuda
class TestWindowProcess(unittest.TestCase):
    """The kernels are exact permutations, so parity must be bit-for-bit.

    torch.equal is a stronger check than gradcheck here: no arithmetic is
    performed on the values, so any deviation is an indexing error, not a
    numerical one. Every dtype is therefore held to exact equality.
    """

    def _cases(self):
        for dtype in available_dtypes():
            for B, H, W, C, window_h, window_w in SHAPES:
                for shift in shifts_for(window_h, window_w):
                    yield dtype, B, H, W, C, (window_h, window_w), shift

    def test_partition_forward(self):
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = spatial(B, H, W, C, dtype)
                with torch.no_grad():
                    expected = pyt_forward(leaf(x), shift, window)
                    fused = WindowProcess.apply(
                        leaf(x), B, H, W, C, (-shift[0], -shift[1]), window)
                self.assertTrue(torch.equal(expected, fused))

    def test_partition_backward(self):
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = spatial(B, H, W, C, dtype)
                grad = windowed(B, H, W, C, window[0], window[1], dtype).cuda()

                a, b = leaf(x), leaf(x)
                pyt_forward(a, shift, window).backward(grad)
                WindowProcess.apply(
                    b, B, H, W, C, (-shift[0], -shift[1]), window).backward(grad)

                self.assertIsNotNone(a.grad)
                self.assertTrue(torch.equal(a.grad, b.grad))

    def test_merge_forward(self):
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = windowed(B, H, W, C, window[0], window[1], dtype)
                with torch.no_grad():
                    expected = reverse_pyt_forward(leaf(x), shift, window, H, W)
                    fused = WindowProcessReverse.apply(leaf(x), B, H, W, C, shift, window)
                self.assertTrue(torch.equal(expected, fused))

    def test_merge_backward(self):
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = windowed(B, H, W, C, window[0], window[1], dtype)
                grad = spatial(B, H, W, C, dtype).cuda()

                a, b = leaf(x), leaf(x)
                reverse_pyt_forward(a, shift, window, H, W).backward(grad)
                WindowProcessReverse.apply(b, B, H, W, C, shift, window).backward(grad)

                self.assertIsNotNone(a.grad)
                self.assertTrue(torch.equal(a.grad, b.grad))

    def test_round_trip(self):
        """WindowProcessReverse must undo WindowProcess for the same shift."""
        for dtype, B, H, W, C, window, shift in self._cases():
            with self.subTest(dtype=dtype, shape=(B, H, W, C), window=window, shift=shift):
                x = leaf(spatial(B, H, W, C, dtype), requires_grad=False)
                windows = WindowProcess.apply(
                    x, B, H, W, C, (-shift[0], -shift[1]), window)
                back = WindowProcessReverse.apply(
                    windows.contiguous(), B, H, W, C, shift, window)
                self.assertTrue(torch.equal(x, back))


@requires_cuda
class TestScalarWindowStillWorks(unittest.TestCase):
    """An int shift/window must behave exactly as the (h, w) pair it expands to.

    This is the compatibility path used by models/swin_transformer.py, which
    passes ints and must keep working unchanged.
    """

    def test_int_and_pair_agree(self):
        B, H, W, C, window, shift = 2, 56, 56, 96, 7, 3
        x = spatial(B, H, W, C, torch.float32)
        with torch.no_grad():
            as_int = WindowProcess.apply(leaf(x), B, H, W, C, -shift, window)
            as_pair = WindowProcess.apply(
                leaf(x), B, H, W, C, (-shift, -shift), (window, window))
        self.assertTrue(torch.equal(as_int, as_pair))


@requires_cuda
class TestPreconditions(unittest.TestCase):
    """Invalid arguments must raise instead of reading the wrong memory."""

    def setUp(self):
        self.B, self.H, self.W, self.C = 2, 16, 16, 32
        self.x = leaf(spatial(self.B, self.H, self.W, self.C, torch.float32), False)

    def _apply(self, **overrides):
        kwargs = dict(B=self.B, H=self.H, W=self.W, C=self.C, shift=(0, 0), window=(8, 8))
        kwargs.update(overrides)
        return WindowProcess.apply(
            self.x, kwargs['B'], kwargs['H'], kwargs['W'], kwargs['C'],
            kwargs['shift'], kwargs['window'])

    def test_window_must_divide_height(self):
        with self.assertRaisesRegex(RuntimeError, 'divisible by window_h'):
            self._apply(window=(5, 8))

    def test_window_must_divide_width(self):
        with self.assertRaisesRegex(RuntimeError, 'divisible by window_w'):
            self._apply(window=(8, 5))

    def test_shift_must_be_smaller_than_window(self):
        with self.assertRaisesRegex(RuntimeError, 'smaller than the window size'):
            self._apply(shift=(8, 0))

    def test_shape_must_match_tensor(self):
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            self._apply(C=self.C * 2)

    def test_non_contiguous_input_is_rejected(self):
        transposed = self.x.transpose(1, 2)
        with self.assertRaises(RuntimeError):
            WindowProcess.apply(
                transposed, self.B, self.W, self.H, self.C, (0, 0), (8, 8))


if __name__ == '__main__':
    if not (CUDA_AVAILABLE and EXTENSION_AVAILABLE):
        print('No CUDA device or extension not built: every test here will be skipped.')
        print('Run test_index_math.py to check the index math without a GPU.\n')
    torch.manual_seed(0)
    unittest.main(verbosity=2)
