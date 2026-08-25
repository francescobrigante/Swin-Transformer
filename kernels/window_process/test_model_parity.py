# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# End-to-end check: a SwinTransformer must produce identical output with and
# without the fused window kernels. The kernels replace torch.roll plus
# window_partition, which are exact data movements, so the two paths are not
# merely close -- they are bit-for-bit equal.
#
#   python test_model_parity.py
# --------------------------------------------------------

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    import swin_window_process  # noqa: F401
    EXTENSION_AVAILABLE = True
except ImportError:
    EXTENSION_AVAILABLE = False

try:
    from models.swin_transformer import SwinTransformer
    MODEL_AVAILABLE = True
except ImportError:                                  # timm missing, for instance
    SwinTransformer = None
    MODEL_AVAILABLE = False


requires_everything = unittest.skipIf(
    not (torch.cuda.is_available() and EXTENSION_AVAILABLE and MODEL_AVAILABLE),
    'requires a CUDA device, the compiled extension and an importable model')


def set_fused(model, enabled):
    """Flip the flag on every block, so both runs share exactly the same weights."""
    switched = 0
    for module in model.modules():
        if hasattr(module, 'fused_window_process'):
            module.fused_window_process = enabled
            switched += 1
    return switched


@requires_everything
class TestModelParity(unittest.TestCase):

    def _model(self, img_size=56, window_size=7):
        torch.manual_seed(0)
        model = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=10,
            embed_dim=48,
            depths=[2, 2],
            num_heads=[3, 6],
            window_size=window_size,
            drop_path_rate=0.0,
            fused_window_process=False,
        )
        return model.cuda().eval()

    def test_forward_is_identical(self):
        model = self._model()
        x = torch.randn(2, 3, 56, 56, device='cuda')

        with torch.no_grad():
            eager = model(x)
            blocks = set_fused(model, True)
            fused = model(x)
            set_fused(model, False)

        # The shifted blocks are the ones that take the fused path at all.
        self.assertGreater(blocks, 0, 'no block exposes fused_window_process')
        self.assertTrue(torch.equal(eager, fused))

    def test_gradients_are_identical(self):
        model = self._model()
        x = torch.randn(2, 3, 56, 56, device='cuda')
        target = torch.randn(2, 10, device='cuda')

        def grads():
            model.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(x), target).backward()
            return [p.grad.clone() for p in model.parameters() if p.grad is not None]

        eager = grads()
        set_fused(model, True)
        fused = grads()
        set_fused(model, False)

        self.assertEqual(len(eager), len(fused))
        for a, b in zip(eager, fused):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    if not (torch.cuda.is_available() and EXTENSION_AVAILABLE and MODEL_AVAILABLE):
        print('Skipping: needs a CUDA device, the built extension and the model.\n')
    unittest.main(verbosity=2)
