# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# Written by Francesco Brigante
# --------------------------------------------------------
# Verifies that the CUDA sources translate cleanly to HIP for ROCm.
#
# torch.utils.cpp_extension.CUDAExtension hipifies its sources automatically
# when torch.version.hip is set, so building on ROCm needs no separate source
# tree and no change to setup.py. What it does need is a way to tell whether the
# translation is complete, and that check must not require an AMD GPU -- or any
# GPU. torch.utils.hipify is pure Python, so this runs anywhere, including in CI
# on a CPU runner.
#
# It fails if a CUDA-only include survives translation, or if a kernel launch
# reaches HIP without an explicit stream (the null-stream bug). It only reports,
# without failing, the symbols HIP keeps as is -- either because it implements
# them natively (blockIdx, __ldg, dim3, ...) or because they resolve through the
# CUDA-compatibility headers (at::cuda::getCurrentCUDAStream on torch >= 2.9).
#
#   python hipify_check.py            # verify
#   python hipify_check.py --diff     # verify and show the generated HIP source
# --------------------------------------------------------

import argparse
import difflib
import os
import re
import shutil
import sys
import tempfile

from torch.utils.hipify import hipify_python


SOURCES = ['swin_window_process.cpp', 'swin_window_process_kernel.cu']

# Includes a HIP build genuinely does not provide: these must be rewritten, or
# the ROCm compile fails outright. hipify has rewritten them in every torch
# version.
MUST_BE_TRANSLATED = [
    'cuda_runtime.h',
    'cuda_fp16.h',
    'ATen/cuda/CUDAContext.h',
    'c10/cuda/CUDAException.h',
]

# API symbols older hipify rewrites to an at::hip / C10_HIP spelling, and newer
# hipify (torch >= 2.9) deliberately leaves alone because they resolve to the HIP
# runtime through the CUDA-compatibility headers. Either outcome is correct, so
# this only reports which one happened -- it is never a failure.
CUDA_COMPAT_SHIMS = [
    'at::cuda::getCurrentCUDAStream',
    'C10_CUDA_KERNEL_LAUNCH_CHECK',
]

# Deliberately unchanged: HIP implements these with the same spelling and the
# same semantics, so translating them would be wrong.
KNOWN_PORTABLE = [
    '__global__',
    '__ldg',
    'blockIdx',
    'threadIdx',
    'blockDim',
    'gridDim',
    'dim3',
    'AT_DISPATCH_FLOATING_TYPES_AND2',
    'at::ScalarType::BFloat16',
]


def hipify_into(destination):
    """Translate SOURCES out of place and return {source: hipified path}."""
    here = os.path.dirname(os.path.abspath(__file__))
    staging = os.path.join(destination, 'src')
    os.makedirs(staging)
    for name in SOURCES:
        shutil.copy(os.path.join(here, name), staging)

    results = hipify_python.hipify(
        project_directory=staging,
        output_directory=os.path.join(destination, 'out'),
        includes=('*',),
        is_pytorch_extension=True,
        out_of_place_only=True,
        show_progress=False,
    )

    mapping = {}
    for source, result in results.items():
        hipified = getattr(result, 'hipified_path', None) or source
        mapping[os.path.basename(source)] = hipified
    return staging, mapping


def report(staging, mapping, show_diff):
    failures = []

    for name in SOURCES:
        original = os.path.join(staging, name)
        hipified = mapping.get(name)
        if hipified is None or not os.path.exists(hipified):
            failures.append(f'{name}: hipify produced no output')
            continue

        before = open(original).read()
        after = open(hipified).read()

        print(f'\n{name} -> {os.path.basename(hipified)}')

        translated, survived = [], []
        for token in MUST_BE_TRANSLATED:
            if token not in before:
                continue
            (survived if token in after else translated).append(token)

        for token in translated:
            print(f'  translated  {token}')
        for token in survived:
            print(f'  SURVIVED    {token}')
            failures.append(f'{name}: {token} was not translated')

        shims = [t for t in CUDA_COMPAT_SHIMS if t in before]
        for token in shims:
            how = 'rewritten' if token not in after else 'kept (resolves via the CUDA-compat headers)'
            print(f'  shim        {token}  ({how})')

        portable = [t for t in KNOWN_PORTABLE if t in before]
        for token in portable:
            if token in after:
                print(f'  portable    {token}  (native in HIP, left as is)')
            else:
                failures.append(f'{name}: {token} was rewritten but should be portable')

        if not translated and not shims and not portable:
            print('  nothing device specific in this file')

        if show_diff:
            diff = difflib.unified_diff(
                before.splitlines(True), after.splitlines(True),
                fromfile=name, tofile=os.path.basename(hipified))
            sys.stdout.writelines(diff)

    return failures


def check_launch_form(mapping):
    """Every kernel launch must carry an explicit stream argument after translation.

    A launch that reaches HIP on the null stream is the ROCm form of the
    default-stream bug. hipify may rewrite ``kernel<<<...>>>(...)`` to
    ``hipLaunchKernelGGL(...)`` or (newer HIP accepts the triple chevron) leave
    it as is, and it may wrap the call across several lines; and it may or may
    not rewrite ``getCurrentCUDAStream`` to ``getCurrentHIPStream...``. Accept
    every combination -- what matters is that a real stream getter is passed,
    not ``0``.
    """
    hipified = mapping.get('swin_window_process_kernel.cu')
    if hipified is None:
        return ['no hipified kernel source to inspect']

    text = open(hipified).read()
    calls = re.findall(r'hipLaunchKernelGGL\s*\((.*?)\)\s*;', text, re.DOTALL)
    calls += re.findall(r'<<<(.*?)>>>', text, re.DOTALL)
    if not calls:
        return ['no kernel launch found in the translated source']

    stream_getters = ('getCurrentHIPStream', 'getCurrentCUDAStream')
    failures = []
    for call in calls:
        if not any(g in call for g in stream_getters):
            failures.append('launch without an explicit stream: '
                            + ' '.join(call.split())[:80])
    if not failures:
        print(f'\n{len(calls)} kernel launches, all carrying an explicit stream')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diff', action='store_true',
                        help='print the generated HIP source as a unified diff')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        staging, mapping = hipify_into(tmp)
        failures = report(staging, mapping, args.diff)
        failures += check_launch_form(mapping)

    if failures:
        print('\nFAILED')
        for failure in failures:
            print(f'  {failure}')
        raise SystemExit(1)

    print('\nOK: the CUDA sources translate to HIP with no unmapped symbols.')
    print('Symbol level only: a symbol can translate and still not compile, '
          'because\nHIP and CUDA do not always give it the same overload set. '
          '__ldg is the\ncase in point -- it has no HIP overload for c10::Half. '
          'Only a real build\nfinds that.')


if __name__ == '__main__':
    main()
