# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
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
# It fails if any CUDA-specific symbol survives translation, and prints the
# symbols that are deliberately left alone because HIP implements them natively.
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

# Present in the CUDA source and required to disappear from the HIP output.
# A survivor here means hipify has no rule for it and the ROCm build would fail
# to compile, or would silently bind to the wrong runtime.
MUST_BE_TRANSLATED = [
    'cuda_runtime.h',
    'cuda_fp16.h',
    'ATen/cuda/CUDAContext.h',
    'c10/cuda/CUDAException.h',
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

        portable = [t for t in KNOWN_PORTABLE if t in before]
        for token in portable:
            if token in after:
                print(f'  portable    {token}  (native in HIP, left as is)')
            else:
                failures.append(f'{name}: {token} was rewritten but should be portable')

        if not translated and not portable:
            print('  nothing device specific in this file')

        if show_diff:
            diff = difflib.unified_diff(
                before.splitlines(True), after.splitlines(True),
                fromfile=name, tofile=os.path.basename(hipified))
            sys.stdout.writelines(diff)

    return failures


def check_launch_form(mapping):
    """Every kernel launch must carry an explicit stream argument after translation.

    hipLaunchKernelGGL takes (kernel, grid, block, sharedMem, stream, ...). A
    launch that reaches HIP without a stream runs on the null stream, which is
    the ROCm form of the default-stream bug.
    """
    hipified = mapping.get('swin_window_process_kernel.cu')
    if hipified is None:
        return ['no hipified kernel source to inspect']

    text = open(hipified).read()
    launches = re.findall(r'hipLaunchKernelGGL\((.*?)\n', text)
    if not launches:
        return ['no hipLaunchKernelGGL call found in the translated source']

    failures = []
    for launch in launches:
        if 'getCurrentHIPStream' not in launch:
            failures.append(f'launch without an explicit stream: {launch.strip()[:70]}')
    print(f'\n{len(launches)} kernel launches, all carrying an explicit HIP stream'
          if not failures else '')
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


if __name__ == '__main__':
    main()
