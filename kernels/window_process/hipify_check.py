# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# Written by Francesco Brigante
# --------------------------------------------------------

"""Check that the CUDA sources translate cleanly to HIP for ROCm.

CUDAExtension hipifies its own sources when torch.version.hip is set, so a ROCm
build needs no second source tree. What it does need is a way to tell whether
that translation is complete. torch.utils.hipify is pure Python, so this answers
it with no AMD GPU -- with no GPU at all -- and runs on a CPU runner in CI.

Fails on a CUDA-only include that survives translation, and on a kernel launch
that reaches HIP without an explicit stream. Reports, without failing, the
symbols HIP keeps verbatim.

    python hipify_check.py            # verify
    python hipify_check.py --diff     # verify, and print the generated HIP
"""

import argparse
import difflib
import os
import re
import shutil
import sys
import tempfile

from torch.utils.hipify import hipify_python


SOURCES = ['swin_window_process.cpp', 'swin_window_process_kernel.cu']

# Includes a HIP build does not provide. If hipify leaves one of these behind,
# the ROCm compile fails outright, so a survivor here is a failure.
MUST_BE_TRANSLATED = [
    'cuda_runtime.h',
    'cuda_fp16.h',
    'ATen/cuda/CUDAContext.h',
    'c10/cuda/CUDAException.h',
]

# Symbols older hipify rewrites to an at::hip / C10_HIP spelling, and newer
# hipify (torch >= 2.9) leaves alone because they already reach the HIP runtime
# through the CUDA-compatibility headers. Both outcomes are correct, so this
# reports which one happened and never fails on it.
CUDA_COMPAT_SHIMS = [
    'at::cuda::getCurrentCUDAStream',
    'C10_CUDA_KERNEL_LAUNCH_CHECK',
]

# Symbols HIP spells and implements exactly as CUDA does. Rewriting one of these
# would be a bug in the translation, so a rewrite here is a failure.
# Printed on success. hipify works on symbols, and a symbol is not a signature.
SYMBOL_LEVEL_CAVEAT = """\
Symbol level only: a symbol can translate and still not compile, because HIP and
CUDA do not always give it the same overload set. __ldg is the case in point --
it has no HIP overload for c10::Half. Only a real build finds that."""

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


def read(path):
    with open(path) as handle:
        return handle.read()


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
    """Classify each source's device-specific symbols and return the failures."""
    failures = []

    for name in SOURCES:
        original = os.path.join(staging, name)
        hipified = mapping.get(name)
        if hipified is None or not os.path.exists(hipified):
            failures.append(f'{name}: hipify produced no output')
            continue

        before = read(original)
        after = read(hipified)

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
            how = ('rewritten' if token not in after
                   else 'kept (resolves via the CUDA-compat headers)')
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
    """Every kernel launch must still carry an explicit stream after translation.

    A launch that reaches HIP on the null stream is the ROCm form of the
    default-stream bug.

    The spelling of a launch is not fixed, so this accepts all of it: hipify may
    turn the triple chevron into hipLaunchKernelGGL or leave it alone, may wrap
    the call over several lines, and may or may not rename getCurrentCUDAStream.
    Only one thing is rejected -- a literal 0 where the stream belongs.
    """
    hipified = mapping.get('swin_window_process_kernel.cu')
    if hipified is None:
        return ['no hipified kernel source to inspect']

    text = read(hipified)
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    print(SYMBOL_LEVEL_CAVEAT)


if __name__ == '__main__':
    main()
