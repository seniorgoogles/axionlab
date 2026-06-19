#!/usr/bin/env python3
"""Verify the threshold structure of a dumped layer, so you can explain it.

For each neuron (row) it reports:
  - step       : spacing between consecutive thresholds
  - uniform    : are the spacings all equal? (arithmetic sequence)
  - 1/step     : the quantizer slope
  - pow2       : is 1/step a power of two? -> the quantizer is a bit-shift (no encoder)

    python3 scripts/check_thresholds.py path/to/MultiThreshold_0_*.txt

Reading:
  * MT_0 (input quant): one row, step constant, 1/step = 128 = 2^7  -> pow2 True
    => code = round(x*128), a shift, NO encoder needed.
  * MT_1/MT_2 (hidden): step constant PER row but differs per row, 1/step not a
    power of two => can't be a global shift => use the precomputed thresholds
    (thermometer encoder), which avoids a per-channel multiplier.
"""

import math
import sys

import numpy as np


def is_pow2(v, tol=1e-3):
    if v <= 0:
        return False, 0
    n = round(math.log2(v))
    return abs(v - 2 ** n) < tol * max(v, 1), n


def main():
    path = sys.argv[1]
    T = np.loadtxt(path, comments="#", ndmin=2)
    print(f"{path}: {T.shape[0]} neurons x {T.shape[1]} thresholds\n")
    steps = []
    for i, row in enumerate(T):
        d = np.diff(np.sort(row))
        smin, smed, smax = float(d.min()), float(np.median(d)), float(d.max())
        uniform = (smax - smin) <= 1.0 + 1e-6     # tolerate ±1 integer-rounding jitter
        inv = (1.0 / smed) if smed else float("inf")
        p2, n = is_pow2(inv)
        steps.append(smed)
        tag = f" (2^{n})" if p2 else ""
        if i < 5 or i == T.shape[0] - 1:           # print first few + last
            print(f"row {i:>3}: step~{smed:<10.6g} [min {smin:g}, max {smax:g}] "
                  f"uniform(±1)={uniform!s:<5} 1/step={inv:<10.6g} pow2={p2}{tag}")
    same_step = (max(steps) - min(steps)) <= 1.0 + 1e-6
    print(f"\nsame step across ALL neurons? {same_step}")
    print("=> " + ("global power-of-two scale -> a bit-shift, NO encoder needed"
                   if same_step and is_pow2(1.0 / steps[0])[0]
                   else "per-neuron / non-pow2 scale -> use the thermometer encoder"))


if __name__ == "__main__":
    main()
