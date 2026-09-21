# SPDX-License-Identifier: Apache-2.0
"""Compile-time tile selection. No device values enter this policy."""


def select_tiles(hidden, intermediate, rows, block_m=None, block_n=None, block_k=None):
    """Return (M, N, K) scheduling groups; each scaled product stays 128x128.

    Defaults bound the live gate weight group to 8 MiB and use a conservative
    24 MiB workspace estimate. Explicit settings are checked for ISA/scale
    alignment; the compiler remains responsible for the final SBUF allocation.
    """
    if min(hidden, intermediate, rows) < 1 or hidden % 128 or intermediate % 128:
        raise ValueError("Require positive rows and H/I multiples of128")
    k = 4096 if block_k is None else block_k
    if not isinstance(k, int) or k <= 0 or k % 128:
        raise ValueError("BLOCK_K must be a positive multiple of128")
    n = max(hidden, 2 * intermediate) if block_n is None else block_n
    if block_n is None:
        while n > 128 and 2 * min(n, 2 * intermediate) * min(k, hidden) > 8 * 1024**2:
            n = max(128, (n // 256) * 128)
    if not isinstance(n, int) or n <= 0 or n % 128:
        raise ValueError("BLOCK_N must be a positive multiple of128")
    m = min(rows, 512) if block_m is None else block_m
    if block_m is None:
        weight_bytes = 2 * min(n, 2 * intermediate) * min(k, hidden)
        scale_bytes = 4 * 3 * (intermediate // 128) * (hidden // 128) * 128
        while m > 1 and ((6 * hidden + 10 * intermediate) * m +
                         6 * hidden * min(m, 128) + weight_bytes + scale_bytes > 24 * 1024**2):
            m = max(1, m // 2)
    if not isinstance(m, int) or not 1 <= m <= 512:
        raise ValueError("BLOCK_M must be an integer from1 through512")
    return m, n, k
