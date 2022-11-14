import numpy as np
import scipy.linalg as la
import timeit
from typing import Type


def format_float_duration(seconds):
    if seconds < 1e-6:
        return "{:7.2f} ns".format(seconds * 1e9)
    elif seconds < 1e-3:
        return "{:7.2f} µs".format(seconds * 1e6)
    elif seconds < 1e-0:
        return "{:7.2f} ms".format(seconds * 1e3)
    else:
        return "{:7.2f} s".format(seconds * 1e0)


def cholesky(n: int, dtype: Type = np.float64) -> float:
    a = np.eye(n, dtype=dtype)
    return timeit.timeit(lambda: la.cho_factor(a.T), number=10)


def partial_lu(n: int, dtype: Type = np.float64) -> float:
    a = np.eye(n, dtype=dtype)
    return timeit.timeit(lambda: la.lu_factor(a.T), number=1)


def full_lu(n: int, dtype: Type = np.float64) -> float:
    a = np.eye(n, dtype=dtype)
    if dtype == np.float32:
        return timeit.timeit(lambda: la.lapack.sgetc2(a.T), number=1)
    elif dtype == np.float64:
        return timeit.timeit(lambda: la.lapack.dgetc2(a.T), number=1)
    elif dtype == np.complex64:
        return timeit.timeit(lambda: la.lapack.cgetc2(a.T), number=1)
    elif dtype == np.complex128:
        return timeit.timeit(lambda: la.lapack.zgetc2(a.T), number=1)
    else:
        return np.NaN


sizes = [128, 256, 512, 1024, 4096]
dtypes = [np.float32, np.float64, np.complex64, np.complex128]

for dtype in dtypes:
    print("cholesky", dtype.__name__)
    for n in sizes:
        print(f"{n:5}: {format_float_duration(cholesky(n, dtype))}")
    print()
print()

for dtype in dtypes:
    print("partial pivoting lu", dtype.__name__)
    for n in sizes:
        print(f"{n:5}: {format_float_duration(partial_lu(n, dtype))}")
    print("")
print()

for dtype in dtypes:
    print("full pivoting lu", dtype.__name__)
    for n in sizes[:-2]:
        print(f"{n:5}: {format_float_duration(full_lu(n, dtype))}")
    print("")
print()
