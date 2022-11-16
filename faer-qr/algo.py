import numpy as np
import scipy.linalg as la

n = 3


def qr(a):
    a = a.copy()
    m, n = a.shape
    q = np.eye(m, dtype=a.dtype)
    for i in range(n - (m == n)):
        h = np.eye(m, dtype=a.dtype)
        h[i:, i:] = make_householder(a[i:, i])
        q = q @ h
        a = h @ a
    return q, a


def make_householder(a):
    sign = a[0] / np.abs(a[0])

    v = a / (a[0] + sign * np.linalg.norm(a))
    v[0] = 1
    tau = 2 / (v.conj().T @ v)

    h = np.eye(a.shape[0], dtype=a.dtype)
    h -= tau * (v[:, None] @ v[:, None].conj().T)
    return h


a = np.random.randn(n, n)
a = np.random.randn(n, n) + 1.0j * np.random.randn()

with np.printoptions(precision=3):
    q, r = qr(a)
    print()
    print(a)
    print(r)
    print(q.conj().T @ q)
    print(q @ r - a)
