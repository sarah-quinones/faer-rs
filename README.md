# faer

`faer` is a collection of crates that implement low level linear algebra routines in pure Rust.
The aim is to eventually provide a fully featured library for linear algebra with focus on portability, correctness, and performance.

See the Wiki and the `docs.rs` documentation for code examples and usage instructions.

Questions about using the library, contributing, and future directions can be discussed in the [Discord server](https://discord.gg/Ak5jDsAFVZ).

[![Documentation](https://docs.rs/faer-core/badge.svg)](https://docs.rs/faer-core)
[![Crate](https://img.shields.io/crates/v/faer-core.svg)](https://crates.io/crates/faer-core)

## faer-core

The core module implements matrix structures, as well as BLAS-like matrix operations such as matrix multiplication and solving triangular linear systems.

## faer-cholesky (WIP)

The Cholesky module implements the LLT and LDLT matrix decompositions. These allow for solving symmetric/hermitian (+positive definite for LLT) linear systems.

## faer-qr (WIP)

The QR module implements the QR decomposition with no pivoting, as well as the version with column pivoting.

## faer-lu (WIP)

The LU module implements the LU factorization with no pivoting, as well as the versions with row pivoting and full pivoting.

## Coming soon

- `faer-svd`
- `faer-eigen`

# Benchmarks

The benchmarks were run on an `11th Gen Intel(R) Core(TM) i5-11400 @ 2.60GHz`.

## Matrix multiplication

Multiplication of two square matrices of size `n`.
```
        faer (serial)      faer (parallel)   ndarray (openblas)      nalgebra (matrixmultiply)
   32           2.5µs                1.5µs                1.5µs                          2.8µs
   64          15.2µs                 10µs                8.1µs                         16.8µs
   96          44.2µs               18.2µs               26.2µs                         51.3µs
  128         102.7µs               19.8µs               36.7µs                        117.3µs
  192         334.8µs                 76µs               50.6µs                        381.1µs
  256         708.8µs              171.6µs              153.5µs                        746.3µs
  384           1.8ms                378µs              343.1µs                            2ms
  512           4.2ms              940.7µs                  1ms                          4.8ms
  640           8.7ms                2.3ms                1.9ms                          9.4ms
  768          15.2ms                3.7ms                2.9ms                         16.2ms
  896          24.4ms                6.2ms                5.3ms                           26ms
 1024          38.2ms                  9ms                7.4ms                         39.1ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of size `n`.
```
        faer (serial)      faer (parallel)   ndarray (openblas)      nalgebra (matrixmultiply)
   32           2.9µs                2.7µs               36.4µs                          8.6µs
   64          10.4µs               10.4µs               25.5µs                         50.4µs
   96          29.9µs               30.6µs               44.1µs                        158.6µs
  128            59µs               46.1µs              121.1µs                        375.3µs
  192         204.8µs              113.9µs              209.6µs                          971µs
  256         407.8µs              162.5µs              520.4µs                            2ms
  384           1.2ms              307.8µs                1.2ms                          6.8ms
  512           2.6ms              655.6µs                  3ms                         15.7ms
  640           4.9ms                1.4ms                4.9ms                         30.4ms
  768           8.2ms                2.1ms                8.7ms                         53.6ms
  896          12.7ms                3.6ms               11.3ms                         84.6ms
 1024          19.5ms                5.1ms               21.4ms                        125.9ms
```
