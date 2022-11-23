# faer

`faer` is a collection of crates that implement low level linear algebra routines in pure Rust.
The aim is to eventually provide a fully featured library for linear algebra with focus on portability, correctness, and performance.

See the Wiki and the `docs.rs` documentation for code examples and usage instructions.

Questions about using the library, contributing, and future directions can be discussed in the [Discord server](https://discord.gg/Ak5jDsAFVZ).

## faer-core

[![Documentation](https://docs.rs/faer-core/badge.svg)](https://docs.rs/faer-core)
[![Crate](https://img.shields.io/crates/v/faer-core.svg)](https://crates.io/crates/faer-core)

The core module implements matrix structures, as well as BLAS-like matrix operations such as matrix multiplication and solving triangular linear systems.

## faer-cholesky (WIP)

The Cholesky module implements the LLT and LDLT matrix decompositions. These allow for solving symmetric/hermitian (+positive definite for LLT) linear systems.

## faer-qr (WIP)

The QR module implements the QR decomposition with no pivoting, as well as the version with column pivoting.

## faer-lu

[![Documentation](https://docs.rs/faer-lu/badge.svg)](https://docs.rs/faer-lu)
[![Crate](https://img.shields.io/crates/v/faer-lu.svg)](https://crates.io/crates/faer-lu)

The LU module implements the LU factorization with row pivoting, as well as the version with full pivoting.

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

Solving `AX = B` in place where `A` and `B` are two square matrices of size `n`, and `A` is a triangular matrix.

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

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix.

```
        faer (serial)      faer (parallel)   ndarray (openblas)      nalgebra (matrixmultiply)
   32           3.2µs               31.3µs                  8µs                          8.4µs
   64           9.8µs                 37µs               23.1µs                         50.4µs
   96          24.2µs               59.2µs               46.6µs                        159.2µs
  128            37µs               62.4µs              134.5µs                        386.6µs
  192          96.5µs                 92µs              217.4µs                        954.2µs
  256           186µs              140.9µs              506.8µs                          1.9ms
  384           534µs              249.6µs                1.2ms                          7.2ms
  512           1.1ms              414.8µs                3.1ms                         17.4ms
  640           1.9ms              567.3µs                5.1ms                         32.6ms
  768           3.2ms              837.2µs                8.5ms                         56.8ms
  896           4.8ms                1.2ms               11.3ms                         88.3ms
 1024           7.2ms                1.8ms               21.5ms                        136.1ms
```

## LU decomposition with partial pivoting

Factorizing a matrix as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.
```
        faer (serial)      faer (parallel)   ndarray (openblas)      nalgebra (matrixmultiply)
   32           6.2µs                5.1µs                9.5µs                          7.1µs
   64          18.2µs               18.1µs               19.3µs                         37.1µs
   96          40.6µs               40.4µs               37.9µs                        109.3µs
  128          78.7µs               83.1µs                1.3ms                        250.4µs
  192         196.4µs                312µs                210µs                        821.2µs
  256           399µs              466.5µs              324.3µs                            2ms
  384           1.1ms              986.6µs              676.7µs                          6.7ms
  512           2.4ms                1.7ms                1.2ms                         11.4ms
  640           4.2ms                2.8ms                1.8ms                         21.2ms
  768             7ms                  4ms                2.7ms                         36.5ms
  896          10.4ms                5.5ms                4.2ms                         57.9ms
 1024          15.6ms                8.2ms                5.4ms                         91.2ms
```

## LU decomposition with full pivoting

Factorizing a matrix as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.
```
        faer (serial)      faer (parallel)   ndarray (openblas)      nalgebra (matrixmultiply)
   32          13.3µs              733.4µs          UNAVAILABLE                         15.9µs
   64          44.1µs              595.8µs          UNAVAILABLE                        111.3µs
   96         109.7µs              682.3µs          UNAVAILABLE                        367.5µs
  128           229µs              974.1µs          UNAVAILABLE                        831.2µs
  192         578.9µs                1.7ms          UNAVAILABLE                          2.8ms
  256           1.3ms                2.6ms          UNAVAILABLE                          6.5ms
  384           4.3ms                4.9ms          UNAVAILABLE                         22.1ms
  512          10.8ms                8.1ms          UNAVAILABLE                         53.4ms
  640          18.9ms                 13ms          UNAVAILABLE                        102.7ms
  768          32.2ms               18.3ms          UNAVAILABLE                        177.2ms
  896          49.3ms               26.1ms          UNAVAILABLE                        281.6ms
 1024          78.6ms               35.7ms          UNAVAILABLE                        430.1ms
```
