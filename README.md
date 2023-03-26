# faer

`faer` is a collection of crates that implement low level linear algebra routines in pure Rust.
The aim is to eventually provide a fully featured library for linear algebra with focus on portability, correctness, and performance.

See the [official website](https://faer-rs.github.io) and the `docs.rs` documentation for code examples and usage instructions.

Questions about using the library, contributing, and future directions can be discussed in the [Discord server](https://discord.gg/Ak5jDsAFVZ).

## faer-core

[![Documentation](https://docs.rs/faer-core/badge.svg)](https://docs.rs/faer-core)
[![Crate](https://img.shields.io/crates/v/faer-core.svg)](https://crates.io/crates/faer-core)

The core module implements matrix structures, as well as BLAS-like matrix operations such as matrix multiplication and solving triangular linear systems.

## faer-cholesky

[![Documentation](https://docs.rs/faer-cholesky/badge.svg)](https://docs.rs/faer-cholesky)
[![Crate](https://img.shields.io/crates/v/faer-cholesky.svg)](https://crates.io/crates/faer-cholesky)

The Cholesky module implements the LLT and LDLT matrix decompositions. These allow for solving symmetric/hermitian (+positive definite for LLT) linear systems.

## faer-lu

[![Documentation](https://docs.rs/faer-lu/badge.svg)](https://docs.rs/faer-lu)
[![Crate](https://img.shields.io/crates/v/faer-lu.svg)](https://crates.io/crates/faer-lu)

The LU module implements the LU factorization with row pivoting, as well as the version with full pivoting.

## faer-qr

[![Documentation](https://docs.rs/faer-qr/badge.svg)](https://docs.rs/faer-qr)
[![Crate](https://img.shields.io/crates/v/faer-qr.svg)](https://crates.io/crates/faer-qr)

The QR module implements the QR decomposition with no pivoting, as well as the version with column pivoting.

## Coming soon

- `faer-svd`
- `faer-eigen`

# Contributing

If you'd like to contribute to `faer`, check out the list of "good first issue"
issues. These are all (or should be) issues that are suitable for getting
started, and they generally include a detailed set of instructions for what to
do. Please ask questions on the Discord server or the issue itself if anything
is unclear!

# Benchmarks

The benchmarks were run on an `11th Gen Intel(R) Core(TM) i5-11400 @ 2.60GHz` with 12 threads.  
- `nalgebra` is used with the `matrixmultiply` backend
- `ndarray` is used with the `openblas` backend
- `eigen` is compiled with `-march=native -O3 -fopenmp`

All computations are done on column major matrices containing `f64` values.

## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      1.2µs      1.1µs      1.1µs      1.9µs      1.2µs
   64      8.1µs      8.1µs      7.8µs       11µs      5.1µs
   96     27.7µs     10.9µs     26.1µs     33.9µs       10µs
  128     65.6µs     17.2µs     35.3µs     78.6µs     32.7µs
  192      218µs     54.2µs       54µs    258.3µs     51.7µs
  256    512.8µs    118.5µs    156.8µs    606.9µs    142.9µs
  384      1.7ms    369.1µs    407.6µs        2ms    330.4µs
  512      4.1ms    853.1µs      1.3ms      4.7ms      1.2ms
  640        8ms      1.7ms      2.3ms      9.3ms        2ms
  768     13.9ms        3ms      3.6ms     16.1ms      3.2ms
  896     22.2ms      4.9ms      6.5ms     25.8ms      5.9ms
 1024       34ms      7.2ms      8.9ms     39.2ms      8.3ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.5µs      2.4µs      8.2µs      7.2µs        3µs
   64     10.2µs     10.3µs     25.8µs     34.7µs     13.6µs
   96     29.5µs     26.6µs     54.4µs    102.4µs     36.8µs
  128     59.1µs     40.8µs    145.2µs    239.2µs       83µs
  192      174µs     93.3µs    263.4µs    832.8µs    214.3µs
  256    382.1µs    164.3µs    625.9µs      1.9ms    485.9µs
  384      1.2ms    328.7µs      1.4ms      6.9ms      1.4ms
  512      2.7ms    683.1µs      3.6ms     15.6ms      3.2ms
  640      4.8ms      1.7ms      5.7ms     29.9ms      5.4ms
  768      8.2ms      2.5ms      9.3ms     53.1ms      9.1ms
  896     12.5ms      3.6ms     13.5ms     83.8ms     13.8ms
 1024     19.1ms      5.3ms     25.2ms    127.7ms     22.9ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.3µs     10.8µs      8.2µs      7.2µs        3µs
   64     10.7µs     17.7µs     25.8µs     34.7µs     13.6µs
   96     25.5µs     37.3µs     54.1µs    102.6µs     36.7µs
  128     39.5µs     51.1µs    145.1µs    241.7µs       83µs
  192    100.3µs     86.9µs    263.4µs    832.7µs    214.1µs
  256    188.6µs    136.6µs      633µs      1.9ms    485.4µs
  384    519.1µs    268.1µs      1.4ms        7ms      1.4ms
  512      1.1ms    450.2µs      3.6ms     15.6ms      3.2ms
  640        2ms    654.2µs      5.7ms     29.9ms      5.4ms
  768      3.2ms    923.6µs      9.3ms     53.2ms      9.1ms
  896      4.8ms      1.5ms     13.6ms     83.8ms     13.8ms
 1024      7.3ms      2.5ms     25.3ms    127.5ms     22.9ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.6µs      4.6µs      3.2µs      2.3µs      2.2µs
   64     12.7µs     12.7µs     38.4µs     11.1µs      8.7µs
   96     28.9µs     28.9µs    100.6µs     32.1µs     20.2µs
  128     41.7µs     41.8µs    282.3µs     77.6µs     38.2µs
  192    110.9µs    116.6µs    303.7µs    252.6µs       99µs
  256    189.8µs    178.5µs    696.7µs    617.8µs    204.4µs
  384    509.6µs    483.2µs      1.2ms      2.1ms    562.1µs
  512      1.2ms    693.1µs      3.7ms        6ms      1.2ms
  640        2ms      1.3ms      3.3ms     11.3ms      2.1ms
  768      3.4ms      1.9ms      5.4ms     19.8ms      3.6ms
  896      5.2ms      2.7ms      6.9ms     30.9ms      5.5ms
 1024      8.1ms      3.5ms     14.7ms     46.4ms      8.3ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      5.3µs      5.4µs      5.6µs        5µs      3.8µs
   64     24.1µs     24.2µs     17.4µs     22.4µs     15.2µs
   96     53.2µs     53.2µs     34.3µs     67.8µs     36.2µs
  128    109.6µs    108.6µs     97.1µs    160.2µs    128.6µs
  192    270.7µs    287.5µs    186.7µs    496.6µs    423.8µs
  256    524.7µs    588.1µs    315.9µs      1.3ms    825.8µs
  384      1.4ms      1.3ms    672.2µs      4.6ms      1.8ms
  512        3ms      2.6ms      1.9ms     11.3ms      4.3ms
  640      5.1ms      4.1ms      2.8ms       21ms      5.6ms
  768      8.3ms        6ms      3.3ms     36.1ms      8.7ms
  896       12ms      9.2ms      4.7ms     56.9ms     11.4ms
 1024     17.7ms     12.3ms      6.8ms     90.1ms     17.3ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.8µs     10.8µs          -     14.4µs     10.4µs
   64     44.5µs     44.5µs          -    104.9µs     67.5µs
   96    109.7µs    110.5µs          -    344.8µs    196.4µs
  128    228.9µs    228.2µs          -      823µs    452.6µs
  192    589.7µs    589.2µs          -      2.7ms      1.4ms
  256      1.3ms      1.3ms          -      6.6ms      3.3ms
  384      4.6ms      4.4ms          -       22ms     10.7ms
  512     11.8ms      8.7ms          -     52.6ms     26.6ms
  640     20.1ms     12.9ms          -    101.4ms     49.3ms
  768     33.4ms     18.1ms          -    175.3ms     85.1ms
  896     52.3ms     25.9ms          -    277.6ms    133.8ms
 1024     80.1ms     34.2ms          -    438.1ms    203.3ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.3µs     10.3µs     15.5µs        8µs      7.2µs
   64     32.8µs     32.8µs     60.3µs     43.7µs     44.9µs
   96     68.7µs     68.6µs    319.7µs    141.4µs     78.9µs
  128    123.5µs    123.3µs    806.9µs      326µs    157.1µs
  192    323.2µs    322.9µs      1.7ms        1ms    383.4µs
  256    661.5µs    713.2µs      4.9ms      2.5ms    797.5µs
  384      1.9ms      1.7ms      8.1ms      8.2ms      2.1ms
  512      4.2ms      3.2ms     16.4ms     19.5ms      4.5ms
  640      7.6ms      4.7ms     22.9ms     37.7ms      8.2ms
  768     12.5ms        7ms     41.8ms     63.3ms     13.5ms
  896     19.1ms      9.8ms     56.2ms    100.8ms     20.8ms
 1024     28.4ms     13.9ms     80.5ms    153.1ms     30.9ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     12.4µs     25.7µs          -       18µs      9.3µs
   64     50.8µs     71.4µs          -    128.5µs     39.1µs
   96    120.4µs    147.2µs          -    423.3µs    102.1µs
  128    241.2µs    272.9µs          -    996.2µs    218.8µs
  192      675µs    729.1µs          -      3.3ms      626µs
  256      1.5ms      1.5ms          -      7.7ms      1.7ms
  384      5.1ms      3.7ms          -     25.5ms      5.5ms
  512     12.1ms        7ms          -       60ms     14.2ms
  640     23.4ms     10.8ms          -    116.6ms     25.4ms
  768     39.5ms     15.1ms          -    199.9ms       44ms
  896     63.2ms     24.2ms          -    318.4ms     68.6ms
 1024     93.2ms     37.6ms          -    494.3ms    108.2ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     14.4µs     32.9µs     10.6µs     21.3µs     10.8µs
   64     55.3µs     79.6µs     37.8µs     98.6µs     46.7µs
   96    138.2µs    144.7µs    162.8µs    284.4µs    118.8µs
  128    227.1µs      216µs    487.6µs    646.1µs    342.4µs
  192    606.2µs    509.8µs    658.8µs      2.2ms    969.2µs
  256      1.2ms      843µs      1.2ms      5.5ms        2ms
  384      3.2ms        2ms      2.3ms       20ms      5.1ms
  512      6.9ms      3.8ms      4.5ms     47.8ms       12ms
  640     12.7ms      6.5ms      7.3ms     90.9ms     19.3ms
  768     21.4ms     10.4ms       11ms    154.1ms     31.2ms
  896       32ms       16ms     16.7ms      241ms     44.2ms
 1024     47.7ms     21.7ms     23.8ms    375.3ms     69.5ms
```

## Matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32    115.6µs    138.3µs    124.7µs     94.2µs    219.3µs
   64    409.7µs    374.9µs    665.4µs    576.7µs        1ms
   96    913.3µs    905.7µs      3.4ms      1.7ms      2.6ms
  128      1.7ms      1.6ms      9.8ms      4.5ms      4.3ms
  192        4ms      4.1ms     28.1ms     15.1ms      9.6ms
  256      7.8ms      7.1ms     75.5ms     45.3ms     17.3ms
  384     21.1ms     15.4ms    167.3ms    119.4ms     43.5ms
  512     45.8ms     28.7ms    455.4ms    456.2ms     84.7ms
  640       81ms     45.8ms    621.6ms    657.8ms    135.5ms
  768    132.9ms     80.4ms      1.37s      1.43s    208.9ms
  896    201.3ms    112.6ms         2s      2.09s    295.2ms
 1024    300.9ms    155.9ms       3.5s      3.91s    438.9ms
```
