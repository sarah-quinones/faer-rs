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
   96     27.8µs     10.6µs     26.1µs     34.2µs     10.3µs
  128     66.1µs       17µs     36.1µs     79.5µs     33.4µs
  192    218.2µs     57.2µs     54.5µs    259.6µs     53.5µs
  256    514.5µs    125.5µs    156.5µs    609.6µs    145.8µs
  384      1.7ms    376.4µs      447µs        2ms      335µs
  512      4.1ms    851.2µs      1.4ms      4.8ms        1ms
  640        8ms      1.7ms      2.4ms      9.4ms        2ms
  768       14ms        3ms      3.6ms     16.3ms      3.8ms
  896     22.3ms      4.8ms      6.7ms     26.2ms      5.9ms
 1024     34.4ms      7.1ms      9.5ms     39.4ms      8.4ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.4µs      2.4µs      8.9µs      7.1µs      2.9µs
   64     10.2µs     10.2µs     26.6µs     34.8µs     13.4µs
   96     29.4µs     26.5µs     57.5µs    102.4µs     37.2µs
  128     59.4µs     42.6µs    155.2µs    241.4µs     81.7µs
  192    173.8µs     91.6µs    277.6µs    830.5µs    213.6µs
  256    382.7µs    164.5µs    709.6µs        2ms      503µs
  384      1.1ms    346.6µs      1.5ms        7ms      1.4ms
  512      2.7ms      682µs      3.7ms     16.3ms      3.2ms
  640      4.8ms      1.3ms      5.9ms     30.8ms      5.5ms
  768      8.2ms      2.5ms      9.6ms     53.9ms      9.2ms
  896     12.6ms      3.6ms     13.9ms     84.8ms       14ms
 1024     19.3ms      5.5ms     25.5ms    126.8ms     22.8ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.3µs     10.6µs      8.9µs      7.1µs      2.9µs
   64     10.6µs     17.2µs       27µs     35.1µs     13.4µs
   96     25.4µs     38.7µs     56.9µs    102.6µs     36.8µs
  128     39.3µs     46.2µs    151.9µs    242.7µs     81.9µs
  192    100.1µs     89.4µs    283.5µs    830.1µs    213.8µs
  256    188.2µs    137.6µs    717.2µs        2ms      505µs
  384    519.4µs      270µs      1.5ms      7.3ms      1.4ms
  512      1.1ms    454.3µs      3.8ms     17.8ms      3.2ms
  640        2ms    682.4µs        6ms     32.9ms      5.5ms
  768      3.3ms    953.5µs      9.6ms     56.4ms      9.2ms
  896      4.9ms      1.4ms     14.1ms     89.2ms     13.9ms
 1024      7.3ms      2.2ms     25.2ms    132.5ms     23.1ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.5µs      4.5µs      3.1µs      2.3µs      2.3µs
   64     12.5µs     12.5µs     36.4µs       11µs      8.8µs
   96     28.9µs     28.8µs     70.9µs     32.3µs     20.4µs
  128       41µs     41.1µs    121.6µs     81.9µs       38µs
  192    110.7µs    118.2µs    228.2µs    253.6µs     99.3µs
  256    189.5µs    179.6µs    569.1µs    610.3µs    203.7µs
  384    510.6µs    465.1µs      1.3ms      2.1ms      565µs
  512      1.2ms    674.7µs        4ms      5.4ms      1.2ms
  640        2ms      1.3ms      3.5ms     10.3ms      2.1ms
  768      3.4ms      1.8ms      5.7ms     17.6ms      3.6ms
  896      5.3ms      2.6ms      7.2ms     27.8ms      5.4ms
 1024      8.2ms      3.4ms       15ms     41.4ms      8.1ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      5.3µs      5.3µs      5.7µs      4.9µs      3.8µs
   64     21.4µs       23µs     17.4µs     22.1µs     15.1µs
   96     52.8µs     49.9µs     34.5µs     68.2µs     36.3µs
  128     98.6µs    103.8µs    100.2µs    160.7µs    125.2µs
  192    262.3µs    273.4µs    190.3µs    503.8µs    408.2µs
  256    537.9µs    550.3µs    318.1µs      1.3ms    927.1µs
  384      1.4ms      1.3ms    664.4µs      4.5ms      2.1ms
  512        3ms      2.6ms      1.2ms     11.3ms      4.9ms
  640      4.8ms        4ms      2.5ms     20.9ms      6.2ms
  768      7.9ms        6ms      3.9ms       36ms        9ms
  896     12.1ms      8.9ms      5.2ms     56.8ms     11.7ms
 1024     17.5ms     12.4ms      7.5ms     87.9ms     17.8ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.9µs     10.9µs          -     14.7µs     10.8µs
   64     44.9µs       45µs          -    104.7µs     66.2µs
   96      111µs    110.8µs          -    345.6µs    201.9µs
  128    228.9µs    227.8µs          -    824.3µs    450.8µs
  192    588.8µs    589.1µs          -      2.8ms      1.4ms
  256      1.4ms      1.4ms          -      6.6ms      3.3ms
  384      4.3ms      4.5ms          -     22.2ms     10.9ms
  512     11.2ms      8.8ms          -     53.6ms     27.1ms
  640     18.8ms     12.9ms          -    102.7ms     50.5ms
  768       32ms     18.3ms          -    177.2ms     86.6ms
  896     49.2ms     26.1ms          -    281.3ms    136.7ms
 1024     76.9ms     34.5ms          -    434.3ms    208.7ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.3µs     10.3µs     15.3µs        8µs      7.1µs
   64     32.9µs     32.9µs     59.8µs     43.8µs     44.6µs
   96       69µs     69.1µs    326.5µs    143.3µs     79.2µs
  128    123.4µs    123.2µs    843.9µs    332.4µs    156.5µs
  192      324µs    323.6µs      1.7ms      1.1ms    384.6µs
  256    661.5µs    689.4µs        3ms      2.5ms    798.4µs
  384      1.9ms      1.6ms      8.5ms      8.1ms      2.1ms
  512      4.3ms      2.8ms     20.8ms     19.3ms      4.5ms
  640      7.6ms      4.4ms     25.7ms     36.8ms      8.2ms
  768     12.7ms      6.6ms     45.4ms     62.1ms     13.4ms
  896     19.3ms      9.4ms     60.1ms     98.6ms     20.7ms
 1024     28.7ms     13.4ms       87ms    152.1ms       31ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     12.5µs     25.1µs          -     18.2µs        9µs
   64     51.4µs     77.1µs          -    127.1µs     36.2µs
   96    121.3µs    145.7µs          -    425.8µs    100.8µs
  128    237.3µs    271.2µs          -    995.2µs    215.9µs
  192    659.7µs    733.8µs          -      3.3ms    617.6µs
  256      1.5ms      1.8ms          -      7.7ms      1.7ms
  384      4.9ms      3.6ms          -     25.6ms      5.5ms
  512     11.9ms        7ms          -     60.3ms     14.3ms
  640     22.9ms     10.8ms          -    117.1ms     25.5ms
  768     38.7ms     15.2ms          -    201.4ms     43.9ms
  896     61.9ms       21ms          -    320.5ms     69.3ms
 1024     92.3ms     46.6ms          -      487ms      107ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     14.5µs     31.7µs     10.6µs       21µs     10.7µs
   64     54.2µs     72.1µs     38.2µs      103µs     45.6µs
   96    134.9µs    128.8µs    158.8µs    284.1µs    119.5µs
  128    228.2µs    199.9µs    273.3µs    651.5µs    328.1µs
  192    598.7µs    465.3µs    722.4µs      2.2ms    953.8µs
  256      1.2ms    829.8µs      1.3ms      5.6ms        2ms
  384      3.2ms      1.8ms      2.7ms     18.9ms      5.1ms
  512        7ms      3.7ms        5ms     45.6ms     12.3ms
  640     12.7ms      6.6ms      8.2ms     84.8ms     19.4ms
  768     21.5ms     10.1ms     12.2ms    145.2ms     31.8ms
  896     32.6ms     15.4ms     20.3ms    227.4ms       44ms
 1024     47.7ms     21.1ms     25.6ms    354.9ms     68.8ms
```
