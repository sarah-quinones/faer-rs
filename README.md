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

## faer-svd

[![Documentation](https://docs.rs/faer-svd/badge.svg)](https://docs.rs/faer-svd)
[![Crate](https://img.shields.io/crates/v/faer-svd.svg)](https://crates.io/crates/faer-svd)

The SVD module implements the singular value decomposition for real matrices (complex support will be following soon).

## Coming soon

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
   32      1.1µs      1.1µs      1.1µs      1.9µs      1.2µs
   64        8µs        8µs      7.9µs     10.8µs      5.1µs
   96     27.6µs     11.2µs     26.2µs     34.4µs     10.1µs
  128     65.5µs     16.9µs     35.5µs     79.3µs     32.8µs
  192    216.9µs     53.4µs     70.3µs    259.5µs     52.2µs
  256    511.1µs    116.3µs      200µs    602.1µs    143.9µs
  384      1.7ms    363.5µs    430.7µs        2ms    327.7µs
  512      4.1ms    845.5µs      1.3ms      4.7ms      1.2ms
  640        8ms      1.7ms      2.3ms      9.3ms        2ms
  768       14ms      3.1ms      3.6ms     16.1ms      3.2ms
  896     22.3ms      5.6ms      6.5ms     25.8ms      5.9ms
 1024     34.2ms      8.2ms      9.6ms     38.9ms      8.3ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.4µs      2.4µs      8.5µs      7.1µs      2.9µs
   64     10.2µs     10.3µs     25.8µs     34.4µs     13.5µs
   96     29.4µs     26.6µs     54.4µs    102.8µs     36.4µs
  128     58.7µs     41.4µs      145µs    240.4µs       83µs
  192    173.4µs       93µs    262.6µs      801µs    213.1µs
  256    377.2µs    166.2µs      633µs      1.8ms      491µs
  384      1.1ms    334.9µs      1.4ms      6.4ms      1.3ms
  512      2.6ms      667µs      3.5ms       15ms      3.2ms
  640      4.8ms      1.5ms      5.7ms     30.2ms      5.4ms
  768      8.1ms      2.4ms      9.3ms     51.8ms      9.3ms
  896     12.4ms      3.6ms     13.4ms     81.6ms       14ms
 1024     19.1ms      5.3ms     20.2ms    124.6ms     22.8ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.7µs     11.1µs      8.6µs        7µs      2.9µs
   64     11.6µs     18.1µs     25.9µs     34.5µs     13.3µs
   96     27.8µs     37.4µs     54.4µs    103.2µs     36.4µs
  128     41.1µs     51.5µs    145.3µs    240.5µs     83.1µs
  192    104.7µs     92.5µs    263.4µs    803.6µs    213.5µs
  256    189.1µs    135.5µs    639.8µs      1.8ms    491.5µs
  384    522.9µs    269.7µs      1.4ms      6.6ms      1.3ms
  512      1.1ms    449.4µs      3.5ms     15.6ms      3.2ms
  640        2ms    635.6µs      5.7ms     30.7ms      5.5ms
  768      3.2ms        1ms      9.4ms     52.7ms      9.3ms
  896      4.8ms      2.3ms     13.5ms       83ms       14ms
 1024      7.2ms      2.5ms     20.1ms    126.3ms     22.8ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32        2µs        2µs      3.3µs      2.5µs      2.3µs
   64      7.5µs      7.3µs       38µs     12.9µs      8.8µs
   96     19.2µs     19.2µs      117µs     38.2µs     20.5µs
  128     30.9µs     30.9µs    165.6µs     92.6µs     37.7µs
  192       90µs     98.6µs    298.7µs      295µs     99.1µs
  256    167.9µs      152µs    703.3µs    677.1µs    204.8µs
  384    477.6µs    421.5µs      1.2ms      2.2ms    557.9µs
  512      1.1ms    627.5µs      3.8ms      5.7ms      1.2ms
  640      1.9ms      1.2ms      3.3ms     10.8ms      2.1ms
  768      3.3ms      1.7ms      5.5ms     18.8ms      3.5ms
  896      5.1ms      2.6ms        7ms     29.3ms      5.5ms
 1024      7.9ms      3.3ms     14.9ms     43.5ms      8.2ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.8µs      3.7µs      5.6µs      4.9µs        4µs
   64     18.1µs     19.8µs     17.5µs     22.4µs     15.5µs
   96     47.6µs     44.7µs     34.8µs     68.7µs     36.6µs
  128     93.2µs       95µs     98.7µs    167.3µs    127.4µs
  192    247.3µs    260.7µs    189.1µs    527.7µs    426.2µs
  256    480.8µs      502µs    374.7µs      1.3ms    820.9µs
  384      1.3ms      1.2ms      1.2ms      4.5ms      1.9ms
  512      2.8ms      2.4ms      1.6ms     11.1ms      4.4ms
  640      4.9ms      3.9ms      2.3ms     20.7ms      5.6ms
  768      7.7ms        6ms      3.4ms     35.8ms      8.7ms
  896     11.4ms      8.2ms      4.8ms     56.4ms     11.3ms
 1024     16.9ms     11.2ms      6.9ms     89.2ms     17.4ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      8.3µs      8.3µs          -     14.9µs       12µs
   64     34.5µs     34.6µs          -    105.1µs     71.7µs
   96     93.3µs     92.1µs          -    345.5µs    206.3µs
  128    208.8µs    205.7µs          -    834.6µs    463.7µs
  192    569.8µs    567.5µs          -      2.7ms      1.4ms
  256      1.3ms      1.4ms          -      6.5ms      3.3ms
  384      4.4ms      4.2ms          -     21.9ms     10.8ms
  512     11.1ms      8.3ms          -     52.5ms     26.5ms
  640     19.9ms     12.7ms          -    101.5ms     49.8ms
  768     33.6ms     18.1ms          -    175.4ms     86.2ms
  896     52.3ms     25.7ms          -    280.3ms    134.4ms
 1024     79.6ms       36ms          -      430ms    205.4ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     11.9µs     11.9µs     15.4µs      7.8µs      6.9µs
   64     34.9µs     34.8µs     60.5µs     43.2µs     47.9µs
   96     71.4µs     71.4µs    327.4µs    141.6µs     79.8µs
  128    125.2µs    125.2µs    824.9µs    320.2µs    155.7µs
  192    323.2µs    323.4µs      1.8ms      1.1ms    383.1µs
  256    654.9µs    709.5µs      5.2ms      2.4ms    799.8µs
  384      1.9ms      1.7ms        8ms        8ms      2.1ms
  512      4.1ms      3.1ms     16.4ms     18.7ms      4.5ms
  640      7.4ms      4.5ms     22.2ms       36ms        8ms
  768     12.2ms      6.7ms     35.1ms     62.2ms     13.3ms
  896     18.6ms      9.4ms     46.5ms     98.2ms     20.5ms
 1024     27.7ms     13.2ms     66.3ms    151.4ms     30.6ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     23.7µs     37.6µs          -     17.9µs      9.5µs
   64     96.4µs    123.3µs          -    125.7µs     38.5µs
   96    231.6µs    263.1µs          -    421.4µs     97.8µs
  128    452.9µs    491.7µs          -    984.7µs    218.5µs
  192      1.1ms      1.2ms          -      3.3ms    617.1µs
  256      2.3ms      2.2ms          -      7.6ms      1.4ms
  384      6.6ms      4.5ms          -     25.4ms      5.4ms
  512     14.7ms      7.8ms          -     59.6ms     14.2ms
  640     26.5ms     12.3ms          -    115.8ms     25.9ms
  768     44.4ms     17.3ms          -    199.8ms       46ms
  896     68.5ms       23ms          -    319.5ms     69.1ms
 1024    104.8ms     42.1ms          -      492ms    118.6ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32       14µs     30.6µs     10.3µs     20.9µs     10.7µs
   64     51.4µs     75.5µs     38.1µs     98.4µs     45.7µs
   96    134.2µs    134.7µs    188.5µs    291.5µs    118.9µs
  128    222.8µs    203.9µs    349.4µs    663.6µs    345.5µs
  192    607.6µs    480.5µs    643.4µs      2.2ms    965.1µs
  256      1.1ms    832.3µs      1.1ms      5.6ms        2ms
  384      3.2ms      1.9ms      2.4ms     19.1ms      5.1ms
  512      6.7ms      3.6ms      4.6ms     44.6ms     11.9ms
  640     12.5ms      6.6ms      7.3ms     85.4ms     19.3ms
  768     20.5ms      9.9ms     11.3ms    145.1ms     31.7ms
  896     32.1ms     14.9ms     16.8ms    228.9ms     44.4ms
 1024       47ms     21.9ms       24ms    359.6ms     68.9ms
```

## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      140µs    171.6µs     92.6µs    103.9µs    228.8µs
   64    494.3µs    461.6µs    680.3µs    566.5µs        1ms
   96      1.1ms      1.1ms      1.7ms      1.7ms      2.5ms
  128        2ms      1.9ms      2.9ms      4.5ms      4.2ms
  192      4.6ms      4.5ms      6.7ms     14.8ms      9.9ms
  256      8.8ms      7.5ms     11.6ms       46ms     17.2ms
  384     23.2ms     15.8ms     25.7ms      121ms     42.1ms
  512     50.1ms     29.2ms     51.6ms      449ms     82.8ms
  640     85.8ms     54.4ms     78.1ms    652.2ms    129.1ms
  768    140.1ms       81ms    122.7ms      1.43s      202ms
  896    211.1ms    114.4ms    172.8ms      2.09s    281.4ms
 1024      317ms      157ms    254.7ms      3.89s    415.7ms
```

## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      1.2ms      1.3ms      5.4ms      5.1ms      3.1ms
   64      3.4ms      3.3ms     15.8ms     20.2ms      8.8ms
   96        7ms      5.9ms     30.2ms     44.6ms     18.9ms
  128     11.5ms      8.8ms     47.8ms     81.2ms     35.8ms
  192     24.3ms     17.1ms     63.6ms    183.9ms     56.5ms
  256       42ms     26.8ms     83.9ms    385.2ms     94.3ms
  384     92.2ms     49.5ms      135ms    922.1ms    209.9ms
  512    166.6ms     80.1ms    305.2ms      2.05s    392.3ms
  640    262.6ms    119.1ms    293.4ms       3.3s      632ms
  768    388.2ms    190.3ms    445.7ms      5.25s    924.7ms
  896    540.4ms    253.5ms    553.1ms      7.41s      1.31s
 1024      735ms    329.6ms    856.1ms     10.83s      1.72s
```
