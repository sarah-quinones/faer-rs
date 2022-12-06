# faer

`faer` is a collection of crates that implement low level linear algebra routines in pure Rust.
The aim is to eventually provide a fully featured library for linear algebra with focus on portability, correctness, and performance.

See the Wiki and the `docs.rs` documentation for code examples and usage instructions.

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

## faer-qr (WIP)

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
   32      1.3µs      1.1µs      1.1µs      1.9µs      1.2µs
   64      8.1µs      8.1µs      7.8µs     10.8µs        5µs
   96     27.7µs     10.8µs     26.1µs     34.2µs     10.4µs
  128     65.7µs     17.6µs     36.2µs       79µs     33.6µs
  192    217.5µs     57.2µs     55.5µs    258.6µs     53.3µs
  256    513.7µs    126.9µs    158.1µs    610.1µs    148.1µs
  384      1.7ms    388.9µs    448.5µs        2ms      339µs
  512      4.1ms    873.6µs      1.4ms      4.7ms      1.1ms
  640        8ms      1.7ms      2.4ms      9.4ms        2ms
  768     13.9ms      2.9ms      3.7ms     16.1ms      3.2ms
  896     22.2ms      4.8ms      6.6ms     26.2ms      5.9ms
 1024       34ms      7.2ms      9.1ms     39.8ms      8.3ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.9µs      2.4µs        9µs      7.1µs      2.9µs
   64     10.2µs     10.2µs       27µs     34.7µs     13.7µs
   96     29.4µs     25.4µs     58.3µs    102.1µs     36.7µs
  128     59.2µs     40.1µs    153.1µs    241.7µs     82.4µs
  192      174µs     93.4µs    281.7µs    831.6µs    213.9µs
  256    381.7µs    166.8µs    724.7µs      1.9ms      486µs
  384      1.1ms    332.8µs      1.5ms      6.9ms      1.4ms
  512      2.7ms    699.4µs      3.7ms     15.6ms      3.2ms
  640      4.8ms      1.4ms      5.9ms     29.9ms      5.4ms
  768      8.2ms      2.5ms      9.5ms     53.1ms      9.1ms
  896     12.5ms      3.7ms     13.7ms     83.8ms     13.8ms
 1024     19.1ms      5.4ms     24.9ms    125.9ms     22.9ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.5µs     10.2µs      9.1µs      7.2µs      2.9µs
   64     10.6µs     17.7µs       27µs     34.5µs     13.7µs
   96     25.7µs     39.5µs     58.2µs    102.7µs     36.9µs
  128     39.4µs     47.2µs    155.7µs    241.4µs     82.4µs
  192    100.8µs     88.8µs    284.8µs    831.3µs    214.2µs
  256    189.3µs    148.4µs    743.2µs      1.9ms    487.1µs
  384    519.6µs      270µs      1.5ms        7ms      1.4ms
  512      1.1ms    454.6µs      3.7ms     15.9ms      3.2ms
  640        2ms    680.8µs      5.9ms     30.5ms      5.4ms
  768      3.2ms      969µs      9.6ms     53.8ms      9.1ms
  896      4.8ms      1.4ms     13.9ms     84.8ms     13.8ms
 1024      7.3ms      2.3ms       25ms    127.9ms       23ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.8µs      4.5µs        3µs      2.2µs      2.2µs
   64     12.4µs     12.5µs     38.6µs       11µs      8.5µs
   96     28.7µs     28.8µs     73.5µs       32µs     19.7µs
  128     40.9µs       41µs    127.5µs     81.4µs     36.1µs
  192    110.5µs      120µs    239.7µs    252.5µs     94.7µs
  256    187.5µs    177.7µs      654µs    598.6µs    195.7µs
  384    507.9µs    464.2µs      1.4ms      2.1ms    548.5µs
  512      1.2ms    690.9µs      3.9ms      5.9ms      1.2ms
  640        2ms      1.3ms      3.6ms     11.3ms      2.1ms
  768      3.4ms      1.8ms      5.6ms     19.6ms      3.6ms
  896      5.2ms      2.7ms      7.3ms     30.5ms      5.5ms
 1024      8.1ms      3.4ms     15.1ms     45.9ms      8.3ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      5.4µs      5.4µs      5.5µs      4.8µs      3.9µs
   64     21.5µs     21.5µs     17.5µs     22.1µs     15.1µs
   96     52.7µs     49.9µs     34.7µs     67.9µs     36.3µs
  128     97.9µs    102.8µs     99.5µs    160.7µs    126.1µs
  192    260.6µs    290.2µs    192.4µs    499.9µs    406.5µs
  256    539.5µs    555.9µs    327.4µs      1.3ms      1.2ms
  384      1.4ms      1.3ms    681.5µs      4.5ms      2.2ms
  512      2.9ms      2.4ms      1.3ms     11.2ms      5.7ms
  640        5ms        4ms      2.5ms     20.8ms      7.2ms
  768      8.2ms      6.1ms      3.8ms       36ms     10.1ms
  896     11.8ms      8.8ms      5.3ms     56.4ms     12.7ms
 1024     17.8ms     12.7ms      7.7ms     88.5ms     18.5ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     11.8µs     11.3µs          -     14.6µs     10.4µs
   64     47.4µs     47.3µs          -    104.2µs     67.7µs
   96    117.1µs      117µs          -    344.9µs    201.5µs
  128    238.5µs    241.5µs          -    819.3µs    458.4µs
  192    606.6µs    611.2µs          -      2.7ms      1.4ms
  256      1.4ms      1.4ms          -      6.6ms      3.3ms
  384      4.4ms      4.5ms          -     22.1ms     10.9ms
  512     11.3ms      8.7ms          -     53.2ms     26.9ms
  640       19ms     12.9ms          -    102.4ms     50.4ms
  768     32.2ms     18.4ms          -    176.9ms     86.8ms
  896     49.6ms     26.8ms          -    279.9ms    135.1ms
 1024     77.8ms     35.3ms          -    436.9ms    204.3ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     11.1µs       11µs     15.5µs        8µs        7µs
   64     34.3µs     34.3µs     60.9µs     43.4µs     44.3µs
   96     70.8µs     70.8µs      327µs    143.9µs     78.6µs
  128      126µs    125.9µs    833.4µs    328.6µs    154.3µs
  192    326.5µs    326.5µs      1.7ms        1ms    382.7µs
  256    665.5µs    720.3µs        3ms      2.5ms    813.6µs
  384      1.9ms      1.6ms      8.5ms      8.1ms      2.1ms
  512      4.2ms      2.9ms     18.3ms     18.9ms      4.5ms
  640      7.6ms      4.5ms       25ms     36.4ms        8ms
  768     12.6ms      6.7ms     45.3ms     61.7ms     13.2ms
  896     19.2ms      9.6ms     60.3ms     97.9ms     20.5ms
 1024     28.5ms     13.5ms     87.6ms    151.4ms     30.4ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32       11µs       11µs          -     17.8µs      8.9µs
   64     47.2µs     47.2µs          -    128.2µs     35.9µs
   96    113.6µs    113.5µs          -    425.6µs     97.6µs
  128    228.8µs    228.8µs          -        1ms    223.9µs
  192    641.8µs    643.3µs          -      3.3ms    636.8µs
  256      1.4ms      1.5ms          -      7.7ms      1.5ms
  384      4.9ms      3.6ms          -     25.7ms      5.8ms
  512     11.7ms      6.9ms          -     60.8ms     15.9ms
  640     22.2ms     10.8ms          -    118.1ms     28.9ms
  768     37.9ms     15.3ms          -    202.1ms     49.1ms
  896     60.8ms       21ms          -    321.5ms       78ms
 1024     92.2ms       28ms          -    495.1ms      121ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     14.7µs     31.9µs     10.4µs     20.9µs     10.5µs
   64     52.9µs       73µs       38µs     99.4µs     45.2µs
   96      135µs      130µs    162.7µs    284.4µs    119.1µs
  128    222.8µs      205µs    285.2µs    650.1µs    325.1µs
  192    615.7µs    470.6µs    742.7µs      2.2ms    950.4µs
  256      1.2ms    812.9µs      1.3ms      5.5ms      2.1ms
  384      3.2ms      1.9ms      2.6ms     18.8ms      5.3ms
  512      7.1ms      3.5ms      5.1ms     45.3ms     12.7ms
  640     12.7ms      6.6ms      8.1ms     84.7ms     20.3ms
  768     21.1ms     10.2ms     12.2ms    144.2ms     32.1ms
  896       32ms     15.6ms     18.1ms    226.3ms     43.9ms
 1024     47.9ms     22.1ms     25.8ms    357.7ms       69ms
```
