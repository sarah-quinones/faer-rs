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

All computations are done on `f64` and column major matrices.

## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      1.2µs      1.2µs      1.1µs      1.9µs      1.2µs
   64      8.1µs      8.1µs      7.8µs     10.8µs      5.1µs
   96       28µs     11.1µs     26.1µs     34.1µs       10µs
  128     65.9µs     16.5µs     35.2µs     79.2µs     32.5µs
  192    218.5µs     53.4µs     53.8µs    257.6µs     52.2µs
  256    513.8µs      123µs    154.6µs    603.3µs    143.5µs
  384      1.7ms    375.5µs    426.2µs        2ms    328.5µs
  512      4.1ms    853.5µs      1.3ms      4.7ms        1ms
  640        8ms      1.6ms      2.3ms      9.3ms      1.9ms
  768       14ms      2.9ms      3.6ms     16.1ms      3.2ms
  896     22.2ms      4.7ms      6.5ms     25.9ms      5.8ms
 1024     34.1ms      7.1ms      8.9ms     39.1ms      8.1ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.4µs      2.4µs        8µs      7.1µs      2.9µs
   64     10.3µs     10.3µs     25.1µs     35.2µs     13.9µs
   96     29.4µs     24.8µs     53.1µs    102.6µs     36.8µs
  128       59µs     40.8µs    142.7µs    242.6µs     81.4µs
  192    173.5µs     92.2µs    261.4µs    832.3µs    214.6µs
  256    380.5µs      164µs    669.7µs      1.9ms      501µs
  384      1.1ms    325.2µs      1.4ms      7.3ms      1.4ms
  512      2.7ms    679.6µs      3.5ms     17.5ms      3.2ms
  640      4.8ms      1.3ms      5.7ms     32.5ms      5.5ms
  768      8.2ms      2.3ms      9.3ms     55.5ms      9.2ms
  896     12.4ms      3.6ms     13.5ms       88ms     13.9ms
 1024       19ms      5.3ms     24.3ms      133ms     22.9ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.3µs     10.6µs      8.1µs      7.1µs      2.9µs
   64     10.6µs     25.3µs     25.1µs     35.2µs     13.9µs
   96     25.5µs     40.4µs     53.3µs    102.6µs     36.6µs
  128     39.3µs     64.3µs    143.8µs    239.7µs     81.3µs
  192    100.5µs     93.7µs      261µs    832.1µs    214.5µs
  256    188.9µs    141.1µs    669.7µs      1.9ms    500.6µs
  384    522.5µs      262µs      1.4ms      7.2ms      1.4ms
  512      1.1ms    446.9µs      3.5ms     17.4ms      3.2ms
  640        2ms    622.6µs      5.6ms     32.5ms      5.5ms
  768      3.3ms    952.1µs      9.2ms     55.5ms      9.2ms
  896      4.8ms      1.4ms     13.4ms     87.9ms     13.9ms
 1024      7.3ms      2.2ms     24.2ms    132.5ms     22.9ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.3µs      4.3µs      3.1µs      2.6µs      2.3µs
   64       12µs     12.1µs     36.8µs     11.3µs      8.8µs
   96       28µs     28.1µs     71.5µs     31.9µs     20.2µs
  128     40.1µs     40.4µs    118.9µs     81.6µs     37.6µs
  192    109.3µs    114.9µs      227µs    251.6µs     98.3µs
  256    187.5µs    175.8µs      538µs    590.4µs      205µs
  384    507.4µs    455.6µs      1.1ms      2.1ms    557.1µs
  512      1.2ms    658.7µs      3.7ms      5.3ms      1.2ms
  640        2ms      1.2ms      3.3ms     10.2ms      2.1ms
  768      3.4ms      1.8ms      5.3ms     17.6ms      3.5ms
  896      5.2ms      2.6ms      6.9ms     27.9ms      5.4ms
 1024      8.2ms      3.4ms     14.5ms     41.9ms      8.1ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.6µs      4.6µs      5.6µs      5.1µs      3.8µs
   64     20.1µs       20µs     17.5µs     22.3µs     15.6µs
   96     41.1µs     41.1µs     34.2µs     67.7µs     36.1µs
  128     80.7µs     92.9µs     95.5µs    158.8µs    123.6µs
  192    209.6µs    224.9µs    186.2µs      502µs    404.1µs
  256    493.8µs    514.4µs    316.3µs      1.3ms    819.8µs
  384      1.2ms      1.2ms    659.7µs      4.6ms      1.8ms
  512      2.6ms        2ms      1.2ms     11.3ms      4.3ms
  640      4.5ms      3.8ms      2.2ms       21ms      5.7ms
  768      7.8ms      5.5ms      3.3ms     36.2ms      8.8ms
  896     11.6ms      7.5ms      4.7ms     56.9ms     11.3ms
 1024     16.2ms     11.8ms      6.7ms       90ms     18.3ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.3µs    189.6µs          -     14.9µs     11.4µs
   64     43.6µs    402.1µs          -    104.6µs     75.4µs
   96    108.5µs    654.1µs          -    344.7µs      209µs
  128    224.7µs    969.4µs          -    814.2µs    459.1µs
  192    591.1µs      1.7ms          -      2.7ms      1.4ms
  256      1.4ms      3.4ms          -      6.5ms      3.3ms
  384      4.4ms      6.4ms          -     22.1ms       11ms
  512     11.3ms     11.1ms          -     53.5ms       27ms
  640     19.2ms     17.3ms          -    102.5ms     50.7ms
  768     32.4ms     24.1ms          -    176.9ms     87.1ms
  896     49.7ms     36.3ms          -    283.4ms    136.8ms
 1024     77.9ms     45.4ms          -    434.7ms    208.8ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     13.8µs     13.8µs     15.4µs        8µs        7µs
   64     39.3µs     39.2µs     61.4µs     43.9µs     48.1µs
   96     77.4µs     77.2µs    318.4µs    143.5µs     78.8µs
  128    133.7µs    164.8µs    863.9µs    334.9µs    156.3µs
  192      316µs    433.1µs      2.1ms      1.1ms    381.2µs
  256    641.2µs    594.3µs      3.7ms      2.5ms    822.1µs
  384      1.8ms      1.2ms      7.8ms      8.1ms      2.1ms
  512      4.3ms      2.4ms     15.7ms     18.9ms      4.5ms
  640      7.4ms      3.4ms     22.2ms     36.3ms        8ms
  768     12.2ms      4.9ms       40ms     61.5ms     13.2ms
  896     18.5ms      6.9ms     53.8ms     97.6ms     20.6ms
 1024       28ms      9.6ms     76.9ms    149.9ms     30.4ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.1µs    190.9µs          -     18.1µs        9µs
   64     45.5µs    399.6µs          -    128.5µs     40.3µs
   96    109.3µs    632.3µs          -    426.4µs     99.1µs
  128    218.5µs    886.4µs          -        1ms    216.5µs
  192      632µs      1.5ms          -      3.3ms    639.6µs
  256      1.4ms      2.2ms          -      7.7ms      1.4ms
  384      4.8ms      4.2ms          -     25.5ms      5.6ms
  512       12ms      7.2ms          -     60.2ms     14.3ms
  640     22.5ms     14.2ms          -    116.5ms     25.5ms
  768     37.8ms     20.6ms          -    200.7ms     43.8ms
  896     61.1ms     28.3ms          -    329.2ms     68.3ms
 1024     91.1ms     36.9ms          -    492.8ms    107.2ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     13.7µs       30µs     10.7µs     21.1µs     10.6µs
   64     51.4µs     72.9µs     37.8µs     98.6µs     46.1µs
   96    128.5µs    119.8µs    149.6µs    285.2µs    118.8µs
  128    217.1µs    181.5µs    301.9µs    638.8µs      328µs
  192    588.4µs    430.8µs      639µs      2.2ms    942.4µs
  256      1.1ms    770.5µs      1.1ms      5.6ms        2ms
  384      3.2ms      1.6ms      2.2ms     19.2ms      5.1ms
  512      6.8ms      3.2ms      4.5ms     46.1ms     11.8ms
  640     12.5ms      6.2ms      7.2ms     86.1ms     19.1ms
  768     20.5ms      9.1ms     10.9ms    148.3ms     30.8ms
  896     32.1ms     14.6ms     16.5ms    231.7ms     44.2ms
 1024     45.7ms     19.8ms     23.6ms    368.2ms     69.1ms
```
