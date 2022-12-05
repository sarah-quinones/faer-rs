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
   64      8.1µs      8.2µs      7.8µs     10.9µs      5.1µs
   96     27.9µs     10.5µs     26.1µs     34.2µs     10.1µs
  128     66.1µs     17.3µs     35.2µs     79.3µs     32.9µs
  192    218.4µs       53µs     54.1µs    258.9µs     51.7µs
  256    513.7µs    124.1µs    154.6µs    607.1µs      143µs
  384      1.7ms    378.6µs    347.4µs        2ms    327.4µs
  512      4.1ms    855.1µs        1ms      4.8ms        1ms
  640        8ms      1.6ms      2.2ms      9.3ms        2ms
  768       14ms      2.9ms      3.6ms     16.1ms      3.2ms
  896     22.2ms      4.7ms      6.5ms     25.9ms      5.8ms
 1024     34.1ms      7.1ms        9ms     39.2ms      8.2ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.4µs      2.4µs      8.2µs      7.1µs      2.9µs
   64     10.1µs     10.1µs     26.1µs     34.7µs     13.6µs
   96     29.4µs     27.2µs     56.8µs    102.6µs     36.9µs
  128     59.1µs     41.9µs    148.9µs    243.1µs     81.3µs
  192    174.2µs     91.8µs    275.4µs    832.3µs    214.3µs
  256    382.5µs      164µs    713.2µs      1.9ms    490.1µs
  384      1.1ms    321.8µs      1.5ms      6.9ms      1.4ms
  512      2.7ms    677.9µs      3.7ms     15.7ms      3.2ms
  640      4.8ms      1.3ms        6ms     30.1ms      5.4ms
  768      8.2ms      2.5ms      9.6ms     53.3ms      9.1ms
  896     12.5ms      3.6ms     13.9ms     83.9ms     13.8ms
 1024     19.1ms      5.4ms     24.9ms    129.7ms     22.8ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.3µs     10.1µs      8.1µs      7.1µs      2.9µs
   64     10.7µs     17.9µs       26µs       35µs     13.6µs
   96     25.7µs     39.3µs     56.1µs    103.9µs     36.9µs
  128     39.5µs     47.1µs      149µs    240.2µs     81.4µs
  192      101µs       89µs      276µs    832.6µs    214.3µs
  256    189.1µs    134.8µs    715.1µs      1.9ms    490.2µs
  384    520.5µs      267µs      1.5ms      6.9ms      1.4ms
  512      1.1ms    448.9µs      3.7ms     15.7ms      3.2ms
  640        2ms    677.7µs      5.9ms     30.1ms      5.4ms
  768      3.2ms      947µs      9.5ms     53.3ms      9.1ms
  896      4.8ms      1.4ms     13.8ms     83.9ms     13.8ms
 1024      7.3ms      2.1ms     24.8ms    129.3ms     22.8ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.5µs      4.5µs      3.2µs      2.2µs      2.2µs
   64     12.5µs     12.6µs     38.1µs     11.2µs      8.6µs
   96     28.8µs     28.8µs     72.3µs     31.9µs       20µs
  128     40.9µs     41.1µs    122.9µs     81.3µs     36.7µs
  192    110.5µs    119.2µs    232.3µs    252.2µs     95.5µs
  256    190.3µs    178.3µs    544.9µs    600.5µs    197.4µs
  384    508.8µs    461.9µs      1.3ms      2.1ms    547.1µs
  512      1.2ms    691.3µs      3.8ms        6ms      1.2ms
  640        2ms      1.3ms      3.5ms     11.4ms      2.1ms
  768      3.4ms      1.8ms      5.5ms       20ms      3.6ms
  896      5.2ms      2.6ms      7.1ms     30.9ms      5.4ms
 1024      8.1ms      3.4ms     14.9ms     46.9ms      8.2ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      5.3µs      5.3µs      5.6µs      4.9µs      3.8µs
   64       23µs       23µs     17.3µs     21.9µs     15.1µs
   96     51.1µs     53.1µs     34.5µs     67.7µs     36.4µs
  128    101.8µs    100.8µs     95.8µs    159.2µs      129µs
  192    263.1µs    277.7µs    185.7µs    503.1µs    408.2µs
  256      544µs    569.6µs    315.8µs      1.3ms    794.9µs
  384      1.4ms      1.3ms      655µs      4.6ms      1.8ms
  512      2.9ms      2.6ms      1.2ms     11.8ms      4.2ms
  640        5ms        4ms      2.4ms     21.6ms      5.5ms
  768      8.1ms      6.1ms      3.7ms     37.3ms      8.4ms
  896     12.2ms      8.7ms      5.2ms     58.6ms     11.2ms
 1024     17.6ms     12.8ms      7.5ms     93.1ms       17ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     11.4µs     11.4µs          -     14.5µs     11.9µs
   64     47.4µs     47.2µs          -      105µs     67.5µs
   96    117.5µs    117.1µs          -    344.5µs    199.9µs
  128      240µs    237.4µs          -    817.6µs    454.8µs
  192    608.6µs    607.5µs          -      2.7ms      1.4ms
  256      1.5ms      1.5ms          -      6.6ms      3.2ms
  384      4.6ms        5ms          -     22.1ms     10.9ms
  512     11.8ms      8.8ms          -     53.2ms     26.9ms
  640     20.2ms     12.9ms          -    102.6ms     50.6ms
  768     33.7ms     18.4ms          -    177.5ms     87.1ms
  896     52.6ms     26.1ms          -    283.8ms    136.3ms
 1024     81.1ms     34.9ms          -    440.4ms    208.9ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     11.2µs     11.1µs     15.1µs        8µs      7.1µs
   64     34.5µs     34.5µs     60.4µs     43.7µs       45µs
   96       71µs       71µs    318.4µs    141.3µs     79.3µs
  128    126.4µs    126.4µs      818µs    327.1µs    155.4µs
  192    327.5µs    327.9µs      1.6ms        1ms    387.3µs
  256    669.2µs    681.9µs      2.9ms      2.5ms    807.2µs
  384      1.9ms      1.6ms      7.9ms      8.1ms      2.1ms
  512      4.3ms      2.9ms     17.8ms       19ms      4.5ms
  640      7.7ms      4.5ms     24.6ms     36.5ms        8ms
  768     12.7ms      6.7ms     44.1ms     61.7ms     13.2ms
  896     19.4ms      9.4ms     59.1ms     97.5ms     20.5ms
 1024     28.7ms     13.3ms     85.2ms    149.6ms     30.3ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32       11µs       11µs          -     17.9µs        9µs
   64     47.2µs     47.4µs          -    128.6µs     36.3µs
   96    113.8µs    113.9µs          -      423µs     99.4µs
  128    221.8µs    221.6µs          -    994.2µs    217.9µs
  192    649.7µs    648.4µs          -      3.3ms    629.5µs
  256      1.4ms      1.5ms          -      7.7ms      1.4ms
  384      4.8ms      3.5ms          -     25.5ms      5.5ms
  512     11.7ms      6.8ms          -       60ms     14.2ms
  640     22.2ms     10.6ms          -    116.6ms     25.4ms
  768     37.2ms     14.8ms          -    200.2ms     44.2ms
  896     59.3ms     20.2ms          -    327.2ms     68.2ms
 1024     89.6ms       27ms          -    490.3ms    107.3ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     14.7µs     31.9µs     10.7µs     21.1µs     10.5µs
   64     54.5µs     73.3µs     38.2µs       99µs     45.5µs
   96    133.9µs    132.2µs    157.4µs    281.8µs    119.1µs
  128    225.5µs    205.8µs    268.3µs    668.4µs    328.3µs
  192    611.6µs    483.3µs    715.6µs      2.2ms    946.5µs
  256      1.2ms    827.5µs      1.3ms      5.5ms        2ms
  384      3.2ms      1.9ms      2.6ms     19.1ms        5ms
  512        7ms      3.6ms        5ms     45.9ms     11.8ms
  640     12.7ms      6.6ms      7.9ms     85.9ms     19.1ms
  768     21.4ms     10.2ms       12ms    146.3ms     30.8ms
  896     32.6ms     15.8ms     17.7ms      229ms     43.9ms
 1024     47.7ms     21.2ms     25.3ms    357.4ms     68.6ms
```
