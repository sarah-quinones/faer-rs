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
   32      1.2µs      1.1µs      1.1µs      1.9µs      1.2µs
   64      8.1µs      8.1µs      7.8µs     10.8µs      5.1µs
   96     27.7µs     10.8µs     26.1µs     34.2µs     10.3µs
  128     65.7µs     16.9µs     35.7µs     78.6µs     33.2µs
  192    218.5µs     65.3µs     61.3µs    257.9µs       54µs
  256    514.4µs    130.5µs    210.3µs    605.7µs    145.4µs
  384      1.7ms    403.8µs    439.1µs        2ms    334.4µs
  512      4.1ms    902.5µs      1.3ms      4.7ms        1ms
  640        8ms      1.7ms      2.3ms      9.3ms      2.1ms
  768     13.9ms        3ms      3.7ms     16.2ms      3.3ms
  896     22.3ms      4.8ms      6.5ms     26.1ms      5.9ms
 1024     33.9ms      7.3ms      8.9ms     39.3ms      8.3ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.4µs      2.4µs      8.2µs      7.2µs      2.9µs
   64     10.5µs     10.5µs     26.8µs     35.2µs     13.6µs
   96     29.5µs     25.8µs     55.3µs    102.9µs       37µs
  128     59.6µs     42.9µs    145.1µs    242.2µs     81.4µs
  192    175.2µs     92.5µs    264.4µs    834.2µs    214.4µs
  256    384.6µs    164.8µs    665.3µs        2ms    503.7µs
  384      1.1ms    338.3µs      1.4ms      6.8ms      1.4ms
  512      2.7ms    695.8µs      3.5ms     15.5ms      3.2ms
  640      4.8ms      1.4ms      5.7ms     29.8ms      5.5ms
  768      8.2ms      2.4ms      9.4ms     52.6ms      9.2ms
  896     12.6ms      3.8ms     13.4ms       84ms     13.9ms
 1024     19.2ms      5.4ms     24.3ms    130.6ms     23.6ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.3µs     10.6µs      8.2µs      7.2µs      2.9µs
   64     10.6µs     25.5µs     26.3µs     35.2µs     13.6µs
   96     25.3µs     40.8µs     55.2µs    102.9µs     36.8µs
  128     39.1µs     64.3µs    147.5µs    240.4µs     81.6µs
  192    100.3µs     93.7µs      265µs    831.1µs    214.6µs
  256    188.4µs    140.1µs    662.9µs        2ms    504.4µs
  384    520.7µs    260.9µs      1.4ms      7.3ms      1.4ms
  512      1.1ms    451.6µs      3.6ms     17.1ms      3.2ms
  640        2ms    696.9µs      5.9ms     32.3ms      5.5ms
  768      3.2ms      993µs      9.5ms     55.4ms      9.2ms
  896      4.8ms      1.6ms     13.7ms     87.5ms       14ms
 1024      7.4ms      2.5ms     24.8ms      133ms       23ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.2µs      4.3µs      3.2µs      2.3µs      2.2µs
   64       12µs     12.1µs       39µs     11.3µs      8.6µs
   96     27.5µs     27.7µs     74.4µs     32.4µs     20.1µs
  128     39.5µs     39.7µs    128.9µs     81.9µs     36.5µs
  192    107.6µs    115.4µs    241.2µs    253.3µs     95.8µs
  256    186.3µs    176.4µs    575.4µs    637.1µs    198.5µs
  384    505.4µs    465.7µs      1.3ms      2.2ms    549.5µs
  512      1.2ms    683.7µs      3.7ms      6.1ms      1.2ms
  640        2ms      1.4ms      3.3ms     11.4ms      2.1ms
  768      3.4ms      1.9ms      5.5ms     19.5ms      3.6ms
  896      5.3ms      2.9ms      6.9ms     30.5ms      5.5ms
 1024      8.2ms      3.6ms     14.6ms     44.4ms      8.2ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.4µs      4.5µs      5.7µs      4.9µs      3.8µs
   64     17.6µs     17.5µs     17.3µs     22.1µs     15.3µs
   96     39.9µs     40.4µs     34.8µs     67.6µs     36.4µs
  128     78.4µs     76.2µs    100.5µs      160µs    126.9µs
  192    198.4µs    215.7µs      193µs    502.3µs    408.8µs
  256      407µs    407.6µs    323.8µs      1.4ms    954.7µs
  384      1.1ms    954.8µs      691µs      4.6ms        2ms
  512      2.4ms      1.8ms      1.5ms     11.7ms      4.9ms
  640      4.2ms      2.8ms      2.3ms     21.6ms      6.7ms
  768      6.9ms      4.3ms      3.5ms     37.3ms      9.6ms
  896     10.4ms      6.4ms      4.8ms     58.3ms     11.9ms
 1024     15.6ms      8.7ms      6.9ms     91.4ms     20.2ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.4µs    191.1µs          -     14.9µs     10.5µs
   64     43.8µs    406.2µs          -    105.2µs     68.3µs
   96    108.7µs    657.4µs          -    347.4µs      203µs
  128    224.9µs      973µs          -    829.5µs    453.7µs
  192    619.1µs      1.7ms          -      2.8ms      1.4ms
  256      1.4ms      3.4ms          -      6.6ms      3.3ms
  384      4.6ms      6.5ms          -       22ms     10.7ms
  512     11.8ms     11.1ms          -       53ms     26.4ms
  640     20.1ms     17.6ms          -    101.7ms     49.3ms
  768     33.6ms     24.4ms          -    175.7ms     85.1ms
  896     52.3ms     36.1ms          -    279.1ms    134.3ms
 1024     80.1ms     45.6ms          -      433ms      203ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     13.6µs     13.6µs     15.3µs        8µs      7.1µs
   64     38.7µs     38.8µs     60.2µs     43.6µs     44.9µs
   96     76.6µs     76.8µs    327.3µs    142.2µs     78.8µs
  128    131.8µs    162.1µs    981.7µs    326.9µs    154.4µs
  192    314.1µs    342.3µs      2.1ms      1.1ms    386.4µs
  256    631.3µs    568.4µs      3.9ms      2.5ms      815µs
  384      1.8ms      1.2ms      8.3ms      8.1ms      2.1ms
  512      4.2ms      2.3ms     19.1ms       19ms      4.5ms
  640      7.2ms      3.4ms     22.8ms     36.3ms      8.1ms
  768     11.9ms        5ms     41.1ms     61.5ms     13.3ms
  896     18.2ms        7ms     55.8ms     97.5ms     20.6ms
 1024     27.8ms      9.7ms     80.2ms    152.5ms       31ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      9.9µs    184.7µs          -     18.1µs      9.1µs
   64     44.3µs    398.9µs          -    128.7µs       36µs
   96      108µs    628.5µs          -    425.4µs       99µs
  128    215.5µs    891.2µs          -    997.4µs    220.5µs
  192    627.9µs      1.5ms          -      3.3ms    617.3µs
  256      1.4ms      2.2ms          -      7.7ms      1.5ms
  384      4.8ms      4.2ms          -     25.5ms      5.6ms
  512     11.6ms      8.7ms          -     60.2ms     14.1ms
  640     22.4ms     15.2ms          -    116.7ms     25.5ms
  768       38ms     21.3ms          -    200.6ms     44.3ms
  896     60.9ms     28.8ms          -    318.6ms     68.3ms
 1024     90.8ms       38ms          -    488.9ms    119.1ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     13.8µs     30.7µs     10.3µs     21.1µs     10.5µs
   64     49.7µs     73.3µs       38µs     99.7µs     45.4µs
   96    125.6µs    125.5µs    159.5µs    283.8µs    119.4µs
  128    227.5µs    206.1µs      304µs    652.4µs    332.5µs
  192    596.8µs    428.6µs    758.1µs      2.2ms    946.3µs
  256      1.2ms      803µs      1.2ms      5.6ms      2.1ms
  384      3.2ms      1.8ms      2.3ms       19ms      5.2ms
  512      7.3ms      3.9ms      5.3ms     46.1ms     12.2ms
  640     12.8ms      6.9ms      7.3ms     86.3ms     20.3ms
  768     21.7ms     10.9ms     11.6ms    149.1ms     32.2ms
  896       33ms     16.5ms     17.2ms    233.1ms     44.7ms
 1024       49ms     22.9ms     27.8ms    403.7ms     72.6ms
```
