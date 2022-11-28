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
   64      8.1µs      8.2µs      7.8µs     10.8µs        5µs
   96     27.7µs     10.5µs     26.1µs     34.2µs      9.8µs
  128     65.8µs     16.5µs     35.2µs     80.1µs     32.4µs
  192    217.7µs     52.8µs       54µs    259.4µs     51.9µs
  256    513.4µs    122.1µs      154µs    608.4µs    143.2µs
  384      1.7ms    370.5µs    349.7µs        2ms      328µs
  512      4.1ms      830µs        1ms      4.8ms        1ms
  640        8ms      1.6ms      2.2ms      9.3ms      1.8ms
  768       14ms      2.8ms      3.6ms     16.1ms      3.1ms
  896     22.3ms      4.7ms      6.4ms       26ms      5.8ms
 1024     34.1ms        7ms      8.8ms     39.1ms      8.1ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.5µs      2.4µs      8.3µs      7.1µs        3µs
   64     10.2µs     10.2µs     24.9µs     35.2µs     13.3µs
   96     29.5µs     25.1µs     52.7µs    102.7µs     36.7µs
  128     59.2µs     41.7µs    142.7µs    239.8µs       82µs
  192    175.4µs     92.1µs    259.4µs    831.1µs    214.2µs
  256    385.1µs    162.9µs    606.3µs      1.9ms    503.2µs
  384      1.1ms      322µs      1.4ms      6.9ms      1.4ms
  512      2.7ms    683.1µs      3.5ms     15.7ms      3.2ms
  640      4.8ms      1.3ms      5.6ms     29.9ms      5.4ms
  768      8.2ms      2.4ms      9.1ms     52.9ms      9.1ms
  896     12.5ms      3.5ms     13.2ms     83.9ms     13.8ms
 1024     19.2ms      5.2ms     24.1ms    128.2ms     22.6ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.3µs     10.6µs      8.5µs      7.1µs        3µs
   64     10.7µs     25.4µs       25µs     35.2µs     13.3µs
   96     25.8µs     40.5µs     53.1µs    102.8µs       37µs
  128     39.5µs     64.2µs    143.1µs      242µs     82.1µs
  192    101.1µs     93.8µs    260.7µs    831.5µs    214.4µs
  256    189.1µs    140.6µs    620.8µs        2ms    504.1µs
  384    520.1µs    260.8µs      1.4ms      7.3ms      1.4ms
  512      1.1ms    444.9µs      3.5ms     17.4ms      3.2ms
  640        2ms    672.3µs      5.7ms     32.6ms      5.5ms
  768      3.2ms    962.5µs      9.2ms     55.6ms      9.2ms
  896      4.8ms      1.4ms     13.4ms     88.2ms     13.9ms
 1024      7.3ms      2.2ms     24.3ms      135ms     22.8ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.4µs      4.5µs      3.2µs      2.3µs      2.3µs
   64     12.4µs     12.4µs       38µs     11.6µs      8.8µs
   96     28.2µs     28.3µs     72.9µs     32.9µs     20.4µs
  128     40.8µs     40.9µs    121.9µs     77.1µs     38.2µs
  192    109.8µs    118.2µs    232.1µs    250.3µs     99.3µs
  256    189.3µs    175.5µs    539.3µs    594.5µs    203.6µs
  384    510.5µs    466.2µs      1.1ms      2.1ms    559.7µs
  512      1.2ms    669.5µs      3.7ms      5.5ms      1.2ms
  640        2ms      1.2ms      3.3ms     10.5ms      2.1ms
  768      3.4ms      1.8ms      5.4ms     17.9ms      3.6ms
  896      5.2ms      2.6ms      6.9ms     28.2ms      5.4ms
 1024      8.2ms      3.4ms     14.6ms     42.3ms      8.1ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.5µs      4.5µs      5.5µs      4.8µs      3.8µs
   64     20.1µs     17.8µs     17.3µs     22.2µs     15.2µs
   96     40.8µs     40.7µs     34.6µs     67.6µs     36.4µs
  128       92µs     82.7µs     95.2µs    159.5µs    125.1µs
  192    236.1µs    246.7µs      184µs    504.1µs    407.7µs
  256      491µs    457.9µs    313.2µs      1.3ms    802.3µs
  384      1.2ms      1.1ms      650µs      4.6ms      1.8ms
  512      2.6ms      2.3ms      1.2ms     11.4ms      4.4ms
  640      4.7ms      3.5ms      2.2ms       21ms      5.4ms
  768      7.7ms      5.4ms      3.3ms     36.1ms      8.5ms
  896     11.6ms        7ms      4.7ms     56.8ms       11ms
 1024     16.2ms     11.1ms      6.7ms     90.4ms       17ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.5µs     10.4µs          -     14.5µs     10.5µs
   64     43.8µs     43.9µs          -    105.7µs     66.5µs
   96    109.3µs    109.5µs          -    345.5µs    201.9µs
  128    225.5µs    227.4µs          -    816.7µs    450.8µs
  192    600.9µs    599.8µs          -      2.7ms      1.4ms
  256      1.4ms      1.4ms          -      6.5ms      3.2ms
  384      4.4ms      4.5ms          -       22ms     10.8ms
  512     11.3ms      8.6ms          -     52.9ms     26.4ms
  640     19.2ms     12.8ms          -    101.5ms     49.5ms
  768     32.4ms     17.9ms          -    175.3ms     85.3ms
  896     49.9ms     25.8ms          -    278.8ms    134.1ms
 1024     77.5ms     34.1ms          -    431.3ms      204ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.4µs     10.4µs     15.3µs        8µs      6.9µs
   64     34.9µs     34.9µs     60.1µs     43.8µs     44.2µs
   96     72.9µs       73µs    320.8µs    141.1µs     80.3µs
  128    128.6µs    128.9µs    807.8µs    330.6µs    156.5µs
  192      310µs      310µs      1.6ms      1.1ms    386.2µs
  256    629.8µs    680.4µs      3.4ms      2.5ms    817.9µs
  384      1.8ms      1.4ms      7.9ms      8.1ms      2.1ms
  512      4.2ms      2.2ms     16.1ms     19.2ms      4.5ms
  640      7.2ms      3.4ms     22.6ms     36.7ms        8ms
  768     11.9ms        5ms     40.5ms     61.9ms     13.2ms
  896     18.2ms      6.9ms     54.5ms       98ms     20.6ms
 1024     27.6ms      9.6ms       78ms    149.9ms     30.4ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32       10µs     10.1µs          -     17.8µs        9µs
   64     44.7µs     44.7µs          -    128.7µs     35.4µs
   96    109.2µs    109.3µs          -    423.4µs     98.1µs
  128      218µs      218µs          -    998.6µs    218.5µs
  192    634.9µs    634.9µs          -      3.3ms      628µs
  256      1.4ms      1.5ms          -      7.7ms      1.5ms
  384      4.8ms      3.4ms          -     25.7ms      5.9ms
  512       12ms      6.7ms          -       61ms     15.8ms
  640     22.5ms     10.4ms          -      118ms     28.8ms
  768       38ms     14.7ms          -    201.9ms     48.8ms
  896     61.1ms       20ms          -    326.8ms     77.2ms
 1024     91.9ms     26.6ms          -    497.8ms    119.6ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     13.8µs     29.7µs     10.6µs     21.1µs     10.6µs
   64     49.3µs     73.2µs     37.8µs     99.7µs     46.1µs
   96    128.6µs    119.9µs    157.4µs    281.6µs    119.8µs
  128    205.4µs    181.2µs    265.6µs    656.7µs    328.3µs
  192    585.9µs    425.5µs    647.2µs      2.2ms      949µs
  256      1.1ms    750.5µs      1.1ms      5.6ms        2ms
  384      3.2ms      1.6ms      2.3ms       19ms        5ms
  512      6.8ms      3.3ms      4.5ms     46.3ms     11.9ms
  640     12.3ms      6.2ms      7.2ms     86.1ms     19.1ms
  768     21.1ms      9.7ms     10.9ms    148.2ms     31.1ms
  896       32ms     13.8ms     16.5ms      231ms     44.4ms
 1024     47.3ms       21ms     23.7ms      368ms     68.7ms
```
