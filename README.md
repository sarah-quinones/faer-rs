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
   32      1.2µs      1.1µs      1.1µs      1.9µs      1.2µs
   64        8µs        8µs      7.8µs     10.8µs        5µs
   96     27.5µs     10.9µs     26.1µs       34µs     10.1µs
  128     65.2µs       17µs     35.2µs     79.2µs     32.6µs
  192    216.1µs     54.1µs     53.7µs      258µs     51.8µs
  256    509.8µs    118.1µs    154.9µs    603.9µs      155µs
  384      1.7ms    368.5µs    346.3µs        2ms    327.4µs
  512      4.1ms    846.8µs      1.2ms      4.7ms      1.2ms
  640      7.9ms      1.7ms      2.3ms      9.3ms        2ms
  768       14ms        3ms      3.6ms     16.1ms      3.2ms
  896     22.3ms      4.9ms      6.5ms     25.9ms      5.9ms
 1024     34.3ms      7.1ms        9ms     39.3ms      8.3ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.4µs      2.4µs      8.1µs        7µs      2.8µs
   64       10µs       10µs     25.7µs     34.5µs     13.7µs
   96     29.1µs       27µs     54.3µs    100.9µs     36.7µs
  128       58µs     42.9µs    144.7µs    232.3µs     81.1µs
  192    172.5µs     93.2µs    262.6µs    809.7µs    213.8µs
  256    374.3µs    165.2µs    648.9µs      1.9ms    487.2µs
  384      1.1ms    333.8µs      1.4ms      6.7ms      1.4ms
  512      2.6ms    683.4µs      3.5ms       15ms      3.2ms
  640      4.8ms      1.6ms      5.6ms     29.1ms      5.4ms
  768      8.1ms      2.4ms      9.2ms     51.8ms      9.1ms
  896     12.4ms      3.6ms     13.4ms     81.9ms     13.8ms
 1024     18.9ms      5.3ms     25.2ms    124.9ms     22.8ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.3µs     10.9µs      8.2µs        7µs      2.9µs
   64     10.7µs     17.9µs     25.6µs     34.4µs     13.7µs
   96     25.8µs     36.8µs     54.1µs      101µs     36.8µs
  128     39.2µs     51.1µs    144.8µs    234.7µs     81.2µs
  192     99.9µs     91.4µs    262.3µs    807.4µs    213.8µs
  256    184.6µs    140.3µs    647.7µs      1.9ms    487.2µs
  384    511.3µs    265.1µs      1.4ms      6.7ms      1.4ms
  512      1.1ms    442.3µs      3.5ms       15ms      3.2ms
  640        2ms    633.1µs      5.7ms     29.1ms      5.4ms
  768      3.2ms    985.7µs      9.2ms     51.8ms      9.1ms
  896      4.7ms      2.3ms     13.4ms     81.9ms     13.8ms
 1024      7.1ms      2.4ms     25.1ms      123ms     22.8ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.3µs      4.3µs      3.4µs      2.6µs      2.3µs
   64     12.1µs     12.1µs     36.8µs     12.8µs      8.7µs
   96       28µs       28µs     73.5µs     37.5µs     20.3µs
  128     40.3µs     40.3µs    135.7µs     89.1µs     38.4µs
  192    109.3µs    117.9µs    411.3µs    291.4µs     99.7µs
  256      185µs    175.8µs    700.2µs    680.1µs    207.3µs
  384    500.4µs    472.1µs      1.2ms      2.3ms    564.2µs
  512      1.1ms    675.8µs      3.7ms      6.3ms      1.2ms
  640        2ms      1.3ms      3.3ms     11.9ms      2.1ms
  768      3.4ms      1.8ms      5.3ms     20.7ms      3.6ms
  896      5.2ms      2.7ms      6.9ms     32.1ms      5.5ms
 1024        8ms      3.4ms     14.8ms     47.4ms      8.3ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      5.6µs      5.6µs      5.6µs        5µs      3.8µs
   64     23.2µs     23.2µs     17.6µs     22.2µs     15.1µs
   96     50.3µs     52.8µs     34.4µs     69.2µs     36.2µs
  128     98.2µs    102.4µs     96.8µs    172.7µs    130.5µs
  192    258.1µs      285µs    187.6µs    550.5µs    429.9µs
  256    510.8µs    538.2µs      318µs      1.4ms      838µs
  384      1.3ms      1.3ms    747.1µs      4.7ms      1.9ms
  512      2.9ms      2.4ms      2.1ms     11.7ms      4.3ms
  640      4.9ms      3.8ms      2.3ms     21.5ms      5.7ms
  768      7.8ms      5.7ms      3.3ms     37.2ms      8.7ms
  896     11.5ms      8.5ms      4.7ms     58.8ms     11.3ms
 1024     17.5ms     11.5ms      6.7ms     91.5ms     17.2ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     12.7µs     12.6µs          -     14.8µs     10.8µs
   64     52.2µs     52.2µs          -    105.3µs     68.8µs
   96    127.2µs    127.2µs          -    349.9µs    196.6µs
  128    257.9µs    259.2µs          -    844.8µs    455.1µs
  192      654µs    652.4µs          -      2.8ms      1.4ms
  256      1.5ms      1.5ms          -      6.6ms      3.3ms
  384      4.6ms      4.7ms          -       22ms     10.7ms
  512     11.7ms        9ms          -     52.7ms     26.5ms
  640     19.8ms     13.1ms          -    101.5ms     49.7ms
  768     33.1ms     18.6ms          -    175.5ms     85.8ms
  896     50.8ms       26ms          -    278.2ms    135.1ms
 1024     79.9ms     35.8ms          -    426.1ms    204.5ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     12.6µs     12.6µs     15.5µs      7.7µs      6.9µs
   64       37µs       37µs     60.4µs     43.2µs     44.3µs
   96     74.9µs       75µs    324.6µs    141.1µs     79.3µs
  128    132.1µs    132.2µs    825.1µs    321.1µs    156.7µs
  192    338.5µs    338.5µs        2ms      1.1ms    384.4µs
  256    678.7µs    733.7µs      7.3ms      2.5ms    800.8µs
  384      1.9ms      1.8ms      8.2ms      8.1ms      2.1ms
  512      4.2ms      3.2ms     16.2ms     19.4ms      4.5ms
  640      7.6ms      4.7ms     22.8ms       37ms      8.2ms
  768     12.5ms        7ms     41.7ms     62.6ms     13.4ms
  896       19ms      9.8ms     56.4ms      100ms     20.7ms
 1024     28.3ms     13.8ms     80.5ms    150.7ms     30.8ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     26.4µs     39.3µs          -     17.9µs      9.1µs
   64    107.5µs    136.6µs          -    127.1µs     36.3µs
   96    241.1µs    272.7µs          -    420.2µs       99µs
  128    447.5µs    490.7µs          -      981µs    222.4µs
  192      1.2ms      1.2ms          -      3.3ms    623.3µs
  256      2.3ms      2.2ms          -      7.7ms      1.6ms
  384      6.9ms      4.6ms          -     25.5ms      5.8ms
  512     15.7ms      8.3ms          -     60.8ms     16.2ms
  640     28.8ms     12.5ms          -    117.3ms     28.7ms
  768     47.5ms     17.4ms          -    201.2ms     49.3ms
  896     73.5ms     23.5ms          -    321.2ms     78.3ms
 1024    107.2ms       42ms          -    500.3ms    121.9ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     14.7µs     33.2µs     10.4µs     21.1µs     10.5µs
   64     53.5µs       79µs       38µs       99µs     45.6µs
   96    135.3µs    141.8µs    156.2µs    285.8µs      119µs
  128    221.6µs    209.7µs    333.6µs    658.1µs      343µs
  192    595.3µs      486µs    653.5µs      2.2ms    974.8µs
  256      1.2ms      813µs      1.1ms      5.6ms        2ms
  384      3.2ms      1.9ms      2.3ms     20.4ms      5.1ms
  512      6.9ms      3.7ms      4.6ms     47.8ms       12ms
  640     12.6ms      6.6ms      7.3ms     92.5ms     19.4ms
  768     21.1ms     10.5ms       11ms    156.5ms     31.2ms
  896     31.7ms     14.8ms     16.7ms    242.9ms     44.4ms
 1024     46.8ms     21.3ms     23.8ms    377.4ms     69.7ms
```

## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32    146.8µs    166.1µs     91.5µs    102.5µs    239.4µs
   64    484.5µs    444.5µs    613.1µs    565.9µs    984.2µs
   96      1.1ms      1.1ms      1.7ms      1.7ms      2.6ms
  128      1.9ms      1.9ms      2.9ms      4.6ms      4.3ms
  192      4.6ms      4.4ms      6.6ms     14.9ms      9.7ms
  256      8.8ms      7.6ms     11.3ms     45.3ms     17.4ms
  384     23.1ms     16.2ms     25.1ms    121.7ms     42.4ms
  512     49.6ms     29.7ms     50.6ms    450.1ms     80.7ms
  640     85.8ms     47.5ms     78.1ms    655.2ms    127.2ms
  768    139.8ms     70.9ms    121.6ms      1.42s    196.7ms
  896    210.5ms     99.2ms    170.8ms      2.09s    277.1ms
 1024      313ms      136ms    270.8ms       3.8s    412.6ms
```

## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      1.3ms      1.4ms      5.2ms      5.1ms        3ms
   64      3.7ms      3.5ms     15.3ms     20.2ms      8.1ms
   96      7.3ms      6.2ms     30.3ms     44.7ms     17.1ms
  128       12ms      9.4ms     47.5ms     79.3ms     28.3ms
  192     24.9ms     17.9ms     62.7ms    181.6ms     54.8ms
  256     42.9ms     28.1ms     83.6ms    367.8ms     92.3ms
  384     93.8ms     51.8ms    132.5ms    912.4ms      206ms
  512    167.8ms       84ms    299.9ms      2.01s    385.4ms
  640    263.8ms      125ms    289.9ms      3.25s    625.2ms
  768    388.5ms    190.6ms    439.3ms      5.18s      912ms
  896    540.6ms    260.4ms    551.6ms      7.26s       1.3s
 1024    733.6ms    335.6ms    849.9ms     10.68s      1.71s
```
