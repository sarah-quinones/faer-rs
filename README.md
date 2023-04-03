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
   64        8µs        8µs      7.9µs     10.8µs        5µs
   96     27.7µs       11µs     26.2µs     34.2µs       10µs
  128     64.9µs     17.1µs     35.5µs     80.6µs     32.6µs
  192    216.5µs     53.9µs     69.3µs    261.4µs       52µs
  256      510µs    116.5µs    205.4µs    603.7µs    143.1µs
  384      1.7ms    358.3µs    440.5µs        2ms    327.6µs
  512      4.1ms    841.7µs      1.3ms      4.7ms      1.2ms
  640        8ms      1.7ms      2.3ms      9.2ms        2ms
  768       14ms      3.3ms      3.6ms     16.1ms      3.2ms
  896     22.3ms      5.6ms      6.6ms     25.9ms      5.9ms
 1024     34.2ms      8.2ms      9.7ms     38.8ms      8.3ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.5µs      2.4µs      9.2µs        7µs      2.9µs
   64     10.3µs     10.4µs     25.9µs     34.6µs     13.4µs
   96     29.2µs     25.2µs     54.5µs    100.9µs     36.5µs
  128     58.4µs     41.7µs    145.5µs    237.3µs     82.6µs
  192    173.2µs     93.3µs    265.6µs    802.9µs    212.9µs
  256    375.6µs    165.2µs    647.7µs      1.8ms    494.5µs
  384      1.1ms    334.7µs      1.4ms      6.5ms      1.3ms
  512      2.6ms    670.7µs      3.5ms     15.7ms      3.2ms
  640      4.8ms      1.5ms      5.7ms     30.2ms      5.5ms
  768      8.1ms      2.4ms      9.4ms     51.8ms      9.3ms
  896     12.4ms      3.6ms     13.4ms     81.9ms       14ms
 1024       19ms      5.3ms     19.9ms    123.3ms     22.9ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.5µs     10.5µs      8.6µs        7µs      2.9µs
   64     11.2µs       18µs     25.8µs     34.3µs     13.8µs
   96     26.8µs     37.1µs     54.4µs    100.9µs     36.5µs
  128     40.2µs     50.1µs    145.6µs    237.5µs     82.7µs
  192    102.5µs     88.3µs      263µs    804.1µs      213µs
  256    186.7µs    138.7µs    643.9µs      1.9ms    494.5µs
  384    518.9µs    268.1µs      1.4ms      6.7ms      1.3ms
  512      1.1ms    447.4µs      3.5ms     16.3ms      3.2ms
  640        2ms    634.1µs      5.6ms     31.1ms      5.5ms
  768      3.2ms    929.2µs      9.4ms     53.2ms      9.3ms
  896      4.8ms      2.2ms     13.4ms     83.9ms       14ms
 1024      7.2ms      2.5ms       20ms    127.1ms     22.9ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.9µs        5µs      3.2µs      2.6µs      2.3µs
   64     13.2µs     13.3µs       38µs       13µs      8.8µs
   96     30.2µs     30.2µs     74.5µs     37.1µs     20.3µs
  128       43µs     43.1µs    217.2µs     89.3µs     37.9µs
  192    113.5µs    123.9µs    286.6µs      295µs     98.6µs
  256    188.4µs      178µs    694.5µs    679.2µs    205.5µs
  384    504.2µs    481.9µs      1.2ms      2.2ms    556.7µs
  512      1.1ms    684.5µs      3.7ms      5.6ms      1.2ms
  640        2ms      1.3ms      3.3ms     10.8ms      2.1ms
  768      3.4ms      1.9ms      5.6ms     18.4ms      3.5ms
  896      5.2ms      2.7ms      6.9ms     28.7ms      5.4ms
 1024        8ms      3.4ms     14.9ms     42.8ms      8.2ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      6.3µs      6.3µs      5.7µs      5.1µs      3.9µs
   64     34.8µs     29.3µs     17.5µs     22.5µs     15.2µs
   96     68.6µs     68.5µs     34.5µs     67.4µs     36.4µs
  128    138.1µs    153.2µs     96.8µs    160.3µs    128.9µs
  192    336.5µs    374.2µs    186.5µs      528µs      429µs
  256    667.4µs    719.9µs    309.6µs      1.4ms    840.6µs
  384      1.9ms      1.7ms    665.8µs      4.6ms      1.9ms
  512      3.8ms      3.3ms      1.9ms     11.8ms      4.4ms
  640      6.3ms        6ms      2.2ms     21.6ms      5.6ms
  768      9.1ms      7.8ms      3.4ms     37.4ms      8.7ms
  896     14.6ms     11.4ms      4.7ms     58.5ms     11.3ms
 1024     19.7ms       16ms      6.8ms     90.9ms     17.4ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     12.1µs     12.1µs          -     14.7µs     12.4µs
   64     50.1µs     50.5µs          -    105.6µs     72.6µs
   96    123.3µs    123.3µs          -    345.2µs    206.7µs
  128      253µs    251.1µs          -    826.1µs    465.8µs
  192    645.8µs    642.5µs          -      2.7ms      1.4ms
  256      1.5ms      1.5ms          -      6.6ms      3.4ms
  384      4.7ms      4.5ms          -     21.9ms     10.9ms
  512     12.1ms      9.1ms          -     52.6ms     26.6ms
  640     20.3ms     13.8ms          -    101.5ms     49.7ms
  768     33.4ms     19.6ms          -    175.4ms     86.1ms
  896     50.7ms     28.2ms          -    277.8ms    134.2ms
 1024     79.4ms     38.6ms          -    424.8ms    204.1ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     14.3µs     14.3µs     15.6µs      7.9µs      7.7µs
   64     40.2µs     40.3µs     60.5µs     43.3µs     44.6µs
   96     80.3µs     79.8µs      325µs    138.9µs     78.4µs
  128    139.3µs    139.3µs    827.2µs    325.3µs    154.7µs
  192    351.6µs    350.8µs      1.7ms        1ms    383.5µs
  256    699.1µs    755.8µs      4.9ms      2.5ms    798.8µs
  384        2ms      1.8ms      8.1ms        8ms      2.1ms
  512      4.2ms      3.2ms     15.5ms     18.6ms      4.5ms
  640      7.6ms      4.8ms     22.6ms     35.9ms        8ms
  768     12.6ms      7.1ms     34.7ms       61ms     13.2ms
  896     19.1ms      9.8ms     46.1ms     97.1ms     20.5ms
 1024     28.4ms       14ms     66.6ms    146.9ms     30.5ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     31.3µs     44.4µs          -     17.9µs      9.2µs
   64    129.8µs      152µs          -    127.8µs     36.4µs
   96    287.4µs    333.5µs          -    422.4µs     97.9µs
  128      545µs    599.6µs          -    990.9µs    226.8µs
  192      1.3ms      1.4ms          -      3.3ms    634.5µs
  256      2.7ms      2.5ms          -      7.7ms      1.5ms
  384      7.4ms      4.7ms          -     25.5ms      5.5ms
  512     16.1ms      7.9ms          -     59.8ms     14.7ms
  640     29.4ms     12.6ms          -    116.3ms     26.3ms
  768     47.8ms     17.5ms          -      200ms       46ms
  896     74.3ms     23.7ms          -    317.7ms     70.5ms
 1024    109.5ms     37.1ms          -    484.1ms    113.1ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32       16µs     36.2µs     10.7µs     21.1µs     10.6µs
   64     67.2µs     92.4µs       38µs     98.3µs     45.5µs
   96    153.4µs    161.3µs    154.8µs    287.4µs    118.6µs
  128    258.8µs    272.4µs    262.8µs    673.6µs    347.7µs
  192    680.8µs    573.2µs    629.1µs      2.3ms      968µs
  256      1.3ms      1.1ms      1.1ms      5.6ms        2ms
  384      3.8ms      2.3ms      2.3ms       19ms        5ms
  512      7.4ms      4.5ms      4.4ms     45.3ms       12ms
  640     13.6ms      8.2ms      7.3ms     85.6ms     19.3ms
  768     22.3ms     12.7ms     11.1ms    146.2ms     30.9ms
  896     34.7ms     17.7ms     16.6ms    229.1ms     44.2ms
 1024     50.3ms     26.2ms     23.9ms    356.6ms     68.8ms
```

## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32    158.2µs    186.8µs     89.3µs     98.3µs    228.7µs
   64    544.4µs    496.2µs    679.1µs    565.3µs        1ms
   96      1.2ms      1.2ms      1.7ms      1.7ms      2.5ms
  128      2.1ms        2ms      2.9ms      4.5ms      4.2ms
  192        5ms      4.7ms      6.7ms     14.9ms      9.9ms
  256      9.4ms      7.9ms     11.6ms     46.5ms     17.1ms
  384     24.2ms     16.7ms     25.6ms    122.1ms       42ms
  512     52.2ms     30.2ms     51.3ms    454.5ms     82.8ms
  640     89.4ms     48.7ms       78ms    657.1ms    129.3ms
  768    144.2ms     85.8ms    121.5ms      1.42s    202.5ms
  896    217.9ms    118.5ms    170.9ms      2.08s    281.6ms
 1024    322.8ms    163.2ms    249.8ms      3.86s    417.5ms
```

## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      1.4ms      1.4ms      5.3ms      5.2ms        3ms
   64      3.8ms      3.6ms       15ms     20.3ms        8ms
   96      7.5ms      6.4ms     29.5ms     44.6ms     17.1ms
  128     12.3ms      9.5ms     47.3ms     78.9ms     30.7ms
  192     25.5ms     18.3ms     61.5ms    180.9ms       57ms
  256     43.8ms     28.5ms     83.8ms      362ms    100.9ms
  384     95.4ms     52.4ms    131.9ms    904.7ms    219.7ms
  512    170.8ms     83.9ms    303.6ms      2.02s    404.2ms
  640    269.9ms    124.9ms    287.6ms      3.24s    646.2ms
  768      395ms    191.5ms    438.8ms      5.25s    947.8ms
  896    550.5ms    263.2ms    544.4ms       7.3s      1.32s
 1024    744.7ms    339.1ms    844.6ms     10.62s      1.74s
```
