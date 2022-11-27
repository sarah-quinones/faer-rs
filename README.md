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
   64        8µs        8µs      7.8µs     10.8µs      5.1µs
   96     27.7µs     10.6µs     26.1µs     34.1µs     10.1µs
  128     65.7µs     16.5µs     35.2µs     79.1µs     32.6µs
  192    218.5µs     55.3µs     53.9µs      258µs     52.1µs
  256    515.3µs    123.8µs    154.2µs    604.2µs    143.4µs
  384      1.7ms    378.2µs    372.3µs        2ms    327.9µs
  512      4.1ms    849.5µs      1.3ms      4.7ms        1ms
  640        8ms      1.7ms      2.3ms      9.3ms      1.9ms
  768     13.9ms      2.9ms      3.6ms     16.1ms      3.2ms
  896     22.2ms      4.8ms      6.4ms     25.9ms      5.9ms
 1024       34ms      7.1ms        9ms     39.1ms      8.1ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.3µs      2.3µs      8.4µs      7.2µs      2.8µs
   64      9.8µs      9.8µs       25µs     34.6µs     13.6µs
   96     28.2µs     25.3µs     52.9µs    102.9µs     36.8µs
  128     57.3µs     40.9µs      143µs    240.9µs     81.8µs
  192    169.3µs     90.1µs    265.2µs    832.1µs    213.3µs
  256    371.5µs    161.3µs    622.6µs      1.9ms    505.6µs
  384      1.1ms    314.8µs      1.4ms      6.9ms      1.4ms
  512      2.6ms    676.7µs      3.5ms     15.8ms      3.2ms
  640      4.7ms      1.3ms      5.6ms       30ms      5.4ms
  768        8ms      2.4ms      9.1ms     52.9ms      9.1ms
  896     12.3ms      3.5ms     13.2ms     83.8ms     13.8ms
 1024     18.8ms      5.2ms     23.8ms    125.2ms     22.7ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      3.2µs     10.5µs      8.5µs      7.2µs      2.8µs
   64     10.2µs     25.2µs       25µs     34.9µs     13.6µs
   96     24.7µs     39.5µs       53µs    102.9µs     36.6µs
  128     37.8µs     63.5µs    143.5µs    243.5µs     81.8µs
  192     97.3µs     92.4µs    259.8µs    831.9µs    213.4µs
  256    181.6µs    142.2µs    624.5µs      1.9ms    505.4µs
  384    504.9µs    255.8µs      1.4ms      6.9ms      1.4ms
  512      1.1ms    436.3µs      3.5ms     15.8ms      3.2ms
  640      1.9ms      666µs      5.6ms       30ms      5.4ms
  768      3.2ms    948.6µs      9.2ms     52.9ms      9.1ms
  896      4.7ms      1.3ms     13.3ms     83.8ms     13.9ms
 1024      7.1ms      2.3ms     24.1ms    127.2ms     22.7ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.2µs      4.3µs      3.2µs      2.3µs      2.2µs
   64     11.9µs       12µs     36.9µs       11µs      8.5µs
   96     27.5µs     27.6µs     73.1µs     31.7µs     19.8µs
  128     39.7µs     39.9µs    122.9µs     77.2µs     36.2µs
  192    107.9µs    114.4µs      232µs    251.4µs     94.3µs
  256    183.7µs    174.6µs    542.5µs    607.8µs    194.1µs
  384    498.8µs    461.1µs      1.2ms      2.1ms    544.1µs
  512      1.1ms    668.4µs      3.7ms      5.6ms      1.2ms
  640        2ms      1.2ms      3.3ms     10.5ms      2.1ms
  768      3.4ms      1.8ms      5.4ms     18.2ms      3.5ms
  896      5.2ms      2.6ms      6.8ms     28.4ms      5.4ms
 1024        8ms      3.4ms     14.6ms     42.7ms      8.1ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.6µs      4.6µs      5.7µs      4.9µs      3.9µs
   64     17.4µs     17.5µs     17.3µs     22.1µs       15µs
   96     39.6µs     39.3µs     34.2µs     68.1µs     36.2µs
  128     81.7µs     75.4µs     96.4µs    161.2µs    125.1µs
  192    195.1µs    211.1µs    185.9µs    498.6µs    404.8µs
  256    414.6µs    403.6µs      314µs      1.3ms    836.4µs
  384      1.1ms    942.9µs    649.5µs      4.6ms      1.9ms
  512      2.4ms      1.7ms      1.4ms     11.4ms      4.2ms
  640      4.1ms      2.9ms      2.2ms       21ms      5.5ms
  768      6.7ms      4.3ms      3.3ms     36.3ms      8.8ms
  896     10.2ms      6.2ms      4.7ms     56.9ms     11.4ms
 1024     15.2ms      8.4ms      6.7ms     89.9ms     18.1ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     10.3µs    190.6µs          -     14.7µs     10.1µs
   64     43.5µs    400.4µs          -    105.5µs     68.3µs
   96    108.2µs    649.3µs          -    346.9µs    203.5µs
  128    225.7µs    972.3µs          -    827.1µs    452.6µs
  192    591.6µs      1.7ms          -      2.7ms      1.4ms
  256      1.4ms      3.3ms          -      6.6ms      3.3ms
  384      4.4ms      6.4ms          -       22ms     10.7ms
  512     11.3ms     11.1ms          -     52.9ms     26.2ms
  640     19.2ms     17.4ms          -    101.6ms     49.1ms
  768     32.5ms       24ms          -    175.6ms     84.8ms
  896     49.8ms     35.2ms          -    281.5ms    133.6ms
 1024     77.8ms     45.1ms          -      425ms    205.5ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     13.5µs     13.5µs     15.3µs        8µs        7µs
   64     38.5µs     38.5µs     60.3µs     43.7µs     45.6µs
   96     75.3µs     75.3µs    317.8µs    143.1µs     79.1µs
  128    129.5µs    160.5µs    921.8µs    329.3µs    155.3µs
  192      305µs    443.2µs      2.1ms      1.1ms    383.6µs
  256    616.7µs    580.9µs      3.8ms      2.5ms    833.5µs
  384      1.7ms      1.2ms        8ms      8.2ms      2.1ms
  512        4ms      2.3ms     16.4ms     19.5ms      4.5ms
  640        7ms      3.4ms     22.7ms     37.5ms      8.1ms
  768     11.7ms      4.9ms     40.7ms     63.2ms     13.4ms
  896     17.9ms      6.8ms     54.8ms    101.2ms       21ms
 1024     26.8ms      9.4ms     78.5ms    154.2ms     30.8ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      9.9µs    187.2µs          -     18.2µs      9.1µs
   64     44.9µs      397µs          -    128.9µs     37.2µs
   96    108.2µs    625.3µs          -    427.1µs     99.7µs
  128    216.2µs      874µs          -        1ms    217.8µs
  192    620.5µs      1.5ms          -      3.3ms    648.4µs
  256      1.4ms      2.1ms          -      7.7ms      1.5ms
  384      4.9ms      4.2ms          -     25.5ms      5.7ms
  512     12.3ms      7.5ms          -     60.4ms     14.8ms
  640     23.2ms     14.9ms          -    116.9ms     26.4ms
  768     39.1ms     20.6ms          -    200.6ms     45.2ms
  896     62.9ms     28.3ms          -    319.1ms     70.4ms
 1024     94.8ms     36.8ms          -    493.1ms      110ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     13.9µs     30.5µs     10.7µs     21.2µs     10.4µs
   64       49µs     73.1µs     38.1µs     99.7µs     45.1µs
   96      124µs    124.5µs    157.9µs    285.7µs    118.8µs
  128    225.6µs    205.5µs    279.3µs    653.9µs    326.3µs
  192    583.7µs    429.2µs    643.2µs      2.2ms    941.1µs
  256      1.2ms    796.8µs      1.1ms      5.6ms        2ms
  384      3.2ms      1.7ms      2.3ms     19.1ms      5.1ms
  512      7.1ms      3.7ms      4.5ms     46.3ms     11.7ms
  640     12.7ms      6.2ms      7.2ms     86.4ms     19.2ms
  768     21.4ms     10.3ms     10.9ms    148.6ms     31.2ms
  896     32.6ms       15ms     16.5ms    231.4ms     44.2ms
 1024       48ms     21.9ms     23.6ms    366.6ms     69.3ms
```
