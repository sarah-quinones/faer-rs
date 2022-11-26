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

# Benchmarks

The benchmarks were run on an `11th Gen Intel(R) Core(TM) i5-11400 @ 2.60GHz` with 12 threads.  
- `nalgebra` is used with the `matrixmultiply` backend
- `ndarray` is used with the `openblas` backend
- `eigen` is compiled with `-march=native -O3 -fopenmp`

All computations are done on `f64` and column major matrices.

## Matrix multiplication

Multiplication of two square matrices of dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      1.4µs      1.1µs      1.1µs      1.9µs      1.34µs
   64        8µs        8µs      7.8µs     10.9µs      5.47µs
   96     27.7µs       11µs     26.1µs     34.4µs      12.8µs
  128     65.2µs     17.5µs     37.9µs       79µs      41.4µs
  192    218.2µs     70.3µs     52.8µs    260.7µs      64.3µs
  256    515.1µs    136.8µs    208.3µs    599.2µs       185µs
  384      1.7ms    415.2µs    455.8µs        2ms       429µs
  512      5.8ms    979.5µs      1.3ms      4.7ms      1.26ms
  640      8.6ms      2.1ms      2.4ms      9.3ms      2.72ms
  768     15.8ms      3.7ms      3.6ms     16.3ms      4.56ms
  896       25ms      6.2ms      6.1ms     26.3ms       6.6ms
 1024       36ms      9.7ms      9.2ms     39.3ms      8.58ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      2.3µs      2.3µs      8.4µs      7.2µs      4.26µs
   64      9.8µs       10µs     26.7µs     35.1µs      18.4µs
   96     29.1µs     25.8µs     56.1µs      103µs        47µs
  128     57.9µs     40.8µs    153.4µs    248.1µs       101µs
  192    170.7µs     93.1µs    275.4µs    834.9µs       257µs
  256    371.7µs    170.4µs    664.3µs      1.9ms       580µs
  384      1.1ms      339µs      1.5ms      7.2ms      1.56ms
  512      2.6ms    716.9µs      3.7ms     17.4ms      3.83ms
  640      4.8ms      1.4ms      5.8ms     32.9ms      6.27ms
  768      8.1ms      2.1ms      9.7ms     57.5ms      10.3ms
  896     12.5ms      3.6ms     13.1ms     89.8ms        15ms
 1024     19.4ms      5.2ms       25ms    144.8ms      24.4ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      3.2µs     10.6µs      8.5µs      7.2µs      3.95µs
   64     10.3µs     25.3µs     26.5µs     35.1µs      18.6µs
   96     24.7µs     39.3µs     55.7µs    102.8µs      47.1µs
  128       38µs     63.7µs    151.5µs    247.6µs       101µs
  192     97.2µs     93.4µs    270.9µs    834.2µs       258µs
  256    180.9µs    140.5µs    648.9µs      1.9ms       564µs
  384    506.3µs    253.1µs      1.4ms      6.8ms      1.48ms
  512      1.1ms    448.5µs      3.6ms     15.8ms      3.61ms
  640        2ms    690.5µs      5.6ms     30.6ms      5.86ms
  768      3.2ms      1.2ms      9.4ms     54.1ms      9.79ms
  896      4.8ms      1.9ms     12.7ms       85ms      14.8ms
 1024      7.3ms      2.6ms     24.6ms    133.1ms        25ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      3.4µs      3.5µs      3.3µs      2.4µs      2.73µs
   64     10.2µs     10.3µs     41.5µs     11.2µs      9.78µs
   96     24.8µs     24.9µs     79.1µs     32.2µs        22µs
  128     35.7µs       36µs    129.3µs     79.9µs      38.6µs
  192    101.2µs    111.6µs    285.5µs    255.6µs      98.5µs
  256    174.9µs      165µs    712.1µs    607.8µs       201µs
  384    485.6µs    450.1µs      1.3ms      2.1ms       537µs
  512      1.1ms    654.8µs      3.8ms      5.5ms       1.2ms
  640      1.9ms      1.3ms      3.4ms     10.4ms      2.05ms
  768      3.4ms      1.9ms      5.6ms       18ms       3.5ms
  896      5.2ms      2.9ms      6.9ms     28.1ms      5.31ms
 1024      8.5ms      3.6ms     15.4ms     43.1ms      8.06ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      4.5µs      4.6µs      5.6µs      4.9µs      4.33µs
   64     17.3µs     17.3µs     17.4µs     22.2µs      16.7µs
   96     38.9µs     39.1µs     34.7µs     67.7µs      39.1µs
  128     75.9µs       83µs    102.7µs      161µs       117µs
  192    197.6µs    231.6µs    197.6µs      503µs       316µs
  256    408.6µs    395.6µs    330.5µs      1.3ms       729µs
  384      1.1ms    951.2µs    886.1µs      4.6ms      1.69ms
  512      2.3ms      1.7ms      1.6ms     11.2ms      3.42ms
  640      4.1ms      2.9ms      2.3ms     21.1ms      4.97ms
  768      6.8ms      4.4ms      3.6ms     36.4ms      7.52ms
  896     10.2ms      6.5ms      4.9ms     57.2ms      10.6ms
 1024     15.6ms      8.9ms        7ms     88.5ms      16.6ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32     10.4µs    198.3µs          -       15µs      9.88µs
   64     43.6µs    423.7µs          -    105.5µs      56.8µs
   96    108.4µs    690.8µs          -    351.5µs       174µs
  128    224.6µs        1ms          -    830.6µs       399µs
  192    590.9µs      1.8ms          -      2.8ms      1.21ms
  256      1.4ms      2.8ms          -      6.6ms      2.89ms
  384      4.6ms      6.7ms          -     22.2ms      9.37ms
  512     11.3ms     11.2ms          -     53.3ms      23.5ms
  640     19.3ms     17.9ms          -    102.7ms      43.4ms
  768     32.8ms     25.4ms          -    177.5ms      74.8ms
  896     49.9ms     36.2ms          -    287.8ms       118ms
 1024     80.2ms     46.8ms          -    431.1ms       180ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32     13.9µs     13.4µs     15.7µs      8.1µs      7.01µs
   64       38µs       38µs     61.3µs     43.6µs      47.4µs
   96     74.7µs     74.7µs    356.7µs    140.1µs      83.9µs
  128      129µs    161.1µs      1.1ms    327.9µs       165µs
  192    305.6µs      370µs      2.2ms        1ms       396µs
  256    632.5µs    565.1µs      3.8ms      2.5ms       816µs
  384      1.9ms      1.2ms      8.2ms      8.1ms      2.15ms
  512      4.7ms      2.3ms     16.7ms     19.1ms      4.51ms
  640      7.7ms      3.5ms     23.6ms     36.5ms      8.16ms
  768     13.2ms        5ms     42.5ms     62.3ms      13.3ms
  896     19.8ms        7ms     57.6ms    101.3ms      20.6ms
 1024     30.9ms     11.2ms     81.6ms    153.5ms      30.5ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32       10µs    195.2µs          -     18.2µs      9.26µs
   64     44.5µs    417.8µs          -    128.6µs      37.5µs
   96    108.5µs    656.2µs          -    422.7µs       106µs
  128    218.2µs    915.4µs          -    999.1µs       228µs
  192    617.6µs      1.5ms          -      3.3ms       674µs
  256      1.4ms      2.2ms          -      7.7ms      1.59ms
  384      4.8ms      5.4ms          -     25.8ms      5.56ms
  512     11.8ms      9.7ms          -     61.1ms      14.6ms
  640     22.3ms     15.7ms          -    118.2ms      26.4ms
  768     37.9ms     21.9ms          -    203.4ms        46ms
  896     60.7ms     29.8ms          -    328.8ms      71.2ms
 1024     97.4ms     40.2ms          -    491.8ms       114ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32     13.8µs     31.4µs     10.4µs     21.1µs        13µs
   64     48.6µs     73.3µs     38.2µs    100.5µs      55.5µs
   96    120.6µs    126.3µs    164.8µs    283.8µs       140µs
  128    222.4µs    204.2µs    354.4µs    653.4µs       347µs
  192    584.8µs    428.8µs    682.1µs      2.2ms      1.06ms
  256      1.2ms    869.4µs      1.2ms      5.8ms      2.12ms
  384      3.2ms      1.8ms      2.5ms     19.1ms      5.55ms
  512      7.2ms      3.9ms      4.7ms     44.7ms      12.2ms
  640     12.8ms      6.8ms      7.7ms     85.3ms      20.2ms
  768     21.8ms       11ms     11.7ms    144.2ms      32.8ms
  896     32.9ms     16.5ms     19.2ms    226.6ms      48.4ms
 1024     49.3ms     23.1ms     24.9ms    348.7ms      80.6ms
```
