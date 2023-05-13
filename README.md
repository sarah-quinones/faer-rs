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

The Cholesky module implements the LLT and LDLT matrix decompositions. These allow for solving symmetric/Hermitian (+positive definite for LLT) linear systems.

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

The SVD module implements the singular value decomposition.

## faer-evd

[![Documentation](https://docs.rs/faer-evd/badge.svg)](https://docs.rs/faer-evd)
[![Crate](https://img.shields.io/crates/v/faer-evd.svg)](https://crates.io/crates/faer-evd)

The EVD module implements the eigenvalue decomposition for Hermitian and non Hermitian matrices .

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
   32      1.1µs      1.1µs      1.1µs      1.7µs      1.2µs
   64        8µs        8µs      7.8µs     10.6µs      5.1µs
   96     27.7µs     10.9µs     26.2µs     34.8µs       10µs
  128     65.3µs       17µs     35.3µs     78.8µs     32.7µs
  192    216.4µs     53.8µs     66.6µs    262.3µs     51.8µs
  256    510.8µs    117.4µs    202.1µs    604.8µs    142.9µs
  384      1.7ms    339.4µs    437.9µs        2ms      327µs
  512        4ms    787.8µs      1.3ms      4.7ms      1.2ms
  640      7.9ms      1.6ms      2.3ms      9.2ms        2ms
  768     13.8ms      2.9ms      3.6ms       16ms      3.2ms
  896     22.1ms      4.6ms      6.5ms     25.8ms      5.9ms
 1024     33.9ms      6.6ms      9.7ms       39ms      8.3ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.3µs      2.3µs      8.5µs      7.1µs      2.9µs
   64       10µs     10.2µs       26µs     34.4µs     13.4µs
   96     28.5µs     24.1µs       55µs    101.1µs     36.8µs
  128     57.8µs       40µs    145.1µs    232.8µs       82µs
  192      170µs     93.2µs    263.7µs    783.1µs    213.9µs
  256    371.8µs    166.5µs    650.7µs      1.9ms    494.7µs
  384      1.1ms    325.1µs      1.4ms      7.2ms      1.3ms
  512      2.6ms    664.3µs      3.6ms     16.9ms      3.2ms
  640      4.7ms      1.5ms      5.7ms     33.3ms      5.5ms
  768        8ms      2.4ms      9.5ms     55.7ms      9.3ms
  896     12.3ms      3.6ms     13.7ms     88.8ms       14ms
 1024     18.8ms      5.2ms     20.1ms    130.4ms     22.8ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32        5µs     12.4µs      8.5µs      7.1µs      2.9µs
   64     13.9µs     19.7µs     25.9µs     34.3µs     13.4µs
   96     32.6µs     40.3µs     54.8µs    101.1µs     36.9µs
  128     45.7µs     53.1µs      145µs    232.8µs       82µs
  192      115µs     97.3µs    262.5µs    782.2µs    213.6µs
  256    199.1µs    143.9µs      641µs      1.9ms    493.7µs
  384    544.4µs    279.5µs      1.4ms      6.4ms      1.3ms
  512      1.1ms    456.9µs      3.5ms     15.6ms      3.2ms
  640        2ms    653.6µs      5.6ms     30.2ms      5.5ms
  768      3.2ms    956.5µs      9.3ms     51.7ms      9.3ms
  896      4.8ms      1.5ms     13.4ms     81.8ms       14ms
 1024      7.2ms      2.4ms     19.9ms    122.5ms     22.6ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      2.2µs      2.3µs      3.2µs        2µs      2.2µs
   64      7.7µs        8µs     38.2µs     10.1µs      8.6µs
   96     20.4µs     20.6µs     95.3µs     29.7µs     19.8µs
  128       32µs     32.3µs      298µs     74.3µs     36.2µs
  192     90.8µs     99.9µs    301.1µs    252.5µs     94.7µs
  256    166.7µs    157.4µs    694.4µs      610µs      197µs
  384    470.7µs    451.1µs      1.2ms      2.1ms    543.6µs
  512      1.1ms    640.8µs      3.7ms      5.1ms      1.2ms
  640      1.9ms      1.3ms      3.2ms       10ms      2.1ms
  768      3.3ms      1.8ms      5.5ms     17.3ms      3.5ms
  896      5.1ms      2.7ms      6.9ms     27.4ms      5.4ms
 1024      7.8ms      3.4ms     14.5ms     40.4ms      8.1ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      4.2µs      4.3µs      5.7µs      4.9µs      3.8µs
   64     16.7µs     17.4µs     17.5µs     22.4µs     15.6µs
   96     38.9µs     40.7µs     34.9µs     70.2µs     36.6µs
  128     74.1µs     75.3µs     99.1µs    172.7µs    128.2µs
  192    202.6µs    234.3µs      188µs    537.6µs      418µs
  256    418.9µs    416.5µs    319.2µs      1.3ms    827.9µs
  384      1.1ms      905µs    880.7µs      4.5ms      1.9ms
  512      2.4ms      1.7ms      1.5ms     11.2ms      4.3ms
  640      4.1ms      2.5ms      2.3ms     20.9ms      5.6ms
  768      6.7ms      3.6ms      3.4ms     36.2ms      8.7ms
  896     10.2ms      5.2ms      4.8ms     57.1ms     11.3ms
 1024       15ms      7.3ms      6.8ms     89.3ms     17.2ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      5.8µs      5.8µs          -     14.8µs     12.2µs
   64     24.6µs     24.5µs          -    105.1µs     72.1µs
   96       67µs     67.2µs          -    347.5µs      207µs
  128    155.7µs    155.9µs          -    836.3µs    460.8µs
  192    456.6µs    457.2µs          -      2.7ms      1.4ms
  256      1.2ms      1.2ms          -      6.6ms      3.4ms
  384      3.8ms      3.8ms          -       22ms     10.9ms
  512     10.2ms      7.9ms          -     52.6ms     26.6ms
  640     17.7ms     11.9ms          -    101.4ms       50ms
  768     31.2ms     17.7ms          -    175.3ms     86.1ms
  896     47.7ms     25.1ms          -    280.5ms    135.4ms
 1024     75.4ms     38.8ms          -    431.7ms    204.4ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     11.9µs     11.9µs     15.1µs      7.9µs      7.1µs
   64     36.6µs     36.6µs     61.1µs     43.1µs     44.7µs
   96     75.2µs     75.2µs    321.3µs    139.2µs     79.3µs
  128    132.3µs    132.9µs    828.1µs      323µs    153.8µs
  192    335.3µs    335.3µs      1.6ms      1.1ms    388.7µs
  256    672.7µs    732.2µs      3.4ms      2.5ms    796.3µs
  384      1.9ms      1.7ms      8.1ms      8.1ms      2.1ms
  512      4.1ms      3.1ms     15.7ms     18.8ms      4.5ms
  640      7.4ms      4.6ms     22.8ms     36.1ms        8ms
  768     12.3ms      6.7ms     35.2ms     61.7ms     13.2ms
  896     18.7ms      9.4ms     46.6ms     98.1ms     20.5ms
 1024     27.7ms     13.1ms     68.3ms    150.3ms     30.5ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     17.1µs     29.5µs          -     17.9µs      9.6µs
   64     68.4µs     91.2µs          -      128µs     37.6µs
   96    163.6µs    187.4µs          -    422.2µs     98.3µs
  128    319.2µs    372.5µs          -    991.2µs    218.3µs
  192    841.8µs    894.6µs          -      3.3ms      633µs
  256      1.8ms      1.7ms          -      7.7ms      1.5ms
  384      5.7ms      3.9ms          -     25.6ms      5.9ms
  512     13.3ms      7.4ms          -     60.5ms     16.2ms
  640     24.8ms     11.3ms          -    117.3ms     28.9ms
  768       42ms       16ms          -    201.8ms       50ms
  896     66.4ms     21.5ms          -    323.8ms     78.2ms
 1024     98.1ms     38.9ms          -    499.7ms    123.2ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     18.1µs     34.2µs     10.7µs     21.1µs     10.7µs
   64     58.2µs     75.9µs     37.9µs     99.1µs       46µs
   96    142.4µs    134.6µs      205µs    285.3µs    118.8µs
  128    219.5µs    190.7µs    355.3µs    657.8µs    344.9µs
  192    592.3µs      458µs    658.1µs      2.2ms    966.3µs
  256      1.1ms    740.5µs      1.1ms      5.6ms        2ms
  384      3.1ms      1.6ms      2.4ms     18.8ms        5ms
  512      6.5ms      2.9ms      4.5ms     44.7ms     11.9ms
  640       12ms      4.9ms      7.3ms     84.9ms     19.3ms
  768     19.9ms      7.7ms     11.3ms    145.2ms     31.3ms
  896     30.3ms     12.2ms     16.7ms      228ms     44.1ms
 1024     44.6ms     19.2ms       24ms      356ms     68.8ms
```

## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32    117.7µs    141.8µs     90.3µs    109.8µs    222.7µs
   64    412.1µs    391.4µs    683.9µs    547.3µs      1.1ms
   96    923.2µs    920.9µs      1.7ms      1.7ms      2.6ms
  128      1.7ms      1.6ms      2.9ms      4.7ms      4.6ms
  192        4ms        4ms      6.7ms     14.9ms      9.9ms
  256      7.8ms        7ms     11.7ms     46.5ms     17.7ms
  384     20.8ms       15ms     25.9ms    123.7ms     43.7ms
  512     45.5ms     27.7ms     51.7ms    466.2ms     84.2ms
  640     80.3ms     44.2ms     79.3ms    662.8ms      135ms
  768      131ms     76.1ms    123.4ms      1.48s    209.6ms
  896    197.6ms    109.9ms    172.4ms      2.13s    293.2ms
 1024    296.3ms      153ms    254.7ms      4.02s      438ms
```

## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32      1.2ms      1.3ms      5.4ms      5.4ms      3.1ms
   64      3.4ms      3.3ms     15.6ms       20ms      8.3ms
   96      6.9ms      5.6ms     30.5ms     45.2ms     17.7ms
  128     11.4ms      8.5ms       48ms     80.4ms     32.4ms
  192     23.8ms     16.3ms     63.4ms    185.7ms     55.1ms
  256     41.2ms     25.7ms     83.6ms    357.3ms     91.1ms
  384     90.7ms     48.8ms      134ms    911.4ms    204.9ms
  512    164.8ms     80.3ms    303.4ms      2.02s    384.1ms
  640    259.1ms    119.7ms      291ms      3.23s    626.8ms
  768    381.5ms    186.1ms    440.1ms      5.17s    922.5ms
  896    531.1ms    253.3ms    550.9ms      7.27s      1.29s
 1024    721.6ms      328ms    850.8ms     10.63s      1.71s
```

## Hermitian matrix eigenvalue decomposition

Computing the EVD of a hermitian matrix with shape `(n, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32     65.3µs     56.7µs    121.5µs     49.9µs     51.1µs
   64    235.5µs      221µs    611.2µs    295.5µs    208.4µs
   96    560.8µs    531.9µs      2.6ms    882.9µs    521.2µs
  128    953.2µs    875.3µs      5.2ms      1.9ms      1.1ms
  192      2.3ms      2.1ms     15.4ms      5.8ms      3.1ms
  256      4.1ms      3.6ms     33.9ms     13.5ms      6.8ms
  384     10.5ms      8.9ms      106ms     43.2ms     21.9ms
  512     21.5ms     16.3ms    179.5ms    101.5ms     54.4ms
  640     37.1ms     26.3ms    272.1ms    192.9ms    100.8ms
  768     59.6ms     38.5ms    405.7ms    328.6ms      171ms
  896     89.2ms     52.6ms    597.7ms    512.2ms    265.7ms
 1024    130.4ms     71.9ms    901.7ms    777.8ms      406ms
```

## Non Hermitian matrix eigenvalue decomposition

Computing the EVD of a matrix with shape `(n, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
   32    207.4µs    208.6µs    173.4µs          -    224.7µs
   64        1ms      1.2ms    993.5µs          -      1.1ms
   96      2.7ms      3.1ms      5.7ms          -      3.2ms
  128      5.1ms      5.2ms     11.4ms          -      9.3ms
  192     13.2ms     16.5ms     22.7ms          -     27.2ms
  256     23.6ms       26ms     49.6ms          -     88.4ms
  384     57.2ms     62.9ms    103.7ms          -    241.6ms
  512    128.7ms      133ms    294.8ms          -      906ms
  640      215ms    201.5ms    418.8ms          -      1.18s
  768    327.1ms    294.8ms    565.5ms          -      2.89s
  896    448.8ms    381.8ms    693.6ms          -      3.63s
 1024    723.6ms    585.2ms    935.1ms          -      7.01s
```
