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
   32      1.4µs      1.1µs      1.1µs      2.5µs       827µs
   64      9.1µs      8.1µs      7.9µs     14.3µs       177µs
   96     28.2µs     11.2µs     26.4µs     36.2µs       219µs
  128     66.2µs     17.7µs     53.1µs     81.8µs      69.2µs
  192    220.5µs     70.3µs     53.7µs    266.4µs       122µs
  256      520µs    132.8µs    172.7µs    614.9µs       257µs
  384      1.8ms    460.2µs      374µs      2.1ms       435µs
  512      4.2ms        1ms      1.2ms        5ms      1.32ms
  640      8.4ms      2.2ms      2.2ms      9.9ms      2.06ms
  768     14.8ms        4ms      3.4ms     17.1ms      2.68ms
  896       24ms      6.7ms      5.7ms       28ms      5.24ms
 1024     36.3ms     10.5ms      8.5ms     43.3ms      7.16ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      2.4µs      2.4µs      7.8µs      8.9µs      9.12µs
   64     10.4µs     10.5µs     22.3µs     38.2µs      27.4µs
   96     29.7µs     26.4µs     45.1µs    104.2µs      79.2µs
  128     59.4µs     40.3µs    125.1µs    250.5µs       141µs
  192    175.8µs     90.2µs    222.1µs    882.1µs       306µs
  256    383.8µs      167µs    544.6µs        2ms       637µs
  384      1.2ms    351.3µs      1.2ms      7.3ms      1.92ms
  512      2.7ms    718.4µs      3.2ms     17.5ms      4.36ms
  640        5ms      1.5ms      5.1ms     32.9ms      7.29ms
  768      8.4ms      2.1ms      8.9ms     57.5ms      12.5ms
  896       13ms      4.3ms     11.8ms       90ms      18.6ms
 1024     20.4ms      5.4ms     22.7ms    140.1ms      28.4ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      3.2µs       14µs      8.2µs      9.1µs      7.88µs
   64     10.4µs     25.9µs     21.4µs     37.9µs        27µs
   96       25µs     39.6µs     46.6µs    103.7µs      60.9µs
  128     38.9µs     62.2µs    127.3µs    248.9µs       120µs
  192     98.9µs     94.2µs    220.1µs    846.9µs       372µs
  256    184.2µs    137.5µs    675.8µs        2ms       695µs
  384      519µs    261.6µs      1.2ms      7.3ms      1.93ms
  512      1.1ms    453.3µs      3.2ms     17.5ms      4.35ms
  640        2ms    700.6µs      5.1ms       34ms      7.86ms
  768      3.3ms        1ms      8.7ms     57.9ms      12.6ms
  896      5.2ms      1.7ms     12.5ms       90ms      18.5ms
 1024      7.5ms      2.3ms     22.4ms    144.3ms      28.4ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      3.5µs      3.5µs      3.3µs      3.5µs      5.12µs
   64     10.2µs     10.3µs     40.8µs     15.4µs      13.9µs
   96     25.2µs     25.4µs     83.3µs     32.7µs      29.9µs
  128     36.3µs     37.5µs      147µs     81.5µs        50µs
  192    104.4µs    111.3µs    321.5µs    262.8µs       121µs
  256    178.3µs    167.7µs    591.5µs    625.3µs       235µs
  384    495.8µs      465µs      1.1ms      2.2ms       606µs
  512      1.2ms      706µs      3.3ms      5.7ms      1.36ms
  640        2ms      1.5ms      2.9ms     10.9ms      2.84ms
  768      3.8ms        2ms      4.8ms     18.4ms      4.19ms
  896      5.5ms        3ms      6.4ms     28.8ms      6.13ms
 1024      8.5ms      3.7ms     12.7ms     43.6ms      8.37ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32      4.6µs      4.6µs      5.7µs      6.4µs       8.2µs
   64     17.5µs     17.9µs     17.8µs     28.1µs      46.5µs
   96     39.7µs     42.9µs     35.4µs     69.2µs      43.5µs
  128     77.9µs     77.9µs     98.3µs    162.9µs       130µs
  192    200.2µs    220.1µs    205.3µs    516.4µs       317µs
  256    405.2µs      403µs    346.8µs      1.4ms       670µs
  384      1.1ms    980.7µs    962.2µs      4.7ms       1.5ms
  512      2.4ms      1.8ms      1.5ms     11.6ms      3.34ms
  640      4.2ms      3.2ms      1.9ms     21.9ms      4.84ms
  768        7ms      4.7ms      4.4ms     37.6ms      7.44ms
  896     10.7ms      6.9ms      4.2ms     59.6ms        11ms
 1024     16.2ms      9.3ms      6.1ms     95.6ms      17.6ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32     10.6µs      214µs          -     15.2µs      17.2µs
   64     44.2µs    421.3µs          -    107.3µs      89.7µs
   96    110.4µs    666.3µs          -    355.2µs       248µs
  128    229.6µs    972.8µs          -      843µs       531µs
  192    593.6µs      1.7ms          -      2.8ms      1.56ms
  256      1.4ms      2.6ms          -      6.7ms      3.65ms
  384      4.5ms      5.3ms          -     22.6ms      11.7ms
  512     11.8ms      8.6ms          -     54.5ms      28.6ms
  640     19.7ms     13.7ms          -    105.1ms      51.7ms
  768     33.6ms     19.3ms          -    182.8ms      77.9ms
  896     51.4ms     27.8ms          -    290.4ms       123ms
 1024     84.2ms     44.7ms          -      455ms       192ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32     13.7µs       14µs     16.5µs     11.4µs      8.94µs
   64     38.7µs     39.6µs     66.9µs     52.6µs      51.7µs
   96       76µs     77.6µs    344.3µs    146.8µs      88.2µs
  128    131.6µs    158.8µs    917.6µs    341.5µs       169µs
  192    310.9µs    329.6µs      1.8ms      1.1ms       402µs
  256    653.9µs    565.9µs      3.4ms      2.5ms       863µs
  384      1.9ms      1.2ms      6.8ms      8.5ms      2.17ms
  512      4.6ms      2.3ms     14.5ms     20.3ms      4.71ms
  640      7.8ms      3.5ms     20.4ms       39ms      8.46ms
  768     13.1ms      5.3ms     37.3ms     66.5ms      14.3ms
  896       20ms      7.4ms     51.7ms    107.1ms      21.9ms
 1024     31.5ms     10.4ms     71.4ms    165.6ms      32.6ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32       10µs    264.3µs          -     21.9µs      18.5µs
   64     44.8µs    429.9µs          -    131.8µs        41µs
   96    111.6µs    685.7µs          -    433.5µs       107µs
  128    229.6µs    942.1µs          -        1ms       228µs
  192    632.6µs      1.5ms          -      3.4ms       638µs
  256      1.4ms      2.4ms          -      7.9ms       1.7ms
  384        5ms      4.4ms          -     26.1ms      6.34ms
  512     12.2ms      7.4ms          -       62ms      16.7ms
  640       24ms     12.6ms          -    121.5ms      30.2ms
  768     40.9ms     17.1ms          -    207.5ms      50.2ms
  896     65.1ms     23.9ms          -    335.6ms      81.2ms
 1024      102ms       37ms          -    512.4ms       129ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra       eigen
   32     13.9µs     30.6µs     10.6µs     35.5µs      31.4µs
   64     49.6µs     71.8µs     38.5µs    138.9µs      68.5µs
   96    125.4µs    136.8µs    185.5µs    293.6µs       159µs
  128    233.3µs    213.5µs      307µs    672.5µs       489µs
  192    599.5µs    431.9µs    565.9µs      2.2ms      1.19ms
  256      1.2ms    886.9µs      994µs      5.7ms      2.37ms
  384      3.2ms      1.8ms      2.6ms     19.8ms      7.32ms
  512      7.4ms        4ms      5.3ms     46.5ms      14.6ms
  640     13.4ms      6.9ms      8.7ms     88.6ms      24.8ms
  768     22.5ms     11.3ms       17ms    151.5ms      35.8ms
  896     33.8ms       17ms     19.8ms    239.4ms      62.5ms
 1024     50.1ms     23.7ms     29.6ms    378.4ms      83.3ms
```
