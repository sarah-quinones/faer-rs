# faer

[![Documentation](https://docs.rs/faer/badge.svg)](https://docs.rs/faer)
[![Crate](https://img.shields.io/crates/v/faer.svg)](https://crates.io/crates/faer)

`faer` is a Rust crate that implements low level linear algebra routines and a high level wrapper for ease of use, in pure Rust.
The aim is to provide a fully featured library for linear algebra with focus on portability, correctness, and performance.

See the [official website](https://faer-rs.github.io) and the [docs.rs](https://docs.rs/faer/latest/faer) documentation for code examples and usage instructions.

Questions about using the library, contributing, and future directions can be discussed in the [Discord server](https://discord.gg/Ak5jDsAFVZ).

# Contributing

If you'd like to contribute to `faer`, check out the list of "good first issue"
issues. These are all (or should be) issues that are suitable for getting
started, and they generally include a detailed set of instructions for what to
do. Please ask questions on the Discord server or the issue itself if anything
is unclear!

# Minimum supported Rust version

The current MSRV is Rust 1.67.0.

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
    4       40ns       41ns      139ns       29ns       17ns
    8       77ns       80ns       63ns      161ns       85ns
   16      189ns      193ns      201ns      363ns      219ns
   32      1.1µs      1.1µs      1.1µs      1.5µs      1.2µs
   64      7.9µs      7.9µs      7.9µs     10.5µs      5.1µs
   96     27.5µs     11.2µs     26.2µs     34.9µs     10.1µs
  128     65.5µs     17.1µs     35.7µs     78.3µs     32.9µs
  192    216.6µs     54.4µs     57.3µs    260.7µs     51.7µs
  256    510.8µs    117.8µs    183.2µs    602.6µs    142.9µs
  384      1.7ms    339.1µs    575.8µs        2ms    327.9µs
  512        4ms    785.6µs      1.3ms      4.7ms        1ms
  640      7.9ms      1.6ms      2.3ms      9.2ms      1.9ms
  768     13.8ms      2.9ms      3.6ms       16ms      3.2ms
  896     22.2ms      4.6ms      6.5ms     25.7ms      5.9ms
 1024     33.9ms      6.6ms      9.7ms     39.1ms      8.3ms
```

## Triangular solve

Solving `AX = B` in place where `A` and `B` are two square matrices of dimension `n`, and `A` is a triangular matrix.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4       20ns       19ns      755ns       39ns       65ns
    8      118ns      118ns      1.5µs      308ns      156ns
   16      498ns      502ns      3.3µs      1.5µs      671ns
   32      2.1µs      2.1µs      8.6µs      6.6µs      2.9µs
   64      9.7µs      9.8µs     25.9µs     34.2µs     13.8µs
   96     27.7µs     24.5µs     55.2µs    101.4µs     36.9µs
  128     56.4µs     39.9µs    145.2µs      232µs     81.7µs
  192    167.8µs       92µs    263.6µs    815.5µs    213.6µs
  256    367.7µs      163µs      660µs      1.9ms    488.1µs
  384      1.1ms    317.5µs      1.4ms      7.4ms      1.4ms
  512      2.6ms    662.7µs      3.5ms     17.2ms      3.3ms
  640      4.7ms      1.2ms      5.7ms     33.6ms      5.5ms
  768        8ms      2.3ms      9.4ms     56.2ms      9.3ms
  896     12.3ms      3.6ms     13.6ms     89.3ms       14ms
 1024     18.7ms      5.2ms     20.1ms    131.9ms     22.9ms
```

## Triangular inverse

Computing `A^-1` where `A` is a square triangular matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4      162ns      5.2µs      771ns       38ns       65ns
    8      514ns      5.9µs      1.5µs      308ns      156ns
   16      1.6µs      7.7µs      3.4µs      1.5µs      672ns
   32      4.2µs     10.5µs      8.7µs      6.6µs      2.9µs
   64     12.5µs     18.1µs     25.7µs     34.2µs     13.8µs
   96     30.6µs     39.8µs       55µs    101.4µs     36.9µs
  128     42.7µs     51.9µs    144.9µs      232µs     81.6µs
  192      110µs     89.7µs    262.9µs    815.7µs    213.3µs
  256    191.7µs    138.3µs    645.5µs      1.9ms    486.9µs
  384    533.5µs    274.7µs      1.4ms      6.7ms      1.4ms
  512      1.1ms    449.4µs      3.5ms     15.6ms      3.3ms
  640        2ms    861.3µs      5.6ms     30.2ms      5.5ms
  768      3.2ms      1.2ms      9.3ms     51.8ms      9.3ms
  896      4.8ms      1.7ms     13.4ms     81.9ms       14ms
 1024      7.2ms      2.4ms     19.9ms    122.8ms     22.7ms
```

## Cholesky decomposition

Factorizing a square matrix with dimension `n` as `L×L.T`, where `L` is lower triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4       49ns       49ns      149ns       52ns       43ns
    8      128ns      128ns      329ns       99ns      125ns
   16      408ns      408ns      950ns      412ns      376ns
   32      1.8µs      1.8µs      3.3µs      1.8µs      2.3µs
   64        7µs        7µs     34.6µs     10.5µs        9µs
   96       18µs     18.2µs     70.5µs     31.3µs       21µs
  128     30.1µs     30.4µs    202.2µs     77.4µs     40.3µs
  192     86.4µs     92.7µs    301.3µs    259.8µs    105.2µs
  256    161.7µs    149.4µs    711.5µs    607.4µs    216.6µs
  384    462.9µs    423.9µs      1.2ms      2.1ms    596.5µs
  512      1.1ms    619.5µs      3.8ms      5.4ms      1.3ms
  640      1.9ms      1.3ms      3.3ms     10.4ms      2.2ms
  768      3.3ms      1.8ms      5.4ms     17.9ms      3.7ms
  896        5ms      2.7ms      6.9ms     28.4ms      5.6ms
 1024      7.8ms      3.4ms     14.5ms     41.2ms      8.4ms
```

## LU decomposition with partial pivoting

Factorizing a square matrix with dimension `n` as `P×L×U`, where `P` is a permutation matrix, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4      103ns       99ns      180ns       77ns       98ns
    8      210ns      217ns      405ns      241ns      278ns
   16      649ns      625ns      1.4µs      859ns      880ns
   32      2.7µs      2.6µs      5.6µs      4.4µs      3.9µs
   64     12.4µs     12.5µs     17.4µs     22.9µs     15.6µs
   96     30.2µs     31.6µs     34.4µs     67.9µs     36.7µs
  128     61.3µs     60.7µs     97.4µs    159.4µs      126µs
  192    163.5µs    187.3µs    182.4µs    527.8µs    425.5µs
  256      352µs    360.9µs    491.1µs      1.3ms    824.9µs
  384    968.8µs    781.3µs    909.5µs      4.5ms      1.9ms
  512      2.1ms      1.5ms      1.5ms     11.1ms      4.3ms
  640      3.8ms      2.2ms      2.2ms     20.7ms      5.6ms
  768      6.2ms      3.2ms      3.4ms     35.8ms      8.6ms
  896      9.5ms      4.6ms      4.7ms     56.1ms     11.4ms
 1024     14.4ms      6.5ms      6.7ms       88ms     17.1ms
```

## LU decomposition with full pivoting

Factorizing a square matrix with dimension `n` as `P×L×U×Q.T`, where `P` and `Q` are permutation matrices, `L` is unit lower triangular and `U` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4      132ns      134ns          -      111ns      164ns
    8      386ns      415ns          -      418ns      493ns
   16      1.7µs      1.7µs          -      2.3µs      2.1µs
   32      5.9µs        6µs          -     14.7µs     12.2µs
   64     25.8µs     25.4µs          -    106.4µs     72.2µs
   96     67.7µs     67.9µs          -    347.3µs    206.3µs
  128    156.4µs    155.2µs          -    819.1µs    460.9µs
  192    463.4µs    460.6µs          -      2.8ms      1.4ms
  256      1.1ms      1.1ms          -      6.6ms      3.3ms
  384      3.8ms      3.8ms          -     22.1ms       11ms
  512     10.1ms      7.9ms          -     53.4ms     27.4ms
  640     17.7ms       12ms          -    102.5ms     50.7ms
  768     31.2ms     17.5ms          -    176.9ms     87.3ms
  896     47.3ms     25.1ms          -      280ms    136.1ms
 1024     76.1ms     33.9ms          -      431ms    207.9ms
```

## QR decomposition with no pivoting

Factorizing a square matrix with dimension `n` as `QR`, where `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4      132ns      132ns      758ns      138ns      273ns
    8      345ns      346ns      1.7µs      321ns      777ns
   16      1.1µs      1.1µs      4.8µs      1.3µs      2.2µs
   32      4.4µs      4.4µs     15.3µs      6.9µs      7.4µs
   64     30.5µs     30.1µs     61.7µs     43.4µs     45.2µs
   96     65.2µs     65.2µs    322.4µs    141.3µs     79.1µs
  128    118.4µs    118.3µs    842.4µs    320.9µs    154.3µs
  192    315.3µs    316.1µs      1.6ms      1.1ms    383.7µs
  256    643.8µs    693.4µs      2.8ms      2.4ms    794.6µs
  384      1.9ms      1.7ms      7.6ms      8.1ms      2.1ms
  512      4.1ms        3ms     16.1ms       19ms      4.5ms
  640      7.4ms      4.5ms     22.5ms     36.2ms        8ms
  768     12.2ms      6.6ms     34.7ms     62.1ms     13.2ms
  896     18.6ms      9.2ms     46.3ms     97.7ms     20.4ms
 1024     27.7ms     12.9ms     65.9ms      150ms     30.2ms
```

## QR decomposition with column pivoting

Factorizing a square matrix with dimension `n` as `QRP`, where `P` is a permutation matrix, `Q` is unitary and `R` is upper triangular.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4      167ns      185ns          -      172ns      373ns
    8      430ns      433ns          -      552ns        1µs
   16      1.7µs      1.7µs          -      2.8µs      2.9µs
   32      5.9µs        6µs          -     17.6µs      9.5µs
   64     33.2µs     50.6µs          -    126.9µs     37.9µs
   96     85.6µs    104.7µs          -    421.8µs    104.7µs
  128    182.3µs    209.2µs          -    987.7µs    218.1µs
  192    548.2µs    600.4µs          -      3.3ms    628.1µs
  256      1.3ms      1.4ms          -      7.6ms      1.6ms
  384      4.6ms      3.5ms          -     25.4ms      5.6ms
  512     11.4ms      6.7ms          -       60ms     15.1ms
  640     22.2ms     10.5ms          -    116.2ms     26.6ms
  768     37.7ms     14.8ms          -    199.7ms     46.2ms
  896     60.7ms     20.1ms          -    317.9ms     71.1ms
 1024     90.2ms     30.7ms          -    488.3ms      114ms
```

## Matrix inverse

Computing the inverse of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4      795ns      7.5µs      534ns       77ns      381ns
    8      2.2µs      8.9µs      995ns      825ns      794ns
   16      5.3µs       12µs      2.9µs        4µs      2.7µs
   32     15.2µs     29.9µs     10.3µs       19µs     10.8µs
   64     49.8µs     66.2µs     40.5µs    101.2µs     45.9µs
   96    127.1µs    122.7µs    182.1µs    285.3µs    119.2µs
  128    199.9µs    172.7µs    314.9µs    661.3µs      341µs
  192      543µs    419.8µs    587.1µs      2.2ms    963.8µs
  256        1ms    668.3µs      1.1ms      5.6ms        2ms
  384      2.9ms      1.4ms      2.4ms     18.7ms      5.1ms
  512      6.2ms      2.6ms      4.6ms     44.2ms     11.9ms
  640     11.5ms      5.5ms      7.2ms       83ms     19.2ms
  768     19.2ms      8.7ms     11.2ms    142.3ms     30.9ms
  896     29.5ms     12.9ms     16.7ms    223.1ms     44.1ms
 1024     43.5ms     18.2ms     23.9ms    347.1ms     68.8ms
```

## Square matrix singular value decomposition

Computing the SVD of a square matrix with dimension `n`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4        2µs      1.9µs        3µs      1.3µs      1.8µs
    8      9.7µs     24.4µs      8.2µs      3.9µs      9.1µs
   16       32µs     57.8µs     25.9µs     16.9µs     49.8µs
   32      107µs    132.1µs     90.3µs     95.9µs      222µs
   64    409.1µs    381.5µs    562.5µs      555µs    987.6µs
   96    903.9µs    913.1µs      1.7ms      1.7ms      2.7ms
  128      1.6ms      1.5ms      2.9ms      4.6ms      4.3ms
  192        4ms        4ms      6.7ms     14.8ms      9.9ms
  256      7.8ms        7ms     11.7ms     47.4ms     17.3ms
  384     20.9ms     15.1ms     25.8ms    121.1ms     42.9ms
  512     45.3ms     28.1ms       52ms    472.1ms     83.9ms
  640       80ms     44.5ms     79.1ms    665.7ms    133.8ms
  768    130.9ms     78.5ms    123.9ms      1.48s    208.9ms
  896    198.4ms    110.9ms    182.8ms      2.11s    295.4ms
 1024    297.8ms      152ms    253.8ms      3.95s    433.6ms
```

## Thin matrix singular value decomposition

Computing the SVD of a rectangular matrix with shape `(4096, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4     73.4µs     73.5µs      311µs    127.5µs     76.7µs
    8    170.8µs    180.7µs    813.8µs    364.3µs    302.3µs
   16    440.4µs      513µs      2.1ms      1.4ms    775.5µs
   32      1.2ms      1.2ms      5.3ms      5.2ms      3.1ms
   64      3.4ms      3.2ms     15.7ms     19.9ms        8ms
   96      6.8ms      5.4ms     30.1ms     44.5ms     17.2ms
  128     11.2ms      8.3ms     47.4ms     79.4ms     30.9ms
  192     23.6ms     16.1ms       63ms    182.2ms     60.7ms
  256     40.7ms     25.5ms       84ms    353.1ms    101.3ms
  384     90.7ms     48.3ms      133ms    904.4ms    219.7ms
  512    164.7ms     80.2ms    303.4ms      2.02s    400.7ms
  640    258.7ms    119.7ms      289ms      3.24s    646.8ms
  768    381.7ms      187ms    440.1ms      5.15s      952ms
  896    532.6ms    252.7ms    550.2ms      7.23s      1.33s
 1024    724.4ms      327ms    849.6ms     10.64s      1.75s
```

## Hermitian matrix eigenvalue decomposition

Computing the EVD of a Hermitian matrix with shape `(n, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4      1.3µs      1.3µs      1.4µs      675ns        1µs
    8      3.9µs        4µs      6.6µs      2.3µs      3.4µs
   16     13.2µs     13.6µs     25.9µs     10.3µs     12.5µs
   32     50.9µs     51.1µs    167.1µs     50.8µs     49.7µs
   64    223.9µs    217.5µs      1.2ms    293.9µs    211.2µs
   96      519µs    518.2µs      2.6ms      876µs      518µs
  128    931.7µs    885.5µs      5.4ms      1.9ms      1.1ms
  192      2.2ms      2.1ms       16ms      5.8ms      3.1ms
  256      4.1ms      3.5ms     33.9ms     13.2ms      6.6ms
  384     10.5ms      8.8ms    105.5ms     42.7ms     21.2ms
  512     21.9ms     16.5ms      175ms     99.3ms     51.4ms
  640     37.6ms     26.5ms    266.2ms    187.4ms     94.2ms
  768     60.4ms     38.1ms    403.3ms    322.6ms    161.9ms
  896     90.4ms     52.2ms    615.3ms    502.5ms    249.9ms
 1024    132.1ms     68.4ms      909ms    764.1ms      392ms
```

## Non Hermitian matrix eigenvalue decomposition

Computing the EVD of a matrix with shape `(n, n)`.

```
    n       faer  faer(par)    ndarray   nalgebra      eigen
    4      4.8µs      5.1µs      3.5µs          -      3.1µs
    8     15.6µs     16.7µs      9.6µs          -     10.5µs
   16     54.7µs     54.4µs     35.9µs          -     44.4µs
   32    270.7µs    235.6µs    172.6µs          -    199.3µs
   64      1.1ms      1.1ms        1ms          -      1.1ms
   96      2.7ms      2.9ms      5.5ms          -      3.1ms
  128      4.9ms      5.6ms     11.6ms          -      9.2ms
  192     14.4ms     14.3ms     22.4ms          -     26.9ms
  256     24.4ms     26.2ms     49.9ms          -     86.6ms
  384     56.4ms     62.6ms      107ms          -    246.1ms
  512    126.8ms    130.1ms    281.7ms          -    887.6ms
  640    205.8ms    192.6ms    415.6ms          -       1.2s
  768    323.5ms    285.6ms    547.2ms          -      2.84s
  896    438.1ms    375.8ms    704.3ms          -      3.67s
 1024    687.8ms    579.3ms    957.1ms          -         7s
```
