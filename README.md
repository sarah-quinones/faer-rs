# faer

`faer` is a collection of crates that implement linear algebra routines with a pure Rust backend and a focus on portability, correctness, and performance.

## faer-core

The core module implements matrix structures, as well as BLAS-like matrix operations such as matrix multiplication and solving triangular linear systems.

# Benchmarks

The benchmarks were run on an `11th Gen Intel(R) Core(TM) i5-11400 @ 2.60GHz`.

## Matrix multiplication

Multiplication of two square matrices of size `n`.
```
        faer (serial)      faer (parallel)   ndarray (openblas)      nalgebra (matrixmultiply)
   32           2.5µs                1.5µs                1.5µs                          2.8µs
   64          15.2µs                 10µs                8.1µs                         16.8µs
   96          44.2µs               18.2µs               26.2µs                         51.3µs
  128         102.7µs               19.8µs               36.7µs                        117.3µs
  192         334.8µs                 76µs               50.6µs                        381.1µs
  256         708.8µs              171.6µs              153.5µs                        746.3µs
  384           1.8ms                378µs              343.1µs                            2ms
  512           4.2ms              940.7µs                  1ms                          4.8ms
  640           8.7ms                2.3ms                1.9ms                          9.4ms
  768          15.2ms                3.7ms                2.9ms                         16.2ms
  896          24.4ms                6.2ms                5.3ms                           26ms
 1024          38.2ms                  9ms                7.4ms                         39.1ms
```
