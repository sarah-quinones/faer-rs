# 0.22
- accelerated matrix multiply backend on `x86_64` targets.
- accelerated column pivoted qr factorization
- accelerated matrix multiply for non primitive (and `Complex<Primitive>`) types.
- implemented an extended precision simd floating point type (exported as `fx128`, complex number as `cxf128`).
- make dense unpivoted qr rank revealing
- removed lblt regularization
- implemented `FromIterator` for `Col` and `Row`.
- stabilized matrix-free solvers.
- implemented matrix-free krylov-schur eigensolver.
- renamed bunch-kaufman to lblt.
- implemented pivoting strategies for the LBLT factorization
- implemented pivoted LLT/LDLT

# 0.20 - 0.21
- Project refactor

# 0.19
- Support matrix-scalar multiplication/division without the `Scale` wrapper for `f32`/`f64`.
- Implemented conjugate gradient, BiCGSTAB, and LSMR iterative solvers (currently gated by the `unstable` feature).
- Implemented Hermitian matrix pseudoinverse implementation. Thanks @lishen_ for the contribution.
- Implemented column and row mean and variance in `faer::stats`.
- Added more iterator and parallel iterator functions (`MatRef::[col|row]_partition`, `MatRef::par_[col|row]_partition`, etc.).
- Added `full` and `zeros` constructors to owned Col, Row, and Matrix ([issue-125](https://github.com/sarah-ek/faer-rs/issues/125)).
- Added `shape` function to return both the row and the column count of a matrix.
- Added several missing associated functions from the mut and owning variants of matrices.
- Implemented `core::iter::{Sum, Product}` for `c32` and `c64`.
- Sparse Cholesky can now be used with user-provided permutations.
- Simplified matrix constructors, adding a variant with a `_generic` prefix for the old behavior.
- LDLT and Bunch-Kaufman decompositions now stores the diagonal blocks instead of their inverses. This helps avoid infinities and NaNs when dealing with singular matrices.
- Integrated `nano-gemm` as a backend for small matrix multiplication.
- Significant performance improvements for small LLT and LDLT decompositions.

# 0.18
- Refactored the project so that `faer` contains all the core and decomposition implementations. `faer-{core,cholesky,lu,qr,svd,evd,sparse}` are now deprecated and will no longer be updated.
- Improved the multithreaded performance of the Eigenvalue decomposition for large matrices.
- Decomposition solve functions now accept column vectors as well as matrices.
- Implemented the L1 norm, and the squared L2 norm.
- Implemented conversions from sparse to dense matrices, by calling `mat.to_dense()`.
- Sparse matrices now support duplicated entries. Note that `faer` will not add duplicated entries to a matrix unless the opposite is explicitly mentioned in the function documentation. `faer` also will deduplicate entries when created with `Sparse{Col,Row}Mat::try_new_from_indices` and other similar functions.
- Implemented conversions from unsorted to sorted sparse matrices by calling `mat.to_sorted()` (or `mat.sort_indices()` for owned matrices).
- Implemented `{Col,Row}::try_as_slice[_mut]` functions that return data as a slice if it is contiguous.
- Implemented `.for_each_with_index` and `.map_with_index` for the matrix zipping API, which passes the matrix row and column indices as well as the values.
- Added `rand` support for randomly generating matrices in the `faer::stats` module, as well as for `faer::complex_native::{c32,c64}`.
- Implemented a pseudoinverse helper for the high level SVD and thin SVD decompositions.

# 0.17
- Implemented sparse matrix arithmetic operators (other than sparse-sparse matrix multiplication), and added mutable sparse views as well as owning sparse matrix containers.
- Implemented `try_from_triplets` for sparse matrices.
- Re-exported subcrates in `faer::modules`.
- Improved performance of the SVD decomposition for small matrices.
- Implemented `col!`, `row!` and `concat!` macros. Thanks @DeliciousHair for the contribution.
- Implemented more `c32/c64` operations. Thanks @edyounis for the contribution.
- Implemented the Kronecker product in `faer_core`. Thanks @edyounis for the contribution.
- Implemented (de)serialization of `Mat`. Thanks @cramt for the contribution.

# 0.16
- Implemented the index operator for row and column structures. Thanks @DeliciousHair for the contribution.
- Exposed a few sparse matrix operations in the high level API.
- Implemented sparse LU and QR, and exposed sparse decompositions in the high level API.
- Better assertion error messages in no_std mode.

# 0.15
- Implemented initial API of `Row`/`RowRef`/`RowMut` and `Col`/`ColRef`/`ColMut` structs for handling matrices with a single row or column.
- Implemented `[Mat|Col|Row]::norm_l2` and `[Mat|Col|Row]::norm_max` for computing the L2 norm of a matrix or its maximum absolute value.
- Fixed several bugs in the eigenvalue decompositions. Special thanks to @AlexMath for tracking down the errors.
- Updated `zipped!` macro API, which now requires a matching `unzipped!` for matching the closure arguments.
- Removed the limitation on the number of matrices that can be passed to `zipped!`.
- Added a `zipped!(...).map(|unzipped!(...)| { ... })` API to allow mapping a zipped pack of matrices and returns the result as a matrix.
- Updated `polars` dependency to 0.34.
- Speed improvements for complex matrix multiplication on AMD cpus.
- New SIMD functions in the Entity trait for aligned loads and stores.
- Renamed multiple methods such as `MatMut::transpose` to `MatMut::transpose_mut`.

# 0.14
- Implemented sparse data structures in `faer_core::sparse`.
- Implemented sparse Cholesky decompositions, simplicial and supernodal. Only the low level API is currently exposed in `faer-sparse`.
- Implemented dynamic regularization for the Bunch-Kaufman Cholesky decomposition.
- Implemented diagonal wrappers that can be used to interpret a matrix as a diagonal matrix, using `{MatRef,MatMut}::diagonal` and `{MatRef,MatMut}::column_vector_as_diagonal`.
- Implemented matrix multiplication syntax sugar for diagonal wrappers, and permutation matrices.
- Implemented `compute_thin_r` and `compute_thin_q` in `faer::solvers::{Qr,ColPivQr}`.
- Implemented initial SIMD support for aarch64.

# 0.13
- Implemented the Bunch-Kaufman Cholesky decomposition for hermitian indefinite matrices.
- Implemented dynamic regularization for the diagonal LDLT.
- Support conversions involving complex values using `IntoFaerComplex`, `IntoNalgebraComplex` and `IntoNdarrayComplex`.
- Refactored the Entity trait for better ergonomics.
- `faer` scalar traits are now prefixed with `faer_` to avoid conflicts with standard library and popular library traits.
- `no_std` and `no_rayon` are now supported, with the optional features `std` and `rayon` (enabled by default).
- Performance improvements in the eigenvalue decomposition and thin matrix multiplication.

# 0.12
- Implemented matrix chunked iterators and parallel chunked iterators.
- Renamed `{Mat,MatMut}::fill_with_zero` to `fill_zeros`
- Renamed `{Mat,MatMut}::fill_with_constant` to `fill`
- More ergonomic `polars` api.
- Refactored Entity and ComplexField SIMD api.
- Switched from DynStack/GlobalMemBuffer to PodStack/GlobalPodBuffer.
- Fixed usize overflow bug in eigenvalue decomposition.

# 0.11
- High level api implemented in `faer`.
- Renamed `Mat::with_dims` to `Mat::from_fn`.
- Renamed `{Mat,MatMut}::set_zeros` to `fill_with_zero`.
- Renamed `{Mat,MatMut}::set_constant` to `fill_with_constant`.

# 0.10
- Performance improvements for small matrices.
- Simpler SVD/EVD API for fixed precision floating point types.
- Simpler math operators (+, -, *). Thanks @geo-ant and @DJDuque.
- More robust pivoted decompositions for rank deficient matrices.
- Better overflow/underflow handling in matrix decompositions, as well as non finite inputs.
- Provide control over global parallelism settings in `faer-core`.
- Various bug fixes for complex number handling.

# 0.9
- Implement the non Hermitian eigenvalue decomposition.
- Improve performance of matrix multiplication.
- Improve performance of LU decomposition with partial pivoting.

# 0.8
- Refactor the core traits for better SIMD support for non native types, using a structure-of-arrays layout.
- Implement the Hermitian eigenvalue decomposition.

# 0.7
- Add `to_owned` function for converting a `MatRef/MatMut` to a `Mat`. Thanks @Tastaturtaste
- Allow comparison of conjugated matrices
- Performance improvements for `f32`, `c32`, and `c64`
- Performance improvements for small/medium matrix decompositions
- Refactor the `ComplexField` trait to allow for non `Copy` types. Note that types other than `f32`, `f64`, `c32`, `c64` are not yet supported.

# 0.6
- Start keeping track of changes.
- Complex SVD support.
- Improve performance for thin SVD.
- Fixed an edge case in complex Householder computation where the input vector is all zeros.

