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
