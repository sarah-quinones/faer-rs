# 0.8.0
- Refactor the core traits for better SIMD support for non native types, using a structure-of-arrays layout.
- Implement the hermitian eigenvalue decomposition.

# 0.7.0
- Add `to_owned` function for converting a `MatRef/MatMut` to a `Mat`. Thanks @Tastaturtaste
- Allow comparison of conjugated matrices
- Performance improvements for `f32`, `c32`, and `c64`
- Performance improvements for small/medium matrix decompositions
- Refactor the `ComplexField` trait to allow for non `Copy` types. Note that types other than `f32`, `f64`, `c32`, `c64` are not yet supported.

# 0.6.0
- Start keeping track of changes.
- Complex SVD support.
- Improve performance for thin SVD.
- Fixed an edge case in complex Householder computation where the input vector is all zeros.
