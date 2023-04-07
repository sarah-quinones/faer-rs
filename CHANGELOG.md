# 0.7.0
- Add `to_owned` function for converting a `MatRef/MatMut` to a `Mat`. Thanks @Tastaturtaste
- Allow comparison of conjugated matrices
- Performance improvements for `f32`, `c32`, and `c64`
- Performance improvements for small/medium matrix decompositions
- Refactored the `ComplexField` trait to allow for non `Copy` types. Note that types other than `f32`, `f64`, `c32`, `c64` are not yet supported.

# 0.6.0
- Started keeping track of changes.
- Complex SVD support.
- Improved performance for thin SVD.
- Fixed an edge case in complex Householder computation where the input vector is all zeros.
