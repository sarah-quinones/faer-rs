//! Native complex floating point types whose real and imaginary parts are stored contiguously.
//!
//! The types [`c32`] and [`c64`] respectively have the same layout as [`num_complex::Complex32`]
//! and [`num_complex::Complex64`].
//!
//! They differ in the way they are treated by the `faer` library: When stored in a matrix,
//! `Mat<c32>` and `Mat<c64>` internally contain a single container of contiguously stored
//! `c32` and `c64` values, whereas `Mat<num_complex::Complex32>` and
//! `Mat<num_complex::Complex64>` internally contain two containers, separately storing the real
//! and imaginary parts of the complex values.
//!
//! Matrix operations using `c32` and `c64` are usually more efficient and should be preferred in
//! most cases. `num_complex::Complex` matrices have better support for generic data types.
//!
//! The drawing below represents a simplified layout of the `Mat` structure for each of `c32` and
//! `num_complex::Complex32`.
//!
//! ```notcode
//! ┌──────────────────┐
//! │ Mat<c32>         │
//! ├──────────────────┤
//! │ ptr: *mut c32 ─ ─│─ ─ ─ ─ ┐
//! │ nrows: usize     │   ┌─────────┐
//! │ ncols: usize     │   │ z0: c32 │
//! │        ...       │   │ z1: c32 │
//! └──────────────────┘   │ z2: c32 │
//!                        │   ...   │
//!                        └─────────┘
//!
//! ┌───────────────────────┐
//! │ Mat<Complex32>        │
//! ├───────────────────────┤
//! │ ptr_real: *mut f32 ─ ─│─ ─ ─ ─ ┐
//! │ ptr_imag: *mut f32 ─ ─│─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ┐
//! │ nrows: usize          │   ┌──────────┐   ┌──────────┐
//! │ ncols: usize          │   │ re0: f32 │   │ im0: f32 │
//! │           ...         │   │ re1: f32 │   │ im1: f32 │
//! └───────────────────────┘   │ re2: f32 │   │ im2: f32 │
//!                             │    ...   │   │    ...   │
//!                             └──────────┘   └──────────┘
//! ```

mod c32;
mod c64;
mod c32conj;
mod c64conj;

pub use c32::c32;
pub use c64::c64;
pub use c32conj::c32conj;
pub use c64conj::c64conj;
