#set text(font: "New Computer Modern")

#show raw: set text(font: "New Computer Modern Mono", size: 1.2em)

#show par: set block(spacing: 0.55em)

#show heading: set block(above: 1.4em, below: 1em)

#show link: underline

#set page(numbering: "1")

#set par(leading: 0.55em, justify: true)

#set heading(numbering: "1.1")

#show heading.where(level: 1): it => pagebreak(weak:true) + block({
    set text(font: "New Computer Modern", weight: "black")
    v(2cm)
    block(text(18pt)[Chapter #counter(heading).display()])
    v(1cm)
    block(text(22pt)[#it.body])
    v(1cm)
})

#import "@preview/codly:0.1.0"
#import "@preview/tablex:0.0.6": tablex, rowspanx, colspanx, gridx, hlinex, vlinex
#import "@preview/colorful-boxes:1.2.0": colorbox

#let icon(codepoint) = {
  box(
    height: 0.8em,
    baseline: 0.05em,
    image(codepoint)
  )
  h(0.1em)
}

#show: codly.codly-init.with()

#codly.codly(
  languages: (
    rust: (name: "Rust", icon: icon("brand-rust.svg"), color: rgb("#CE412B")),
  ),
  breakable: false,
  width-numbers: none,
)

#outline()

== Introduction
_`faer-rs`_ is a general-purpose linear algebra library for the Rust
programming language, with a focus on correctness, portability, and
performance.
In this book, we'll be assuming version `0.16.0` of the library.

A matrix is a 2-dimensional array of numerical values, which can represent
different things depending on the context. In the context of linear algebra,
it is often used to represent a linear transformation mapping vectors from
one finite-dimensional space to another.

Column vectors are typically elements of the vector space, but may also be used
interchangeably with $n×1$ matrices. Row vectors are also similarly used
interchangeably with $1×n$ matrices.

= Dense linear algebra
== Creating a matrix
_`faer`_ provides several ways to create matrices and matrix views.

The main matrix types are `faer_core::Mat`, `faer_core::MatRef` and `faer_core::MatMut`,
which can be thought of as being analogous to `Vec`, `&[_]` and `&mut [_]`.

The most flexible way to initialize a matrix is to initialize a zero matrix,
then fill out the values by hand.

```rust
let mut a = Mat::<f64>::zeros(4, 3);

for j in 0..a.ncols() {
    for i in 0..a.nrows() {
        a[(i, j)] = 9.0;
    }
}
```

Given a callable object that outputs the matrix elements, `Mat::from_fn`, can also be used.

```rust
let a = Mat::from_fn(3, 4, |i, j| (i + j) as f64);
```

For common matrices such as the zero matrix and the identity matrix, shorthands
are provided.

```rust
use faer_core::Mat;

// creates a 10×4 matrix whose values are all `0.0`.
let a = Mat::<f64>::zeros(10, 4);

// creates a 5×4 matrix containing `0.0` except on the main diagonal,
// which contains `1.0` instead.
let a = Mat::<f64>::identity(5, 4);
```
#colorbox(
  title: "Note",
  color: "green",
  radius: 2pt,
  width: auto
)[
  In some cases, users may wish to avoid the cost of initializing the matrix to zero,
  in which case, unsafe code may be used to allocate an uninitialized matrix, which
  can then be filled out before it's used.
  ```rust
  // `a` is initially a 0×0 matrix.
  let mut a = Mat::<f64>::with_capacity(4, 3);

  // `a` is now a 4×3 matrix, whose values are uninitialized.
  unsafe { a.set_dims(4, 3) };

  for j in 0..a.ncols() {
      for i in 0..a.nrows() {
          // we cannot write `a[(i, j)] = 9.0`, as that would
          // create a reference to uninitialized data,
          // which is currently disallowed by Rust.
          a.write(i, j, 9.0);
      }
  }
  ```
]

== Creating a matrix view
In some situations, it may be desirable to create a matrix view over existing
data.
In that case, we can use `faer_core::MatRef` (or `faer_core::MatMut` for
mutable views).

They can be created in a safe way using:

`faer_core::mat::{from_column_major_slice, from_row_major_slice}`,

`faer_core::mat::{from_column_major_slice_mut, from_row_major_slice_mut}`,

for contiguous matrix storage, or:

`{from_column_major_slice_with_stride, from_row_major_slice_with_stride}`,

`{from_column_major_slice_with_stride_mut, from_row_major_slice_with_stride_mut}`,

for strided matrix storage.

#colorbox(
  title: "Note",
  color: "green",
  radius: 2pt,
  width: auto
)[
  A lower level pointer API is also provided for handling uninitialized data
  or arbitrary strides `{from_raw_parts, from_raw_parts_mut}`.
]

== Converting to a view

A `Mat` instance `m` can be converted to `MatRef` or `MatMut` by writing `m.as_ref()`
or `m.as_mut()`.

== Reborrowing a mutable view

Immutable matrix views can be freely copied around, since they are non-owning
wrappers around a pointer and the matrix dimensions/strides.

Mutable matrices however are limited by Rust's borrow checker. Copying them
would be unsound since only a single active mutable view is allowed at a time.

This means the following code does not compile.

```rust
use faer::{Mat, MatMut};

fn takes_view_mut(m: MatMut<f64>) {}

let mut a = Mat::<f64>::new();
let view = a.as_mut();

takes_view_mut(view);

// This would have failed to compile since `MatMut` is never `Copy`
// takes_view_mut(view);
```

The alternative is to temporarily give up ownership over the data, by creating
a view with a shorter lifetime, then recovering the ownership when the view is
no longer being used.

This is also called reborrowing.

```rust
use faer::{Mat, MatMut, MatRef};
use reborrow::*;

fn takes_view(m: MatRef<f64>) {}
fn takes_view_mut(m: MatMut<f64>) {}

let mut a = Mat::<f64>::new();
let mut view = a.as_mut();

takes_view_mut(view.rb_mut());
takes_view_mut(view.rb_mut());
takes_view(view.rb()); // We can also reborrow immutably

{
    let short_view = view.rb_mut();

    // This would have failed to compile since we can't use the original view
    // while the reborrowed view is still being actively used
    // takes_view_mut(view);

    takes_view_mut(short_view);
}

// We can once again use the original view
takes_view_mut(view.rb_mut());

// Or consume it to convert it to an immutable view
takes_view(view.into_const());
```

== Splitting a matrix view, slicing a submatrix
A matrix view can be split up along its row axis, column axis or both.
This is done using `MatRef::split_at_row`, `MatRef::split_at_col` or
`MatRef::split_at` (or `MatMut::split_at_row_mut`, `MatMut::split_at_col_mut` or
`MatMut::split_at_mut`).

These functions take the middle index at which the split is performed, and return
the two sides (in top/bottom or left/right order) or the four corners (top
left, top right, bottom left, bottom right)

We can also take a submatrix using `MatRef::subrows`, `MatRef::subcols` or
`MatRef::submatrix` (or `MatMut::subrows_mut`, `MatMut::subcols_mut` or
`MatMut::submatrix_mut`).

Alternatively, we can also use `MatRef::get` or `MatMut::get_mut`, which take
as parameters the row and column ranges.

#colorbox(
  title: "Warning",
  color: "red",
  radius: 2pt,
  width: auto
)[
  Note that `MatRef::submatrix` (and `MatRef::subrows`, `MatRef::subcols`) takes
  as a parameter, the first row and column of the submatrix, then the number
  of rows and columns of the submatrix.

  On the other hand, `MatRef::get` takes a range from the first row and column
  to the last row and column.
]

== Matrix arithmetic operations
_`faer`_ matrices implement most of the arithmetic operators, so two matrices
can be added simply by writing `&a + &b`, the result of the expression is a
`faer::Mat`, which allows chaining operations (e.g. `(&a + &b) * &c`), although
at the cost of allocating temporary matrices.

#colorbox(
  title: "Note",
  color: "green",
  radius: 2pt,
  width: auto
)[
  Temporary allocations can be avoided by using the zip api:
```rust
use faer::{Mat, zipped, unzipped};

let a = Mat::<f64>::zeros(4, 3);
let b = Mat::<f64>::zeros(4, 3);
let mut c = Mat::<f64>::zeros(4, 3);

// Sums `a` and `b` and stores the result in `c`.
zipped!(&mut c, &a, &b).for_each(|unzipped!(c, a, b)| *c = *a + *b);

// Sums `a`, `b` and `c` into a new matrix `d`.
let d = zipped!(&mut c, &a, &b).map(|unzipped!(c, a, b)| *a + *b + *c);
```
  For matrix multiplication, the non-allocating api is provided in the
  `faer_core::mul` module.

```rust
use faer::{Mat, Parallelism};
use faer_core::mul::matmul;

let a = Mat::<f64>::zeros(4, 3);
let b = Mat::<f64>::zeros(3, 5);

let mut c = Mat::<f64>::zeros(4, 5);

// Computes `faer::scale(3.0) * &a * &b` and stores the result in `c`.
matmul(c.as_mut(), a.as_ref(), b.as_ref(), None, 3.0, Parallelism::None);

// Computes `faer::scale(3.0) * &a * &b + 5.0 * &c` and stores the result in `c`.
matmul(c.as_mut(), a.as_ref(), b.as_ref(), Some(5.0), 3.0, Parallelism::None);
```
]

== Solving a linear system
Several applications require solving a linear system of the form $A x = b$.
The recommended method can vary depending on the properties of $A$, and the
desired numerical accuracy.

=== $A$ is triangular
In this case, one can use $A$ and $b$ directly to find $x$, using the functions
provided in `faer_core::solve`.

```rust
use faer::{Mat, Parallelism};
use faer_core::solve::solve_lower_triangular_in_place;

let a = Mat::<f64>::from_fn(4, 4, |i, j| if i >= j { 1.0 } else { 0.0 });
let b = Mat::<f64>::from_fn(4, 2, |i, j| (i - j) as f64);

let mut x = Mat::<f64>::zeros(4, 2);
x.copy_from(&b);
solve_lower_triangular_in_place(a.as_ref(), x.as_mut(), Parallelism::None);

// x now contains the approximate solution
```

In the case where $A$ has a unit diagonal, one can use
`solve_unit_lower_triangular_in_place`, which avoids reading the diagonal, and
instead implicitly uses the value `1.0` as a replacement.

=== $A$ is real-symmetric/complex-Hermitian
If $A$ is Hermitian and positive definite, users can use the Cholesky LLT
decomposition.

```rust
use faer::{mat, Side};
use faer::prelude::*;

let a = mat![
    [10.0, 2.0],
    [2.0, 10.0f64],
];
let b = mat![[15.0], [-3.0f64]];

// Compute the Cholesky decomposition,
// reading only the lower triangular half of the matrix.
let llt = a.cholesky(Side::Lower).unwrap();

let x = llt.solve(&b);
```

Alternatively, a lower-level API could be used to avoid temporary allocations.
The corresponding code for other decompositions follows the same pattern, so we
will avoid repeating it.

```rust
use faer::{mat, Parallelism, Conj};
use faer_cholesky::llt::compute::cholesky_in_place_req;
use faer_cholesky::llt::compute::{cholesky_in_place, LltRegularization, LltParams};
use faer_cholesky::llt::solve::solve_in_place_req;
use faer_cholesky::llt::solve::solve_in_place_with_conj;
use dyn_stack::{PodStack, GlobalPodBuffer};

let a = mat![
    [10.0, 2.0],
    [2.0, 10.0f64],
];
let mut b = mat![[15.0], [-3.0f64]];

let mut llt = Mat::<f64>::zeros(2, 2);
let no_par = Parallelism::None;

// Compute the size and alignment of the required scratch space
let cholesky_memory = cholesky_in_place_req::<f64>(
    a.nrows(),
    Parallelism::None,
    LltParams::default(),
).unwrap();
let solve_memory = solve_in_place_req::<f64>(
    a.nrows(),
    b.ncols(),
    Parallelism::None,
).unwrap();

// Allocate the scratch space
let mut memory = GlobalPodBuffer::new(cholesky_memory.or(solve_memory));
let mut stack = PodStack::new(&mut mem);

// Compute the decomposition
llt.copy_from(&a);
cholesky_in_place(
    llt.as_mut(),
    LltRegularization::default(), // no regularization
    no_par,
    stack.rb_mut(),               // scratch space
    LltParams::default(),         // default settings
);
// Solve the linear system
solve_in_place_with_conj(llt.as_ref(), Conj::No, b.as_mut(), no_par, stack);
```

If $A$ is not positive definite, the Bunch-Kaufman LBLT decomposition is recommended instead.
```rust
use faer::{mat, Side};
use faer::prelude::*;

let a = mat![
    [10.0, 2.0],
    [2.0, -10.0f64],
];
let b = mat![[15.0], [-3.0f64]];

// Compute the Bunch-Kaufman LBLT decomposition,
// reading only the lower triangular half of the matrix.
let lblt = a.lblt(Side::Lower);

let x = lblt.solve(&b);
```

=== $A$ is square
For a square matrix $A$, we can use the LU decomposition with partial pivoting,
or the full pivoting variant which is slower but can be more accurate when the
matrix is nearly singular.

```rust
use faer::mat;
use faer::prelude::*;

let a = mat![
    [10.0, 3.0],
    [2.0, -10.0f64],
];
let b = mat![[15.0], [-3.0f64]];

// Compute the LU decomposition with partial pivoting,
let plu = a.partial_piv_lu();
let x1 = plu.solve(&b);

// or the LU decomposition with full pivoting.
let flu = a.full_piv_lu();
let x2 = flu.solve(&b);
```

=== $A$ is a tall matrix (least squares solution)
When the linear system is over-determined, an exact solution may not
necessarily exist, in which case we can get a best-effort result by computing
the least squares solution.
That is, the solution that minimizes $||A x - b||$.

This can be done using the QR decomposition.

```rust
use faer::mat;
use faer::prelude::*;

let a = mat![
    [10.0, 3.0],
    [2.0, -10.0],
    [3.0, -45.0f64],
];
let b = mat![[15.0], [-3.0], [13.1f64]];

// Compute the QR decomposition.
let qr = a.qr();
let x = qr.solve_lstsq(&b);
```

== Computing the singular value decomposition
```rust
use faer::mat;
use faer::prelude::*;

let a = mat![
    [10.0, 3.0],
    [2.0, -10.0],
    [3.0, -45.0f64],
];

// Compute the SVD decomposition.
let svd = a.svd();
// Compute the thin SVD decomposition.
let svd = a.thin_svd();
// Compute the singular values.
let svd = a.singular_values();
```

== Computing the eigenvalue decomposition
```rust
use faer::mat;
use faer::prelude::*;
use faer::complex_native::c64;

let a = mat![
    [10.0, 3.0],
    [2.0, -10.0f64],
];

// Compute the eigendecomposition.
let evd = a.eigendecomposition::<c64>();

// Compute the eigenvalues.
let evd = a.eigen_values::<c64>();

// Compute the eigendecomposition assuming `a` is Hermitian.
let evd = a.selfadjoint_eigendecomposition();

// Compute the eigenvalues assuming `a` is Hermitian.
let evd = a.selfadjoint_eigenvalues();
```

= Sparse linear algebra
