#set text(font: "New Computer Modern")

#show par: set block(spacing: 0.55em)

#show heading: set block(above: 1.4em, below: 1em)

#set par(leading: 0.55em, justify: true)

#import "@preview/codly:0.1.0"

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
  )
)

= Introduction
`faer-rs` is a general-purpose linear algebra library for the Rust programming language.
With a focus on correctness, portability, and performance. In this book, we'll
be assuming version `0.15.0` of the library.

`faer` is designed around a high level API that sacrifices some amount of
performance and customizability in exchange for ease of use, as well as a low
level API that offers more control over memory allocations and multithreading
capabilities. The two APIs share the same data structures and can be used
together or separately, depending on the user's needs.

This book assumes some level of familiarity with Rust, linear algebra and `faer`'s API.
Users who are new to the library are encouraged to get started by taking a look
at the library's examples directory #footnote[`faer-rs/faer-libs/faer/examples`] and
browsing the `docs.rs` documentation #footnote[https://docs.rs/faer/0.15.0/faer/index.html].

We will go into detail over the various operations and matrix decompositions
that are provided by the library, as well as their implementation details. We
will also explain the architecture of `faer`'s data structures and how low
level operations are handled using vectorized SIMD instructions.

#pagebreak()

= Data layout and the `Entity` trait

In most linear algebra libraries, matrix data is stored contiguously in memory,
regardless of the scalar type. This can be done in two ways, either a row-major
layout or a column-major layout.

Consider the matrix
$ mat(
  a_11, a_12;
  a_21, a_22;
  a_31, a_32;
) $
Storing it in row-major layout would place the values in memory in the following order:
$ (
  a_11, a_12,
  a_21, a_22,
  a_31, a_32
), $
while storing it in column-major order would place the values in memory in this order:
$ (
  a_11, a_21, a_31,
  a_12, a_22, a_32
). $

`faer`, on the other hand, first splits each scalar into its atomic units,
then stores each unit matrix separately in a contiguous fashion. The library
does not mandate the usage of one layout or the other, but heavily prefers to receive
data in column-major layout (with notable exceptions).

The way in which a scalar can be split is chosen by the scalar type itself.
For example, a complex floating point type may choose to either be stored as one unit
or as a group of two units.

Given the following complex matrix:
$ mat(
  a_11 + b_11, a_12 + b_12;
  a_21 + b_21, a_22 + b_22;
  a_31 + b_31, a_32 + b_32;
), $
and assuming column-major layout, we can either choose the following storage scheme in which
the full number is considered a single unit:
$ (
  a_11, b_11, a_21, b_21, a_31, b_31,
  a_12, b_12, a_22, b_22, a_32, b_32
), $

or the following scheme in which the real and imaginary parts are considered two distinct units
$ (
  a_11, a_21, a_31,
  a_12, a_22, a_32
)\
(
  b_11, b_21, b_31,
  b_12, b_22, b_32
). $

The former is commonly referred to as AoS layout (array of structures), while
the latter is called SoA (structure of arrays). The choice of which one to use
depends on the context. As a general rule, types that are natively vectorizable
(have direct CPU support for arithmetic operations) prefer to be laid out in
AoS layout. On the other hand, types that do not have native vectorization
support but can still be vectorized by combining more primitive operations
prefer to be laid out in SoA layout.

Types that are not vectorizable may be in either one, but the AoS layout is
typically easier to work with, in that scenario.

== `Entity` trait
The `Entity` trait determines how a type prefers to be stored in memory,
through its associated type `Group`.

Given some type `E` that implements `Entity`, we can manipulate groups of
arbitrary types in a generic way.

For example, `faer_core::GroupFor<E, E::Unit>` is an `E`-group of `E::Unit`, which can be
thought of as a raw representation of `E`.

Pre-existing data can be referred to using a reference to a slice or a raw
pointer, for example `GroupFor<E, &[E::Unit]` or `GroupFor<E, *const E::Unit>`.

The `Entity` trait requires associated functions to convert from one `E`-group type to another.
For example, we can take a reference to each element in a group with
`E::faer_as_ref`, or `E::faer_as_mut`.

```rust
use faer_core::{Entity, GroupFor};

fn value_to_unit_references<E: Entity>(value: E) {
    let units: GroupFor<E, E::Unit> = value.into_units();
    let references: GroupFor<E, &E::Unit> = E::faer_as_ref(&units);
}
```

We can map one group type to another using `E::faer_map`.
```rust
use faer_core::{Entity, GroupFor};

fn slice_to_ptr<E: Entity>(
    slice: GroupFor<E, &[E::Unit]>
) -> GroupFor<E, *const E::Unit> {
    E::faer_map(slice, |slice| slice.as_ptr())
}
```

We can also zip and unzip groups of values with `E::faer_zip` and `E::faer_unzip`.
```rust
use faer_core::{Entity, GroupFor};

unsafe fn ptr_to_slice<'a, E: Entity>(
    ptr: GroupFor<E, *const E::Unit>,
    len: GroupFor<E, usize>
) -> GroupFor<E, &'a [E::Unit]> {
    let zipped: GroupFor<E, (*const E::Unit, usize)> = E::faer_zip(ptr, len);
    E::faer_map(zipped, |(ptr, len)| std::slice::from_raw_parts(ptr, len))
}

unsafe fn split_at<E: Entity>(
    slice: GroupFor<E, &[E::Unit]>,
    mid: usize
) -> (GroupFor<E, &[E::Unit]>, GroupFor<E, &[E::Unit]>) {
    E::faer_unzip(E::faer_map(slice, |slice| slice.split_at(mid)))
}
```

== Matrix layout
Matrices in `faer` fall into two broad categories with respect to layout. Owned
matrices (`Mat`) which are always stored in column-major layout, and matrix views
(`MatRef`/`MatMut`) which allow any strided layout.

Note that even though matrix views allow for any row and column stride, they
are still typically optimized for column major layout, since that happens to be
the preferred layout for most matrix decompositions.

#pagebreak()

Matrix views are roughly defined as:
```rust
struct MatRef<'a, E: Entity> {
    ptr: GroupFor<E, *const E::Unit>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
    __marker: PhantomData<&'a E>,
}

struct MatMut<'a, E: Entity> {
    ptr: GroupFor<E, *mut E::Unit>,
    nrows: usize,
    ncols: usize,
    row_stride: isize,
    col_stride: isize,
    __marker: PhantomData<&'a mut E>,
}
```

The actual implementation is slightly different in order to allow `MatRef` to
have `Copy` semantics, as well as make use of the fact that `ptr` is never null
to allow for niche optimizations (such as `Option<MatRef<'_, E>>` having the
same layout as `MatRef<'_, E>`).

`ptr` is a group of non-null pointers to units, each pointing to a matrix with an
underlying contiguous allocation. In other words, even though the data itself
is strided, it has to have a contiguous underlying storage in order to allow
for pointer arithmetic to be valid.

`nrows`, `ncols`, `row_stride` and `col_stride` are the matrix dimensions and
strides, which must be the same for every unit matrix in the group.

Finally, `__marker` imbues `MatRef` and `MatMut` with the correct variance,
This allows `MatRef<'short_lifetime, E>` to be a subtype of
`MatRef<'long_lifetime, E>`, which allows for better ergonomics.

In addition to `Copy` semantics for `MatRef`, both `MatRef` and `MatMut` naturally
provide `Move` semantics, as do most Rust types. On top of that, they also provide
`Reborrow` semantics, which currently need to be explicitly used, unlike native
references which are implicitly reborrowed.

#pagebreak()

Reborrowing is the act of temporarily borrowing a matrix view as another matrix
view with a shorter lifetime. For example, given a `MatMut<'a, E>`, we would like
to pass it to functions taking `MatMut<'_, E>` by value without having to consume
our object. Unlike `MatRef<'a, E>`, this is not done automatically as `MatMut`
is not `Copy`. The solution is to mutably reborrow our `MatMut` object like this
```rust
fn function_taking_mat_ref(mat: MatRef<'_, E>) {}
fn function_taking_mat_mut(mat: MatMut<'_, E>) {}

fn mutable_reborrow_example(mut mat: MatMut<'_, E>) {
    use faer::prelude::*;

    function_taking_mat_mut(mat.rb_mut());
    function_taking_mat_mut(mat.rb_mut());
    function_taking_mat_ref(mat.rb());
    function_taking_mat_ref(mat.rb());
    function_taking_mat_mut(mat);

    // does not compile, since `mat` was moved in the previous call
    // function_taking_mat_mut(mat);
}
```

Owned matrices on the other hand are roughly defined as:
```rust
struct Mat<E: Entity> {
    ptr: GroupFor<E, *mut E::Unit>,
    nrows: usize,
    ncols: usize,
    row_capacity: usize,
    col_capacity: usize,
    __marker: PhantomData<E>,
}

impl<E: Entity> Drop for Mat<E> {
    fn drop(&mut self) {
        // deallocate the storage
    }
}
```
Unlike matrix views, we don't need to explicitly store the strides. We know that
the row stride is equal to `1`, since the layout is column major, and the column
stride is equal to `row_capacity`.

We also have two new fields: `row_capacity` and `col_capacity`, which represent
how much storage we have for resizing the matrix without having to reallocate.

`Mat` can be converted to `MatRef` using `Mat::as_ref(&self)` or `MatMut` using
`Mat::as_mut(&mut self)`.

#pagebreak()

= Componentwise operations
Componentwise operations are operations that take $n$ matrices with matching
dimensions, producing an output of the same shape. Addition and subtraction
are examples of commonly used componentwise operations.

Componentwise operations can be expressed in `faer` using the `zipped!`
macro, followed by a call to `for_each` (for in-place iteration) or `map` (for
producing an output value).

```rust
use faer_core::{zipped, unzipped};

fn a_plus_3b(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    zipped!(a, b).map(|unzipped!(a, b)| {
        *a + 3.0 * *b
    })
}

fn swap_a_b(a: MatMut<'_, f64>, b: MatMut<'_, f64>) {
    zipped!(a, b).for_each(|unzipped!(mut a, mut b)| {
        (*a, *b) = (*b, *a);
    })
}
```

`zipped!` function calls can be more efficient than naive nested loops. The
reason for this is that `zipped!` analyzes the layout of the input matrices in
order to determine the optimal iteration order. For example whether it should
iterate over rows first, before columns. Or whether the iteration should happen
in reverse order (starting from the last row/column) instead of the forward
order.

Currently, `zipped!` determines the iteration order based on the preferred
iteration order of the first matrix, but this may change in a future release.

#pagebreak()

= Matrix multiplication
In this section we will give a detailed overview of the techniques used to
speed up matrix multiplication in `faer`. The approach we use is a
reimplementation of BLIS's matrix multiplication algorithm with some
modifications.

Consider three matrices $A$, $B$ and $C$, such that we want to perform the operation
$ C += A B. $

We can chunk $A$, $B$ and $C$ in a way that is compatible with matrix multiplication:
$
A = mat(
 A_11   , A_12   , ...      , A_(1 k);
 A_21   , A_22   , ...      , A_(2 k);
 dots.v , dots.v , dots.down, dots.v ;
 A_(m 1), A_(m 2), ...      , A_(m k);
),
B = mat(
 B_11   , B_12   , ...      , B_(1 n);
 B_21   , B_22   , ...      , B_(2 n);
 dots.v , dots.v , dots.down, dots.v ;
 B_(k 1), B_(k 2), ...      , B_(k n);
),
C = mat(
 C_11   , C_12   , ...      , C_(1 n);
 C_21   , C_22   , ...      , C_(2 n);
 dots.v , dots.v , dots.down, dots.v ;
 C_(m 1), C_(m 2), ...      , C_(m n);
).
$
