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

_`faer`_ is designed around a high level API that sacrifices some amount of
performance and customizability in exchange for ease of use, as well as a low
level API that offers more control over memory allocations and multithreading
capabilities. The two APIs share the same data structures and can be used
together or separately, depending on the user's needs.

This book assumes some level of familiarity with Rust, linear algebra and _`faer`_'s API.
Users who are new to the library are encouraged to get started by taking a look
at the user guide, the library's examples directory
#footnote[`faer-rs/faer-libs/faer/examples`] and browsing the `docs.rs`
documentation #footnote[https://docs.rs/faer/0.16.0/faer/index.html].

We will go into detail over the various operations and matrix decompositions
that are provided by the library, as well as their implementation details. We
will also explain the architecture of _`faer`_'s data structures and how low
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

_`faer`_, on the other hand, first splits each scalar into its atomic units,
then stores each unit matrix separately in a contiguous fashion. The library
does not mandate the usage of one layout or the other, but heavily prefers to receive
data in column-major layout, with the notable exception of matrix multiplication which
we try to optimize for both column-major and row-major layouts.

The way in which a scalar can be split is chosen by the scalar type itself.
For example, a complex floating point type may choose to either be stored as one unit
or as a group of two units.

Given the following complex matrix:
$ mat(
  a_11 + i b_11, a_12 + i b_12;
  a_21 + i b_21, a_22 + i b_22;
  a_31 + i b_31, a_32 + i b_32;
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
),\
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
typically easier to work with in that scenario.

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
Matrices in _`faer`_ fall into two broad categories with respect to layout. Owned
matrices (`Mat`) which are always stored in column-major layout, and matrix views
(`MatRef`/`MatMut`) which allow any strided layout.

Note that even though matrix views allow for any row and column stride, they
are still typically optimized for column major layout, since that happens to be
the preferred layout for most matrix decompositions.

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

= Vector operations
== Componentwise operations
Componentwise operations are operations that take $n$ matrices with matching
dimensions, producing an output of the same shape. Addition and subtraction
are examples of commonly used componentwise operations.

Componentwise operations can be expressed in _`faer`_ using the `zipped!`
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

== Vectorized operations
SIMD (Single Instruction, Multiple Data) refers to the usage of CPU instructions
that take vectors of inputs, packed together in CPU registers, and perform the
same operation on all of them. As an example, classic addition takes two scalars
as an input and produces one output, while SIMD addition could take two vectors,
each containing 4 scalars, and adds them componentwise, producing an output vector
of 4 scalars. Correct SIMD usage is a crucial part of any linear algebra
library, given that most linear algebra operations lend themselves well to
vectorization.

== SIMD with _`pulp`_

_`faer`_ provides a common interface for generic and composable SIMD, using the
_`pulp`_ crate as a backend. _`pulp`_'s high level API abstracts away the differences
between various instruction sets and provides a common API that's generic over
them (but not the scalar type). This allows users to write a generic implementation
that gets turned into several functions, one for each possible instruction set
among a predetermined subset. Finally, the generic implementation can be used along
with an `Arch` structure that determines the best implementation at runtime.

Here's an example of how _`pulp`_ could be used to compute the expression $x^2 +
2y - |z|$, and store it into an output vector.

```rust
use core::iter::zip;

fn compute_expr(out: &mut[f64], x: &[f64], y: &[f64], z: &[f64]) {
    struct Impl<'a> {
        out: &'a mut [f64],
        x: &'a [f64],
        y: &'a [f64],
        z: &'a [f64],
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) {
            let Self { out, x, y, z } = self;

            let (out_head, out_tail) = S::f64s_as_mut_simd(out);
            let (x_head, x_tail) = S::f64s_as_simd(x);
            let (y_head, y_tail) = S::f64s_as_simd(y);
            let (z_head, z_tail) = S::f64s_as_simd(z);

            let two = simd.f64s_splat(2.0);
            for (out, (&x, (&y, &z))) in zip(
                out_head,
                zip(x_head, zip(y_head, z_head)),
            ) {
                *out = simd.f64s_add(
                    x,
                    simd.f64s_sub(simd.f64s_mul(two, y), simd.f64s_abs(z)),
                );
            }

            for (out, (&x, (&y, &z))) in zip(
                out_tail,
                zip(x_tail, zip(y_tail, z_tail)),
            ) {
                *out = x - 2.0 * y - z.abs();
            }
        }
    }

    pulp::Arch::new().dispatch(Impl { out, x, y, z });
}
```

There's a lot of things going on at the same time in this code example. Let us
go over them step by step.

_`pulp`_'s generic SIMD implementation happens through the `WithSimd` trait,
which takes `self` by value to pass in the function parameters. It additionally
provides another parameter to `with_simd` describing the instruction set being
used. `WithSimd::with_simd` *must* be marked with the `#[inline(always)]` attribute.
Forgetting to do so could lead to a significant performance drop.

Inside the body of the function, we split up each of `out`, `x`, `y` and
`z` into two parts using `S::f64s_as[_mut]_simd`. The first part (`head`) is a
slice of `S::f64s`, representing the vectorizable part of the original slice.
The second part (`tail`) contains the remainder that doesn't fit into a vector
register.

Handling the head section is done using vectorized operation. Currently these
need to take `simd` as a parameter, in order to guarantee its availability in a
sound way. This is what allows the API to be safe. The tail section is handled
using scalar operations.

The final step is actually calling into our SIMD implementation. This is done
by creating an instance of `pulp::Arch` that performs the runtime detection
(and caches the result, so that future invocations are as fast as possible),
then calling `Arch::dispatch` which takes a type that implements `WithSimd`,
and chooses the best SIMD implementation for it.

=== Memory alignment

Instead of splitting the input and output slices into two sections
(vectorizable head + non-vectorizable tail), an alternative approach would be
to split them up into three sections instead (vectorizable head + vectorizable
body + vectorizable tail). This can be accomplished using masked loads and
stores, which can speed things up if the slices are _similarly aligned_.

Similarly aligned slices are slices which have the same base address modulo
the byte size of the CPU's vector registers. The simplest way to guarantee this
is to allocate the slices in aligned memory (such that the base address is a
multiple of the register size in bytes), in which case the slices are similarly
aligned, and any subslices of them (with a shared offset and size) will also be
similarly aligned. Aligned allocation is done automatically for matrices in _`faer`_,
which helps uphold these guarantees for maximum performance.

Here's an example of how one might write an implementation that makes use of
memory alignment, using _`pulp`_.

```rust
use core::iter::zip;
use pulp::{Read, Write};

#[inline(always)]
fn compute_expr_register<S: pulp::Simd>(
    simd: S,
    mut out: impl Write<Output = S::f64s>,
    x: impl Read<Output = S::f64s>,
    y: impl Read<Output = S::f64s>,
    z: impl Read<Output = S::f64s>,
) {
    let zero = simd.f64s_splat(0.0);
    let x = x.read_or(zero);
    let y = y.read_or(zero);
    let z = z.read_or(zero);
    let two = simd.f64s_splat(2.0);
    out.write(simd.f64s_add(
        x,
        simd.f64s_sub(simd.f64s_mul(two, y), simd.f64s_abs(z)),
    ));
}
impl pulp::WithSimd for Impl<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let Self { out, x, y, z } = self;
        let offset = simd.f64s_align_offset(out.as_ptr(), out.len());

        let (out_head, out_body, out_tail) =
            simd.f64s_as_aligned_mut_simd(out, offset);
        let (x_head, x_body, x_tail) = simd.f64s_as_aligned_simd(x, offset);
        let (y_head, y_body, y_tail) = simd.f64s_as_aligned_simd(y, offset);
        let (z_head, z_body, z_tail) = simd.f64s_as_aligned_simd(z, offset);

        compute_expr_register(simd, out_head, x_head, y_head, z_head);
        for (out, (x, (y, z))) in zip(
            out_body,
            zip(x_body, zip(y_body, z_body)),
        ) {
            compute_expr_register(simd, out, x, y, z);
        }
        compute_expr_register(simd, out_tail, x_tail, y_tail, z_tail);
    }
}
```

_`faer`_ adds one more abstraction layer on top of _`pulp`_, in order to make the
SIMD operations generic over the scalar type. This is done using the
`faer_core::group_helpers::SimdFor<E, S>` struct that's effectively a thin
wrapper over `S`, and only exposes operations specific to the type `E`.

Here's how one might implement the previous operation for a generic real scalar type.

```rust
use faer_core::group_helpers::{SliceGroup, SliceGroupMut};
use faer_core::RealField;

struct Impl<'a, E: RealField> {
    out: SliceGroupMut<'a, E>,
    x: SliceGroup<'a, E>,
    y: SliceGroup<'a, E>,
    z: SliceGroup<'a, E>,
}
```

`&[f64]` and `&mut [f64]` are replaced by `SliceGroup<'_, E>` and
`SliceGroupMut<'_, E>`, to accomodate the fact that `E` might be an SoA type
that wants to be decomposed into multiple units. Aside from that change, most
of the code looks similar to what we had before.

```rust
use core::iter::zip;
use faer_core::{RealField, SimdGroupFor, group_helpers::SimdFor};
use pulp::{Read, Write};
use reborrow::*;
#[inline(always)]
fn compute_expr_register<E: RealField, S: pulp::Simd>(
    simd: SimdFor<E, S>,
    mut out: impl Write<Output = SimdGroupFor<E, S>>,
    x: impl Read<Output = SimdGroupFor<E, S>>,
    y: impl Read<Output = SimdGroupFor<E, S>>,
    z: impl Read<Output = SimdGroupFor<E, S>>,
) {
    let zero = simd.splat(E::faer_zero());
    let two = simd.splat(E::faer_from_f64(2.0));
    let x = x.read_or(zero);
    let y = y.read_or(zero);
    let z = z.read_or(zero);
    out.write(simd.add(x, simd.sub(simd.mul(two, y), simd.abs(z))));
}

```

```rust
impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let Self { out, x, y, z } = self;
        let simd = SimdFor::<E, S>::new(simd);
        let offset = simd.align_offset(out.rb());
        let (out_head, out_body, out_tail) =
            simd.as_aligned_simd_mut(out, offset);
        let (x_head, x_body, x_tail) = simd.as_aligned_simd(x, offset);
        let (y_head, y_body, y_tail) = simd.as_aligned_simd(y, offset);
        let (z_head, z_body, z_tail) = simd.as_aligned_simd(z, offset);
        compute_expr_register(simd, out_head, x_head, y_head, z_head);
        for (out, (x, (y, z))) in zip(
            out_body.into_mut_iter(),
            zip(
                x_body.into_ref_iter(),
                zip(y_body.into_ref_iter(), z_body.into_ref_iter())
            ),
        ) {
            compute_expr_register(simd, out, x, y, z);
        }
        compute_expr_register(simd, out_tail, x_tail, y_tail, z_tail);
    }
}
```

== SIMD reductions
The previous examples focused on _vertical_ operations, which compute the
output of a componentwise operation and storing the result in an output vector.
Another interesting kind of operations is _horizontal_ ones, which accumulate
the result of one or more vector into one or more scalar values.

One example of this is the dot product, which takes two vectors $a$ and $b$ of
size $n$ and computes $sum_(i = 0)^n a_i b_i$.

One way to implement it would be like this:

```rust
use faer_core::group_helpers::{SliceGroup, SliceGroupMut};
use faer_core::RealField;

struct Impl<'a, E: RealField> {
    a: SliceGroup<'a, E>,
    b: SliceGroup<'a, E>,
}
#[inline(always)]
fn dot_register<E: RealField, S: pulp::Simd>(
    simd: SimdFor<E, S>,
    acc: SimdGroupFor<E, S>,
    b: impl Read<Output = SimdGroupFor<E, S>>,
    a: impl Read<Output = SimdGroupFor<E, S>>,
) -> SimdGroupFor<E, S> {
    let zero = simd.splat(E::faer_zero());
    let a = a.read_or(zero);
    let b = b.read_or(zero);
    simd.mul_add(a, b, acc)
}

impl<E: RealField> pulp::WithSimd for Impl<'_, E> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let Self { a, b } = self;
        let simd = SimdFor::<E, S>::new(simd);
        let offset = simd.align_offset(a);

        let (a_head, a_body, a_tail) = simd.as_aligned_simd(a, offset);
        let (b_head, b_body, b_tail) = simd.as_aligned_simd(b, offset);

        let mut acc = simd.splat(E::faer_zero());
        acc = dot_register(simd, acc, a_head, b_head);
        for (a, b) in zip(a_body, b_body) {
            acc = dot_register(simd, acc, a, b);
        }
        acc = dot_register(simd, acc, a_tail, b_tail);

        simd.reduce_add(simd.rotate_left(acc, offset.rotate_left_amount()))
    }
}
```
The code looks similar to what we've written before. An interesting addition
is the use of `simd.rotate_left` in the last line. The reason for this is to
make sure our computed reduction doesn't depend on the memory offset, which can
help avoid variations in the output due to the non-associativity of floating point
arithmetic.

For example, suppose our register size is 4 elements, and we want to compute
the dot product of 13 elements from each of $a$ and $b$.

In the case where the memory is aligned, this is what the head, body and tail
of $a$ and $b$ look like:

$
a_("head") &= (a_1, a_2, a_3, a_4),\
b_("head") &= (b_1, b_2, b_3, b_4),\
\
a_("body") &= [(a_5, a_6, a_7, a_8), (a_9, a_10, a_11, a_12)],\
b_("body") &= [(b_5, b_6, b_7, b_8), (b_9, b_10, b_11, b_12)],\
\
a_("tail") &= (a_13, 0, 0, 0),\
b_("tail") &= (b_13, 0, 0, 0).\
$

Right before we perform the rotation, the accumulator contains the following result
$
  "acc"_1 &= a_1 b_1 + a_5 b_5 + a_9 b_9 + a_13 b_13,\
  "acc"_2 &= a_2 b_2 + a_6 b_6 + a_10 b_10 ,\
  "acc"_3 &= a_3 b_3 + a_7 b_7 + a_11 b_11 ,\
  "acc"_4 &= a_4 b_4 + a_8 b_8 + a_12 b_12 .\
$

If we assume the reduction operation `simd.reduce_add` sums the elements
sequentially, we get the final result:
$
  "acc"_"aligned" = &(a_1 b_1 + a_5 b_5 + a_9 b_9 + a_13 b_13)\
  &+ (a_2 b_2 + a_6 b_6 + a_10 b_10 )\
  &+ (a_3 b_3 + a_7 b_7 + a_11 b_11 )\
  &+ (a_4 b_4 + a_8 b_8 + a_12 b_12 ).\
$

Now let's take a look at the case where the memory is unaligned, for example
with an offset of 1. In this case $a$ and $b$ look like:

$
a'_("head") &= (0, a_1, a_2, a_3),\
b'_("head") &= (0, b_1, b_2, b_3),\
\
a'_("body") &= [(a_4, a_5, a_6, a_7), (a_8, a_9, a_10, a_11)],\
b'_("body") &= [(a_4, b_5, b_6, b_7), (b_8, b_9, b_10, b_11)],\
\
a'_("tail") &= (a_12, a_13, 0, 0),\
b'_("tail") &= (b_12, b_13, 0, 0).\
$

Right before we perform the rotation, the accumulator contains the following result
$
  "acc'"_1 &=  a_4 b_4 + a_8 b_8 + a_12 b_12,\
  "acc'"_2 &= a_1 b_1 + a_5 b_5 + a_9 b_9 + a_13 b_13,\
  "acc'"_3 &= a_2 b_2 + a_6 b_6 + a_10 b_10 ,\
  "acc'"_4 &= a_3 b_3 + a_7 b_7 + a_11 b_11 .\
$

If we use `simd.reduce_add` directly, without going through `simd.rotate_left` first, we get
this result:

$
  "result"_("unaligned"(1)) = &( a_4 b_4 + a_8 b_8 + a_12 b_12)\
  +&(a_1 b_1 + a_5 b_5 + a_9 b_9 + a_13 b_13)\
  +&(a_2 b_2 + a_6 b_6 + a_10 b_10 )\
  +&(a_3 b_3 + a_7 b_7 + a_11 b_11 )\
$

Mathematically, the result is equivalent, but since floating point operations round the result,
we would get a slightly different result for the aligned and unaligned cases.

Our solution is to first rotate the accumulator to the left by the alignment offset.
Doing this right before the accumulation would give us the rotated accumulator:
$
  "rotate"_1 &=& "acc'"_2 &= a_1 b_1 + a_5 b_5 + a_9 b_9 + a_13 b_13 &&= "acc"_1,\
  "rotate"_2 &=& "acc'"_3 &= a_2 b_2 + a_6 b_6 + a_10 b_10  &&= "acc"_2,\
  "rotate"_3 &=& "acc'"_4 &= a_3 b_3 + a_7 b_7 + a_11 b_11  &&= "acc"_3,\
  "rotate"_4 &=& "acc'"_1 &=  a_4 b_4 + a_8 b_8 + a_12 b_12 &&= "acc"_4.\
$

Summing these sequentially would then give us the exact result as $"acc"_"aligned"$.

#pagebreak()

= Matrix multiplication
In this section we will give a detailed overview of the techniques used to
speed up matrix multiplication in _`faer`_. The approach we use is a
reimplementation of BLIS's matrix multiplication algorithm with some
modifications.

Consider three matrices $A$, $B$ and $C$, such that we want to perform the operation
$ C "+=" A B. $

We can chunk $A$, $B$ and $C$ in a way that is compatible with matrix multiplication:
$
A = mat(
 A_11   , A_12   , ...      , A_(1 k);
 A_21   , A_22   , ...      , A_(2 k);
 dots.v , dots.v , dots.down, dots.v ;
 A_(m 1), A_(m 2), ...      , A_(m k);
),\
B = mat(
 B_11   , B_12   , ...      , B_(1 n);
 B_21   , B_22   , ...      , B_(2 n);
 dots.v , dots.v , dots.down, dots.v ;
 B_(k 1), B_(k 2), ...      , B_(k n);
),\
C = mat(
 C_11   , C_12   , ...      , C_(1 n);
 C_21   , C_22   , ...      , C_(2 n);
 dots.v , dots.v , dots.down, dots.v ;
 C_(m 1), C_(m 2), ...      , C_(m n);
).
$

Then the $C "+=" A B$ operation may be decomposed into:
#set math.mat(delim: none, column-gap: 2.0em)
$
mat(
 C_(1 1) "+=" sum_(p = 1)^(k) A_(1 p) B_(p 1), C_(1 2) "+=" sum_(p = 1)^(k) A_(1 p) B_(p 2), ...      , C_(1 n) "+=" sum_(p = 1)^(k) A_(1 p) B_(p n);
 C_(2 1) "+=" sum_(p = 1)^(k) A_(2 p) B_(p 1), C_(2 2) "+=" sum_(p = 1)^(k) A_(2 p) B_(p 2), ...      , C_(2 n) "+=" sum_(p = 1)^(k) A_(2 p) B_(p n);
 dots.v , dots.v , dots.down, dots.v ;
 C_(m 1) "+=" sum_(p = 1)^(k) A_(m p) B_(p 1), C_(m 2) "+=" sum_(p = 1)^(k) A_(m p) B_(p 2), ...      , C_(m n) "+=" sum_(p = 1)^(k) A_(m p) B_(p n);
).
$

#set math.mat(delim: "(", column-gap: 0.5em)

Doing so does not decrease the number of flops (floating point operations). But
this restructuring step can lead to a large speedup if done correctly, by
making use of cache locality on modern CPUs/GPUs.

The general idea revolves around memory reuse. For now, let us consider the
case of a single thread executing the entire operation. The multithreaded case
can be handled with a few adjustments.

 - The algorithm we use computes sequentially $C$ by column blocks. In other words, we first compute $C_(: 1)$, then $C_(: 2)$ and so on.
 - For each column block $j$, we sequentially iterate over $p$, computing all the terms $A_(: p) B_(p j)$, and accumulate them to the output.
 - Then for each row block $i$, we compute $A_(i p) B_(p j)$ and accumulate it to $C_(i j)$.

Since most modern CPUs have a hierarchical cache structure (usually ranging
from L1 (smallest) to L3 (largest)), we would like to make use of this in our
algorithm for maximum efficiency.

The way we exploit this is by choosing the chunk dimensions so that $B_(p j)$
remains in the L3 cache during each iteration of the second loop, and $A_(i p)$
remains in the L2 cache during each iteration of the third loop. This leaves us
with one more cache level to use: the L1 cache.

We make the most use out of this by chunking the inner product once again, resulting in
two more loop levels:
$
A_(i p) = mat(
 A'_1p     ;
 A'_2p     ;
 dots.v    ;
 A'_(m' p);
),\
B_(p j) = mat(
 B'_(p 1)   , B_(p 2)   , ...      , B_(p n');
),\
C_(i j) = mat(
  C'_(1 1)  , C'_(1 2)  , ...      , C'_(1 n')  ;
  C'_(2 1)  , C'_(2 2)  , ...      , C'_(2 n')  ;
  dots.v    , dots.v    , dots.down, dots.v      ;
  C'_(m' 1), C'_(m' 2), ...      , C'_(m' n');
).
$

We iterate over each column block $B'_(p j')$, then over each row block
$A'_(i' p)$,
and accumulate the product $A'_(i' p) B'_(p j')$ to $C'_(i' j')$.

In the outer loop, $B'_(p j')$ is brought from the L3 cache into the L1 cache,
and stays there until the outer loop iteration is done.
Then in the inner loop, it is brought once again from the L1 cache into
registers, while $A'_(i' p)$ is brought from the L2 cache into registers,
so that the computation can be performed.

This last step is done using a vectorized microkernel, which heavily uses
SIMD instructions to maximize efficiency.
The number of registers limits the size of the microkernel.

During each iteration $p'$ of the microkernel, we can bring in $m_r$ elements
from the $A'_(i' p')$, and $n_r$ elements from $B'_(p' j')$, where $m_r$
and $n_r$ are the dimensions of the microkernel (as well as the dimensions of
each block $C'_(i' j')$). We use the following algorithm:


 - Load one element from $B'$,
 - Load $m_r$ elements from $A'$,
 - Multiply the $m_r$ elements from $A'$ by the element from $B'$, and accumulate the result to $C'$

Consider x86-64 as an example, with the AVX2+FMA instruction set (256-bit
registers), and suppose the scalar type has a size of 64 bits so that each
register can hold $N = 256/64 = 4$ scalars. We have 16 available registers in
total, this means we can load one element from $B'$ into one register, and $m_r$
elements from $A'$ into $m_r / N$ registers.

Since we don't want to constantly read and write to $C'$ from main memory, we use a local accumulator
that occupies $m_r / N n_r$ registers.

In this case we have a total of 16 available registers, which need to hold $1 +
m_r / N + m_r / N n_r$ registers. A good choice for our case is $m_r = 3N$,
$n_r = 4$, which requires exactly 16 registers.

To determine the chunk sizes, we compute them starting from the innermost loop
to the outermost loop.

Given that we've already computed $m_r$ and $n_r$, we determine $k_c$ (the
number of columns of $A_(i p)$, and also the number of rows of $B_(p j)$) so
that $B'_(p, j')$ fits into the L1 cache, then we determine $m_c$ (the number
of rows of $A_(i p)$) so that $A_(i p)$ fits into the L2 cache.
And finally we determine $n_c$ (the number of columns of $B_(p j)$) so that
$B_(: j)$ fits into the L3 cache.

Note that bringing data into the cache is typically done automatically by the CPU.
However, in our case, we want to perform that explicitly by storing each L3/L2 chunk
into packed storage, which allows for contiguous access that's friendly to the CPU's
hardware prefetcher and minimizes TLB (Translation Lookaside Buffer) misses.

In order to parallelize the algorithm, we have a few options. The second
loop can't be easily parallelized without allocating extra storage for the
accumulators, since we have a data dependency in the second loop.
The microkernel also doesn't perform enough work to compensate for the overhead
of core synchronization, so it also makes for a poor multithreading candidate.

This leaves us with the first loop, as well as the third, fourth and fifth
loops. The first loop stores its data in the L3 cache, which is typically
shared between cores. So it's not usually a very attractive candidate for
parallelization.

Loops three through five however can be parallelized in a straightforward way,
and make good use of each core's separate L1 (and often L2) cache, which leads
to a significant speedup.

Since data is packed explicitly during matrix multiplication, the original
layout of the input matrices has little effect on efficiency when the
dimensions are medium or large. This has the side-effect of matrix multiplication
being highly efficient regardless of the matrix layout.

== Special cases
In this section, we will refer to matrix multiplication with dimensions $(m, n)
× (n, k)$ with $(m, n, k)$ as a shorthand.

For special common matrix dimensions, we do not usually want to go through the
aforementioned strategy, because the packing and unpacking steps, as well as
the microkernel indirection can add considerable overhead.

Such cases include:
- inner product $(1, 1, k)$,
- outer product: $(m, n, 1)$,
- matrix-vector: $(m, 1, k)$,
- vector-matrix: $(1, n, k)$, which can be rewritten in terms of matrix-vector by transposing, if we assume that the scalar multiplication is commutative (which _`faer`_ generally does), since $C "+=" A B <=> C^top "+=" B^top A^top$.

The $(1, 1, k)$ case can be optimized for when $A$ is row-major and $B$ is
column-major, and is written similarly to our previous dot product example,
with one difference: Instead of using one accumulator for the result, we use
multiple accumulators, and then sum them together at the end. This can speed up
the computation by making use of instruction level parallelism, since each
accumulator can be computed independently from the others.

For the $(m, n, 1)$ case, we assume $C$ is column-major. If it is row-major, we
can implicitly transpose the matrix multiply operation. The algorithm we use
consists of computing $C$ column by column, which is equivalent to $C_(: j) "+="
A b_(1 j)$. This can be vectorized as a vertical operation, if $A$ is column-major.
If it is not, we can store it to contiguous temporary storage before performing
the computation. Note that this is a relatively cheap operation since its
dimensions are $(m, 1)$, which is usually much smaller than the size of $C$:
$(m, n)$.

For the $(m, 1, k)$ case, there are two interesting cases. The first one is
when $A$ is column major. In this case we assume $C$ is column major
(otherwise, we can compute the result in a temporary vector and accumulate it
to $C$ afterwards). For each column of $A$, we multiply it by the corresponding
element of $B$ and accumulate it to $C$. The inner kernel for this operation is
$C "+=" A_(: k) b_(k 1)$, which is essentially the same as the one from the outer
product.

When $A$ is row-major, we assume $B$ is column-major, and compute $C_(i 1) "+="
A_(i :) B_(: 1)$, which uses the same kernel as the $(1, 1, k)$ case.

== Triangular matrix products
In some cases, one of the matrices $A$ and $B$ is triangular (with a possibly
implicit zero or one diagonal), or we only want to compute the lower or upper
half of the output. _`faer`_ currently uses recursive implementations that are
padded and handled as rectangular matrix multiplication in the base case. For
example, we may want to compute $A B$ where $A$ is lower triangular and $B$ is
upper triangular:

$
mat(
  C_(1 1), C_(1 2);
  C_(2 1), C_(2 2);
) "+="
mat(
  A_(1 1), 0;
  A_(2 1), A_(2 2);
)
mat(
  B_(1 1), B_(1 2);
  0      , B_(2 2);
)
$

This can be split up into a sequence of products:

$
C_(1 1) & "+=" A_(1 1) B_(1 1),\
C_(1 2) & "+=" A_(1 1) B_(1 2),\
C_(2 1) & "+=" A_(2 1) B_(1 1),\
C_(2 2) & "+=" A_(2 1) B_(1 2) + A_(2 2) B_(2 2).
$

The steps $C_(1 1) "+=" A_(1 1) B_(1 1)$ and $C_(2 2) "+=" A_(2 2) B_(2 2)$
are also matrix prodcuts where the LHS is lower triangular and the RHS is upper
triangular, so we call the algorithm recursively for them.

All of these products can be performed either sequentially or in parallel.
In the parallel case, we group them like this to avoid load imbalances between
threads.

#set align(center)
#gridx(
  columns: (auto, auto),
  align: center,
  "Thread 1", vlinex(), "Thread 2",
  $
  C_(1 1) & "+=" A_(1 1) B_(1 1)\
  C_(2 2) & "+=" A_(2 1) B_(1 2) + A_(2 2) B_(2 2)
  $, (),
  $
  C_(1 2) & "+=" A_(1 1) B_(1 2)\
  C_(2 1) & "+=" A_(2 1) B_(1 1)\
  $,
)
#set align(left)

This way each thread performs roughtly the same number of flops, which helps
avoid idling threads that spend time waiting for the others to finish.
Every time we recurse to another triangular matrix multiplication, we can split
up the work again. And if we perform a rectangular matrix multiply, we can rely on
its inherent parallelism.

This is one of the scenarios where _`faer`_'s fine control over multithreading shines,
as we can provide a hint for each nested operation that it doesn't need to use
all the available cores, which can reduce synchronization overhead and conflict
over shared resources by the different threads.

#colorbox(
  title: "PERF",
  color: "blue",
  radius: 2pt,
  width: auto
)[
  The current strategy doesn't take advantage of the CPU's cache hierarchy for
  deciding how to split up the work between threads. This could lead to
  multiple threads contending over the L3 cache for very large matrices.

  In that case it could be worth investigating if it's better to only use
  triangular threading when the sizes fall below a certain threshold.
]

#colorbox(
  title: "PERF",
  color: "blue",
  radius: 2pt,
  width: auto
)[
  Specialized implementations for specific dimensions, similarly to what is
  done for the rectangular matrix multiply. For example a lower triangular
  matrix times a column vector.
]

== Triangular matrix solve
Solving systems of the form $A X = B$ (where $A$ is a triangular matrix) is
another primitive that is implemented using a recursive implementation that
relies on matrix multiplication. For simplicity, we assume $A$ is lower
triangular. The cases where $A$ is unit lower triangular or [unit] upper
triangular are handled similarly.

If we decompose $A$, $B$ and $X$ as follows:
$
A = mat(A_(1 1), ; A_(2 1), A_(2 2)),
B = mat(B_1; B_2),
X = mat(X_1; X_2),
$
then the system can be reformulated as
$
A_(1 1) X_1 &= B_1,\
A_(2 2) X_2 &= B_2 - A_(2 1) X_1.
$
This system can be solved sequentially by first solving the first equation,
then substituting its solution in the second equation. Moreover, the system can
be solved in place, by taking an inout parameter that contains $B$ on input,
and $X$ on output.

Once the recursion reaches a certain threshold, we fall back to a sequential
implementation to avoid the recursion overhead.

= How To...

This chapter aims at providing helpful guides for new contributors and seasond ones who may have forgotten a thing or two. 

== Setup a New Faer Library

Fear implements feature groups as new sub-crates in `fare-libs`. Assuming the new feature group is called `foo`, each subdirectory should
 - contain a copy `katex-header.html`.
 - a `cargo.toml` file, matching the one of the base crate `faer-libs`. 
 - should contain a `faer-libs/faer-foo/benches/bench.rs`.
 - contain a copy of the project license. 

Additionally
 - the new crate needs to be added to the `workspace/members` array in `faer-libs/cargo.toml`. 
 - tests are implemented in `faer-libs/faer-foo/src/lib.rs` and can be run via `cargo test --package faer-linop --lib -- tests`. 
 - benchmarks are implemented in `faer-libs/faer-foo/benches/` and can be run via `cd faer-libs/faer-foo && cargo bench --bench bench -- bar`.

Examples can be found in the `foo`s sibling crates.