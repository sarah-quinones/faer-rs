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
With a focus on correctness, portability, and performance.

In this book, we'll be using version `0.15.0` of the library. All of the
examples are self contained and assume a classic Rust project structure with
the following `Cargo.toml` file.
```toml
[package]
name = "faer-example"
version = "0.0.0"
edition = "2021"

[dependencies]
faer          = "0.15.0"
faer-core     = "0.15.0"
faer-cholesky = "0.15.0"
faer-qr       = "0.15.0"
faer-lu       = "0.15.0"
faer-svd      = "0.15.0"
faer-evd      = "0.15.0"
```

`faer` is designed around a high level API that sacrifices some amount of
performance and customizability in exchange for ease of use, as well as a low
level API that offers more control over memory allocations and multithreading
capabilities. The two APIs share the same data structures and can be used
together or separately, depending on the user's needs.

This book assumes some level of familiarity with `faer`'s API.
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
  a_00, a_01;
  a_10, a_11;
  a_20, a_21;
) $
Storing it in row-major layout would place the values in memory in the following order:
$ (
  a_00, a_01,
  a_10, a_11,
  a_20, a_21
), $
while storing it in column-major order would place the values in memory in this order:
$ (
  a_00, a_10, a_20,
  a_01, a_11, a_21
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
  a_00 + b_00, a_01 + b_01;
  a_10 + b_10, a_11 + b_11;
  a_20 + b_20, a_21 + b_21;
), $
and assuming column-major layout, we can either choose the following storage scheme in which
the full number is considered a single unit:
$ (
  a_00, b_00, a_10, b_10, a_20, b_20,
  a_01, b_01, a_11, b_11, a_21, b_21
), $

or the following scheme in which the real and imaginary parts are considered two distinct units
$ (
  a_00, a_10, a_20,
  a_01, a_11, a_21
)\
(
  b_00, b_10, b_20,
  b_01, b_11, b_21
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
fn value_to_unit_references<E: Entity>(value: E) {
    let units: GroupFor<E, E::Unit> = value.into_units();
    let references: GroupFor<E, &E::Unit> = E::faer_as_ref(&units);
}
```

We can map one group type to another using `E::faer_map`.
```rust
fn slice_to_ptr<E: Entity>(
    slice: GroupFor<E, &[E::Unit]>
) -> GroupFor<E, *const E::Unit> {
    E::faer_map(slice, |slice| slice.as_ptr())
}
```

We can also zip and unzip groups of values with `E::faer_zip` and `E::faer_unzip`.
```rust
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
