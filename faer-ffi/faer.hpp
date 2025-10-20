#pragma once

#include "faer.h"
#include "quad.hpp"
#include "stddef.h"
#include <complex>
#include <vector>

#define LIBFAER_CAST_REAL(x) reinterpret_cast<::FaerV0_23_Real const *>(&x)
#define LIBFAER_CAST_SCALAR(x) reinterpret_cast<::FaerV0_23_Scalar const *>(&x)
#define LIBFAER_ALLOC(layout)                                                  \
  ((mem.ptr ? mem : detail::mem(detail::alloc(layout))).mem_ffi())

#define LIBFAER_SCALAR_DISPATCH(name)                                          \
  detail::ScalarDispatch{                                                      \
      .f32 = libfaer_v0_23_##name##_f32,                                       \
      .f64 = libfaer_v0_23_##name##_f64,                                       \
      .fx128 = libfaer_v0_23_##name##_fx128,                                   \
      .c32 = libfaer_v0_23_##name##_c32,                                       \
      .c64 = libfaer_v0_23_##name##_c64,                                       \
      .cx128 = libfaer_v0_23_##name##_cx128,                                   \
  }                                                                            \
      .for_type(detail::TYPE<T>)

#define LIBFAER_INDEX_DISPATCH(name)                                           \
  detail::ScalarDispatch{                                                      \
      .f32 =                                                                   \
          detail::IndexDispatch{                                               \
              .u32 = libfaer_v0_23_##name##_u32_f32,                           \
              .u64 = libfaer_v0_23_##name##_u64_f32,                           \
          }                                                                    \
              .for_type(detail::TYPE<I>),                                      \
      .f64 =                                                                   \
          detail::IndexDispatch{                                               \
              .u32 = libfaer_v0_23_##name##_u32_f64,                           \
              .u64 = libfaer_v0_23_##name##_u64_f64,                           \
          }                                                                    \
              .for_type(detail::TYPE<I>),                                      \
      .fx128 =                                                                 \
          detail::IndexDispatch{                                               \
              .u32 = libfaer_v0_23_##name##_u32_fx128,                         \
              .u64 = libfaer_v0_23_##name##_u64_fx128,                         \
          }                                                                    \
              .for_type(detail::TYPE<I>),                                      \
      .c32 =                                                                   \
          detail::IndexDispatch{                                               \
              .u32 = libfaer_v0_23_##name##_u32_c32,                           \
              .u64 = libfaer_v0_23_##name##_u64_c32,                           \
          }                                                                    \
              .for_type(detail::TYPE<I>),                                      \
      .c64 =                                                                   \
          detail::IndexDispatch{                                               \
              .u32 = libfaer_v0_23_##name##_u32_c64,                           \
              .u64 = libfaer_v0_23_##name##_u64_c64,                           \
          }                                                                    \
              .for_type(detail::TYPE<I>),                                      \
      .cx128 =                                                                 \
          detail::IndexDispatch{                                               \
              .u32 = libfaer_v0_23_##name##_u32_cx128,                         \
              .u64 = libfaer_v0_23_##name##_u64_cx128,                         \
          }                                                                    \
              .for_type(detail::TYPE<I>),                                      \
  }                                                                            \
      .for_type(detail::TYPE<T>)

namespace faer {
inline namespace v0_23 {

namespace meta {
template <typename T> struct Ffi {
  using Mat = FaerV0_23_MatMut;
  using Vec = FaerV0_23_VecMut;
  using Slice = FaerV0_23_SliceMut;
};
template <typename T> struct Ffi<T const> {
  using Mat = FaerV0_23_MatRef;
  using Vec = FaerV0_23_VecRef;
  using Slice = FaerV0_23_SliceRef;
};
} // namespace meta

namespace detail {
template <typename T> struct Type {};
template <typename T> inline constexpr Type<T> TYPE{};

/// @brief Scalar type dispatch table
/// tparam F function pointer type
/// see IndexDispatch
/// description my long description
template <typename F> struct ScalarDispatch {
  F f32;
  F f64;
  F fx128;
  F c32;
  F c64;
  F cx128;

  constexpr F for_type(Type<float>) const { return this->f32; }
  constexpr F for_type(Type<double>) const { return this->f64; }
  constexpr F for_type(Type<quad::Quad>) const { return this->fx128; }
  constexpr F for_type(Type<std::complex<float>>) const { return this->c32; }
  constexpr F for_type(Type<std::complex<double>>) const { return this->c64; }
  constexpr F for_type(Type<std::complex<quad::Quad>>) const {
    return this->cx128;
  }
};

template <typename F> struct IndexDispatch {
  F u32;
  F u64;

  constexpr F for_type(Type<unsigned>) const {
    if constexpr (sizeof(unsigned) == 4) {
      return this->u32;
    } else if constexpr (sizeof(unsigned) == 8) {
      return this->u64;
    } else {
      static_assert(sizeof(F) == 0, "unsupported type");
    }
  }
  constexpr F for_type(Type<unsigned long>) const {
    if constexpr (sizeof(unsigned long) == 4) {
      return this->u32;
    } else if constexpr (sizeof(unsigned long) == 8) {
      return this->u64;
    } else {
      static_assert(sizeof(F) == 0, "unsupported type");
    }
  }
  constexpr F for_type(Type<unsigned long long>) const {
    if constexpr (sizeof(unsigned long long) == 4) {
      return this->u32;
    } else if constexpr (sizeof(unsigned long long) == 8) {
      return this->u64;
    } else {
      static_assert(sizeof(F) == 0, "unsupported type");
    }
  }
};

template <typename T> ScalarDispatch(T, T, T, T, T, T) -> ScalarDispatch<T>;

} // namespace detail

template <typename T> struct Mat {
  T *ptr;
  size_t nrows;
  size_t ncols;
  ptrdiff_t row_stride;
  ptrdiff_t col_stride;

  constexpr auto ffi() const -> typename meta::Ffi<T>::Mat {
    return {
        .ptr = this->ptr,
        .nrows = this->nrows,
        .ncols = this->ncols,
        .row_stride = this->row_stride,
        .col_stride = this->col_stride,
    };
  }
};

template <typename I> struct Slice {
  I *ptr;
  size_t len;

  constexpr auto mem_ffi() const -> FaerV0_23_MemAlloc {
    return {
        .ptr = this->ptr,
        .len_bytes = sizeof(I) * this->len,
    };
  }

  constexpr auto ffi() const -> typename meta::Ffi<I>::Slice {
    return {
        .ptr = this->ptr,
        .len = sizeof(I) * this->len,
    };
  }
};

template <typename T> struct Vec {
  T *ptr;
  size_t len;
  ptrdiff_t stride;

  constexpr auto ffi() const -> typename meta::Ffi<T>::Vec {
    return {
        .ptr = this->ptr,
        .len = this->len,
        .stride = this->stride,
    };
  }
};

template <typename T> Mat(T *, size_t, size_t, ptrdiff_t, ptrdiff_t) -> Mat<T>;
template <typename T> Vec(T *, size_t, ptrdiff_t) -> Vec<T>;
template <typename T> Slice(T *, size_t) -> Slice<T>;

struct Par {
  enum {
    Seq,
    Rayon,
  } which;
  size_t nthreads;

  constexpr auto ffi() const -> FaerV0_23_Par {
    return {
        .tag = this->which == Seq ? FaerV0_23_ParTag::FaerV0_23_ParTag_Seq
                                  : FaerV0_23_ParTag::FaerV0_23_ParTag_Rayon,
        .nthreads = this->nthreads,
    };
  }
};

inline auto get_global_par() -> Par {
  auto par = libfaer_v0_23_get_global_par();
  return Par{
      .which = par.tag == FaerV0_23_ParTag::FaerV0_23_ParTag_Seq ? Par::Seq
                                                                 : Par::Rayon,
      .nthreads = par.nthreads,
  };
}

inline auto set_global_par(Par par) { libfaer_v0_23_set_global_par(par.ffi()); }

struct Layout {
  size_t size;
  size_t align;

  constexpr static auto from_ffi(FaerV0_23_Layout ffi) -> Layout {
    return {
        .size = ffi.len_bytes,
        .align = ffi.align_bytes,
    };
  }

  constexpr auto ffi() const -> FaerV0_23_Layout {
    return {
        .len_bytes = this->size,
        .align_bytes = this->align,
    };
  }
};

using Mem = Slice<unsigned char>;

namespace detail {
inline auto mem(std::vector<unsigned char> &&v) -> Mem {
  return Mem{
      v.data(),
      v.size(),
  };
}

inline auto alloc(Layout l) -> std::vector<unsigned char> {
  return std::vector<unsigned char>(l.size + l.align);
}
} // namespace detail

namespace linalg {
namespace cholesky {

namespace llt {
namespace factor {
using Params = FaerV0_23_LltParams;

template <typename T> struct Regularization {
  T dynamic_regularization_delta;
  T dynamic_regularization_epsilon;
};

template <typename T> auto params() -> Params {
  constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(LltParams);
  return ptr();
}

template <typename T> auto scratch(size_t n, Par par, Params params) -> Layout {
  constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(llt_factor_in_place_scratch);
  return Layout::from_ffi(ptr(n, par.ffi(), params));
}

template <typename T>
void in_place(Mat<T> A, Regularization<T> regularization = {},
              Par par = get_global_par(), Mem mem = {},
              Params params = factor::params<T>()) {
  constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(llt_factor_in_place);
  ptr(A.ffi(),
      {
          .dynamic_regularization_delta =
              LIBFAER_CAST_REAL(regularization.dynamic_regularization_delta),
          .dynamic_regularization_epsilon =
              LIBFAER_CAST_REAL(regularization.dynamic_regularization_epsilon),
      },
      par.ffi(), LIBFAER_ALLOC(scratch<T>(A.nrows, par, params)), params);
}
} // namespace factor

} // namespace llt

namespace llt_pivoting {}

namespace ldlt {}

namespace lblt {}

} // namespace cholesky
} // namespace linalg

} // namespace v0_23
} // namespace faer

#undef LIBFAER_CAST_REAL
#undef LIBFAER_CAST_SCALAR
#undef LIBFAER_ALLOC
#undef LIBFAER_SCALAR_DISPATCH
#undef LIBFAER_INDEX_DISPATCH
