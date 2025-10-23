#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "faer.h"
#include "quad.hpp"

#define LIBFAER_CAST_REAL(x) reinterpret_cast<::FaerV0_23_Real const*>(&x)
#define LIBFAER_CAST_SCALAR(x) reinterpret_cast<::FaerV0_23_Scalar const*>(&x)
#define LIBFAER_ALLOC(...) \
    ((mem.ptr ? mem : detail::mem(detail::alloc(__VA_ARGS__))).mem_ffi())

#define LIBFAER_SCALAR_DISPATCH(name) \
    static_cast<decltype(&libfaer_v0_23_##name##_f32)>( \
        detail::ScalarDispatch { \
            .f32 = libfaer_v0_23_##name##_f32, \
            .f64 = libfaer_v0_23_##name##_f64, \
            .fx128 = libfaer_v0_23_##name##_fx128, \
            .c32 = libfaer_v0_23_##name##_c32, \
            .c64 = libfaer_v0_23_##name##_c64, \
            .cx128 = libfaer_v0_23_##name##_cx128, \
        } \
            .for_type(detail::TYPE<T>) \
    )

#define LIBFAER_INDEX_DISPATCH(name) \
    static_cast<decltype(&libfaer_v0_23_##name##_u32_f32)>( \
        detail::ScalarDispatch { \
            .f32 = \
                detail::IndexDispatch { \
                    .u32 = libfaer_v0_23_##name##_u32_f32, \
                    .u64 = libfaer_v0_23_##name##_u64_f32, \
                } \
                    .for_type(detail::TYPE<I>), \
            .f64 = \
                detail::IndexDispatch { \
                    .u32 = libfaer_v0_23_##name##_u32_f64, \
                    .u64 = libfaer_v0_23_##name##_u64_f64, \
                } \
                    .for_type(detail::TYPE<I>), \
            .fx128 = \
                detail::IndexDispatch { \
                    .u32 = libfaer_v0_23_##name##_u32_fx128, \
                    .u64 = libfaer_v0_23_##name##_u64_fx128, \
                } \
                    .for_type(detail::TYPE<I>), \
            .c32 = \
                detail::IndexDispatch { \
                    .u32 = libfaer_v0_23_##name##_u32_c32, \
                    .u64 = libfaer_v0_23_##name##_u64_c32, \
                } \
                    .for_type(detail::TYPE<I>), \
            .c64 = \
                detail::IndexDispatch { \
                    .u32 = libfaer_v0_23_##name##_u32_c64, \
                    .u64 = libfaer_v0_23_##name##_u64_c64, \
                } \
                    .for_type(detail::TYPE<I>), \
            .cx128 = \
                detail::IndexDispatch { \
                    .u32 = libfaer_v0_23_##name##_u32_cx128, \
                    .u64 = libfaer_v0_23_##name##_u64_cx128, \
                } \
                    .for_type(detail::TYPE<I>), \
        } \
            .for_type(detail::TYPE<T>) \
    )

namespace faer {
inline namespace v0_23 {
    namespace meta {
        template<typename T>
        struct Ffi {
            using Mat = FaerV0_23_MatMut;
            using Vec = FaerV0_23_VecMut;
            using Slice = FaerV0_23_SliceMut;
        };

        template<typename T>
        struct Ffi<T const> {
            using Mat = FaerV0_23_MatRef;
            using Vec = FaerV0_23_VecRef;
            using Slice = FaerV0_23_SliceRef;
        };
    } // namespace meta

    namespace detail {
        template<typename T>
        struct Type {};

        template<typename T>
        inline constexpr Type<T> TYPE {};

        /// @brief Scalar type dispatch table
        /// tparam F function pointer type
        /// see IndexDispatch
        /// description my long description
        template<typename F>
        struct ScalarDispatch {
            F f32;
            F f64;
            F fx128;
            F c32;
            F c64;
            F cx128;

            constexpr F for_type(Type<float>) const {
                return this->f32;
            }

            constexpr F for_type(Type<double>) const {
                return this->f64;
            }

            constexpr F for_type(Type<quad::Quad>) const {
                return this->fx128;
            }

            constexpr F for_type(Type<std::complex<float>>) const {
                return this->c32;
            }

            constexpr F for_type(Type<std::complex<double>>) const {
                return this->c64;
            }

            constexpr F for_type(Type<std::complex<quad::Quad>>) const {
                return this->cx128;
            }
        };

        template<typename F>
        struct IndexDispatch {
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

        template<typename T>
        ScalarDispatch(T, T, T, T, T, T) -> ScalarDispatch<T>;
        template<typename T>
        IndexDispatch(T, T) -> IndexDispatch<T>;

    } // namespace detail

    template<typename T>
    struct Mat {
        T* ptr;
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

    template<typename I>
    struct Slice {
        I* ptr;
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

    template<typename T>
    struct Vec {
        T* ptr;
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

    template<typename T>
    Mat(T*, size_t, size_t, ptrdiff_t, ptrdiff_t) -> Mat<T>;
    template<typename T>
    Vec(T*, size_t, ptrdiff_t) -> Vec<T>;
    template<typename T>
    Slice(T*, size_t) -> Slice<T>;

    struct Par {
        enum {
            Seq,
            Rayon,
        } which;

        size_t nthreads;

        constexpr auto ffi() const -> FaerV0_23_Par {
            return {
                .tag = this->which == Seq
                    ? FaerV0_23_ParTag::FaerV0_23_ParTag_Seq
                    : FaerV0_23_ParTag::FaerV0_23_ParTag_Rayon,
                .nthreads = this->nthreads,
            };
        }
    };

    struct Conj {
        enum {
            No,
            Yes,
        } which;

        constexpr auto ffi() const -> FaerV0_23_Conj {
            return this->which == No ? FaerV0_23_Conj_No : FaerV0_23_Conj_Yes;
        }
    };

    struct ComputeSvdVectors {
        enum {
            No,
            Thin,
            Full,
        } which;

        constexpr auto ffi() const -> FaerV0_23_ComputeSvdVectors {
            return this->which == No  ? FaerV0_23_ComputeSvdVectors_No
                : this->which == Thin ? FaerV0_23_ComputeSvdVectors_Thin
                                      : FaerV0_23_ComputeSvdVectors_Full;
        }
    };

    struct ComputeEigenvectors {
        enum {
            No,
            Yes,
        } which;

        constexpr auto ffi() const -> FaerV0_23_ComputeEigenvectors {
            return this->which == No ? FaerV0_23_ComputeEigenvectors_No
                                     : FaerV0_23_ComputeEigenvectors_Yes;
        }
    };

    inline auto get_global_par() -> Par {
        auto par = libfaer_v0_23_get_global_par();
        return Par {
            .which = par.tag == FaerV0_23_ParTag::FaerV0_23_ParTag_Seq
                ? Par::Seq
                : Par::Rayon,
            .nthreads = par.nthreads,
        };
    }

    inline auto set_global_par(Par par) {
        libfaer_v0_23_set_global_par(par.ffi());
    }

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

    struct Mem: Slice<std::byte> {
        constexpr Mem() noexcept : Slice {} {}

        constexpr Mem(std::byte* ptr, size_t len) noexcept : Slice {ptr, len} {}

        Mem(std::vector<std::byte>& mem) noexcept :
            Slice {
                mem.data(),
                mem.size(),
            } {}
    };

    namespace detail {
        inline auto mem(std::vector<std::byte>&& v) -> Mem {
            return Mem {
                v.data(),
                v.size(),
            };
        }

        inline auto alloc(Layout l) -> std::vector<std::byte> {
            return std::vector<std::byte>(l.size + l.align);
        }
    } // namespace detail

    namespace linalg {
        namespace cholesky {
            namespace llt {
                namespace factor {
                    using Params = FaerV0_23_LltParams;

                    template<typename T>
                    struct Regularization {
                        T dynamic_regularization_delta;
                        T dynamic_regularization_epsilon;
                    };

                    template<typename T>
                    auto params() -> Params {
                        constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(LltParams);
                        return ptr();
                    }

                    template<typename T>
                    auto in_place_scratch(
                        size_t n,
                        Par par = get_global_par(),
                        Params params = factor::params<T>()
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(
                            llt_factor_in_place_scratch
                        );
                        return Layout::from_ffi(ptr(n, par.ffi(), params));
                    }

                    template<typename T>
                    void in_place(
                        Mat<T> A,
                        Regularization<T> regularization = {},
                        Par par = get_global_par(),
                        Mem mem = {},
                        Params params = factor::params<T>()
                    ) {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(llt_factor_in_place);
                        ptr(A.ffi(),
                            {
                                .dynamic_regularization_delta =
                                    LIBFAER_CAST_REAL(
                                        regularization
                                            .dynamic_regularization_delta
                                    ),
                                .dynamic_regularization_epsilon =
                                    LIBFAER_CAST_REAL(
                                        regularization
                                            .dynamic_regularization_epsilon
                                    ),
                            },
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<T>(A.nrows, par, params)
                            ),
                            params);
                    }
                } // namespace factor

                namespace solve {
                    template<typename T>
                    auto in_place_scratch(size_t n, size_t rhs_ncols, Par par)
                        -> Layout {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(llt_solve_in_place_scratch);
                        return Layout::from_ffi(ptr(n, rhs_ncols, par.ffi()));
                    }

                    template<typename T>
                    void in_place(
                        Mat<T const> L,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(llt_solve_in_place);
                        ptr(L.ffi(),
                            conj_A.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<T>(L.nrows, rhs.ncols, par)
                            ));
                    }
                } // namespace solve
            } // namespace llt

            namespace llt_pivoting {
                namespace factor {
                    using Params = FaerV0_23_PivLltParams;

                    template<typename T>
                    auto params() -> Params {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(PivLltParams);
                        return ptr();
                    }

                    template<typename I, typename T>
                    auto in_place_scratch(
                        size_t n,
                        Par par = get_global_par(),
                        Params params = factor::params<T>()
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            piv_llt_factor_in_place_scratch
                        );
                        return Layout::from_ffi(ptr(n, par.ffi(), params));
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T> A,
                        Slice<I> perm_fwd,
                        Slice<I> perm_bwd,
                        Par par = get_global_par(),
                        Mem mem = {},
                        Params params = factor::params<T>()
                    ) {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(piv_llt_factor_in_place);
                        ptr(A.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(A.nrows, par, params)
                            ),
                            params);
                    }
                } // namespace factor

                namespace solve {
                    template<typename I, typename T>
                    auto in_place_scratch(size_t n, size_t rhs_ncols, Par par)
                        -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            piv_llt_solve_in_place_scratch
                        );
                        return Layout::from_ffi(ptr(n, rhs_ncols, par.ffi()));
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T const> L,
                        Slice<I const> perm_fwd,
                        Slice<I const> perm_bwd,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(piv_llt_solve_in_place);
                        ptr(L.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            conj_A.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(L.nrows, rhs.ncols, par)
                            ));
                    }

                } // namespace solve
            } // namespace llt_pivoting

            namespace ldlt {
                namespace factor {
                    using Params = FaerV0_23_LdltParams;

                    template<typename T>
                    struct Regularization {
                        Slice<int8_t> signs;
                        T dynamic_regularization_delta;
                        T dynamic_regularization_epsilon;
                    };

                    template<typename T>
                    auto params() -> Params {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(LdltParams);
                        return ptr();
                    }

                    template<typename T>
                    auto in_place_scratch(
                        size_t n,
                        Par par = get_global_par(),
                        Params params = factor::params<T>()
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(
                            ldlt_factor_in_place_scratch
                        );
                        return Layout::from_ffi(ptr(n, par.ffi(), params));
                    }

                    template<typename T>
                    void in_place(
                        Mat<T> A,
                        Regularization<T> regularization = {},
                        Par par = get_global_par(),
                        Mem mem = {},
                        Params params = factor::params<T>()
                    ) {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(ldlt_factor_in_place);
                        ptr(A.ffi(),
                            FaerV0_23_LdltRegularization {
                                .dynamic_regularization_delta =
                                    LIBFAER_CAST_REAL(
                                        regularization
                                            .dynamic_regularization_delta
                                    ),
                                .dynamic_regularization_epsilon =
                                    LIBFAER_CAST_REAL(
                                        regularization
                                            .dynamic_regularization_epsilon
                                    ),
                                .dynamic_regularization_signs =
                                    regularization.signs.ffi(),
                            },
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<T>(A.nrows, par, params)
                            ),
                            params);
                    }
                } // namespace factor

                namespace solve {
                    template<typename T>
                    auto in_place_scratch(size_t n, size_t rhs_ncols, Par par)
                        -> Layout {
                        constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(
                            ldlt_solve_in_place_scratch
                        );
                        return Layout::from_ffi(ptr(n, rhs_ncols, par.ffi()));
                    }

                    template<typename T>
                    void in_place(
                        Mat<T const> L,
                        Vec<T const> D,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(ldlt_solve_in_place);
                        ptr(L.ffi(),
                            D.ffi(),
                            conj_A.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<T>(L.nrows, rhs.ncols, par)
                            ));
                    }
                } // namespace solve
            } // namespace ldlt

            namespace lblt {
                namespace factor {
                    using Params = FaerV0_23_LbltParams;

                    template<typename T>
                    auto params() -> Params {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(LbltParams);
                        return ptr();
                    }

                    template<typename I, typename T>
                    auto in_place_scratch(
                        size_t n,
                        Par par = get_global_par(),
                        Params params = factor::params<T>()
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            lblt_factor_in_place_scratch
                        );
                        return Layout::from_ffi(ptr(n, par.ffi(), params));
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T> A,
                        Vec<T> subdiag,
                        Slice<I> perm_fwd,
                        Slice<I> perm_bwd,
                        Par par = get_global_par(),
                        Mem mem = {},
                        Params params = factor::params<T>()
                    ) {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(lblt_factor_in_place);
                        ptr(A.ffi(),
                            subdiag.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(A.nrows, par, params)
                            ),
                            params);
                    }
                } // namespace factor

                namespace solve {
                    template<typename I, typename T>
                    auto in_place_scratch(size_t n, size_t rhs_ncols, Par par)
                        -> Layout {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(lblt_solve_in_place_scratch);
                        return Layout::from_ffi(ptr(n, rhs_ncols, par.ffi()));
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T const> L,
                        Vec<T const> diag,
                        Vec<T const> subdiag,
                        Slice<I const> perm_fwd,
                        Slice<I const> perm_bwd,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(lblt_solve_in_place);
                        ptr(L.ffi(),
                            diag.ffi(),
                            subdiag.ffi(),
                            conj_A.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(L.nrows, rhs.ncols, par)
                            ));
                    }
                } // namespace solve
            } // namespace lblt
        } // namespace cholesky

        namespace qr {
            namespace no_pivoting {
                namespace factor {
                    using Params = FaerV0_23_QrParams;

                    template<typename T>
                    auto params() -> Params {
                        constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(QrParams);
                        return ptr();
                    }

                    template<typename T>
                    auto in_place_scratch(
                        size_t nrows,
                        size_t ncols,
                        size_t block_size,
                        Par par = get_global_par(),
                        Params params = factor::params<T>()
                    ) -> Layout {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(qr_factor_in_place_scratch);
                        return Layout::from_ffi(
                            ptr(nrows, ncols, block_size, par.ffi(), params)
                        );
                    }

                    template<typename T>
                    void in_place(
                        Mat<T> A,
                        Mat<T> Q_coeff,
                        Par par = get_global_par(),
                        Mem mem = {},
                        Params params = factor::params<T>()
                    ) {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(qr_factor_in_place);
                        ptr(A.ffi(),
                            Q_coeff.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<T>(
                                    A.nrows,
                                    A.ncols,
                                    Q_coeff.nrows,
                                    par,
                                    params
                                )
                            ),
                            params);
                    }
                } // namespace factor

                namespace solve {
                    template<typename T>
                    auto in_place_scratch(
                        size_t n,
                        size_t block_size,
                        size_t rhs_ncols,
                        Par par
                    ) -> Layout {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(qr_solve_in_place_scratch);
                        return Layout::from_ffi(
                            ptr(n, block_size, rhs_ncols, par.ffi())
                        );
                    }

                    template<typename T>
                    void in_place(
                        Mat<T const> Q_basis,
                        Mat<T const> Q_coeff,
                        Mat<T const> R,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(qr_solve_in_place);
                        ptr(Q_basis.ffi(),
                            Q_coeff.ffi(),
                            R.ffi(),
                            conj_A.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<T>(
                                    Q_basis.nrows,
                                    Q_coeff.nrows,
                                    rhs.ncols,
                                    par
                                )
                            ));
                    }

                } // namespace solve

                namespace lstsq {
                    template<typename T>
                    auto in_place_scratch(
                        size_t nrows,
                        size_t ncols,
                        size_t block_size,
                        size_t rhs_ncols,
                        Par par
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(
                            qr_solve_lstsq_in_place_scratch
                        );
                        return Layout::from_ffi(
                            ptr(nrows, ncols, block_size, rhs_ncols, par.ffi())
                        );
                    }

                    template<typename T>
                    void in_place(
                        Mat<T const> Q_basis,
                        Mat<T const> Q_coeff,
                        Mat<T const> R,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(qr_solve_lstsq_in_place);
                        ptr(Q_basis.ffi(),
                            Q_coeff.ffi(),
                            R.ffi(),
                            conj_A.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<T>(
                                    Q_basis.nrows,
                                    (std::max)(Q_basis.ncols, R.ncols),
                                    Q_coeff.nrows,
                                    rhs.ncols,
                                    par
                                )
                            ));
                    }
                } // namespace lstsq
            } // namespace no_pivoting

            namespace col_pivoting {
                namespace factor {
                    using Params = FaerV0_23_ColPivQrParams;

                    template<typename T>
                    auto params() -> Params {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(ColPivQrParams);
                        return ptr();
                    }

                    template<typename I, typename T>
                    auto in_place_scratch(
                        size_t nrows,
                        size_t ncols,
                        size_t block_size,
                        Par par = get_global_par(),
                        Params params = factor::params<T>()
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            colpiv_qr_factor_in_place_scratch
                        );
                        return Layout::from_ffi(
                            ptr(nrows, ncols, block_size, par.ffi(), params)
                        );
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T> A,
                        Mat<T> Q_coeff,
                        Slice<I> perm_fwd,
                        Slice<I> perm_bwd,
                        Par par = get_global_par(),
                        Mem mem = {},
                        Params params = factor::params<T>()
                    ) {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(colpiv_qr_factor_in_place);
                        ptr(A.ffi(),
                            Q_coeff.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(
                                    A.nrows,
                                    A.ncols,
                                    Q_coeff.nrows,
                                    par,
                                    params
                                )
                            ),
                            params);
                    }
                } // namespace factor

                namespace solve {
                    template<typename I, typename T>
                    auto in_place_scratch(
                        size_t n,
                        size_t block_size,
                        size_t rhs_ncols,
                        Par par
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            colpiv_qr_solve_in_place_scratch
                        );
                        return Layout::from_ffi(
                            ptr(n, block_size, rhs_ncols, par.ffi())
                        );
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T const> Q_basis,
                        Mat<T const> Q_coeff,
                        Mat<T const> R,
                        Slice<I const> perm_fwd,
                        Slice<I const> perm_bwd,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(colpiv_qr_solve_in_place);
                        ptr(Q_basis.ffi(),
                            Q_coeff.ffi(),
                            R.ffi(),
                            conj_A.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(
                                    Q_basis.nrows,
                                    Q_coeff.nrows,
                                    rhs.ncols,
                                    par
                                )
                            ));
                    }

                } // namespace solve

                namespace lstsq {
                    template<typename I, typename T>
                    auto in_place_scratch(
                        size_t nrows,
                        size_t ncols,
                        size_t block_size,
                        size_t rhs_ncols,
                        Par par
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            colpiv_qr_solve_lstsq_in_place_scratch
                        );
                        return Layout::from_ffi(
                            ptr(nrows, ncols, block_size, rhs_ncols, par.ffi())
                        );
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T const> Q_basis,
                        Mat<T const> Q_coeff,
                        Mat<T const> R,
                        Slice<I const> perm_fwd,
                        Slice<I const> perm_bwd,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            colpiv_qr_solve_lstsq_in_place
                        );
                        ptr(Q_basis.ffi(),
                            Q_coeff.ffi(),
                            R.ffi(),
                            conj_A.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(
                                    Q_basis.nrows,
                                    (std::max)(Q_basis.ncols, R.ncols),
                                    Q_coeff.nrows,
                                    rhs.ncols,
                                    par
                                )
                            ));
                    }
                } // namespace lstsq
            } // namespace col_pivoting
        } // namespace qr

        namespace lu {
            namespace partial_pivoting {
                namespace factor {
                    using Params = FaerV0_23_PartialPivLuParams;

                    template<typename T>
                    auto params() -> Params {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(PartialPivLuParams);
                        return ptr();
                    }

                    template<typename I, typename T>
                    auto in_place_scratch(
                        size_t nrows,
                        size_t ncols,
                        Par par = get_global_par(),
                        Params params = factor::params<T>()
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            partial_piv_lu_factor_in_place_scratch
                        );
                        return Layout::from_ffi(
                            ptr(nrows, ncols, par.ffi(), params)
                        );
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T> A,
                        Slice<I> perm_fwd,
                        Slice<I> perm_bwd,
                        Par par = get_global_par(),
                        Mem mem = {},
                        Params params = factor::params<T>()
                    ) {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            partial_piv_lu_factor_in_place
                        );
                        ptr(A.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(
                                    A.nrows,
                                    A.ncols,
                                    par,
                                    params
                                )
                            ),
                            params);
                    }
                } // namespace factor

                namespace solve {
                    template<typename I, typename T>
                    auto in_place_scratch(size_t n, size_t rhs_ncols, Par par)
                        -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            partial_piv_lu_solve_in_place_scratch
                        );
                        return Layout::from_ffi(ptr(n, rhs_ncols, par.ffi()));
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T const> L,
                        Mat<T const> U,
                        Slice<I const> perm_fwd,
                        Slice<I const> perm_bwd,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            partial_piv_lu_solve_in_place
                        );
                        ptr(L.ffi(),
                            U.ffi(),
                            conj_A.ffi(),
                            perm_fwd.ffi(),
                            perm_bwd.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(L.nrows, rhs.ncols, par)
                            ));
                    }
                } // namespace solve
            } // namespace partial_pivoting

            namespace full_pivoting {
                namespace factor {
                    using Params = FaerV0_23_FullPivLuParams;

                    template<typename T>
                    auto params() -> Params {
                        constexpr auto ptr =
                            LIBFAER_SCALAR_DISPATCH(FullPivLuParams);
                        return ptr();
                    }

                    template<typename I, typename T>
                    auto in_place_scratch(
                        size_t nrows,
                        size_t ncols,
                        Par par = get_global_par(),
                        Params params = factor::params<T>()
                    ) -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            full_piv_lu_factor_in_place_scratch
                        );
                        return Layout::from_ffi(
                            ptr(nrows, ncols, par.ffi(), params)
                        );
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T> A,
                        Slice<I> row_perm_fwd,
                        Slice<I> row_perm_bwd,
                        Slice<I> col_perm_fwd,
                        Slice<I> col_perm_bwd,
                        Par par = get_global_par(),
                        Mem mem = {},
                        Params params = factor::params<T>()
                    ) {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(full_piv_lu_factor_in_place);
                        ptr(A.ffi(),
                            row_perm_fwd.ffi(),
                            row_perm_bwd.ffi(),
                            col_perm_fwd.ffi(),
                            col_perm_bwd.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(
                                    A.nrows,
                                    A.ncols,
                                    par,
                                    params
                                )
                            ),
                            params);
                    }
                } // namespace factor

                namespace solve {
                    template<typename I, typename T>
                    auto in_place_scratch(size_t n, size_t rhs_ncols, Par par)
                        -> Layout {
                        constexpr auto ptr = LIBFAER_INDEX_DISPATCH(
                            full_piv_lu_solve_in_place_scratch
                        );
                        return Layout::from_ffi(ptr(n, rhs_ncols, par.ffi()));
                    }

                    template<typename I, typename T>
                    void in_place(
                        Mat<T const> L,
                        Mat<T const> U,
                        Slice<I const> row_perm_fwd,
                        Slice<I const> row_perm_bwd,
                        Slice<I const> col_perm_fwd,
                        Slice<I const> col_perm_bwd,
                        Conj conj_A,
                        Mat<T> rhs,
                        Par par = get_global_par(),
                        Mem mem = {}
                    ) {
                        constexpr auto ptr =
                            LIBFAER_INDEX_DISPATCH(full_piv_lu_solve_in_place);
                        ptr(L.ffi(),
                            U.ffi(),
                            conj_A.ffi(),
                            row_perm_fwd.ffi(),
                            row_perm_bwd.ffi(),
                            col_perm_fwd.ffi(),
                            col_perm_bwd.ffi(),
                            rhs.ffi(),
                            par.ffi(),
                            LIBFAER_ALLOC(
                                in_place_scratch<I, T>(L.nrows, rhs.ncols, par)
                            ));
                    }
                } // namespace solve
            } // namespace full_pivoting
        } // namespace lu

        namespace svd {
            using Params = FaerV0_23_SvdParams;

            template<typename T>
            auto params() -> Params {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(SvdParams);
                return ptr();
            }

            template<typename T>
            auto scratch(
                size_t nrows,
                size_t ncols,
                ComputeSvdVectors U,
                ComputeSvdVectors V,
                Par par = get_global_par(),
                Params params = svd::params<T>()
            ) -> Layout {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(svd_scratch);
                return Layout::from_ffi(
                    ptr(nrows, ncols, U.ffi(), V.ffi(), par.ffi(), params)
                );
            }

            template<typename T>
            void compute(
                Mat<T const> A,
                Mat<T> U,
                Vec<T> S,
                Mat<T> V,
                Par par = get_global_par(),
                Mem mem = {},
                Params params = svd::params<T>()
            ) {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(svd);
                ptr(A.ffi(),
                    U.ffi(),
                    S.ffi(),
                    V.ffi(),
                    par.ffi(),
                    LIBFAER_ALLOC(
                        scratch<T>(
                            A.nrows,
                            A.ncols,
                            {
                                U.ncols == 0 ? ComputeSvdVectors::Thin
                                    : U.ncols == U.nrows
                                    ? ComputeSvdVectors::Full
                                    : ComputeSvdVectors::Thin,
                            },
                            {
                                V.ncols == 0 ? ComputeSvdVectors::Thin
                                    : V.ncols == V.nrows
                                    ? ComputeSvdVectors::Full
                                    : ComputeSvdVectors::Thin,
                            },
                            par,
                            params
                        )
                    ),
                    params);
            }
        } // namespace svd

        namespace self_adjoint_evd {
            using Params = FaerV0_23_SelfAdjointEvdParams;

            template<typename T>
            auto params() -> Params {
                constexpr auto ptr =
                    LIBFAER_SCALAR_DISPATCH(SelfAdjointEvdParams);
                return ptr();
            }

            template<typename T>
            auto scratch(
                size_t n,
                ComputeEigenvectors U,
                Par par = get_global_par(),
                Params params = self_adjoint_evd::params<T>()
            ) -> Layout {
                constexpr auto ptr =
                    LIBFAER_SCALAR_DISPATCH(self_adjoint_evd_scratch);
                return Layout::from_ffi(ptr(n, U.ffi(), par.ffi(), params));
            }

            template<typename T>
            void compute(
                Mat<T const> A,
                Mat<T> U,
                Vec<T> S,
                Vec<T> S_im,
                Par par = get_global_par(),
                Mem mem = {},
                Params params = self_adjoint_evd::params<T>()
            ) {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(self_adjoint_evd);
                ptr(A.ffi(),
                    U.ffi(),
                    S.ffi(),
                    S_im.ffi(),
                    par.ffi(),
                    LIBFAER_ALLOC(
                        scratch<T>(
                            A.nrows,
                            {
                                U.ncols == 0 ? ComputeEigenvectors::No
                                             : ComputeEigenvectors::Yes,
                            },
                            par,
                            params
                        )
                    ),
                    params);
            }
        } // namespace self_adjoint_evd

        namespace evd {
            using Params = FaerV0_23_EvdParams;

            template<typename T>
            auto params() -> Params {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(EvdParams);
                return ptr();
            }

            template<typename T>
            auto scratch(
                size_t n,
                ComputeEigenvectors UL,
                ComputeEigenvectors UR,
                Par par = get_global_par(),
                Params params = evd::params<T>()
            ) -> Layout {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(evd_scratch);
                return Layout::from_ffi(
                    ptr(n, UL.ffi(), UR.ffi(), par.ffi(), params)
                );
            }

            template<typename T>
            void compute(
                Mat<T const> A,
                Mat<T> UL,
                Mat<T> UR,
                Vec<T> S,
                Vec<T> S_im,
                Par par = get_global_par(),
                Mem mem = {},
                Params params = evd::params<T>()
            ) {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(evd);
                ptr(A.ffi(),
                    UL.ffi(),
                    UR.ffi(),
                    S.ffi(),
                    S_im.ffi(),
                    par.ffi(),
                    LIBFAER_ALLOC(
                        scratch<T>(
                            A.nrows,
                            {
                                UL.ncols == 0 ? ComputeEigenvectors::No
                                              : ComputeEigenvectors::Yes,
                            },
                            {
                                UR.ncols == 0 ? ComputeEigenvectors::No
                                              : ComputeEigenvectors::Yes,
                            },
                            par,
                            params
                        )
                    ),
                    params);
            }
        } // namespace evd

        namespace generalized_evd {
            using Params = FaerV0_23_GevdParams;

            template<typename T>
            auto params() -> Params {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(GevdParams);
                return ptr();
            }

            template<typename T>
            auto scratch(
                size_t n,
                ComputeEigenvectors UL,
                ComputeEigenvectors UR,
                Par par = get_global_par(),
                Params params = generalized_evd::params<T>()
            ) -> Layout {
                constexpr auto ptr =
                    LIBFAER_SCALAR_DISPATCH(generalized_evd_scratch);
                return Layout::from_ffi(
                    ptr(n, UL.ffi(), UR.ffi(), par.ffi(), params)
                );
            }

            template<typename T>
            void compute(
                Mat<T> A,
                Mat<T> B,
                Mat<T> UL,
                Mat<T> UR,
                Vec<T> alpha,
                Vec<T> alpha_im,
                Vec<T> beta,
                Par par = get_global_par(),
                Mem mem = {},
                Params params = generalized_evd::params<T>()
            ) {
                constexpr auto ptr = LIBFAER_SCALAR_DISPATCH(generalized_evd);
                ptr(A.ffi(),
                    B.ffi(),
                    UL.ffi(),
                    UR.ffi(),
                    alpha.ffi(),
                    alpha_im.ffi(),
                    beta.ffi(),
                    par.ffi(),
                    LIBFAER_ALLOC(
                        scratch<T>(
                            A.nrows,
                            {
                                UL.ncols == 0 ? ComputeEigenvectors::No
                                              : ComputeEigenvectors::Yes,
                            },
                            {
                                UR.ncols == 0 ? ComputeEigenvectors::No
                                              : ComputeEigenvectors::Yes,
                            },
                            par,
                            params
                        )
                    ),
                    params);
            }
        } // namespace generalized_evd
    } // namespace linalg

    namespace detail {
        inline void check_well_formed() {
            using I = uint32_t;
            using T = float;
            using std::ignore;

            using namespace linalg;
            {
                using namespace cholesky;
                ignore = llt::factor::in_place<T>;
                ignore = llt::solve::in_place<T>;

                ignore = llt_pivoting::factor::in_place<I, T>;
                ignore = llt_pivoting::solve::in_place<I, T>;

                ignore = ldlt::factor::in_place<T>;
                ignore = ldlt::solve::in_place<T>;
                ignore = lblt::factor::in_place<I, T>;
                ignore = lblt::solve::in_place<I, T>;
            }

            {
                using namespace qr;

                ignore = no_pivoting::factor::in_place<T>;
                ignore = no_pivoting::solve::in_place<T>;
                ignore = no_pivoting::lstsq::in_place<T>;
                ignore = col_pivoting::factor::in_place<I, T>;
                ignore = col_pivoting::solve::in_place<I, T>;
                ignore = col_pivoting::lstsq::in_place<I, T>;
            }

            {
                using namespace lu;

                ignore = partial_pivoting::factor::in_place<I, T>;
                ignore = partial_pivoting::solve::in_place<I, T>;
                ignore = full_pivoting::factor::in_place<I, T>;
                ignore = full_pivoting::solve::in_place<I, T>;
            }

            {
                ignore = svd::compute<T>;
                ignore = evd::compute<T>;
                ignore = generalized_evd::compute<T>;
            }
        }
    } // namespace detail
} // namespace v0_23
} // namespace faer

#undef LIBFAER_CAST_REAL
#undef LIBFAER_CAST_SCALAR
#undef LIBFAER_ALLOC
#undef LIBFAER_SCALAR_DISPATCH
#undef LIBFAER_INDEX_DISPATCH
