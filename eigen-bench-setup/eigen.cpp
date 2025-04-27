#include <eigen3/Eigen/Dense>

using usize = std::uintptr_t;
using f32 = float;
using f64 = double;
using c32 = std::complex<f32>;
using c64 = std::complex<f64>;

namespace double_f64 {
struct DoubleF64 {
  f64 x0;
  f64 x1;

  DoubleF64() = default;
  inline DoubleF64(double x) : x0{x}, x1{} {}
  inline DoubleF64(double x, double y) : x0{x}, x1{y} {}
};

/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
inline double quick_two_sum(double a, double b, double &err) {
  double s = a + b;
  err = b - (s - a);
  return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
inline double quick_two_diff(double a, double b, double &err) {
  double s = a - b;
  err = (a - s) - b;
  return s;
}

/* Computes fl(a+b) and err(a+b).  */
inline double two_sum(double a, double b, double &err) {
  double s = a + b;
  double bb = s - a;
  err = (a - (s - bb)) + (b - bb);
  return s;
}

/* Computes fl(a-b) and err(a-b).  */
inline double two_diff(double a, double b, double &err) {
  double s = a - b;
  double bb = s - a;
  err = (a - (s - bb)) - (b + bb);
  return s;
}

/* Computes fl(a*b) and err(a*b). */
inline double two_prod(double a, double b, double &err) {
  double p = a * b;
  err = std::fma(a, b, -p);
  return p;
}

inline DoubleF64 operator+(const DoubleF64 &a, const DoubleF64 &b) {
  double s, e;

  s = two_sum(a.x0, b.x0, e);
  e += (a.x1 + b.x1);
  s = quick_two_sum(s, e, e);
  return DoubleF64{s, e};
}

inline DoubleF64 operator-(const DoubleF64 &a) {
  return DoubleF64{-a.x0, -a.x1};
}

inline DoubleF64 operator-(const DoubleF64 &a, const DoubleF64 &b) {
  double s1, s2, t1, t2;
  s1 = two_diff(a.x0, b.x0, s2);
  t1 = two_diff(a.x1, b.x1, t2);
  s2 += t1;
  s1 = quick_two_sum(s1, s2, s2);
  s2 += t2;
  s1 = quick_two_sum(s1, s2, s2);
  return DoubleF64{s1, s2};
}

inline DoubleF64 operator*(const DoubleF64 &a, const DoubleF64 &b) {
  double p1, p2;

  p1 = two_prod(a.x0, b.x0, p2);
  p2 += (a.x0 * b.x1 + a.x1 * b.x0);
  p1 = quick_two_sum(p1, p2, p2);
  return DoubleF64(p1, p2);
}

inline DoubleF64 operator/(const DoubleF64 &a, const DoubleF64 &b) {
  double s1, s2;
  double q1, q2;
  DoubleF64 r;

  q1 = a.x0 / b.x0; /* approximate quotient */

  /* compute  this - q1 * dd */
  r = b * q1;
  s1 = two_diff(a.x0, r.x0, s2);
  s2 -= r.x1;
  s2 += a.x1;

  /* get next approximation */
  q2 = (s1 + s2) / b.x0;

  /* renormalize */
  r.x0 = quick_two_sum(q1, q2, r.x1);
  return r;
}

inline DoubleF64 &operator+=(DoubleF64 &a, const DoubleF64 &b) {
  a = a + b;
  return a;
}
inline DoubleF64 &operator-=(DoubleF64 &a, const DoubleF64 &b) {
  a = a - b;
  return a;
}
inline DoubleF64 &operator*=(DoubleF64 &a, const DoubleF64 &b) {
  a = a * b;
  return a;
}
inline DoubleF64 &operator/=(DoubleF64 &a, const DoubleF64 &b) {
  a = a / b;
  return a;
}

inline DoubleF64 sqrt(DoubleF64 const &a) {
  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto infty = std::numeric_limits<double>::infinity();
  if (a.x0 == 0.0) {
    return DoubleF64{};
  } else if (a.x0 < 0.0) {
    return DoubleF64{nan, nan};
  } else if (a.x0 == infty) {
    return DoubleF64{infty, infty};
  } else {
    auto x = 1.0 / std::sqrt(a.x0);
    auto ax = DoubleF64{a.x0 * x};
    return ax + (a - ax * ax) * DoubleF64{x * 0.5};
  }
}

inline DoubleF64 fabs(DoubleF64 const &a) { return a.x0 < 0.0 ? -a : a; }
inline DoubleF64 abs(DoubleF64 const &a) { return a.x0 < 0.0 ? -a : a; }

inline bool isfinite(DoubleF64 const &a) { return std::isfinite(a.x0); }
inline bool isinf(DoubleF64 const &a) { return std::isinf(a.x0); }
inline bool isnan(DoubleF64 const &a) {
  return std::isnan(a.x0) || std::isnan(a.x1);
}

inline bool operator==(const DoubleF64 &a, const DoubleF64 &b) {
  return a.x0 == b.x0 && a.x1 == b.x1;
}
inline bool operator!=(const DoubleF64 &a, const DoubleF64 &b) {
  return !(a == b);
}
inline bool operator<(const DoubleF64 &a, const DoubleF64 &b) {
  return (a.x0 < b.x0) || (a.x0 == b.x0 && a.x1 < b.x1);
}
inline bool operator>(const DoubleF64 &a, const DoubleF64 &b) {
  return (a.x0 > b.x0) || (a.x0 == b.x0 && a.x1 > b.x1);
}
inline bool operator<=(const DoubleF64 &a, const DoubleF64 &b) {
  return (a.x0 < b.x0) || (a.x0 == b.x0 && a.x1 <= b.x1);
}
inline bool operator>=(const DoubleF64 &a, const DoubleF64 &b) {
  return (a.x0 > b.x0) || (a.x0 == b.x0 && a.x1 >= b.x1);
}
} // namespace double_f64

using fx128 = double_f64::DoubleF64;
using cx128 = std::complex<double_f64::DoubleF64>;

namespace std {
template <> struct numeric_limits<fx128> {
  static constexpr auto is_specialized = true;
  static constexpr auto is_signed = true;
  static constexpr auto is_integer = false;
  static constexpr auto is_exact = false;
  static constexpr auto has_infinity = true;
  static constexpr auto has_quiet_NaN = true;
  static constexpr auto has_signaling_NaN = true;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = std::round_to_nearest;
  static constexpr auto is_iec559 = false;
  static constexpr auto is_bounded = true;
  static constexpr auto is_modulo = false;
  static constexpr auto digits = 100;

  static fx128 epsilon() { return fx128{1e-30}; }
  static fx128 min() { return fx128{1e-200}; }
  static fx128 max() { return 1.0 / min(); }
  static fx128 quiet_NaN() {
    return fx128{
        std::numeric_limits<f64>::quiet_NaN(),
        std::numeric_limits<f64>::quiet_NaN(),
    };
  }
  static fx128 infinity() {
    return fx128{
        std::numeric_limits<f64>::infinity(),
        std::numeric_limits<f64>::infinity(),
    };
  }
};
} // namespace std

constexpr usize LLT = 0;
constexpr usize LDLT = 1;
constexpr usize PLU = 2;
constexpr usize FLU = 3;
constexpr usize QR = 4;
constexpr usize CQR = 5;
constexpr usize SVD = 6;
constexpr usize HEVD = 7;
constexpr usize EVD = 8;

constexpr usize F32 = 0;
constexpr usize F64 = 1;
constexpr usize FX128 = 2;
constexpr usize C32 = 3;
constexpr usize C64 = 4;
constexpr usize CX128 = 5;

template <typename T> using Mat = Eigen::Matrix<T, -1, -1>;
template <typename T> constexpr bool is_complex = false;
template <typename T> constexpr bool is_complex<std::complex<T>> = true;

template <typename T>
void *make_decomp_impl(usize decomp, usize nrows, usize ncols) {
  bool is_square = false;

  switch (decomp) {
  case LLT:
  case LDLT:
  case PLU:
  case HEVD:
  case EVD:
    is_square = true;
    break;
  default:
    is_square = false;
  }

  if (is_square && nrows != ncols) {
    std::terminate();
  }

  switch (decomp) {
  case LLT:
    return new Eigen::LLT<Mat<T>>(nrows);
  case LDLT:
    return new Eigen::LDLT<Mat<T>>(nrows);
  case PLU:
    return new Eigen::PartialPivLU<Mat<T>>(nrows);
  case FLU:
    return new Eigen::FullPivLU<Mat<T>>(nrows, ncols);
  case QR:
    return new Eigen::HouseholderQR<Mat<T>>(nrows, ncols);
  case CQR:
    return new Eigen::ColPivHouseholderQR<Mat<T>>(nrows, ncols);
  case SVD:
    return new Eigen::BDCSVD<Mat<T>>(nrows, ncols);
  case HEVD:
    return new Eigen::SelfAdjointEigenSolver<Mat<T>>(nrows);
  case EVD:
    if constexpr (is_complex<T>) {
      return new Eigen::ComplexEigenSolver<Mat<T>>(nrows);
    } else {
      return new Eigen::EigenSolver<Mat<T>>(nrows);
    }
  default:
    std::terminate();
  }
}

template <typename T>
void factorize_impl(usize decomp, void *ptr, void *data, usize nrows,
                    usize ncols, usize stride) {
  auto A = Eigen::Map<Mat<T> const, Eigen::Unaligned, Eigen::OuterStride<-1>>(
      (T const *)data, nrows, ncols, Eigen::OuterStride<-1>(stride));

  switch (decomp) {
  case LLT:
    ((Eigen::LLT<Mat<T>> *)ptr)->compute(A);
    return;
  case LDLT:
    ((Eigen::LDLT<Mat<T>> *)ptr)->compute(A);
    return;
  case PLU:
    ((Eigen::PartialPivLU<Mat<T>> *)ptr)->compute(A);
    return;
  case FLU:
    ((Eigen::FullPivLU<Mat<T>> *)ptr)->compute(A);
    return;
  case QR:
    ((Eigen::HouseholderQR<Mat<T>> *)ptr)->compute(A);
    return;
  case CQR:
    ((Eigen::ColPivHouseholderQR<Mat<T>> *)ptr)->compute(A);
    return;
  case SVD:
    ((Eigen::BDCSVD<Mat<T>> *)ptr)->compute(A);
    return;
  case HEVD:
    ((Eigen::SelfAdjointEigenSolver<Mat<T>> *)ptr)->compute(A);
    return;
  case EVD:
    if constexpr (is_complex<T>) {
      ((Eigen::ComplexEigenSolver<Mat<T>> *)ptr)->compute(A);
      return;
    } else {
      ((Eigen::EigenSolver<Mat<T>> *)ptr)->compute(A);
      return;
    }
  default:
    std::terminate();
  }
}

template <typename T> void free_impl(usize decomp, void *ptr) {
  switch (decomp) {
  case LLT:
    delete (Eigen::LLT<Mat<T>> *)ptr;
    return;
  case LDLT:
    delete (Eigen::LDLT<Mat<T>> *)ptr;
    return;
  case PLU:
    delete (Eigen::PartialPivLU<Mat<T>> *)ptr;
    return;
  case FLU:
    delete (Eigen::FullPivLU<Mat<T>> *)ptr;
    return;
  case QR:
    delete (Eigen::HouseholderQR<Mat<T>> *)ptr;
    return;
  case CQR:
    delete (Eigen::ColPivHouseholderQR<Mat<T>> *)ptr;
    return;
  case SVD:
    delete (Eigen::BDCSVD<Mat<T>> *)ptr;
    return;
  case HEVD:
    delete (Eigen::SelfAdjointEigenSolver<Mat<T>> *)ptr;
    return;
  case EVD:
    if constexpr (is_complex<T>) {
      delete (Eigen::ComplexEigenSolver<Mat<T>> *)ptr;
      return;
    } else {
      delete (Eigen::EigenSolver<Mat<T>> *)ptr;
      return;
    }
  default:
    std::terminate();
  }
}

extern "C" void *libeigen_make_decomp(usize decomp, usize dtype, usize nrows,
                                      usize ncols) {
  switch (dtype) {
  case F32:
    return make_decomp_impl<f32>(decomp, nrows, ncols);
  case F64:
    return make_decomp_impl<f64>(decomp, nrows, ncols);
  case FX128:
    return make_decomp_impl<fx128>(decomp, nrows, ncols);
  case C32:
    return make_decomp_impl<c32>(decomp, nrows, ncols);
  case C64:
    return make_decomp_impl<c64>(decomp, nrows, ncols);
  case CX128:
    return make_decomp_impl<cx128>(decomp, nrows, ncols);
  default:
    std::terminate();
  }
}

extern "C" void libeigen_factorize(usize decomp, usize dtype, void *ptr,
                                   void *data, usize nrows, usize ncols,
                                   usize stride) {
  switch (dtype) {
  case F32:
    return factorize_impl<f32>(decomp, ptr, data, nrows, ncols, stride);
  case F64:
    return factorize_impl<f64>(decomp, ptr, data, nrows, ncols, stride);
  case FX128:
    return factorize_impl<fx128>(decomp, ptr, data, nrows, ncols, stride);
  case C32:
    return factorize_impl<c32>(decomp, ptr, data, nrows, ncols, stride);
  case C64:
    return factorize_impl<c64>(decomp, ptr, data, nrows, ncols, stride);
  case CX128:
    return factorize_impl<cx128>(decomp, ptr, data, nrows, ncols, stride);
  default:
    std::terminate();
  }
}

extern "C" void libeigen_free_decomp(usize decomp, usize dtype, void *ptr) {
  switch (dtype) {
  case F32:
    return free_impl<f32>(decomp, ptr);
  case F64:
    return free_impl<f64>(decomp, ptr);
  case FX128:
    return free_impl<fx128>(decomp, ptr);
  case C32:
    return free_impl<c32>(decomp, ptr);
  case C64:
    return free_impl<c64>(decomp, ptr);
  case CX128:
    return free_impl<cx128>(decomp, ptr);
  default:
    std::terminate();
  }
}
