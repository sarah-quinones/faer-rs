#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>
#include <limits>

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

using f128 = double_f64::DoubleF64;
using c128 = std::complex<double_f64::DoubleF64>;

namespace std {
template <> struct numeric_limits<f128> {
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

  static f128 epsilon() { return f128{1e-30}; }
  static f128 min() { return f128{1e-200}; }
  static f128 max() { return 1.0 / min(); }
  static f128 quiet_NaN() {
    return f128{
        std::numeric_limits<f64>::quiet_NaN(),
        std::numeric_limits<f64>::quiet_NaN(),
    };
  }
  static f128 infinity() {
    return f128{
        std::numeric_limits<f64>::infinity(),
        std::numeric_limits<f64>::infinity(),
    };
  }
};
} // namespace std

template <typename F> auto time1(F f) -> double {
  auto start = std::chrono::steady_clock::now();
  f();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

template <typename F> auto timeit(F f) -> double {
  auto min = 1e-0;
  auto once = (time1)(f);
  if (once > min) {
    return once;
  }
  auto ten = (time1)([&] {
    for (std::size_t i = 0; i < 10; ++i) {
      f();
    }
  });
  if (ten > min) {
    return ten / 10.0;
  }

  unsigned long long n = std::ceil(min * 10 / ten);
  return (time1)([&] {
           for (std::size_t i = 0; i < n; ++i) {
             f();
           }
         }) /
         double(n);
}

#define INSTANTIATE(T)                                                         \
                                                                               \
  extern "C" void gemm_##T(double *out, std::size_t const *inputs,             \
                           std::size_t count) {                                \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      auto b = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      auto c = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setZero();                                                             \
      b.setZero();                                                             \
      c.setZero();                                                             \
                                                                               \
      out[i] = timeit([&] { c.noalias() += a * b; });                          \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void trsm_##T(double *out, std::size_t const *inputs,             \
                           std::size_t count) {                                \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      auto b = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setZero();                                                             \
      b.setZero();                                                             \
      out[i] = timeit(                                                         \
          [&] { a.triangularView<Eigen::UnitLower>().solveInPlace(b); });      \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void trinv_##T(double *out, std::size_t const *inputs,            \
                            std::size_t count) {                               \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      auto b = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setZero();                                                             \
      b.setZero();                                                             \
      out[i] = timeit(                                                         \
          [&] { a.triangularView<Eigen::UnitLower>().solveInPlace(b); });      \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void chol_##T(double *out, std::size_t const *inputs,             \
                           std::size_t count) {                                \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setIdentity();                                                         \
      auto b = a.llt();                                                        \
      out[i] = timeit([&] { b.compute(a); });                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void plu_##T(double *out, std::size_t const *inputs,              \
                          std::size_t count) {                                 \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setRandom();                                                           \
      auto b = a.partialPivLu();                                               \
      out[i] = timeit([&] { b.compute(a); });                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void flu_##T(double *out, std::size_t const *inputs,              \
                          std::size_t count) {                                 \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setRandom();                                                           \
      auto b = a.fullPivLu();                                                  \
      out[i] = timeit([&] { b.compute(a); });                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void qr_##T(double *out, std::size_t const *inputs,               \
                         std::size_t count) {                                  \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setRandom();                                                           \
      auto b = a.householderQr();                                              \
      out[i] = timeit([&] { b.compute(a); });                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void colqr_##T(double *out, std::size_t const *inputs,            \
                            std::size_t count) {                               \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setRandom();                                                           \
      auto b = a.colPivHouseholderQr();                                        \
      out[i] = timeit([&] { b.compute(a); });                                  \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void inverse_##T(double *out, std::size_t const *inputs,          \
                              std::size_t count) {                             \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setRandom();                                                           \
      auto b = a;                                                              \
      out[i] = timeit([&] { b = a.inverse(); });                               \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void svd_##T(double *out, std::size_t const *inputs,              \
                          std::size_t count) {                                 \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setRandom();                                                           \
      Eigen::BDCSVD<Eigen::Matrix<T, -1, -1>> svd(                             \
          n, n, Eigen::ComputeFullU | Eigen::ComputeFullV);                    \
      out[i] = timeit([&] { svd.compute(a); });                                \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void rectangular_svd_##T(double *out, std::size_t const *inputs,  \
                                      std::size_t count) {                     \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(4096, n);                              \
      a.setRandom();                                                           \
      Eigen::BDCSVD<Eigen::Matrix<T, -1, -1>> svd(                             \
          n, n, Eigen::ComputeThinU | Eigen::ComputeThinV);                    \
      out[i] = timeit([&] { svd.compute(a); });                                \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" void symmetric_evd_##T(double *out, std::size_t const *inputs,    \
                                    std::size_t count) {                       \
    for (std::size_t i = 0; i < count; ++i) {                                  \
      auto n = inputs[i];                                                      \
                                                                               \
      auto a = Eigen::Matrix<T, -1, -1>(n, n);                                 \
      a.setRandom();                                                           \
      a = (a + a.adjoint()).eval();                                            \
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, -1, -1>> evd(n);          \
      out[i] = timeit([&] { evd.compute(a, Eigen::ComputeEigenvectors); });    \
    }                                                                          \
  }                                                                            \
  static_assert(true, "")

INSTANTIATE(f32);
INSTANTIATE(f64);
INSTANTIATE(f128);
INSTANTIATE(c32);
INSTANTIATE(c64);
INSTANTIATE(c128);
