#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>

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

using f32 = float;
using f64 = double;
using c32 = std::complex<f32>;
using c64 = std::complex<f64>;

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
  static_assert(true, "")

INSTANTIATE(f32);
INSTANTIATE(f64);
INSTANTIATE(c32);
INSTANTIATE(c64);
