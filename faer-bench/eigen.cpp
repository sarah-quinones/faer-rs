#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/QR>
#include <iomanip>
#include <iostream>
#include <vector>

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

extern "C" void gemm(double *out, std::size_t const *inputs,
                     std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    auto b = Eigen::MatrixXd(n, n);
    auto c = Eigen::MatrixXd(n, n);
    a.setZero();
    b.setZero();
    c.setZero();

    out[i] = timeit([&] { c.noalias() += a * b; });
  }
}

extern "C" void trsm(double *out, std::size_t const *inputs,
                     std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    auto b = Eigen::MatrixXd(n, n);
    a.setZero();
    b.setZero();
    out[i] =
        timeit([&] { a.triangularView<Eigen::UnitLower>().solveInPlace(b); });
  }
}

extern "C" void trinv(double *out, std::size_t const *inputs,
                      std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    auto b = Eigen::MatrixXd(n, n);
    a.setZero();
    b.setZero();
    out[i] =
        timeit([&] { a.triangularView<Eigen::UnitLower>().solveInPlace(b); });
  }
}

extern "C" void chol(double *out, std::size_t const *inputs,
                     std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    a.setIdentity();
    auto b = a.llt();
    out[i] = timeit([&] { b.compute(a); });
  }
}

extern "C" void plu(double *out, std::size_t const *inputs, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a.partialPivLu();
    out[i] = timeit([&] { b.compute(a); });
  }
}

extern "C" void flu(double *out, std::size_t const *inputs, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a.fullPivLu();
    out[i] = timeit([&] { b.compute(a); });
  }
}

extern "C" void qr(double *out, std::size_t const *inputs, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a.householderQr();
    out[i] = timeit([&] { b.compute(a); });
  }
}

extern "C" void colqr(double *out, std::size_t const *inputs,
                      std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a.colPivHouseholderQr();
    out[i] = timeit([&] { b.compute(a); });
  }
}

extern "C" void inverse(double *out, std::size_t const *inputs,
                        std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    auto n = inputs[i];

    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a;
    out[i] = timeit([&] { b = a.inverse(); });
  }
}
