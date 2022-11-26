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

void fmt(double time) {
  auto unit = " s";
  if (time < 1e-6) {
    time *= 1e9;
    unit = "ns";
  } else if (time < 1e-3) {
    time *= 1e6;
    unit = "µs";
  } else if (time < 1e-0) {
    time *= 1e3;
    unit = "ms";
  }

  std::cout << std::setprecision(3) << std::setw(10) << time << unit << '\n';
}

int main() {
  std::vector<Eigen::Index> inputs{32,  64,  96,  128, 192, 256,
                                   384, 512, 640, 768, 896, 1024};
  for (int i = 0; i < 10; ++i) {
    timeit([] {});
  }

  std::cout << "gemm" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    auto b = Eigen::MatrixXd(n, n);
    auto c = Eigen::MatrixXd(n, n);
    a.setZero();
    b.setZero();
    c.setZero();
    fmt(timeit([&] { c.noalias() += a * b; }));
  }

  std::cout << "trsm" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    auto b = Eigen::MatrixXd(n, n);
    a.setZero();
    b.setZero();
    fmt(timeit([&] { a.triangularView<Eigen::UnitLower>().solveInPlace(b); }));
  }

  std::cout << "triangular inverse" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    auto b = Eigen::MatrixXd(n, n);
    a.setZero();
    b.setZero();
    fmt(timeit([&] { a.triangularView<Eigen::UnitLower>().solveInPlace(b); }));
  }

  std::cout << "cholesky decomposition" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    a.setIdentity();
    auto b = a.llt();
    fmt(timeit([&] { b.compute(a); }));
  }

  std::cout << "lu partial piv" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a.partialPivLu();
    fmt(timeit([&] { b.compute(a); }));
  }

  std::cout << "lu full piv" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a.fullPivLu();
    fmt(timeit([&] { b.compute(a); }));
  }

  std::cout << "qr" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a.householderQr();
    fmt(timeit([&] { b.compute(a); }));
  }

  std::cout << "col piv qr" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a.colPivHouseholderQr();
    fmt(timeit([&] { b.compute(a); }));
  }

  std::cout << "inverse" << '\n';
  for (auto n : inputs) {
    auto a = Eigen::MatrixXd(n, n);
    a.setRandom();
    auto b = a;
    fmt(timeit([&] { b = a.inverse(); }));
  }
}
