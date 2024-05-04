#include <diol.hpp>
#include <eigen3/Eigen/Core>

#include <mkl/mkl_lapack.h>

#define CAT_(A, B) A##B
#define CAT(A, B) CAT_(A, B)
#define MAKE_LAPACK(prefix, func)                                              \
  inline static constexpr auto func = CAT(CAT(prefix, func), _);

using namespace diol;
template <typename T> T const *addr(T const &ptr) { return &ptr; }
template <typename T> T *addr(T &&ptr) { return &ptr; }
MKL_INT max(MKL_INT a, MKL_INT b) { return a > b ? a : b; }

using f32 = float;
using f64 = double;
using c32 = std::complex<f32>;
using c64 = std::complex<f64>;

template <typename T> using Mat = Eigen::Matrix<T, -1, -1>;

#define MAKE_MKL(ty, mkl_ty, prefix, sym)                                      \
  template <> struct Mkl<ty> {                                                 \
    using T = mkl_ty;                                                          \
    MAKE_LAPACK(prefix, potrf);                                                \
    MAKE_LAPACK(prefix, geqrf);                                                \
    MAKE_LAPACK(prefix, geqp3);                                                \
    MAKE_LAPACK(prefix, getrf);                                                \
    MAKE_LAPACK(prefix, getc2);                                                \
    MAKE_LAPACK(prefix, gesdd);                                                \
    MAKE_LAPACK(CAT(prefix, sym), evd);                                        \
    MAKE_LAPACK(prefix, geev);                                                 \
  };

template <typename E> struct Mkl;

MAKE_MKL(f32, f32, s, sy);
MAKE_MKL(f64, f64, d, sy);
MAKE_MKL(c32, MKL_Complex8, c, he);
MAKE_MKL(c64, MKL_Complex16, z, he);

template <typename T> using Data = typename Mkl<T>::T *;

template <typename T> MKL_INT stride(Mat<T> const &mat) {
  return mat.outerStride();
}
template <typename T> auto data(Mat<T> &mat) { return Data<T>(mat.data()); }

template <typename T> Mat<T> rand(MKL_INT m, MKL_INT n) {
  Mat<T> A(m, n);
  A.setRandom();
  return A;
}

template <typename T> Mat<T> rand_pos_def(MKL_INT n) {
  Mat<T> A(n, n);
  A.setRandom();
  Mat<T> H = T(0.5) * (A + A.adjoint());
  H += T(n) * Mat<T>::Identity(n, n);
  return H;
}

template <typename T> Mat<T> randh(MKL_INT n) {
  Mat<T> A(n, n);
  A.setRandom();
  Mat<T> H = T(0.5) * (A + A.adjoint());
  return H;
}

template <typename T> void cholesky(Bencher bencher, PlotArg arg) {
  MKL_INT n = arg.n;
  std::srand(0);
  Mat<T> H_orig = rand_pos_def<T>(n);
  Mat<T> H = H_orig;

  std::move(bencher).bench([&] {
    H = H_orig;

    Mkl<T>::potrf("L", addr(n), data(H), addr(stride(H)), addr((MKL_INT)0));
  });
}

template <typename T> void qr(Bencher bencher, PlotArg arg) {
  MKL_INT n = arg.n;
  std::srand(0);
  Mat<T> H_orig = rand<T>(n, n);
  Mat<T> H = H_orig;
  Mat<T> tau(n, 1);

  T lwork_;
  Mkl<T>::geqrf(addr(n), addr(n), data(H), addr(stride(H)), data(tau),
                Data<T>(&lwork_), addr((MKL_INT)-1), addr((MKL_INT)0));
  MKL_INT lwork = 2 * (MKL_INT)std::real(lwork_);
  Mat<T> work(lwork, 1);

  std::move(bencher).bench([&] {
    H = H_orig;

    Mkl<T>::geqrf(addr(n), addr(n), data(H), addr(stride(H)), data(tau),
                  data(work), addr(lwork), addr((MKL_INT)0));
  });
}

template <typename T> void piv_qr(Bencher bencher, PlotArg arg) {
  MKL_INT n = arg.n;
  std::srand(0);
  Mat<T> H_orig = rand<T>(n, n);
  Mat<T> H = H_orig;
  Mat<T> tau(n, 1);
  Mat<MKL_INT> p(n, 1);
  Mat<typename Mat<T>::RealScalar> rwork(2 * n, 1);

  T lwork_;
  if constexpr (std::same_as<T, typename Mkl<T>::T>) {
    Mkl<T>::geqp3(addr(n), addr(n), data(H), addr(stride(H)), p.data(),
                  data(tau), Data<T>(&lwork_), addr((MKL_INT)-1),
                  addr((MKL_INT)0));
  } else {
    Mkl<T>::geqp3(addr(n), addr(n), data(H), addr(stride(H)), p.data(),
                  data(tau), Data<T>(&lwork_), addr((MKL_INT)-1), data(rwork),
                  addr((MKL_INT)0));
  }
  auto lwork = 2 * (MKL_INT)std::real(lwork_);
  Mat<T> work(lwork, 1);

  std::move(bencher).bench([&] {
    H = H_orig;

    if constexpr (std::same_as<T, typename Mkl<T>::T>) {
      Mkl<T>::geqp3(addr(n), addr(n), data(H), addr(stride(H)), p.data(),
                    data(tau), data(work), addr(lwork), addr((MKL_INT)0));
    } else {
      Mkl<T>::geqp3(addr(n), addr(n), data(H), addr(stride(H)), p.data(),
                    data(tau), data(work), addr(lwork), data(rwork),
                    addr((MKL_INT)0));
    }
  });
}

template <typename T> void lu(Bencher bencher, PlotArg arg) {
  MKL_INT n = arg.n;
  std::srand(0);
  Mat<T> H_orig = rand<T>(n, n);
  Mat<T> H = H_orig;
  Mat<MKL_INT> p(n, 1);
  Mat<T> work(n, max(n, 16));

  std::move(bencher).bench([&] {
    H = H_orig;

    Mkl<T>::getrf(addr(n), addr(n), data(H), addr(stride(H)), p.data(),
                  addr((MKL_INT)0));
  });
}

template <typename T> void piv_lu(Bencher bencher, PlotArg arg) {
  MKL_INT n = arg.n;
  std::srand(0);
  Mat<T> H_orig = rand<T>(n, n);
  Mat<T> H = H_orig;
  Mat<MKL_INT> p(n, 1);
  Mat<MKL_INT> q(n, 1);
  Mat<T> work(n, max(n, 16));

  std::move(bencher).bench([&] {
    H = H_orig;

    Mkl<T>::getc2(addr(n), data(H), addr(stride(H)), p.data(), q.data(),
                  addr((MKL_INT)0));
  });
}
template <typename T> void svd(Bencher bencher, PlotArg arg) {
  MKL_INT m = arg.n;
  MKL_INT n = arg.n;
  MKL_INT mn = m * n;
  MKL_INT mx = max(m, n);

  std::srand(0);
  Mat<T> H_orig = rand<T>(m, n);
  Mat<T> H = H_orig;
  Mat<T> U(m, m);
  Mat<T> V(n, n);
  Mat<typename Mat<T>::RealScalar> S(m, 1);

  Mat<typename Mat<T>::RealScalar> rwork(
      max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn), 1);
  Mat<MKL_INT> iwork(8 * max(m, n), 1);

  T lwork_;
  if constexpr (std::same_as<T, typename Mkl<T>::T>) {
    Mkl<T>::gesdd("A", addr(m), addr(n), data(H), addr(stride(H)), data(S),
                  data(U), addr(stride(U)), data(V), addr(stride(V)),
                  Data<T>(&lwork_), addr((MKL_INT)-1), iwork.data(),
                  addr((MKL_INT)0));
  } else {
    Mkl<T>::gesdd("A", addr(m), addr(n), data(H), addr(stride(H)), data(S),
                  data(U), addr(stride(U)), data(V), addr(stride(V)),
                  Data<T>(&lwork_), addr((MKL_INT)-1), data(rwork),
                  iwork.data(), addr((MKL_INT)0));
  }

  auto lwork = 2 * (MKL_INT)std::real(lwork_);
  Mat<T> work(lwork, 1);

  std::move(bencher).bench([&] {
    H = H_orig;

    if constexpr (std::same_as<T, typename Mkl<T>::T>) {
      Mkl<T>::gesdd("A", addr(m), addr(n), data(H), addr(stride(H)), data(S),
                    data(U), addr(stride(U)), data(V), addr(stride(V)),
                    data(work), addr(lwork), iwork.data(), addr((MKL_INT)0));
    } else {
      Mkl<T>::gesdd("A", addr(m), addr(n), data(H), addr(stride(H)), data(S),
                    data(U), addr(stride(U)), data(V), addr(stride(V)),
                    data(work), addr(lwork), data(rwork), iwork.data(),
                    addr((MKL_INT)0));
    }
  });
}
template <typename T> void thin_svd(Bencher bencher, PlotArg arg) {
  MKL_INT m = 4096;
  MKL_INT n = arg.n;
  MKL_INT mn = m * n;
  MKL_INT mx = max(m, n);

  std::srand(0);
  Mat<T> H_orig = rand<T>(m, n);
  Mat<T> H = H_orig;
  Mat<T> U(m, m);
  Mat<T> V(n, n);
  Mat<typename Mat<T>::RealScalar> S(m, 1);

  Mat<typename Mat<T>::RealScalar> rwork(
      max(5 * mn * mn + 5 * mn, 2 * mx * mn + 2 * mn * mn + mn), 1);
  Mat<MKL_INT> iwork(8 * max(m, n), 1);

  T lwork_;
  if constexpr (std::same_as<T, typename Mkl<T>::T>) {
    Mkl<T>::gesdd("S", addr(m), addr(n), data(H), addr(stride(H)), data(S),
                  data(U), addr(stride(U)), data(V), addr(stride(V)),
                  Data<T>(&lwork_), addr((MKL_INT)-1), iwork.data(),
                  addr((MKL_INT)0));
  } else {
    Mkl<T>::gesdd("S", addr(m), addr(n), data(H), addr(stride(H)), data(S),
                  data(U), addr(stride(U)), data(V), addr(stride(V)),
                  Data<T>(&lwork_), addr((MKL_INT)-1), data(rwork),
                  iwork.data(), addr((MKL_INT)0));
  }

  auto lwork = 2 * (MKL_INT)std::real(lwork_);
  Mat<T> work(lwork, 1);

  std::move(bencher).bench([&] {
    H = H_orig;

    if constexpr (std::same_as<T, typename Mkl<T>::T>) {
      Mkl<T>::gesdd("S", addr(m), addr(n), data(H), addr(stride(H)), data(S),
                    data(U), addr(stride(U)), data(V), addr(stride(V)),
                    data(work), addr(lwork), iwork.data(), addr((MKL_INT)0));
    } else {
      Mkl<T>::gesdd("S", addr(m), addr(n), data(H), addr(stride(H)), data(S),
                    data(U), addr(stride(U)), data(V), addr(stride(V)),
                    data(work), addr(lwork), data(rwork), iwork.data(),
                    addr((MKL_INT)0));
    }
  });
}
template <typename T> void eigh(Bencher bencher, PlotArg arg) {
  MKL_INT n = arg.n;

  std::srand(0);
  Mat<T> H_orig = randh<T>(n);
  Mat<T> H = H_orig;
  Mat<T> U(n, n);
  Mat<typename Mat<T>::RealScalar> W(n, 1);

  T lwork_ = 0;
  typename Mat<T>::RealScalar lrwork_ = 0;
  MKL_INT liwork_ = 0;

  if constexpr (std::same_as<T, typename Mkl<T>::T>) {
    Mkl<T>::evd("V", "L", addr(n), data(H), addr(stride(H)), data(W),
                Data<T>(&lwork_), addr((MKL_INT)-1), &liwork_,
                addr((MKL_INT)-1), addr((MKL_INT)0));
  } else {
    Mkl<T>::evd("V", "L", addr(n), data(H), addr(stride(H)), data(W),
                Data<T>(&lwork_), addr((MKL_INT)-1), &lrwork_,
                addr((MKL_INT)-1), &liwork_, addr((MKL_INT)-1),
                addr((MKL_INT)0));
  }

  MKL_INT lwork = 2 * (MKL_INT)(std::real(lwork_));
  MKL_INT lrwork = 2 * (MKL_INT)(lrwork_);
  MKL_INT liwork = 2 * (MKL_INT)(liwork_);

  Mat<T> work(lwork, 1);
  Mat<typename Mat<T>::RealScalar> rwork(lrwork, 1);
  Mat<MKL_INT> iwork(liwork, 1);

  std::move(bencher).bench([&] {
    H = H_orig;

    if constexpr (std::same_as<T, typename Mkl<T>::T>) {
      Mkl<T>::evd("V", "L", addr(n), data(H), addr(stride(H)), data(W),
                  data(work), addr(lwork), iwork.data(), addr(liwork),
                  addr((MKL_INT)0));
    } else {
      Mkl<T>::evd("V", "L", addr(n), data(H), addr(stride(H)), data(W),
                  data(work), addr(lwork), data(rwork), addr(lrwork),
                  iwork.data(), addr(liwork), addr((MKL_INT)0));
    }
  });
}
template <typename T> void eig(Bencher bencher, PlotArg arg) {
  MKL_INT n = arg.n;

  std::srand(0);
  Mat<T> H_orig = rand<T>(n, n);
  Mat<T> H = H_orig;
  Mat<T> U(n, n);
  Mat<T> V(n, n);
  Mat<T> W(n, 1);
  Mat<typename Mat<T>::RealScalar> W_im(n, 1);
  Mat<typename Mat<T>::RealScalar> rwork(2 * n, 1);

  T lwork_;

  if constexpr (std::same_as<T, typename Mkl<T>::T>) {
    Mkl<T>::geev("V", "N", addr(n), data(H), addr(stride(H)), data(W),
                 data(W_im), data(U), addr(stride(U)), data(V), addr(stride(V)),
                 Data<T>(&lwork_), addr((MKL_INT)-1), addr((MKL_INT)0));
  } else {
    Mkl<T>::geev("V", "N", addr(n), data(H), addr(stride(H)), data(W), data(U),
                 addr(stride(U)), data(V), addr(stride(V)), Data<T>(&lwork_),
                 addr((MKL_INT)-1), data(rwork), addr((MKL_INT)0));
  }

  MKL_INT lwork = 2 * (MKL_INT)std::real(lwork_);

  Mat<T> work(lwork, 1);

  std::move(bencher).bench([&] {
    H = H_orig;

    if constexpr (std::same_as<T, typename Mkl<T>::T>) {
      Mkl<T>::geev("V", "N", addr(n), data(H), addr(stride(H)), data(W),
                   data(W_im), data(U), addr(stride(U)), data(V),
                   addr(stride(V)), data(work), addr(lwork), addr((MKL_INT)0));
    } else {
      Mkl<T>::geev("V", "N", addr(n), data(H), addr(stride(H)), data(W),
                   data(U), addr(stride(U)), data(V), addr(stride(V)),
                   data(work), addr(lwork), data(rwork), addr((MKL_INT)0));
    }
  });
}

std::string glue_name(std::string prefix, std::string name, std::string type) {
  return prefix + "_" + name + "<" + type + ">";
}
template <typename T>
void register_funcs(std::string prefix, std::string ty, Bench &bench) {
  PlotArg args[] = {
      4,   8,   12,   16,   24,   32,   48,   64,   128,
      256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096,
  };

  auto do_it = [&](std::string name, FnPtr<Bencher, PlotArg> fn) {
    bench.register_funcs<PlotArg>({{{glue_name(prefix, name, ty), fn}}}, args);
  };

  do_it("cholesky", cholesky<T>);
  do_it("qr", qr<T>);
  do_it("piv_qr", piv_qr<T>);
  do_it("lu", lu<T>);
  do_it("piv_lu", piv_lu<T>);
  do_it("svd", svd<T>);
  do_it("thin_svd", thin_svd<T>);
  do_it("eigh", eigh<T>);
  do_it("eig", eig<T>);
}

f64 flops_per_sec(size_t n, f64 time) {
  return f64(n) * f64(n) * f64(n) / time;
}

int main() {
  auto config = BenchConfig::from_args();
  config.set_metric("n³/s", Monotonicity::HigherIsBetter, flops_per_sec);
  auto bench = Bench::from_config(config);
#ifdef BENCH_MKL
  std::string prefix = "mkl";
#else
  std::string prefix = "openblas";
#endif

  register_funcs<f32>(prefix, "f32", bench);
  register_funcs<f64>(prefix, "f64", bench);
  register_funcs<c32>(prefix, "c32", bench);
  register_funcs<c64>(prefix, "c64", bench);
  bench.run();
}
