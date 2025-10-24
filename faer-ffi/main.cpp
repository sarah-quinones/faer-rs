#include <fmt/format.h>
#include <fmt/ostream.h>

#include <eigen3/Eigen/Core>

#include "faer.hpp"

int main() {
    using T = quad::f128;
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(10, 10);
    A.setRandom();
    A = A * A.transpose();

    using namespace faer::linalg::cholesky;

    Mat L = A;
    llt::factor::in_place(faer::from_eigen(L));
    L.triangularView<Eigen::StrictlyUpper>().setZero();

    fmt::print(
        "{}\n\n"
        "{}\n\n"
        "error norm: {}\n\n",
        fmt::streamed(A),
        fmt::streamed((L * L.transpose()).eval()),
        fmt::streamed((A - L * L.transpose()).norm())
    );
}
