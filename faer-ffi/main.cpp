#include <fmt/format.h>
#include <fmt/ostream.h>

#include <eigen3/Eigen/Core>
#include <iostream>
#include <type_traits>

#include "faer.hpp"

namespace faer {

template<typename T>
auto from_eigen(T&& mat) -> Mat<std::remove_pointer_t<decltype(mat.data())>> {
    return {
        .ptr = mat.data(),
        .nrows = static_cast<size_t>(mat.rows()),
        .ncols = static_cast<size_t>(mat.cols()),
        .row_stride = mat.IsRowMajor ? mat.outerStride() : mat.innerStride(),
        .col_stride = mat.IsRowMajor ? mat.innerStride() : mat.outerStride(),
    };
}
} // namespace faer

int main() {
    using T = quad::f128;
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    Mat A(10, 10);
    A.setRandom();
    std::cout << A << "\n\n";

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
