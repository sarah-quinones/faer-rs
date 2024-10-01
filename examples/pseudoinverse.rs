use faer::{assert_matrix_eq, mat, Mat};

fn main() {
    #[cfg(feature = "svd")]
    {
        let matrix = mat![
            [-0.29945184, -0.5228196],
            [0.84136125, -1.15768694],
            [1.25678304, -0.46203532]
        ];

        let m = matrix.nrows();
        let n = matrix.ncols();
        let svd = matrix.svd();

        let s_diag = svd.s_diagonal();
        let mut s_inv = Mat::zeros(n, m);
        for i in 0..Ord::min(m, n) {
            s_inv[(i, i)] = 1.0 / s_diag[i];
        }

        let pseudoinv = svd.v() * &s_inv * svd.u().adjoint();

        assert_matrix_eq!(
            &pseudoinv * &matrix,
            Mat::identity(2, 2),
            comp = abs,
            tol = 1e-10
        );
    }
}
