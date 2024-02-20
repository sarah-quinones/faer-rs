use faer::{assert_matrix_eq, mat, prelude::*, Side};

fn main() {
    let matrix = mat![
        [2.28583329, -0.90628668, -1.71493024],
        [-0.90628668, 4.00729077, 2.17332502],
        [-1.71493024, 2.17332502, 1.97196187]
    ];

    let chol = match matrix.cholesky(Side::Lower) {
        Ok(chol) => chol,
        Err(_) => panic!("Non positive definite matrix"),
    };

    let rhs = mat![
        [-0.29945184, -0.5228196],
        [0.84136125, -1.15768694],
        [1.25678304, -0.46203532]
    ];

    let sol = chol.solve(&rhs);
    assert_matrix_eq!(rhs, &matrix * &sol, comp = abs, tol = 1e-10);
}
