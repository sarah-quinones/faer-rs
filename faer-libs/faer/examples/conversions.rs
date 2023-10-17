use faer::{assert_matrix_eq, mat, IntoFaer, IntoNalgebra, IntoNdarray};

fn main() {
    let matrix = mat![
        [2.28583329, -0.90628668, -1.71493024],
        [-0.90628668, 4.00729077, 2.17332502],
        [-1.71493024, 2.17332502, 1.97196187]
    ];

    let nalgebra = matrix.as_ref().into_nalgebra();
    let ndarray = matrix.as_ref().into_ndarray();

    // compare multiplication using faer, with multiplication using nalgebra
    assert_matrix_eq!(
        &matrix * &matrix,
        (nalgebra * nalgebra).view_range(.., ..).into_faer(),
        comp = abs,
        tol = 1e-14
    );

    // compare addition using faer, with addition using ndarray
    assert_matrix_eq!(
        &matrix + &matrix,
        (&ndarray + &ndarray).view().into_faer(),
        comp = abs,
        tol = 1e-14
    );
}
