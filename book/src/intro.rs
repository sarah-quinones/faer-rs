use faer::{dbgf, mat};
use faer_core::scale;

pub fn main() {
    let m = mat![
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.1, 3.1, 4.1],
        [1.2, 2.2, 3.2, 4.2f64]
    ];

    // we can add or subtract matrices
    let m_plus_m = &m + &m;
    // scaling by a constant and matrix multiplication are also possible
    // using familiar math notation
    let mt_times_2m = m.transpose() * scale(2.0) * &m;

    // matrices can be printed using the Debug trait
    println!("{mt_times_2m:6.2?}");

    // the dbgf macro is provided for ease of debugging, by displaying
    // the location of the print statement, similarly to how the `dbg!`
    // macro works.
    dbgf!("6.2?", &m_plus_m, &mt_times_2m);
}
