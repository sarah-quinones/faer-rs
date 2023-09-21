use faer::{dbgf, polars::polars_to_faer_f64};
use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let directory = "./faer/examples/";

    for (filename, cols) in [
        (
            "diabetes_data_raw.parquet",
            [
                "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6",
            ]
            .as_slice(),
        ),
        (
            "iris.parquet",
            [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ]
            .as_slice(),
        ),
    ] {
        dbg!(filename);

        let data = LazyFrame::scan_parquet(
            format!("{directory}{filename}"),
            ScanArgsParquet {
                n_rows: None,
                cache: true,
                parallel: ParallelStrategy::Auto,
                rechunk: true,
                row_count: None,
                low_memory: false,
                cloud_options: None,
                use_statistics: true,
            },
        )
        .and_then(|df| polars_to_faer_f64(df, cols, 0.0))
        .unwrap();
        dbgf!("6.2?", data);
    }

    Ok(())
}
