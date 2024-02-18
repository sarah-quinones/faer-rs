use faer::{dbgf, polars::polars_to_faer_f64};
use polars::prelude::*;

fn main() -> PolarsResult<()> {
    let directory = "./faer/examples/";
    for filename in ["diabetes_data_raw.parquet", "iris.parquet"] {
        dbg!(filename);

        let data = LazyFrame::scan_parquet(
            format!("{directory}{filename}"),
            ScanArgsParquet {
                n_rows: None,
                cache: true,
                parallel: ParallelStrategy::Auto,
                rechunk: true,
                row_index: None,
                low_memory: false,
                cloud_options: None,
                use_statistics: true,
                ..Default::default()
            },
        )
        .and_then(|lf| polars_to_faer_f64(lf))
        .unwrap();
        dbgf!("6.2?", data);
    }

    Ok(())
}
