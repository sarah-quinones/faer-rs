use diol::result::*;
use equator::assert;
use std::{
    fs::File,
    io::{self, BufReader, BufWriter},
    iter::zip,
};

fn merge(a: BenchResult, b: BenchResult) -> BenchResult {
    let mut a = a;

    for (a, b) in zip(&mut a.groups, b.groups) {
        assert!(a.args == b.args);
        a.function.extend(b.function);
    }
    a
}

trait Merge {
    fn merge(self, other: Self) -> Self;
}

impl Merge for BenchResult {
    fn merge(self, other: Self) -> Self {
        merge(self, other)
    }
}

fn main() -> io::Result<()> {
    let open = |name: &str| -> io::Result<BenchResult> {
        Ok(serde_json::de::from_reader(BufReader::new(File::open(
            name,
        )?))?)
    };

    if let (Ok(mkl_mt), Ok(openblas_mt), Ok(faer_mt)) = (
        open("./diol_mkl_mt.json"),
        open("./diol_openblas_mt.json"),
        open("./diol_faer_mt.json"),
    ) {
        serde_json::ser::to_writer(
            BufWriter::new(File::create("./diol_mt.json")?),
            &faer_mt.merge(mkl_mt).merge(openblas_mt),
        )?;
    }

    if let (Ok(mkl_st), Ok(openblas_st), Ok(faer_st), Ok(nalgebra_st)) = (
        open("./diol_mkl_st.json"),
        open("./diol_openblas_st.json"),
        open("./diol_faer_st.json"),
        open("./diol_nalgebra_st.json"),
    ) {
        serde_json::ser::to_writer(
            BufWriter::new(File::create("./diol_st.json")?),
            &faer_st.merge(mkl_st).merge(openblas_st).merge(nalgebra_st),
        )?;
    }

    Ok(())
}
