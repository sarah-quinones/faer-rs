use csv::Writer;
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
        open("./target/diol_mkl_mt.json"),
        open("./target/diol_openblas_mt.json"),
        open("./target/diol_faer_mt.json"),
    ) {
        let merged = faer_mt.merge(mkl_mt).merge(openblas_mt);

        let mut idx = 0;
        for ty in ["f32", "f64", "c32", "c64"] {
            for algo in [
                "cholesky", "qr", "piv_qr", "lu", "piv_lu", "svd", "thin_svd", "eigh", "eig",
            ] {
                let mut writer =
                    Writer::from_writer(File::create(format!("./target/mt_{algo}_{ty}.csv"))?);
                let group = &merged.groups[idx];

                writer.write_record(&["n", "faer", "mkl", "openblas"])?;

                for arg in 0..group.args.len() {
                    let mut data = vec![];
                    match group.arg(arg) {
                        BenchArg::Plot(n) => data.push(format!("{}", n.0)),
                        _ => panic!(),
                    };
                    for func in 0..group.function.len() {
                        data.push(format!(
                            "{}",
                            group.at(Func(func), Arg(arg)).1.unwrap().mean_stddev().0
                        ));
                    }

                    writer.write_record(data)?;
                }
                idx += 1;
            }
        }

        serde_json::ser::to_writer(
            BufWriter::new(File::create("./target/diol_mt.json")?),
            &merged,
        )?;
    }

    if let (Ok(mkl_st), Ok(openblas_st), Ok(faer_st), Ok(nalgebra_st)) = (
        open("./target/diol_mkl_st.json"),
        open("./target/diol_openblas_st.json"),
        open("./target/diol_faer_st.json"),
        open("./target/diol_nalgebra_st.json"),
    ) {
        let merged = faer_st.merge(mkl_st).merge(openblas_st).merge(nalgebra_st);

        let mut idx = 0;
        for ty in ["f32", "f64", "c32", "c64"] {
            for algo in [
                "cholesky", "qr", "piv_qr", "lu", "piv_lu", "svd", "thin_svd", "eigh", "eig",
            ] {
                let mut writer =
                    Writer::from_writer(File::create(format!("./target/st_{algo}_{ty}.csv"))?);
                let group = &merged.groups[idx];

                writer.write_record(&["n", "faer", "mkl", "openblas", "nalgebra"])?;

                for arg in 0..group.args.len() {
                    let mut data = vec![];
                    match group.arg(arg) {
                        BenchArg::Plot(n) => data.push(format!("{}", n.0)),
                        _ => panic!(),
                    };
                    for func in 0..group.function.len() {
                        data.push(format!(
                            "{}",
                            group.at(Func(func), Arg(arg)).1.unwrap().mean_stddev().0
                        ));
                    }

                    writer.write_record(data)?;
                }
                idx += 1;
            }
        }

        serde_json::ser::to_writer(
            BufWriter::new(File::create("./target/diol_st.json")?),
            &merged,
        )?;
    }

    Ok(())
}
