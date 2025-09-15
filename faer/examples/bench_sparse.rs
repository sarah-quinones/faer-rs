#![allow(non_snake_case, dead_code)]

use curl::easy::Easy;
use diol::prelude::*;
use faer::Side;
use faer::reborrow::*;
use faer::sparse::*;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::sync::LazyLock;

#[cfg(suitesparse)]
use suitesparse_sys::*;

static MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
	HashMap::from([
		("nd24k", "http://sparse-files.engr.tamu.edu/MM/ND/nd24k.tar.gz"),
		("nd3k", "http://sparse-files.engr.tamu.edu/MM/ND/nd3k.tar.gz"),
		("af shell7", "http://sparse-files.engr.tamu.edu/MM/Schenk_AFE/af_shell7.tar.gz"),
		("G3 circuit", "http://sparse-files.engr.tamu.edu/MM/AMD/G3_circuit.tar.gz"),
	])
});

fn download(name: &str, sym: bool) -> SparseColMat<usize, f64> {
	let url = MAP[name];

	let mut dst = Vec::new();
	{
		let mut easy = Easy::new();
		easy.url(url).unwrap();

		let mut transfer = easy.transfer();
		transfer
			.write_function(|data| {
				dst.extend_from_slice(data);
				Ok(data.len())
			})
			.unwrap();
		transfer.perform().unwrap();
	}

	let tmp = std::env::temp_dir().join("__tmp__.mtx");

	let tar = flate2::read::GzDecoder::new(&*dst);
	let mut mtx = tar::Archive::new(tar);
	let mut size = 0;

	for entry in mtx.entries().unwrap() {
		let mut entry = entry.unwrap();
		if entry.path().unwrap().extension() == Some(OsStr::new("mtx")) && entry.size() > size {
			size = entry.size();
			entry.unpack(&tmp).unwrap();
		}
	}

	let matrix_market_rs::MtxData::Sparse([nrows, ncols], coord, val, _) = matrix_market_rs::MtxData::<f64, 2>::from_file(&tmp).unwrap() else {
		panic!();
	};

	SparseColMat::try_new_from_triplets(
		nrows,
		ncols,
		&if sym {
			coord
				.iter()
				.zip(&val)
				.flat_map(|(&[row, col], &val)| {
					let val = if row == col { val / 2.0 } else { val };
					[Triplet::new(row, col, val), Triplet::new(col, row, val)]
				})
				.collect::<Vec<_>>()
		} else {
			coord
				.iter()
				.zip(&val)
				.map(|(&[row, col], &val)| Triplet::new(row, col, val))
				.collect::<Vec<_>>()
		},
	)
	.unwrap()
}

#[cfg(suitesparse)]
fn suitesparse_llt(bencher: Bencher, name: String) {
	let A = download(&name, true);

	unsafe {
		let mut common = core::mem::zeroed::<cholmod_common>();
		let mut A = cholmod_sparse_struct {
			nrow: A.nrows(),
			ncol: A.ncols(),
			nzmax: A.compute_nnz(),
			p: A.col_ptr().as_ptr() as _,
			i: A.row_idx().as_ptr() as _,
			nz: A.col_nnz().map(|x| x.as_ptr() as _).unwrap_or(core::ptr::null_mut()),
			x: A.val().as_ptr() as _,
			z: core::ptr::null_mut(),
			stype: -1,
			itype: CHOLMOD_LONG as _,
			xtype: CHOLMOD_REAL as _,
			dtype: CHOLMOD_DOUBLE as _,
			sorted: 1,
			packed: A.col_nnz().is_none() as _,
		};

		cholmod_l_start(&mut common);
		cholmod_l_defaults(&mut common);
		common.nmethods = 1;
		common.method[0].ordering = CHOLMOD_AMD as _;

		let mut L = cholmod_l_analyze(&mut A, &mut common);
		bencher.bench(|| {
			cholmod_l_factorize(&mut A, L, &mut common);
		});
		cholmod_l_free_factor(&mut L, &mut common);
		cholmod_l_finish(&mut common);
	}
}

fn faer_llt(bencher: Bencher, name: String) {
	let A = download(&name, true);

	let symbolic = linalg::solvers::SymbolicLlt::try_new(A.symbolic(), Side::Lower).unwrap();
	bencher.bench(|| linalg::solvers::Llt::try_new_with_symbolic(symbolic.clone(), A.rb(), Side::Lower));
}

fn main() -> eyre::Result<()> {
	spindle::with_lock(rayon::current_num_threads(), || {
		let bench = Bench::from_args()?;

		bench.register_many(
			"llt",
			{
				let list = diol::variadics::Nil;
				#[cfg(suitesparse)]
				let list = diol::variadics::Cons {
					head: suitesparse_llt,
					tail: list,
				};
				#[cfg(faer)]
				let list = diol::variadics::Cons { head: faer_llt, tail: list };
				list
			},
			["nd3k", "af shell7", "G3 circuit"].map(String::from),
		);

		bench.run()?;

		Ok(())
	})
}
