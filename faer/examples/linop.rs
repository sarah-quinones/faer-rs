#![allow(non_snake_case)]

use dyn_stack::{MemStack, StackReq};
use reborrow::{Reborrow, ReborrowMut};

use faer::{dyn_stack, reborrow};

use faer::mat::{AsMatMut, MatMut, MatRef};
use faer::matrix_free::{BiLinOp, LinOp};
use faer::sparse::{SparseRowMatRef, linalg};
use faer::{Accum, Conj, Par};

use faer::c64;

use faer::traits::ext::ComplexFieldExt;
use faer::traits::{ComplexField, Index};

#[derive(Debug)]
pub struct BinvAOperator<'a, I: Index, T: ComplexField> {
	pub Binv: linalg::lu::LuRef<'a, I, T>,
	pub A: SparseRowMatRef<'a, I, T>,
}

#[derive(Debug)]
pub struct BinvAOperatorScratch<'a, I: Index> {
	pub Binv: &'a linalg::lu::SymbolicLu<I>,
}

impl<I: Index, T: ComplexField> LinOp<T> for BinvAOperatorScratch<'_, I> {
	fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		self.Binv.solve_in_place_scratch::<T>(rhs_ncols, par)
	}

	fn nrows(&self) -> usize {
		self.Binv.nrows()
	}

	fn ncols(&self) -> usize {
		self.Binv.nrows()
	}

	#[allow(unused_variables)]
	fn apply(
		&self,
		out: MatMut<'_, T>,
		rhs: MatRef<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) {
		panic!()
	}

	#[allow(unused_variables)]
	fn conj_apply(
		&self,
		out: MatMut<'_, T>,
		rhs: MatRef<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) {
		panic!()
	}
}

impl<I: Index, T: ComplexField> BiLinOp<T> for BinvAOperatorScratch<'_, I> {
	fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		faer::linalg::temp_mat_scratch::<T>(self.Binv.nrows(), rhs_ncols).and(
			self.Binv
				.solve_transpose_in_place_scratch::<T>(rhs_ncols, par),
		)
	}

	#[allow(unused_variables)]
	fn transpose_apply(
		&self,
		out: MatMut<'_, T>,
		rhs: MatRef<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) {
		panic!()
	}

	#[allow(unused_variables)]
	fn adjoint_apply(
		&self,
		out: MatMut<'_, T>,
		rhs: MatRef<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) {
		panic!()
	}
}

impl<I: Index, T: ComplexField> LinOp<T> for BinvAOperator<'_, I, T> {
	fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		LinOp::<T>::apply_scratch(
			&BinvAOperatorScratch {
				Binv: self.Binv.symbolic(),
			},
			rhs_ncols,
			par,
		)
	}

	fn nrows(&self) -> usize {
		self.Binv.symbolic().nrows()
	}

	fn ncols(&self) -> usize {
		self.A.ncols()
	}

	fn apply(
		&self,
		out: MatMut<'_, T>,
		rhs: MatRef<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) {
		let mut out = out;

		linalg::matmul::sparse_dense_matmul(
			out.rb_mut(),
			Accum::Replace,
			self.A,
			rhs,
			T::one(),
			par,
		);
		self.Binv
			.solve_in_place_with_conj(Conj::No, out.rb_mut(), par, stack);
	}

	fn conj_apply(
		&self,
		out: MatMut<'_, T>,
		rhs: MatRef<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) {
		let mut out = out;
		linalg::matmul::sparse_dense_matmul(
			out.rb_mut(),
			Accum::Replace,
			self.A.conjugate(),
			rhs,
			T::one(),
			par,
		);
		self.Binv
			.solve_in_place_with_conj(Conj::Yes, out.rb_mut(), par, stack);
	}
}

// not needed for partial_eigen. left here as an example
impl<I: Index, T: ComplexField> BiLinOp<T> for BinvAOperator<'_, I, T> {
	fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
		BiLinOp::<T>::transpose_apply_scratch(
			&BinvAOperatorScratch {
				Binv: self.Binv.symbolic(),
			},
			rhs_ncols,
			par,
		)
	}

	fn transpose_apply(
		&self,
		out: MatMut<'_, T>,
		rhs: MatRef<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) {
		let (m, n) = (rhs.nrows(), rhs.ncols());

		let (mut tmp, stack) =
			faer::linalg::temp_mat_zeroed::<T, _, _>(m, n, stack);
		let mut tmp = tmp.as_mat_mut();
		tmp.copy_from(rhs);

		self.Binv.solve_transpose_in_place_with_conj(
			Conj::No,
			tmp.rb_mut(),
			par,
			stack,
		);
		linalg::matmul::sparse_dense_matmul(
			out,
			Accum::Replace,
			self.A.transpose(),
			tmp.rb(),
			T::one(),
			par,
		);
	}

	fn adjoint_apply(
		&self,
		out: MatMut<'_, T>,
		rhs: MatRef<'_, T>,
		par: Par,
		stack: &mut MemStack,
	) {
		let (m, n) = (rhs.nrows(), rhs.ncols());

		let (mut tmp, stack) =
			faer::linalg::temp_mat_zeroed::<T, _, _>(m, n, stack);
		let mut tmp = tmp.as_mat_mut();
		tmp.copy_from(rhs);

		self.Binv.solve_transpose_in_place_with_conj(
			Conj::Yes,
			tmp.rb_mut(),
			par,
			stack,
		);
		linalg::matmul::sparse_dense_matmul(
			out,
			Accum::Replace,
			self.A.adjoint(),
			tmp.rb(),
			T::one(),
			par,
		);
	}
}

use dyn_stack::MemBuffer;
use faer::prelude::default;
use faer::sparse::{SparseRowMat, Triplet};
use faer::{Col, Mat};

fn main() -> Result<(), Box<dyn core::error::Error>> {
	use faer::stats::prelude::*;
	let rng = &mut StdRng::seed_from_u64(0);

	let dist = ComplexDistribution::new(StandardNormal, StandardNormal);

	let n = 200usize;
	let density = 0.1;

	let mut A_triplets = Vec::new();
	let mut B_triplets = Vec::new();

	for M in [&mut A_triplets, &mut B_triplets] {
		M.extend((0..n).map(|i| Triplet::new(i, i, dist.rand::<c64>(rng))));

		for j in 0..n {
			for i in 0..n {
				if StandardUniform.rand::<f64>(rng) > density {
					let val = dist.rand::<c64>(rng);
					M.push(Triplet::new(i, j, val));
				}
			}
		}
	}

	let A =
		SparseRowMat::<usize, c64>::try_new_from_triplets(n, n, &A_triplets)?;
	let B =
		SparseRowMat::<usize, c64>::try_new_from_triplets(n, n, &B_triplets)?;

	let n = A.nrows();
	let n_eig = 2;

	// since we stored B in CSR format, we start by converting it to CSC, since
	// that's what decompositions usually prefer
	//
	// alternatively: we could compute the LU of B.transpose(), then replace all
	// the `solve` calls with `solve_transpose` and vice-versa
	let B = B.to_col_major()?;

	// this example uses a single thread
	let par = Par::Seq;

	// symbolic factorization
	let symbolic = linalg::lu::factorize_symbolic_lu(B.symbolic(), default())?;
	let mut numeric = linalg::lu::NumericLu::new();

	// computing allocation size
	let factorize_scratch =
		symbolic.factorize_numeric_lu_scratch::<c64>(par, default());
	let eigen_scratch = faer::matrix_free::eigen::partial_eigen_scratch(
		&BinvAOperatorScratch { Binv: &symbolic } as &dyn LinOp<c64>,
		n_eig,
		par,
		default(),
	);

	// we only allocate for the eigendecomp after the factorization is done
	// using its allocation, so we use `any_of` here.
	//
	// call stack representation:
	// ```
	// CALL factorization(stack)
	//     factorization allocates scratch space from stack
	//     factorization deallocates its scratch space
	//
	// CALL eigendecomp(stack)
	//     eigendecomp allocates scratch space from stack
	//     eigendecomp deallocates its scratch space
	// ```
	//
	// if we were executing both at the same time, or performing the
	// factorization while the eigendecomp factorization is still allocated,
	// we'd use `all_of` instead
	//
	// example:
	// ```
	// eigendecomp allocates scratch space from stack
	//
	// CALL factorization(stack)
	//     factorization allocates scratch space from stack
	//     factorization deallocates its scratch space
	//
	// eigendecomp deallocates its scratch space
	// ```
	let scratch = StackReq::any_of(&[factorize_scratch, eigen_scratch]);

	// preallocation
	let mut mem = MemBuffer::new(scratch);

	let stack = MemStack::new(&mut mem);
	let Binv = if true {
		// if factorizing for the first time
		symbolic.factorize_numeric_lu(
			&mut numeric,
			B.rb(),
			par,
			stack,
			default(),
		)?
	} else {
		// if you need to rematerialize Binv
		linalg::lu::LuRef::new_unchecked(&symbolic, &numeric)
	};

	let mut eigvecs = Mat::<c64>::zeros(n, n_eig);
	let mut eigvals = vec![c64::ZERO; n_eig];

	let v0 = Col::<c64>::ones(n);

	faer::matrix_free::eigen::partial_eigen(
		eigvecs.rb_mut(),
		&mut eigvals,
		&BinvAOperator { Binv, A: A.rb() },
		v0.rb(),
		f64::EPSILON,
		par,
		stack,
		default(),
	);

	println!("computed eigenvalues and corresponding eigenvectors:");
	for k in 0..n_eig {
		let eigval = eigvals[k];
		let eigvec = eigvecs.col(k);

		print!("eigval: {eigval:16.2}, eigvec: [");

		let mut i = 0;
		while i < n {
			if i > 0 {
				print!(", ");
			}

			if i >= 2 && i + 2 < n {
				print!("…");

				i = n - 2;
				continue;
			}

			let ei = eigvec[i];
			print!("{ei:12.2}");

			i += 1;
		}

		println!("]");
	}

	{
		use faer::prelude::*;

		println!("sanity check:");
		let Binv_A = B.sp_lu()?.solve(A.to_dense());
		let eig = Binv_A.eigen().unwrap();
		let mut sort = Vec::from_iter(0..n);
		sort.sort_unstable_by(|&i, &j| {
			let i = eig.S()[i].norm();
			let j = eig.S()[j].norm();
			j.total_cmp(&i)
		});

		for k in 0..n_eig {
			let eigval = eig.S()[sort[k]];
			let eigvec = eig.U().col(sort[k]);

			let normalized0 = |x: ColRef<'_, c64>| x / x.norm_l2();
			let normalized =
				|x: ColRef<'_, c64>| normalized0((x / Scale(x[0])).rb());

			let err = (eigval / eigvals[k] - c64::ONE).norm()
				+ (normalized(eigvec) - normalized(eigvecs.col(k))).norm_l2();

			print!("eigval: {eigval:16.2}, eigvec: [");

			let mut i = 0;
			while i < n {
				if i > 0 {
					print!(", ");
				}

				if i >= 2 && i + 2 < n {
					print!("…");

					i = n - 2;
					continue;
				}

				let ei = eigvec[i];
				print!("{ei:12.2}");

				i += 1;
			}
			println!("], gap: {err:8.2e}");
		}
	}

	Ok(())
}
