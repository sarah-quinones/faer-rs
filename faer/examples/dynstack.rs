#![allow(non_snake_case)]

use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::mat::AsMatMut;
use faer::prelude::*;
use faer::{Accum, dyn_stack};

fn foo_scratch(n: usize) -> StackReq {
	let tmp = faer::linalg::temp_mat_scratch::<f64>(n, n);
	let llt = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(n, Par::Seq, default());
	tmp.and(StackReq::any_of(&[tmp.and(llt), tmp]))
}

// A <- A * B
fn foo(mut A: MatMut<'_, f64>, B: MatRef<'_, f64>, stack: &mut MemStack) {
	let n = A.nrows();
	let (mut tmp0, stack) = unsafe { faer::linalg::temp_mat_uninit(n, n, stack) };
	let mut tmp0: MatMut<'_, f64> = tmp0.as_mat_mut();
	{
		let (mut tmp1, stack) = unsafe { faer::linalg::temp_mat_uninit(n, n, stack) };
		let _tmp1: MatMut<'_, f64> = tmp1.as_mat_mut();

		faer::linalg::matmul::matmul(&mut tmp0, Accum::Replace, &A, &B, 1.0, Par::Seq);
		A.copy_from(&tmp0);

		let _ = faer::linalg::cholesky::llt::factor::cholesky_in_place(A.rb_mut(), default(), Par::Seq, stack, default());
	}

	let (_tmp1, _) = unsafe { faer::linalg::temp_mat_uninit::<f64, _, _>(n, n, stack) };
}

fn main() {
	let n = 128;
	let nthreads = 100;

	let scratch = foo_scratch(n);
	let mut workspace = MemBuffer::new(scratch);
	let stack = MemStack::new(&mut workspace);

	//          stack
	// [....................]
	// let (mut stacks, mut stack) = stack.make_with(nthreads, |_| MemStack::new(&mut []));

	// stacks    stack
	// [...][.................]

	// for i in 0..nthreads {
	// 	let scratch = foo_scratch(n);
	// 	let new;
	// 	(new, stack) = stack.make_aligned_uninit(scratch.size_bytes(), scratch.align_bytes());

	// 	// stacks foo_scratch  stack
	// 	// [...][..][..][..][....]

	// 	stacks[i] = MemStack::new(new);
	// }
	// stacks    foo_scratch
	// [...][..][..][..][..][..]

	// use rayon::prelude::*;
	(0..nthreads).into_iter().for_each(|_| {
		let mut A = Mat::zeros(n, n);
		let B = Mat::zeros(n, n);
		foo(A.as_mut(), B.as_ref(), stack);
	});
}
