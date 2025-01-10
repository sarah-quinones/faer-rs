#![allow(missing_docs)]

/// compile-time bound-checked indexing types
pub mod bound;

#[doc(hidden)]
pub mod thread {
	use crate::*;

	/// executes the two operations, possibly in parallel, while splitting the amount of parallelism
	/// between the two
	#[inline]
	pub fn join_raw(op_a: impl Send + FnOnce(Par), op_b: impl Send + FnOnce(Par), parallelism: Par) {
		fn implementation(op_a: &mut (dyn Send + FnMut(Par)), op_b: &mut (dyn Send + FnMut(Par)), parallelism: Par) {
			match parallelism {
				Par::Seq => (op_a(parallelism), op_b(parallelism)),
				#[cfg(feature = "rayon")]
				Par::Rayon(n_threads) => {
					let n_threads = n_threads.get();
					if n_threads == 1 {
						(op_a(Par::Seq), op_b(Par::Seq))
					} else {
						let parallelism = Par::Rayon(core::num::NonZeroUsize::new(n_threads - n_threads / 2).unwrap());
						rayon::join(|| op_a(parallelism), || op_b(parallelism))
					}
				},
			};
		}
		let mut op_a = Some(op_a);
		let mut op_b = Some(op_b);
		implementation(
			&mut |parallelism| (op_a.take().unwrap())(parallelism),
			&mut |parallelism| (op_b.take().unwrap())(parallelism),
			parallelism,
		)
	}

	/// unsafe [`Send`] and [`Sync`] pointer type
	pub struct Ptr<T>(pub *mut T);
	unsafe impl<T> Send for Ptr<T> {}
	unsafe impl<T> Sync for Ptr<T> {}
	impl<T> Copy for Ptr<T> {}
	impl<T> Clone for Ptr<T> {
		#[inline]
		fn clone(&self) -> Self {
			*self
		}
	}

	/// the amount of threads that should ideally execute an operation with the given parallelism
	#[inline]
	pub fn parallelism_degree(parallelism: Par) -> usize {
		match parallelism {
			Par::Seq => 1,
			#[cfg(feature = "rayon")]
			Par::Rayon(n_threads) => n_threads.get(),
		}
	}

	/// returns the start and length of a subsegment of `0..n`, split between `chunk_count`
	/// consumers, for the consumer at index `idx`
	///
	/// for the same `n` and `chunk_count`, different values of `idx` between in `0..chunk_count`
	/// will represent distinct subsegments
	#[inline]
	pub fn par_split_indices(n: usize, idx: usize, chunk_count: usize) -> (usize, usize) {
		let chunk_size = n / chunk_count;
		let rem = n % chunk_count;

		let idx_to_col_start = move |idx| {
			if idx < rem { idx * (chunk_size + 1) } else { rem + idx * chunk_size }
		};

		let start = idx_to_col_start(idx);
		let end = idx_to_col_start(idx + 1);
		(start, end - start)
	}
}

/// simd helper utilities based on lifetime-bound indices
pub mod simd;

/// approximate comparators for testing purposes
pub mod approx;
