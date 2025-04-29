use crate::internal_prelude::*;
use crate::linalg::matmul::triangular::DiagonalKind;
use crate::{MatRef, assert};
use linalg::matmul::triangular::BlockStructure;

pub fn has_spicy_matmul<T: ComplexField>() -> bool {
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if const { T::IS_NATIVE_F64 || T::IS_NATIVE_F32 || T::IS_NATIVE_C64 || T::IS_NATIVE_C32 } {
		if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
			return true;
		}
	}
	false
}

pub fn spicy_matmul_scratch<T: ComplexField>(nrows: usize, ncols: usize, depth: usize, gather: bool, diag: bool) -> StackReq {
	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if const { T::IS_NATIVE_F64 || T::IS_NATIVE_F32 || T::IS_NATIVE_C64 || T::IS_NATIVE_C32 } {
		if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
			return StackReq::EMPTY;
		}
	}

	let diag = temp_mat_scratch::<T>(nrows, if diag { depth } else { 0 });
	let gather = temp_mat_scratch::<T>(nrows, if gather { ncols } else { 0 });
	diag.and(gather)
}

#[math]
pub fn spicy_matmul<I: Index, T: ComplexField>(
	C: MatMut<'_, T>,
	C_block: BlockStructure,
	row_idx: Option<&[I]>,
	col_idx: Option<&[I]>,
	beta: Accum,

	A: MatRef<'_, T>,
	conj_A: Conj,

	B: MatRef<'_, T>,
	conj_B: Conj,

	D: Option<DiagRef<'_, T>>,
	alpha: T,

	par: Par,

	stack: &mut MemStack,
) {
	let mut C = C;
	assert!(all(
		A.ncols() == B.nrows(),
		A.nrows() == row_idx.map(|idx| idx.len()).unwrap_or(C.nrows()),
		B.ncols() == col_idx.map(|idx| idx.len()).unwrap_or(C.ncols()),
	));

	let nrows = A.nrows();
	let ncols = B.ncols();
	let depth = A.ncols();
	if nrows == 0 || ncols == 0 {
		return;
	}
	let par = if (nrows * ncols).saturating_mul(depth) > 32usize * 32usize * 32usize {
		par
	} else {
		Par::Seq
	};

	if let Some(row_idx) = row_idx {
		for &i in row_idx {
			assert!(i.zx() < C.nrows());
		}
	}
	if let Some(col_idx) = col_idx {
		for &j in col_idx {
			assert!(j.zx() < C.ncols());
		}
	}

	#[cfg(all(target_arch = "x86_64", feature = "std"))]
	if const { T::IS_NATIVE_F64 || T::IS_NATIVE_F32 || T::IS_NATIVE_C64 || T::IS_NATIVE_C32 } {
		use private_gemm_x86::*;

		let feat = if std::arch::is_x86_feature_detected!("avx512f") {
			Some(InstrSet::Avx512)
		} else if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
			Some(InstrSet::Avx256)
		} else {
			None
		};

		if let Some(feat) = feat {
			let mut C = C;
			let mut A = A;
			let mut B = B;
			let mut row_idx = row_idx;
			let mut col_idx = col_idx;

			if matches!(C_block, BlockStructure::StrictTriangularLower | BlockStructure::UnitTriangularLower) {
				if nrows == 0 {
					return;
				}
				A = A.get(1.., ..);
				if let Some(row_idx) = &mut row_idx {
					*row_idx = &row_idx[1..];
				} else {
					C = C.get_mut(1.., ..);
				}
			}

			if matches!(C_block, BlockStructure::StrictTriangularUpper | BlockStructure::UnitTriangularUpper) {
				if ncols == 0 {
					return;
				}
				B = B.get(.., 1..);

				if let Some(col_idx) = &mut col_idx {
					*col_idx = &col_idx[1..];
				} else {
					C = C.get_mut(.., 1..);
				}
			}

			unsafe {
				private_gemm_x86::gemm(
					const {
						if T::IS_NATIVE_F64 {
							DType::F64
						} else if T::IS_NATIVE_F32 {
							DType::F32
						} else if T::IS_NATIVE_C64 {
							DType::C64
						} else {
							DType::C32
						}
					},
					const {
						if size_of::<I>() == 4 {
							IType::U32
						} else if size_of::<I>() == 8 {
							IType::U64
						} else {
							unreachable!()
						}
					},
					feat,
					nrows,
					ncols,
					depth,
					C.as_ptr_mut() as _,
					C.row_stride(),
					C.col_stride(),
					row_idx.map(|idx| idx.as_ptr()).unwrap_or(core::ptr::null()) as _,
					col_idx.map(|idx| idx.as_ptr()).unwrap_or(core::ptr::null()) as _,
					match C_block {
						BlockStructure::Rectangular => DstKind::Full,
						BlockStructure::TriangularLower => DstKind::Lower,
						BlockStructure::StrictTriangularLower => DstKind::Lower,
						BlockStructure::UnitTriangularLower => DstKind::Lower,
						BlockStructure::TriangularUpper => DstKind::Upper,
						BlockStructure::StrictTriangularUpper => DstKind::Upper,
						BlockStructure::UnitTriangularUpper => DstKind::Upper,
					},
					match beta {
						crate::Accum::Replace => Accum::Replace,
						crate::Accum::Add => Accum::Add,
					},
					A.as_ptr() as _,
					A.row_stride(),
					A.col_stride(),
					conj_A == Conj::Yes,
					D.map(|D| D.column_vector().as_ptr() as _).unwrap_or(core::ptr::null()),
					D.map(|D| D.column_vector().row_stride()).unwrap_or(0),
					B.as_ptr() as _,
					B.row_stride(),
					B.col_stride(),
					conj_B == Conj::Yes,
					&raw const alpha as _,
					par.degree(),
				);
				return;
			}
		}
	}

	let (mut out, stack) = unsafe { temp_mat_uninit::<T, _, _>(nrows, if row_idx.is_some() || col_idx.is_some() { ncols } else { 0 }, stack) };
	let mut out = out.as_mat_mut();

	let (mut scaled, stack) = unsafe { temp_mat_uninit::<T, _, _>(nrows, if D.is_some() { depth } else { 0 }, stack) };
	let mut scaled = scaled.as_mat_mut();

	let _ = stack;

	let A = if let Some(D) = D {
		for k in 0..depth {
			let d = real(D[k]);
			zip!(scaled.rb_mut().col_mut(k), A.col(k)).for_each(|unzip!(x, y)| *x = mul_real(*y, d));
		}
		scaled.rb()
	} else {
		A
	};

	{
		let mut C = if row_idx.is_some() || col_idx.is_some() {
			out.rb_mut()
		} else {
			C.rb_mut()
		};

		let size = Ord::min(nrows, ncols);

		if C_block.is_dense() {
			linalg::matmul::triangular::matmul_with_conj(
				C,
				C_block,
				if row_idx.is_some() || col_idx.is_some() { Accum::Replace } else { beta },
				A,
				BlockStructure::Rectangular,
				conj_A,
				B,
				BlockStructure::Rectangular,
				conj_B,
				alpha,
				par,
			);
		} else {
			linalg::matmul::triangular::matmul_with_conj(
				C.rb_mut().get_mut(..size, ..size),
				C_block,
				if row_idx.is_some() || col_idx.is_some() { Accum::Replace } else { beta },
				A.get(..size, ..),
				BlockStructure::Rectangular,
				conj_A,
				B.get(.., ..size),
				BlockStructure::Rectangular,
				conj_B,
				copy(alpha),
				par,
			);

			if C_block.is_lower() && nrows > ncols {
				linalg::matmul::matmul_with_conj(
					C.rb_mut().get_mut(size.., ..size),
					if row_idx.is_some() || col_idx.is_some() { Accum::Replace } else { beta },
					A.get(size.., ..),
					conj_A,
					B.get(.., ..size),
					conj_B,
					alpha,
					par,
				);
			} else if ncols > nrows {
				linalg::matmul::matmul_with_conj(
					C.rb_mut().get_mut(..size, size..),
					if row_idx.is_some() || col_idx.is_some() { Accum::Replace } else { beta },
					A.get(..size, ..),
					conj_A,
					B.get(.., size..),
					conj_B,
					alpha,
					par,
				);
			}
		}
	}

	let lower = C_block.is_lower();
	let upper = C_block.is_upper();

	let diag = matches!(C_block.diag_kind(), DiagonalKind::Generic) as usize;

	match (row_idx, col_idx) {
		(Some(row_idx), Some(col_idx)) => match beta {
			Accum::Replace => {
				for (j, &jj) in col_idx.iter().enumerate() {
					for (i, &ii) in row_idx.iter().enumerate() {
						if (lower && j < i + diag) || (upper && i < j + diag) {
							C[(ii.zx(), jj.zx())] = copy(out[(i, j)]);
						}
					}
				}
			},
			Accum::Add => {
				for (j, &jj) in col_idx.iter().enumerate() {
					for (i, &ii) in row_idx.iter().enumerate() {
						if (lower && j < i + diag) || (upper && i < j + diag) {
							C[(ii.zx(), jj.zx())] = C[(ii.zx(), jj.zx())] + out[(i, j)];
						}
					}
				}
			},
		},
		(Some(row_idx), None) => match beta {
			Accum::Replace => {
				for j in 0..ncols {
					for (i, &ii) in row_idx.iter().enumerate() {
						if (lower && j < i + diag) || (upper && i < j + diag) {
							C[(ii.zx(), j)] = copy(out[(i, j)]);
						}
					}
				}
			},
			Accum::Add => {
				for j in 0..ncols {
					for (i, &ii) in row_idx.iter().enumerate() {
						if (lower && j < i + diag) || (upper && i < j + diag) {
							C[(ii.zx(), j)] = C[(ii.zx(), j)] + out[(i, j)];
						}
					}
				}
			},
		},
		(None, Some(col_idx)) => match beta {
			Accum::Replace => {
				for (j, &jj) in col_idx.iter().enumerate() {
					for i in 0..nrows {
						if (lower && j < i + diag) || (upper && i < j + diag) {
							C[(i, jj.zx())] = copy(out[(i, j)]);
						}
					}
				}
			},
			Accum::Add => {
				for (j, &jj) in col_idx.iter().enumerate() {
					for i in 0..nrows {
						if (lower && j < i + diag) || (upper && i < j + diag) {
							C[(i, jj.zx())] = C[(i, jj.zx())] + out[(i, j)];
						}
					}
				}
			},
		},
		(None, None) => {},
	}
}
