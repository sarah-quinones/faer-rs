#![allow(non_snake_case)]
use core::ffi::c_void;
use faer::dyn_stack::StackReq;
use faer::{c32, c64, cx128, fx128};
trait RealField: faer::traits::RealField + Copy + 'static {}
impl<T: faer::traits::RealField + Copy + 'static> RealField for T {}
trait ComplexField: faer::traits::ComplexField + Copy + 'static {}
impl<T: faer::traits::ComplexField + Copy + 'static> ComplexField for T {}
type AsIs<T> = T;
#[derive(Copy, Clone)]
pub enum Never {}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MatRef {
	pub ptr: *const c_void,
	pub nrows: usize,
	pub ncols: usize,
	pub row_stride: isize,
	pub col_stride: isize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MatMut {
	pub ptr: *mut c_void,
	pub nrows: usize,
	pub ncols: usize,
	pub row_stride: isize,
	pub col_stride: isize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VecRef {
	pub ptr: *const c_void,
	pub len: usize,
	pub stride: isize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VecMut {
	pub ptr: *mut c_void,
	pub len: usize,
	pub stride: isize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SliceRef {
	pub ptr: *const c_void,
	pub len: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SliceMut {
	pub ptr: *mut c_void,
	pub len: usize,
}
pub enum Scalar {}
pub enum Index {}
pub enum Real {}
#[repr(C)]
#[derive(Copy, Clone)]
pub enum Accum {
	Replace,
	Add,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub enum Conj {
	No,
	Yes,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub enum ParTag {
	Seq,
	Rayon,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Par {
	pub tag: ParTag,
	pub nthreads: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub enum Block {
	Rectangular,
	TriangularLower,
	TriangularUpper,
	StrictTriangularLower,
	StrictTriangularUpper,
	UnitTriangularLower,
	UnitTriangularUpper,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Layout {
	pub len_bytes: usize,
	pub align_bytes: usize,
}
impl From<StackReq> for Layout {
	fn from(value: StackReq) -> Self {
		Self {
			len_bytes: value.size_bytes(),
			align_bytes: value.align_bytes(),
		}
	}
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MemAlloc {
	pub ptr: *mut c_void,
	pub len_bytes: usize,
}
impl MemAlloc {
	fn faer(self) -> &'static mut faer::dyn_stack::MemStack {
		faer::dyn_stack::MemStack::new(
			SliceMut {
				ptr: self.ptr,
				len: self.len_bytes,
			}
			.slice(),
		)
	}
}
impl Conj {
	fn faer(self) -> faer::Conj {
		match self {
			Conj::No => faer::Conj::No,
			Conj::Yes => faer::Conj::Yes,
		}
	}
}
impl Block {
	fn faer(self) -> faer::linalg::matmul::triangular::BlockStructure {
		match self {
			Block::Rectangular => faer::linalg::matmul::triangular::BlockStructure::Rectangular,
			Block::TriangularLower => faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
			Block::TriangularUpper => faer::linalg::matmul::triangular::BlockStructure::TriangularUpper,
			Block::StrictTriangularLower => faer::linalg::matmul::triangular::BlockStructure::StrictTriangularLower,
			Block::StrictTriangularUpper => faer::linalg::matmul::triangular::BlockStructure::StrictTriangularUpper,
			Block::UnitTriangularLower => faer::linalg::matmul::triangular::BlockStructure::UnitTriangularLower,
			Block::UnitTriangularUpper => faer::linalg::matmul::triangular::BlockStructure::UnitTriangularUpper,
		}
	}
}
fn scalar<T: ComplexField>(x: *const Scalar) -> T {
	unsafe { (x as *const T).read() }
}
fn real<T: RealField>(x: *const Real) -> T {
	unsafe { (x as *const T).read() }
}
impl MatRef {
	fn faer<'a, T>(self) -> faer::MatRef<'a, T> {
		unsafe { faer::MatRef::from_raw_parts(self.ptr as *const T, self.nrows, self.ncols, self.row_stride, self.col_stride) }
	}
}
impl MatMut {
	fn faer<'a, T>(self) -> faer::MatMut<'a, T> {
		unsafe { faer::MatMut::from_raw_parts_mut(self.ptr as *mut T, self.nrows, self.ncols, self.row_stride, self.col_stride) }
	}
}
#[allow(dead_code)]
impl VecRef {
	fn diag<'a, T>(self) -> faer::diag::DiagRef<'a, T> {
		unsafe { faer::diag::DiagRef::from_raw_parts(self.ptr as *const T, self.len, self.stride) }
	}

	fn col<'a, T>(self) -> faer::col::ColRef<'a, T> {
		unsafe { faer::col::ColRef::from_raw_parts(self.ptr as *const T, self.len, self.stride) }
	}

	fn row<'a, T>(self) -> faer::row::RowRef<'a, T> {
		unsafe { faer::row::RowRef::from_raw_parts(self.ptr as *const T, self.len, self.stride) }
	}
}
#[allow(dead_code)]
impl VecMut {
	fn diag<'a, T>(self) -> faer::diag::DiagMut<'a, T> {
		unsafe { faer::diag::DiagMut::from_raw_parts_mut(self.ptr as *mut T, self.len, self.stride) }
	}

	fn col<'a, T>(self) -> faer::col::ColMut<'a, T> {
		unsafe { faer::col::ColMut::from_raw_parts_mut(self.ptr as *mut T, self.len, self.stride) }
	}

	fn row<'a, T>(self) -> faer::row::RowMut<'a, T> {
		unsafe { faer::row::RowMut::from_raw_parts_mut(self.ptr as *mut T, self.len, self.stride) }
	}
}
impl SliceRef {
	fn slice<'a, T>(self) -> &'a [T] {
		if self.ptr.is_null() && self.len == 0 {
			&[]
		} else {
			unsafe { core::slice::from_raw_parts(self.ptr as _, self.len) }
		}
	}
}
impl SliceMut {
	fn slice<'a, T>(self) -> &'a mut [T] {
		if self.ptr.is_null() && self.len == 0 {
			&mut []
		} else {
			unsafe { core::slice::from_raw_parts_mut(self.ptr as _, self.len) }
		}
	}
}
impl Accum {
	fn faer(self) -> faer::Accum {
		match self {
			Accum::Replace => faer::Accum::Replace,
			Accum::Add => faer::Accum::Add,
		}
	}
}
impl Par {
	fn faer(self) -> faer::Par {
		match self.tag {
			ParTag::Seq => faer::Par::Seq,
			ParTag::Rayon => faer::Par::rayon(self.nthreads),
		}
	}
}
macro_rules! funcs {
	({$(
		pub fn $func:ident<
			$($ty:ident),* $(,)?
		>(
			$($input:ident : $input_ty:ty),* $(,)?
		) $(-> $output_ty: ty)?
			$body:block
	)*}) => {
		$($crate::funcs!(
			impl 0 @ fn $func<$($ty,)*>(
				$($input: $input_ty,)*
			) $(-> $output_ty)? $body
		);)*
	};

	(impl 0 @
		fn $func:ident<
			$($ty:ident,)*
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		$crate::funcs!(
			impl 1 @ fn $func<$($ty,)*>$inputs -> $output_ty $body
		);
	};
	(impl 0 @
		fn $func:ident<
			$($ty:ident,)*
		>$inputs: tt
			$body:block
	) => {
		$crate::funcs!(
			impl 1 @ fn $func<$($ty,)*>$inputs -> () $body
		);
	};

	(impl 1 @
		fn $func:ident<
			$ty0:ident,
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		$crate::funcs!(
			impl 2 @ fn $func<$ty0 in (f32, f64, fx128, c32, c64, cx128,),>$inputs -> $output_ty $body
		);
	};

	(impl 1 @
		fn $func:ident<
			$ty1:ident,
			$ty0:ident,
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		$crate::funcs!(
			impl 2 @ fn $func<$ty1, $ty0 in (f32, f64, fx128, c32, c64, cx128,),>$inputs -> $output_ty $body
		);
	};

	(impl 2 @
		fn $func:ident<
			$ty0:ident in ($($ty0_list: ident,)*),
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		pastey::paste! {
			$(
				#[unsafe(no_mangle)]
				pub unsafe extern "C" fn [<libfaer_v0_23_ $func _ $ty0_list>] $inputs -> $output_ty { type $ty0 = $ty0_list;#[allow(unused_unsafe)] unsafe {$body} }
			)*
		}
	};

	(impl 2 @
		fn $func:ident<
			$ty1:ident,
			$ty0:ident in ($($ty0_list: ident,)*),
		>$inputs: tt -> $output_ty: ty
			$body:block
	) => {
		pastey::paste! {
			$(
				#[unsafe(no_mangle)]
				pub unsafe extern "C" fn [<libfaer_v0_23_ $func _u32_ $ty0_list >] $inputs -> $output_ty { type $ty0 = $ty0_list; type $ty1 = u32;#[allow(unused_unsafe)] unsafe {$body} }
			)*
		}
		pastey::paste! {
			$(
				#[unsafe(no_mangle)]
				pub unsafe extern "C" fn [<libfaer_v0_23_ $func _u64_ $ty0_list >] $inputs -> $output_ty { type $ty0 = $ty0_list; type $ty1 = u64;#[allow(unused_unsafe)] unsafe {$body} }
			)*
		}
	};
}
pub(crate) use funcs;
macro_rules! cerr {
	({$(

		#[ok($ok_path:path)]
		#[err($err_path:path)]
		pub enum $name:ident {
			Ok {$($field:ident: $field_ty:ty),* $(,)?},
			$($err:ident {$($err_field:ident: $err_field_ty:ty),* $(,)?}),*$(,)?
		}

	)*}) => {$(
		#[repr(C)]
		#[derive(Copy, Clone)]
		pub enum $name {
			Ok {$($field: $field_ty,)*},
			$($err {$($err_field: $err_field_ty,)*},)*
			Unknown,
		}

		impl Into<$name> for Result<$ok_path, $err_path> {
			fn into(self) -> $name {
				match self {
					#[allow(unused_variables)]
					Ok(this) => $name::Ok {
						$($field: this.$field,)*
					},
					$(Err(AsIs::<$err_path>::$err { $($err_field,)* }) => $name::$err {
						$($err_field,)*
					},)*
					#[allow(unreachable_patterns)]
					Err(_) => $name::Unknown,
				}
			}
		}
	)*};
}
macro_rules! cparams {
	({$(

		#[repr($path:path)]
		pub struct $name:ident {
			$($field:ident: $field_ty:ty),* $(,)?
		}

	)*}) => {$(
		#[repr(C)]
		#[derive(Copy, Clone)]
		pub struct $name {
			$($field: $field_ty,)*
		}

		funcs!({
			pub fn $name<T>() -> $name {
				let default: $path = faer::auto!(T);
				$name {
					$($field: default.$field.into(),)*
				}
			}
		});

		impl $name {
			#[allow(dead_code)]
			fn faer<T: ComplexField>(self) -> faer::Spec<$path, T> {
				<$path>::from(self).into()
			}
		}

		impl From<$name> for $path {
			#[allow(unused_variables)]
			fn from(value: $name) -> $path {
				$path {
					$($field: value.$field.into(),)*
					..faer::auto!(f32)
				}
			}
		}
		impl From<$path> for $name {
			#[allow(unused_variables)]
			fn from(value: $path) -> $name {
				$name {
					$($field: value.$field.into(),)*
				}
			}
		}
	)*};
}
#[cfg(feature = "linalg")]
pub mod linalg {
	use super::*;
	use faer::linalg as la;
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum ComputeSvdVectors {
		No,
		Thin,
		Full,
	}
	impl ComputeSvdVectors {
		fn faer(self) -> la::svd::ComputeSvdVectors {
			match self {
				ComputeSvdVectors::No => la::svd::ComputeSvdVectors::No,
				ComputeSvdVectors::Thin => la::svd::ComputeSvdVectors::Thin,
				ComputeSvdVectors::Full => la::svd::ComputeSvdVectors::Full,
			}
		}
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum ComputeEigenvectors {
		No,
		Yes,
	}
	impl ComputeEigenvectors {
		fn faer(self) -> la::evd::ComputeEigenvectors {
			match self {
				ComputeEigenvectors::No => la::evd::ComputeEigenvectors::No,
				ComputeEigenvectors::Yes => la::evd::ComputeEigenvectors::Yes,
			}
		}
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum PivotingStrategy {
		/// searches for the k-th pivot in the k-th column
		Partial,
		/// searches for the k-th pivot in the k-th column, as well as the tail of the diagonal of
		/// the matrix
		PartialDiag,
		/// searches for pivots that are locally optimal
		Rook,
		/// searches for pivots that are locally optimal, as well as the tail of the diagonal of the
		/// matrix
		RookDiag,

		/// searches for pivots that are globally optimal
		Full,
	}
	impl From<PivotingStrategy> for la::cholesky::lblt::factor::PivotingStrategy {
		fn from(value: PivotingStrategy) -> Self {
			match value {
				PivotingStrategy::Partial => la::cholesky::lblt::factor::PivotingStrategy::Partial,
				PivotingStrategy::PartialDiag => la::cholesky::lblt::factor::PivotingStrategy::PartialDiag,
				PivotingStrategy::Rook => la::cholesky::lblt::factor::PivotingStrategy::Rook,
				PivotingStrategy::RookDiag => la::cholesky::lblt::factor::PivotingStrategy::RookDiag,
				PivotingStrategy::Full => la::cholesky::lblt::factor::PivotingStrategy::Full,
			}
		}
	}
	impl From<la::cholesky::lblt::factor::PivotingStrategy> for PivotingStrategy {
		fn from(value: la::cholesky::lblt::factor::PivotingStrategy) -> Self {
			match value {
				la::cholesky::lblt::factor::PivotingStrategy::Partial => PivotingStrategy::Partial,
				la::cholesky::lblt::factor::PivotingStrategy::PartialDiag => PivotingStrategy::PartialDiag,
				la::cholesky::lblt::factor::PivotingStrategy::Rook => PivotingStrategy::Rook,
				la::cholesky::lblt::factor::PivotingStrategy::RookDiag => PivotingStrategy::RookDiag,
				la::cholesky::lblt::factor::PivotingStrategy::Full => PivotingStrategy::Full,
				_ => PivotingStrategy::Partial,
			}
		}
	}
	cerr!({
		#[ok(faer::linalg::cholesky::llt::factor::LltInfo)]
		#[err(faer::linalg::cholesky::llt::factor::LltError)]
		pub enum LltStatus {
			Ok { dynamic_regularization_count: usize },
			NonPositivePivot { index: usize },
		}
		#[ok(faer::linalg::cholesky::llt_pivoting::factor::PivLltInfo)]
		#[err(faer::linalg::cholesky::llt::factor::LltError)]
		pub enum PivLltStatus {
			Ok { rank: usize, transposition_count: usize },
			NonPositivePivot { index: usize },
		}
		#[ok(faer::linalg::cholesky::ldlt::factor::LdltInfo)]
		#[err(faer::linalg::cholesky::ldlt::factor::LdltError)]
		pub enum LdltStatus {
			Ok { dynamic_regularization_count: usize },
			ZeroPivot { index: usize },
		}
		#[ok(faer::linalg::cholesky::lblt::factor::LbltInfo)]
		#[err(Never)]
		pub enum LbltStatus {
			Ok { transposition_count: usize },
		}
		#[ok(faer::linalg::qr::no_pivoting::factor::QrInfo)]
		#[err(Never)]
		pub enum QrStatus {
			Ok { rank: usize },
		}
		#[ok(faer::linalg::qr::col_pivoting::factor::ColPivQrInfo)]
		#[err(Never)]
		pub enum ColPivQrStatus {
			Ok { transposition_count: usize },
		}
		#[ok(faer::linalg::lu::partial_pivoting::factor::PartialPivLuInfo)]
		#[err(Never)]
		pub enum PartialPivLuStatus {
			Ok { transposition_count: usize },
		}
		#[ok(faer::linalg::lu::full_pivoting::factor::FullPivLuInfo)]
		#[err(Never)]
		pub enum FullPivLuStatus {
			Ok { transposition_count: usize },
		}
	});
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum SvdStatus {
		Ok { padding: usize },
		NoConvergence { padding: usize },
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum EvdStatus {
		Ok { padding: usize },
		NoConvergence { padding: usize },
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub enum GevdStatus {
		Ok { padding: usize },
		NoConvergence { padding: usize },
	}
	impl Into<SvdStatus> for Result<(), la::svd::SvdError> {
		fn into(self) -> SvdStatus {
			match self {
				Ok(()) => SvdStatus::Ok { padding: 0 },
				Err(la::svd::SvdError::NoConvergence) => SvdStatus::NoConvergence { padding: 0 },
			}
		}
	}
	impl Into<EvdStatus> for Result<(), la::evd::EvdError> {
		fn into(self) -> EvdStatus {
			match self {
				Ok(()) => EvdStatus::Ok { padding: 0 },
				Err(la::evd::EvdError::NoConvergence) => EvdStatus::NoConvergence { padding: 0 },
			}
		}
	}
	impl Into<GevdStatus> for Result<(), la::gevd::GevdError> {
		fn into(self) -> GevdStatus {
			match self {
				Ok(()) => GevdStatus::Ok { padding: 0 },
				Err(la::gevd::GevdError::NoConvergence) => GevdStatus::NoConvergence { padding: 0 },
			}
		}
	}
	cparams!({
		#[repr(faer::linalg::cholesky::llt::factor::LltParams)]
		pub struct LltParams {
			recursion_threshold: usize,
			block_size: usize,
		}
		#[repr(faer::linalg::cholesky::llt_pivoting::factor::PivLltParams)]
		pub struct PivLltParams {
			block_size: usize,
		}
		#[repr(faer::linalg::cholesky::ldlt::factor::LdltParams)]
		pub struct LdltParams {
			recursion_threshold: usize,
			block_size: usize,
		}
		#[repr(faer::linalg::cholesky::lblt::factor::LbltParams)]
		pub struct LbltParams {
			pivoting: PivotingStrategy,
			par_threshold: usize,
			block_size: usize,
		}
		#[repr(faer::linalg::qr::no_pivoting::factor::QrParams)]
		pub struct QrParams {
			blocking_threshold: usize,
			par_threshold: usize,
		}
		#[repr(faer::linalg::qr::col_pivoting::factor::ColPivQrParams)]
		pub struct ColPivQrParams {
			blocking_threshold: usize,
			par_threshold: usize,
		}
		#[repr(faer::linalg::lu::partial_pivoting::factor::PartialPivLuParams)]
		pub struct PartialPivLuParams {
			recursion_threshold: usize,
			block_size: usize,
			par_threshold: usize,
		}
		#[repr(faer::linalg::lu::full_pivoting::factor::FullPivLuParams)]
		pub struct FullPivLuParams {
			par_threshold: usize,
		}
		#[repr(faer::linalg::svd::bidiag::BidiagParams)]
		pub struct BidiagParams {
			par_threshold: usize,
		}
		#[repr(faer::linalg::evd::tridiag::TridiagParams)]
		pub struct TridiagParams {
			par_threshold: usize,
		}
		#[repr(faer::linalg::evd::hessenberg::HessenbergParams)]
		pub struct HessenbergParams {
			par_threshold: usize,
			blocking_threshold: usize,
		}
		#[repr(faer::linalg::gevd::gen_hessenberg::GeneralizedHessenbergParams)]
		pub struct GeneralizedHessenbergParams {
			block_size: usize,
			blocking_threshold: usize,
		}
		#[repr(faer::linalg::evd::EvdFromSchurParams)]
		pub struct EvdFromSchurParams {
			recursion_threshold: usize,
		}
		#[repr(faer::linalg::evd::schur::SchurParams)]
		pub struct SchurParams {
			recommended_shift_count: extern "C" fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
			recommended_deflation_window: extern "C" fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
			blocking_threshold: usize,
			nibble_threshold: usize,
		}
		#[repr(faer::linalg::gevd::GeneralizedSchurParams)]
		pub struct GeneralizedSchurParams {
			relative_cost_estimate_of_shift_chase_to_matmul: extern "C" fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
			recommended_shift_count: extern "C" fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
			recommended_deflation_window: extern "C" fn(matrix_dimension: usize, active_block_dimension: usize) -> usize,
			blocking_threshold: usize,
			nibble_threshold: usize,
		}
		#[repr(faer::linalg::svd::SvdParams)]
		pub struct SvdParams {
			bidiag: BidiagParams,
			qr: QrParams,
			recursion_threshold: usize,
			qr_ratio_threshold: f64,
		}
		#[repr(faer::linalg::evd::SelfAdjointEvdParams)]
		pub struct SelfAdjointEvdParams {
			tridiag: TridiagParams,
			recursion_threshold: usize,
		}
		#[repr(faer::linalg::evd::EvdParams)]
		pub struct EvdParams {
			hessenberg: HessenbergParams,
			schur: SchurParams,
			evd_from_schur: EvdFromSchurParams,
		}
		#[repr(faer::linalg::gevd::GevdParams)]
		pub struct GevdParams {
			hessenberg: GeneralizedHessenbergParams,
			schur: GeneralizedSchurParams,
			evd_from_schur: GevdFromSchurParams,
		}
	});
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub struct GevdFromSchurParams {
		pub padding: usize,
	}
	impl GevdFromSchurParams {
		#[allow(dead_code)]
		fn faer<T: ComplexField>(self) -> faer::Spec<la::gevd::GevdFromSchurParams, T> {
			<la::gevd::GevdFromSchurParams>::from(self).into()
		}
	}
	impl From<GevdFromSchurParams> for la::gevd::GevdFromSchurParams {
		#[allow(unused_variables)]
		fn from(value: GevdFromSchurParams) -> la::gevd::GevdFromSchurParams {
			la::gevd::GevdFromSchurParams { ..faer::auto!(f32) }
		}
	}
	impl From<la::gevd::GevdFromSchurParams> for GevdFromSchurParams {
		#[allow(unused_variables)]
		fn from(_: la::gevd::GevdFromSchurParams) -> GevdFromSchurParams {
			GevdFromSchurParams { padding: 0 }
		}
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub struct LltRegularization {
		/// regularized value
		pub dynamic_regularization_delta: *const Real,
		/// regularization threshold
		pub dynamic_regularization_epsilon: *const Real,
	}
	impl LltRegularization {
		fn faer<T: RealField>(self) -> faer::linalg::cholesky::llt::factor::LltRegularization<T> {
			{
				faer::linalg::cholesky::llt::factor::LltRegularization {
					dynamic_regularization_delta: real(self.dynamic_regularization_delta),
					dynamic_regularization_epsilon: real(self.dynamic_regularization_epsilon),
				}
			}
		}
	}
	#[repr(C)]
	#[derive(Copy, Clone)]
	pub struct LdltRegularization {
		/// regularized value
		pub dynamic_regularization_delta: *const Real,
		/// regularization threshold
		pub dynamic_regularization_epsilon: *const Real,

		/// i8
		pub dynamic_regularization_signs: SliceMut,
	}
	impl LdltRegularization {
		fn faer<'a, T: RealField>(self) -> faer::linalg::cholesky::ldlt::factor::LdltRegularization<'a, T> {
			{
				faer::linalg::cholesky::ldlt::factor::LdltRegularization {
					dynamic_regularization_delta: real(self.dynamic_regularization_delta),
					dynamic_regularization_epsilon: real(self.dynamic_regularization_epsilon),
					dynamic_regularization_signs: if self.dynamic_regularization_signs.ptr.is_null() {
						None
					} else {
						Some(self.dynamic_regularization_signs.slice())
					},
				}
			}
		}
	}
	funcs!({
		pub fn matmul<T>(C: MatMut, accum: Accum, A: MatRef, B: MatRef, alpha: *const Scalar, par: Par) {
			la::matmul::matmul(C.faer::<T>(), accum.faer(), A.faer::<T>(), B.faer::<T>(), scalar(alpha), par.faer());
		}
		pub fn matmul_triangular<T>(
			C: MatMut,
			C_block: Block,
			accum: Accum,
			A: MatRef,
			A_block: Block,
			B: MatRef,
			B_block: Block,
			alpha: *const Scalar,
			par: Par,
		) {
			la::matmul::triangular::matmul(
				C.faer::<T>(),
				C_block.faer(),
				accum.faer(),
				A.faer::<T>(),
				A_block.faer(),
				B.faer::<T>(),
				B_block.faer(),
				scalar(alpha),
				par.faer(),
			);
		}
		// SOLVE
		pub fn solve_triangular_lower_in_place<T>(L: MatRef, L_conj: Conj, rhs: MatMut, par: Par) {
			la::triangular_solve::solve_lower_triangular_in_place_with_conj(L.faer::<T>(), L_conj.faer(), rhs.faer(), par.faer());
		}
		pub fn solve_triangular_upper_in_place<T>(U: MatRef, U_conj: Conj, rhs: MatMut, par: Par) {
			la::triangular_solve::solve_upper_triangular_in_place_with_conj(U.faer::<T>(), U_conj.faer(), rhs.faer(), par.faer());
		}
		pub fn solve_unit_triangular_lower_in_place<T>(L: MatRef, L_conj: Conj, rhs: MatMut, par: Par) {
			la::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(L.faer::<T>(), L_conj.faer(), rhs.faer(), par.faer());
		}
		pub fn solve_unit_triangular_upper_in_place<T>(U: MatRef, U_conj: Conj, rhs: MatMut, par: Par) {
			la::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(U.faer::<T>(), U_conj.faer(), rhs.faer(), par.faer());
		}
		// INVERSE
		pub fn inverse_triangular_lower_in_place<T>(L_inv: MatMut, L: MatRef, par: Par) {
			la::triangular_inverse::invert_lower_triangular(L_inv.faer::<T>(), L.faer(), par.faer());
		}
		pub fn inverse_triangular_upper_in_place<T>(L_inv: MatMut, L: MatRef, par: Par) {
			la::triangular_inverse::invert_upper_triangular(L_inv.faer::<T>(), L.faer(), par.faer());
		}
		pub fn inverse_unit_triangular_lower_in_place<T>(L_inv: MatMut, L: MatRef, par: Par) {
			la::triangular_inverse::invert_unit_lower_triangular(L_inv.faer::<T>(), L.faer(), par.faer());
		}
		pub fn inverse_unit_triangular_upper_in_place<T>(L_inv: MatMut, L: MatRef, par: Par) {
			la::triangular_inverse::invert_unit_upper_triangular(L_inv.faer::<T>(), L.faer(), par.faer());
		}
		// LLT
		pub fn llt_factor_in_place_scratch<T>(dim: usize, par: Par, params: LltParams) -> Layout {
			la::cholesky::llt::factor::cholesky_in_place_scratch::<T>(dim, par.faer(), params.faer()).into()
		}
		pub fn llt_factor_in_place<T>(A: MatMut, regularization: LltRegularization, par: Par, mem: MemAlloc, params: LltParams) -> LltStatus {
			la::cholesky::llt::factor::cholesky_in_place::<T>(A.faer(), regularization.faer(), par.faer(), mem.faer(), params.faer()).into()
		}
		pub fn llt_solve_in_place_scratch<T>(dim: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::cholesky::llt::solve::solve_in_place_scratch::<T>(dim, rhs_ncols, par.faer()).into()
		}
		pub fn llt_solve_in_place<T>(L: MatRef, A_conj: Conj, rhs: MatMut, par: Par, mem: MemAlloc) {
			la::cholesky::llt::solve::solve_in_place_with_conj(L.faer::<T>(), A_conj.faer(), rhs.faer(), par.faer(), mem.faer())
		}
		pub fn llt_reconstruct_scratch<T>(dim: usize, par: Par) -> Layout {
			la::cholesky::llt::reconstruct::reconstruct_scratch::<T>(dim, par.faer()).into()
		}
		pub fn llt_reconstruct<T>(A: MatMut, L: MatRef, par: Par, mem: MemAlloc) {
			la::cholesky::llt::reconstruct::reconstruct(A.faer::<T>(), L.faer(), par.faer(), mem.faer())
		}
		pub fn llt_inverse_scratch<T>(dim: usize, par: Par) -> Layout {
			la::cholesky::llt::inverse::inverse_scratch::<T>(dim, par.faer()).into()
		}
		pub fn llt_inverse<T>(A_inv: MatMut, L: MatRef, par: Par, mem: MemAlloc) {
			la::cholesky::llt::inverse::inverse(A_inv.faer::<T>(), L.faer(), par.faer(), mem.faer())
		}
		pub fn piv_llt_factor_in_place_scratch<I, T>(dim: usize, par: Par, params: PivLltParams) -> Layout {
			la::cholesky::llt_pivoting::factor::cholesky_in_place_scratch::<I, T>(dim, par.faer(), params.faer()).into()
		}
		pub fn piv_llt_factor_in_place<I, T>(
			A: MatMut,
			perm_fwd: SliceMut,
			perm_bwd: SliceMut,
			par: Par,
			mem: MemAlloc,
			params: PivLltParams,
		) -> PivLltStatus {
			la::cholesky::llt_pivoting::factor::cholesky_in_place::<I, T>(
				A.faer(),
				perm_fwd.slice(),
				perm_bwd.slice(),
				par.faer(),
				mem.faer(),
				params.faer(),
			)
			.map(|x| x.0)
			.into()
		}
		pub fn piv_llt_solve_in_place_scratch<I, T>(dim: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::cholesky::llt_pivoting::solve::solve_in_place_scratch::<I, T>(dim, rhs_ncols, par.faer()).into()
		}
		pub fn piv_llt_solve_in_place<I, T>(L: MatRef, perm_fwd: SliceRef, perm_bwd: SliceRef, A_conj: Conj, rhs: MatMut, par: Par, mem: MemAlloc) {
			la::cholesky::llt_pivoting::solve::solve_in_place_with_conj(
				L.faer::<T>(),
				faer::perm::PermRef::<I>::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn piv_llt_reconstruct_scratch<I, T>(dim: usize, par: Par) -> Layout {
			la::cholesky::llt_pivoting::reconstruct::reconstruct_scratch::<I, T>(dim, par.faer()).into()
		}
		pub fn piv_llt_reconstruct<I, T>(A: MatMut, L: MatRef, perm_fwd: SliceRef, perm_bwd: SliceRef, par: Par, mem: MemAlloc) {
			la::cholesky::llt_pivoting::reconstruct::reconstruct(
				A.faer::<T>(),
				L.faer(),
				faer::perm::PermRef::<I>::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn piv_llt_inverse_scratch<I, T>(dim: usize, par: Par) -> Layout {
			la::cholesky::llt_pivoting::inverse::inverse_scratch::<I, T>(dim, par.faer()).into()
		}
		pub fn piv_llt_inverse<I, T>(A_inv: MatMut, L: MatRef, perm_fwd: SliceRef, perm_bwd: SliceRef, par: Par, mem: MemAlloc) {
			la::cholesky::llt_pivoting::inverse::inverse(
				A_inv.faer::<T>(),
				L.faer(),
				faer::perm::PermRef::<I>::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		// LDLT
		pub fn ldlt_factor_in_place_scratch<T>(dim: usize, par: Par, params: LdltParams) -> Layout {
			la::cholesky::ldlt::factor::cholesky_in_place_scratch::<T>(dim, par.faer(), params.faer()).into()
		}
		pub fn ldlt_factor_in_place<T>(A: MatMut, regularization: LdltRegularization, par: Par, mem: MemAlloc, params: LdltParams) -> LdltStatus {
			la::cholesky::ldlt::factor::cholesky_in_place::<T>(A.faer(), regularization.faer(), par.faer(), mem.faer(), params.faer()).into()
		}
		pub fn ldlt_solve_in_place_scratch<T>(dim: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::cholesky::ldlt::solve::solve_in_place_scratch::<T>(dim, rhs_ncols, par.faer()).into()
		}
		pub fn ldlt_solve_in_place<T>(L: MatRef, D: VecRef, A_conj: Conj, rhs: MatMut, par: Par, mem: MemAlloc) {
			la::cholesky::ldlt::solve::solve_in_place_with_conj(L.faer::<T>(), D.diag(), A_conj.faer(), rhs.faer(), par.faer(), mem.faer())
		}
		pub fn ldlt_reconstruct_scratch<T>(dim: usize, par: Par) -> Layout {
			la::cholesky::ldlt::reconstruct::reconstruct_scratch::<T>(dim, par.faer()).into()
		}
		pub fn ldlt_reconstruct<T>(A: MatMut, L: MatRef, D: VecRef, par: Par, mem: MemAlloc) {
			la::cholesky::ldlt::reconstruct::reconstruct(A.faer::<T>(), L.faer(), D.diag(), par.faer(), mem.faer())
		}
		pub fn ldlt_inverse_scratch<T>(dim: usize, par: Par) -> Layout {
			la::cholesky::ldlt::inverse::inverse_scratch::<T>(dim, par.faer()).into()
		}
		pub fn ldlt_inverse<T>(A_inv: MatMut, L: MatRef, D: VecRef, par: Par, mem: MemAlloc) {
			la::cholesky::ldlt::inverse::inverse(A_inv.faer::<T>(), L.faer(), D.diag(), par.faer(), mem.faer())
		}
		// LBLT
		pub fn lblt_factor_in_place_scratch<I, T>(dim: usize, par: Par, params: LbltParams) -> Layout {
			la::cholesky::lblt::factor::cholesky_in_place_scratch::<I, T>(dim, par.faer(), params.faer()).into()
		}
		pub fn lblt_factor_in_place<I, T>(
			A: MatMut,
			subdiag: VecMut,
			perm_fwd: SliceMut,
			perm_bwd: SliceMut,
			par: Par,
			mem: MemAlloc,
			params: LbltParams,
		) -> LbltStatus {
			Ok(la::cholesky::lblt::factor::cholesky_in_place::<I, T>(
				A.faer(),
				subdiag.diag(),
				perm_fwd.slice(),
				perm_bwd.slice(),
				par.faer(),
				mem.faer(),
				params.faer(),
			)
			.0)
			.into()
		}
		pub fn lblt_solve_in_place_scratch<I, T>(dim: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::cholesky::lblt::solve::solve_in_place_scratch::<I, T>(dim, rhs_ncols, par.faer()).into()
		}
		pub fn lblt_solve_in_place<I, T>(
			L: MatRef,
			diag: VecRef,
			subdiag: VecRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::cholesky::lblt::solve::solve_in_place_with_conj::<I, T>(
				L.faer(),
				diag.diag(),
				subdiag.diag(),
				A_conj.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn lblt_reconstruct_scratch<I, T>(dim: usize, par: Par) -> Layout {
			la::cholesky::lblt::reconstruct::reconstruct_scratch::<I, T>(dim, par.faer()).into()
		}
		pub fn lblt_reconstruct<I, T>(
			A: MatMut,
			L: MatRef,
			diag: VecRef,
			subdiag: VecRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			par: Par,
			mem: MemAlloc,
		) {
			la::cholesky::lblt::reconstruct::reconstruct::<I, T>(
				A.faer(),
				L.faer(),
				diag.diag(),
				subdiag.diag(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn lblt_inverse_scratch<I, T>(dim: usize, par: Par) -> Layout {
			la::cholesky::lblt::inverse::inverse_scratch::<I, T>(dim, par.faer()).into()
		}
		pub fn lblt_inverse<I, T>(
			A_inv: MatMut,
			L: MatRef,
			diag: VecRef,
			subdiag: VecRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			par: Par,
			mem: MemAlloc,
		) {
			la::cholesky::lblt::inverse::inverse::<I, T>(
				A_inv.faer(),
				L.faer(),
				diag.diag(),
				subdiag.diag(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		// HOUSEHOLDER
		pub fn apply_householder_on_the_left_scratch<T>(dim: usize, block_size: usize, rhs_ncols: usize) -> Layout {
			la::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(dim, block_size, rhs_ncols).into()
		}
		pub fn apply_householder_on_the_left<T>(
			householder_basis: MatRef,
			householder_factor: MatRef,
			householder_conj: Conj,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj::<T>(
				householder_basis.faer(),
				householder_factor.faer(),
				householder_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn apply_householder_transpose_on_the_left_scratch<T>(dim: usize, block_size: usize, rhs_ncols: usize) -> Layout {
			la::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(dim, block_size, rhs_ncols).into()
		}
		pub fn apply_householder_transpose_on_the_left<T>(
			householder_basis: MatRef,
			householder_factor: MatRef,
			householder_conj: Conj,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj::<T>(
				householder_basis.faer(),
				householder_factor.faer(),
				householder_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn apply_householder_on_the_right_scratch<T>(dim: usize, block_size: usize, rhs_ncols: usize) -> Layout {
			la::householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(dim, block_size, rhs_ncols).into()
		}
		pub fn apply_householder_on_the_right<T>(
			householder_basis: MatRef,
			householder_factor: MatRef,
			householder_conj: Conj,
			lhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::householder::apply_block_householder_sequence_on_the_right_in_place_with_conj::<T>(
				householder_basis.faer(),
				householder_factor.faer(),
				householder_conj.faer(),
				lhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn apply_householder_transpose_on_the_right_scratch<T>(dim: usize, block_size: usize, rhs_ncols: usize) -> Layout {
			la::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_scratch::<T>(dim, block_size, rhs_ncols).into()
		}
		pub fn apply_householder_transpose_on_the_right<T>(
			householder_basis: MatRef,
			householder_factor: MatRef,
			householder_conj: Conj,
			lhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::householder::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj::<T>(
				householder_basis.faer(),
				householder_factor.faer(),
				householder_conj.faer(),
				lhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		// QR
		pub fn qr_recommended_block_size<T>(nrows: usize, ncols: usize) -> usize {
			la::qr::no_pivoting::factor::recommended_block_size::<T>(nrows, ncols)
		}
		pub fn qr_factor_in_place_scratch<T>(nrows: usize, ncols: usize, block_size: usize, par: Par, params: QrParams) -> Layout {
			la::qr::no_pivoting::factor::qr_in_place_scratch::<T>(nrows, ncols, block_size, par.faer(), params.faer()).into()
		}
		pub fn qr_factor_in_place<T>(A: MatMut, Q_coeff: MatMut, par: Par, mem: MemAlloc, params: QrParams) -> QrStatus {
			Ok(la::qr::no_pivoting::factor::qr_in_place::<T>(
				A.faer(),
				Q_coeff.faer(),
				par.faer(),
				mem.faer(),
				params.faer(),
			))
			.into()
		}
		pub fn qr_solve_in_place_scratch<T>(dim: usize, block_size: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::qr::no_pivoting::solve::solve_in_place_scratch::<T>(dim, block_size, rhs_ncols, par.faer()).into()
		}
		pub fn qr_solve_in_place<T>(Q_basis: MatRef, Q_coeff: MatRef, R: MatRef, A_conj: Conj, rhs: MatMut, par: Par, mem: MemAlloc) {
			la::qr::no_pivoting::solve::solve_in_place_with_conj::<T>(
				Q_basis.faer(),
				Q_coeff.faer(),
				R.faer(),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn qr_solve_transpose_in_place_scratch<T>(dim: usize, block_size: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::qr::no_pivoting::solve::solve_transpose_in_place_scratch::<T>(dim, block_size, rhs_ncols, par.faer()).into()
		}
		pub fn qr_solve_transpose_in_place<T>(Q_basis: MatRef, Q_coeff: MatRef, R: MatRef, A_conj: Conj, rhs: MatMut, par: Par, mem: MemAlloc) {
			la::qr::no_pivoting::solve::solve_transpose_in_place_with_conj::<T>(
				Q_basis.faer(),
				Q_coeff.faer(),
				R.faer(),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn qr_solve_lstsq_in_place_scratch<T>(nrows: usize, ncols: usize, block_size: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::qr::no_pivoting::solve::solve_lstsq_in_place_scratch::<T>(nrows, ncols, block_size, rhs_ncols, par.faer()).into()
		}
		pub fn qr_solve_lstsq_in_place<T>(Q_basis: MatRef, Q_coeff: MatRef, R: MatRef, A_conj: Conj, rhs: MatMut, par: Par, mem: MemAlloc) {
			la::qr::no_pivoting::solve::solve_lstsq_in_place_with_conj::<T>(
				Q_basis.faer(),
				Q_coeff.faer(),
				R.faer(),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn qr_reconstruct_scratch<T>(nrows: usize, ncols: usize, block_size: usize, par: Par) -> Layout {
			la::qr::no_pivoting::reconstruct::reconstruct_scratch::<T>(nrows, ncols, block_size, par.faer()).into()
		}
		pub fn qr_reconstruct<T>(A: MatMut, Q_basis: MatRef, Q_coeff: MatRef, R: MatRef, par: Par, mem: MemAlloc) {
			la::qr::no_pivoting::reconstruct::reconstruct::<T>(A.faer(), Q_basis.faer(), Q_coeff.faer(), R.faer(), par.faer(), mem.faer())
		}
		pub fn qr_inverse_scratch<T>(dim: usize, block_size: usize, par: Par) -> Layout {
			la::qr::no_pivoting::inverse::inverse_scratch::<T>(dim, block_size, par.faer()).into()
		}
		pub fn qr_inverse<T>(A: MatMut, Q_basis: MatRef, Q_coeff: MatRef, R: MatRef, par: Par, mem: MemAlloc) {
			la::qr::no_pivoting::inverse::inverse::<T>(A.faer(), Q_basis.faer(), Q_coeff.faer(), R.faer(), par.faer(), mem.faer())
		}
		pub fn colpiv_qr_factor_in_place_scratch<I, T>(nrows: usize, ncols: usize, block_size: usize, par: Par, params: ColPivQrParams) -> Layout {
			la::qr::col_pivoting::factor::qr_in_place_scratch::<I, T>(nrows, ncols, block_size, par.faer(), params.faer()).into()
		}
		pub fn colpiv_qr_factor_in_place<I, T>(
			A: MatMut,
			Q_coeff: MatMut,
			perm_fwd: SliceMut,
			perm_bwd: SliceMut,
			par: Par,
			mem: MemAlloc,
			params: ColPivQrParams,
		) -> ColPivQrStatus {
			Ok(la::qr::col_pivoting::factor::qr_in_place::<I, T>(
				A.faer(),
				Q_coeff.faer(),
				perm_fwd.slice(),
				perm_bwd.slice(),
				par.faer(),
				mem.faer(),
				params.faer(),
			)
			.0)
			.into()
		}
		pub fn colpiv_qr_solve_in_place_scratch<I, T>(dim: usize, block_size: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::qr::col_pivoting::solve::solve_in_place_scratch::<I, T>(dim, block_size, rhs_ncols, par.faer()).into()
		}
		pub fn colpiv_qr_solve_in_place<I, T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::qr::col_pivoting::solve::solve_in_place_with_conj::<I, T>(
				Q_basis.faer(),
				Q_coeff.faer(),
				R.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn colpiv_qr_solve_transpose_in_place_scratch<I, T>(dim: usize, block_size: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::qr::col_pivoting::solve::solve_transpose_in_place_scratch::<I, T>(dim, block_size, rhs_ncols, par.faer()).into()
		}
		pub fn colpiv_qr_solve_transpose_in_place<I, T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::qr::col_pivoting::solve::solve_transpose_in_place_with_conj::<I, T>(
				Q_basis.faer(),
				Q_coeff.faer(),
				R.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn colpiv_qr_solve_lstsq_in_place_scratch<I, T>(nrows: usize, ncols: usize, block_size: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::qr::col_pivoting::solve::solve_lstsq_in_place_scratch::<I, T>(nrows, ncols, block_size, rhs_ncols, par.faer()).into()
		}
		pub fn colpiv_qr_solve_lstsq_in_place<I, T>(
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::qr::col_pivoting::solve::solve_lstsq_in_place_with_conj::<I, T>(
				Q_basis.faer(),
				Q_coeff.faer(),
				R.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn colpiv_qr_reconstruct_scratch<I, T>(nrows: usize, ncols: usize, block_size: usize, par: Par) -> Layout {
			la::qr::col_pivoting::reconstruct::reconstruct_scratch::<I, T>(nrows, ncols, block_size, par.faer()).into()
		}
		pub fn colpiv_qr_reconstruct<I, T>(
			A: MatMut,
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			par: Par,
			mem: MemAlloc,
		) {
			la::qr::col_pivoting::reconstruct::reconstruct::<I, T>(
				A.faer(),
				Q_basis.faer(),
				Q_coeff.faer(),
				R.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn colpiv_qr_inverse_scratch<I, T>(dim: usize, block_size: usize, par: Par) -> Layout {
			la::qr::col_pivoting::inverse::inverse_scratch::<I, T>(dim, block_size, par.faer()).into()
		}
		pub fn colpiv_qr_inverse<I, T>(
			A: MatMut,
			Q_basis: MatRef,
			Q_coeff: MatRef,
			R: MatRef,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			par: Par,
			mem: MemAlloc,
		) {
			la::qr::col_pivoting::inverse::inverse::<I, T>(
				A.faer(),
				Q_basis.faer(),
				Q_coeff.faer(),
				R.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		// LU
		pub fn partial_piv_lu_factor_in_place_scratch<I, T>(dim: usize, block_size: usize, par: Par, params: PartialPivLuParams) -> Layout {
			la::lu::partial_pivoting::factor::lu_in_place_scratch::<I, T>(dim, block_size, par.faer(), params.faer()).into()
		}
		pub fn partial_piv_lu_factor_in_place<I, T>(
			A: MatMut,
			perm_fwd: SliceMut,
			perm_bwd: SliceMut,
			par: Par,
			mem: MemAlloc,
			params: PartialPivLuParams,
		) -> PartialPivLuStatus {
			Ok(la::lu::partial_pivoting::factor::lu_in_place::<I, T>(
				A.faer(),
				perm_fwd.slice(),
				perm_bwd.slice(),
				par.faer(),
				mem.faer(),
				params.faer(),
			)
			.0)
			.into()
		}
		pub fn partial_piv_lu_solve_in_place_scratch<I, T>(dim: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::lu::partial_pivoting::solve::solve_in_place_scratch::<I, T>(dim, rhs_ncols, par.faer()).into()
		}
		pub fn partial_piv_lu_solve_in_place<I, T>(
			L: MatRef,
			U: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::lu::partial_pivoting::solve::solve_in_place_with_conj::<I, T>(
				L.faer(),
				U.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn partial_piv_lu_solve_transpose_in_place_scratch<I, T>(dim: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<I, T>(dim, rhs_ncols, par.faer()).into()
		}
		pub fn partial_piv_lu_solve_transpose_in_place<I, T>(
			L: MatRef,
			U: MatRef,
			A_conj: Conj,
			perm_fwd: SliceRef,
			perm_bwd: SliceRef,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::lu::partial_pivoting::solve::solve_transpose_in_place_with_conj::<I, T>(
				L.faer(),
				U.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn partial_piv_lu_reconstruct_scratch<I, T>(nrows: usize, ncols: usize, par: Par) -> Layout {
			la::lu::partial_pivoting::reconstruct::reconstruct_scratch::<I, T>(nrows, ncols, par.faer()).into()
		}
		pub fn partial_piv_lu_reconstruct<I, T>(A: MatMut, L: MatRef, U: MatRef, perm_fwd: SliceRef, perm_bwd: SliceRef, par: Par, mem: MemAlloc) {
			la::lu::partial_pivoting::reconstruct::reconstruct::<I, T>(
				A.faer(),
				L.faer(),
				U.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn partial_piv_lu_inverse_scratch<I, T>(dim: usize, par: Par) -> Layout {
			la::lu::partial_pivoting::inverse::inverse_scratch::<I, T>(dim, par.faer()).into()
		}
		pub fn partial_piv_lu_inverse<I, T>(A: MatMut, L: MatRef, U: MatRef, perm_fwd: SliceRef, perm_bwd: SliceRef, par: Par, mem: MemAlloc) {
			la::lu::partial_pivoting::inverse::inverse::<I, T>(
				A.faer(),
				L.faer(),
				U.faer(),
				faer::perm::PermRef::new_unchecked(perm_fwd.slice(), perm_bwd.slice(), Ord::min(perm_fwd.len, perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn full_piv_lu_factor_in_place_scratch<I, T>(dim: usize, block_size: usize, par: Par, params: FullPivLuParams) -> Layout {
			la::lu::full_pivoting::factor::lu_in_place_scratch::<I, T>(dim, block_size, par.faer(), params.faer()).into()
		}
		pub fn full_piv_lu_factor_in_place<I, T>(
			A: MatMut,
			row_perm_fwd: SliceMut,
			row_perm_bwd: SliceMut,
			col_perm_fwd: SliceMut,
			col_perm_bwd: SliceMut,
			par: Par,
			mem: MemAlloc,
			params: FullPivLuParams,
		) -> FullPivLuStatus {
			Ok(la::lu::full_pivoting::factor::lu_in_place::<I, T>(
				A.faer(),
				row_perm_fwd.slice(),
				row_perm_bwd.slice(),
				col_perm_fwd.slice(),
				col_perm_bwd.slice(),
				par.faer(),
				mem.faer(),
				params.faer(),
			)
			.0)
			.into()
		}
		pub fn full_piv_lu_solve_in_place_scratch<I, T>(dim: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::lu::full_pivoting::solve::solve_in_place_scratch::<I, T>(dim, rhs_ncols, par.faer()).into()
		}
		pub fn full_piv_lu_solve_in_place<I, T>(
			L: MatRef,
			U: MatRef,
			A_conj: Conj,
			row_perm_fwd: SliceRef,
			row_perm_bwd: SliceRef,
			col_perm_fwd: SliceRef,
			col_perm_bwd: SliceRef,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::lu::full_pivoting::solve::solve_in_place_with_conj::<I, T>(
				L.faer(),
				U.faer(),
				faer::perm::PermRef::new_unchecked(row_perm_fwd.slice(), row_perm_bwd.slice(), Ord::min(row_perm_fwd.len, row_perm_bwd.len)),
				faer::perm::PermRef::new_unchecked(col_perm_fwd.slice(), col_perm_bwd.slice(), Ord::min(col_perm_fwd.len, col_perm_bwd.len)),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn full_piv_lu_solve_transpose_in_place_scratch<I, T>(dim: usize, rhs_ncols: usize, par: Par) -> Layout {
			la::lu::full_pivoting::solve::solve_transpose_in_place_scratch::<I, T>(dim, rhs_ncols, par.faer()).into()
		}
		pub fn full_piv_lu_solve_transpose_in_place<I, T>(
			L: MatRef,
			U: MatRef,
			A_conj: Conj,
			row_perm_fwd: SliceRef,
			row_perm_bwd: SliceRef,
			col_perm_fwd: SliceRef,
			col_perm_bwd: SliceRef,
			rhs: MatMut,
			par: Par,
			mem: MemAlloc,
		) {
			la::lu::full_pivoting::solve::solve_transpose_in_place_with_conj::<I, T>(
				L.faer(),
				U.faer(),
				faer::perm::PermRef::new_unchecked(row_perm_fwd.slice(), row_perm_bwd.slice(), Ord::min(row_perm_fwd.len, row_perm_bwd.len)),
				faer::perm::PermRef::new_unchecked(col_perm_fwd.slice(), col_perm_bwd.slice(), Ord::min(col_perm_fwd.len, col_perm_bwd.len)),
				A_conj.faer(),
				rhs.faer(),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn full_piv_lu_reconstruct_scratch<I, T>(nrows: usize, ncols: usize, par: Par) -> Layout {
			la::lu::full_pivoting::reconstruct::reconstruct_scratch::<I, T>(nrows, ncols, par.faer()).into()
		}
		pub fn full_piv_lu_reconstruct<I, T>(
			A: MatMut,
			L: MatRef,
			U: MatRef,
			row_perm_fwd: SliceRef,
			row_perm_bwd: SliceRef,
			col_perm_fwd: SliceRef,
			col_perm_bwd: SliceRef,
			par: Par,
			mem: MemAlloc,
		) {
			la::lu::full_pivoting::reconstruct::reconstruct::<I, T>(
				A.faer(),
				L.faer(),
				U.faer(),
				faer::perm::PermRef::new_unchecked(row_perm_fwd.slice(), row_perm_bwd.slice(), Ord::min(row_perm_fwd.len, row_perm_bwd.len)),
				faer::perm::PermRef::new_unchecked(col_perm_fwd.slice(), col_perm_bwd.slice(), Ord::min(col_perm_fwd.len, col_perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		pub fn full_piv_lu_inverse_scratch<I, T>(dim: usize, par: Par) -> Layout {
			la::lu::full_pivoting::inverse::inverse_scratch::<I, T>(dim, par.faer()).into()
		}
		pub fn full_piv_lu_inverse<I, T>(
			A: MatMut,
			L: MatRef,
			U: MatRef,
			row_perm_fwd: SliceRef,
			row_perm_bwd: SliceRef,
			col_perm_fwd: SliceRef,
			col_perm_bwd: SliceRef,
			par: Par,
			mem: MemAlloc,
		) {
			la::lu::full_pivoting::inverse::inverse::<I, T>(
				A.faer(),
				L.faer(),
				U.faer(),
				faer::perm::PermRef::new_unchecked(row_perm_fwd.slice(), row_perm_bwd.slice(), Ord::min(row_perm_fwd.len, row_perm_bwd.len)),
				faer::perm::PermRef::new_unchecked(col_perm_fwd.slice(), col_perm_bwd.slice(), Ord::min(col_perm_fwd.len, col_perm_bwd.len)),
				par.faer(),
				mem.faer(),
			)
		}
		// SVD
		pub fn svd_scratch<T>(
			nrows: usize,
			ncols: usize,
			compute_U: ComputeSvdVectors,
			compute_V: ComputeSvdVectors,
			par: Par,
			params: SvdParams,
		) -> Layout {
			la::svd::svd_scratch::<T>(nrows, ncols, compute_U.faer(), compute_V.faer(), par.faer(), params.faer()).into()
		}
		pub fn svd<T>(A: MatRef, U: MatMut, S: VecMut, V: MatMut, par: Par, mem: MemAlloc, params: SvdParams) -> SvdStatus {
			let U = if U.ncols == 0 { None } else { Some(U.faer()) };
			let V = if V.ncols == 0 { None } else { Some(V.faer()) };
			la::svd::svd::<T>(A.faer(), S.diag(), U, V, par.faer(), mem.faer(), params.faer()).into()
		}
		// EVD (self adjoint)
		pub fn self_adjoint_evd_scratch<T>(dim: usize, compute_U: ComputeEigenvectors, par: Par, params: SelfAdjointEvdParams) -> Layout {
			la::evd::self_adjoint_evd_scratch::<T>(dim, compute_U.faer(), par.faer(), params.faer()).into()
		}
		pub fn self_adjoint_evd<T>(A: MatRef, U: MatMut, S: VecMut, par: Par, mem: MemAlloc, params: SelfAdjointEvdParams) -> EvdStatus {
			let U = if U.ncols == 0 { None } else { Some(U.faer()) };
			la::evd::self_adjoint_evd::<T>(A.faer(), S.diag(), U, par.faer(), mem.faer(), params.faer()).into()
		}
		// EVD (general)
		pub fn evd_scratch<T>(
			dim: usize,
			compute_left: ComputeEigenvectors,
			compute_right: ComputeEigenvectors,
			par: Par,
			params: EvdParams,
		) -> Layout {
			la::evd::evd_scratch::<T>(dim, compute_left.faer(), compute_right.faer(), par.faer(), params.faer()).into()
		}
		pub fn evd<T>(A: MatRef, UL: MatMut, UR: MatMut, S: VecMut, S_im: VecMut, par: Par, mem: MemAlloc, params: EvdParams) -> EvdStatus {
			type R = <T as faer::traits::ComplexField>::Real;
			if const { <T as faer::traits::ComplexField>::IS_REAL } {
				let UL = if UL.ncols == 0 { None } else { Some(UL.faer()) };
				let UR = if UR.ncols == 0 { None } else { Some(UR.faer()) };
				la::evd::evd_real::<R>(A.faer(), S.diag(), S_im.diag(), UL, UR, par.faer(), mem.faer(), params.faer()).into()
			} else {
				let UL = if UL.ncols == 0 { None } else { Some(UL.faer()) };
				let UR = if UR.ncols == 0 { None } else { Some(UR.faer()) };
				la::evd::evd_cplx::<R>(A.faer(), S.diag(), UL, UR, par.faer(), mem.faer(), params.faer()).into()
			}
		}
		// EVD (general)
		pub fn generalized_evd_scratch<T>(
			dim: usize,
			compute_left: ComputeEigenvectors,
			compute_right: ComputeEigenvectors,
			par: Par,
			params: GevdParams,
		) -> Layout {
			la::gevd::gevd_scratch::<T>(dim, compute_left.faer(), compute_right.faer(), par.faer(), params.faer()).into()
		}
		pub fn generalized_evd<T>(
			A: MatMut,
			B: MatMut,
			UL: MatMut,
			UR: MatMut,
			alpha: VecMut,
			alpha_im: VecMut,
			beta: VecMut,
			par: Par,
			mem: MemAlloc,
			params: GevdParams,
		) -> GevdStatus {
			type R = <T as faer::traits::ComplexField>::Real;
			if const { <T as faer::traits::ComplexField>::IS_REAL } {
				let UL = if UL.ncols == 0 { None } else { Some(UL.faer()) };
				let UR = if UR.ncols == 0 { None } else { Some(UR.faer()) };
				la::gevd::gevd_real::<R>(
					A.faer(),
					B.faer(),
					alpha.diag(),
					alpha_im.diag(),
					beta.diag(),
					UL,
					UR,
					par.faer(),
					mem.faer(),
					params.faer(),
				)
				.into()
			} else {
				let UL = if UL.ncols == 0 { None } else { Some(UL.faer()) };
				let UR = if UR.ncols == 0 { None } else { Some(UR.faer()) };
				la::gevd::gevd_cplx::<R>(
					A.faer(),
					B.faer(),
					alpha.diag(),
					beta.diag(),
					UL,
					UR,
					par.faer(),
					mem.faer(),
					params.faer(),
				)
				.into()
			}
		}
	});
}
#[unsafe(no_mangle)]
pub extern "C" fn libfaer_v0_23_get_global_par() -> Par {
	match faer::get_global_parallelism() {
		faer::Par::Seq => Par {
			tag: ParTag::Seq,
			nthreads: 1,
		},
		faer::Par::Rayon(nthreads) => Par {
			tag: ParTag::Rayon,
			nthreads: nthreads.get(),
		},
	}
}
#[unsafe(no_mangle)]
pub extern "C" fn libfaer_v0_23_set_global_par(par: Par) {
	faer::set_global_parallelism(match par.tag {
		ParTag::Seq => faer::Par::Seq,
		ParTag::Rayon => faer::Par::rayon(par.nthreads),
	})
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libfaer_v0_23_alloc(size: usize, align: usize) -> *mut c_void {
	unsafe {
		std::alloc::alloc(match core::alloc::Layout::from_size_align(size, align) {
			Ok(layout) => layout,
			Err(_) => return core::ptr::null_mut(),
		}) as *mut c_void
	}
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn libfaer_v0_23_dealloc(ptr: *mut c_void, size: usize, align: usize) {
	unsafe { std::alloc::dealloc(ptr as *mut u8, core::alloc::Layout::from_size_align_unchecked(size, align)) }
}
