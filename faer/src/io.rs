use crate::prelude::*;

/// npy format conversions
#[cfg(feature = "npy")]
pub mod npy {
	use super::*;

	/// memory view over a buffer in `npy` format
	pub struct Npy<'a> {
		aligned_bytes: &'a [u8],
		nrows: usize,
		ncols: usize,
		prefix_len: usize,
		dtype: NpyDType,
		fortran_order: bool,
	}

	/// data type of an `npy` buffer
	#[derive(Debug, Copy, Clone, PartialEq, Eq)]
	pub enum NpyDType {
		/// 32-bit floating point
		F32,
		/// 64-bit floating point
		F64,
		/// 32-bit complex floating point
		C32,
		/// 64-bit complex floating point
		C64,
		/// unknown type
		Other,
	}

	/// trait implemented for native types that can be read from a `npy` buffer
	pub trait FromNpy: bytemuck::Pod {
		/// data type of the buffer data
		const DTYPE: NpyDType;
	}

	impl FromNpy for f32 {
		const DTYPE: NpyDType = NpyDType::F32;
	}
	impl FromNpy for f64 {
		const DTYPE: NpyDType = NpyDType::F64;
	}
	impl FromNpy for c32 {
		const DTYPE: NpyDType = NpyDType::C32;
	}
	impl FromNpy for c64 {
		const DTYPE: NpyDType = NpyDType::C64;
	}

	impl<'a> Npy<'a> {
		fn parse_npyz(data: &[u8], npyz: npyz::NpyFile<&[u8]>) -> Result<(NpyDType, usize, usize, usize, bool), std::io::Error> {
			let ver_major = data[6] - b'\x00';
			let length = if ver_major <= 1 {
				2usize
			} else if ver_major <= 3 {
				4usize
			} else {
				return Err(std::io::Error::new(std::io::ErrorKind::Other, "unsupported version"));
			};
			let header_len = if length == 2 {
				u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize
			} else {
				u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize
			};
			let dtype = || -> NpyDType {
				match npyz.dtype() {
					npyz::DType::Plain(str) => {
						let is_complex = match str.type_char() {
							npyz::TypeChar::Float => false,
							npyz::TypeChar::Complex => true,
							_ => return NpyDType::Other,
						};

						let byte_size = str.size_field();
						if byte_size == 8 && is_complex {
							NpyDType::C32
						} else if byte_size == 16 && is_complex {
							NpyDType::C64
						} else if byte_size == 4 && !is_complex {
							NpyDType::F32
						} else if byte_size == 16 && !is_complex {
							NpyDType::F64
						} else {
							NpyDType::Other
						}
					},
					_ => NpyDType::Other,
				}
			};

			let dtype = dtype();
			let order = npyz.header().order();
			let shape = npyz.shape();
			let nrows = shape.get(0).copied().unwrap_or(1) as usize;
			let ncols = shape.get(1).copied().unwrap_or(1) as usize;
			let prefix_len = 8 + length + header_len;
			let fortran_order = order == npyz::Order::Fortran;
			Ok((dtype, nrows, ncols, prefix_len, fortran_order))
		}

		/// parse a npy file from a memory buffer
		#[inline]
		pub fn new(data: &'a [u8]) -> Result<Self, std::io::Error> {
			let npyz = npyz::NpyFile::new(data)?;

			let (dtype, nrows, ncols, prefix_len, fortran_order) = Self::parse_npyz(data, npyz)?;

			Ok(Self {
				aligned_bytes: data,
				prefix_len,
				nrows,
				ncols,
				dtype,
				fortran_order,
			})
		}

		/// returns the data type of the memory buffer
		#[inline]
		pub fn dtype(&self) -> NpyDType {
			self.dtype
		}

		/// checks if the memory buffer is aligned, in which case the data can be referenced
		/// in-place
		#[inline]
		pub fn is_aligned(&self) -> bool {
			self.aligned_bytes.as_ptr().align_offset(64) == 0
		}

		/// if the memory buffer is aligned, and the provided type matches the one stored in the
		/// buffer, returns a matrix view over the data
		#[inline]
		pub fn as_aligned_ref<T: FromNpy>(&self) -> MatRef<'_, T> {
			assert!(self.is_aligned());
			assert!(self.dtype == T::DTYPE);

			if self.fortran_order {
				MatRef::from_column_major_slice(bytemuck::cast_slice(&self.aligned_bytes[self.prefix_len..]), self.nrows, self.ncols)
			} else {
				MatRef::from_row_major_slice(bytemuck::cast_slice(&self.aligned_bytes[self.prefix_len..]), self.nrows, self.ncols)
			}
		}

		/// if the provided type matches the one stored in the buffer, returns a matrix containing
		/// the data
		#[inline]
		pub fn to_mat<T: FromNpy>(&self) -> Mat<T> {
			assert!(self.dtype == T::DTYPE);

			let mut mat = Mat::<T>::with_capacity(self.nrows, self.ncols);
			unsafe { mat.set_dims(self.nrows, self.ncols) };

			let data = &self.aligned_bytes[self.prefix_len..];

			if self.fortran_order {
				for j in 0..self.ncols {
					bytemuck::cast_slice_mut(mat.col_as_slice_mut(j))
						.copy_from_slice(&data[j * self.nrows * core::mem::size_of::<T>()..][..self.nrows * core::mem::size_of::<T>()])
				}
			} else {
				for j in 0..self.ncols {
					for i in 0..self.nrows {
						mat[(i, j)] =
							bytemuck::cast_slice::<u8, T>(&data[(i * self.ncols + j) * core::mem::size_of::<T>()..][..core::mem::size_of::<T>()])[0];
					}
				}
			};

			mat
		}
	}
}
