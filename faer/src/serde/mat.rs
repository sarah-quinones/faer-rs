use crate::internal_prelude::*;
use core::marker::PhantomData;
use serde::de::{DeserializeSeed, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};

impl<T> Serialize for MatRef<'_, T>
where
	T: Serialize,
{
	fn serialize<S>(&self, s: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
	where
		S: Serializer,
	{
		struct MatSequenceSerializer<'a, T>(MatRef<'a, T>);

		impl<'a, T> Serialize for MatSequenceSerializer<'a, T>
		where
			T: Serialize,
		{
			fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
			where
				S: Serializer,
			{
				let mut seq = s.serialize_seq(Some(self.0.nrows() * self.0.ncols()))?;

				for i in 0..self.0.nrows() {
					for j in 0..self.0.ncols() {
						seq.serialize_element(&self.0[(i, j)])?;
					}
				}

				seq.end()
			}
		}

		let mut structure = s.serialize_struct("Mat", 3)?;

		structure.serialize_field("nrows", &self.nrows())?;

		structure.serialize_field("ncols", &self.ncols())?;

		structure.serialize_field("data", &MatSequenceSerializer(*self))?;

		structure.end()
	}
}

impl<T> Serialize for MatMut<'_, T>
where
	T: Serialize,
{
	fn serialize<S>(&self, s: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
	where
		S: Serializer,
	{
		self.as_ref().serialize(s)
	}
}

impl<T> Serialize for Mat<T>
where
	T: Serialize,
{
	fn serialize<S>(&self, s: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
	where
		S: Serializer,
	{
		self.as_ref().serialize(s)
	}
}

impl<'a, T: 'a + Deserialize<'a>> Deserialize<'a> for Mat<T> {
	fn deserialize<D>(d: D) -> Result<Self, <D as serde::Deserializer<'a>>::Error>
	where
		D: serde::Deserializer<'a>,
	{
		#[derive(Deserialize)]
		#[allow(non_camel_case_types)]

		enum Field {
			nrows,
			ncols,
			data,
		}

		const FIELDS: &'static [&'static str] = &["nrows", "ncols", "data"];

		struct MatVisitor<T>(PhantomData<T>);

		enum MatrixOrVec<T> {
			Matrix(Mat<T>),
			Vec(alloc::vec::Vec<T>),
		}

		impl<T> MatrixOrVec<T> {
			fn into_mat(self, nrows: usize, ncols: usize) -> Mat<T> {
				match self {
					MatrixOrVec::Matrix(m) => m,
					MatrixOrVec::Vec(mut v) => {
						let me = Mat::from_fn(nrows, ncols, |i, j| unsafe { core::ptr::read(&v[i * ncols + j]) });

						unsafe { v.set_len(0) };

						me
					},
				}
			}
		}

		struct MatrixOrVecDeserializer<'a, T: Deserialize<'a>> {
			marker: PhantomData<&'a T>,
			nrows: Option<usize>,
			ncols: Option<usize>,
		}

		impl<'a, T: Deserialize<'a>> MatrixOrVecDeserializer<'a, T> {
			fn new(nrows: Option<usize>, ncols: Option<usize>) -> Self {
				Self {
					marker: PhantomData,
					nrows,
					ncols,
				}
			}
		}

		impl<'a, T> DeserializeSeed<'a> for MatrixOrVecDeserializer<'a, T>
		where
			T: Deserialize<'a>,
		{
			type Value = MatrixOrVec<T>;

			fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
			where
				D: serde::Deserializer<'a>,
			{
				deserializer.deserialize_seq(self)
			}
		}

		impl<'a, T> Visitor<'a> for MatrixOrVecDeserializer<'a, T>
		where
			T: Deserialize<'a>,
		{
			type Value = MatrixOrVec<T>;

			fn expecting(&self, formatter: &mut alloc::fmt::Formatter) -> alloc::fmt::Result {
				formatter.write_str("a sequence")
			}

			fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
			where
				A: SeqAccess<'a>,
			{
				match (self.ncols, self.nrows) {
					(Some(ncols), Some(nrows)) => {
						let mut data = Mat::<T>::with_capacity(nrows, ncols);

						let expected_length = nrows * ncols;

						{
							let stride = data.col_stride() as usize;

							let data = data.as_ptr_mut();

							for i in 0..expected_length {
								let el = seq
									.next_element::<T>()?
									.ok_or_else(|| serde::de::Error::invalid_length(i, &alloc::format!("{} elements", expected_length).as_str()))?;

								let (i, j) = (i / ncols, i % ncols);

								unsafe { data.add(i + j * stride).write(el) };
							}
						}

						unsafe {
							data.set_dims(nrows, ncols);
						}

						let mut additional = 0usize;

						while let Some(_) = seq.next_element::<T>()? {
							additional += 1;
						}

						if additional > 0 {
							return Err(serde::de::Error::invalid_length(
								additional + expected_length,
								&alloc::format!("{} elements", expected_length).as_str(),
							));
						}

						Ok(MatrixOrVec::Matrix(data))
					},
					_ => {
						let mut data = alloc::vec::Vec::new();

						while let Some(el) = seq.next_element::<T>()? {
							data.push(el);
						}

						Ok(MatrixOrVec::Vec(data))
					},
				}
			}
		}

		impl<'a, T: 'a + Deserialize<'a>> Visitor<'a> for MatVisitor<T> {
			type Value = Mat<T>;

			fn expecting(&self, formatter: &mut alloc::fmt::Formatter) -> alloc::fmt::Result {
				formatter.write_str("a faer matrix")
			}

			fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
			where
				A: SeqAccess<'a>,
			{
				let nrows = seq
					.next_element::<usize>()?
					.ok_or_else(|| serde::de::Error::invalid_length(0, &"nrows"))?;

				let ncols = seq
					.next_element::<usize>()?
					.ok_or_else(|| serde::de::Error::invalid_length(1, &"ncols"))?;

				let data = seq.next_element_seed(MatrixOrVecDeserializer::<T>::new(Some(nrows), Some(ncols)))?;

				let mat = data.ok_or_else(|| serde::de::Error::missing_field("data"))?.into_mat(nrows, ncols);

				Ok(mat)
			}

			fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
			where
				A: serde::de::MapAccess<'a>,
			{
				let mut nrows = None;

				let mut ncols = None;

				let mut data: Option<MatrixOrVec<T>> = None;

				while let Some(key) = map.next_key()? {
					match key {
						Field::nrows => {
							if nrows.is_some() {
								return Err(serde::de::Error::duplicate_field("nrows"));
							}

							let value = map.next_value()?;

							nrows = Some(value);
						},
						Field::ncols => {
							if ncols.is_some() {
								return Err(serde::de::Error::duplicate_field("ncols"));
							}

							let value = map.next_value()?;

							ncols = Some(value);
						},
						Field::data => {
							if data.is_some() {
								return Err(serde::de::Error::duplicate_field("data"));
							}

							data = Some(map.next_value_seed(MatrixOrVecDeserializer::<T>::new(nrows.clone(), ncols.clone()))?);
						},
					}
				}

				let nrows = nrows.ok_or_else(|| serde::de::Error::missing_field("nrows"))?;

				let ncols = ncols.ok_or_else(|| serde::de::Error::missing_field("ncols"))?;

				let data = data.ok_or_else(|| serde::de::Error::missing_field("data"))?.into_mat(nrows, ncols);

				Ok(data)
			}
		}

		d.deserialize_struct("Mat", FIELDS, MatVisitor(PhantomData))
	}
}

#[cfg(test)]

mod tests {

	use super::*;
	use serde_test::{Token, assert_de_tokens_error, assert_tokens};

	#[test]

	fn matrix_serialization_normal() {
		let value = Mat::from_fn(3, 4, |i, j| (i + (j * 10)) as f64);

		assert_tokens(
			&value,
			&[
				Token::Struct { name: "Mat", len: 3 },
				Token::Str("nrows"),
				Token::U64(3),
				Token::Str("ncols"),
				Token::U64(4),
				Token::Str("data"),
				Token::Seq { len: Some(12) },
				Token::F64(0.0),
				Token::F64(10.0),
				Token::F64(20.0),
				Token::F64(30.0),
				Token::F64(1.0),
				Token::F64(11.0),
				Token::F64(21.0),
				Token::F64(31.0),
				Token::F64(2.0),
				Token::F64(12.0),
				Token::F64(22.0),
				Token::F64(32.0),
				Token::SeqEnd,
				Token::StructEnd,
			],
		)
	}

	#[test]

	fn matrix_serialization_wide() {
		let value = Mat::from_fn(12, 1, |i, j| (i + (j * 10)) as f64);

		assert_tokens(
			&value,
			&[
				Token::Struct { name: "Mat", len: 3 },
				Token::Str("nrows"),
				Token::U64(12),
				Token::Str("ncols"),
				Token::U64(1),
				Token::Str("data"),
				Token::Seq { len: Some(12) },
				Token::F64(0.0),
				Token::F64(1.0),
				Token::F64(2.0),
				Token::F64(3.0),
				Token::F64(4.0),
				Token::F64(5.0),
				Token::F64(6.0),
				Token::F64(7.0),
				Token::F64(8.0),
				Token::F64(9.0),
				Token::F64(10.0),
				Token::F64(11.0),
				Token::SeqEnd,
				Token::StructEnd,
			],
		)
	}

	#[test]

	fn matrix_serialization_tall() {
		let value = Mat::from_fn(1, 12, |i, j| (i + (j * 10)) as f64);

		assert_tokens(
			&value,
			&[
				Token::Struct { name: "Mat", len: 3 },
				Token::Str("nrows"),
				Token::U64(1),
				Token::Str("ncols"),
				Token::U64(12),
				Token::Str("data"),
				Token::Seq { len: Some(12) },
				Token::F64(0.0),
				Token::F64(10.0),
				Token::F64(20.0),
				Token::F64(30.0),
				Token::F64(40.0),
				Token::F64(50.0),
				Token::F64(60.0),
				Token::F64(70.0),
				Token::F64(80.0),
				Token::F64(90.0),
				Token::F64(100.0),
				Token::F64(110.0),
				Token::SeqEnd,
				Token::StructEnd,
			],
		)
	}

	#[test]

	fn matrix_serialization_zero() {
		let value = Mat::from_fn(0, 0, |i, j| (i + (j * 10)) as f64);

		assert_tokens(
			&value,
			&[
				Token::Struct { name: "Mat", len: 3 },
				Token::Str("nrows"),
				Token::U64(0),
				Token::Str("ncols"),
				Token::U64(0),
				Token::Str("data"),
				Token::Seq { len: Some(0) },
				Token::SeqEnd,
				Token::StructEnd,
			],
		)
	}

	#[test]

	fn matrix_serialization_errors_too_small() {
		assert_de_tokens_error::<Mat<f64>>(
			&[
				Token::Struct { name: "Mat", len: 3 },
				Token::Str("nrows"),
				Token::U64(3),
				Token::Str("ncols"),
				Token::U64(4),
				Token::Str("data"),
				Token::Seq { len: Some(12) },
				Token::F64(0.0),
				Token::F64(10.0),
				Token::F64(20.0),
				Token::F64(30.0),
				Token::F64(1.0),
				Token::F64(11.0),
				Token::F64(21.0),
				Token::F64(31.0),
				Token::F64(2.0),
				Token::SeqEnd,
			],
			"invalid length 9, expected 12 elements",
		)
	}

	#[test]

	fn matrix_serialization_errors_too_large() {
		assert_de_tokens_error::<Mat<f64>>(
			&[
				Token::Struct { name: "Mat", len: 3 },
				Token::Str("nrows"),
				Token::U64(3),
				Token::Str("ncols"),
				Token::U64(4),
				Token::Str("data"),
				Token::Seq { len: Some(12) },
				Token::F64(0.0),
				Token::F64(10.0),
				Token::F64(20.0),
				Token::F64(30.0),
				Token::F64(1.0),
				Token::F64(11.0),
				Token::F64(21.0),
				Token::F64(31.0),
				Token::F64(2.0),
				Token::F64(12.0),
				Token::F64(22.0),
				Token::F64(32.0),
				Token::F64(32.0),
				Token::F64(32.0),
				Token::SeqEnd,
			],
			"invalid length 14, expected 12 elements",
		)
	}
}
