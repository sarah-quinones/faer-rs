//! Serde implementations for Mat

use core::marker::PhantomData;

use faer_entity::Entity;
use serde::{
    de::{DeserializeSeed, SeqAccess, Visitor},
    ser::{SerializeSeq, SerializeStruct},
    Deserialize, Serialize, Serializer,
};

use crate::Mat;

impl<E: Entity> Serialize for Mat<E>
where
    E: Serialize,
{
    fn serialize<S>(&self, s: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        struct MatSequenceSerializer<'a, E: Entity>(&'a Mat<E>);

        impl<'a, E: Entity> Serialize for MatSequenceSerializer<'a, E>
        where
            E: Serialize,
        {
            fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let mut seq = s.serialize_seq(Some(self.0.nrows() * self.0.ncols()))?;
                for i in 0..self.0.nrows() {
                    for j in 0..self.0.ncols() {
                        seq.serialize_element(&self.0.read(i, j))?;
                    }
                }
                seq.end()
            }
        }

        let mut structure = s.serialize_struct("Mat", 3)?;
        structure.serialize_field("nrows", &self.nrows())?;
        structure.serialize_field("ncols", &self.ncols())?;
        structure.serialize_field("data", &MatSequenceSerializer(self))?;
        structure.end()
    }
}

impl<'a, E: Entity> Deserialize<'a> for Mat<E>
where
    E: Deserialize<'a>,
{
    fn deserialize<D>(d: D) -> Result<Self, <D as serde::Deserializer<'a>>::Error>
    where
        D: serde::Deserializer<'a>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Nrows,
            Ncols,
            Data,
        }
        const FIELDS: &'static [&'static str] = &["nrows", "ncols", "data"];
        struct MatVisitor<E: Entity>(PhantomData<E>);
        impl<'a, E: Entity + Deserialize<'a>> Visitor<'a> for MatVisitor<E> {
            type Value = Mat<E>;

            fn expecting(&self, formatter: &mut alloc::fmt::Formatter) -> alloc::fmt::Result {
                formatter.write_str("a faer matrix")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'a>,
            {
                enum MatrixOrVec<E: Entity> {
                    Matrix(Mat<E>),
                    Vec(Vec<E>),
                }
                impl<E: Entity> MatrixOrVec<E> {
                    fn into_mat(self, nrows: usize, ncols: usize) -> Mat<E> {
                        match self {
                            MatrixOrVec::Matrix(m) => m,
                            MatrixOrVec::Vec(v) => {
                                Mat::from_fn(nrows, ncols, |i, j| v[i * ncols + j])
                            }
                        }
                    }
                }
                struct MatrixOrVecDeserializer<'a, E: Entity + Deserialize<'a>> {
                    marker: PhantomData<&'a E>,
                    nrows: Option<usize>,
                    ncols: Option<usize>,
                }
                impl<'a, E: Entity + Deserialize<'a>> MatrixOrVecDeserializer<'a, E> {
                    fn new(nrows: Option<usize>, ncols: Option<usize>) -> Self {
                        Self {
                            marker: PhantomData,
                            nrows,
                            ncols,
                        }
                    }
                }
                impl<'a, E: Entity> DeserializeSeed<'a> for MatrixOrVecDeserializer<'a, E>
                where
                    E: Deserialize<'a>,
                {
                    type Value = MatrixOrVec<E>;

                    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
                    where
                        D: serde::Deserializer<'a>,
                    {
                        deserializer.deserialize_seq(self)
                    }
                }
                impl<'a, E: Entity> Visitor<'a> for MatrixOrVecDeserializer<'a, E>
                where
                    E: Deserialize<'a>,
                {
                    type Value = MatrixOrVec<E>;

                    fn expecting(
                        &self,
                        formatter: &mut alloc::fmt::Formatter,
                    ) -> alloc::fmt::Result {
                        formatter.write_str("a sequence")
                    }

                    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                    where
                        A: SeqAccess<'a>,
                    {
                        match (self.ncols, self.nrows) {
                            (Some(ncols), Some(nrows)) => {
                                let ncols = ncols;
                                let nrows = nrows;
                                let mut data = Mat::<E>::with_capacity(nrows, ncols);
                                unsafe {
                                    data.set_dims(nrows, ncols);
                                }
                                let mut i = 0;
                                while let Some(el) = seq.next_element::<E>()? {
                                    data.write(i / ncols, i % ncols, el);
                                    i += 1;
                                }
                                if i < nrows * ncols {
                                    return Err(serde::de::Error::invalid_length(
                                        i,
                                        &format!("{} elements", nrows * ncols).as_str(),
                                    ));
                                }
                                Ok(MatrixOrVec::Matrix(data))
                            }
                            _ => {
                                let mut data = Vec::new();
                                while let Some(el) = seq.next_element::<E>()? {
                                    data.push(el);
                                }
                                Ok(MatrixOrVec::Vec(data))
                            }
                        }
                    }
                }
                let mut nrows = None;
                let mut ncols = None;
                let mut data: Option<MatrixOrVec<E>> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Nrows => {
                            if nrows.is_some() {
                                return Err(serde::de::Error::duplicate_field("nrows"));
                            }
                            let value = map.next_value()?;
                            nrows = Some(value);
                        }
                        Field::Ncols => {
                            if ncols.is_some() {
                                return Err(serde::de::Error::duplicate_field("ncols"));
                            }
                            let value = map.next_value()?;
                            ncols = Some(value);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(serde::de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value_seed(MatrixOrVecDeserializer::<E>::new(
                                nrows.clone(),
                                ncols.clone(),
                            ))?);
                        }
                    }
                }
                let nrows = nrows.ok_or_else(|| serde::de::Error::missing_field("nrows"))?;
                let ncols = ncols.ok_or_else(|| serde::de::Error::missing_field("ncols"))?;
                let data = data
                    .ok_or_else(|| serde::de::Error::missing_field("data"))?
                    .into_mat(nrows, ncols);
                Ok(data)
            }
        }
        d.deserialize_struct("Mat", FIELDS, MatVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_test::{assert_de_tokens_error, assert_tokens, Token};
    #[test]
    fn matrix_serialization_normal() {
        let value = Mat::from_fn(3, 4, |i, j| (i + (j * 10)) as f64);
        assert_tokens(
            &value,
            &[
                Token::Struct {
                    name: "Mat",
                    len: 3,
                },
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
                Token::Struct {
                    name: "Mat",
                    len: 3,
                },
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
                Token::Struct {
                    name: "Mat",
                    len: 3,
                },
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
                Token::Struct {
                    name: "Mat",
                    len: 3,
                },
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
    fn matrix_serialization_errors_with_too_smaller_size() {
        assert_de_tokens_error::<Mat<f64>>(
            &[
                Token::Struct {
                    name: "Mat",
                    len: 3,
                },
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
}
