//! Serde implementations for Mat

use faer_entity::Entity;
use serde::{Deserialize, Serialize, Serializer};

use crate::Mat;

#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
struct SerdeMat<E: Entity> {
    col_length: usize,
    data: Vec<E>,
}

impl<E: Entity> Into<Mat<E>> for SerdeMat<E> {
    fn into(self) -> Mat<E> {
        let row_length = self.data.len() / self.col_length;
        Mat::from_fn(row_length, self.col_length, |i, j| {
            self.data[j + (i * self.col_length)]
        })
    }
}

impl<E: Entity> Into<SerdeMat<E>> for &Mat<E> {
    fn into(self) -> SerdeMat<E> {
        let mut data = Vec::with_capacity(self.nrows() * self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                data.push(self.read(i, j));
            }
        }

        SerdeMat {
            col_length: self.ncols(),
            data,
        }
    }
}

impl<E: Entity> Serialize for Mat<E>
where
    E: Serialize,
{
    fn serialize<S>(&self, s: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        let mat: SerdeMat<E> = self.into();
        mat.serialize(s)
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
        let mat: SerdeMat<E> = SerdeMat::<E>::deserialize(d)?;
        if mat.data.len() % mat.col_length != 0 {
            return Err(serde::de::Error::custom(
                "serialized matrix is now valid as col_length isnt a divisor of the length of the data",
            ));
        }
        Ok(mat.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_test::{assert_de_tokens_error, assert_tokens, Token};
    #[test]
    fn mat_to_serde_mat_and_back_again() {
        let mat = Mat::from_fn(3, 4, |i, j| (i + (j * 10)) as f64);
        let serde_mat: SerdeMat<f64> = (&mat).into();
        println!("{:?}", serde_mat);
        let mat_again: Mat<f64> = serde_mat.into();
        assert_eq!(mat_again, mat);
    }

    #[test]
    fn mat_serialization() {
        let mat = Mat::from_fn(3, 4, |i, j| (i + (j * 10)) as f64);
        assert_tokens(
            &mat,
            &[
                Token::Struct {
                    len: 2,
                    name: "SerdeMat",
                },
                Token::Str("col_length"),
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
    fn deserialization_can_fail() {
        assert_de_tokens_error::<Mat<f64>>(
            &[
                Token::Struct {
                    len: 2,
                    name: "SerdeMat",
                },
                Token::Str("col_length"),
                Token::U64(3),
                Token::Str("data"),
                Token::Seq { len: Some(12) },
                Token::F64(0.0),
                Token::F64(1.0),
                Token::F64(2.0),
                Token::F64(3.0),
                Token::SeqEnd,
                Token::StructEnd,
            ],
            "serialized matrix is now valid as col_length isnt a divisor of the length of the data",
        )
    }
}
