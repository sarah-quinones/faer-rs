use crate::{Col, ComplexField, Mat, Row};
use rand::distributions::Distribution;
use rand_distr::{Standard, StandardNormal};

/// The normal distribution, `N(mean, std_dev**2)`.
pub struct Normal<E: ComplexField> {
    mean: E,
    std_dev: E::Real,
}

/// The normal distribution, `N(mean, std_dev**2)` for `0 <= i < nrows`, `0 <= j < ncols`.
pub struct NormalMat<E: ComplexField> {
    /// Number of rows of the sampled matrix.
    pub nrows: usize,
    /// Number of columns of the sampled matrix.
    pub ncols: usize,
    /// Normal distribution parameters for a single scalar.
    pub normal: Normal<E>,
}

/// The standard normal distribution, `N(0, 1)` for `0 <= i < nrows`, `0 <= j < ncols`.
pub struct StandardNormalMat {
    /// Number of rows of the sampled matrix.
    pub nrows: usize,
    /// Number of columns of the sampled matrix.
    pub ncols: usize,
}

/// The standard distribution. Samples uniformly distributed values for `0 <= i < nrows`, `0 <= j
/// < ncols`.
pub struct StandardMat {
    /// Number of rows of the sampled matrix.
    pub nrows: usize,
    /// Number of columns of the sampled matrix.
    pub ncols: usize,
}

/// The normal distribution, `N(mean, std_dev**2)` for `0 <= j < ncols`.
pub struct NormalRow<E: ComplexField> {
    /// Number of columns of the sampled row.
    pub ncols: usize,
    /// Normal distribution parameters for a single scalar.
    pub normal: Normal<E>,
}

/// The standard normal distribution, `N(0, 1)` for `0 <= j < ncols`.
pub struct StandardNormalRow {
    /// Number of columns of the sampled row.
    pub ncols: usize,
}

/// The standard distribution. Samples uniformly distributed values for `0 <= j < ncols`.
pub struct StandardRow {
    /// Number of columns of the sampled row.
    pub ncols: usize,
}

/// The normal distribution, `N(mean, std_dev**2)` for `0 <= i < nrows`.
pub struct NormalCol<E: ComplexField> {
    /// Number of rows of the sampled column.
    pub nrows: usize,
    /// Normal distribution parameters for a single scalar.
    pub normal: Normal<E>,
}

/// The standard normal distribution, `N(0, 1)` for `0 <= i < nrows`.
pub struct StandardNormalCol {
    /// Number of rows of the sampled column.
    pub nrows: usize,
}

/// The standard distribution. Samples uniformly distributed values for `0 <= i < nrows`.
pub struct StandardCol {
    /// Number of rows of the sampled column.
    pub nrows: usize,
}

/// Uniformly samples a unitary matrix from the unitary group, in the sense of the [Haar measure](https://en.wikipedia.org/wiki/Haar_measure).
pub struct UnitaryMat {
    /// Dimension of the sampled matrix.
    pub dimension: usize,
}

impl<E: ComplexField> Normal<E> {
    /// Construct, from dimensions, mean and standard deviation.
    ///
    /// Parameters:
    /// - mean (`μ`, unrestricted)
    /// - standard deviation (`σ`, must be finite)
    pub fn new(mean: E, std_dev: E::Real) -> Result<Self, rand_distr::NormalError> {
        if !std_dev.faer_is_finite() {
            return Err(rand_distr::NormalError::BadVariance);
        }

        Ok(Self { mean, std_dev })
    }

    /// Construct, from dimensions, mean and coefficient of variation.
    ///
    /// Parameters:
    /// - mean (`μ`, unrestricted)
    /// - coefficient of variantion (`cv = abs(σ / μ)`)
    pub fn from_mean_cv(mean: E, cv: E::Real) -> Result<Self, rand_distr::NormalError> {
        if !cv.faer_is_finite() || cv < E::Real::faer_zero() {
            return Err(rand_distr::NormalError::BadVariance);
        }

        Ok(Self {
            mean,
            std_dev: mean.faer_abs().faer_mul(cv),
        })
    }
}

impl<E: ComplexField> Distribution<Mat<E>> for NormalMat<E>
where
    StandardNormal: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Mat<E> {
        Mat::from_fn(self.nrows, self.ncols, |_, _| {
            self.normal.mean.faer_add(
                StandardNormal
                    .sample(rng)
                    .faer_scale_real(self.normal.std_dev),
            )
        })
    }
}

impl<E: ComplexField> Distribution<Mat<E>> for StandardNormalMat
where
    StandardNormal: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Mat<E> {
        Mat::from_fn(self.nrows, self.ncols, |_, _| StandardNormal.sample(rng))
    }
}

impl<E: ComplexField> Distribution<Mat<E>> for StandardMat
where
    Standard: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Mat<E> {
        Mat::from_fn(self.nrows, self.ncols, |_, _| Standard.sample(rng))
    }
}

impl<E: ComplexField> Distribution<Mat<E>> for UnitaryMat
where
    StandardNormal: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Mat<E> {
        let qr = StandardNormalMat {
            nrows: self.dimension,
            ncols: self.dimension,
        }
        .sample(rng)
        .qr();

        let r_diag = qr.factors.as_ref().diagonal().column_vector();
        let mut q = qr.compute_q();

        for j in 0..self.dimension {
            let r = r_diag.read(j);
            let r = if r == E::faer_zero() {
                E::faer_one()
            } else {
                r.faer_scale_real(r.faer_abs().faer_inv())
            };

            crate::zipped!(q.as_mut().col_mut(j)).for_each(|crate::unzipped!(mut q)| {
                q.write(q.read().faer_mul(r));
            });
        }

        q
    }
}

impl<E: ComplexField> Distribution<Col<E>> for NormalCol<E>
where
    StandardNormal: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Col<E> {
        Col::from_fn(self.nrows, |_| {
            self.normal.mean.faer_add(
                StandardNormal
                    .sample(rng)
                    .faer_scale_real(self.normal.std_dev),
            )
        })
    }
}

impl<E: ComplexField> Distribution<Col<E>> for StandardNormalCol
where
    StandardNormal: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Col<E> {
        Col::from_fn(self.nrows, |_| StandardNormal.sample(rng))
    }
}

impl<E: ComplexField> Distribution<Col<E>> for StandardCol
where
    Standard: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Col<E> {
        Col::from_fn(self.nrows, |_| Standard.sample(rng))
    }
}

impl<E: ComplexField> Distribution<Row<E>> for NormalRow<E>
where
    StandardNormal: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Row<E> {
        Row::from_fn(self.ncols, |_| {
            self.normal.mean.faer_add(
                StandardNormal
                    .sample(rng)
                    .faer_scale_real(self.normal.std_dev),
            )
        })
    }
}

impl<E: ComplexField> Distribution<Row<E>> for StandardNormalRow
where
    StandardNormal: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Row<E> {
        Row::from_fn(self.ncols, |_| StandardNormal.sample(rng))
    }
}

impl<E: ComplexField> Distribution<Row<E>> for StandardRow
where
    Standard: Distribution<E>,
{
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Row<E> {
        Row::from_fn(self.ncols, |_| Standard.sample(rng))
    }
}
