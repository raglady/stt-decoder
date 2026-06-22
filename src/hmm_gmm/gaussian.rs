cfg_select! {
    feature = "f64" => {
        use std::f64::consts::PI;
    }
    _ => {
        use std::f32::consts::PI;
    }
}

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

use crate::types::Float;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gaussian {
    mean: Array1<Float>,
    covar: Array1<Float>, // diagonal covariance
    dim: usize,
}

// Only for diagonal covariance
impl Gaussian {
    pub fn new(dim: usize) -> Self {
        Gaussian {
            mean: Array1::zeros([dim]),
            covar: Array1::zeros([dim]),
            dim,
        }
    }

    pub fn get_dimension(&self) -> usize {
        self.dim
    }

    pub fn get_mean(&self) -> ArrayView1<'_, Float> {
        self.mean.view()
    }

    pub fn set_mean(&mut self, mean: ArrayView1<Float>) {
        self.mean = mean.to_owned();
    }

    pub fn set_covar(&mut self, covar: ArrayView1<Float>) {
        self.covar = covar.to_owned();
    }

    pub fn get_covar(&self) -> ArrayView1<'_, Float> {
        self.covar.view()
    }

    #[inline]
    pub fn log_multivar_gauss_dist(&self, obs: ArrayView1<Float>) -> Float {
        assert_eq!(obs.len(), self.dim);
        let ln_det_covar: Float = self.covar.iter().map(|v| v.ln()).sum();

        -0.5 * self.dim as Float * (2.0 * PI).ln()
            - 0.5 * ln_det_covar
            - 0.5 * ((&obs - &self.mean.view()).powi(2) / &self.covar).sum()
    }
}
