use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::{hmm_gmm::gaussian::Gaussian, log_sum_exp, types::Float};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gmm {
    // gaussian, poids
    gaussians_weights: Vec<(Gaussian, Float)>,
    n_components: usize,
}

impl Gmm {
    pub fn new(n_components: usize, dim: usize) -> Self {
        let weight = 1.0 / n_components as Float;
        let gaussians_weights = (0..n_components)
            .map(|_| (Gaussian::new(dim), weight))
            .collect();

        Gmm {
            gaussians_weights,
            n_components,
        }
    }

    pub fn from_vec_gaussian_weight(vec_gaussian_weight: &[(Gaussian, Float)]) -> Self {
        Self {
            gaussians_weights: vec_gaussian_weight.to_vec(),
            n_components: vec_gaussian_weight.len(),
        }
    }

    pub fn initialize(&mut self, data: ArrayView2<Float>) {
        if self.get_component(0).0.get_mean().iter().sum::<Float>() == 0.0 {
            if self.num_component() > 1 {
                unimplemented!("begin with one component and use mixing up after !")
            }
            let mean = data.mean_axis(Axis(0)).unwrap().to_vec();
            let covar = data.var_axis(Axis(0), 0.0).to_vec();
            self.gaussians_weights[0]
                .0
                .set_mean(Array1::from_vec(mean).view());
            self.gaussians_weights[0]
                .0
                .set_covar(Array1::from_vec(covar).view());
            self.gaussians_weights[0].1 = 1.0;
        }
    }

    pub fn num_component(&self) -> usize {
        self.n_components
    }

    pub fn get_component(&self, index: usize) -> &(Gaussian, Float) {
        &self.gaussians_weights[index]
    }

    pub fn set_component(&mut self, index: usize, gaussian: &Gaussian, weight: Float) {
        self.gaussians_weights[index] = (gaussian.clone(), weight);
    }

    #[inline]
    pub fn log_probability_density_state_component(
        &self,
        obs: ArrayView1<Float>,
        component: usize,
    ) -> Float {
        self.gaussians_weights[component].1.ln()
            + self.gaussians_weights[component]
                .0
                .log_multivar_gauss_dist(obs)
    }

    #[inline]
    pub fn log_probability_density(&self, obs: ArrayView1<Float>) -> Float {
        log_sum_exp(
            &self
                .gaussians_weights
                .iter()
                .map(|component| component.1.ln() + component.0.log_multivar_gauss_dist(obs))
                .collect::<Vec<Float>>(),
        )
    }

    pub fn split_up(&mut self) {
        let mut new_gmm = Vec::new();
        let mut sorted_gaussian = self
            .gaussians_weights
            .iter()
            .enumerate()
            .collect::<Vec<_>>();
        sorted_gaussian.sort_by(|a, b| a.1.1.total_cmp(&b.1.1));
        sorted_gaussian.reverse();

        for (index, (gaussian, weight)) in self.gaussians_weights.iter().enumerate() {
            if index == sorted_gaussian[0].0 {
                let new_mean1 = gaussian
                    .get_mean()
                    .iter()
                    .enumerate()
                    .map(|(index, mean)| mean - (gaussian.get_covar()[index].sqrt() * 0.2))
                    .collect::<Vec<_>>();
                let new_mean2 = gaussian
                    .get_mean()
                    .iter()
                    .enumerate()
                    .map(|(index, mean)| mean + (gaussian.get_covar()[index].sqrt() * 0.2))
                    .collect::<Vec<_>>();
                let mut gaussian1 = gaussian.clone();
                let mut gaussian2 = gaussian.clone();
                gaussian1.set_mean(Array1::from_vec(new_mean1).view());
                gaussian2.set_mean(Array1::from_vec(new_mean2).view());
                new_gmm.append(&mut vec![
                    (gaussian1, *weight / 2.0),
                    (gaussian2, *weight / 2.0),
                ]);
            } else {
                new_gmm.push((gaussian.clone(), *weight));
            }
        }
        self.gaussians_weights = new_gmm;
        self.n_components += 1;
    }
}
