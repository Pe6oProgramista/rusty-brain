use ndarray::prelude::*;
use super::{IsOptimizer, Adam};

impl IsOptimizer for Adam {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.e_gradient.shape() == &[1, 0] {
            self.e_gradient = Array2::zeros(weights.dim());
            self.e_velocity = Array2::zeros(weights.dim());
            self.velocity = Array2::zeros(weights.dim());
        }



        return weights + &self.velocity
    }
}

impl Default for Adam {
    fn default() -> Adam {
        Adam {
            epsilon: f64::powi(10., -8),
            rho: 0.9,
            e_gradient: arr2(&[[]]),
            e_velocity: arr2(&[[]]),
            velocity: arr2(&[[]])
        }
    }
}