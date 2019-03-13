use ndarray::prelude::*;
use super::{IsOptimizer, RMSProp};

impl IsOptimizer for RMSProp {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.shape() == &[1, 0] {
            self.velocity = Array2::zeros(weights.dim());
            self.e_gradient = Array2::zeros(weights.dim());
        }

        self.e_gradient = self.rho * &self.e_gradient + (1. - self.rho) * gradient.mapv(|x: f64| x.powi(2));
        let velocity = - self.learning_rate * gradient / (&self.e_gradient + self.epsilon).mapv(|x: f64| x.sqrt());

        return weights + &velocity
    }
}

impl Default for RMSProp {
    fn default() -> RMSProp {
        RMSProp {
            learning_rate: 0.1,
            epsilon: f64::powi(10., -8),
            rho: 0.9,
            e_gradient: arr2(&[[]]),
            velocity: arr2(&[[]])
        }
    }
}