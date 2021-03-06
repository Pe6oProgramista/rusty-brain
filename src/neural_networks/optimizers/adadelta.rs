use ndarray::prelude::*;
use super::{IsOptimizer, Adadelta};

impl IsOptimizer for Adadelta {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.e_gradient.shape() == &[1, 0] {
            self.e_gradient = Array2::zeros(weights.dim());
            self.e_velocity = Array2::zeros(weights.dim());
            self.velocity = Array2::zeros(weights.dim());
        }

        self.e_gradient = self.rho * &self.e_gradient + (1. - self.rho) * gradient.mapv(|x: f64| x.powi(2));

        self.e_velocity = self.rho * &self.e_velocity + (1. - self.rho) * self.velocity.mapv(|x: f64| x.powi(2));

        self.velocity = - (self.epsilon + &self.e_velocity).mapv(|x: f64| x.sqrt()) /
            (self.epsilon + &self.e_gradient).mapv(|x: f64| x.sqrt()) * gradient;

        return weights + &self.velocity
    }
}

impl Default for Adadelta {
    fn default() -> Adadelta {
        Adadelta {
            epsilon: f64::powi(10., -8),
            rho: 0.9,
            e_gradient: arr2(&[[]]),
            e_velocity: arr2(&[[]]),
            velocity: arr2(&[[]])
        }
    }
}