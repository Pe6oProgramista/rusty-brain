use ndarray::prelude::*;
use super::{Optimizer, SGD};

impl Optimizer for SGD {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.shape() == &[1, 0] {
            self.velocity = Array2::zeros(weights.dim());
        }
        // println!("v{:?} - g{:?}", self.velocity.shape(), gradient.shape());

        assert!(self.momentum >= 0. && self.momentum <= 1., "momentum not in range [0, 1]");

        self.velocity = self.momentum * &self.velocity - self.learning_rate * gradient;

        return weights + &self.velocity
    }
}

impl Default for SGD {
    fn default() -> SGD {
        SGD {
            learning_rate: 0.01,
            momentum: 0.,
            decay: 0.,
            nesterov: false,
            velocity: arr2(&[[]])
        }
    }
}