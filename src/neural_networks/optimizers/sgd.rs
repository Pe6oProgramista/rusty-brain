use ndarray::prelude::*;
use super::{Optimizer, SGD};

impl Optimizer for SGD {
    fn run(&mut self, weights: ArrayD<f64>, gradient: f64) -> ArrayD<f64> {
        if self.velocity.shape() == &[] {
            self.velocity = ArrayD::<f64>::zeros(IxDyn(weights.shape()));
        }

        assert!(self.momentum >= 0. && self.momentum <= 1., "momentum not in range [0, 1]");

        self.velocity = self.momentum * &self.velocity - self.learning_rate * gradient;

        if self.nesterov {
            self.velocity = self.momentum * &self.velocity - self.learning_rate * gradient;
        }

        return &weights + &self.velocity
    }
}

impl Default for SGD {
    fn default() -> SGD {
        SGD {
            learning_rate: 0.01,
            momentum: 0.,
            decay: 0.,
            nesterov: false,
            velocity: ArrayD::<f64>::zeros(IxDyn(&[]))
        }
    }
}