use ndarray::prelude::*;
use super::{IsOptimizer, Momentum};

impl IsOptimizer for Momentum {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.shape() == &[1, 0] {
            self.velocity = Array2::zeros(weights.dim());
        }

        assert!(self.momentum >= 0. && self.momentum <= 1., "momentum is not in range [0, 1]");

        self.velocity = self.momentum * &self.velocity - self.learning_rate * gradient;

        return weights + &self.velocity
    }
}

impl Default for Momentum {
    fn default() -> Momentum {
        Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            velocity: arr2(&[[]])
        }
    }
}