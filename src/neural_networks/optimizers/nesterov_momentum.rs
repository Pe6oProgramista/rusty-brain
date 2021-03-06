use ndarray::prelude::*;
use super::{IsOptimizer, NesterovMomentum};

impl IsOptimizer for NesterovMomentum {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.shape() == &[1, 0] {
            self.velocity = Array2::zeros(weights.dim());
        }

        assert!(self.momentum >= 0. && self.momentum <= 1., "momentum not in range [0, 1]");

        self.velocity = self.momentum.powi(2) * &self.velocity - (1. + self.momentum) * self.learning_rate * gradient;
        
        return weights + &self.velocity
    }
}

impl Default for NesterovMomentum {
    fn default() -> NesterovMomentum {
        NesterovMomentum {
            learning_rate: 0.1,
            momentum: 0.9,
            velocity: arr2(&[[]])
        }
    }
}