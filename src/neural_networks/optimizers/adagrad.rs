use ndarray::prelude::*;
use super::{IsOptimizer, Adagrad};

impl IsOptimizer for Adagrad {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.g.shape() == &[1, 0] {
            self.g = Array2::zeros(weights.dim());
        }

        self.g = gradient.mapv(|x: f64| x.powi(2)) + &self.g;
        let velocity = - self.learning_rate * gradient / (self.epsilon + &self.g.mapv(|x: f64| x.sqrt()));

        return weights + &velocity
    }
}

impl Default for Adagrad {
    fn default() -> Adagrad {
        Adagrad {
            learning_rate: 0.1,
            epsilon: f64::powi(10., -8),
            g: arr2(&[[]])
        }
    }
}