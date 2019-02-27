use ndarray::prelude::*;
use super::{IsOptimizer, Adagrad};

impl IsOptimizer for Adagrad {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.G.shape() == &[1, 0] {
            self.G = Array2::zeros(weights.dim());
        }

        self.G = gradient.mapv(|x: f64| x.powi(2)) + &self.G;
        let velocity = - self.learning_rate / (self.epsilon + &self.G).mapv(|x: f64| x.sqrt()) * gradient;

        return weights + &velocity
    }
}

impl Default for Adagrad {
    fn default() -> Adagrad {
        Adagrad {
            learning_rate: 0.1,
            epsilon: f64::powi(10., -8),
            G: arr2(&[[]])
        }
    }
}