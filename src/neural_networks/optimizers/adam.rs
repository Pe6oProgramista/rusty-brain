use ndarray::prelude::*;
use super::{IsOptimizer, Adam};

impl IsOptimizer for Adam {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.shape() == &[1, 0] {
            self.velocity = Array2::zeros(weights.dim());
            self.m = Array2::zeros(weights.dim());
            self.v = Array2::zeros(weights.dim());
        }

        self.m = self.b1 * &self.m + (1. - self.b1) * gradient;
        self.v = self.b2 * &self.v + (1. - self.b2) * gradient.mapv(|x: f64| x.powi(2));

        let m_bc = &self.m / (1. - self.b1.powi(self.t));
        let v_bc = &self.v / (1. - self.b2.powi(self.t));

        self.velocity = - self.learning_rate * m_bc / 
            (v_bc.mapv(|x: f64| x.sqrt()) + self.epsilon);

        self.t += 1;
        return weights + &self.velocity
    }
}

impl Default for Adam {
    fn default() -> Adam {
        Adam {
            learning_rate: 0.1,
            epsilon: f64::powi(10., -8),
            b1: 0.9,
            b2: 0.999,
            m: arr2(&[[]]),
            v: arr2(&[[]]),
            t: 1,
            velocity: arr2(&[[]])
        }
    }
}