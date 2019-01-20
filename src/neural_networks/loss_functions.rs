use ndarray::prelude::*;

#[derive(Clone)]
pub enum LossFn {
    MeanSquare
}

impl LossFn {
    pub fn run(&self, prediction: &Array2<f64>, output: &Array2<f64>) -> f64 {
        match self {
            LossFn::MeanSquare => (prediction - output).reversed_axes().dot(&(prediction - output))[[0, 0]] / (2. * prediction.shape()[0] as f64),
            _ => 0.
        }
    }

    pub fn gradient(&self, prediction: &Array2<f64>, output: &Array2<f64>) -> Array2<f64> {
        match self {
            LossFn::MeanSquare => (prediction - output) / prediction.shape()[0] as f64,
            _ => Array2::zeros(prediction.dim())
        }
    }
}

impl Default for LossFn {
    fn default() -> Self {
        LossFn::MeanSquare
    }
}