use ndarray::prelude::*;

pub enum LossFn {
    MeanSquare
}

impl LossFn {
    pub fn run(&self, prediction: &Array2<f64>, output: &Array2<f64>) -> f64 {
        match self {
            LossFn::MeanSquare => (prediction - output).map(|x| { x.powf(2.) }).sum() / (2. * prediction.shape()[0] as f64),
            _ => 0.
        }
    }

    pub fn gradient(&self, prediction: &Array2<f64>, output: &Array2<f64>) -> Array2<f64> {
        match self {
            LossFn::MeanSquare => prediction - output,
            _ => Array2::zeros(prediction.dim())
        }
    }
}

impl Default for LossFn {
    fn default() -> Self {
        LossFn::MeanSquare
    }
}