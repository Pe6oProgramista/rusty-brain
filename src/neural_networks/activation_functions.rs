use ndarray::prelude::*;

#[derive(Clone)]
pub enum ActivationFn {
    Linear,
    Sigmoid
}

impl ActivationFn {
    pub fn run(&self, prediction: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFn::Sigmoid => prediction.map(|x| { 1. / (1. + f64::exp(-x)) }),
            ActivationFn::Linear => prediction.clone(),
            _ => prediction.clone()
        }
    }

    pub fn gradient(&self, prediction: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFn::Sigmoid => {
                let prediction = self.run(prediction);
                prediction.map(|x| { x * (1. - x) })
            },
            ActivationFn::Linear => Array2::ones(prediction.dim()),
            _ => Array2::ones(prediction.dim())
        }
    }
}

impl Default for ActivationFn {
    fn default() -> Self {
        ActivationFn::Sigmoid
    }
}