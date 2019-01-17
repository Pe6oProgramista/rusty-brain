use ndarray::prelude::*;

pub enum ActivationFn {
    Sigmoid
}

impl ActivationFn {
    pub fn run(&self, prediction: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFn::Sigmoid => prediction.map(|x| { 1. / (1. + f64::exp(-x)) }),
            _ => prediction.clone()
        }
    }

    pub fn gradient(&self, prediction: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFn::Sigmoid => {
                let prediction = self.run(prediction);
                prediction.map(|x| { x * (1. - x) })
            },
            _ => prediction.clone()
        }
    }
}

impl Default for ActivationFn {
    fn default() -> Self {
        ActivationFn::Sigmoid
    }
}