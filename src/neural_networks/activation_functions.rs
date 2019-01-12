use ndarray::prelude::*;

pub enum ActivationFn {
    Sigmoid
}

impl ActivationFn {
    fn run(&self, input: ArrayD<f64>) -> ArrayD<f64> {
        match self {
            ActivationFn::Sigmoid => input.map(|x| { 1. / (1. + f64::exp(-x)) }),
            _ => input.clone()
        }
    }

    fn gradient(&self, input: ArrayD<f64>) -> ArrayD<f64> {
        match self {
            ActivationFn::Sigmoid => self.run(input.clone()) * (1. - self.run(input.clone())),
            _ => input.clone()
        }
    }
}

impl Default for ActivationFn {
    fn default() -> Self {
        ActivationFn::Sigmoid
    }
}