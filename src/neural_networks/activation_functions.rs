use serde_derive::{Serialize, Deserialize};
use ndarray::prelude::*;
use utils::*;

#[derive(Clone, Serialize, Deserialize)]
pub enum ActivationFn {
    Linear,
    Sigmoid,
    Softmax,
    Relu
}

impl ActivationFn {
    pub fn run(&self, prediction: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFn::Linear => prediction.clone(),
            ActivationFn::Sigmoid => prediction.map(|x| {
                let res = 1. / (1. + f64::exp(-x));
                res
            }),
            ActivationFn::Softmax => {
                let (samples, _) = prediction.dim();
                
                let max = prediction.map_axis(Axis(1), |x| max_arr1(&x))
                    .into_shape((samples, 1_usize))
                    .unwrap();
                let e_pred = (prediction - &max).map(|x| f64::exp(*x));
                let sum = e_pred.sum_axis(Axis(1))
                    .into_shape((samples, 1_usize))
                    .unwrap();
                
                &e_pred / &sum
            },
            ActivationFn::Relu => {
                prediction.map(|x| if *x > 0. {*x} else {0.})
            },
            _ => prediction.clone()
        }
    }

    pub fn gradient(&self, prediction: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFn::Linear => Array2::ones(prediction.dim()),
            ActivationFn::Sigmoid |
            ActivationFn::Softmax => {
                let prediction = self.run(prediction);
                (1. - &prediction) * &prediction
            },
            ActivationFn::Relu => {
                prediction.map(|x| if *x > 0. {1.} else {0.})
            },
            _ => Array2::ones(prediction.dim())
        }
    }
}

impl Default for ActivationFn {
    fn default() -> Self {
        ActivationFn::Linear
    }
}