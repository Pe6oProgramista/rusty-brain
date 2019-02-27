use ndarray::prelude::*;
use ndarray::Zip;

#[derive(Clone)]
pub enum LossFn {
    MeanSquare,
    CrossEntropy
}

impl LossFn {
    pub fn run(&self, prediction: &Array2<f64>, output: &Array2<f64>) -> f64 {
        let samples_cnt = prediction.shape()[0] as f64;
        match self {
            LossFn::MeanSquare => (prediction - output).mapv(|x: f64| x.powi(2)).sum() / (2. * samples_cnt),
            LossFn::CrossEntropy => {
                let mut loss = output.clone();
                Zip::from(&mut loss).and(prediction).apply(|a, &b| {
                    let mut pred = b;
                    if pred >= 1. {
                        pred = 1. - 1e-15_f64;
                    } else if pred <= 0. {
                        pred = 1e-15_f64;
                    }
                    *a = *a * pred.ln() + (1. - *a) * (1. - pred).ln();
                });
                
                loss.sum() / -samples_cnt
            },
            _ => 0.
        }
    }

    pub fn gradient(&self, prediction: &Array2<f64>, output: &Array2<f64>) -> Array2<f64> {
        match self {
            LossFn::MeanSquare => (prediction - output),
            LossFn::CrossEntropy => {
                let mut prediction = prediction.clone();
                Zip::from(&mut prediction).apply(|p| {
                    if *p >= 1. {
                        *p = 1. - 1e-15_f64;
                    } else if *p <= 0. {
                        *p = 1e-15_f64;
                    }
                });
                (&prediction - output) / ((1. - &prediction) * &prediction)
            },
            _ => Array2::zeros(prediction.dim())
        }
    }
}

impl Default for LossFn {
    fn default() -> Self {
        LossFn::MeanSquare
    }
}