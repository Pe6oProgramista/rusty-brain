use ndarray::prelude::*;

pub mod sgd;

pub trait Optimizer {
    fn run(&mut self, weights: ArrayD<f64>, gradient: f64) -> ArrayD<f64>;
}

pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
    pub decay: f64,
    pub nesterov: bool,
    pub velocity: ArrayD<f64>
}