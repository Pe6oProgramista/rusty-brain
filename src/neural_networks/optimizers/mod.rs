use ndarray::prelude::*;

pub mod sgd;

pub trait Optimizer: Clone {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64>;
}

#[derive(Clone)]
pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
    pub decay: f64,
    pub nesterov: bool,
    pub velocity: Array2<f64>
}