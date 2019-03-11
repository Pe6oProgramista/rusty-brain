use serde_derive::{Serialize, Deserialize};
use ndarray::prelude::*;

pub mod momentum;
pub mod nesterov_momentum;
pub mod adagrad;
pub mod adadelta;
pub mod adam;

pub trait IsOptimizer : Clone {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Momentum {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: Array2<f64>
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NesterovMomentum {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: Array2<f64>
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Adagrad {
    pub learning_rate: f64,
    pub epsilon: f64,
    pub g: Array2<f64>
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Adadelta {
    pub epsilon: f64,
    pub rho: f64,
    pub e_gradient: Array2<f64>,
    pub e_velocity: Array2<f64>,
    pub velocity: Array2<f64>
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Adam {
    pub epsilon: f64,
    pub rho: f64,
    pub e_gradient: Array2<f64>,
    pub e_velocity: Array2<f64>,
    pub velocity: Array2<f64>
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Optimizer {
    Momentum(Momentum),
    NesterovMomentum(NesterovMomentum),
    Adagrad(Adagrad),
    Adadelta(Adadelta),
    Adam(Adam)
}

macro_rules! use_Optimizer {(
        let Opt($pat:pat) = $e:expr;
        $($body:tt)*
    ) => ({
        match $e {
            Optimizer::Momentum($pat) => { $($body)* },
            Optimizer::NesterovMomentum($pat) => { $($body)* },
            Optimizer::Adagrad($pat) => { $($body)* },
            Optimizer::Adadelta($pat) => { $($body)* },
            Optimizer::Adam($pat) => { $($body)* }
        }
    })
}

impl IsOptimizer for Optimizer {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        use_Optimizer!{
            let Opt(ref mut op) = *self;
            op.run(weights, gradient)
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::Momentum(Default::default())
    }
}