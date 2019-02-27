use ndarray::prelude::*;

pub mod momentum;
pub mod nesterov_momentum;
pub mod adagrad;
pub mod adadelta;

pub trait IsOptimizer : Clone {
    fn run(&mut self, weights: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64>;
}

#[derive(Clone)]
pub struct Momentum {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: Array2<f64>
}

#[derive(Clone)]
pub struct NesterovMomentum {
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: Array2<f64>
}

#[derive(Clone)]
pub struct Adagrad {
    pub learning_rate: f64,
    pub epsilon: f64,
    pub G: Array2<f64>
}

#[derive(Clone)]
pub struct Adadelta {
    pub epsilon: f64,
    pub rho: f64,
    pub E_gradient: Array2<f64>,
    pub E_velocity: Array2<f64>,
    pub velocity: Array2<f64>
}

#[derive(Clone)]
pub enum Optimizer {
    Momentum(Momentum),
    NesterovMomentum(NesterovMomentum),
    Adagrad(Adagrad),
    Adadelta(Adadelta)
}

macro_rules! use_Optimizer {(
        let Opt($pat:pat) = $e:expr;
        $($body:tt)*
    ) => ({
        match $e {
            Optimizer::Momentum($pat) => { $($body)* },
            Optimizer::NesterovMomentum($pat) => { $($body)* },
            Optimizer::Adagrad($pat) => { $($body)* },
            Optimizer::Adadelta($pat) => { $($body)* }
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