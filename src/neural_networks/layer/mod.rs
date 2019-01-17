use ndarray::prelude::*;
use super::optimizers::*;
use super::activation_functions::*;

pub mod dense;

pub trait LayerTrait<O>
    where O: Optimizer
{
    fn get_input_shape(&self) -> Vec<usize>;

    fn set_input_shape(&mut self, shape: &Vec<usize>);

    fn get_output_shape(&self) -> Vec<usize>;

    fn set_units(&mut self, units: usize);

    fn set_optimizer(&mut self, optimizer: &O);

    fn init_weights(&mut self);

    fn parameters(&self) -> usize;

    fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64>;

    fn backward_prop(&mut self, gradient: &Array2<f64>) -> Array2<f64>;
}

pub struct Dense<O>
    where O: Optimizer
{
    pub input: Array2<f64>,
    pub output: Array2<f64>,
    pub input_shape: Vec<usize>,
    pub units: usize,
    pub weights: Array2<f64>,
    pub optimizer: O,
    pub activation_fn: ActivationFn
}