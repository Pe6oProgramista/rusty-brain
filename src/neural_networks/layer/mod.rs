use ndarray::prelude::*;
use super::optimizers::*;
use super::activation_functions::*;

pub mod dense;

pub trait LayerTrait<O> : Clone
    where O: Optimizer
{
    fn get_input_shape(&self) -> Vec<usize>;

    fn set_input_shape<'a>(&'a mut self, shape: &Vec<usize>) -> &'a mut Self;

    fn get_output_shape(&self) -> Vec<usize>;

    fn set_units<'a>(&'a mut self, units: usize) -> &'a mut Self;

    fn get_optimizer(&self) -> O;

    fn set_optimizer<'a>(&'a mut self, optimizer: &O) -> &'a mut Self;

    fn get_activation_fn(&self) -> ActivationFn;

    fn set_activation_fn<'a>(&'a mut self, activation_fn: &ActivationFn) -> &'a mut Self;

    fn get_weights(&self) -> Array2<f64>;

    fn init_weights<'a>(&'a mut self) -> &'a mut Self;

    fn parameters(&self) -> usize;

    fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64>;

    fn backward_prop(&mut self, gradient: &Array2<f64>) -> Array2<f64>;
}

#[derive(Clone)]
pub struct Dense<O>
    where O: Optimizer
{
    input: Array2<f64>,
    output: Array2<f64>,
    input_shape: Vec<usize>,
    units: usize,
    weights: Array2<f64>,
    optimizer: O,
    activation_fn: ActivationFn
}