use ndarray::prelude::*;
use super::optimizers::*;

pub mod dense;
pub mod activation;

pub trait LayerTrait {
    fn get_input_shape(&self) -> Vec<usize> {
    vec![]
    }

    fn set_input_shape(&mut self, shape: &Vec<usize>) {}

    fn get_output_shape(&self) -> Vec<usize> {
        vec![]
    }

    fn set_units(&mut self, units: usize) {}

    fn parameters(&self) -> usize {
        0
    }

    fn forward_prop(&mut self, input: ArrayD<f64>) {}

    fn backward_prop(&mut self, gradient: ArrayD<f64>) {}
}

pub struct Dense {
    pub input: ArrayD<f64>,
    pub input_shape: Vec<usize>,
    pub units: usize,
    pub weights: ArrayD<f64>,
    pub optimizer: Box<Optimizer>
}

pub struct Activation {
    pub func_name: String
}