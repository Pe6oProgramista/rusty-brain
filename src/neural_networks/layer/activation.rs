use ndarray::prelude::*;
use super::{LayerTrait, Activation};
use neural_networks::activation_functions::*;

impl Activation {
    pub fn new(input_shape: &Vec<usize>, units: usize) -> Self {
        Activation {
            input_shape: input_shape.clone(),
            activation_fn: ActivationFn::Sigmoid
        }
    }
}

impl Default for Activation {
    fn default() -> Self {
        Activation {
            input_shape: Vec::new(),
            activation_fn: ActivationFn::Sigmoid
        }
    }
}

impl LayerTrait for Activation {
    fn get_input_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    fn set_input_shape(&mut self, shape: &Vec<usize>) {
        
    }

    fn get_output_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    fn set_units(&mut self, units: usize) {
        
    }

    fn parameters(&self) -> usize {
        0
    }

    fn forward_prop(&mut self, input: ArrayD<f64>) {}

    fn backward_prop(&mut self, gradient: ArrayD<f64>) {}
}