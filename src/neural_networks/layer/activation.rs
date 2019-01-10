use ndarray::prelude::*;
use super::{LayerTrait, Activation};

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