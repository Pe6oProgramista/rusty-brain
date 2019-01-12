use ndarray::prelude::*;
use super::{LayerTrait, Dense};
use neural_networks::optimizers::*;

impl Dense<SGD> {
    pub fn new(input_shape: &Vec<usize>, units: usize) -> Self {
        Dense {
            input: arr1(&[]).into_dyn(),
            input_shape: input_shape.clone(),
            units: units,
            weights: ArrayD::<f64>::zeros(IxDyn(input_shape.clone().as_slice())),
            optimizer: SGD { ..Default::default() },
            activation_fn: Default::default()
        }
    }
}

impl Default for Dense<SGD> {
    fn default() -> Self {
        Dense {
            input: arr1(&[]).into_dyn(),
            input_shape: Vec::new(),
            units: 0,
            weights: arr1(&[]).into_dyn(),
            optimizer: SGD { ..Default::default()},
            activation_fn: Default::default()
        }
    }
}

impl<T> LayerTrait for Dense<T>
    where T: Optimizer
{
    fn get_input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn set_input_shape(&mut self, shape: &Vec<usize>) {
        self.input_shape.clone_from_slice(shape)
    }

    fn get_output_shape(&self) -> Vec<usize> {
        let mut output_shape = self.input_shape.clone();
        if let Some(last) = output_shape.last_mut() {
            *last = self.units;
        }
        output_shape
    }

    fn set_units(&mut self, units: usize) {
        self.units = units
    }

    fn parameters(&self) -> usize {
        0
    }

    fn forward_prop(&mut self, input: ArrayD<f64>) {}

    fn backward_prop(&mut self, gradient: ArrayD<f64>) {}
}