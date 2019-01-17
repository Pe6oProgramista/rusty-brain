use ndarray::prelude::*;
use ndarray::Zip;

use super::{LayerTrait, Dense};
use neural_networks::optimizers::*;
use neural_networks::activation_functions::*;

impl<O> Dense<O> 
    where O: Optimizer
{
    pub fn new(input_shape: &Vec<usize>, units: usize, optimizer: O, activation_fn: ActivationFn) -> Self {
        Dense {
            input: arr2(&[[]]),
            output: arr2(&[[]]),
            input_shape: input_shape.clone(),
            units: units,
            weights: Array::linspace(0., 1., input_shape[1]).into_shape((1, input_shape[1])).unwrap(),
            optimizer: optimizer,
            activation_fn: activation_fn
        }
    }
}

impl Default for Dense<SGD> {
    fn default() -> Self {
        Dense {
            input: arr2(&[[]]),
            output: arr2(&[[]]),
            input_shape: Vec::new(),
            units: 0,
            weights: arr2(&[[]]),
            optimizer: SGD { ..Default::default() },
            activation_fn: Default::default()
        }
    }
}

impl<O> LayerTrait<O> for Dense<O>
    where O: Optimizer
{
    fn get_input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn set_input_shape(&mut self, shape: &Vec<usize>) {
        self.input_shape = shape.clone();
        self.init_weights()
    }

    fn get_output_shape(&self) -> Vec<usize> {
        let mut output_shape = self.input_shape.clone();
        if let Some(last) = output_shape.last_mut() {
            *last = self.units;
        }
        output_shape
    }

    fn set_units(&mut self, units: usize) {
        self.units = units;
        self.init_weights()
    }

    fn set_optimizer(&mut self, optimizer: &O) {
        self.optimizer = optimizer.clone()
    }

    fn init_weights(&mut self) {
        self.weights = Array::linspace(0., self.units as f64, self.units * self.input_shape[1])
            .into_shape((self.units, self.input_shape[1]))
            .unwrap();
    }

    fn parameters(&self) -> usize {
        self.weights.shape()[0] * self.weights.shape()[1]
        // self.weights.shape().iter().fold(0,|a, &b| a + b)
    }

    fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64> {
        assert!(input.shape().to_vec() == self.input_shape,
            "input isn't with same shape as input_shape");
        assert!(self.units == self.weights.shape()[0] && input.shape()[1] == self.weights.shape()[1],
            "weight isn't with the same shape as input and output: u{:?} - in{:?} - w{:?}", self.units, input.shape(), self.weights.shape());

        self.input = input.clone();
        self.output = input.dot(&self.weights.clone().reversed_axes());
        self.activation_fn.run(&self.output)
    }

    fn backward_prop(&mut self, gradient: &Array2<f64>) -> Array2<f64> {
        let mut gradient = gradient.clone();
        let activation_grad = self.activation_fn.gradient(&self.output);
        Zip::from(&mut gradient).and(&activation_grad).apply(|a, &b| *a *= b);
        let weights = self.weights.clone();

        let grad_wrt_w = gradient.clone().reversed_axes().dot(&self.input);
        self.weights = self.optimizer.run(&self.weights, &grad_wrt_w);

        gradient.dot(&weights)
    }
}