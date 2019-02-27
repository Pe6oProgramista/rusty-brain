use ndarray::prelude::*;
use ndarray::Zip;
use rand::prelude::*;

use super::{LayerTrait, Dense};
use neural_networks::optimizers::*;
use neural_networks::activation_functions::*;

impl Dense {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build(&self) -> Self {
        self.clone()
    }
}

impl Default for Dense {
    fn default<'a>() -> Self {
        Dense {
            input: arr2(&[[]]),
            output: arr2(&[[]]),
            input_shape: Vec::new(),
            units: 0,
            weights: arr2(&[[]]),
            optimizer: Default::default(),
            activation_fn: Default::default()
        }
    }
}

impl LayerTrait for Dense {
    fn get_input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn set_input_shape<'a>(&'a mut self, shape: &Vec<usize>) -> &'a mut Self {
        self.input_shape = shape.clone();
        self.input_shape[1] += 1;
        self
    }

    fn get_output_shape(&self) -> Vec<usize> {
        let mut output_shape = self.input_shape.clone();
        if let Some(last) = output_shape.last_mut() {
            *last = self.units;
        }
        output_shape
    }

    fn set_units<'a>(&'a mut self, units: usize) -> &'a mut Self {
        self.units = units;
        self
    }

    fn get_optimizer(&self) -> Optimizer {
        self.optimizer.clone()
    }

    fn set_optimizer<'a>(&'a mut self, optimizer: &Optimizer) -> &'a mut Self {
        self.optimizer = optimizer.clone();
        self
    }

    fn get_activation_fn(&self) -> ActivationFn {
        self.activation_fn.clone()
    }

    fn set_activation_fn<'a>(&'a mut self, activation_fn: &ActivationFn) -> &'a mut Self {
        self.activation_fn = activation_fn.clone();
        self
    }

    fn get_weights(&self) -> Array2<f64> {
        self.weights.clone()
    }

    fn init_weights<'a>(&'a mut self) -> &'a mut Self {
        if(self.input_shape != vec![]) { 
            self.weights = unsafe { Array2::<f64>::uninitialized((self.units, self.input_shape[1])) };
            Zip::from(&mut self.weights).apply(|x| *x = 2. * random::<f64>() - 1.);
            
            // self.weights = Array::linspace(-1., 1., self.units * self.input_shape[1])
            //     .into_shape((self.units, self.input_shape[1]))
            //     .unwrap();
        }
        self
    }

    fn parameters(&self) -> usize {
        self.weights.shape().iter().fold(1,|acc, &x| acc * x)
    }

    fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let input = stack!(Axis(1), Array2::ones((input.shape()[0], 1)), *input);

        assert!(input.shape()[1] == self.input_shape[1],
            "input isn't with same shape as input_shape {:?} {:?}", input.shape(), self.input_shape);
        assert!(self.units == self.weights.shape()[0] && input.shape()[1] == self.weights.shape()[1],
            "weight isn't with the same shape as input and output: u{:?} - in{:?} - w{:?}", self.units, input.shape(), self.weights.shape());

        self.input = input.clone();
        self.output = input.dot(&self.weights.clone().reversed_axes());
        self.activation_fn.run(&self.output)
    }

    fn backward_prop(&mut self, gradient: &Array2<f64>) -> Array2<f64> {
        let mut gradient = gradient.clone();
        let activation_grad = self.activation_fn.gradient(&self.output);
        gradient = &gradient * &activation_grad;
        let weights = self.weights.clone();

        let grad_wrt_w = gradient.clone().reversed_axes().dot(&self.input) / gradient.shape()[0] as f64;
        self.weights = self.optimizer.run(&self.weights, &grad_wrt_w);

        gradient.dot(&weights).slice_move(s![.., 1..])
    }
}