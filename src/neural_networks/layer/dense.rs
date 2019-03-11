use serde_derive::{Serialize, Deserialize};
use ndarray::prelude::*;
use ndarray::Zip;
use rand::prelude::*;

use super::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct Dense {
    input: Array2<f64>,
    output: Array2<f64>,
    inputs_cnt: usize,
    units: usize,
    weights: Array2<f64>,
    optimizer: Optimizer,
    activation_fn: ActivationFn
}

impl Dense {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Default for Dense {
    fn default() -> Self {
        Dense {
            input: arr2(&[[]]),
            output: arr2(&[[]]),
            inputs_cnt: 0,
            units: 0,
            weights: arr2(&[[]]),
            optimizer: Default::default(),
            activation_fn: Default::default()
        }
    }
}

#[typetag::serde(name = "dense")]
impl Layer for Dense {
    fn build(&self) -> Box<Layer> {
        Box::new(self.clone())
    }

    fn get_inputs_cnt(&self) -> usize {
        self.inputs_cnt
    }

    fn set_inputs_cnt(&mut self, inputs_cnt: usize) -> &mut Layer {
        self.inputs_cnt = inputs_cnt + 1;
        self
    }

    fn get_units(&self) -> usize {
        self.units
    }

    fn set_units(&mut self, units: usize) -> &mut Layer {
        self.units = units;
        self
    }

    fn get_optimizer(&self) -> Optimizer {
        self.optimizer.clone()
    }

    fn set_optimizer(&mut self, optimizer: &Optimizer) -> &mut Layer {
        self.optimizer = optimizer.clone();
        self
    }

    fn get_activation_fn(&self) -> ActivationFn {
        self.activation_fn.clone()
    }

    fn set_activation_fn(&mut self, activation_fn: &ActivationFn) -> &mut Layer {
        self.activation_fn = activation_fn.clone();
        self
    }

    fn get_weights(&self) -> Array2<f64> {
        self.weights.clone()
    }

    fn init_weights(&mut self) -> &mut Layer {
        if self.inputs_cnt != 0 { 
            self.weights = unsafe { Array2::<f64>::uninitialized((self.units, self.inputs_cnt)) };
            Zip::from(&mut self.weights).apply(|x| *x = 2. * random::<f64>() - 1.);
            
            // self.weights = Array::linspace(-1., 1., self.units * self.inputs_cnt)
            //     .into_shape((self.units, self.inputs_cnt))
            //     .unwrap();
        }
        self
    }

    fn parameters(&self) -> usize {
        self.weights.shape().iter().fold(1,|acc, &x| acc * x)
    }

    fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let input = stack!(Axis(1), Array2::ones((input.shape()[0], 1)), *input);

        assert!(input.shape()[1] == self.inputs_cnt,
            "input isn't with same shape as inputs_cnt {:?} {:?}", input.shape(), self.inputs_cnt);
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