use ndarray::prelude::*;

use super::NeuralNetwork;
use super::layer::LayerTrait;
use neural_networks::optimizers::*;

impl<L> NeuralNetwork<L, SGD> 
    where L: LayerTrait<SGD>
{
    pub fn new(layers: Vec<L>) -> Self {
        NeuralNetwork {
            layers: layers,
            optimizer: SGD { ..Default::default() },
            loss_fn: Default::default()
        }
    }
}

impl<L> Default for NeuralNetwork<L, SGD> 
    where L: LayerTrait<SGD>
{
    fn default() -> Self {
        NeuralNetwork {
            layers: Vec::<L>::new(),
            optimizer: SGD { ..Default::default() },
            loss_fn: Default::default()
        }
    }
}

impl<L, O> NeuralNetwork<L, O> 
    where L: LayerTrait<O>,
          O: Optimizer
{
    pub fn add(&mut self, mut layer: L) {
        if self.layers.len() != 0 {
            layer.set_input_shape(&self.layers.last().unwrap().get_output_shape());
        } else {
            layer.init_weights();
        }
        layer.set_optimizer(&self.optimizer);

        self.layers.push(layer);
    }

    pub fn parameters(&self) -> usize {
        let mut p = 0;
        for layer in &self.layers {
            println!("{}", layer.parameters());
            p += layer.parameters();
        }
        p
    }

    pub fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut last_output = input.clone();
        for layer in &mut self.layers {
            last_output = layer.forward_prop(&last_output);
        }
        last_output
    }

    pub fn backward_prop(&mut self, gradient: &Array2 <f64>) -> Array2<f64> {
        let mut gradient = gradient.clone();
        for mut layer in &mut self.layers.iter_mut().rev() {
            gradient = layer.backward_prop(&gradient);
        }
        gradient
    }
}