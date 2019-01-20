use ndarray::prelude::*;
use std::cmp::min;

use super::NeuralNetwork;
use super::layer::LayerTrait;
use super::optimizers::*;
use super::loss_functions::*;

impl<L> NeuralNetwork<L, SGD> 
    where L: LayerTrait<SGD>
{
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build(&self) -> Self {
        self.clone()
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
    pub fn get_layers(&self) -> Vec<L> {
        self.layers.clone()
    }

    pub fn add<'a>(&'a mut self, mut layer: L) -> &'a mut Self {
        if self.layers.len() != 0 {
            layer.set_input_shape(&self.layers.last().unwrap().get_output_shape());
        }
        layer.init_weights();
        layer.set_optimizer(&self.optimizer);

        self.layers.push(layer);
        self
    }

    pub fn get_optimizer(&self) -> O {
        self.optimizer.clone()
    }

    pub fn set_optimizer<'a>(&'a mut self, optimizer: &O) -> &'a mut Self {
        self.optimizer = optimizer.clone();
        self
    }

    pub fn get_loss_fn(&self) -> LossFn {
        self.loss_fn.clone()
    }

    pub fn set_loss_fn<'a>(&'a mut self, loss_fn: &LossFn) -> &'a mut Self {
        self.loss_fn = loss_fn.clone();
        self
    }

    pub fn parameters(&self) -> usize {
        let mut p = 0;
        for layer in &self.layers {
            p += layer.parameters();
        }
        p
    }

    pub fn fit(&mut self, input: &Array2<f64>, output: &Array2<f64>, batch_size: usize, epochs: usize) {
        let mut errors = Vec::<f64>::new();
        for _ in 0..epochs {
            errors.push(self.train_on_batch(input, output));
            // println!("{}", err);
            // let samples = input.shape()[0];
            // for i in Array::range(0., samples as f64, batch_size as f64).iter() {
            //     let (start, stop) = (i as usize, min(samples, i as usize + batch_size));
            //     self.train_on_batch(input.slice(s![start..stop, ..]), output.slice(s![start..stop, ..]));
            // }
        }
    }

    pub fn predict(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.forward_prop(input)
    }
    
    fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut last_output = input.clone();
        for layer in &mut self.layers {
            last_output = layer.forward_prop(&last_output);
        }
        last_output
    }

    fn backward_prop(&mut self, gradient: &Array2 <f64>) -> Array2<f64> {
        let mut gradient = gradient.clone();
        for mut layer in &mut self.layers.iter_mut().rev() {
            gradient = layer.backward_prop(&gradient);
        }
        gradient
    }

    fn train_on_batch(&mut self, input: &Array2<f64>, output: &Array2<f64>) -> f64 {
        let prediction = self.forward_prop(&input);
        let error = self.loss_fn.run(&prediction, output);

        let gradient = self.loss_fn.gradient(&prediction, output);
        self.backward_prop(&gradient);
        
        error
    }
}