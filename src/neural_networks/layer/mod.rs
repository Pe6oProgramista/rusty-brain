use ndarray::prelude::*;
use super::optimizers::*;
use super::activation_functions::*;

pub mod dense;

#[typetag::serde(tag = "type")]
pub trait Layer : LayerClone {
    fn build(&self) -> Box<Layer>;

    fn get_inputs_cnt(&self) -> usize;

    fn set_inputs_cnt(&mut self, shape: usize) -> &mut Layer;

    fn get_units(&self) -> usize;

    fn set_units(&mut self, units: usize) -> &mut Layer;

    fn get_optimizer(&self) -> Optimizer;

    fn set_optimizer(&mut self, optimizer: &Optimizer) -> &mut Layer;

    fn get_activation_fn(&self) -> ActivationFn;

    fn set_activation_fn(&mut self, activation_fn: &ActivationFn) -> &mut Layer;

    fn get_weights(&self) -> Array2<f64>;

    fn init_weights(&mut self) -> &mut Layer;

    fn set_weights(&mut self, weights: &Array2<f64>) -> &mut Layer;

    fn parameters(&self) -> usize;

    fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64>;

    fn backward_prop(&mut self, gradient: &Array2<f64>) -> Array2<f64>;
}

pub trait LayerClone {
    fn clone_box(&self) -> Box<Layer>;
}

impl<L> LayerClone for L
where
    L: 'static + Layer + Clone,
{
    fn clone_box(&self) -> Box<Layer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<Layer> {
    fn clone(&self) -> Box<Layer> {
        self.clone_box()
    }
}