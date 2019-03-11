use serde_derive::{Serialize, Deserialize};

pub mod layer;
pub mod optimizers;
pub mod activation_functions;
pub mod loss_functions;
pub mod network;

#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Box<layer::Layer>>,
    optimizer: optimizers::Optimizer,
    loss_fn: loss_functions::LossFn
}