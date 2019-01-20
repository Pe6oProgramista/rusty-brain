pub mod layer;
pub mod optimizers;
pub mod activation_functions;
pub mod loss_functions;
pub mod network;

#[derive(Clone)]
pub struct NeuralNetwork<L, O>
    where L: layer::LayerTrait<O>,
          O: optimizers::Optimizer
{
    layers: Vec<L>,
    optimizer: O,
    loss_fn: loss_functions::LossFn
}