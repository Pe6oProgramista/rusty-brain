pub mod layer;
pub mod optimizers;
pub mod activation_functions;
pub mod loss_functions;
pub mod network;

#[derive(Clone)]
pub struct NeuralNetwork<L>
    where L: layer::LayerTrait
{
    layers: Vec<L>,
    optimizer: optimizers::Optimizer,
    loss_fn: loss_functions::LossFn
}