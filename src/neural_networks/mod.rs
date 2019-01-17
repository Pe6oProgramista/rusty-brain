pub mod layer;
pub mod optimizers;
pub mod activation_functions;
pub mod loss_functions;
pub mod network;

pub struct NeuralNetwork<L, O>
    where L: layer::LayerTrait<O>,
          O: optimizers::Optimizer
{
    pub layers: Vec<L>,
    pub optimizer: O,
    pub loss_fn: loss_functions::LossFn
}