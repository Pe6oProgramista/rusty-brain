pub mod sgd;

pub trait Optimizer {
    fn update(&self) {}
}

pub struct SGD {
    pub a: u32
}