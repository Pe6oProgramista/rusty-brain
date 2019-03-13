#[macro_use]
extern crate ndarray;
extern crate regex;
extern crate rand;
extern crate plotlib;
extern crate serde;
extern crate serde_derive;
extern crate serde_json;
extern crate typetag;

pub mod neural_networks;
pub mod file_reader;
pub mod utils;

pub use crate::file_reader::*;
pub use crate::neural_networks::*;
pub use crate::neural_networks::layer::*;
pub use crate::neural_networks::layer::dense::*;
pub use crate::neural_networks::optimizers::*;
pub use crate::neural_networks::activation_functions::*;
pub use crate::neural_networks::loss_functions::*;