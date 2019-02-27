extern crate regex;
#[macro_use]
extern crate ndarray;
extern crate rand;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use regex::Regex;
use ndarray::prelude::*;
use rand::prelude::*;

pub mod supervised;
use supervised::regression as rg;

pub mod neural_networks;
use neural_networks::*;
use neural_networks::layer::*;
use neural_networks::optimizers::*;
use neural_networks::activation_functions::*;
use neural_networks::loss_functions::*;

pub mod utils;
use utils::*;

fn main() {
    let mut network = NeuralNetwork::new().set_optimizer(&Optimizer::Momentum(Default::default())).set_loss_fn(&LossFn::CrossEntropy).build();
    network.add(Dense::new().set_input_shape(&vec![149, 4]).set_units(5).set_activation_fn(&ActivationFn::Softmax).build());
    network.add(Dense::new().set_units(3).set_activation_fn(&ActivationFn::Softmax).build());

    let (input, output) = get_data();
    network.fit(&input, &output, 26, 1_000);// 1-Iris-setosa 2-Iris-versicolor 3-Iris-virginica

    // let test_in = arr2(&[[0.04741, 0.00, 11.930, 0., 0.5730, 6.0300, 80.80, 2.5050, 1., 273.0, 21.00, 396.90, 7.88]]);
    // let test_in2 = arr2(&[[0.06860, 0.00, 2.890, 0., 0.4450, 7.4160, 62.50, 3.4952, 2., 276.0, 18.00, 396.90, 6.19]]);
    // let test_out = 11.90;
    // let test_out2 = 33.20;
    // let p = network.predict(&test_in);
    // println!("{} ----> {}", test_out, p);

    let test_in = arr2(&[[4.4,2.9,1.4,0.2]]); // 4.4,2.9,1.4,0.2 - [1 0 0]  5.9,3.0,5.1,1.8 - [0 0 1]  7.0,3.2,4.7,1.4 - [0 1 0]
    let test_out = arr2(&[[1,0,0]]);
    let p = network.predict(&test_in);
    println!("{:?}", p);
}

fn get_data() -> (Array2<f64>, Array2<f64>) {
    let path = Path::new("./src/datasets/housing.data");
    let display = path.display();
    let mut file = match File::open(&path) {
            Err(why) => panic!("couldn't open {}: {}",display, why),
            Ok(file) => file,
        };

    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read: {}", why),
        Ok(_) => {},
    }
    let re = Regex::new(r"[\t ]+").unwrap();

    let mut vec: Vec<Vec<f64>> = Vec::new();
    for v in s.split('\n').collect::<Vec<&str>>().iter() {
        vec.push(re.split(v.trim()).map(|s| s.parse::<f64>().unwrap()).collect());
    }

    let mut data = Array2::<f64>::zeros((vec.len(), vec[0].len()));
    for i in 0..vec.len() {
        for j in 0..vec[0].len() {
            data[[i, j]] = vec[i][j];
        }
    }
    let data = data;

    let mut inputs = Array2::<f64>::ones((data.shape()[0], 13));
    let mut outputs = Array2::<f64>::ones((data.shape()[0], 1));

    inputs.assign(&data.slice(s![.., 0..13]));
    outputs.assign(&data.slice(s![.., 13..]));

    (inputs, outputs)
}

fn get_data2() -> (Array2<f64>, Array2<f64>) {
    let path = Path::new("./src/datasets/iris.data");
    let display = path.display();
    let mut file = match File::open(&path) {
            Err(why) => panic!("couldn't open {}: {}",display, why),
            Ok(file) => file,
        };

    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read: {}", why),
        Ok(_) => {},
    }
    let re = Regex::new(r",").unwrap();

    let mut vec: Vec<Vec<f64>> = Vec::new();
    for v in s.split('\n').collect::<Vec<&str>>().iter() {
        vec.push(re.split(v.trim()).map(|s| s.parse::<f64>().unwrap()).collect());
    }

    let mut data = Array2::<f64>::zeros((vec.len(), vec[0].len()));
    for i in 0..vec.len() {
        for j in 0..vec[0].len() {
            data[[i, j]] = vec[i][j];
        }
    }
    let data = data;

    let mut inputs = Array2::<f64>::ones((data.shape()[0], 4));
    let mut outputs = Array2::<f64>::ones((data.shape()[0], 3));

    inputs.assign(&data.slice(s![.., 0..4]));
    outputs.assign(&data.slice(s![.., 4..]));

    (inputs, outputs)
}
