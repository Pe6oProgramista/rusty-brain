extern crate regex;
#[macro_use]
extern crate ndarray;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use regex::Regex;
use ndarray::prelude::*;

pub mod supervised;
use supervised::regression as rg;

pub mod neural_networks;
use neural_networks::*;
use neural_networks::layer::*;
use neural_networks::optimizers::*;
use neural_networks::activation_functions::*;
use neural_networks::loss_functions::*;

fn main() {
    let mut network: NeuralNetwork<Dense<SGD>, SGD> = NeuralNetwork{ ..Default::default() };
    network.add(Dense{input_shape: vec![1, 2], units: 5, ..Default::default()});
    network.add(Dense{units: 1, ..Default::default()});

    let input = arr2(&[[1., 2.]]);

    network.forward_prop(&input);
    network.backward_prop(&Array2::ones((1, 1)));
    println!("n{}", network.parameters());

    return;
    rg::gg();
    let mut learning_rate = 0.001;
    let input_n = 13;

    let path = Path::new("D:\\Projects\\diplomna\\rusty-brain\\src\\housing.data");
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

    let mut data = ArrayD::<f64>::zeros(IxDyn(&[vec.len(), vec[0].len()]));
    for i in 0..vec.len() {
        for j in 0..vec[0].len() {
            data[[i, j]] = vec[i][j];
        }
    }
    let data = data;
    // println!("{:?}", data);

    // ArrayD::<f64>::uninitialized(IxDyn(&[data.dim()[0], 13]))
    let mut inputs = Array2::<f64>::ones((data.dim()[0], input_n + 1));
    let mut outputs = unsafe { Array2::<f64>::uninitialized((data.dim()[0], data.dim()[1] - input_n)) };
    inputs.slice_mut(s![.., 1..]).assign(&data.slice(s![.., 0..13]));
    outputs.assign(&data.slice(s![.., 13..]));
    println!("{:?} - {:?}", inputs.index_axis(Axis(0), 0), outputs.index_axis(Axis(0), 0));

    let mut w = Array2::<f64>::ones((inputs.shape()[1], 1));
    // println!("{:?}", w);

    let mut errors: Vec<f64> = Vec::new();
    let mut e_percentage = 100.;

    for i in 0.. {
        if e_percentage < 18. { break; }

        let pred = inputs.dot(&w);
        
        let e = (&pred - &outputs).map(|x| x.powf(2.)).sum() / (2. * outputs.shape()[0] as f64);
        errors.push(e);

        if errors.len() > 1 {
            if errors[i] > errors[i - 1] {
                learning_rate *= 0.6;
            } else if errors[i] < errors[i - 1] {
                learning_rate *= 1.2;
            }
        }

        e_percentage = 100. * ((&outputs - &pred) / &outputs).map(|x| x.abs()).sum() / outputs.shape()[0] as f64;
        // if i % 1000 == 0 {println!("{}   {}", e_percentage, e);}
        println!("{}   {}", e_percentage, e);

        let deltas = (&pred - &outputs).reversed_axes().dot(&inputs);
        w = w - (learning_rate * &deltas.reversed_axes() / outputs.shape()[0] as f64);
    }

    let test_in = Array1::from_vec(vec![1., 0.04741, 0.00, 11.930, 0., 0.5730, 6.0300, 80.80, 2.5050, 1., 273.0, 21.00, 396.90, 7.88]);
    let test_out = 11.90;
    let test = test_in.dot(&w);
    println!("{} --> {}", test_out, test);
}
