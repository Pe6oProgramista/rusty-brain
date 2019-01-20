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
    let mut network = NeuralNetwork::new();
    network.add(Dense::new().set_input_shape(&vec![26, 2]).set_units(2).set_activation_fn(&ActivationFn::Linear).build());
    network.add(Dense::new().set_units(1).set_activation_fn(&ActivationFn::Linear).build());

    let input = arr2(&[[1., 2.],
                    [2., 3.],
                    [3., 4.],
                    [4., 5.],
                    [5., 6.],
                    [6., 7.],
                    [7., 8.],
                    [8., 9.],
                    [9., 10.],
                    [10., 11.],
                    [11., 12.],
                    [12., 13.],
                    [13., 14.],
                    [14., 15.],
                    [15., 16.],
                    [16., 17.],
                    [17., 18.],
                    [18., 19.],
                    [19., 20.],
                    [20., 21.],
                    [21., 22.],
                    [22., 23.],
                    [23., 24.],
                    [24., 25.],
                    [25., 26.],
                    [26., 27.]]);

    let output = arr2(&[[21.],
                        [35.],
                        [49.],
                        [63.],
                        [77.],
                        [91.],
                        [105.],
                        [119.],
                        [133.],
                        [147.],
                        [161.],
                        [175.],
                        [189.],
                        [203.],
                        [217.],
                        [231.],
                        [245.],
                        [259.],
                        [273.],
                        [287.],
                        [301.],
                        [315.],
                        [329.],
                        [343.],
                        [357.],
                        [371.]]);

    network.fit(&input, &output, 26, 1_000);

    let test = arr2(&[[27., 28.]]);
    let p = network.predict(&test);
    println!("----{:?}", p);

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
