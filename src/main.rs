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

use ndarray::*;

use neural_networks::*;
use neural_networks::layer::*;
use neural_networks::layer::dense::*;
use neural_networks::optimizers::*;
use neural_networks::activation_functions::*;
use neural_networks::loss_functions::*;
use file_reader::*;

extern crate gnuplot;
use gnuplot::*;
fn main() {
	let mut network = NeuralNetwork::new().set_optimizer(&Optimizer::RMSProp(Default::default())).set_loss_fn(&LossFn::CrossEntropy).build();
	network.add(Dense::new().set_inputs_cnt(4).set_units(5).set_activation_fn(&ActivationFn::Softmax).build());
	network.add(Dense::new().set_units(3).set_activation_fn(&ActivationFn::Softmax).build());

	let (input, output) = FileReader::new("./datasets/iris.data", ",").get_data(4);
	let _error = network.fit(&input, &output, 149, 1_000); // 1-Iris-setosa 2-Iris-versicolor 3-Iris-virginica

	let test_in = arr2(&[[4.4,2.9,1.4,0.2]]); // 4.4,2.9,1.4,0.2 - [1 0 0]  5.9,3.0,5.1,1.8 - [0 0 1]  7.0,3.2,4.7,1.4 - [0 1 0]
	let test_out = arr2(&[[1,0,0]]);

	let p = network.predict(&test_in);
	println!("{} ----> {}", test_out, p);
  {
	// let mut network = NeuralNetwork::new().set_optimizer(&Optimizer::Momentum(Default::default())).set_loss_fn(&LossFn::CrossEntropy).build();
	// network.add(Dense::new().set_inputs_cnt(4).set_units(5).set_activation_fn(&ActivationFn::Softmax).build());
	// network.add(Dense::new().set_units(3).set_activation_fn(&ActivationFn::Softmax).build());

	// let (input, output) = FileReader::new("./datasets/iris.data", ",").get_data(4);
	// let _error = network.fit(&input, &output, 149, 1_000); // 1-Iris-setosa 2-Iris-versicolor 3-Iris-virginica

	// let test_in = arr2(&[[4.4,2.9,1.4,0.2]]); // 4.4,2.9,1.4,0.2 - [1 0 0]  5.9,3.0,5.1,1.8 - [0 0 1]  7.0,3.2,4.7,1.4 - [0 1 0]
	// let test_out = arr2(&[[1,0,0]]);
	
	// let _serialized = network.save("./models/classification.json").unwrap();

	// let mut model = NeuralNetwork::from_json("./models/classification.json");
	// let p = model.predict(&test_in);
	// println!("{} ----> {}", test_out, p);

	// {
	//     // network.set_optimizer(&Optimizer::Momentum(Momentum {
	//     //         learning_rate: 0.1,
	//     //         momentum: 0.,
	//     //         velocity: arr2(&[[]])
	//     //     }));

	//     // let gd_e = network.fit(&input, &output, 149, 1000);

	//     // let sgd_e = network.fit(&input, &output, 15, 1000);

	//     // network.set_optimizer(&Optimizer::Momentum(Default::default()));
	//     // let momentum_e = network.fit(&input, &output, 15, 1000);

	//     // network.set_optimizer(&Optimizer::Adagrad(Default::default()));
	//     // let adagrad_e = network.fit(&input, &output, 15, 1000);

	//     // network.set_optimizer(&Optimizer::Adadelta(Default::default()));
	//     // let adadelta_e = network.fit(&input, &output, 15, 1000);

	//     // let mut dots = Vec::new();
	//     // let mut iters: Vec<f64> = (0..1000).map(|x| x as f64).collect();


	//     // dots = iters.iter().zip(gd_e.iter()).map(|(x, y)| (*x, *y)).collect();
	//     // let gd = line::Line::new(&dots)
	//     //     .style(line::Style::new()
	//     //         .colour("red"));

	//     // dots = iters.iter().zip(sgd_e.iter()).map(|(x, y)| (*x, *y)).collect();
	//     // let sgd = line::Line::new(&dots)
	//     //     .style(line::Style::new()
	//     //         .colour("green"));

	//     // dots = iters.iter().zip(momentum_e.iter()).map(|(x, y)| (*x, *y)).collect();
	//     // let momentum = line::Line::new(&dots)
	//     //     .style(line::Style::new()
	//     //         .colour("blue"));

	//     // dots = iters.iter().zip(adagrad_e.iter()).map(|(x, y)| (*x, *y)).collect();
	//     // let adagrad = line::Line::new(&dots)
	//     //     .style(line::Style::new()
	//     //         .colour("yellow"));

	//     // dots = iters.iter().zip(adadelta_e.iter()).map(|(x, y)| (*x, *y)).collect();
	//     // let adadelta = line::Line::new(&dots)
	//     //     .style(line::Style::new()
	//     //         .colour("pink"));

	//     // let mut v = ContinuousView::new()
	//     // .add(&gd)
	//     // .add(&sgd)
	//     // .add(&momentum)
	//     // .add(&adagrad)
	//     // .add(&adadelta);

	//     // Page::empty().add_plot(&v).save("scatter.svg");
	// }
  }
}

//  - burzina sravnima sus C i eliminira bugovete
//  	(golqma chast ot tqh sa memory problemi i pri nn sa mnogo trudni za namirane)
//  - pametta
//  - llvm kod
//  - portable e