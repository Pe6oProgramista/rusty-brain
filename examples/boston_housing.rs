extern crate rusty_brain;
#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;

use rusty_brain::*

fn main() {
    let mut network = NeuralNetwork::new().set_optimizer(&Optimizer::Momentum(Default::default())).set_loss_fn(&LossFn::MeanSquare).build();
    network.add(Dense::new().set_inputs_cnt(13).set_units(5).set_activation_fn(&ActivationFn::Softmax).build());
    network.add(Dense::new().set_units(1).set_activation_fn(&ActivationFn::Linear).build());

    let (input, output) = FileReader::new("./datasets/housing.data", "").get_data(13);
    let error = network.fit(&input, &output, 101, 1_000);

    let test_in = arr2(&[[0.04741, 0.00, 11.930, 0., 0.5730, 6.0300, 80.80, 2.5050, 1., 273.0, 21.00, 396.90, 7.88]]);
    let test_in2 = arr2(&[[0.06860, 0.00, 2.890, 0., 0.4450, 7.4160, 62.50, 3.4952, 2., 276.0, 18.00, 396.90, 6.19]]);
    let test_out = 11.90;
    let test_out2 = 33.20;
    
    let _serialized = network.save("./models/regression.json").unwrap();

    let mut model = NeuralNetwork::from_json("./models/regression.json");
    let p = model.predict(&test_in);
    println!("{} ----> {}", test_out, p);
}
