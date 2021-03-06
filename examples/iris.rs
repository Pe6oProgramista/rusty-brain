extern crate rusty_brain;
extern crate ndarray;

use ndarray::prelude::*;

use rusty_brain::*;

fn main() {
    let mut network = NeuralNetwork::new().set_optimizer(&Optimizer::Adam(Default::default())).set_loss_fn(&LossFn::CrossEntropy).build();
    network.add(Dense::new().set_inputs_cnt(4).set_units(10).set_activation_fn(&ActivationFn::Softmax).build());
    network.add(Dense::new().set_units(3).set_activation_fn(&ActivationFn::Softmax).build());

    let (features_names, input, output) = FileReader::new("./datasets/iris.data", ",").get_data(4);
    let error = network.fit(&input, &output, 149, 1_000); // 1-Iris-setosa 2-Iris-versicolor 3-Iris-virginica

    let test_in = arr2(&[[4.4,2.9,1.4,0.2]]); // 4.4,2.9,1.4,0.2 - [1 0 0]  5.9,3.0,5.1,1.8 - [0 0 1]  7.0,3.2,4.7,1.4 - [0 1 0]
    let test_out = arr2(&[[1,0,0]]);
    
    let _serialized = network.save("./models/classification.json").unwrap();

    let mut model = NeuralNetwork::from_json("./models/classification.json");
    let p = model.predict(&test_in);
    println!("{} ----> {}", test_out, p);

    plot_data(&input);
    plot_error(&error);
}