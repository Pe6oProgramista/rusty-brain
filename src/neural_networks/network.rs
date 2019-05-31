use ndarray::prelude::*;
use ndarray::Axis;
use ndarray::Zip;
use std::cmp::min;
use rand::{thread_rng, seq::SliceRandom};
use std::ptr::copy_nonoverlapping;
use std::fs::File;
use std::io::prelude::*;
use std::path::*;
use serde_json;

use super::NeuralNetwork;
use super::layer::*;
use super::optimizers::*;
use super::loss_functions::*;

use gnuplot::*;
use gnuplot::Axis as gpAxis;

impl NeuralNetwork {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from_json(pathname: &str) -> Self {
        let path = Path::new(pathname);
        let mut file = File::open(&path).expect(&format!("couldn't open {}: ", path.display()));

        let mut json = String::new();
        file.read_to_string(&mut json).expect("couldn't read: ");

        serde_json::from_str(&json).expect("couldn't deserialize: ")
    }
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        NeuralNetwork {
            layers: Vec::<Box<Layer>>::new(),
            optimizer: Default::default(),
            loss_fn: Default::default()
        }
    }
}

impl NeuralNetwork {
    pub fn build(&self) -> Self {
        self.clone()
    }

    pub fn get_layers(&self) -> Vec<Box<Layer>> {
        self.layers.clone()
    }

    pub fn add<'a>(&'a mut self, mut layer: Box<Layer>) -> &'a mut Self {
        if self.layers.len() != 0 {
            layer.set_inputs_cnt(self.layers.last().unwrap().get_units());
        }
        if layer.get_weights().shape() == &[1, 0] {
            layer.init_weights();
        }
        layer.set_optimizer(&self.optimizer);

        self.layers.push(layer.clone());
        self
    }

    pub fn get_optimizer(&self) -> Optimizer {
        self.optimizer.clone()
    }

    pub fn set_optimizer<'a>(&'a mut self, optimizer: &Optimizer) -> &'a mut Self {
        for layer in self.layers.iter_mut() {
            layer.set_optimizer(optimizer);
        }
        self.optimizer = optimizer.clone();
        self
    }

    pub fn get_loss_fn(&self) -> LossFn {
        self.loss_fn.clone()
    }

    pub fn set_loss_fn<'a>(&'a mut self, loss_fn: &LossFn) -> &'a mut Self {
        self.loss_fn = loss_fn.clone();
        self
    }

    pub fn init_weights<'a>(&'a mut self) -> &'a mut Self {
        for layer in self.layers.iter_mut() {
            layer.init_weights();
        }
        self
    }

    pub fn parameters(&self) -> usize {
        let mut p = 0;
        for layer in &self.layers {
            p += layer.parameters();
        }
        p
    }

    pub fn fit(&mut self, input: &Array2<f64>, output: &Array2<f64>, batch_size: usize, epochs: usize) -> Vec<f64> {        
        let mut errors = Vec::<f64>::new();
        let (samples, features) = input.dim();
        let (_, lables) = output.dim();

        for iter in 0..epochs {
            let (shuffled_input, shuffled_output) = self.shuffle_data(input, output);

            let mut i = 0;
            while i < samples {
                let (start, stop) = (i, min(samples, i + batch_size));

                let mut batch_in = Array2::<f64>::ones((min(batch_size, stop - start), features));
                let mut batch_out = Array2::<f64>::ones((min(batch_size, stop - start), lables));

                batch_in.assign(&shuffled_input.slice(s![start..stop, ..]));
                batch_out.assign(&shuffled_output.slice(s![start..stop, ..]));

                let e = self.train_on_batch(&batch_in, &batch_out);
                errors.push(e);
                println!("Iter. {} --> {:?}", iter, e);
                
                i += batch_size;
            }
        }
        errors
    }

    pub fn predict(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.forward_prop(input)
    }

    pub fn save(&self, path: &str) -> Result<String, ()> {
        let serialized = serde_json::to_string(self).unwrap();

        let path = Path::new(path);
        let display = path.display();

        let mut file = File::create(&path)
            .expect(&format!("couldn't create {}: ", path.display()));

        file.write_all(serialized.as_bytes())
            .expect(&format!("couldn't write to {}: ", path.display()));

        Ok(format!("successfully wrote to {}", display))
    }
    
    fn forward_prop(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut last_output = input.clone();
        for layer in &mut self.layers {
            last_output = layer.forward_prop(&last_output);
        }
        last_output
    }

    fn backward_prop(&mut self, gradient: &Array2 <f64>) -> Array2<f64> {
        let mut gradient = gradient.clone();
        for mut layer in &mut self.layers.iter_mut().rev() {
            gradient = layer.backward_prop(&gradient);
        }
        gradient
    }

    fn train_on_batch(&mut self, input: &Array2<f64>, output: &Array2<f64>) -> f64 {
        let prediction = self.forward_prop(&input);
        let error = self.loss_fn.run(&prediction, output);

        let gradient = self.loss_fn.gradient(&prediction, output);
        self.backward_prop(&gradient);
        
        error
    }

    fn shuffle_data(&self, input: &Array2<f64>, output: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let samples = input.shape()[0];
        let mut indices: Vec<_> = (0..samples).collect();

        indices.shuffle(&mut thread_rng());

        let mut shuffled_input = unsafe { Array2::uninitialized(input.dim()) };
        let mut shuffled_output = unsafe { Array2::uninitialized(output.dim()) };

        unsafe {
            for i in 0..samples {
                Zip::from(shuffled_input.index_axis_mut(Axis(0), indices[i]))
                    .and(input.index_axis(Axis(0), i))
                    .apply(|to, from| {
                        copy_nonoverlapping(from, to, 1)
                    });
                Zip::from(shuffled_output.index_axis_mut(Axis(0), indices[i]))
                    .and(output.index_axis(Axis(0), i))
                    .apply(|to, from| {
                        copy_nonoverlapping(from, to, 1)
                    });
            }
        }
        (shuffled_input, shuffled_output)
    }
}

// let x: Vec<usize> = (0..errors.len()).collect();
// fg.axes2d()
//     .set_title("A plot", &[])
//     .set_legend(Graph(0.5), Graph(0.9), &[], &[])
//     .set_x_label("iterations", &[])
//     .set_y_label("Error", &[])
//     .lines(&x, &errors, &[Caption("Error"), Color("red")]);
// fg.show();
// fg.clear_axes();