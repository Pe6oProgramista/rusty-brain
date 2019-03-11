use ndarray::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::path::*;
use regex::Regex;

#[derive(Clone)]
pub struct FileReader {
    path: PathBuf,
    labels: Vec<String>,
    data: Option<Array2<f64>>,
    separator: String
}

impl FileReader {
    pub fn new(filename: &str, separator: &str) -> Self {
        FileReader {
            path: PathBuf::from(filename),
            labels: Vec::new(),
            data: None,
            separator: separator.to_string()
        }
    }

    pub fn get_data(&mut self, input_features: usize) -> (Array2<f64>, Array2<f64>) {
        if self.data.is_none() {
            self.data = Some(self.read_data());
        }
        let data = self.data.as_ref().map(|x| x).unwrap();
        let (samples, features) = data.dim();

        let mut inputs = Array2::<f64>::ones((samples, input_features));
        let mut outputs = Array2::<f64>::ones((samples, features - input_features));

        inputs.assign(&data.slice(s![.., 0..input_features]));
        outputs.assign(&data.slice(s![.., input_features..]));

        (inputs, outputs)
    }

    fn read_data(&mut self) -> Array2<f64> {
        let mut file = File::open(&self.path).expect(&format!("couldn't open {}: ", self.path.display()));

        let mut s = String::new();
        file.read_to_string(&mut s).expect("couldn't read: ");

        if self.separator.trim().is_empty() {
            self.separator = "[\t ]+".to_string();
        }
        let re = Regex::new(&self.separator).unwrap();

        let mut vec: Vec<Vec<f64>> = Vec::new();
        let lines = s.split('\n').collect::<Vec<&str>>();
        self.labels = re.split(lines[0].trim()).map(|s| s.to_string()).collect();

        for v in lines[1..].iter() {
            vec.push(re.split(v.trim()).map(|s| s.parse::<f64>().unwrap()).collect());
        }

        let mut data = Array2::<f64>::zeros((vec.len(), self.labels.len()));
        for i in 0..vec.len() {
            for j in 0..self.labels.len() {
                data[[i, j]] = vec[i][j];
            }
        }
        data
    }
}