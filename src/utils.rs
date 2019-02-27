use ndarray::prelude::*;

pub fn max_arr1(array: &ArrayView1<f64>) -> f64 {
    let mut max = array[0];
    for a in array.iter() {
        if *a > max {
            max = *a;
        }
    }
    max
}

pub fn max_arr2(array: &ArrayView2<f64>) -> f64 {
    let mut max = array[[0, 0]];
    for a in array.iter() {
        if *a > max {
            max = *a;
        }
    }
    max
}