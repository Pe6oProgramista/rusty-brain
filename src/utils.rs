use ndarray::prelude::*;

use gnuplot::*;
use gnuplot::Axis as gpAxis;

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

pub fn plot_data(data: &Array2<f64>) {
    let (samples, features) = data.dim();

    let mut fg = Figure::new();
    for row in 0..features {
        for col in 0..features {
            fg.axes2d()
                .set_pos_grid(features as u32, features as u32, (row * features + col) as u32)
                // .set_title("A plot", &[])
                // .set_legend(Graph(0.5), Graph(0.9), &[], &[])
                // .set_x_label("iterations", &[])
                // .set_y_label("Error", &[])
                .set_x_ticks(None, &[], &[])
                .set_y_ticks(None, &[], &[])
                .points(data.clone().reversed_axes().row(row), data.clone().reversed_axes().row(col), &[PointSymbol('O'), PointSize(0.5)]);
                //&[Caption("Points"), PointSymbol('D'), Color("#ffaa77"), PointSize(2.0)
        }
    }
    fg.set_terminal("pngcairo", "exampleData.png");
    fg.show().close();
}

pub fn plot_error(errors: &Vec<f64>) {
    let mut fg = Figure::new();

    let x: Vec<usize> = (0..errors.len()).collect();
    fg.axes2d()
        .set_title("A plot", &[])
        .set_legend(Graph(0.5), Graph(0.9), &[], &[])
        .set_x_label("iterations", &[])
        .set_y_label("Error", &[])
        .lines(&x, errors, &[Caption("Error"), Color("red")]);
    fg.set_terminal("pngcairo", "exampleError.png");
    fg.show().close();
}