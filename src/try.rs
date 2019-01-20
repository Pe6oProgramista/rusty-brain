use std::time::{ SystemTime};

fn main() {
    let now = SystemTime::now();
    let mut x:i64 = 5;
    for _ in 0..1_000_000_000_00 as i64 {
        x += 1;
    }

    // let mut x = Some(5 as i64);
    // for _ in 0..1000000000 as i64 {
    //     match x {
    //         Some(ref mut y) => *y += 1,
    //         None => ()
    //     }
    // }
    println!("{}", now.elapsed().unwrap().as_secs());
    // println!("{}", x);
}